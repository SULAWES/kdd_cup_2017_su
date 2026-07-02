from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, TargetRow, block_name, combine_date_time, make_target_rows, target_volume

from src3_explore.common.metrics import mape_value, summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
from src3_explore.common.visibility import load_phase1_context, load_train1_latest_fold_context


ATTR_GROUPS = {
    "etc": ("0", "1"),
    "model": ("0", "1", "2", "3", "4", "5", "6", "7"),
    "veh_type": ("blank", "0", "1"),
}


def attr_count(attr_agg: Mapping, row: TargetRow, attr_name: str, attr_value: str, red: bool) -> float:
    if red:
        return float(attr_agg.get((row.start, row.tollgate_id, row.direction, attr_name, attr_value), 0))
    total = 0.0
    for clock in OBS_TIMES[block_name(row.start)]:
        obs_start = combine_date_time(row.start.date(), clock)
        total += float(attr_agg.get((obs_start, row.tollgate_id, row.direction, attr_name, attr_value), 0))
    return total


def fit_component_ratios(rows: Sequence[TargetRow], attr_agg: Mapping, label_agg: Mapping, attr_name: str) -> dict[tuple, float]:
    values = defaultdict(list)
    for row in rows:
        for attr_value in ATTR_GROUPS[attr_name]:
            green = attr_count(attr_agg, row, attr_name, attr_value, red=False)
            red = attr_count(attr_agg, row, attr_name, attr_value, red=True)
            if green > 0:
                values[(row.combo, block_name(row.start), attr_value)].append(red / green)
    return {key: float(np.median(items)) for key, items in values.items() if items}


def predict_components(rows, known_attr, ratios, attr_name: str) -> np.ndarray:
    pred = []
    for row in rows:
        total = 0.0
        for attr_value in ATTR_GROUPS[attr_name]:
            green = attr_count(known_attr, row, attr_name, attr_value, red=False)
            ratio = ratios.get((row.combo, block_name(row.start), attr_value), 0.0)
            total += green * ratio
        pred.append(total)
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def evaluate_attr(context, attr_name: str):
    train_rows = make_target_rows(context.train_days, context.combos)
    ratios = fit_component_ratios(train_rows, context.train_attr_agg, context.train_agg, attr_name)
    pred = predict_components(context.rows, context.known_attr_agg, ratios, attr_name)
    actual = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    rows = []
    for row, y, p in zip(context.rows, actual, pred):
        rows.append(
            {
                "date": str(row.start.date()),
                "combo": f"{row.tollgate_id}_{row.direction}",
                "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                "attr_name": attr_name,
                "actual": f"{y:.6f}",
                "component_prediction": f"{float(p):.6f}",
                "signed_error": f"{float(p - y):.6f}",
            }
        )
    return rows, mape_value(actual, pred)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    summary = []
    grouped_rows = []
    artifacts = []
    for attr_name in ATTR_GROUPS:
        train_rows, train_mape = evaluate_attr(train_context, attr_name)
        phase1_rows, phase1_mape = evaluate_attr(phase1_context, attr_name)
        train_csv = output_dir / "mechanisms" / f"etc_component_{attr_name}_train1_fold.csv"
        phase1_csv = output_dir / "mechanisms" / f"etc_component_{attr_name}_phase1.csv"
        write_csv(train_csv, train_rows)
        write_csv(phase1_csv, phase1_rows)
        artifacts.extend([str(train_csv), str(phase1_csv)])
        for context_name, rows in (("train1_latest_fold", train_rows), ("phase1_observation", phase1_rows)):
            for fields in (["combo"], ["slot"], ["combo", "slot"]):
                for item in summarize_errors(rows, fields, prediction_field="component_prediction"):
                    item["component"] = attr_name
                    item["context"] = context_name
                    item["dimension"] = "/".join(fields)
                    item["value"] = "/".join(str(item.pop(field)) for field in fields)
                    grouped_rows.append(item)
        summary.append(
            {
                "component": attr_name,
                "train1_latest_fold_mape": f"{train_mape:.6f}",
                "phase1_observation_mape": f"{phase1_mape:.6f}",
            }
        )
    summary_csv = output_dir / "mechanisms" / "etc_component_model_summary.csv"
    grouped_csv = output_dir / "mechanisms" / "etc_component_model_grouped_errors.csv"
    chart = output_dir / "mechanisms" / "etc_component_model_phase1_mape.svg"
    write_csv(summary_csv, summary)
    write_csv(grouped_csv, grouped_rows)
    write_bar_svg(
        chart,
        [
            {"label": row["component"], "mape": row["phase1_observation_mape"]}
            for row in sorted(summary, key=lambda item: float(item["phase1_observation_mape"]), reverse=True)
        ],
        "label",
        "mape",
        "Component model phase1 MAPE",
    )
    artifacts.extend([str(summary_csv), str(grouped_csv), str(chart)])
    best = min(summary, key=lambda row: float(row["train1_latest_fold_mape"]))
    card = ExperimentCard(
        name="etc_component_model",
        hypothesis="ETC/non-ETC、vehicle_model、vehicle_type 不应只作为普通特征；它们也可能是可生成、可 reconcile 的子流量分量。",
        data_visibility=(
            "component ratio 只在训练窗口拟合；held-out 和 phase1 红窗标签只用于评分固定的 component-sum 预测。"
        ),
        prototype="分别按 ETC、vehicle model、vehicle type 拟合 green→red component ratio，再把各 component red 预测求和并输出分组误差。",
        metrics={"best_train_component": best["component"], "best_train_mape": best["train1_latest_fold_mape"]},
        result=f"最佳 train1 component 为 {best['component']}，MAPE={best['train1_latest_fold_mape']}；component generator 作为 standalone 模型偏弱，但车辆结构 share 有解释价值。",
        insight="即使不能直接降 MAPE，也能判断 component share 是否稳定，以及 total prediction 与 component prediction 是否存在可 reconcile 的方向性误差。",
        next_step="保留为解释和 reconciliation 候选。只有在 rolling folds 中证明某个 component 稳定改善，再与 total-flow prediction 做约束融合。",
        artifacts=tuple(artifacts),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ETC and vehicle component mechanism model")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
