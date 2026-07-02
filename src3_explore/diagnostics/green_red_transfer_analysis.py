from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from kddcup2017_task2.data import OBS_TIMES, TargetRow, block_name, combine_date_time, target_volume

from src3_explore.common.metrics import bucket_quantiles, mape_value, summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
from src3_explore.common.visibility import VisibilityContext, load_phase1_context, load_train1_latest_fold_context


TARGET_TIMES_BY_BLOCK = {
    "morning": (time(8, 0), time(8, 20), time(8, 40), time(9, 0), time(9, 20), time(9, 40)),
    "evening": (time(17, 0), time(17, 20), time(17, 40), time(18, 0), time(18, 20), time(18, 40)),
}


@dataclass(frozen=True)
class TransferMatrix:
    weights: np.ndarray
    intercept: np.ndarray

    def predict(self, green: np.ndarray) -> np.ndarray:
        pred = np.asarray(green, dtype=float) @ self.weights + self.intercept
        return np.maximum(pred, 0.0)


def fit_nonnegative_transfer_matrix(green: np.ndarray, red: np.ndarray) -> TransferMatrix:
    x = np.asarray(green, dtype=float)
    y = np.asarray(red, dtype=float)
    x_aug = np.column_stack([x, np.ones(len(x), dtype=float)])
    weights = np.zeros((6, 6), dtype=float)
    intercept = np.zeros(6, dtype=float)
    for target_idx in range(6):
        coef, _ = nnls(x_aug, y[:, target_idx])
        weights[:, target_idx] = coef[:6]
        intercept[target_idx] = coef[-1]
    return TransferMatrix(weights=weights, intercept=intercept)


def block_vectors(context: VisibilityContext, source_agg, label_agg, days: Sequence, combos: Sequence[tuple[str, str]]):
    green_rows = []
    red_rows = []
    meta = []
    for day in days:
        for combo in combos:
            for block, clocks in OBS_TIMES.items():
                green = [float(source_agg.get((combine_date_time(day, clock), combo[0], combo[1]), 0)) for clock in clocks]
                red = [
                    float(label_agg.get((combine_date_time(day, clock), combo[0], combo[1]), 0))
                    for clock in TARGET_TIMES_BY_BLOCK[block]
                ]
                green_rows.append(green)
                red_rows.append(red)
                meta.append({"date": str(day), "combo": f"{combo[0]}_{combo[1]}", "block": block})
    return np.asarray(green_rows, dtype=float), np.asarray(red_rows, dtype=float), meta


def evaluate_transfer(
    transfer: TransferMatrix,
    context: VisibilityContext,
    source_agg,
    label_agg,
    days: Sequence,
) -> tuple[list[dict[str, object]], float]:
    green, red, meta = block_vectors(context, source_agg, label_agg, days, context.combos)
    pred = transfer.predict(green)
    rows = []
    green_sum = np.asarray(green, dtype=float).sum(axis=1)
    green_buckets = bucket_quantiles(green_sum, labels=("weak", "normal", "strong"))
    for block_idx, info in enumerate(meta):
        for slot_idx in range(6):
            actual = float(red[block_idx, slot_idx])
            prediction = float(pred[block_idx, slot_idx])
            rows.append(
                {
                    **info,
                    "red_slot": slot_idx,
                    "actual": f"{actual:.6f}",
                    "prediction": f"{prediction:.6f}",
                    "signed_error": f"{prediction - actual:.6f}",
                    "abs_pct_error": f"{abs(prediction - actual) / max(abs(actual), 1.0):.6f}",
                    "green_sum": f"{float(green_sum[block_idx]):.6f}",
                    "green_strength_bucket": green_buckets[block_idx],
                }
            )
    return rows, mape_value(red.ravel(), pred.ravel())


def green_shape_clusters(green: np.ndarray, n_clusters: int = 4) -> list[int]:
    from sklearn.cluster import KMeans

    x = np.asarray(green, dtype=float)
    denom = np.maximum(x.sum(axis=1, keepdims=True), 1.0)
    shape = x / denom
    k = max(1, min(n_clusters, len(shape)))
    if k == 1:
        return [0] * len(shape)
    model = KMeans(n_clusters=k, random_state=13, n_init=10)
    return [int(value) for value in model.fit_predict(shape)]


def ratio_surface_rows(green: np.ndarray, red: np.ndarray, meta: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    green_sum = np.asarray(green, dtype=float).sum(axis=1)
    buckets = bucket_quantiles(green_sum, labels=("weak", "normal", "strong"))
    rows = []
    for idx, bucket in enumerate(buckets):
        for slot_idx in range(6):
            rows.append(
                {
                    "green_strength_bucket": bucket,
                    "block": meta[idx]["block"],
                    "red_slot": slot_idx,
                    "green_sum": f"{green_sum[idx]:.6f}",
                    "red": f"{float(red[idx, slot_idx]):.6f}",
                    "red_to_green_ratio": f"{float(red[idx, slot_idx]) / max(float(green_sum[idx]), 1.0):.6f}",
                }
            )
    return rows


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    green_fit, red_fit, fit_meta = block_vectors(
        train_context,
        train_context.train_agg,
        train_context.train_agg,
        train_context.train_days,
        train_context.combos,
    )
    transfer = fit_nonnegative_transfer_matrix(green_fit, red_fit)
    train_rows, train_mape = evaluate_transfer(
        transfer,
        train_context,
        train_context.known_agg,
        train_context.label_agg,
        train_context.eval_days,
    )
    phase1_rows, phase1_mape = evaluate_transfer(
        fit_nonnegative_transfer_matrix(
            *block_vectors(phase1_context, phase1_context.train_agg, phase1_context.train_agg, phase1_context.train_days, phase1_context.combos)[:2]
        ),
        phase1_context,
        phase1_context.known_agg,
        phase1_context.label_agg,
        phase1_context.eval_days,
    )
    clusters = green_shape_clusters(green_fit)
    cluster_rows = [{**meta, "cluster": cluster} for meta, cluster in zip(fit_meta, clusters)]
    ratio_rows = ratio_surface_rows(green_fit, red_fit, fit_meta)
    transfer_rows = [
        {"green_slot": i, "red_slot": j, "weight": f"{float(transfer.weights[i, j]):.8f}"}
        for i in range(6)
        for j in range(6)
    ]
    for j in range(6):
        transfer_rows.append({"green_slot": "intercept", "red_slot": j, "weight": f"{float(transfer.intercept[j]):.8f}"})

    out_dir = output_dir / "diagnostics"
    transfer_csv = out_dir / "green_red_transfer_matrix.csv"
    train_csv = out_dir / "green_red_transfer_train1_fold.csv"
    phase1_csv = out_dir / "green_red_transfer_phase1_observation.csv"
    train_group_csv = out_dir / "green_red_transfer_train1_fold_grouped.csv"
    phase1_group_csv = out_dir / "green_red_transfer_phase1_grouped.csv"
    cluster_csv = out_dir / "green_shape_clusters.csv"
    ratio_csv = out_dir / "green_red_ratio_surface.csv"
    chart = out_dir / "green_red_transfer_phase1_grouped_mape.svg"
    train_grouped = []
    phase1_grouped = []
    for fields in (["block"], ["red_slot"], ["green_strength_bucket"], ["combo", "block"]):
        for item in summarize_errors(train_rows, fields):
            item["dimension"] = "/".join(fields)
            item["value"] = "/".join(str(item.pop(field)) for field in fields)
            train_grouped.append(item)
        for item in summarize_errors(phase1_rows, fields):
            item["dimension"] = "/".join(fields)
            item["value"] = "/".join(str(item.pop(field)) for field in fields)
            phase1_grouped.append(item)
    write_csv(transfer_csv, transfer_rows)
    write_csv(train_csv, train_rows)
    write_csv(phase1_csv, phase1_rows)
    write_csv(train_group_csv, train_grouped)
    write_csv(phase1_group_csv, phase1_grouped)
    write_csv(cluster_csv, cluster_rows)
    write_csv(ratio_csv, ratio_rows)
    chart_rows = [
        {"label": f"{row['dimension']}={row['value']}", "mape": row["mape"]}
        for row in sorted(phase1_grouped, key=lambda item: float(item["mape"]), reverse=True)
    ]
    write_bar_svg(chart, chart_rows, "label", "mape", "Green-red transfer phase1 MAPE by group", max_items=18)
    card = ExperimentCard(
        name="green_red_transfer_analysis",
        hypothesis="同一天 6 个绿色观察 slot 到 6 个红色目标 slot 之间存在可解释的形状迁移，但这种迁移应受到非负和低复杂度约束。",
        data_visibility=(
            "transfer 矩阵、shape cluster 和 ratio surface 只用 train1 拟合或选择；phase1 标签只用于观察固定原型表现，不参与矩阵秩、聚类数或平滑参数选择。"
        ),
        prototype="拟合非负 6x6 green-to-red transfer matrix，聚类 green shape，并输出 red/green ratio surface 和分组误差。",
        metrics={
            "train1_latest_fold_mape": f"{train_mape:.6f}",
            "phase1_observation_mape": f"{phase1_mape:.6f}",
            "worst_phase1_group": chart_rows[0]["label"] if chart_rows else "none",
        },
        result=(
            f"单纯 6x6 线性迁移较弱，train1 fold MAPE={train_mape:.6f}，phase1 MAPE={phase1_mape:.6f}；"
            f"最差 phase1 分组为 {chart_rows[0]['label'] if chart_rows else 'none'}。"
        ),
        insight="分数不好仍有价值：它说明 green shape 的线性可迁移部分有限，并能暴露哪些 slot/combo 需要非线性或 regime 条件。",
        next_step="归档为诊断基线。只在 residual_atlas 高误差组与特定 green shape cluster 重合时，再考虑扩展非线性 transfer。",
        artifacts=(
            str(transfer_csv),
            str(train_csv),
            str(phase1_csv),
            str(train_group_csv),
            str(phase1_group_csv),
            str(cluster_csv),
            str(ratio_csv),
            str(chart),
        ),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze green-window to red-window transfer")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
