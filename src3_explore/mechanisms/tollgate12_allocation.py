from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import TARGET_TIMES, combine_date_time

from src3_explore.common.metrics import robust_z_scores
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
from src3_explore.common.visibility import load_phase1_context


def allocation_rows(aggregate, days) -> list[dict[str, object]]:
    rows = []
    for day in days:
        for clock in TARGET_TIMES:
            start = combine_date_time(day, clock)
            y1 = float(aggregate.get((start, "1", "0"), 0))
            y2 = float(aggregate.get((start, "2", "0"), 0))
            z12 = y1 + y2
            rows.append(
                {
                    "date": str(day),
                    "slot": f"{start.hour:02d}:{start.minute:02d}",
                    "block": "morning" if start.hour < 12 else "evening",
                    "y1": f"{y1:.6f}",
                    "y2": f"{y2:.6f}",
                    "z12": f"{z12:.6f}",
                    "r2": f"{(y2 / z12) if z12 > 0 else 0.0:.6f}",
                }
            )
    return rows


def add_anomaly_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = {}
    for idx, row in enumerate(rows):
        groups.setdefault((row["slot"], row["block"]), []).append(idx)
    out = [dict(row) for row in rows]
    for idxs in groups.values():
        z = robust_z_scores([float(rows[idx]["r2"]) for idx in idxs])
        for idx, score in zip(idxs, z):
            out[idx]["r2_robust_z"] = f"{float(score):.6f}"
            out[idx]["allocation_anomaly"] = abs(float(score)) >= 2.5
    return out


def anomaly_count_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["block"]), str(row["slot"]))
        groups[key] = groups.get(key, 0) + int(bool(row["allocation_anomaly"]))
    return [
        {"block": block, "slot": slot, "allocation_anomalies": count}
        for (block, slot), count in sorted(groups.items())
    ]


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    context = load_phase1_context(data_dir)
    train_rows = add_anomaly_scores(allocation_rows(context.train_agg, context.train_days))
    phase1_rows = add_anomaly_scores(allocation_rows(context.label_agg, context.eval_days))
    out_dir = output_dir / "mechanisms"
    train_csv = out_dir / "tollgate12_allocation_train1.csv"
    phase1_csv = out_dir / "tollgate12_allocation_phase1_observation.csv"
    anomaly_csv = out_dir / "tollgate12_allocation_train1_anomaly_counts.csv"
    chart = out_dir / "tollgate12_allocation_train1_anomaly_counts.svg"
    write_csv(train_csv, train_rows)
    write_csv(phase1_csv, phase1_rows)
    anomaly_rows = anomaly_count_rows(train_rows)
    write_csv(anomaly_csv, anomaly_rows)
    write_bar_svg(
        chart,
        [{"label": f"{row['block']} {row['slot']}", "count": row["allocation_anomalies"]} for row in anomaly_rows],
        "label",
        "count",
        "Tollgate 1/2 allocation anomaly counts",
    )
    anomaly_count = sum(1 for row in train_rows if row["allocation_anomaly"])
    phase1_anomaly_count = sum(1 for row in phase1_rows if row["allocation_anomaly"])
    card = ExperimentCard(
        name="tollgate12_allocation",
        hypothesis="tollgate 1 和 tollgate 2 entry 的总量 z12 可能稳定，但分配比例 r2 可能出现计量偏差或 allocation anomaly，导致独立预测失效。",
        data_visibility=(
            "使用 train1 标签建立 robust r2 分布和阈值；phase1 allocation 只作为最终观察输出，不用来调阈值。"
        ),
        prototype="按 slot/block 计算 z12=y1+y2 与 r2=y2/(y1+y2)，用 robust z-score 标记分配异常，并输出 slot 级异常计数。",
        metrics={
            "train1_rows": len(train_rows),
            "train1_allocation_anomalies": anomaly_count,
            "phase1_observation_allocation_anomalies": phase1_anomaly_count,
        },
        result=f"train1 中标记到 {anomaly_count} 个 broad allocation flags，phase1 最终观察中有 {phase1_anomaly_count} 个；该信号更像 residual 解释器，而不是直接纠偏规则。",
        insight="即使异常阈值较宽，也能检查 1_0/2_0 的高残差是否来自总量变化还是 tollgate 间分配变化。",
        next_step="保留并扩展。下一步把 residual_atlas 中 1_0/2_0 高误差行与 r2 anomaly join，确认是否值得做 z12/r2 reconciliation。",
        artifacts=(str(train_csv), str(phase1_csv), str(anomaly_csv), str(chart)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tollgate 1/2 allocation mechanism analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
