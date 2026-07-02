from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import TARGET_TIMES, combine_date_time

from src3_explore.common.metrics import robust_z_scores
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
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


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    context = load_phase1_context(data_dir)
    train_rows = add_anomaly_scores(allocation_rows(context.train_agg, context.train_days))
    phase1_rows = add_anomaly_scores(allocation_rows(context.label_agg, context.eval_days))
    out_dir = output_dir / "mechanisms"
    train_csv = out_dir / "tollgate12_allocation_train1.csv"
    phase1_csv = out_dir / "tollgate12_allocation_phase1_observation.csv"
    write_csv(train_csv, train_rows)
    write_csv(phase1_csv, phase1_rows)
    anomaly_count = sum(1 for row in train_rows if row["allocation_anomaly"])
    card = ExperimentCard(
        name="tollgate12_allocation",
        hypothesis="Total entry demand for tollgate 1+2 and the y2 share may expose allocation or metering anomalies.",
        data_visibility=(
            "Uses train1 labels for mechanism fitting and reports phase1 allocation only as final observation; "
            "no phase1 labels are used to tune thresholds."
        ),
        prototype="Compute z12=y1+y2 and r2=y2/(y1+y2), then flag robust r2 deviations within slot/block.",
        metrics={"train1_rows": len(train_rows), "train1_allocation_anomalies": anomaly_count},
        result=f"Wrote allocation tables to {train_csv} and {phase1_csv}.",
        insight="Allocation anomalies are plausible failure modes for models that predict tollgates independently.",
        next_step="Compare phase1 high residuals on 1_0/2_0 with r2 anomaly flags before adding reconciliation.",
        artifacts=(str(train_csv), str(phase1_csv)),
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

