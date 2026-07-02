from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, project_paths, target_volume

from src3_explore.common.metrics import safe_corr
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.trajectory import INTERSECTIONS, read_trajectory_aggregate, route_count_at_lag
from src3_explore.common.visibility import load_phase1_context


def kernel_feature_rows(context, trajectory_agg, lags: Sequence[int]) -> list[dict[str, object]]:
    rows = []
    for row in make_target_rows(context.train_days, context.combos):
        y = float(target_volume(context.train_agg, row))
        for intersection in INTERSECTIONS:
            for lag in lags:
                rows.append(
                    {
                        "date": str(row.start.date()),
                        "combo": f"{row.tollgate_id}_{row.direction}",
                        "tollgate_id": row.tollgate_id,
                        "direction": row.direction,
                        "block": "morning" if row.start.hour < 12 else "evening",
                        "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                        "intersection": intersection,
                        "lag_minutes": lag,
                        "route_count": route_count_at_lag(row, trajectory_agg, intersection, lag),
                        "actual": y,
                    }
                )
    return rows


def summarize_kernel(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups = {}
    for row in rows:
        key = (row["intersection"], row["tollgate_id"], row["lag_minutes"], row["block"])
        groups.setdefault(key, {"x": [], "y": []})
        groups[key]["x"].append(float(row["route_count"]))
        groups[key]["y"].append(float(row["actual"]))
    out = []
    for key, value in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), int(item[0][2]), str(item[0][3]))):
        x = np.asarray(value["x"], dtype=float)
        y = np.asarray(value["y"], dtype=float)
        corr = safe_corr(x, y)
        out.append(
            {
                "intersection": key[0],
                "tollgate_id": key[1],
                "lag_minutes": key[2],
                "block": key[3],
                "corr": f"{corr:.6f}",
                "route_count_mean": f"{float(x.mean()) if len(x) else 0.0:.6f}",
                "actual_mean": f"{float(y.mean()) if len(y) else 0.0:.6f}",
            }
        )
    return out


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    context = load_phase1_context(data_dir)
    paths = project_paths(data_dir)
    traj = read_trajectory_aggregate([data_dir / "dataSets" / "training" / "trajectories(table 5)_training.csv"])
    lags = (20, 40, 60, 80, 100, 120)
    detail = kernel_feature_rows(context, traj, lags)
    summary = summarize_kernel(detail)
    out_dir = output_dir / "mechanisms"
    detail_csv = out_dir / "route_arrival_kernel_rows.csv"
    summary_csv = out_dir / "route_arrival_kernel_summary.csv"
    write_csv(detail_csv, detail)
    write_csv(summary_csv, summary)
    best = max(summary, key=lambda row: abs(float(row["corr"]))) if summary else {"corr": "0"}
    card = ExperimentCard(
        name="route_arrival_kernel",
        hypothesis="Route trajectory counts should explain tollgate volume after an interpretable lead-lag kernel.",
        data_visibility=(
            "Mechanism analysis uses train1 trajectory and volume labels only. It does not tune or rerun the "
            "five-node GNN; phase1 labels are not used."
        ),
        prototype="Compute correlations between upstream intersection-to-tollgate counts at 20-120 minute lags and target volumes.",
        metrics={"rows": len(detail), "max_abs_corr": f"{abs(float(best['corr'])):.6f}"},
        result=f"Wrote route kernel summary to {summary_csv}.",
        insight="Large lag-specific correlations identify route mechanisms that are richer than the five-node tollgate graph.",
        next_step="Only promote route features that remain visible from green windows and survive rolling-fold checks.",
        artifacts=(str(detail_csv), str(summary_csv)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Route arrival lead-lag kernel analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

