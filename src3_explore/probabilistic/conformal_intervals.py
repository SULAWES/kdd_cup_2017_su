from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.candidate_cache import build_prediction_payload
from src3_explore.common.metrics import bucket_quantiles, interval_coverage, summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv


def conformal_radius(abs_residuals: np.ndarray, alpha: float = 0.1) -> float:
    residuals = np.asarray(abs_residuals, dtype=float)
    if len(residuals) == 0:
        return 0.0
    q = np.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals)
    q = min(max(float(q), 0.0), 1.0)
    return float(np.quantile(residuals, q, method="higher"))


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    payload = build_prediction_payload(data_dir)
    context = payload["context"]
    cal_actual = payload["calibration_actual"]
    cal_pred = payload["calibration_prediction"]
    val_actual = payload["validation_actual"]
    val_pred = payload["validation_prediction"]
    matrix = payload["validation_matrix"]
    radius = conformal_radius(np.abs(cal_actual - cal_pred), alpha=0.1)
    lower = np.maximum(val_pred - radius, 0.0)
    upper = val_pred + radius
    uncertainty = np.std(matrix, axis=1) / np.maximum(np.mean(matrix, axis=1), 1.0)
    buckets = bucket_quantiles(uncertainty, labels=("low", "mid", "high"))
    rows = []
    for idx, row in enumerate(context.rows):
        rows.append(
            {
                "date": str(row.start.date()),
                "combo": f"{row.tollgate_id}_{row.direction}",
                "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                "actual": f"{val_actual[idx]:.6f}",
                "prediction": f"{val_pred[idx]:.6f}",
                "lower": f"{lower[idx]:.6f}",
                "upper": f"{upper[idx]:.6f}",
                "covered": bool(lower[idx] <= val_actual[idx] <= upper[idx]),
                "uncertainty": f"{uncertainty[idx]:.6f}",
                "uncertainty_bucket": buckets[idx],
            }
        )
    summary = summarize_errors(rows, ["uncertainty_bucket"])
    coverage = interval_coverage(val_actual, lower, upper)
    csv_path = output_dir / "probabilistic" / "conformal_intervals_phase1.csv"
    summary_csv = output_dir / "probabilistic" / "conformal_intervals_by_uncertainty.csv"
    write_csv(csv_path, rows)
    write_csv(summary_csv, summary)
    card = ExperimentCard(
        name="conformal_intervals",
        hypothesis="Calibration residuals and ensemble spread should expose when predictions are unreliable.",
        data_visibility=(
            "Conformal radius is fitted on the latest train1 calibration fold only. Phase1 labels are used only "
            "to evaluate fixed interval coverage."
        ),
        prototype="Symmetric split-conformal interval around the official hour-weight ensemble plus uncertainty buckets from candidate spread.",
        metrics={
            "radius": f"{radius:.6f}",
            "coverage": f"{coverage['coverage']:.6f}",
            "mean_width": f"{coverage['mean_width']:.6f}",
        },
        result=f"Wrote conformal interval rows to {csv_path}.",
        insight="If high-spread buckets also have higher MAPE or lower coverage, the ensemble knows part of its uncertainty.",
        next_step="Calibrate separate radii by train1-selected regimes only if pooled coverage is poorly calibrated.",
        artifacts=(str(csv_path), str(summary_csv)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Conformal interval and ensemble uncertainty diagnostics")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

