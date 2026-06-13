from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import infer_dates
from kddcup2017_task2.ensemble import latest_training_fold_split
from kddcup2017_task2.model import mape
from kddcup2017_task2_exp.observation_adjust_exp import (
    BASE_CONFIGS,
    build_base_predictions,
    fit_apply_adjustment,
    fit_expected_obs,
    observation_strengths,
)
from kddcup2017_task2_exp.trajectory_ensemble_exp import load_inputs
from kddcup2017_task2_exp.trajectory_rolling_exp import fold_candidate_matrices


def fold_day_sets(data, validation_start):
    all_days = infer_dates(data["train1"])
    validation_end = validation_start + timedelta(days=6)
    validation_days = [day for day in all_days if validation_start <= day <= validation_end]
    available_days = [day for day in all_days if day < validation_start]
    calibration_train_days, calibration_days = latest_training_fold_split(available_days)
    return calibration_train_days, calibration_days, available_days, validation_days


def run_fold(data, validation_start, beta_max: float):
    (
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_actual,
        validation_matrix,
    ) = fold_candidate_matrices(data, validation_start, include_route_means=False)
    calibration_train_days, _, available_days, _ = fold_day_sets(data, validation_start)
    combos = sorted({row.combo for row in calibration_rows})
    combo_block, combo_block_dow = fit_expected_obs(data["train1"], calibration_train_days, combos)
    validation_combo_block, validation_combo_block_dow = fit_expected_obs(data["train1"], available_days, combos)

    rows = []
    base_predictions = [
        build_base_predictions(*config, calibration_rows, calibration_actual, calibration_matrix, validation_rows, validation_matrix)
        for config in BASE_CONFIGS
    ]
    for base in base_predictions:
        rows.append(
            {
                "validation_start": validation_start.isoformat(),
                "base_name": base["base_name"],
                "expected_mode": "none",
                "smoothing": "",
                "adjustment_scope": "none",
                "calibration_mape": base["calibration_mape"],
                "validation_mape": mape(validation_actual, base["validation_predictions"]),
                "pred_mean": float(base["validation_predictions"].mean()),
                "betas": "",
            }
        )
        for expected_mode in ("combo_block", "combo_block_dow"):
            for smoothing in (5.0, 20.0, 50.0):
                calibration_strength = observation_strengths(
                    calibration_rows,
                    data["train1"],
                    combo_block,
                    combo_block_dow,
                    expected_mode,
                    smoothing,
                )
                validation_strength = observation_strengths(
                    validation_rows,
                    data["train1"],
                    validation_combo_block,
                    validation_combo_block_dow,
                    expected_mode,
                    smoothing,
                )
                for adjustment_scope in ("global", "block", "hour"):
                    cal_preds, val_preds, betas = fit_apply_adjustment(
                        calibration_rows,
                        calibration_actual,
                        base["calibration_predictions"],
                        calibration_strength,
                        validation_rows,
                        base["validation_predictions"],
                        validation_strength,
                        adjustment_scope,
                        beta_max,
                    )
                    beta_text = " | ".join(
                        f"{'/'.join(key)}:{value:.3f}" for key, value in sorted(betas.items())
                    )
                    rows.append(
                        {
                            "validation_start": validation_start.isoformat(),
                            "base_name": base["base_name"],
                            "expected_mode": expected_mode,
                            "smoothing": smoothing,
                            "adjustment_scope": adjustment_scope,
                            "calibration_mape": mape(calibration_actual, cal_preds),
                            "validation_mape": mape(validation_actual, val_preds),
                            "pred_mean": float(val_preds.mean()),
                            "betas": beta_text,
                        }
                    )
    return rows


def summarize(rows: Sequence[Mapping[str, object]]):
    grouped = {}
    for row in rows:
        key = (
            row["base_name"],
            row["expected_mode"],
            row["smoothing"],
            row["adjustment_scope"],
        )
        grouped.setdefault(key, []).append(float(row["validation_mape"]))
    summary = []
    for key, scores in grouped.items():
        summary.append(
            {
                "base_name": key[0],
                "expected_mode": key[1],
                "smoothing": key[2],
                "adjustment_scope": key[3],
                "mean_validation_mape": float(np.mean(scores)),
                "min_validation_mape": float(np.min(scores)),
                "max_validation_mape": float(np.max(scores)),
                "fold_scores": ",".join(f"{score:.6f}" for score in scores),
            }
        )
    return sorted(summary, key=lambda row: float(row["mean_validation_mape"]))


def write_detail(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "validation_start",
        "base_name",
        "expected_mode",
        "smoothing",
        "adjustment_scope",
        "calibration_mape",
        "validation_mape",
        "pred_mean",
        "betas",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in ("calibration_mape", "validation_mape"):
                output[key] = f"{float(output[key]):.6f}"
            output["pred_mean"] = f"{float(output['pred_mean']):.3f}"
            writer.writerow(output)


def write_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_name",
        "expected_mode",
        "smoothing",
        "adjustment_scope",
        "mean_validation_mape",
        "min_validation_mape",
        "max_validation_mape",
        "fold_scores",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in ("mean_validation_mape", "min_validation_mape", "max_validation_mape"):
                output[key] = f"{float(output[key]):.6f}"
            writer.writerow(output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train1-only rolling observation adjustment experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_observation_adjust_rolling.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/experiments/src1_observation_adjust_rolling_summary.csv"),
    )
    parser.add_argument("--beta-max", type=float, default=0.8)
    args = parser.parse_args(argv)
    data = load_inputs(args.data_dir)
    train_days = infer_dates(data["train1"])
    validation_starts = [train_days[-14], train_days[-7]]
    rows = []
    for idx, validation_start in enumerate(validation_starts, start=1):
        print(f"[fold {idx}/{len(validation_starts)}] validation_start={validation_start}")
        fold_rows = run_fold(data, validation_start, args.beta_max)
        fold_rows.sort(key=lambda row: float(row["validation_mape"]))
        rows.extend(fold_rows)
        for row in fold_rows[:5]:
            print(
                f"  base={row['base_name']} expected={row['expected_mode']} "
                f"smoothing={row['smoothing']} adjustment={row['adjustment_scope']} "
                f"validation_mape={float(row['validation_mape']):.6f}"
            )
    summary = summarize(rows)
    write_detail(args.output, rows)
    write_summary(args.summary_output, summary)
    for row in summary[:10]:
        print(
            f"summary base={row['base_name']} expected={row['expected_mode']} "
            f"smoothing={row['smoothing']} adjustment={row['adjustment_scope']} "
            f"mean_validation_mape={float(row['mean_validation_mape']):.6f} "
            f"fold_scores={row['fold_scores']}"
        )
    print(f"best_mean_validation_mape={float(summary[0]['mean_validation_mape']):.6f}")
    print(f"output={args.output}")
    print(f"summary_output={args.summary_output}")
