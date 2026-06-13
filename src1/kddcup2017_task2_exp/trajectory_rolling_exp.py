from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import infer_combos, infer_dates, make_target_rows, target_volume
from kddcup2017_task2.ensemble import (
    attr_observation_windows_only,
    filter_attr_days,
    filter_days,
    fit_ensemble_prediction_matrix,
    latest_training_fold_split,
    observation_windows_only,
)
from kddcup2017_task2.model import mape
from kddcup2017_task2_exp.trajectory_ensemble_exp import (
    apply_scoped_weights,
    load_inputs,
    optimize_scoped_capped_blend,
)
from kddcup2017_task2_exp.trajectory_exp import (
    filter_trajectory_days,
    merge_trajectory_aggregates,
    train_predict_group,
    trajectory_observation_windows_only,
)
from kddcup2017_task2.data import merge_aggregates, merge_attr_aggregates


def fold_candidate_matrices(data, validation_start, include_route_means: bool):
    train1 = data["train1"]
    train1_attr = data["train1_attr"]
    train1_traj = data["train1_traj"]
    weather = data["weather"]
    combos = infer_combos(train1)
    all_days = infer_dates(train1)
    validation_end = validation_start + timedelta(days=6)
    validation_days = [day for day in all_days if validation_start <= day <= validation_end]
    available_days = [day for day in all_days if day < validation_start]
    calibration_train_days, calibration_days = latest_training_fold_split(available_days)

    calibration_train = filter_days(train1, calibration_train_days)
    calibration_train_attr = filter_attr_days(train1_attr, calibration_train_days)
    calibration_train_traj = filter_trajectory_days(train1_traj, calibration_train_days)
    calibration_known = merge_aggregates(calibration_train, observation_windows_only(train1, calibration_days))
    calibration_known_attr = merge_attr_aggregates(
        calibration_train_attr,
        attr_observation_windows_only(train1_attr, calibration_days),
    )
    calibration_known_traj = merge_trajectory_aggregates(
        calibration_train_traj,
        trajectory_observation_windows_only(train1_traj, calibration_days),
    )
    calibration_rows = make_target_rows(calibration_days, combos)
    calibration_matrix, _ = fit_ensemble_prediction_matrix(
        calibration_train,
        calibration_known,
        weather,
        calibration_train_attr,
        calibration_known_attr,
        calibration_train_days,
        calibration_rows,
        combos,
    )
    calibration_traj = train_predict_group(
        "low_volume_block",
        calibration_train,
        calibration_known,
        weather,
        calibration_train_attr,
        calibration_known_attr,
        calibration_train_traj,
        calibration_known_traj,
        calibration_train_days,
        calibration_rows,
        combos,
        include_route_means,
    )
    calibration_matrix = np.column_stack([calibration_matrix, calibration_traj])
    calibration_actual = np.array([target_volume(train1, row) for row in calibration_rows], dtype=float)

    validation_train = filter_days(train1, available_days)
    validation_train_attr = filter_attr_days(train1_attr, available_days)
    validation_train_traj = filter_trajectory_days(train1_traj, available_days)
    validation_known = merge_aggregates(validation_train, observation_windows_only(train1, validation_days))
    validation_known_attr = merge_attr_aggregates(
        validation_train_attr,
        attr_observation_windows_only(train1_attr, validation_days),
    )
    validation_known_traj = merge_trajectory_aggregates(
        validation_train_traj,
        trajectory_observation_windows_only(train1_traj, validation_days),
    )
    validation_rows = make_target_rows(validation_days, combos)
    validation_matrix, _ = fit_ensemble_prediction_matrix(
        validation_train,
        validation_known,
        weather,
        validation_train_attr,
        validation_known_attr,
        available_days,
        validation_rows,
        combos,
    )
    validation_traj = train_predict_group(
        "low_volume_block",
        validation_train,
        validation_known,
        weather,
        validation_train_attr,
        validation_known_attr,
        validation_train_traj,
        validation_known_traj,
        available_days,
        validation_rows,
        combos,
        include_route_means,
    )
    validation_matrix = np.column_stack([validation_matrix, validation_traj])
    validation_actual = np.array([target_volume(train1, row) for row in validation_rows], dtype=float)
    return calibration_rows, calibration_actual, calibration_matrix, validation_rows, validation_actual, validation_matrix


def run_fold(data, validation_start, include_route_means: bool, cap: float):
    cal_rows, cal_y, cal_m, val_rows, val_y, val_m = fold_candidate_matrices(
        data,
        validation_start,
        include_route_means,
    )
    weights, cal_score = optimize_scoped_capped_blend(cal_y, cal_m, cal_rows, "block", cap)
    preds = apply_scoped_weights(val_m, val_rows, weights, "block")
    return {
        "validation_start": validation_start.isoformat(),
        "include_route_means": include_route_means,
        "trajectory_cap": cap,
        "calibration_mape": cal_score,
        "validation_mape": mape(val_y, preds),
        "pred_mean": float(preds.mean()),
    }


def write_results(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "validation_start",
        "include_route_means",
        "trajectory_cap",
        "calibration_mape",
        "validation_mape",
        "pred_mean",
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train1-only rolling check for trajectory-capped ensemble")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_trajectory_rolling_caps.csv"))
    parser.add_argument("--caps", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    args = parser.parse_args(argv)
    data = load_inputs(args.data_dir)
    train_days = infer_dates(data["train1"])
    validation_starts = [train_days[-14], train_days[-7]]
    results = []
    for validation_start in validation_starts:
        for include_route_means in (False, True):
            for cap in args.caps:
                result = run_fold(data, validation_start, include_route_means, cap)
                results.append(result)
                print(
                    f"validation_start={result['validation_start']} "
                    f"include_route_means={include_route_means} cap={cap:.2f} "
                    f"calibration_mape={result['calibration_mape']:.6f} "
                    f"validation_mape={result['validation_mape']:.6f}"
                )
    write_results(args.output, results)
    by_cap = {}
    for row in results:
        key = (row["include_route_means"], row["trajectory_cap"])
        by_cap.setdefault(key, []).append(float(row["validation_mape"]))
    best_key, best_scores = min(by_cap.items(), key=lambda item: float(np.mean(item[1])))
    print(
        f"best_mean_validation_mape={float(np.mean(best_scores)):.6f} "
        f"include_route_means={best_key[0]} cap={best_key[1]:.2f}"
    )
    print(f"output={args.output}")
