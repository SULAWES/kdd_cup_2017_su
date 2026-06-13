from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from kddcup2017_task2.data import (
    infer_combos,
    infer_dates,
    load_weather,
    make_target_rows,
    merge_aggregates,
    merge_attr_aggregates,
    project_paths,
    read_volume_aggregate,
    read_volume_attr_aggregate,
    target_volume,
)
from kddcup2017_task2.ensemble import (
    ENSEMBLE_MODEL_NAMES,
    apply_scoped_blend,
    attr_observation_windows_only,
    filter_attr_days,
    filter_days,
    fit_ensemble_prediction_matrix,
    latest_training_fold_split,
    observation_windows_only,
    optimize_scoped_blend_weights,
)
from kddcup2017_task2.model import mape
from kddcup2017_task2_exp.trajectory_exp import (
    filter_trajectory_days,
    merge_trajectory_aggregates,
    read_trajectory_aggregate,
    train_predict_group,
    trajectory_observation_windows_only,
)


def load_inputs(data_dir: Path):
    paths = project_paths(data_dir)
    return {
        "paths": paths,
        "train1": read_volume_aggregate([paths["train1_volume"]]),
        "train2": read_volume_aggregate([paths["train2_volume"]]),
        "test1_obs": read_volume_aggregate([paths["test1_volume"]]),
        "train1_attr": read_volume_attr_aggregate([paths["train1_volume"]]),
        "test1_attr": read_volume_attr_aggregate([paths["test1_volume"]]),
        "train1_traj": read_trajectory_aggregate(
            [data_dir / "dataSets" / "training" / "trajectories(table 5)_training.csv"]
        ),
        "test1_traj": read_trajectory_aggregate(
            [data_dir / "dataSets" / "testing_phase1" / "trajectories(table 5)_test1.csv"]
        ),
        "weather": load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase1"]]),
    }


def build_candidate_matrices(data, include_route_means: bool):
    train1 = data["train1"]
    train2 = data["train2"]
    test1_obs = data["test1_obs"]
    train1_attr = data["train1_attr"]
    test1_attr = data["test1_attr"]
    train1_traj = data["train1_traj"]
    test1_traj = data["test1_traj"]
    weather = data["weather"]
    combos = infer_combos(train1)
    train_days_all = infer_dates(train1)
    calibration_train_days, calibration_days = latest_training_fold_split(train_days_all)

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
    calibration_traj_pred = train_predict_group(
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
    calibration_matrix = np.column_stack([calibration_matrix, calibration_traj_pred])
    calibration_actual = np.array([target_volume(train1, row) for row in calibration_rows], dtype=float)

    validation_rows = make_target_rows(infer_dates(train2), combos)
    validation_matrix, _ = fit_ensemble_prediction_matrix(
        train1,
        merge_aggregates(train1, test1_obs),
        weather,
        train1_attr,
        merge_attr_aggregates(train1_attr, test1_attr),
        train_days_all,
        validation_rows,
        combos,
    )
    validation_traj_pred = train_predict_group(
        "low_volume_block",
        train1,
        merge_aggregates(train1, test1_obs),
        weather,
        train1_attr,
        merge_attr_aggregates(train1_attr, test1_attr),
        train1_traj,
        merge_trajectory_aggregates(train1_traj, test1_traj),
        train_days_all,
        validation_rows,
        combos,
        include_route_means,
    )
    validation_matrix = np.column_stack([validation_matrix, validation_traj_pred])
    validation_actual = np.array([target_volume(train2, row) for row in validation_rows], dtype=float)
    return calibration_rows, calibration_actual, calibration_matrix, validation_rows, validation_actual, validation_matrix


def optimize_capped_blend_weights(actual, prediction_matrix, trajectory_cap: float):
    actual = np.asarray(actual, dtype=float)
    n_models = prediction_matrix.shape[1]
    trajectory_cap = float(trajectory_cap)

    def objective(weights):
        return mape(actual, prediction_matrix @ weights)

    def is_feasible(weights):
        weights = np.asarray(weights, dtype=float)
        return (
            np.all(np.isfinite(weights))
            and abs(float(np.sum(weights)) - 1.0) <= 1e-6
            and np.all(weights >= -1e-8)
            and np.all(weights[:-1] <= 1.0 + 1e-8)
            and -1e-8 <= weights[-1] <= trajectory_cap + 1e-8
        )

    def add_start(starts, weights):
        weights = np.asarray(weights, dtype=float)
        if is_feasible(weights):
            starts.append(weights.copy())

    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)
    bounds = [(0.0, 1.0)] * (n_models - 1) + [(0.0, trajectory_cap)]
    starts = []
    no_traj = np.zeros(n_models)
    no_traj[:-1] = 1.0 / (n_models - 1)
    add_start(starts, no_traj)
    for idx in range(n_models - 1):
        start = np.zeros(n_models)
        start[idx] = 1.0
        add_start(starts, start)
        if trajectory_cap > 0:
            capped_start = np.zeros(n_models)
            capped_start[idx] = 1.0 - trajectory_cap
            capped_start[-1] = trajectory_cap
            add_start(starts, capped_start)
    if trajectory_cap >= 1.0:
        start = np.zeros(n_models)
        start[-1] = 1.0
        add_start(starts, start)
    if trajectory_cap > 0:
        capped_uniform = np.zeros(n_models)
        capped_uniform[:-1] = (1.0 - trajectory_cap) / (n_models - 1)
        capped_uniform[-1] = trajectory_cap
        add_start(starts, capped_uniform)

    best_weights = None
    best_score = float("inf")
    for start in starts:
        start_score = float(objective(start))
        if np.isfinite(start_score) and start_score < best_score:
            best_weights = start.copy()
            best_score = start_score
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        result_weights = np.asarray(result.x, dtype=float)
        result_score = float(objective(result_weights)) if is_feasible(result_weights) else float("inf")
        if np.isfinite(result_score) and result_score < best_score:
            best_weights = result_weights
            best_score = result_score
    if best_weights is None:
        raise RuntimeError("no feasible capped blend weights found")
    best_weights = np.maximum(best_weights, 0.0)
    best_weights[-1] = min(best_weights[-1], trajectory_cap)
    best_weights /= best_weights.sum()
    return best_weights, float(objective(best_weights))


def optimize_scoped_capped_blend(actual, prediction_matrix, rows, weight_scope: str, trajectory_cap: float):
    def scope_key(row):
        if weight_scope == "global":
            return ("global",)
        if weight_scope == "hour":
            return (f"{row.start.hour:02d}",)
        if weight_scope == "block":
            return ("morning" if row.start.hour < 12 else "evening",)
        raise ValueError(f"unknown weight scope: {weight_scope}")

    keys = [scope_key(row) for row in rows]
    predictions = np.zeros(len(rows), dtype=float)
    weights_by_scope = {}
    for key in sorted(set(keys)):
        idx = [row_idx for row_idx, row_key in enumerate(keys) if row_key == key]
        weights, _ = optimize_capped_blend_weights(actual[idx], prediction_matrix[idx], trajectory_cap)
        weights_by_scope[key] = weights
        predictions[idx] = prediction_matrix[idx] @ weights
    return weights_by_scope, mape(actual, predictions)


def apply_scoped_weights(prediction_matrix, rows, weights_by_scope, weight_scope: str):
    def scope_key(row):
        if weight_scope == "global":
            return ("global",)
        if weight_scope == "hour":
            return (f"{row.start.hour:02d}",)
        if weight_scope == "block":
            return ("morning" if row.start.hour < 12 else "evening",)
        raise ValueError(f"unknown weight scope: {weight_scope}")

    predictions = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        predictions[idx] = prediction_matrix[idx] @ weights_by_scope[scope_key(row)]
    return predictions


def run_one(data, weight_scope: str, include_route_means: bool, trajectory_cap: float):
    (
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_actual,
        validation_matrix,
    ) = build_candidate_matrices(data, include_route_means)
    if trajectory_cap >= 1.0:
        weights_by_scope, calibration_mape, _ = optimize_scoped_blend_weights(
            calibration_actual,
            calibration_matrix,
            calibration_rows,
            weight_scope,
        )
        predictions = apply_scoped_blend(validation_matrix, validation_rows, weights_by_scope, weight_scope)
    else:
        weights_by_scope, calibration_mape = optimize_scoped_capped_blend(
            calibration_actual,
            calibration_matrix,
            calibration_rows,
            weight_scope,
            trajectory_cap,
        )
        predictions = apply_scoped_weights(validation_matrix, validation_rows, weights_by_scope, weight_scope)
    names = ENSEMBLE_MODEL_NAMES + ("trajectory_low_volume_block",)
    weight_parts = []
    for key, weights in weights_by_scope.items():
        key_name = "/".join(str(item) for item in key)
        values = ",".join(f"{name}:{weight:.4f}" for name, weight in zip(names, weights))
        weight_parts.append(f"{key_name}={values}")
    single_scores = {
        name: mape(validation_actual, validation_matrix[:, idx])
        for idx, name in enumerate(names)
    }
    return {
        "weight_scope": weight_scope,
        "include_route_means": include_route_means,
        "trajectory_cap": trajectory_cap,
        "calibration_mape": calibration_mape,
        "validation_mape": mape(validation_actual, predictions),
        "pred_mean": float(predictions.mean()),
        "weights": " | ".join(weight_parts),
        "single_scores": ",".join(f"{name}:{score:.6f}" for name, score in single_scores.items()),
    }


def write_results(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "weight_scope",
        "include_route_means",
        "trajectory_cap",
        "calibration_mape",
        "validation_mape",
        "pred_mean",
        "weights",
        "single_scores",
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
    parser = argparse.ArgumentParser(description="Add trajectory candidate to four-model ensemble")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_trajectory_ensemble.csv"))
    parser.add_argument("--weight-scopes", nargs="+", default=["global", "block", "hour"])
    parser.add_argument("--trajectory-caps", nargs="+", type=float, default=[1.0])
    args = parser.parse_args(argv)
    data = load_inputs(args.data_dir)
    results = []
    for include_route_means in (False, True):
        for weight_scope in args.weight_scopes:
            for trajectory_cap in args.trajectory_caps:
                result = run_one(data, weight_scope, include_route_means, trajectory_cap)
                results.append(result)
                print(
                    f"weight_scope={weight_scope} include_route_means={include_route_means} "
                    f"trajectory_cap={trajectory_cap:.2f} "
                    f"calibration_mape={result['calibration_mape']:.6f} "
                    f"validation_mape={result['validation_mape']:.6f} pred_mean={result['pred_mean']:.3f}"
                )
    results.sort(key=lambda item: float(item["validation_mape"]))
    write_results(args.output, results)
    print(f"best_validation_mape={float(results[0]['validation_mape']):.6f}")
    print(f"output={args.output}")
