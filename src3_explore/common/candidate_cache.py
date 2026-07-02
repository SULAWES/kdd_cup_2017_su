from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, TargetRow, block_name, combine_date_time, make_target_rows, project_paths, target_volume
from kddcup2017_task2.ensemble import (
    ENSEMBLE_MODEL_NAMES,
    apply_scoped_blend,
    attr_observation_windows_only,
    filter_attr_days,
    filter_days,
    fit_ensemble_prediction_matrix,
    latest_training_fold_split,
    mape_sample_weight,
    observation_windows_only,
    optimize_scoped_blend_weights,
    predict_mlp,
    predict_ratio_lag7,
)
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features, train_and_predict

from .metrics import bucket_quantiles, mape_value
from .reporting import read_csv, write_csv
from .trajectory import (
    merge_trajectory_aggregates,
    read_trajectory_aggregate,
    trajectory_green_count,
    trajectory_observation_windows_only,
)
from .visibility import load_phase1_context


def cache_path(output_dir: Path, name: str = "phase1_candidate_predictions.csv") -> Path:
    return output_dir / "cache" / name


def obs_sum(known_agg: Mapping, row: TargetRow) -> float:
    return float(
        sum(
            known_agg.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0)
            for clock in OBS_TIMES[block_name(row.start)]
        )
    )


def expected_obs_by_combo_block(train_agg: Mapping, train_days: Sequence, combos: Sequence[tuple[str, str]]) -> dict:
    values: dict[tuple, list[float]] = defaultdict(list)
    for combo in combos:
        for day in train_days:
            for block, clocks in OBS_TIMES.items():
                total = sum(train_agg.get((combine_date_time(day, clock), combo[0], combo[1]), 0) for clock in clocks)
                values[(combo, block)].append(float(total))
    return {key: float(np.median(items)) if items else 0.0 for key, items in values.items()}


def green_strengths(rows: Sequence[TargetRow], known_agg: Mapping, expected: Mapping) -> np.ndarray:
    strengths = []
    for row in rows:
        current = obs_sum(known_agg, row)
        baseline = float(expected.get((row.combo, block_name(row.start)), 0.0))
        strengths.append(np.log((current + 20.0) / (baseline + 20.0)))
    return np.asarray(strengths, dtype=float)


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def predict_xgb_diagnostic_fallback(x_train, y_log, x_pred, sample_weight):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=240,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        min_samples_leaf=8,
        random_state=13,
    )
    model.fit(x_train, y_log, sample_weight=sample_weight)
    return np.maximum(np.expm1(model.predict(x_pred)), 0.0)


def fit_prediction_matrix_visible(
    train_agg,
    known_agg,
    weather,
    train_attr_agg,
    known_attr_agg,
    train_days,
    pred_rows: Sequence[TargetRow],
    combos,
):
    if xgboost_available():
        return fit_ensemble_prediction_matrix(
            train_agg,
            known_agg,
            weather,
            train_attr_agg,
            known_attr_agg,
            train_days,
            pred_rows,
            combos,
        ), "official_xgboost"

    predictions = {}
    _, low_volume_preds, _, _ = train_and_predict(
        train_agg,
        known_agg,
        weather,
        train_attr_agg,
        known_attr_agg,
        train_days,
        pred_rows,
        combos,
        "extra",
        20.0,
        "low_volume_block",
        "log",
        False,
        0.3,
        DEFAULT_DROP_FEATURES,
    )
    predictions["low_volume_block"] = low_volume_preds

    train_rows = make_target_rows(train_days, combos)
    y_train = np.asarray([target_volume(train_agg, row) for row in train_rows], dtype=float)
    y_log = np.log1p(y_train)
    sample_weight = mape_sample_weight(y_train)

    builder = FeatureBuilder(train_agg, weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = builder.transform(train_rows, train_agg, train_attr_agg)
    pred_features = builder.transform(pred_rows, known_agg, known_attr_agg)
    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(filter_features(train_features, DEFAULT_DROP_FEATURES))
    x_pred = vectorizer.transform(filter_features(pred_features, DEFAULT_DROP_FEATURES))

    predictions["xgb"] = predict_xgb_diagnostic_fallback(x_train, y_log, x_pred, sample_weight)
    predictions["mlp"] = predict_mlp(x_train, y_log, x_pred)
    predictions["ratio_lag_7"] = predict_ratio_lag7(
        x_train,
        y_train,
        x_pred,
        train_features,
        pred_features,
        sample_weight,
    )
    matrix = np.column_stack([predictions[name] for name in ENSEMBLE_MODEL_NAMES])
    return (matrix, predictions), "diagnostic_hgb_fallback_for_missing_xgboost"


def calibration_actual_values(train_agg: Mapping, rows: Sequence[TargetRow]) -> np.ndarray:
    return np.asarray([target_volume(train_agg, row) for row in rows], dtype=float)


def etc_share(row: TargetRow, known_attr_agg: Mapping) -> float:
    total = 0.0
    etc = 0.0
    for clock in OBS_TIMES[block_name(row.start)]:
        obs_start = combine_date_time(row.start.date(), clock)
        for value in ("0", "1"):
            count = float(known_attr_agg.get((obs_start, row.tollgate_id, row.direction, "etc", value), 0))
            total += count
            if value == "1":
                etc += count
    return etc / total if total > 0 else 0.0


def build_prediction_payload(data_dir: Path):
    context = load_phase1_context(data_dir)
    train_days_all = list(context.train_days)
    calibration_train_days, calibration_days = latest_training_fold_split(train_days_all)
    calibration_train = filter_days(context.train_agg, calibration_train_days)
    calibration_train_attr = filter_attr_days(context.train_attr_agg, calibration_train_days)
    calibration_known = {
        **calibration_train,
        **observation_windows_only(context.train_agg, calibration_days),
    }
    calibration_known_attr = {
        **calibration_train_attr,
        **attr_observation_windows_only(context.train_attr_agg, calibration_days),
    }
    from kddcup2017_task2.data import merge_aggregates, merge_attr_aggregates

    calibration_known = merge_aggregates(calibration_train, observation_windows_only(context.train_agg, calibration_days))
    calibration_known_attr = merge_attr_aggregates(
        calibration_train_attr,
        attr_observation_windows_only(context.train_attr_agg, calibration_days),
    )
    calibration_rows = make_target_rows(calibration_days, context.combos)
    (calibration_matrix, _), calibration_backend = fit_prediction_matrix_visible(
        calibration_train,
        calibration_known,
        context.weather,
        calibration_train_attr,
        calibration_known_attr,
        calibration_train_days,
        calibration_rows,
        context.combos,
    )
    calibration_actual = calibration_actual_values(context.train_agg, calibration_rows)
    weights_by_scope, calibration_mape, _ = optimize_scoped_blend_weights(
        calibration_actual,
        calibration_matrix,
        calibration_rows,
        "hour",
    )
    (validation_matrix, candidate_predictions), validation_backend = fit_prediction_matrix_visible(
        context.train_agg,
        context.known_agg,
        context.weather,
        context.train_attr_agg,
        context.known_attr_agg,
        context.train_days,
        context.rows,
        context.combos,
    )
    validation_pred = apply_scoped_blend(validation_matrix, context.rows, weights_by_scope, "hour")
    validation_actual = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    return {
        "context": context,
        "calibration_rows": calibration_rows,
        "calibration_matrix": calibration_matrix,
        "calibration_actual": calibration_actual,
        "calibration_prediction": apply_scoped_blend(calibration_matrix, calibration_rows, weights_by_scope, "hour"),
        "calibration_mape": calibration_mape,
        "validation_matrix": validation_matrix,
        "validation_actual": validation_actual,
        "validation_prediction": validation_pred,
        "candidate_predictions": candidate_predictions,
        "candidate_backend": validation_backend,
        "calibration_backend": calibration_backend,
    }


def ensure_phase1_candidate_cache(data_dir: Path, output_dir: Path, force: bool = False) -> Path:
    path = cache_path(output_dir)
    if path.exists() and not force:
        return path
    payload = build_prediction_payload(data_dir)
    context = payload["context"]
    matrix = payload["validation_matrix"]
    actual = payload["validation_actual"]
    pred = payload["validation_prediction"]

    expected = expected_obs_by_combo_block(context.train_agg, context.train_days, context.combos)
    strengths = green_strengths(context.rows, context.known_agg, expected)
    strength_buckets = bucket_quantiles(strengths)
    disagreement = (np.max(matrix, axis=1) - np.min(matrix, axis=1)) / np.maximum(np.mean(matrix, axis=1), 1.0)
    disagreement_buckets = bucket_quantiles(disagreement)
    etc_values = np.asarray([etc_share(row, context.known_attr_agg) for row in context.rows], dtype=float)
    etc_buckets = bucket_quantiles(etc_values)

    paths = project_paths(data_dir)
    train_traj = read_trajectory_aggregate([data_dir / "dataSets" / "training" / "trajectories(table 5)_training.csv"])
    test1_traj = read_trajectory_aggregate(
        [data_dir / "dataSets" / "testing_phase1" / "trajectories(table 5)_test1.csv"]
    )
    known_traj = merge_trajectory_aggregates(train_traj, test1_traj)
    traj_values = np.asarray([trajectory_green_count(row, known_traj) for row in context.rows], dtype=float)
    traj_buckets = bucket_quantiles(traj_values)

    rows = []
    for idx, row in enumerate(context.rows):
        item = {
            "date": str(row.start.date()),
            "tollgate_id": row.tollgate_id,
            "direction": row.direction,
            "combo": f"{row.tollgate_id}_{row.direction}",
            "block": block_name(row.start),
            "hour": f"{row.start.hour:02d}",
            "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
            "weekday": row.start.weekday(),
            "actual": f"{actual[idx]:.6f}",
            "prediction": f"{pred[idx]:.6f}",
            "signed_error": f"{pred[idx] - actual[idx]:.6f}",
            "abs_pct_error": f"{abs(pred[idx] - actual[idx]) / max(abs(actual[idx]), 1.0):.6f}",
            "green_obs_sum": f"{obs_sum(context.known_agg, row):.6f}",
            "green_obs_strength": f"{strengths[idx]:.6f}",
            "green_obs_strength_bucket": strength_buckets[idx],
            "ETC_share": f"{etc_values[idx]:.6f}",
            "ETC_share_bucket": etc_buckets[idx],
            "trajectory_signal": f"{traj_values[idx]:.6f}",
            "trajectory_signal_bucket": traj_buckets[idx],
            "model_disagreement": f"{disagreement[idx]:.6f}",
            "model_disagreement_bucket": disagreement_buckets[idx],
        }
        for model_idx, name in enumerate(ENSEMBLE_MODEL_NAMES):
            item[f"candidate_{name}"] = f"{matrix[idx, model_idx]:.6f}"
        rows.append(item)
    write_csv(path, rows)
    meta = [
        {
            "metric": "calibration_mape_latest_train1_fold",
            "value": f"{payload['calibration_mape']:.6f}",
        },
        {"metric": "phase1_observation_mape", "value": f"{mape_value(actual, pred):.6f}"},
        {"metric": "candidate_backend", "value": payload["candidate_backend"]},
        {"metric": "calibration_backend", "value": payload["calibration_backend"]},
    ]
    write_csv(cache_path(output_dir, "phase1_candidate_predictions_meta.csv"), meta)
    return path


def load_candidate_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path)
