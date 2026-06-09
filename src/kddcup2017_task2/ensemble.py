from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import (
    OBS_TIMES,
    TargetRow,
    infer_combos,
    infer_dates,
    load_weather,
    make_target_rows,
    make_target_rows_like_sample,
    merge_aggregates,
    merge_attr_aggregates,
    project_paths,
    read_sample_shape,
    read_volume_aggregate,
    read_volume_attr_aggregate,
    target_volume,
    write_submission,
)
from .features import FeatureBuilder, Vectorizer
from .model import mape
from .pipeline import DEFAULT_DROP_FEATURES, filter_features, train_and_predict


ENSEMBLE_MODEL_NAMES = ("low_volume_block", "xgb", "mlp", "ratio_lag_7")


def fit_ensemble_prediction_matrix(
    train_agg,
    known_agg,
    weather,
    train_attr_agg,
    known_attr_agg,
    train_days,
    pred_rows: Sequence[TargetRow],
    combos,
):
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
    y_train = np.array([target_volume(train_agg, row) for row in train_rows], dtype=float)
    y_log = np.log1p(y_train)
    sample_weight = mape_sample_weight(y_train)

    builder = FeatureBuilder(train_agg, weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = builder.transform(train_rows, train_agg, train_attr_agg)
    pred_features = builder.transform(pred_rows, known_agg, known_attr_agg)
    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(filter_features(train_features, DEFAULT_DROP_FEATURES))
    x_pred = vectorizer.transform(filter_features(pred_features, DEFAULT_DROP_FEATURES))

    predictions["xgb"] = predict_xgb(x_train, y_log, x_pred, sample_weight)
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
    return matrix, predictions


def mape_sample_weight(y):
    denom = np.maximum(np.asarray(y, dtype=float), 1.0)
    weights = (float(np.mean(denom)) / denom) ** 0.3
    return weights / np.mean(weights)


def predict_xgb(x_train, y_log, x_pred, sample_weight):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.05,
        objective="reg:squarederror",
        random_state=13,
        n_jobs=-1,
    )
    model.fit(x_train, y_log, sample_weight=sample_weight)
    return np.maximum(np.expm1(model.predict(x_pred)), 0.0)


def predict_mlp(x_train, y_log, x_pred):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(48, 24),
            alpha=0.01,
            learning_rate_init=0.003,
            max_iter=1200,
            random_state=13,
            early_stopping=True,
            validation_fraction=0.15,
        ),
    )
    model.fit(x_train, y_log)
    return np.maximum(np.expm1(model.predict(x_pred)), 0.0)


def predict_ratio_lag7(x_train, y_train, x_pred, train_features, pred_features, sample_weight):
    from sklearn.ensemble import ExtraTreesRegressor

    train_base = np.array([max(features.get("lag_7", 0.0), 1.0) for features in train_features], dtype=float)
    pred_base = np.array([max(features.get("lag_7", 0.0), 1.0) for features in pred_features], dtype=float)
    target = np.log((np.asarray(y_train, dtype=float) + 1.0) / train_base)
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=10,
        random_state=17,
        n_jobs=-1,
    )
    model.fit(x_train, target, sample_weight=sample_weight)
    return np.maximum(np.exp(model.predict(x_pred)) * pred_base - 1.0, 0.0)


def optimize_blend_weights(actual, prediction_matrix):
    from scipy.optimize import minimize

    actual = np.asarray(actual, dtype=float)
    n_models = prediction_matrix.shape[1]

    def objective(weights):
        return mape(actual, prediction_matrix @ weights)

    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)
    starts = [np.ones(n_models) / n_models] + [np.eye(n_models)[idx] for idx in range(n_models)]
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if best is None or result.fun < best.fun:
            best = result
    return np.asarray(best.x, dtype=float), float(best.fun)


def blend_scope_key(row: TargetRow, scope: str):
    if scope == "global":
        return ("global",)
    if scope == "hour":
        return (f"{row.start.hour:02d}",)
    if scope == "block":
        return ("morning" if row.start.hour < 12 else "evening",)
    raise ValueError(f"unknown blend weight scope: {scope}")


def optimize_scoped_blend_weights(actual, prediction_matrix, rows: Sequence[TargetRow], scope: str):
    keys = [blend_scope_key(row, scope) for row in rows]
    predictions = np.zeros(len(rows), dtype=float)
    weights_by_scope = {}
    scores_by_scope = {}
    for key in sorted(set(keys)):
        idx = [row_idx for row_idx, row_key in enumerate(keys) if row_key == key]
        weights, score = optimize_blend_weights(actual[idx], prediction_matrix[idx])
        weights_by_scope[key] = weights
        scores_by_scope[key] = score
        predictions[idx] = prediction_matrix[idx] @ weights
    return weights_by_scope, float(mape(actual, predictions)), scores_by_scope


def apply_scoped_blend(prediction_matrix, rows: Sequence[TargetRow], weights_by_scope, scope: str):
    predictions = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        key = blend_scope_key(row, scope)
        if key not in weights_by_scope:
            raise KeyError(f"missing blend weights for scope {key}")
        predictions[idx] = prediction_matrix[idx] @ weights_by_scope[key]
    return predictions


def format_scope_key(key) -> str:
    return "_".join(str(item) for item in key)


def latest_training_fold_split(days):
    latest_day = max(days)
    valid_start = latest_day - timedelta(days=6)
    valid_days = [day for day in days if valid_start <= day <= latest_day]
    train_days = [day for day in days if day < valid_start]
    return train_days, valid_days


def filter_days(aggregate, days):
    day_set = set(days)
    return {key: value for key, value in aggregate.items() if key[0].date() in day_set}


def filter_attr_days(aggregate, days):
    day_set = set(days)
    return {key: value for key, value in aggregate.items() if key[0].date() in day_set}


def observation_windows_only(aggregate, days):
    day_set = set(days)
    clocks = set(OBS_TIMES["morning"] + OBS_TIMES["evening"])
    return {
        key: value
        for key, value in aggregate.items()
        if key[0].date() in day_set and key[0].time() in clocks
    }


def attr_observation_windows_only(aggregate, days):
    day_set = set(days)
    clocks = set(OBS_TIMES["morning"] + OBS_TIMES["evening"])
    return {
        key: value
        for key, value in aggregate.items()
        if key[0].date() in day_set and key[0].time() in clocks
    }


def validate_latest_fold_ensemble(args) -> None:
    paths = project_paths(args.data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test1_obs = read_volume_aggregate([paths["test1_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    test1_attr = read_volume_attr_aggregate([paths["test1_volume"]])
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase1"]])
    combos = infer_combos(train1)
    train_days_all = infer_dates(train1)

    calibration_train_days, calibration_days = latest_training_fold_split(train_days_all)
    calibration_train = filter_days(train1, calibration_train_days)
    calibration_train_attr = filter_attr_days(train1_attr, calibration_train_days)
    calibration_known = merge_aggregates(calibration_train, observation_windows_only(train1, calibration_days))
    calibration_known_attr = merge_attr_aggregates(
        calibration_train_attr,
        attr_observation_windows_only(train1_attr, calibration_days),
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
    calibration_actual = np.array([target_volume(train1, row) for row in calibration_rows], dtype=float)
    weight_scope = getattr(args, "weight_scope", "hour")
    weights_by_scope, calibration_score, scope_scores = optimize_scoped_blend_weights(
        calibration_actual,
        calibration_matrix,
        calibration_rows,
        weight_scope,
    )

    valid_days = infer_dates(train2)
    valid_rows = make_target_rows(valid_days, combos)
    validation_matrix, candidate_predictions = fit_ensemble_prediction_matrix(
        train1,
        merge_aggregates(train1, test1_obs),
        weather,
        train1_attr,
        merge_attr_aggregates(train1_attr, test1_attr),
        train_days_all,
        valid_rows,
        combos,
    )
    validation_actual = np.array([target_volume(train2, row) for row in valid_rows], dtype=float)
    predictions = apply_scoped_blend(validation_matrix, valid_rows, weights_by_scope, weight_scope)
    score = mape(validation_actual, predictions)

    print("calibration=latest_training_fold")
    print("leakage_check=uses only labels before validation period")
    print(f"weight_scope={weight_scope}")
    print(f"calibration_rows={len(calibration_rows)}")
    print(f"calibration_mape={calibration_score:.6f}")
    for key, weights in weights_by_scope.items():
        scope_name = format_scope_key(key)
        print(f"scope_{scope_name}_calibration_mape={scope_scores[key]:.6f}")
        for name, weight in zip(ENSEMBLE_MODEL_NAMES, weights):
            print(f"weight_{scope_name}_{name}={weight:.6f}")
    for name, candidate in candidate_predictions.items():
        print(f"single_{name}_mape={mape(validation_actual, candidate):.6f}")
    print(f"validation_rows={len(valid_rows)}")
    print(f"validation_mape={score:.6f}")
    print(f"actual_mean={validation_actual.mean():.3f}")
    print(f"pred_mean={predictions.mean():.3f}")
    if args.validation_output:
        write_submission(args.validation_output, valid_rows, predictions)
        print(f"validation_prediction={args.validation_output}")


def predict_phase2_ensemble(args) -> None:
    paths = project_paths(args.data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test1_obs = read_volume_aggregate([paths["test1_volume"]])
    test2_obs = read_volume_aggregate([paths["test2_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    train2_attr = read_volume_attr_aggregate([paths["train2_volume"]])
    test1_attr = read_volume_attr_aggregate([paths["test1_volume"]])
    test2_attr = read_volume_attr_aggregate([paths["test2_volume"]])
    weather = load_weather(
        [paths["weather_train"], paths["weather_train_orig"], paths["weather_phase1"], paths["weather_phase2"]]
    )
    combos = infer_combos(train1)

    calibration_rows = make_target_rows(infer_dates(train2), combos)
    calibration_matrix, _ = fit_ensemble_prediction_matrix(
        train1,
        merge_aggregates(train1, test1_obs),
        weather,
        train1_attr,
        merge_attr_aggregates(train1_attr, test1_attr),
        infer_dates(train1),
        calibration_rows,
        combos,
    )
    calibration_actual = np.array([target_volume(train2, row) for row in calibration_rows], dtype=float)
    weight_scope = getattr(args, "weight_scope", "hour")
    weights_by_scope, calibration_score, scope_scores = optimize_scoped_blend_weights(
        calibration_actual,
        calibration_matrix,
        calibration_rows,
        weight_scope,
    )

    train_all = merge_aggregates(train1, train2)
    known = merge_aggregates(train_all, test2_obs)
    train_all_attr = merge_attr_aggregates(train1_attr, train2_attr)
    known_attr = merge_attr_aggregates(train_all_attr, test2_attr)
    _, sample_combos = read_sample_shape(paths["sample_volume"])
    pred_combos = sample_combos or infer_combos(train_all)
    first_pred_day = min(infer_dates(test2_obs))
    pred_rows = make_target_rows_like_sample(paths["sample_volume"], first_pred_day)
    prediction_matrix, _ = fit_ensemble_prediction_matrix(
        train_all,
        known,
        weather,
        train_all_attr,
        known_attr,
        infer_dates(train_all),
        pred_rows,
        pred_combos,
    )
    predictions = apply_scoped_blend(prediction_matrix, pred_rows, weights_by_scope, weight_scope)
    write_submission(args.output, pred_rows, predictions)

    print("calibration=train1_to_train2")
    print("leakage_check=legal_for_phase2_only; do not report this as a no-leak phase1 validation score")
    print(f"weight_scope={weight_scope}")
    print(f"calibration_rows={len(calibration_rows)}")
    print(f"calibration_mape={calibration_score:.6f}")
    for key, weights in weights_by_scope.items():
        scope_name = format_scope_key(key)
        print(f"scope_{scope_name}_calibration_mape={scope_scores[key]:.6f}")
        for name, weight in zip(ENSEMBLE_MODEL_NAMES, weights):
            print(f"weight_{scope_name}_{name}={weight:.6f}")
    print(f"prediction_rows={len(pred_rows)}")
    print(f"pred_mean={predictions.mean():.3f}")
    print(f"submission={args.output}")
