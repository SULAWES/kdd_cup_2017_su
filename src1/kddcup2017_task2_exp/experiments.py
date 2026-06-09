from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import (
    block_name,
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
    write_submission,
)
from kddcup2017_task2.ensemble import (
    attr_observation_windows_only,
    filter_attr_days,
    filter_days,
    latest_training_fold_split,
    observation_windows_only,
)
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.model import RidgeRegressor, mape
from kddcup2017_task2.pipeline import (
    DEFAULT_DROP_FEATURES,
    apply_history_blend,
    filter_features,
    group_key,
    select_low_volume_combos,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    group: str = "low_volume_block"
    target_mode: str = "volume"
    target_transform: str = "log"
    include_weather: bool = False
    sample_weight_power: float = 0.3
    low_volume_ratio: float = 0.6
    drop_features: tuple[str, ...] = DEFAULT_DROP_FEATURES
    history_blend: float = 0.0
    prediction_scale: float = 1.0
    params: Mapping[str, object] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class EnsembleSpec:
    name: str
    candidate_names: tuple[str, ...]
    weight_scope: str = "global"
    shrink_to_global: float = 0.0
    optimize_global_shrink: bool = False
    post_scale_scope: str = ""
    note: str = ""


def candidate_specs() -> dict[str, CandidateSpec]:
    specs = [
        CandidateSpec(
            name="sota_single_extra",
            family="extra",
            params={"n_estimators": 600, "max_depth": 14, "min_samples_leaf": 10},
            note="Current best single-model settings kept as an experiment baseline.",
        ),
        CandidateSpec(
            name="extra_leaf6",
            family="extra",
            params={"n_estimators": 800, "max_depth": 14, "min_samples_leaf": 6},
            note="Lower leaf size to fit more local variation.",
        ),
        CandidateSpec(
            name="extra_depth18_leaf8",
            family="extra",
            params={"n_estimators": 800, "max_depth": 18, "min_samples_leaf": 8},
            note="Deeper trees with moderate smoothing.",
        ),
        CandidateSpec(
            name="extra_leaf14",
            family="extra",
            params={"n_estimators": 800, "max_depth": 14, "min_samples_leaf": 14},
            note="More smoothing for low-volume MAPE stability.",
        ),
        CandidateSpec(
            name="extra_weight04",
            family="extra",
            sample_weight_power=0.4,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Increase low-volume weighting strength.",
        ),
        CandidateSpec(
            name="extra_weight02",
            family="extra",
            sample_weight_power=0.2,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Reduce low-volume weighting strength.",
        ),
        CandidateSpec(
            name="extra_weight025",
            family="extra",
            sample_weight_power=0.25,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Slightly reduce low-volume weighting strength.",
        ),
        CandidateSpec(
            name="extra_weight035",
            family="extra",
            sample_weight_power=0.35,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Slightly increase low-volume weighting strength.",
        ),
        CandidateSpec(
            name="extra_lv_ratio05",
            family="extra",
            low_volume_ratio=0.5,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Stricter recent-low-volume trigger.",
        ),
        CandidateSpec(
            name="extra_lv_ratio07",
            family="extra",
            low_volume_ratio=0.7,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Looser recent-low-volume trigger.",
        ),
        CandidateSpec(
            name="extra_lv_ratio08",
            family="extra",
            low_volume_ratio=0.8,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Much looser recent-low-volume trigger.",
        ),
        CandidateSpec(
            name="extra_global_leaf6",
            family="extra",
            group="global",
            params={"n_estimators": 800, "max_depth": 14, "min_samples_leaf": 6},
            note="Global-only control for lower leaf size.",
        ),
        CandidateSpec(
            name="extra_weather",
            family="extra",
            include_weather=True,
            params={"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 10},
            note="Re-test weather under the isolated experiment harness.",
        ),
        CandidateSpec(
            name="hgb_global",
            family="hgb",
            group="global",
            params={
                "max_iter": 350,
                "learning_rate": 0.035,
                "l2_regularization": 0.1,
                "min_samples_leaf": 8,
            },
            note="Histogram gradient boosting on shared features.",
        ),
        CandidateSpec(
            name="lgbm_global",
            family="lgbm",
            group="global",
            params={
                "n_estimators": 450,
                "learning_rate": 0.025,
                "num_leaves": 15,
                "min_child_samples": 8,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
            note="LightGBM global model.",
        ),
        CandidateSpec(
            name="xgb_global",
            family="xgb",
            group="global",
            params={
                "n_estimators": 500,
                "learning_rate": 0.025,
                "max_depth": 3,
                "min_child_weight": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
            },
            note="XGBoost global model similar to the ensemble candidate.",
        ),
        CandidateSpec(
            name="mlp_global",
            family="mlp",
            group="global",
            sample_weight_power=0.0,
            params={
                "hidden_layer_sizes": (48, 24),
                "alpha": 0.01,
                "learning_rate_init": 0.003,
                "max_iter": 1200,
                "early_stopping": True,
                "validation_fraction": 0.15,
            },
            note="MLP candidate mirroring the current SOTA ensemble member.",
        ),
        CandidateSpec(
            name="ratio_lag7_extra",
            family="extra",
            group="global",
            target_mode="ratio_lag7",
            target_transform="raw",
            params={"n_estimators": 500, "max_depth": 14, "min_samples_leaf": 10, "random_state": 17},
            note="Predict ratio to lag_7 instead of raw volume.",
        ),
        CandidateSpec(
            name="ridge_combo_slot",
            family="ridge",
            group="combo_slot",
            sample_weight_power=0.2,
            params={"alpha": 15.0},
            note="Linear control with very fine grouping.",
        ),
    ]
    return {spec.name: spec for spec in specs}


def ensemble_specs() -> dict[str, EnsembleSpec]:
    sota4 = ("sota_single_extra", "xgb_global", "mlp_global", "ratio_lag7_extra")
    specs = [
        EnsembleSpec(
            name="sota4_global",
            candidate_names=sota4,
            weight_scope="global",
            note="Reproduce current four-model style with global non-negative MAPE weights.",
        ),
        EnsembleSpec(
            name="sota4_combo_weights",
            candidate_names=sota4,
            weight_scope="combo",
            note="Learn one non-negative weight vector per tollgate-direction combo.",
        ),
        EnsembleSpec(
            name="sota4_block_weights",
            candidate_names=sota4,
            weight_scope="block",
            note="Learn separate morning and evening weights.",
        ),
        EnsembleSpec(
            name="sota4_combo_block_weights",
            candidate_names=sota4,
            weight_scope="combo_block",
            note="Learn weights for each combo and peak block; high variance risk.",
        ),
        EnsembleSpec(
            name="sota4_hour_weights",
            candidate_names=sota4,
            weight_scope="hour",
            note="Learn one weight vector for each target hour.",
        ),
        EnsembleSpec(
            name="sota4_slot_weights",
            candidate_names=sota4,
            weight_scope="slot",
            note="Learn one weight vector for each 20-minute target slot; high variance risk.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink10",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.10,
            note="Blend 90% block-weight prediction with 10% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink05",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.05,
            note="Blend 95% block-weight prediction with 5% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink15",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.15,
            note="Blend 85% block-weight prediction with 15% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink12",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.12,
            note="Blend 88% block-weight prediction with 12% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink14",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.14,
            note="Blend 86% block-weight prediction with 14% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink16",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.16,
            note="Blend 84% block-weight prediction with 16% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink18",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.18,
            note="Blend 82% block-weight prediction with 18% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink20",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.20,
            note="Blend 80% block-weight prediction with 20% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink25",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.25,
            note="Blend 75% block-weight prediction with 25% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink30",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.30,
            note="Blend 70% block-weight prediction with 30% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink40",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.40,
            note="Blend 60% block-weight prediction with 40% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink50",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.50,
            note="Blend 50% block-weight prediction with 50% global-weight prediction.",
        ),
        EnsembleSpec(
            name="sota4_block_global_blend",
            candidate_names=sota4,
            weight_scope="block",
            optimize_global_shrink=True,
            note="Learn a second-stage blend between block-weight and global-weight predictions.",
        ),
        EnsembleSpec(
            name="sota4_global_scale_global",
            candidate_names=sota4,
            weight_scope="global",
            post_scale_scope="global",
            note="Global four-model weights plus one calibration-fold prediction scale.",
        ),
        EnsembleSpec(
            name="sota4_block_scale_global",
            candidate_names=sota4,
            weight_scope="block",
            post_scale_scope="global",
            note="Block weights plus one global calibration-fold prediction scale.",
        ),
        EnsembleSpec(
            name="sota4_block_scale_block",
            candidate_names=sota4,
            weight_scope="block",
            post_scale_scope="block",
            note="Block weights plus separate morning/evening prediction scales.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink14_scale_global",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.14,
            post_scale_scope="global",
            note="14% global shrinkage plus one global calibration-fold prediction scale.",
        ),
        EnsembleSpec(
            name="sota4_block_shrink14_scale_block",
            candidate_names=sota4,
            weight_scope="block",
            shrink_to_global=0.14,
            post_scale_scope="block",
            note="14% global shrinkage plus separate morning/evening prediction scales.",
        ),
        EnsembleSpec(
            name="sota4_no_xgb",
            candidate_names=("sota_single_extra", "mlp_global", "ratio_lag7_extra"),
            weight_scope="global",
            note="Remove weak standalone XGBoost candidate.",
        ),
        EnsembleSpec(
            name="sota4_no_xgb_block_weights",
            candidate_names=("sota_single_extra", "mlp_global", "ratio_lag7_extra"),
            weight_scope="block",
            note="Remove XGBoost while learning separate morning/evening weights.",
        ),
        EnsembleSpec(
            name="sota4_no_mlp",
            candidate_names=("sota_single_extra", "xgb_global", "ratio_lag7_extra"),
            weight_scope="global",
            note="Remove MLP candidate.",
        ),
        EnsembleSpec(
            name="sota4_no_ratio",
            candidate_names=("sota_single_extra", "xgb_global", "mlp_global"),
            weight_scope="global",
            note="Remove lag_7 ratio candidate.",
        ),
        EnsembleSpec(
            name="sota4_plus_lgbm",
            candidate_names=sota4 + ("lgbm_global",),
            weight_scope="global",
            note="Add LightGBM as an extra diversity candidate.",
        ),
        EnsembleSpec(
            name="sota4_plus_lgbm_block_weights",
            candidate_names=sota4 + ("lgbm_global",),
            weight_scope="block",
            note="Add LightGBM with separate morning/evening weights.",
        ),
        EnsembleSpec(
            name="sota4_plus_boosting",
            candidate_names=sota4 + ("lgbm_global", "hgb_global"),
            weight_scope="global",
            note="Add LightGBM and HistGradientBoosting.",
        ),
        EnsembleSpec(
            name="sota4_plus_boosting_block_weights",
            candidate_names=sota4 + ("lgbm_global", "hgb_global"),
            weight_scope="block",
            note="Add LightGBM and HistGradientBoosting with block weights.",
        ),
        EnsembleSpec(
            name="sota4_plus_extra_variants",
            candidate_names=sota4 + ("extra_leaf6", "extra_lv_ratio07"),
            weight_scope="global",
            note="Add correlated ExtraTrees variants to test whether optimizer ignores or uses them.",
        ),
        EnsembleSpec(
            name="sota4_plus_extra_variants_block_weights",
            candidate_names=sota4 + ("extra_leaf6", "extra_lv_ratio07"),
            weight_scope="block",
            note="Add ExtraTrees variants with separate morning/evening weights.",
        ),
        EnsembleSpec(
            name="sota4_plus_ridge",
            candidate_names=sota4 + ("ridge_combo_slot",),
            weight_scope="global",
            note="Add a linear combo-slot candidate.",
        ),
        EnsembleSpec(
            name="sota4_plus_ridge_block_weights",
            candidate_names=sota4 + ("ridge_combo_slot",),
            weight_scope="block",
            note="Add a linear combo-slot candidate with block weights.",
        ),
        EnsembleSpec(
            name="replace_xgb_lgbm",
            candidate_names=("sota_single_extra", "lgbm_global", "mlp_global", "ratio_lag7_extra"),
            weight_scope="global",
            note="Replace XGBoost with LightGBM.",
        ),
    ]
    return {spec.name: spec for spec in specs}


PRESETS = {
    "quick": (
        "sota_single_extra",
        "extra_leaf6",
        "extra_depth18_leaf8",
        "extra_weight04",
        "hgb_global",
        "xgb_global",
    ),
    "tree": (
        "sota_single_extra",
        "extra_leaf6",
        "extra_depth18_leaf8",
        "extra_leaf14",
        "extra_weight04",
        "extra_global_leaf6",
        "extra_weather",
    ),
    "tuning": (
        "sota_single_extra",
        "extra_weight02",
        "extra_weight025",
        "extra_weight035",
        "extra_weight04",
        "extra_lv_ratio05",
        "extra_lv_ratio07",
        "extra_lv_ratio08",
    ),
    "boosting": (
        "hgb_global",
        "lgbm_global",
        "xgb_global",
    ),
    "ensemble_members": (
        "sota_single_extra",
        "xgb_global",
        "mlp_global",
        "ratio_lag7_extra",
        "lgbm_global",
        "hgb_global",
        "extra_leaf6",
        "extra_lv_ratio07",
        "ridge_combo_slot",
    ),
    "all": tuple(candidate_specs().keys()),
}


ENSEMBLE_PRESETS = {
    "quick": (
        "sota4_global",
        "sota4_combo_weights",
        "sota4_block_weights",
        "sota4_no_xgb",
        "sota4_plus_lgbm",
    ),
    "scopes": (
        "sota4_global",
        "sota4_combo_weights",
        "sota4_block_weights",
        "sota4_combo_block_weights",
    ),
    "granular_scopes": (
        "sota4_global",
        "sota4_block_weights",
        "sota4_hour_weights",
        "sota4_slot_weights",
        "sota4_combo_weights",
        "sota4_combo_block_weights",
    ),
    "shrinkage": (
        "sota4_global",
        "sota4_block_weights",
        "sota4_block_shrink10",
        "sota4_block_shrink20",
        "sota4_block_shrink30",
    ),
    "fine_shrinkage": (
        "sota4_global",
        "sota4_block_weights",
        "sota4_block_shrink05",
        "sota4_block_shrink10",
        "sota4_block_shrink15",
        "sota4_block_shrink20",
        "sota4_block_shrink25",
        "sota4_block_shrink30",
        "sota4_block_shrink40",
        "sota4_block_shrink50",
    ),
    "micro_shrinkage": (
        "sota4_block_shrink10",
        "sota4_block_shrink12",
        "sota4_block_shrink14",
        "sota4_block_shrink15",
        "sota4_block_shrink16",
        "sota4_block_shrink18",
        "sota4_block_shrink20",
    ),
    "second_stage": (
        "sota4_global",
        "sota4_block_weights",
        "sota4_block_shrink14",
        "sota4_block_shrink15",
        "sota4_block_global_blend",
    ),
    "post_scale": (
        "sota4_global",
        "sota4_global_scale_global",
        "sota4_block_weights",
        "sota4_block_scale_global",
        "sota4_block_scale_block",
        "sota4_block_shrink14",
        "sota4_block_shrink14_scale_global",
        "sota4_block_shrink14_scale_block",
    ),
    "members": (
        "sota4_global",
        "sota4_no_xgb",
        "sota4_no_mlp",
        "sota4_no_ratio",
        "sota4_plus_lgbm",
        "sota4_plus_boosting",
        "sota4_plus_extra_variants",
        "sota4_plus_ridge",
        "replace_xgb_lgbm",
    ),
    "block_members": (
        "sota4_block_weights",
        "sota4_no_xgb_block_weights",
        "sota4_plus_lgbm_block_weights",
        "sota4_plus_boosting_block_weights",
        "sota4_plus_extra_variants_block_weights",
        "sota4_plus_ridge_block_weights",
    ),
    "all": tuple(ensemble_specs().keys()),
}


def build_estimator(spec: CandidateSpec):
    params = dict(spec.params)
    if spec.family == "ridge":
        return RidgeRegressor(alpha=float(params.get("alpha", 20.0)))
    if spec.family == "extra":
        from sklearn.ensemble import ExtraTreesRegressor

        params.setdefault("random_state", 13)
        params.setdefault("n_jobs", -1)
        return ExtraTreesRegressor(**params)
    if spec.family == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        params.setdefault("random_state", 13)
        return HistGradientBoostingRegressor(**params)
    if spec.family == "lgbm":
        from lightgbm import LGBMRegressor

        params.setdefault("random_state", 13)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", -1)
        return LGBMRegressor(**params)
    if spec.family == "xgb":
        from xgboost import XGBRegressor

        params.setdefault("objective", "reg:squarederror")
        params.setdefault("random_state", 13)
        params.setdefault("n_jobs", -1)
        return XGBRegressor(**params)
    if spec.family == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        params.setdefault("random_state", 13)
        return make_pipeline(StandardScaler(), MLPRegressor(**params))
    raise ValueError(f"unknown experiment model family: {spec.family}")


def fit_estimator(estimator, x_train, target, sample_weight):
    if sample_weight is None:
        estimator.fit(x_train, target)
        return estimator
    try:
        estimator.fit(x_train, target, sample_weight=sample_weight)
    except (TypeError, ValueError):
        estimator.fit(x_train, target)
    return estimator


def train_and_predict_candidate(
    spec: CandidateSpec,
    train_agg,
    known_agg,
    weather,
    train_attr_agg,
    known_attr_agg,
    train_days,
    pred_rows,
    combos,
):
    if spec.group == "low_volume_block":
        low_volume_combos = select_low_volume_combos(
            train_agg,
            train_days,
            combos,
            ratio=spec.low_volume_ratio,
        )
        rows, global_preds, _ = train_and_predict_candidate(
            replace(spec, group="global"),
            train_agg,
            known_agg,
            weather,
            train_attr_agg,
            known_attr_agg,
            train_days,
            pred_rows,
            combos,
        )
        _, block_preds, _ = train_and_predict_candidate(
            replace(spec, group="block"),
            train_agg,
            known_agg,
            weather,
            train_attr_agg,
            known_attr_agg,
            train_days,
            pred_rows,
            combos,
        )
        preds = np.asarray(global_preds, dtype=float).copy()
        for idx, row in enumerate(rows):
            if row.combo in low_volume_combos:
                preds[idx] = block_preds[idx]
        return rows, preds, {"low_volume_combos": ",".join("_".join(combo) for combo in sorted(low_volume_combos))}

    train_rows = make_target_rows(train_days, combos)
    y_train = np.array([target_volume(train_agg, row) for row in train_rows], dtype=float)
    builder = FeatureBuilder(train_agg, weather, include_weather=spec.include_weather)
    builder.fit_stats(train_rows)
    train_features = builder.transform(train_rows, train_agg, train_attr_agg)
    pred_features = builder.transform(pred_rows, known_agg, known_attr_agg)
    if spec.drop_features:
        train_features = filter_features(train_features, spec.drop_features)
        pred_features = filter_features(pred_features, spec.drop_features)

    preds = np.zeros(len(pred_rows), dtype=float)
    artifacts = {"groups": 0}
    train_groups = {group_key(row, spec.group) for row in train_rows}
    pred_groups = {group_key(row, spec.group) for row in pred_rows}
    for key in sorted(train_groups | pred_groups):
        train_idx = [idx for idx, row in enumerate(train_rows) if group_key(row, spec.group) == key]
        pred_idx = [idx for idx, row in enumerate(pred_rows) if group_key(row, spec.group) == key]
        if not pred_idx:
            continue
        if not train_idx:
            raise ValueError(f"no training rows for group {key}")
        vectorizer = Vectorizer()
        x_train = vectorizer.fit_transform([train_features[idx] for idx in train_idx])
        x_pred = vectorizer.transform([pred_features[idx] for idx in pred_idx])
        target = y_train[train_idx]
        train_subset_features = [train_features[idx] for idx in train_idx]
        pred_subset_features = [pred_features[idx] for idx in pred_idx]
        sample_weight = None
        if spec.sample_weight_power > 0:
            denom = np.maximum(target, 1.0)
            sample_weight = (float(np.mean(denom)) / denom) ** spec.sample_weight_power
            sample_weight = sample_weight / np.mean(sample_weight)
        train_base = None
        pred_base = None
        if spec.target_mode == "ratio_lag7":
            train_base = np.array(
                [max(features.get("lag_7", 0.0), 1.0) for features in train_subset_features],
                dtype=float,
            )
            pred_base = np.array(
                [max(features.get("lag_7", 0.0), 1.0) for features in pred_subset_features],
                dtype=float,
            )
            target = np.log((target + 1.0) / train_base)
        elif spec.target_transform == "log":
            target = np.log1p(target)
        estimator = fit_estimator(build_estimator(spec), x_train, target, sample_weight)
        group_preds = np.asarray(estimator.predict(x_pred), dtype=float)
        if spec.target_mode == "ratio_lag7":
            group_preds = np.exp(group_preds) * pred_base - 1.0
        elif spec.target_transform == "log":
            group_preds = np.expm1(group_preds)
        preds[pred_idx] = np.maximum(group_preds, 0.0)
        artifacts["groups"] += 1
    return pred_rows, preds, artifacts


def load_phase1_validation_inputs(data_dir: Path):
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test1_obs = read_volume_aggregate([paths["test1_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    test1_attr = read_volume_attr_aggregate([paths["test1_volume"]])
    known = merge_aggregates(train1, test1_obs)
    known_attr = merge_attr_aggregates(train1_attr, test1_attr)
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase1"]])
    combos = infer_combos(train1)
    train_days = infer_dates(train1)
    valid_days = infer_dates(train2)
    valid_rows = make_target_rows(valid_days, combos)
    actual = np.array([target_volume(train2, row) for row in valid_rows], dtype=float)
    return train1, known, weather, train1_attr, known_attr, train_days, valid_rows, combos, actual


def load_ensemble_validation_inputs(data_dir: Path) -> dict[str, object]:
    paths = project_paths(data_dir)
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
    calibration_actual = np.array([target_volume(train1, row) for row in calibration_rows], dtype=float)

    valid_days = infer_dates(train2)
    validation_rows = make_target_rows(valid_days, combos)
    validation_actual = np.array([target_volume(train2, row) for row in validation_rows], dtype=float)
    validation_known = merge_aggregates(train1, test1_obs)
    validation_known_attr = merge_attr_aggregates(train1_attr, test1_attr)

    return {
        "weather": weather,
        "combos": combos,
        "calibration_train": calibration_train,
        "calibration_known": calibration_known,
        "calibration_train_attr": calibration_train_attr,
        "calibration_known_attr": calibration_known_attr,
        "calibration_train_days": calibration_train_days,
        "calibration_rows": calibration_rows,
        "calibration_actual": calibration_actual,
        "validation_train": train1,
        "validation_known": validation_known,
        "validation_train_attr": train1_attr,
        "validation_known_attr": validation_known_attr,
        "validation_train_days": train_days_all,
        "validation_rows": validation_rows,
        "validation_actual": validation_actual,
    }


def load_rolling_ensemble_inputs(data_dir: Path, validation_start) -> dict[str, object]:
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"]])
    combos = infer_combos(train1)
    train_days_all = infer_dates(train1)
    validation_end = validation_start + timedelta(days=6)
    validation_days = [day for day in train_days_all if validation_start <= day <= validation_end]
    if len(validation_days) != 7:
        raise ValueError(f"rolling validation fold must contain 7 days, got {len(validation_days)}")
    fold_train_days = [day for day in train_days_all if day < validation_start]
    if len(fold_train_days) < 14:
        raise ValueError("rolling validation fold needs at least 14 prior training days")

    calibration_train_days, calibration_days = latest_training_fold_split(fold_train_days)
    calibration_train = filter_days(train1, calibration_train_days)
    calibration_train_attr = filter_attr_days(train1_attr, calibration_train_days)
    calibration_known = merge_aggregates(calibration_train, observation_windows_only(train1, calibration_days))
    calibration_known_attr = merge_attr_aggregates(
        calibration_train_attr,
        attr_observation_windows_only(train1_attr, calibration_days),
    )
    calibration_rows = make_target_rows(calibration_days, combos)
    calibration_actual = np.array([target_volume(train1, row) for row in calibration_rows], dtype=float)

    validation_train = filter_days(train1, fold_train_days)
    validation_train_attr = filter_attr_days(train1_attr, fold_train_days)
    validation_known = merge_aggregates(validation_train, observation_windows_only(train1, validation_days))
    validation_known_attr = merge_attr_aggregates(
        validation_train_attr,
        attr_observation_windows_only(train1_attr, validation_days),
    )
    validation_rows = make_target_rows(validation_days, combos)
    validation_actual = np.array([target_volume(train1, row) for row in validation_rows], dtype=float)

    return {
        "weather": weather,
        "combos": combos,
        "calibration_train": calibration_train,
        "calibration_known": calibration_known,
        "calibration_train_attr": calibration_train_attr,
        "calibration_known_attr": calibration_known_attr,
        "calibration_train_days": calibration_train_days,
        "calibration_rows": calibration_rows,
        "calibration_actual": calibration_actual,
        "validation_train": validation_train,
        "validation_known": validation_known,
        "validation_train_attr": validation_train_attr,
        "validation_known_attr": validation_known_attr,
        "validation_train_days": fold_train_days,
        "validation_rows": validation_rows,
        "validation_actual": validation_actual,
        "fold_start": validation_days[0],
        "fold_end": validation_days[-1],
    }


def default_rolling_starts(data_dir: Path):
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train_days = infer_dates(train1)
    if len(train_days) < 28:
        raise ValueError("need at least 28 train1 days for two rolling 7-day folds")
    return [train_days[-14], train_days[-7]]


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


def ensemble_scope_key(row, scope: str):
    if scope == "global":
        return ("global",)
    if scope == "combo":
        return row.combo
    if scope == "block":
        return (block_name(row.start),)
    if scope == "combo_block":
        return row.combo + (block_name(row.start),)
    if scope == "hour":
        return (f"{row.start.hour:02d}",)
    if scope == "slot":
        return (f"{row.start.hour:02d}:{row.start.minute:02d}",)
    raise ValueError(f"unknown ensemble weight scope: {scope}")


def compute_candidate_pair(candidate_name: str, inputs: Mapping[str, object]):
    spec = candidate_specs()[candidate_name]
    calibration_rows, calibration_preds, calibration_artifacts = train_and_predict_candidate(
        spec,
        inputs["calibration_train"],
        inputs["calibration_known"],
        inputs["weather"],
        inputs["calibration_train_attr"],
        inputs["calibration_known_attr"],
        inputs["calibration_train_days"],
        inputs["calibration_rows"],
        inputs["combos"],
    )
    validation_rows, validation_preds, validation_artifacts = train_and_predict_candidate(
        spec,
        inputs["validation_train"],
        inputs["validation_known"],
        inputs["weather"],
        inputs["validation_train_attr"],
        inputs["validation_known_attr"],
        inputs["validation_train_days"],
        inputs["validation_rows"],
        inputs["combos"],
    )
    return {
        "calibration_rows": calibration_rows,
        "calibration_preds": calibration_preds,
        "calibration_artifacts": calibration_artifacts,
        "validation_rows": validation_rows,
        "validation_preds": validation_preds,
        "validation_artifacts": validation_artifacts,
    }


def build_ensemble_matrices(
    spec: EnsembleSpec,
    inputs: Mapping[str, object],
    prediction_cache: dict[str, Mapping[str, object]],
):
    calibration_columns = []
    validation_columns = []
    for candidate_name in spec.candidate_names:
        if candidate_name not in prediction_cache:
            prediction_cache[candidate_name] = compute_candidate_pair(candidate_name, inputs)
        pair = prediction_cache[candidate_name]
        calibration_columns.append(pair["calibration_preds"])
        validation_columns.append(pair["validation_preds"])
    return np.column_stack(calibration_columns), np.column_stack(validation_columns)


def apply_scoped_blend(
    spec: EnsembleSpec,
    calibration_rows,
    calibration_actual,
    calibration_matrix,
    validation_rows,
    validation_matrix,
):
    calibration_predictions = np.zeros(len(calibration_rows), dtype=float)
    validation_predictions = np.zeros(len(validation_rows), dtype=float)
    weights_by_scope = {}

    calibration_keys = [ensemble_scope_key(row, spec.weight_scope) for row in calibration_rows]
    validation_keys = [ensemble_scope_key(row, spec.weight_scope) for row in validation_rows]
    for scope_key in sorted(set(calibration_keys) | set(validation_keys)):
        calibration_idx = [idx for idx, key in enumerate(calibration_keys) if key == scope_key]
        validation_idx = [idx for idx, key in enumerate(validation_keys) if key == scope_key]
        if not calibration_idx:
            weights, _ = optimize_blend_weights(calibration_actual, calibration_matrix)
        else:
            weights, _ = optimize_blend_weights(
                calibration_actual[calibration_idx],
                calibration_matrix[calibration_idx],
            )
        weights_by_scope[scope_key] = weights
        if calibration_idx:
            calibration_predictions[calibration_idx] = calibration_matrix[calibration_idx] @ weights
        if validation_idx:
            validation_predictions[validation_idx] = validation_matrix[validation_idx] @ weights

    calibration_score = mape(calibration_actual, calibration_predictions)
    return calibration_predictions, validation_predictions, weights_by_scope, calibration_score


def format_weights(candidate_names: Sequence[str], weights_by_scope: Mapping[tuple, np.ndarray]) -> str:
    parts = []
    for scope_key, weights in weights_by_scope.items():
        key = "/".join(str(item) for item in scope_key)
        values = ",".join(f"{name}:{weight:.4f}" for name, weight in zip(candidate_names, weights))
        parts.append(f"{key}={values}")
    return " | ".join(parts)


def optimize_prediction_scale(actual, predictions):
    from scipy.optimize import minimize_scalar

    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    def objective(scale):
        return mape(actual, np.maximum(predictions * float(scale), 0.0))

    result = minimize_scalar(objective, bounds=(0.75, 1.25), method="bounded", options={"xatol": 1e-8})
    return float(result.x), float(result.fun)


def apply_post_scale(
    scope: str,
    calibration_rows,
    calibration_actual,
    calibration_predictions,
    validation_rows,
    validation_predictions,
):
    scaled_calibration = np.asarray(calibration_predictions, dtype=float).copy()
    scaled_validation = np.asarray(validation_predictions, dtype=float).copy()
    scales = {}
    calibration_keys = [ensemble_scope_key(row, scope) for row in calibration_rows]
    validation_keys = [ensemble_scope_key(row, scope) for row in validation_rows]
    for scope_key in sorted(set(calibration_keys) | set(validation_keys)):
        calibration_idx = [idx for idx, key in enumerate(calibration_keys) if key == scope_key]
        validation_idx = [idx for idx, key in enumerate(validation_keys) if key == scope_key]
        if not calibration_idx:
            scale = 1.0
        else:
            scale, _ = optimize_prediction_scale(
                calibration_actual[calibration_idx],
                scaled_calibration[calibration_idx],
            )
        scales[scope_key] = scale
        if calibration_idx:
            scaled_calibration[calibration_idx] = np.maximum(scaled_calibration[calibration_idx] * scale, 0.0)
        if validation_idx:
            scaled_validation[validation_idx] = np.maximum(scaled_validation[validation_idx] * scale, 0.0)
    return scaled_calibration, scaled_validation, scales


def validate_ensemble_spec_with_inputs(
    spec: EnsembleSpec,
    inputs: Mapping[str, object],
    prediction_cache: dict[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    start_time = time.perf_counter()
    cache = prediction_cache if prediction_cache is not None else {}
    calibration_matrix, validation_matrix = build_ensemble_matrices(spec, inputs, cache)
    calibration_predictions, validation_predictions, weights_by_scope, calibration_score = apply_scoped_blend(
        spec,
        inputs["calibration_rows"],
        inputs["calibration_actual"],
        calibration_matrix,
        inputs["validation_rows"],
        validation_matrix,
    )
    if spec.shrink_to_global > 0:
        global_spec = EnsembleSpec(
            name=f"{spec.name}_global_anchor",
            candidate_names=spec.candidate_names,
            weight_scope="global",
        )
        global_calibration_predictions, global_predictions, global_weights, _ = apply_scoped_blend(
            global_spec,
            inputs["calibration_rows"],
            inputs["calibration_actual"],
            calibration_matrix,
            inputs["validation_rows"],
            validation_matrix,
        )
        validation_predictions = (
            (1.0 - spec.shrink_to_global) * validation_predictions
            + spec.shrink_to_global * global_predictions
        )
        calibration_predictions = (
            (1.0 - spec.shrink_to_global) * calibration_predictions
            + spec.shrink_to_global * global_calibration_predictions
        )
        for scope_key, weights in global_weights.items():
            weights_by_scope[("global_anchor",) + scope_key] = weights
        calibration_score = mape(inputs["calibration_actual"], calibration_predictions)
    if spec.optimize_global_shrink:
        global_spec = EnsembleSpec(
            name=f"{spec.name}_global_anchor",
            candidate_names=spec.candidate_names,
            weight_scope="global",
        )
        global_calibration_predictions, global_predictions, global_weights, _ = apply_scoped_blend(
            global_spec,
            inputs["calibration_rows"],
            inputs["calibration_actual"],
            calibration_matrix,
            inputs["validation_rows"],
            validation_matrix,
        )
        second_stage_matrix = np.column_stack([calibration_predictions, global_calibration_predictions])
        second_stage_weights, calibration_score = optimize_blend_weights(
            inputs["calibration_actual"],
            second_stage_matrix,
        )
        validation_predictions = (
            second_stage_weights[0] * validation_predictions
            + second_stage_weights[1] * global_predictions
        )
        calibration_predictions = second_stage_matrix @ second_stage_weights
        weights_by_scope[("second_stage",)] = second_stage_weights
        for scope_key, weights in global_weights.items():
            weights_by_scope[("global_anchor",) + scope_key] = weights
    post_scales = {}
    if spec.post_scale_scope:
        calibration_predictions, validation_predictions, post_scales = apply_post_scale(
            spec.post_scale_scope,
            inputs["calibration_rows"],
            inputs["calibration_actual"],
            calibration_predictions,
            inputs["validation_rows"],
            validation_predictions,
        )
        calibration_score = mape(inputs["calibration_actual"], calibration_predictions)
    validation_actual = inputs["validation_actual"]
    single_scores = {
        name: mape(validation_actual, validation_matrix[:, idx])
        for idx, name in enumerate(spec.candidate_names)
    }
    score = mape(validation_actual, validation_predictions)
    elapsed = time.perf_counter() - start_time
    return {
        "name": spec.name,
        "weight_scope": spec.weight_scope,
        "shrink_to_global": spec.shrink_to_global,
        "optimize_global_shrink": spec.optimize_global_shrink,
        "post_scale_scope": spec.post_scale_scope,
        "candidate_names": ",".join(spec.candidate_names),
        "calibration_mape": calibration_score,
        "validation_mape": score,
        "actual_mean": float(np.mean(validation_actual)),
        "pred_mean": float(np.mean(validation_predictions)),
        "elapsed_sec": elapsed,
        "weights": format_weights(spec.candidate_names, weights_by_scope),
        "post_scales": "|".join(
            f"{'/'.join(str(item) for item in key)}:{value:.6f}" for key, value in post_scales.items()
        ),
        "single_scores": ",".join(f"{name}:{value:.6f}" for name, value in single_scores.items()),
        "note": spec.note,
    }


def validate_ensemble_spec(
    spec: EnsembleSpec,
    data_dir: Path,
    prediction_cache: dict[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    return validate_ensemble_spec_with_inputs(
        spec,
        load_ensemble_validation_inputs(data_dir),
        prediction_cache,
    )


def validate_candidate(spec: CandidateSpec, data_dir: Path, validation_output: Path | None = None) -> dict[str, object]:
    start_time = time.perf_counter()
    train1, known, weather, train1_attr, known_attr, train_days, valid_rows, combos, actual = (
        load_phase1_validation_inputs(data_dir)
    )
    rows, preds, artifacts = train_and_predict_candidate(
        spec,
        train1,
        known,
        weather,
        train1_attr,
        known_attr,
        train_days,
        valid_rows,
        combos,
    )
    preds = apply_history_blend(rows, preds, known, spec.history_blend, spec.prediction_scale)
    score = mape(actual, preds)
    elapsed = time.perf_counter() - start_time
    if validation_output is not None:
        write_submission(validation_output, rows, preds)
    return {
        "name": spec.name,
        "family": spec.family,
        "group": spec.group,
        "target_mode": spec.target_mode,
        "target_transform": spec.target_transform,
        "include_weather": spec.include_weather,
        "sample_weight_power": spec.sample_weight_power,
        "low_volume_ratio": spec.low_volume_ratio,
        "history_blend": spec.history_blend,
        "prediction_scale": spec.prediction_scale,
        "drop_feature_count": len(spec.drop_features),
        "validation_mape": score,
        "actual_mean": float(np.mean(actual)),
        "pred_mean": float(np.mean(preds)),
        "elapsed_sec": elapsed,
        "artifacts": artifacts,
        "note": spec.note,
    }


def write_results(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "family",
        "group",
        "target_mode",
        "target_transform",
        "include_weather",
        "sample_weight_power",
        "low_volume_ratio",
        "history_blend",
        "prediction_scale",
        "drop_feature_count",
        "validation_mape",
        "actual_mean",
        "pred_mean",
        "elapsed_sec",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["validation_mape"] = f"{float(row['validation_mape']):.6f}"
            row["actual_mean"] = f"{float(row['actual_mean']):.3f}"
            row["pred_mean"] = f"{float(row['pred_mean']):.3f}"
            row["elapsed_sec"] = f"{float(row['elapsed_sec']):.2f}"
            row.pop("artifacts", None)
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_ensemble_results(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "weight_scope",
        "shrink_to_global",
        "optimize_global_shrink",
        "post_scale_scope",
        "candidate_names",
        "calibration_mape",
        "validation_mape",
        "actual_mean",
        "pred_mean",
        "elapsed_sec",
        "weights",
        "post_scales",
        "single_scores",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["calibration_mape"] = f"{float(row['calibration_mape']):.6f}"
            row["validation_mape"] = f"{float(row['validation_mape']):.6f}"
            row["actual_mean"] = f"{float(row['actual_mean']):.3f}"
            row["pred_mean"] = f"{float(row['pred_mean']):.3f}"
            row["elapsed_sec"] = f"{float(row['elapsed_sec']):.2f}"
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def print_result(result: Mapping[str, object]) -> None:
    print(f"name={result['name']}")
    print(f"family={result['family']}")
    print(f"group={result['group']}")
    print(f"target_mode={result['target_mode']}")
    print(f"target_transform={result['target_transform']}")
    print(f"include_weather={result['include_weather']}")
    print(f"sample_weight_power={result['sample_weight_power']}")
    print(f"low_volume_ratio={result['low_volume_ratio']}")
    print(f"validation_rows=420")
    print(f"validation_mape={float(result['validation_mape']):.6f}")
    print(f"actual_mean={float(result['actual_mean']):.3f}")
    print(f"pred_mean={float(result['pred_mean']):.3f}")
    print(f"elapsed_sec={float(result['elapsed_sec']):.2f}")
    print(f"note={result['note']}")


def print_ensemble_result(result: Mapping[str, object]) -> None:
    print(f"name={result['name']}")
    print(f"weight_scope={result['weight_scope']}")
    print(f"shrink_to_global={result['shrink_to_global']}")
    print(f"optimize_global_shrink={result['optimize_global_shrink']}")
    print(f"post_scale_scope={result['post_scale_scope']}")
    print(f"candidate_names={result['candidate_names']}")
    print(f"calibration_mape={float(result['calibration_mape']):.6f}")
    print(f"validation_rows=420")
    print(f"validation_mape={float(result['validation_mape']):.6f}")
    print(f"actual_mean={float(result['actual_mean']):.3f}")
    print(f"pred_mean={float(result['pred_mean']):.3f}")
    print(f"elapsed_sec={float(result['elapsed_sec']):.2f}")
    print(f"weights={result['weights']}")
    print(f"post_scales={result['post_scales']}")
    print(f"single_scores={result['single_scores']}")
    print(f"note={result['note']}")


def list_candidates(args) -> None:
    specs = candidate_specs()
    names = PRESETS[args.preset] if args.preset else tuple(specs.keys())
    for name in names:
        spec = specs[name]
        print(f"{spec.name}\t{spec.family}\t{spec.group}\t{spec.note}")


def list_ensembles(args) -> None:
    specs = ensemble_specs()
    names = ENSEMBLE_PRESETS[args.preset] if args.preset else tuple(specs.keys())
    for name in names:
        spec = specs[name]
        print(f"{spec.name}\t{spec.weight_scope}\t{','.join(spec.candidate_names)}\t{spec.note}")


def validate_command(args) -> None:
    specs = candidate_specs()
    if args.candidate not in specs:
        raise SystemExit(f"unknown candidate: {args.candidate}")
    output = args.validation_output if args.validation_output else None
    result = validate_candidate(specs[args.candidate], args.data_dir, output)
    print_result(result)
    if output:
        print(f"validation_prediction={output}")


def sweep_command(args) -> None:
    specs = candidate_specs()
    names = list(PRESETS[args.preset])
    if args.limit:
        names = names[: args.limit]
    results = []
    for idx, name in enumerate(names, start=1):
        print(f"[{idx}/{len(names)}] running {name}")
        result = validate_candidate(specs[name], args.data_dir)
        results.append(result)
        print(f"  validation_mape={float(result['validation_mape']):.6f}")
    results.sort(key=lambda item: float(item["validation_mape"]))
    write_results(args.output, results)
    print(f"best={results[0]['name']}")
    print(f"best_validation_mape={float(results[0]['validation_mape']):.6f}")
    print(f"results={args.output}")


def ensemble_validate_command(args) -> None:
    specs = ensemble_specs()
    if args.ensemble not in specs:
        raise SystemExit(f"unknown ensemble: {args.ensemble}")
    result = validate_ensemble_spec(specs[args.ensemble], args.data_dir)
    print_ensemble_result(result)


def ensemble_sweep_command(args) -> None:
    specs = ensemble_specs()
    names = list(ENSEMBLE_PRESETS[args.preset])
    if args.limit:
        names = names[: args.limit]
    results = []
    prediction_cache: dict[str, Mapping[str, object]] = {}
    for idx, name in enumerate(names, start=1):
        print(f"[{idx}/{len(names)}] running {name}")
        result = validate_ensemble_spec(specs[name], args.data_dir, prediction_cache)
        results.append(result)
        print(f"  validation_mape={float(result['validation_mape']):.6f}")
    results.sort(key=lambda item: float(item["validation_mape"]))
    write_ensemble_results(args.output, results)
    print(f"best={results[0]['name']}")
    print(f"best_validation_mape={float(results[0]['validation_mape']):.6f}")
    print(f"results={args.output}")


def write_rolling_results(path: Path, summary_rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "mean_validation_mape",
        "min_validation_mape",
        "max_validation_mape",
        "fold_scores",
        "weight_scope",
        "shrink_to_global",
        "optimize_global_shrink",
        "post_scale_scope",
        "candidate_names",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in summary_rows:
            row = dict(result)
            for key in ("mean_validation_mape", "min_validation_mape", "max_validation_mape"):
                row[key] = f"{float(row[key]):.6f}"
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def rolling_ensemble_sweep_command(args) -> None:
    specs = ensemble_specs()
    names = list(ENSEMBLE_PRESETS[args.preset])
    if args.limit:
        names = names[: args.limit]
    starts = default_rolling_starts(args.data_dir)
    fold_results_by_name = {name: [] for name in names}
    for fold_idx, validation_start in enumerate(starts, start=1):
        inputs = load_rolling_ensemble_inputs(args.data_dir, validation_start)
        print(f"[fold {fold_idx}/{len(starts)}] {inputs['fold_start']}..{inputs['fold_end']}")
        prediction_cache: dict[str, Mapping[str, object]] = {}
        for name in names:
            result = validate_ensemble_spec_with_inputs(specs[name], inputs, prediction_cache)
            fold_results_by_name[name].append(result)
            print(f"  {name} validation_mape={float(result['validation_mape']):.6f}")

    summary_rows = []
    for name in names:
        fold_results = fold_results_by_name[name]
        scores = [float(result["validation_mape"]) for result in fold_results]
        spec = specs[name]
        summary_rows.append(
            {
                "name": name,
                "mean_validation_mape": float(np.mean(scores)),
                "min_validation_mape": float(np.min(scores)),
                "max_validation_mape": float(np.max(scores)),
                "fold_scores": ",".join(f"{score:.6f}" for score in scores),
                "weight_scope": spec.weight_scope,
                "shrink_to_global": spec.shrink_to_global,
                "optimize_global_shrink": spec.optimize_global_shrink,
                "post_scale_scope": spec.post_scale_scope,
                "candidate_names": ",".join(spec.candidate_names),
                "note": spec.note,
            }
        )
    summary_rows.sort(key=lambda item: float(item["mean_validation_mape"]))
    write_rolling_results(args.output, summary_rows)
    print(f"best={summary_rows[0]['name']}")
    print(f"best_mean_validation_mape={float(summary_rows[0]['mean_validation_mape']):.6f}")
    print(f"results={args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Isolated Task 2 exploration runner")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    sub = parser.add_subparsers(dest="command")

    list_parser = sub.add_parser("list", help="list experiment candidates")
    list_parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    list_parser.set_defaults(func=list_candidates)

    ensemble_list_parser = sub.add_parser("ensemble-list", help="list ensemble experiment candidates")
    ensemble_list_parser.add_argument("--preset", choices=sorted(ENSEMBLE_PRESETS), default=None)
    ensemble_list_parser.set_defaults(func=list_ensembles)

    validate_parser = sub.add_parser("validate", help="validate one experiment candidate on phase1")
    validate_parser.add_argument("candidate", choices=sorted(candidate_specs()))
    validate_parser.add_argument("--validation-output", type=Path, default=None)
    validate_parser.set_defaults(func=validate_command)

    sweep_parser = sub.add_parser("sweep", help="run a preset of experiment candidates")
    sweep_parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    sweep_parser.add_argument("--limit", type=int, default=0)
    sweep_parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_sweep_results.csv"))
    sweep_parser.set_defaults(func=sweep_command)

    ensemble_validate_parser = sub.add_parser("ensemble-validate", help="validate one ensemble experiment")
    ensemble_validate_parser.add_argument("ensemble", choices=sorted(ensemble_specs()))
    ensemble_validate_parser.set_defaults(func=ensemble_validate_command)

    ensemble_sweep_parser = sub.add_parser("ensemble-sweep", help="run a preset of ensemble experiments")
    ensemble_sweep_parser.add_argument("--preset", choices=sorted(ENSEMBLE_PRESETS), default="quick")
    ensemble_sweep_parser.add_argument("--limit", type=int, default=0)
    ensemble_sweep_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/src1_ensemble_sweep_results.csv"),
    )
    ensemble_sweep_parser.set_defaults(func=ensemble_sweep_command)

    rolling_ensemble_parser = sub.add_parser(
        "rolling-ensemble-sweep",
        help="run ensemble experiments on train1-only rolling folds",
    )
    rolling_ensemble_parser.add_argument("--preset", choices=sorted(ENSEMBLE_PRESETS), default="micro_shrinkage")
    rolling_ensemble_parser.add_argument("--limit", type=int, default=0)
    rolling_ensemble_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/src1_rolling_ensemble_sweep.csv"),
    )
    rolling_ensemble_parser.set_defaults(func=rolling_ensemble_sweep_command)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    args.func(args)
