from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import (
    infer_combos,
    infer_dates,
    load_weather,
    make_target_rows,
    make_target_rows_like_sample,
    merge_aggregates,
    merge_attr_aggregates,
    project_paths,
    read_sample_shape,
    read_volume_attr_aggregate,
    read_volume_aggregate,
    target_volume,
    write_submission,
)
from .features import FeatureBuilder, Vectorizer
from .model import make_regressor, mape


def group_key(row, group: str):
    if group == "global":
        return ("global",)
    if group == "block":
        return ("morning" if row.start.hour < 12 else "evening",)
    if group == "combo":
        return row.combo
    if group == "combo_block":
        return row.combo + ("morning" if row.start.hour < 12 else "evening",)
    if group == "combo_slot":
        return row.combo + (f"{row.start.hour:02d}:{row.start.minute:02d}",)
    raise ValueError(f"unknown group: {group}")


def train_and_predict(
    train_agg,
    known_agg,
    weather,
    train_attr_agg,
    known_attr_agg,
    train_days,
    pred_rows,
    combos,
    model_name,
    alpha,
    group,
    target_transform,
    include_weather,
    sample_weight_power,
):
    train_rows = make_target_rows(train_days, combos)
    y_train = np.array([target_volume(train_agg, row) for row in train_rows], dtype=float)

    builder = FeatureBuilder(train_agg, weather, include_weather=include_weather)
    builder.fit_stats(train_rows)
    train_features = builder.transform(train_rows, train_agg, train_attr_agg)
    pred_features = builder.transform(pred_rows, known_agg, known_attr_agg)

    preds = np.zeros(len(pred_rows), dtype=float)
    artifacts = []
    train_groups = {group_key(row, group) for row in train_rows}
    pred_groups = {group_key(row, group) for row in pred_rows}
    for key in sorted(train_groups | pred_groups):
        train_idx = [i for i, row in enumerate(train_rows) if group_key(row, group) == key]
        pred_idx = [i for i, row in enumerate(pred_rows) if group_key(row, group) == key]
        if not pred_idx:
            continue
        if not train_idx:
            raise ValueError(f"no training rows for group {key}")
        vectorizer = Vectorizer()
        x_train = vectorizer.fit_transform([train_features[i] for i in train_idx])
        x_pred = vectorizer.transform([pred_features[i] for i in pred_idx])
        target = y_train[train_idx]
        sample_weight = None
        if sample_weight_power > 0:
            denom = np.maximum(target, 1.0)
            sample_weight = (float(np.mean(denom)) / denom) ** sample_weight_power
            sample_weight = sample_weight / np.mean(sample_weight)
        if target_transform == "log":
            target = np.log1p(target)
        model = make_regressor(model_name, alpha=alpha).fit(x_train, target, sample_weight=sample_weight)
        group_preds = model.predict(x_pred)
        if target_transform == "log":
            group_preds = np.expm1(group_preds)
        preds[pred_idx] = np.maximum(group_preds, 0.0)
        artifacts.append((key, model, vectorizer))
    return pred_rows, preds, artifacts, builder


def validate(args) -> None:
    paths = project_paths(args.data_dir)
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

    rows, preds, _, _ = train_and_predict(
        train1,
        known,
        weather,
        train1_attr,
        known_attr,
        train_days,
        valid_rows,
        combos,
        args.model,
        args.alpha,
        args.group,
        args.target_transform,
        args.use_weather,
        args.sample_weight_power,
    )
    actual = np.array([target_volume(train2, row) for row in rows], dtype=float)
    score = mape(actual, preds)
    print(f"model={args.model}")
    print(f"group={args.group}")
    print(f"target_transform={args.target_transform}")
    print(f"use_weather={args.use_weather}")
    print(f"sample_weight_power={args.sample_weight_power}")
    print(f"validation_rows={len(rows)}")
    print(f"validation_mape={score:.6f}")
    print(f"actual_mean={actual.mean():.3f}")
    print(f"pred_mean={preds.mean():.3f}")

    if args.validation_output:
        write_submission(args.validation_output, rows, preds)
        print(f"validation_prediction={args.validation_output}")


def predict(args) -> None:
    paths = project_paths(args.data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test2_obs = read_volume_aggregate([paths["test2_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    train2_attr = read_volume_attr_aggregate([paths["train2_volume"]])
    test2_attr = read_volume_attr_aggregate([paths["test2_volume"]])
    train_all = merge_aggregates(train1, train2)
    known = merge_aggregates(train_all, test2_obs)
    train_all_attr = merge_attr_aggregates(train1_attr, train2_attr)
    known_attr = merge_attr_aggregates(train_all_attr, test2_attr)
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase2"]])
    _, sample_combos = read_sample_shape(paths["sample_volume"])
    combos = sample_combos or infer_combos(train_all)
    train_days = infer_dates(train_all)
    first_pred_day = min(infer_dates(test2_obs))
    pred_rows = make_target_rows_like_sample(paths["sample_volume"], first_pred_day)

    rows, preds, _, _ = train_and_predict(
        train_all,
        known,
        weather,
        train_all_attr,
        known_attr,
        train_days,
        pred_rows,
        combos,
        args.model,
        args.alpha,
        args.group,
        args.target_transform,
        args.use_weather,
        args.sample_weight_power,
    )
    write_submission(args.output, rows, preds)
    print(f"model={args.model}")
    print(f"group={args.group}")
    print(f"target_transform={args.target_transform}")
    print(f"use_weather={args.use_weather}")
    print(f"sample_weight_power={args.sample_weight_power}")
    print(f"train_rows={len(train_days) * len(combos) * 12}")
    print(f"prediction_rows={len(rows)}")
    print(f"submission={args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KDD Cup 2017 Task 2 baseline pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model", choices=["extra", "lgbm", "hgb", "ridge"], default="extra")
    parser.add_argument(
        "--group",
        choices=["global", "block", "combo", "combo_block", "combo_slot"],
        default="global",
    )
    parser.add_argument("--target-transform", choices=["log", "raw"], default="log")
    parser.add_argument("--use-weather", action="store_true", help="include weather features; off by default")
    parser.add_argument("--sample-weight-power", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=20.0)
    sub = parser.add_subparsers(dest="command")

    valid = sub.add_parser("validate", help="train on phase1 training data and validate on 2016-10-18..24")
    valid.add_argument("--validation-output", type=Path, default=Path("outputs/validation_phase1_pred.csv"))
    valid.set_defaults(func=validate)

    pred = sub.add_parser("predict", help="train on all released training data and predict phase2")
    pred.add_argument("--output", type=Path, default=Path("outputs/submission_task2_volume.csv"))
    pred.set_defaults(func=predict)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
