from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import (
    OBS_TIMES,
    TargetRow,
    block_name,
    combine_date_time,
    floor_20min,
    infer_combos,
    infer_dates,
    load_weather,
    make_target_rows,
    merge_aggregates,
    merge_attr_aggregates,
    parse_dt,
    project_paths,
    read_volume_aggregate,
    read_volume_attr_aggregate,
    target_volume,
)
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.model import mape
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features, select_low_volume_combos
from kddcup2017_task2_exp.graph_gcn import mape_sample_weight


INTERSECTIONS = ("A", "B", "C")


@dataclass(frozen=True)
class TrajectoryStats:
    count: int = 0
    travel_sum: float = 0.0


@dataclass(frozen=True)
class TrajectoryResult:
    group: str
    include_route_means: bool
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


def read_trajectory_aggregate(paths: Sequence[Path]):
    totals = defaultdict(lambda: [0, 0.0])
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = floor_20min(parse_dt(row["starting_time"]))
                key = (start, str(row["intersection_id"]), str(row["tollgate_id"]))
                totals[key][0] += 1
                totals[key][1] += float(row["travel_time"])
    return {key: TrajectoryStats(count=value[0], travel_sum=value[1]) for key, value in totals.items()}


def merge_trajectory_aggregates(*aggregates):
    totals = defaultdict(lambda: [0, 0.0])
    for aggregate in aggregates:
        for key, value in aggregate.items():
            totals[key][0] += value.count
            totals[key][1] += value.travel_sum
    return {key: TrajectoryStats(count=value[0], travel_sum=value[1]) for key, value in totals.items()}


def filter_trajectory_days(aggregate, days):
    day_set = set(days)
    return {key: value for key, value in aggregate.items() if key[0].date() in day_set}


def trajectory_observation_windows_only(aggregate, days):
    day_set = set(days)
    clocks = set(OBS_TIMES["morning"] + OBS_TIMES["evening"])
    return {
        key: value
        for key, value in aggregate.items()
        if key[0].date() in day_set and key[0].time() in clocks
    }


def add_trajectory_features(features, row: TargetRow, trajectory_agg, include_route_means: bool):
    block = block_name(row.start)
    total_count = 0
    total_travel = 0.0
    obs_counts = []
    for obs_idx, clock in enumerate(OBS_TIMES[block]):
        obs_start = combine_date_time(row.start.date(), clock)
        obs_count = 0
        obs_travel = 0.0
        for intersection in INTERSECTIONS:
            stats = trajectory_agg.get((obs_start, intersection, row.tollgate_id), TrajectoryStats())
            obs_count += stats.count
            obs_travel += stats.travel_sum
            feature_prefix = f"traj_{intersection}_to_{row.tollgate_id}"
            features[f"{feature_prefix}_count_sum"] = features.get(f"{feature_prefix}_count_sum", 0.0) + stats.count
            features[f"{feature_prefix}_travel_sum"] = features.get(f"{feature_prefix}_travel_sum", 0.0) + stats.travel_sum
        features[f"traj_obs_{block}_{obs_idx}_count"] = float(obs_count)
        obs_counts.append(float(obs_count))
        total_count += obs_count
        total_travel += obs_travel
    features["traj_count_sum"] = float(total_count)
    features["traj_count_mean"] = float(total_count) / len(OBS_TIMES[block])
    features["traj_count_trend"] = obs_counts[-1] - obs_counts[0] if obs_counts else 0.0
    features["traj_travel_mean"] = float(total_travel) / max(float(total_count), 1.0)
    for intersection in INTERSECTIONS:
        prefix = f"traj_{intersection}_to_{row.tollgate_id}"
        count = features.get(f"{prefix}_count_sum", 0.0)
        features[f"{prefix}_count_share"] = count / max(float(total_count), 1.0)
        if include_route_means:
            features[f"{prefix}_travel_mean"] = features.get(f"{prefix}_travel_sum", 0.0) / max(count, 1.0)


def build_features(
    rows,
    feature_train_agg,
    known_agg,
    weather,
    feature_train_attr,
    known_attr,
    trajectory_agg,
    include_route_means: bool,
):
    builder = FeatureBuilder(feature_train_agg, weather, include_weather=False)
    builder.fit_stats(make_target_rows(infer_dates(feature_train_agg), sorted({row.combo for row in rows})))
    features = builder.transform(rows, known_agg, known_attr)
    for feature_row, target_row in zip(features, rows):
        add_trajectory_features(feature_row, target_row, trajectory_agg, include_route_means)
    return filter_features(features, DEFAULT_DROP_FEATURES)


def train_extra(train_features, y_train, pred_features, random_state=13):
    from sklearn.ensemble import ExtraTreesRegressor

    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_pred = vectorizer.transform(pred_features)
    model = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=14,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, np.log1p(y_train), sample_weight=mape_sample_weight(y_train))
    return np.maximum(np.expm1(model.predict(x_pred)), 0.0)


def train_predict_group(
    group,
    train_agg,
    known_agg,
    weather,
    train_attr,
    known_attr,
    train_traj,
    known_traj,
    train_days,
    pred_rows,
    combos,
    include_route_means,
):
    train_rows = make_target_rows(train_days, combos)
    y_train = np.array([target_volume(train_agg, row) for row in train_rows], dtype=float)
    train_features = build_features(
        train_rows,
        train_agg,
        train_agg,
        weather,
        train_attr,
        train_attr,
        train_traj,
        include_route_means,
    )
    pred_features = build_features(
        pred_rows,
        train_agg,
        known_agg,
        weather,
        train_attr,
        known_attr,
        known_traj,
        include_route_means,
    )
    if group == "global":
        return train_extra(train_features, y_train, pred_features)
    if group == "block":
        predictions = np.zeros(len(pred_rows), dtype=float)
        for block in ("morning", "evening"):
            train_idx = [idx for idx, row in enumerate(train_rows) if block_name(row.start) == block]
            pred_idx = [idx for idx, row in enumerate(pred_rows) if block_name(row.start) == block]
            if pred_idx:
                block_preds = train_extra(
                    [train_features[idx] for idx in train_idx],
                    y_train[train_idx],
                    [pred_features[idx] for idx in pred_idx],
                    random_state=17 if block == "evening" else 13,
                )
                predictions[pred_idx] = block_preds
        return predictions
    if group == "low_volume_block":
        global_preds = train_predict_group(
            "global",
            train_agg,
            known_agg,
            weather,
            train_attr,
            known_attr,
            train_traj,
            known_traj,
            train_days,
            pred_rows,
            combos,
            include_route_means,
        )
        block_preds = train_predict_group(
            "block",
            train_agg,
            known_agg,
            weather,
            train_attr,
            known_attr,
            train_traj,
            known_traj,
            train_days,
            pred_rows,
            combos,
            include_route_means,
        )
        low_volume_combos = select_low_volume_combos(train_agg, train_days, combos)
        predictions = global_preds.copy()
        for idx, row in enumerate(pred_rows):
            if row.combo in low_volume_combos:
                predictions[idx] = block_preds[idx]
        return predictions
    raise ValueError(f"unknown group: {group}")


def load_inputs(data_dir: Path):
    paths = project_paths(data_dir)
    return {
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


def run_one(group, include_route_means, data):
    start = time.perf_counter()
    train1 = data["train1"]
    train2 = data["train2"]
    test1_obs = data["test1_obs"]
    train1_attr = data["train1_attr"]
    test1_attr = data["test1_attr"]
    train1_traj = data["train1_traj"]
    test1_traj = data["test1_traj"]
    weather = data["weather"]
    known = merge_aggregates(train1, test1_obs)
    known_attr = merge_attr_aggregates(train1_attr, test1_attr)
    known_traj = merge_trajectory_aggregates(train1_traj, test1_traj)
    combos = infer_combos(train1)
    train_days = infer_dates(train1)
    valid_rows = make_target_rows(infer_dates(train2), combos)
    predictions = train_predict_group(
        group,
        train1,
        known,
        weather,
        train1_attr,
        known_attr,
        train1_traj,
        known_traj,
        train_days,
        valid_rows,
        combos,
        include_route_means,
    )
    actual = np.array([target_volume(train2, row) for row in valid_rows], dtype=float)
    return TrajectoryResult(
        group=group,
        include_route_means=include_route_means,
        validation_mape=float(mape(actual, predictions)),
        pred_mean=float(predictions.mean()),
        elapsed_sec=time.perf_counter() - start,
    )


def write_results(path: Path, rows: Sequence[TrajectoryResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "include_route_means", "validation_mape", "pred_mean", "elapsed_sec"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row.group,
                    "include_route_means": row.include_route_means,
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trajectory green-window feature exploration for Task 2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_trajectory_features.csv"))
    parser.add_argument("--groups", nargs="+", default=["global", "block", "low_volume_block"])
    args = parser.parse_args(argv)
    data = load_inputs(args.data_dir)
    results = []
    for group in args.groups:
        for include_route_means in (False, True):
            result = run_one(group, include_route_means, data)
            results.append(result)
            print(
                f"group={group} include_route_means={include_route_means} "
                f"validation_mape={result.validation_mape:.6f} pred_mean={result.pred_mean:.3f}"
            )
    results.sort(key=lambda item: item.validation_mape)
    write_results(args.output, results)
    print(f"best_validation_mape={results[0].validation_mape:.6f}")
    print(f"output={args.output}")
