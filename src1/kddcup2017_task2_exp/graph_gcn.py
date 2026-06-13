from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import (
    TargetRow,
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
from kddcup2017_task2.ensemble import observation_windows_only
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.model import mape
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features


@dataclass
class GraphDataset:
    starts: list
    combos: list[tuple[str, str]]
    x: np.ndarray
    y: np.ndarray


@dataclass
class TrainResult:
    mode: str
    hidden: int
    lr: float
    best_epoch: int
    internal_mape: float
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


@dataclass
class GraphFeatureResult:
    mode: str
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


def mape_sample_weight(y, power: float = 0.3):
    denom = np.maximum(np.asarray(y, dtype=float), 1.0)
    weights = (float(np.mean(denom)) / denom) ** power
    return weights / np.mean(weights)


def normalize_graph(adjacency):
    adjacency = np.asarray(adjacency, dtype=float)
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def graph_adjacency(mode: str, combos: Sequence[tuple[str, str]], train_y: np.ndarray):
    n_nodes = len(combos)
    if mode == "identity":
        return np.eye(n_nodes)
    if mode == "full":
        return normalize_graph(np.ones((n_nodes, n_nodes), dtype=float))
    if mode == "topology":
        adjacency = np.eye(n_nodes)
        for i, left in enumerate(combos):
            for j, right in enumerate(combos):
                if left[0] == right[0] or left[1] == right[1]:
                    adjacency[i, j] = 1.0
        return normalize_graph(adjacency)
    if mode == "corr":
        corr = np.corrcoef(train_y.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.maximum(corr, 0.0)
        return normalize_graph(corr)
    raise ValueError(f"unknown graph mode: {mode}")


def build_graph_dataset(
    label_agg,
    feature_train_agg,
    known_agg,
    feature_train_attr,
    known_attr,
    weather,
    train_days_for_stats,
    sample_days,
    combos,
    vectorizer=None,
    scaler=None,
):
    rows = make_target_rows(sample_days, combos)
    stats_rows = make_target_rows(train_days_for_stats, combos)
    builder = FeatureBuilder(feature_train_agg, weather, include_weather=False)
    builder.fit_stats(stats_rows)
    row_features = builder.transform(rows, known_agg, known_attr)
    row_features = filter_features(row_features, DEFAULT_DROP_FEATURES)

    fit_vectorizer = vectorizer is None
    if vectorizer is None:
        vectorizer = Vectorizer()
        feature_matrix = vectorizer.fit_transform(row_features)
    else:
        feature_matrix = vectorizer.transform(row_features)

    if scaler is None:
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        std[std == 0] = 1.0
        scaler = (mean, std)
    mean, std = scaler
    feature_matrix = (feature_matrix - mean) / std

    starts = sorted({row.start for row in rows})
    start_index = {start: idx for idx, start in enumerate(starts)}
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    x = np.zeros((len(starts), len(combos), feature_matrix.shape[1]), dtype=float)
    y = np.zeros((len(starts), len(combos)), dtype=float)

    for row_idx, row in enumerate(rows):
        sample_idx = start_index[row.start]
        node_idx = combo_index[row.combo]
        x[sample_idx, node_idx] = feature_matrix[row_idx]
        y[sample_idx, node_idx] = float(target_volume(label_agg, row))

    return GraphDataset(starts, list(combos), x, y), vectorizer, scaler


def relu(x):
    return np.maximum(x, 0.0)


def forward(params, adjacency, x):
    w0, b0, w1, b1 = params
    ax = np.einsum("ij,bjf->bif", adjacency, x)
    z1 = ax @ w0 + b0
    h = relu(z1)
    ah = np.einsum("ij,bjh->bih", adjacency, h)
    pred_log = (ah @ w1).squeeze(-1) + b1
    cache = ax, z1, h, ah
    return pred_log, cache


def adam_update(params, grads, moments, step, lr):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    new_params = []
    for idx, (param, grad) in enumerate(zip(params, grads)):
        m, v = moments[idx]
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        new_params.append(param - lr * m_hat / (np.sqrt(v_hat) + eps))
        moments[idx] = (m, v)
    return tuple(new_params)


def train_gcn(
    train_x,
    train_y,
    adjacency,
    hidden: int,
    lr: float,
    epochs: int,
    seed: int,
    eval_x=None,
    eval_y=None,
    l2: float = 1e-4,
):
    rng = np.random.default_rng(seed)
    n_features = train_x.shape[2]
    w0 = rng.normal(0.0, math.sqrt(2.0 / max(n_features, 1)), size=(n_features, hidden))
    b0 = np.zeros(hidden, dtype=float)
    w1 = rng.normal(0.0, math.sqrt(2.0 / max(hidden, 1)), size=(hidden, 1))
    b1 = np.array(0.0)
    params = (w0, b0, w1, b1)
    moments = [(np.zeros_like(param), np.zeros_like(param)) for param in params]

    target_log = np.log1p(train_y)
    target_mean = float(target_log.mean())
    target_std = float(target_log.std())
    if target_std == 0.0:
        target_std = 1.0
    target = (target_log - target_mean) / target_std
    weights = mape_sample_weight(train_y)
    weights = weights / weights.mean()
    best_params = (tuple(param.copy() for param in params), target_mean, target_std)
    best_epoch = epochs
    best_score = float("inf")
    patience = 120
    stale = 0

    for epoch in range(1, epochs + 1):
        pred_log, cache = forward(params, adjacency, train_x)
        diff = pred_log - target
        grad_pred = 2.0 * weights * diff / weights.sum()

        ax, z1, h, ah = cache
        w0, b0, w1, b1 = params
        grad_w1 = ah.reshape(-1, hidden).T @ grad_pred.reshape(-1, 1) + l2 * w1
        grad_b1 = np.asarray(grad_pred.sum())
        grad_ah = grad_pred[:, :, None] * w1.reshape(1, 1, hidden)
        grad_h = np.einsum("ji,bih->bjh", adjacency, grad_ah)
        grad_z1 = grad_h * (z1 > 0)
        grad_w0 = ax.reshape(-1, n_features).T @ grad_z1.reshape(-1, hidden) + l2 * w0
        grad_b0 = grad_z1.sum(axis=(0, 1))
        params = adam_update(params, (grad_w0, grad_b0, grad_w1, grad_b1), moments, epoch, lr)

        if eval_x is not None and (epoch % 10 == 0 or epoch == epochs):
            eval_pred = predict_gcn((params, target_mean, target_std), adjacency, eval_x)
            score = mape(eval_y.reshape(-1), eval_pred.reshape(-1))
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_params = (tuple(param.copy() for param in params), target_mean, target_std)
                stale = 0
            else:
                stale += 10
            if stale >= patience:
                break

    if eval_x is None:
        best_score = float("nan")
        best_params = (params, target_mean, target_std)
    return best_params, best_epoch, best_score


def predict_gcn(model, adjacency, x):
    if len(model) == 3 and isinstance(model[0], tuple):
        params, target_mean, target_std = model
    else:
        params = model
        target_mean = 0.0
        target_std = 1.0
    pred_log, _ = forward(params, adjacency, x)
    pred_log = pred_log * target_std + target_mean
    return np.maximum(np.expm1(pred_log), 0.0)


def load_phase1_data(data_dir: Path):
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test1_obs = read_volume_aggregate([paths["test1_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    test1_attr = read_volume_attr_aggregate([paths["test1_volume"]])
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase1"]])
    combos = infer_combos(train1)
    train_days = infer_dates(train1)
    valid_days = infer_dates(train2)
    return {
        "train1": train1,
        "train2": train2,
        "test1_obs": test1_obs,
        "train1_attr": train1_attr,
        "test1_attr": test1_attr,
        "weather": weather,
        "combos": combos,
        "train_days": train_days,
        "valid_days": valid_days,
    }


def run_one(mode: str, hidden: int, lr: float, epochs: int, seed: int, data) -> TrainResult:
    start_time = time.perf_counter()
    train1 = data["train1"]
    train2 = data["train2"]
    train1_attr = data["train1_attr"]
    test1_attr = data["test1_attr"]
    test1_obs = data["test1_obs"]
    weather = data["weather"]
    combos = data["combos"]
    train_days = data["train_days"]
    valid_days = data["valid_days"]

    train_graph, vectorizer, scaler = build_graph_dataset(
        train1,
        train1,
        train1,
        train1_attr,
        train1_attr,
        weather,
        train_days,
        train_days,
        combos,
    )
    validation_graph, _, _ = build_graph_dataset(
        train2,
        train1,
        merge_aggregates(train1, test1_obs),
        train1_attr,
        merge_attr_aggregates(train1_attr, test1_attr),
        weather,
        train_days,
        valid_days,
        combos,
        vectorizer=vectorizer,
        scaler=scaler,
    )

    latest_internal_start = max(train_days) - (max(train_days) - train_days[-7])
    internal_dates = {day for day in train_days if day >= latest_internal_start}
    internal_mask = np.array([start.date() in internal_dates for start in train_graph.starts])
    fit_mask = ~internal_mask
    adjacency = graph_adjacency(mode, combos, train_graph.y[fit_mask])
    _, best_epoch, internal_mape = train_gcn(
        train_graph.x[fit_mask],
        train_graph.y[fit_mask],
        adjacency,
        hidden=hidden,
        lr=lr,
        epochs=epochs,
        seed=seed,
        eval_x=train_graph.x[internal_mask],
        eval_y=train_graph.y[internal_mask],
    )

    final_adjacency = graph_adjacency(mode, combos, train_graph.y)
    final_params, _, _ = train_gcn(
        train_graph.x,
        train_graph.y,
        final_adjacency,
        hidden=hidden,
        lr=lr,
        epochs=best_epoch,
        seed=seed,
    )
    predictions = predict_gcn(final_params, final_adjacency, validation_graph.x)
    score = mape(validation_graph.y.reshape(-1), predictions.reshape(-1))
    return TrainResult(
        mode=mode,
        hidden=hidden,
        lr=lr,
        best_epoch=best_epoch,
        internal_mape=float(internal_mape),
        validation_mape=float(score),
        pred_mean=float(predictions.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def write_results(path: Path, rows: Sequence[TrainResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "hidden",
                "lr",
                "best_epoch",
                "internal_mape",
                "validation_mape",
                "pred_mean",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.mode,
                    "hidden": row.hidden,
                    "lr": row.lr,
                    "best_epoch": row.best_epoch,
                    "internal_mape": f"{row.internal_mape:.6f}",
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def graph_convolved_tabular_matrix(x, adjacency):
    ax = np.einsum("ij,bjf->bif", adjacency, x)
    augmented = np.concatenate([x, ax, ax - x], axis=2)
    return augmented.reshape(x.shape[0] * x.shape[1], augmented.shape[2])


def run_graph_feature_extra(mode: str, data) -> GraphFeatureResult:
    from sklearn.ensemble import ExtraTreesRegressor

    start_time = time.perf_counter()
    train1 = data["train1"]
    train2 = data["train2"]
    train1_attr = data["train1_attr"]
    test1_attr = data["test1_attr"]
    test1_obs = data["test1_obs"]
    weather = data["weather"]
    combos = data["combos"]
    train_days = data["train_days"]
    valid_days = data["valid_days"]

    train_graph, vectorizer, scaler = build_graph_dataset(
        train1,
        train1,
        train1,
        train1_attr,
        train1_attr,
        weather,
        train_days,
        train_days,
        combos,
    )
    validation_graph, _, _ = build_graph_dataset(
        train2,
        train1,
        merge_aggregates(train1, test1_obs),
        train1_attr,
        merge_attr_aggregates(train1_attr, test1_attr),
        weather,
        train_days,
        valid_days,
        combos,
        vectorizer=vectorizer,
        scaler=scaler,
    )
    adjacency = graph_adjacency(mode, combos, train_graph.y)
    x_train = graph_convolved_tabular_matrix(train_graph.x, adjacency)
    x_valid = graph_convolved_tabular_matrix(validation_graph.x, adjacency)
    y_train = train_graph.y.reshape(-1)
    y_valid = validation_graph.y.reshape(-1)
    model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=10,
        random_state=13,
        n_jobs=-1,
    )
    model.fit(x_train, np.log1p(y_train), sample_weight=mape_sample_weight(y_train))
    predictions = np.maximum(np.expm1(model.predict(x_valid)), 0.0)
    return GraphFeatureResult(
        mode=mode,
        validation_mape=float(mape(y_valid, predictions)),
        pred_mean=float(predictions.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def write_graph_feature_results(path: Path, rows: Sequence[GraphFeatureResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "validation_mape", "pred_mean", "elapsed_sec"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.mode,
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explore lightweight graph-convolution models for Task 2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_graph_gcn_sweep.csv"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--modes", nargs="+", default=["identity", "topology", "full", "corr"])
    parser.add_argument("--hidden", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--graph-feature-output",
        type=Path,
        default=Path("outputs/experiments/src1_graph_feature_extra.csv"),
    )
    args = parser.parse_args(argv)

    data = load_phase1_data(args.data_dir)
    results = []
    for mode in args.modes:
        for hidden in args.hidden:
            result = run_one(mode, hidden, args.lr, args.epochs, args.seed, data)
            results.append(result)
            print(
                f"mode={mode} hidden={hidden} internal_mape={result.internal_mape:.6f} "
                f"validation_mape={result.validation_mape:.6f} best_epoch={result.best_epoch} "
                f"pred_mean={result.pred_mean:.3f}"
            )
    results.sort(key=lambda item: item.validation_mape)
    write_results(args.output, results)
    print(f"best_validation_mape={results[0].validation_mape:.6f}")
    print(f"output={args.output}")

    feature_results = []
    for mode in args.modes:
        result = run_graph_feature_extra(mode, data)
        feature_results.append(result)
        print(
            f"graph_feature_extra mode={mode} validation_mape={result.validation_mape:.6f} "
            f"pred_mean={result.pred_mean:.3f}"
        )
    feature_results.sort(key=lambda item: item.validation_mape)
    write_graph_feature_results(args.graph_feature_output, feature_results)
    print(f"best_graph_feature_validation_mape={feature_results[0].validation_mape:.6f}")
    print(f"graph_feature_output={args.graph_feature_output}")
