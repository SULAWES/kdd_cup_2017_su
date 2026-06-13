from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

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
from kddcup2017_task2_exp.graph_gcn import graph_adjacency


@dataclass(frozen=True)
class MetaGraphData:
    starts: list
    combos: list[tuple[str, str]]
    rows: list
    candidates: np.ndarray
    actual: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class MetaResult:
    method: str
    mode: str
    hidden: int
    dropout: float
    lr: float
    calibration_mape: float
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32)


def rows_to_graph(rows, prediction_matrix, actual, combos):
    starts = sorted({row.start for row in rows})
    start_index = {start: idx for idx, start in enumerate(starts)}
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    candidates = np.zeros((len(starts), len(combos), prediction_matrix.shape[1]), dtype=float)
    y = np.zeros((len(starts), len(combos)), dtype=float)
    hour_values = sorted({row.start.hour for row in rows})
    hour_index = {hour: idx for idx, hour in enumerate(hour_values)}
    features = np.zeros((len(starts), len(combos), prediction_matrix.shape[1] + 8), dtype=float)
    for row_idx, row in enumerate(rows):
        sample_idx = start_index[row.start]
        node_idx = combo_index[row.combo]
        pred = np.maximum(prediction_matrix[row_idx], 0.0)
        candidates[sample_idx, node_idx] = pred
        y[sample_idx, node_idx] = float(actual[row_idx])
        log_pred = np.log1p(pred)
        base = max(pred[0], 1.0)
        ratios = np.log((pred + 1.0) / base)
        hour_one_hot = np.zeros(4, dtype=float)
        if row.start.hour in hour_index and hour_index[row.start.hour] < 4:
            hour_one_hot[hour_index[row.start.hour]] = 1.0
        features[sample_idx, node_idx] = np.concatenate(
            [
                log_pred,
                ratios,
                hour_one_hot,
            ]
        )
    mean = features.reshape(-1, features.shape[-1]).mean(axis=0)
    std = features.reshape(-1, features.shape[-1]).std(axis=0)
    std[std == 0.0] = 1.0
    features = (features - mean) / std
    return MetaGraphData(starts, list(combos), list(rows), candidates, y, features), mean, std


def rows_to_graph_with_scaler(rows, prediction_matrix, actual, combos, mean, std):
    graph, _, _ = rows_to_graph(rows, prediction_matrix, actual, combos)
    raw_features = graph.features * 1.0
    # Rebuild with the supplied scaler so validation never sets feature scale.
    starts = graph.starts
    start_index = {start: idx for idx, start in enumerate(starts)}
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    features = np.zeros_like(graph.features)
    for row_idx, row in enumerate(rows):
        sample_idx = start_index[row.start]
        node_idx = combo_index[row.combo]
        pred = np.maximum(prediction_matrix[row_idx], 0.0)
        log_pred = np.log1p(pred)
        base = max(pred[0], 1.0)
        ratios = np.log((pred + 1.0) / base)
        hour_one_hot = np.zeros(4, dtype=float)
        hour_map = {8: 0, 9: 1, 17: 2, 18: 3}
        if row.start.hour in hour_map:
            hour_one_hot[hour_map[row.start.hour]] = 1.0
        features[sample_idx, node_idx] = (np.concatenate([log_pred, ratios, hour_one_hot]) - mean) / std
    return MetaGraphData(starts, list(combos), list(rows), graph.candidates, graph.actual, features)


def load_meta_data(data_dir: Path):
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
    validation_actual = np.array([target_volume(train2, row) for row in validation_rows], dtype=float)

    calibration_graph, mean, std = rows_to_graph(calibration_rows, calibration_matrix, calibration_actual, combos)
    validation_graph = rows_to_graph_with_scaler(validation_rows, validation_matrix, validation_actual, combos, mean, std)
    return calibration_graph, validation_graph, combos


class GraphMetaBlend(nn.Module):
    def __init__(self, n_features: int, n_nodes: int, hidden: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, 8) * 0.02)
        in_features = n_features + 8
        self.self_1 = nn.Linear(in_features, hidden)
        self.neighbor_1 = nn.Linear(in_features, hidden, bias=False)
        self.self_2 = nn.Linear(hidden, hidden)
        self.neighbor_2 = nn.Linear(hidden, hidden, bias=False)
        self.norm_1 = nn.LayerNorm(hidden)
        self.norm_2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.weight_head = nn.Linear(hidden, len(ENSEMBLE_MODEL_NAMES))
        self.residual_head = nn.Linear(hidden, 1)

    def encode(self, x, adjacency):
        batch = x.shape[0]
        emb = self.node_embedding.unsqueeze(0).expand(batch, -1, -1)
        h0 = torch.cat([x, emb], dim=-1)
        n0 = torch.einsum("ij,bjf->bif", adjacency, h0)
        h = torch.relu(self.norm_1(self.self_1(h0) + self.neighbor_1(n0)))
        h = self.dropout(h)
        n1 = torch.einsum("ij,bjh->bih", adjacency, h)
        h = torch.relu(self.norm_2(self.self_2(h) + self.neighbor_2(n1)))
        return self.dropout(h)

    def forward_blend(self, x, adjacency, candidates):
        h = self.encode(x, adjacency)
        weights = torch.softmax(self.weight_head(h), dim=-1)
        return torch.sum(weights * candidates, dim=-1), weights

    def forward_residual(self, x, adjacency, base_pred):
        h = self.encode(x, adjacency)
        correction = 0.35 * torch.tanh(self.residual_head(h).squeeze(-1))
        return torch.clamp(base_pred * torch.exp(correction), min=0.0)


def mape_loss(actual, pred):
    return torch.mean(torch.abs(actual - pred) / torch.clamp(torch.abs(actual), min=1.0))


def train_meta_model(method, calibration, adjacency, hidden, dropout, lr, epochs, seed, base_pred=None):
    torch.manual_seed(seed)
    model = GraphMetaBlend(calibration.features.shape[-1], len(calibration.combos), hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x_t = to_tensor(calibration.features)
    y_t = to_tensor(calibration.actual)
    candidate_t = to_tensor(calibration.candidates)
    adjacency_t = to_tensor(adjacency)
    base_t = to_tensor(base_pred) if base_pred is not None else None
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = epochs
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        if method == "blend":
            pred, weights = model.forward_blend(x_t, adjacency_t, candidate_t)
            entropy_penalty = 0.0005 * torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, min=1e-8)), dim=-1))
            loss = mape_loss(y_t, pred) + entropy_penalty
        elif method == "residual":
            pred = model.forward_residual(x_t, adjacency_t, base_t)
            loss = mape_loss(y_t, pred)
        else:
            raise ValueError(f"unknown method: {method}")
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs:
            score = float(mape_loss(y_t, pred).detach().cpu().item())
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_score


def predict_meta(method, model, graph, adjacency, base_pred=None):
    model.eval()
    with torch.no_grad():
        if method == "blend":
            pred, _ = model.forward_blend(to_tensor(graph.features), to_tensor(adjacency), to_tensor(graph.candidates))
        else:
            pred = model.forward_residual(to_tensor(graph.features), to_tensor(adjacency), to_tensor(base_pred))
    return pred.cpu().numpy()


def run_one(method, mode, hidden, dropout, lr, epochs, seed, calibration, validation, combos):
    start_time = time.perf_counter()
    adjacency = graph_adjacency(mode, combos, calibration.actual)
    base_calibration = None
    base_validation = None
    if method == "residual":
        weights_by_scope, _, _ = optimize_scoped_blend_weights(
            calibration.actual.reshape(-1),
            calibration.candidates.reshape(-1, calibration.candidates.shape[-1]),
            calibration.rows,
            "hour",
        )
        base_calibration = apply_scoped_blend(
            calibration.candidates.reshape(-1, calibration.candidates.shape[-1]),
            calibration.rows,
            weights_by_scope,
            "hour",
        ).reshape(calibration.actual.shape)
        base_validation = apply_scoped_blend(
            validation.candidates.reshape(-1, validation.candidates.shape[-1]),
            validation.rows,
            weights_by_scope,
            "hour",
        ).reshape(validation.actual.shape)
    model, _, calibration_score = train_meta_model(
        method,
        calibration,
        adjacency,
        hidden,
        dropout,
        lr,
        epochs,
        seed,
        base_pred=base_calibration,
    )
    predictions = predict_meta(method, model, validation, adjacency, base_pred=base_validation)
    return MetaResult(
        method=method,
        mode=mode,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        calibration_mape=calibration_score,
        validation_mape=float(mape(validation.actual.reshape(-1), predictions.reshape(-1))),
        pred_mean=float(predictions.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def write_results(path: Path, rows: Sequence[MetaResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "mode",
                "hidden",
                "dropout",
                "lr",
                "calibration_mape",
                "validation_mape",
                "pred_mean",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "mode": row.mode,
                    "hidden": row.hidden,
                    "dropout": row.dropout,
                    "lr": row.lr,
                    "calibration_mape": f"{row.calibration_mape:.6f}",
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Torch graph meta-ensemble exploration for Task 2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_torch_meta_ensemble.csv"))
    parser.add_argument("--methods", nargs="+", default=["blend", "residual"])
    parser.add_argument("--modes", nargs="+", default=["identity", "topology", "corr", "full"])
    parser.add_argument("--hidden", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    calibration, validation, combos = load_meta_data(args.data_dir)
    results = []
    for method in args.methods:
        for mode in args.modes:
            for hidden in args.hidden:
                for dropout in args.dropout:
                    result = run_one(
                        method,
                        mode,
                        hidden,
                        dropout,
                        args.lr,
                        args.epochs,
                        args.seed,
                        calibration,
                        validation,
                        combos,
                    )
                    results.append(result)
                    print(
                        f"method={method} mode={mode} hidden={hidden} dropout={dropout} "
                        f"calibration_mape={result.calibration_mape:.6f} "
                        f"validation_mape={result.validation_mape:.6f} pred_mean={result.pred_mean:.3f}"
                    )
    results.sort(key=lambda item: item.validation_mape)
    write_results(args.output, results)
    print(f"best_validation_mape={results[0].validation_mape:.6f}")
    print(f"output={args.output}")
