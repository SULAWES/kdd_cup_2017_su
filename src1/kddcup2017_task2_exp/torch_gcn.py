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

from kddcup2017_task2.data import infer_dates, merge_aggregates, merge_attr_aggregates
from kddcup2017_task2.model import mape
from kddcup2017_task2_exp.graph_gcn import build_graph_dataset, graph_adjacency, load_phase1_data, mape_sample_weight


@dataclass(frozen=True)
class TorchGraphResult:
    mode: str
    hidden: int
    embedding: int
    dropout: float
    lr: float
    best_epoch: int
    internal_mape: float
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


class GraphResidualRegressor(nn.Module):
    def __init__(self, n_features: int, n_nodes: int, hidden: int, embedding: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, embedding) * 0.02)
        in_features = n_features + embedding
        self.self_1 = nn.Linear(in_features, hidden)
        self.neighbor_1 = nn.Linear(in_features, hidden, bias=False)
        self.norm_1 = nn.LayerNorm(hidden)
        self.self_2 = nn.Linear(hidden, hidden)
        self.neighbor_2 = nn.Linear(hidden, hidden, bias=False)
        self.norm_2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, adjacency):
        batch = x.shape[0]
        emb = self.node_embedding.unsqueeze(0).expand(batch, -1, -1)
        h0 = torch.cat([x, emb], dim=-1)
        n0 = torch.einsum("ij,bjf->bif", adjacency, h0)
        h = torch.relu(self.norm_1(self.self_1(h0) + self.neighbor_1(n0)))
        h = self.dropout(h)
        n1 = torch.einsum("ij,bjh->bih", adjacency, h)
        h = torch.relu(self.norm_2(self.self_2(h) + self.neighbor_2(n1)))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32)


def predict(model, adjacency, x, target_mean: float, target_std: float):
    model.eval()
    with torch.no_grad():
        pred_z = model(to_tensor(x), to_tensor(adjacency)).cpu().numpy()
    pred_log = pred_z * target_std + target_mean
    return np.maximum(np.expm1(pred_log), 0.0)


def train_model(
    train_x,
    train_y,
    adjacency,
    hidden: int,
    embedding: int,
    dropout: float,
    lr: float,
    epochs: int,
    seed: int,
    eval_x=None,
    eval_y=None,
    weight_decay: float = 1e-4,
):
    torch.manual_seed(seed)
    model = GraphResidualRegressor(train_x.shape[2], train_x.shape[1], hidden, embedding, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    target_log = np.log1p(train_y)
    target_mean = float(target_log.mean())
    target_std = float(target_log.std())
    if target_std == 0:
        target_std = 1.0
    target = (target_log - target_mean) / target_std
    weights = mape_sample_weight(train_y)

    x_t = to_tensor(train_x)
    y_t = to_tensor(target)
    w_t = to_tensor(weights / weights.mean())
    a_t = to_tensor(adjacency)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = epochs
    best_score = float("inf")
    stale = 0
    patience = 180

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t, a_t)
        loss = (w_t * torch.nn.functional.smooth_l1_loss(pred, y_t, reduction="none")).sum() / w_t.sum()
        loss.backward()
        optimizer.step()

        if eval_x is not None and (epoch % 10 == 0 or epoch == epochs):
            eval_pred = predict(model, adjacency, eval_x, target_mean, target_std)
            score = mape(eval_y.reshape(-1), eval_pred.reshape(-1))
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 10
            if stale >= patience:
                break

    if eval_x is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_score, target_mean, target_std


def make_train_validation_graphs(data):
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
    return train_graph, validation_graph


def run_one(mode, hidden, embedding, dropout, lr, epochs, seed, data) -> TorchGraphResult:
    start_time = time.perf_counter()
    train_graph, validation_graph = make_train_validation_graphs(data)
    train_days = data["train_days"]
    combos = data["combos"]
    internal_start = train_days[-7]
    internal_dates = {day for day in train_days if day >= internal_start}
    internal_mask = np.array([start.date() in internal_dates for start in train_graph.starts])
    fit_mask = ~internal_mask
    adjacency = graph_adjacency(mode, combos, train_graph.y[fit_mask])
    _, best_epoch, internal_mape, _, _ = train_model(
        train_graph.x[fit_mask],
        train_graph.y[fit_mask],
        adjacency,
        hidden=hidden,
        embedding=embedding,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        seed=seed,
        eval_x=train_graph.x[internal_mask],
        eval_y=train_graph.y[internal_mask],
    )
    final_adjacency = graph_adjacency(mode, combos, train_graph.y)
    model, _, _, target_mean, target_std = train_model(
        train_graph.x,
        train_graph.y,
        final_adjacency,
        hidden=hidden,
        embedding=embedding,
        dropout=dropout,
        lr=lr,
        epochs=best_epoch,
        seed=seed,
    )
    predictions = predict(model, final_adjacency, validation_graph.x, target_mean, target_std)
    score = mape(validation_graph.y.reshape(-1), predictions.reshape(-1))
    return TorchGraphResult(
        mode=mode,
        hidden=hidden,
        embedding=embedding,
        dropout=dropout,
        lr=lr,
        best_epoch=best_epoch,
        internal_mape=float(internal_mape),
        validation_mape=float(score),
        pred_mean=float(predictions.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def write_results(path: Path, rows: Sequence[TorchGraphResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "hidden",
                "embedding",
                "dropout",
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
                    "embedding": row.embedding,
                    "dropout": row.dropout,
                    "lr": row.lr,
                    "best_epoch": row.best_epoch,
                    "internal_mape": f"{row.internal_mape:.6f}",
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Torch graph neural network exploration for Task 2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_torch_gcn_sweep.csv"))
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--modes", nargs="+", default=["identity", "topology", "corr", "full"])
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--embedding", type=int, default=8)
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args(argv)

    data = load_phase1_data(args.data_dir)
    results = []
    print(f"torch_version={torch.__version__}")
    for mode in args.modes:
        for hidden in args.hidden:
            for dropout in args.dropout:
                result = run_one(
                    mode,
                    hidden,
                    args.embedding,
                    dropout,
                    args.lr,
                    args.epochs,
                    args.seed,
                    data,
                )
                results.append(result)
                print(
                    f"mode={mode} hidden={hidden} dropout={dropout} "
                    f"internal_mape={result.internal_mape:.6f} "
                    f"validation_mape={result.validation_mape:.6f} "
                    f"best_epoch={result.best_epoch} pred_mean={result.pred_mean:.3f}"
                )
    results.sort(key=lambda item: item.validation_mape)
    write_results(args.output, results)
    print(f"best_validation_mape={results[0].validation_mape:.6f}")
    print(f"output={args.output}")
