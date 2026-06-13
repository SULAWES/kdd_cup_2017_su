from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from kddcup2017_task2.data import (
    OBS_TIMES,
    TargetRow,
    WindowKey,
    block_name,
    combine_date_time,
    infer_combos,
    infer_dates,
    make_target_rows,
    merge_aggregates,
    project_paths,
    read_volume_aggregate,
    target_volume,
)
from kddcup2017_task2.ensemble import observation_windows_only
from kddcup2017_task2.model import mape


@dataclass(frozen=True)
class SequenceNormalizer:
    hist_mean: np.ndarray
    hist_std: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    extras_mean: np.ndarray
    extras_std: np.ndarray


@dataclass(frozen=True)
class SequenceArrays:
    hist: np.ndarray
    obs: np.ndarray
    combo: np.ndarray
    hour: np.ndarray
    dow: np.ndarray
    minute: np.ndarray
    extras: np.ndarray
    y: np.ndarray
    normalizer: SequenceNormalizer


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    hidden: int
    dropout: float
    lr: float
    hist_days: int
    best_epoch: int
    internal_mape: float
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


def fit_array_normalizer(values) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(values, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) - mean) / std


def build_holdout_known(
    aggregate: Mapping[WindowKey, int],
    fit_days: Sequence,
    holdout_days: Sequence,
) -> dict[WindowKey, int]:
    fit_day_set = set(fit_days)
    fit_known = {key: value for key, value in aggregate.items() if key[0].date() in fit_day_set}
    holdout_obs = observation_windows_only(aggregate, holdout_days)
    return merge_aggregates(fit_known, holdout_obs)


def load_phase1_inputs(data_dir: Path):
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test1_obs = read_volume_aggregate([paths["test1_volume"]])
    combos = infer_combos(train1)
    train_days = infer_dates(train1)
    valid_days = infer_dates(train2)
    return {
        "train1": train1,
        "train2": train2,
        "test1_obs": test1_obs,
        "combos": combos,
        "train_days": train_days,
        "valid_days": valid_days,
        "train_rows": make_target_rows(train_days, combos),
        "valid_rows": make_target_rows(valid_days, combos),
        "known_validation": merge_aggregates(train1, test1_obs),
    }


def row_obs_values(known_agg: Mapping[WindowKey, int], row: TargetRow) -> list[float]:
    return [
        float(known_agg.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0))
        for clock in OBS_TIMES[block_name(row.start)]
    ]


def target_history_values(known_agg: Mapping[WindowKey, int], row: TargetRow, hist_days: int) -> list[float]:
    values = []
    for days_back in range(hist_days, 0, -1):
        start = row.start - timedelta(days=days_back)
        values.append(float(known_agg.get((start, row.tollgate_id, row.direction), 0)))
    return values


def build_sequence_arrays(
    rows: Sequence[TargetRow],
    known_agg: Mapping[WindowKey, int],
    actual_agg: Mapping[WindowKey, int],
    combos: Sequence[tuple[str, str]],
    hist_days: int,
    normalizer: SequenceNormalizer | None = None,
) -> SequenceArrays:
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    hour_index = {8: 0, 9: 1, 17: 2, 18: 3}
    minute_index = {0: 0, 20: 1, 40: 2}

    hist_log = np.log1p(np.asarray([target_history_values(known_agg, row, hist_days) for row in rows], dtype=float))
    obs_log = np.log1p(np.asarray([row_obs_values(known_agg, row) for row in rows], dtype=float))
    extras_raw = []
    for row, hist_values, obs_values in zip(rows, hist_log, obs_log):
        hist_tail = hist_values[-min(7, len(hist_values)) :] if len(hist_values) else hist_values
        obs_sum = float(np.sum(obs_values))
        hist_tail_mean = float(np.mean(hist_tail)) if len(hist_tail) else 0.0
        extras_raw.append(
            [
                1.0 if block_name(row.start) == "morning" else 0.0,
                1.0 if row.start.weekday() >= 5 else 0.0,
                obs_sum,
                float(np.mean(obs_values)) if len(obs_values) else 0.0,
                float(obs_values[-1] - obs_values[0]) if len(obs_values) else 0.0,
                hist_tail_mean,
                float(hist_values[-1]) if len(hist_values) else 0.0,
                obs_sum - hist_tail_mean,
            ]
        )
    extras = np.asarray(extras_raw, dtype=float)

    if normalizer is None:
        hist_mean, hist_std = fit_array_normalizer(hist_log)
        obs_mean, obs_std = fit_array_normalizer(obs_log)
        extras_mean, extras_std = fit_array_normalizer(extras)
        normalizer = SequenceNormalizer(hist_mean, hist_std, obs_mean, obs_std, extras_mean, extras_std)

    return SequenceArrays(
        hist=standardize(hist_log, normalizer.hist_mean, normalizer.hist_std).astype(np.float32),
        obs=standardize(obs_log, normalizer.obs_mean, normalizer.obs_std).astype(np.float32),
        combo=np.asarray([combo_index[row.combo] for row in rows], dtype=np.int64),
        hour=np.asarray([hour_index[row.start.hour] for row in rows], dtype=np.int64),
        dow=np.asarray([row.start.weekday() for row in rows], dtype=np.int64),
        minute=np.asarray([minute_index[row.start.minute] for row in rows], dtype=np.int64),
        extras=standardize(extras, normalizer.extras_mean, normalizer.extras_std).astype(np.float32),
        y=np.asarray([target_volume(actual_agg, row) for row in rows], dtype=np.float32),
        normalizer=normalizer,
    )


def subset_arrays(data: SequenceArrays, mask: np.ndarray) -> SequenceArrays:
    return SequenceArrays(
        hist=data.hist[mask],
        obs=data.obs[mask],
        combo=data.combo[mask],
        hour=data.hour[mask],
        dow=data.dow[mask],
        minute=data.minute[mask],
        extras=data.extras[mask],
        y=data.y[mask],
        normalizer=data.normalizer,
    )


def tensor_batch(data: SequenceArrays, device: torch.device):
    return (
        torch.as_tensor(data.hist, dtype=torch.float32, device=device),
        torch.as_tensor(data.obs, dtype=torch.float32, device=device),
        torch.as_tensor(data.combo, dtype=torch.long, device=device),
        torch.as_tensor(data.hour, dtype=torch.long, device=device),
        torch.as_tensor(data.dow, dtype=torch.long, device=device),
        torch.as_tensor(data.minute, dtype=torch.long, device=device),
        torch.as_tensor(data.extras, dtype=torch.float32, device=device),
    )


class StaticContext(nn.Module):
    def __init__(self, n_combos: int):
        super().__init__()
        self.combo_embedding = nn.Embedding(n_combos, 4)
        self.hour_embedding = nn.Embedding(4, 3)
        self.dow_embedding = nn.Embedding(7, 3)
        self.minute_embedding = nn.Embedding(3, 2)

    @property
    def width(self) -> int:
        return 4 + 3 + 3 + 2 + 8

    def forward(self, combo, hour, dow, minute, extras):
        return torch.cat(
            [
                self.combo_embedding(combo),
                self.hour_embedding(hour),
                self.dow_embedding(dow),
                self.minute_embedding(minute),
                extras,
            ],
            dim=-1,
        )


class LSTMRegressor(nn.Module):
    def __init__(self, hist_days: int, n_combos: int, hidden: int, dropout: float):
        super().__init__()
        del hist_days
        hist_hidden = max(4, hidden // 2)
        obs_hidden = max(4, hidden // 3)
        self.hist_encoder = nn.LSTM(1, hist_hidden, batch_first=True)
        self.obs_encoder = nn.LSTM(1, obs_hidden, batch_first=True)
        self.context = StaticContext(n_combos)
        head_in = hist_hidden + obs_hidden + self.context.width
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(4, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(4, hidden // 2), 1),
        )

    def forward(self, hist, obs, combo, hour, dow, minute, extras):
        _, (hist_state, _) = self.hist_encoder(hist.unsqueeze(-1))
        _, (obs_state, _) = self.obs_encoder(obs.unsqueeze(-1))
        context = self.context(combo, hour, dow, minute, extras)
        features = torch.cat([hist_state[-1], obs_state[-1], context], dim=-1)
        return self.head(features).squeeze(-1)


def choose_nhead(hidden: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden % candidate == 0:
            return candidate
    return 1


class TransformerRegressor(nn.Module):
    def __init__(self, hist_days: int, n_combos: int, hidden: int, dropout: float):
        super().__init__()
        self.hist_days = hist_days
        self.total_steps = hist_days + 6
        self.value_projection = nn.Linear(1, hidden)
        self.kind_embedding = nn.Embedding(2, hidden)
        self.position_embedding = nn.Embedding(self.total_steps, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=choose_nhead(hidden),
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.context = StaticContext(n_combos)
        self.head = nn.Sequential(
            nn.Linear(hidden + self.context.width, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(4, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(4, hidden // 2), 1),
        )

    def forward(self, hist, obs, combo, hour, dow, minute, extras):
        values = torch.cat([hist, obs], dim=1).unsqueeze(-1)
        batch_size = values.shape[0]
        positions = torch.arange(self.total_steps, device=values.device).unsqueeze(0).expand(batch_size, -1)
        kinds = torch.cat(
            [
                torch.zeros(self.hist_days, dtype=torch.long, device=values.device),
                torch.ones(6, dtype=torch.long, device=values.device),
            ]
        ).unsqueeze(0).expand(batch_size, -1)
        tokens = self.value_projection(values) + self.kind_embedding(kinds) + self.position_embedding(positions)
        encoded = self.encoder(tokens)
        sequence_vec = encoded.mean(dim=1)
        context = self.context(combo, hour, dow, minute, extras)
        return self.head(torch.cat([sequence_vec, context], dim=-1)).squeeze(-1)


def make_model(method: str, hist_days: int, n_combos: int, hidden: int, dropout: float) -> nn.Module:
    if method == "lstm":
        return LSTMRegressor(hist_days, n_combos, hidden, dropout)
    if method == "transformer":
        return TransformerRegressor(hist_days, n_combos, hidden, dropout)
    raise ValueError(f"unknown method: {method}")


def mape_sample_weight(y) -> np.ndarray:
    denom = np.maximum(np.asarray(y, dtype=float), 1.0)
    weights = (float(np.mean(denom)) / denom) ** 0.3
    return weights / np.mean(weights)


def target_z(y) -> tuple[np.ndarray, float, float]:
    y_log = np.log1p(np.asarray(y, dtype=float))
    mean = float(y_log.mean())
    std = float(y_log.std())
    if std < 1e-6:
        std = 1.0
    return ((y_log - mean) / std).astype(np.float32), mean, std


def predict_model(model: nn.Module, data: SequenceArrays, target_mean: float, target_std: float, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_z = model(*tensor_batch(data, device)).detach().cpu().numpy()
    return np.maximum(np.expm1(pred_z * target_std + target_mean), 0.0)


def train_with_early_stop(
    model: nn.Module,
    fit_data: SequenceArrays,
    eval_data: SequenceArrays,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    patience: int,
    eval_every: int,
) -> tuple[nn.Module, int, float, float, float]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    z, target_mean, target_std = target_z(fit_data.y)
    weights = mape_sample_weight(fit_data.y)
    y_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    tensors = tensor_batch(fit_data, device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = epochs
    best_score = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(*tensors)
        loss = (w_t * torch.nn.functional.smooth_l1_loss(pred, y_t, reduction="none")).sum() / w_t.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % eval_every == 0 or epoch == epochs:
            eval_pred = predict_model(model, eval_data, target_mean, target_std, device)
            score = mape(eval_data.y, eval_pred)
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += eval_every
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch, float(best_score), target_mean, target_std


def fit_fixed_epochs(
    model: nn.Module,
    train_data: SequenceArrays,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    z, target_mean, target_std = target_z(train_data.y)
    weights = mape_sample_weight(train_data.y)
    y_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    tensors = tensor_batch(train_data, device)
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(*tensors)
        loss = (w_t * torch.nn.functional.smooth_l1_loss(pred, y_t, reduction="none")).sum() / w_t.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return model, target_mean, target_std


def run_one(
    data,
    method: str,
    hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    seed: int,
    hist_days: int,
    eval_days: int,
    patience: int,
    eval_every: int,
    device: torch.device,
) -> ExperimentResult:
    start_time = time.perf_counter()
    train_days = data["train_days"]
    internal_days = train_days[-eval_days:]
    fit_days = train_days[:-eval_days]
    combos = data["combos"]

    fit_rows = make_target_rows(fit_days, combos)
    eval_rows = make_target_rows(internal_days, combos)
    fit_known = {key: value for key, value in data["train1"].items() if key[0].date() in set(fit_days)}
    eval_known = build_holdout_known(data["train1"], fit_days, internal_days)

    fit_arrays = build_sequence_arrays(fit_rows, fit_known, data["train1"], combos, hist_days)
    eval_arrays = build_sequence_arrays(
        eval_rows,
        eval_known,
        data["train1"],
        combos,
        hist_days,
        fit_arrays.normalizer,
    )
    model = make_model(method, hist_days, len(combos), hidden, dropout)
    _, best_epoch, internal_score, _, _ = train_with_early_stop(
        model,
        fit_arrays,
        eval_arrays,
        lr,
        epochs,
        seed,
        device,
        patience,
        eval_every,
    )

    final_train = build_sequence_arrays(data["train_rows"], data["train1"], data["train1"], combos, hist_days)
    validation = build_sequence_arrays(
        data["valid_rows"],
        data["known_validation"],
        data["train2"],
        combos,
        hist_days,
        final_train.normalizer,
    )
    final_model = make_model(method, hist_days, len(combos), hidden, dropout)
    final_model, target_mean, target_std = fit_fixed_epochs(
        final_model,
        final_train,
        lr,
        best_epoch,
        seed,
        device,
    )
    preds = predict_model(final_model, validation, target_mean, target_std, device)
    return ExperimentResult(
        method=method,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        hist_days=hist_days,
        best_epoch=best_epoch,
        internal_mape=internal_score,
        validation_mape=float(mape(validation.y, preds)),
        pred_mean=float(preds.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def write_results(path: Path, rows: Sequence[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "hidden",
        "dropout",
        "lr",
        "hist_days",
        "best_epoch",
        "internal_mape",
        "validation_mape",
        "pred_mean",
        "elapsed_sec",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "hidden": row.hidden,
                    "dropout": row.dropout,
                    "lr": row.lr,
                    "hist_days": row.hist_days,
                    "best_epoch": row.best_epoch,
                    "internal_mape": f"{row.internal_mape:.6f}",
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def print_result(result: ExperimentResult) -> None:
    print(
        f"method={result.method} hidden={result.hidden} dropout={result.dropout} lr={result.lr} "
        f"hist_days={result.hist_days} best_epoch={result.best_epoch} "
        f"internal_mape={result.internal_mape:.6f} validation_mape={result.validation_mape:.6f} "
        f"pred_mean={result.pred_mean:.3f} elapsed_sec={result.elapsed_sec:.2f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="src2 exploratory LSTM/Transformer sequence models for Task 2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src2_sequence_nn_sweep.csv"))
    parser.add_argument("--methods", nargs="+", choices=["lstm", "transformer"], default=["lstm", "transformer"])
    parser.add_argument("--hidden", nargs="+", type=int, default=[32])
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.1])
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hist-days", type=int, default=21)
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args(argv)

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    data = load_phase1_inputs(args.data_dir)
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"device={device}")
    print("leakage_check=train1 internal holdout exposes only same-day green windows; train2 labels used only for final scoring")

    results = []
    for method in args.methods:
        for hidden in args.hidden:
            for dropout in args.dropout:
                result = run_one(
                    data,
                    method,
                    hidden,
                    dropout,
                    args.lr,
                    args.epochs,
                    args.seed,
                    args.hist_days,
                    args.eval_days,
                    args.patience,
                    args.eval_every,
                    device,
                )
                results.append(result)
                print_result(result)
    results.sort(key=lambda row: row.validation_mape)
    write_results(args.output, results)
    print(f"best_validation_mape={results[0].validation_mape:.6f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
