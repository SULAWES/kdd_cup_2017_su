from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from kddcup2017_task2.data import (
    OBS_TIMES,
    TargetRow,
    block_name,
    combine_date_time,
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
from kddcup2017_task2.ensemble import apply_scoped_blend, optimize_scoped_blend_weights
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.model import mape
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features
from kddcup2017_task2_exp.observation_adjust_exp import fit_expected_obs, observation_strengths
from kddcup2017_task2_exp.trajectory_ensemble_exp import (
    apply_scoped_weights,
    build_candidate_matrices,
    load_inputs as load_trajectory_inputs,
    optimize_scoped_capped_blend,
)


@dataclass(frozen=True)
class NNResult:
    method: str
    architecture: str
    detail: str
    hidden: int
    dropout: float
    lr: float
    best_epoch: int
    internal_mape: float
    calibration_mape: float
    validation_mape: float
    pred_mean: float
    elapsed_sec: float


def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32)


def mape_sample_weight(y):
    denom = np.maximum(np.asarray(y, dtype=float), 1.0)
    weights = (float(np.mean(denom)) / denom) ** 0.3
    return weights / np.mean(weights)


def fit_standardizer(x):
    mean = np.asarray(x, dtype=float).mean(axis=0)
    std = np.asarray(x, dtype=float).std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def standardize(x, mean, std):
    return (np.asarray(x, dtype=float) - mean) / std


def load_phase1_inputs(data_dir: Path):
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
    train_rows = make_target_rows(train_days, combos)
    valid_rows = make_target_rows(valid_days, combos)
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
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "known_validation": merge_aggregates(train1, test1_obs),
        "known_attr_validation": merge_attr_aggregates(train1_attr, test1_attr),
    }


class DenseNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float, architecture: str):
        super().__init__()
        self.architecture = architecture
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU())
        if architecture == "tab_mlp":
            self.body = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
            self.out = nn.Linear(hidden // 2, 1)
        elif architecture == "tab_resnet":
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(hidden, hidden),
                        nn.LayerNorm(hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                    )
                    for _ in range(3)
                ]
            )
            self.out = nn.Linear(hidden, 1)
        else:
            raise ValueError(f"unknown dense architecture: {architecture}")

    def forward(self, x):
        h = self.input(x)
        if self.architecture == "tab_mlp":
            h = self.body(h)
        else:
            for block in self.blocks:
                h = torch.relu(h + 0.4 * block(h))
        return self.out(h).squeeze(-1)


class SequenceNet(nn.Module):
    def __init__(self, hidden: int, dropout: float, architecture: str, n_combos: int):
        super().__init__()
        self.architecture = architecture
        self.combo_embedding = nn.Embedding(n_combos, 4)
        self.hour_embedding = nn.Embedding(4, 3)
        self.dow_embedding = nn.Embedding(7, 3)
        self.minute_embedding = nn.Embedding(3, 2)
        if architecture == "seq_gru":
            self.hist_encoder = nn.GRU(1, hidden, batch_first=True)
            self.obs_encoder = nn.GRU(1, hidden // 2, batch_first=True)
            seq_features = hidden + hidden // 2
        elif architecture == "seq_conv":
            self.hist_encoder = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.obs_encoder = nn.Sequential(
                nn.Conv1d(1, hidden // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            seq_features = hidden + hidden // 2
        else:
            raise ValueError(f"unknown sequence architecture: {architecture}")
        cat_features = 4 + 3 + 3 + 2 + 4
        self.head = nn.Sequential(
            nn.Linear(seq_features + cat_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, hist, obs, combo, hour, dow, minute, extras):
        if self.architecture == "seq_gru":
            _, hist_h = self.hist_encoder(hist.unsqueeze(-1))
            _, obs_h = self.obs_encoder(obs.unsqueeze(-1))
            hist_vec = hist_h[-1]
            obs_vec = obs_h[-1]
        else:
            hist_vec = self.hist_encoder(hist.unsqueeze(1)).squeeze(-1)
            obs_vec = self.obs_encoder(obs.unsqueeze(1)).squeeze(-1)
        h = torch.cat(
            [
                hist_vec,
                obs_vec,
                self.combo_embedding(combo),
                self.hour_embedding(hour),
                self.dow_embedding(dow),
                self.minute_embedding(minute),
                extras,
            ],
            dim=-1,
        )
        return self.head(h).squeeze(-1)


class CalibratorNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float, correction_scale: float):
        super().__init__()
        self.correction_scale = correction_scale
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, base):
        correction = self.correction_scale * torch.tanh(self.net(x).squeeze(-1))
        return torch.clamp(base * torch.exp(correction), min=0.0)


class BlendGateNet(nn.Module):
    def __init__(self, n_features: int, n_models: int, hidden: int, dropout: float, gate_scale: float):
        super().__init__()
        self.gate_scale = gate_scale
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_models),
        )

    def forward(self, x, candidates, prior_weights):
        prior_logits = torch.log(torch.clamp(prior_weights, min=1e-5))
        delta = self.gate_scale * torch.tanh(self.net(x))
        weights = torch.softmax(prior_logits + delta, dim=-1)
        return torch.sum(weights * candidates, dim=-1)


def train_z_model(
    model,
    train_x,
    train_y,
    eval_x,
    eval_y,
    lr: float,
    epochs: int,
    seed: int,
    sequence: bool = False,
):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    target_log = np.log1p(train_y)
    target_mean = float(target_log.mean())
    target_std = float(target_log.std())
    if target_std == 0.0:
        target_std = 1.0
    target = (target_log - target_mean) / target_std
    weights = mape_sample_weight(train_y)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = epochs
    stale = 0
    patience = 220

    y_t = to_tensor(target)
    w_t = to_tensor(weights)
    if sequence:
        tensors = sequence_tensors(train_x)
    else:
        tensors = (to_tensor(train_x),)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(*tensors)
        loss = (w_t * torch.nn.functional.smooth_l1_loss(pred, y_t, reduction="none")).sum() / w_t.sum()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            pred_eval = predict_z_model(model, eval_x, target_mean, target_std, sequence=sequence)
            score = mape(eval_y, pred_eval)
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 10
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, target_mean, target_std


def predict_z_model(model, x, target_mean: float, target_std: float, sequence: bool = False):
    model.eval()
    with torch.no_grad():
        tensors = sequence_tensors(x) if sequence else (to_tensor(x),)
        pred_z = model(*tensors).cpu().numpy()
    return np.maximum(np.expm1(pred_z * target_std + target_mean), 0.0)


def sequence_tensors(data):
    return (
        to_tensor(data["hist"]),
        to_tensor(data["obs"]),
        torch.as_tensor(data["combo"], dtype=torch.long),
        torch.as_tensor(data["hour"], dtype=torch.long),
        torch.as_tensor(data["dow"], dtype=torch.long),
        torch.as_tensor(data["minute"], dtype=torch.long),
        to_tensor(data["extras"]),
    )


def subset_sequence(data, mask):
    return {key: value[mask] for key, value in data.items()}


def build_tabular_arrays(data):
    builder = FeatureBuilder(data["train1"], data["weather"], include_weather=False)
    builder.fit_stats(data["train_rows"])
    train_features = builder.transform(data["train_rows"], data["train1"], data["train1_attr"])
    valid_features = builder.transform(
        data["valid_rows"],
        data["known_validation"],
        data["known_attr_validation"],
    )
    train_features = filter_features(train_features, DEFAULT_DROP_FEATURES)
    valid_features = filter_features(valid_features, DEFAULT_DROP_FEATURES)
    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_valid = vectorizer.transform(valid_features)
    y_train = np.array([target_volume(data["train1"], row) for row in data["train_rows"]], dtype=float)
    y_valid = np.array([target_volume(data["train2"], row) for row in data["valid_rows"]], dtype=float)
    return x_train, y_train, x_valid, y_valid


def fit_final_dense(architecture, hidden, dropout, lr, epochs, seed, x_train, y_train, x_valid):
    mean, std = fit_standardizer(x_train)
    target_log = np.log1p(y_train)
    target_mean = float(target_log.mean())
    target_std = float(target_log.std()) or 1.0
    target = (target_log - target_mean) / target_std
    weights = mape_sample_weight(y_train)
    torch.manual_seed(seed)
    model = DenseNet(x_train.shape[1], hidden, dropout, architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x_t = to_tensor(standardize(x_train, mean, std))
    y_t = to_tensor(target)
    w_t = to_tensor(weights)
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t)
        loss = (w_t * torch.nn.functional.smooth_l1_loss(pred, y_t, reduction="none")).sum() / w_t.sum()
        loss.backward()
        optimizer.step()
    preds = predict_z_model(model, standardize(x_valid, mean, std), target_mean, target_std)
    return preds


def run_tabular(data, architecture, hidden, dropout, lr, epochs, seed) -> NNResult:
    start_time = time.perf_counter()
    x_train, y_train, x_valid, y_valid = build_tabular_arrays(data)
    internal_start = data["train_days"][-7]
    internal_mask = np.array([row.start.date() >= internal_start for row in data["train_rows"]])
    fit_mask = ~internal_mask
    mean, std = fit_standardizer(x_train[fit_mask])
    x_fit = standardize(x_train[fit_mask], mean, std)
    x_eval = standardize(x_train[internal_mask], mean, std)
    model = DenseNet(x_train.shape[1], hidden, dropout, architecture)
    _, best_epoch, internal_score, _, _ = train_z_model(
        model,
        x_fit,
        y_train[fit_mask],
        x_eval,
        y_train[internal_mask],
        lr,
        epochs,
        seed,
    )
    preds = fit_final_dense(architecture, hidden, dropout, lr, best_epoch, seed, x_train, y_train, x_valid)
    return NNResult(
        method="direct",
        architecture=architecture,
        detail="feature_builder",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        best_epoch=best_epoch,
        internal_mape=float(internal_score),
        calibration_mape=float("nan"),
        validation_mape=float(mape(y_valid, preds)),
        pred_mean=float(preds.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def row_obs_values(known_agg: Mapping, row: TargetRow):
    return [
        float(known_agg.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0))
        for clock in OBS_TIMES[block_name(row.start)]
    ]


def target_history_values(known_agg: Mapping, row: TargetRow, days: int = 14):
    from datetime import timedelta

    values = []
    for day_back in range(days, 0, -1):
        start = row.start - timedelta(days=day_back)
        values.append(float(known_agg.get((start, row.tollgate_id, row.direction), 0)))
    return values


def build_sequence_arrays(rows, known_agg, actual_agg, combos, hist_mean=None, hist_std=None, obs_mean=None, obs_std=None):
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    hour_index = {8: 0, 9: 1, 17: 2, 18: 3}
    minute_index = {0: 0, 20: 1, 40: 2}
    hist = np.array([target_history_values(known_agg, row) for row in rows], dtype=float)
    obs = np.array([row_obs_values(known_agg, row) for row in rows], dtype=float)
    hist = np.log1p(hist)
    obs = np.log1p(obs)
    if hist_mean is None:
        hist_mean = hist.mean(axis=0)
        hist_std = hist.std(axis=0)
        hist_std[hist_std == 0.0] = 1.0
        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0)
        obs_std[obs_std == 0.0] = 1.0
    hist_z = (hist - hist_mean) / hist_std
    obs_z = (obs - obs_mean) / obs_std
    extras = []
    for row, obs_raw, hist_raw in zip(rows, obs, hist):
        extras.append(
            [
                1.0 if block_name(row.start) == "morning" else 0.0,
                1.0 if row.start.weekday() >= 5 else 0.0,
                float(np.mean(obs_raw)),
                float(np.mean(hist_raw[-7:])),
            ]
        )
    y = np.array([target_volume(actual_agg, row) for row in rows], dtype=float)
    data = {
        "hist": hist_z.astype(np.float32),
        "obs": obs_z.astype(np.float32),
        "combo": np.array([combo_index[row.combo] for row in rows], dtype=np.int64),
        "hour": np.array([hour_index[row.start.hour] for row in rows], dtype=np.int64),
        "dow": np.array([row.start.weekday() for row in rows], dtype=np.int64),
        "minute": np.array([minute_index[row.start.minute] for row in rows], dtype=np.int64),
        "extras": np.asarray(extras, dtype=np.float32),
    }
    return data, y, hist_mean, hist_std, obs_mean, obs_std


def fit_final_sequence(architecture, hidden, dropout, lr, epochs, seed, train_data, y_train, valid_data, n_combos):
    torch.manual_seed(seed)
    model = SequenceNet(hidden, dropout, architecture, n_combos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    target_log = np.log1p(y_train)
    target_mean = float(target_log.mean())
    target_std = float(target_log.std()) or 1.0
    target = (target_log - target_mean) / target_std
    weights = mape_sample_weight(y_train)
    tensors = sequence_tensors(train_data)
    y_t = to_tensor(target)
    w_t = to_tensor(weights)
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(*tensors)
        loss = (w_t * torch.nn.functional.smooth_l1_loss(pred, y_t, reduction="none")).sum() / w_t.sum()
        loss.backward()
        optimizer.step()
    return predict_z_model(model, valid_data, target_mean, target_std, sequence=True)


def run_sequence(data, architecture, hidden, dropout, lr, epochs, seed) -> NNResult:
    start_time = time.perf_counter()
    train_seq, y_train, hist_mean, hist_std, obs_mean, obs_std = build_sequence_arrays(
        data["train_rows"],
        data["train1"],
        data["train1"],
        data["combos"],
    )
    valid_seq, y_valid, _, _, _, _ = build_sequence_arrays(
        data["valid_rows"],
        data["known_validation"],
        data["train2"],
        data["combos"],
        hist_mean,
        hist_std,
        obs_mean,
        obs_std,
    )
    internal_start = data["train_days"][-7]
    internal_mask = np.array([row.start.date() >= internal_start for row in data["train_rows"]])
    fit_mask = ~internal_mask
    model = SequenceNet(hidden, dropout, architecture, len(data["combos"]))
    _, best_epoch, internal_score, _, _ = train_z_model(
        model,
        subset_sequence(train_seq, fit_mask),
        y_train[fit_mask],
        subset_sequence(train_seq, internal_mask),
        y_train[internal_mask],
        lr,
        epochs,
        seed,
        sequence=True,
    )
    preds = fit_final_sequence(
        architecture,
        hidden,
        dropout,
        lr,
        best_epoch,
        seed,
        train_seq,
        y_train,
        valid_seq,
        len(data["combos"]),
    )
    return NNResult(
        method="direct_sequence",
        architecture=architecture,
        detail="obs6_hist14",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        best_epoch=best_epoch,
        internal_mape=float(internal_score),
        calibration_mape=float("nan"),
        validation_mape=float(mape(y_valid, preds)),
        pred_mean=float(preds.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def calibrator_features(rows, prediction_matrix, base_pred, known_agg, combo_block, combo_block_dow):
    strengths_5 = observation_strengths(rows, known_agg, combo_block, combo_block_dow, "combo_block_dow", 5.0)
    strengths_20 = observation_strengths(rows, known_agg, combo_block, combo_block_dow, "combo_block_dow", 20.0)
    combo_values = sorted({row.combo for row in rows})
    combo_index = {combo: idx for idx, combo in enumerate(combo_values)}
    hour_index = {8: 0, 9: 1, 17: 2, 18: 3}
    features = []
    for idx, row in enumerate(rows):
        combo_oh = np.zeros(len(combo_values), dtype=float)
        combo_oh[combo_index[row.combo]] = 1.0
        hour_oh = np.zeros(4, dtype=float)
        hour_oh[hour_index[row.start.hour]] = 1.0
        block_oh = np.array([1.0 if block_name(row.start) == "morning" else 0.0], dtype=float)
        preds = np.maximum(prediction_matrix[idx], 0.0)
        base = max(float(base_pred[idx]), 1.0)
        features.append(
            np.concatenate(
                [
                    np.log1p(preds),
                    np.log((preds + 1.0) / base),
                    [np.log1p(base), strengths_5[idx], strengths_20[idx]],
                    hour_oh,
                    block_oh,
                    combo_oh,
                ]
            )
        )
    return np.asarray(features, dtype=float)


def train_calibrator(x_train, y_train, base_train, x_valid, y_valid, base_valid, hidden, dropout, lr, epochs, seed, scale):
    torch.manual_seed(seed)
    mean, std = fit_standardizer(x_train)
    model = CalibratorNet(x_train.shape[1], hidden, dropout, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    x_t = to_tensor(standardize(x_train, mean, std))
    y_t = to_tensor(y_train)
    base_t = to_tensor(base_train)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = epochs
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t, base_t)
        loss = torch.mean(torch.abs(y_t - pred) / torch.clamp(torch.abs(y_t), min=1.0))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            pred_valid = predict_calibrator(model, x_valid, base_valid, mean, std)
            score = mape(y_valid, pred_valid)
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 10
            if stale >= 160:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, mean, std


def predict_calibrator(model, x, base, mean, std):
    model.eval()
    with torch.no_grad():
        pred = model(to_tensor(standardize(x, mean, std)), to_tensor(base)).cpu().numpy()
    return np.maximum(pred, 0.0)


def build_base_predictions(name, calibration_rows, calibration_actual, calibration_matrix, validation_rows, validation_matrix):
    if name == "hour4":
        weights, score, _ = optimize_scoped_blend_weights(
            calibration_actual,
            calibration_matrix[:, :4],
            calibration_rows,
            "hour",
        )
        base_cal = apply_scoped_blend(calibration_matrix[:, :4], calibration_rows, weights, "hour")
        base_val = apply_scoped_blend(validation_matrix[:, :4], validation_rows, weights, "hour")
        cal_weight_matrix = weight_matrix(calibration_rows, weights, "hour", 4)
        val_weight_matrix = weight_matrix(validation_rows, weights, "hour", 4)
        cal_candidates = calibration_matrix[:, :4]
        val_candidates = validation_matrix[:, :4]
    elif name == "block4":
        weights, score, _ = optimize_scoped_blend_weights(
            calibration_actual,
            calibration_matrix[:, :4],
            calibration_rows,
            "block",
        )
        base_cal = apply_scoped_blend(calibration_matrix[:, :4], calibration_rows, weights, "block")
        base_val = apply_scoped_blend(validation_matrix[:, :4], validation_rows, weights, "block")
        cal_weight_matrix = weight_matrix(calibration_rows, weights, "block", 4)
        val_weight_matrix = weight_matrix(validation_rows, weights, "block", 4)
        cal_candidates = calibration_matrix[:, :4]
        val_candidates = validation_matrix[:, :4]
    elif name == "traj_hour_cap010":
        weights, score = optimize_scoped_capped_blend(calibration_actual, calibration_matrix, calibration_rows, "hour", 0.10)
        base_cal = apply_scoped_weights(calibration_matrix, calibration_rows, weights, "hour")
        base_val = apply_scoped_weights(validation_matrix, validation_rows, weights, "hour")
        cal_weight_matrix = weight_matrix(calibration_rows, weights, "hour", calibration_matrix.shape[1])
        val_weight_matrix = weight_matrix(validation_rows, weights, "hour", validation_matrix.shape[1])
        cal_candidates = calibration_matrix
        val_candidates = validation_matrix
    elif name == "traj_block_cap020":
        weights, score = optimize_scoped_capped_blend(calibration_actual, calibration_matrix, calibration_rows, "block", 0.20)
        base_cal = apply_scoped_weights(calibration_matrix, calibration_rows, weights, "block")
        base_val = apply_scoped_weights(validation_matrix, validation_rows, weights, "block")
        cal_weight_matrix = weight_matrix(calibration_rows, weights, "block", calibration_matrix.shape[1])
        val_weight_matrix = weight_matrix(validation_rows, weights, "block", validation_matrix.shape[1])
        cal_candidates = calibration_matrix
        val_candidates = validation_matrix
    else:
        raise ValueError(f"unknown base: {name}")
    return score, base_cal, base_val, cal_weight_matrix, val_weight_matrix, cal_candidates, val_candidates


def scoped_key(row, scope: str):
    if scope == "hour":
        return (f"{row.start.hour:02d}",)
    if scope == "block":
        return (block_name(row.start),)
    raise ValueError(f"unknown scope: {scope}")


def weight_matrix(rows, weights_by_scope, scope: str, n_models: int):
    matrix = np.zeros((len(rows), n_models), dtype=float)
    for idx, row in enumerate(rows):
        matrix[idx] = weights_by_scope[scoped_key(row, scope)]
    return matrix


def run_calibrator(data_dir, base_name, hidden, dropout, lr, epochs, seed, scale) -> NNResult:
    start_time = time.perf_counter()
    data = load_trajectory_inputs(data_dir)
    (
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_actual,
        validation_matrix,
    ) = build_candidate_matrices(data, include_route_means=False)
    base_score, base_cal, base_val, _, _, _, _ = build_base_predictions(
        base_name,
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_matrix,
    )
    all_days = infer_dates(data["train1"])
    calibration_days = sorted({row.start.date() for row in calibration_rows})
    calibration_train_days = [day for day in all_days if day < min(calibration_days)]
    combos = sorted({row.combo for row in calibration_rows})
    combo_block, combo_block_dow = fit_expected_obs(data["train1"], calibration_train_days, combos)
    validation_combo_block, validation_combo_block_dow = fit_expected_obs(data["train1"], all_days, combos)
    x_cal = calibrator_features(
        calibration_rows,
        calibration_matrix,
        base_cal,
        data["train1"],
        combo_block,
        combo_block_dow,
    )
    x_val = calibrator_features(
        validation_rows,
        validation_matrix,
        base_val,
        data["test1_obs"],
        validation_combo_block,
        validation_combo_block_dow,
    )
    eval_start = max(row.start.date() for row in calibration_rows) - __import__("datetime").timedelta(days=1)
    eval_mask = np.array([row.start.date() >= eval_start for row in calibration_rows])
    fit_mask = ~eval_mask
    model, best_epoch, internal_score, mean, std = train_calibrator(
        x_cal[fit_mask],
        calibration_actual[fit_mask],
        base_cal[fit_mask],
        x_cal[eval_mask],
        calibration_actual[eval_mask],
        base_cal[eval_mask],
        hidden,
        dropout,
        lr,
        epochs,
        seed,
        scale,
    )
    # Refit on the full calibration fold using the selected epoch count.
    model = CalibratorNet(x_cal.shape[1], hidden, dropout, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    mean, std = fit_standardizer(x_cal)
    x_t = to_tensor(standardize(x_cal, mean, std))
    y_t = to_tensor(calibration_actual)
    base_t = to_tensor(base_cal)
    for _ in range(max(1, best_epoch)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t, base_t)
        loss = torch.mean(torch.abs(y_t - pred) / torch.clamp(torch.abs(y_t), min=1.0))
        loss.backward()
        optimizer.step()
    cal_pred = predict_calibrator(model, x_cal, base_cal, mean, std)
    val_pred = predict_calibrator(model, x_val, base_val, mean, std)
    return NNResult(
        method="calibrator",
        architecture="calibrator_mlp",
        detail=f"{base_name}_scale{scale}",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        best_epoch=best_epoch,
        internal_mape=float(internal_score),
        calibration_mape=float(mape(calibration_actual, cal_pred)),
        validation_mape=float(mape(validation_actual, val_pred)),
        pred_mean=float(val_pred.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def train_gate(
    x_train,
    y_train,
    candidates_train,
    prior_train,
    x_valid,
    y_valid,
    candidates_valid,
    prior_valid,
    hidden,
    dropout,
    lr,
    epochs,
    seed,
    scale,
):
    torch.manual_seed(seed)
    mean, std = fit_standardizer(x_train)
    model = BlendGateNet(x_train.shape[1], candidates_train.shape[1], hidden, dropout, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    x_t = to_tensor(standardize(x_train, mean, std))
    y_t = to_tensor(y_train)
    c_t = to_tensor(candidates_train)
    p_t = to_tensor(prior_train)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = epochs
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t, c_t, p_t)
        loss = torch.mean(torch.abs(y_t - pred) / torch.clamp(torch.abs(y_t), min=1.0))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            pred_valid = predict_gate(model, x_valid, candidates_valid, prior_valid, mean, std)
            score = mape(y_valid, pred_valid)
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 10
            if stale >= 160:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, mean, std


def predict_gate(model, x, candidates, prior, mean, std):
    model.eval()
    with torch.no_grad():
        pred = model(
            to_tensor(standardize(x, mean, std)),
            to_tensor(candidates),
            to_tensor(prior),
        ).cpu().numpy()
    return np.maximum(pred, 0.0)


def run_gate(data_dir, base_name, hidden, dropout, lr, epochs, seed, scale) -> NNResult:
    start_time = time.perf_counter()
    data = load_trajectory_inputs(data_dir)
    (
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_actual,
        validation_matrix,
    ) = build_candidate_matrices(data, include_route_means=False)
    base_score, base_cal, base_val, w_cal, w_val, c_cal, c_val = build_base_predictions(
        base_name,
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_matrix,
    )
    all_days = infer_dates(data["train1"])
    calibration_days = sorted({row.start.date() for row in calibration_rows})
    calibration_train_days = [day for day in all_days if day < min(calibration_days)]
    combos = sorted({row.combo for row in calibration_rows})
    combo_block, combo_block_dow = fit_expected_obs(data["train1"], calibration_train_days, combos)
    validation_combo_block, validation_combo_block_dow = fit_expected_obs(data["train1"], all_days, combos)
    x_cal = calibrator_features(
        calibration_rows,
        c_cal,
        base_cal,
        data["train1"],
        combo_block,
        combo_block_dow,
    )
    x_val = calibrator_features(
        validation_rows,
        c_val,
        base_val,
        data["test1_obs"],
        validation_combo_block,
        validation_combo_block_dow,
    )
    eval_start = max(row.start.date() for row in calibration_rows) - __import__("datetime").timedelta(days=1)
    eval_mask = np.array([row.start.date() >= eval_start for row in calibration_rows])
    fit_mask = ~eval_mask
    _, best_epoch, internal_score, _, _ = train_gate(
        x_cal[fit_mask],
        calibration_actual[fit_mask],
        c_cal[fit_mask],
        w_cal[fit_mask],
        x_cal[eval_mask],
        calibration_actual[eval_mask],
        c_cal[eval_mask],
        w_cal[eval_mask],
        hidden,
        dropout,
        lr,
        epochs,
        seed,
        scale,
    )
    mean, std = fit_standardizer(x_cal)
    torch.manual_seed(seed)
    model = BlendGateNet(x_cal.shape[1], c_cal.shape[1], hidden, dropout, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    x_t = to_tensor(standardize(x_cal, mean, std))
    y_t = to_tensor(calibration_actual)
    c_t = to_tensor(c_cal)
    p_t = to_tensor(w_cal)
    for _ in range(max(1, best_epoch)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t, c_t, p_t)
        loss = torch.mean(torch.abs(y_t - pred) / torch.clamp(torch.abs(y_t), min=1.0))
        loss.backward()
        optimizer.step()
    cal_pred = predict_gate(model, x_cal, c_cal, w_cal, mean, std)
    val_pred = predict_gate(model, x_val, c_val, w_val, mean, std)
    return NNResult(
        method="gate",
        architecture="prior_gate_mlp",
        detail=f"{base_name}_scale{scale}",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        best_epoch=best_epoch,
        internal_mape=float(internal_score),
        calibration_mape=float(mape(calibration_actual, cal_pred)),
        validation_mape=float(mape(validation_actual, val_pred)),
        pred_mean=float(val_pred.mean()),
        elapsed_sec=time.perf_counter() - start_time,
    )


def write_results(path: Path, rows: Sequence[NNResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "architecture",
        "detail",
        "hidden",
        "dropout",
        "lr",
        "best_epoch",
        "internal_mape",
        "calibration_mape",
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
                    "architecture": row.architecture,
                    "detail": row.detail,
                    "hidden": row.hidden,
                    "dropout": row.dropout,
                    "lr": row.lr,
                    "best_epoch": row.best_epoch,
                    "internal_mape": f"{row.internal_mape:.6f}",
                    "calibration_mape": "" if np.isnan(row.calibration_mape) else f"{row.calibration_mape:.6f}",
                    "validation_mape": f"{row.validation_mape:.6f}",
                    "pred_mean": f"{row.pred_mean:.3f}",
                    "elapsed_sec": f"{row.elapsed_sec:.2f}",
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Torch non-graph neural-network explorations for Task 2")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_torch_nn_sweep.csv"))
    parser.add_argument("--methods", nargs="+", default=["tabular", "sequence", "calibrator", "gate"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--calibrator-bases", nargs="+", default=["hour4", "traj_hour_cap010", "traj_block_cap020"])
    parser.add_argument("--correction-scales", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    args = parser.parse_args(argv)

    print(f"torch_version={torch.__version__}")
    results = []
    phase1_data = None
    if "tabular" in args.methods or "sequence" in args.methods:
        phase1_data = load_phase1_inputs(args.data_dir)
    if "tabular" in args.methods:
        for architecture in ("tab_mlp", "tab_resnet"):
            for hidden in args.hidden:
                for dropout in args.dropout:
                    result = run_tabular(phase1_data, architecture, hidden, dropout, args.lr, args.epochs, args.seed)
                    results.append(result)
                    print_result(result)
    if "sequence" in args.methods:
        for architecture in ("seq_gru", "seq_conv"):
            for hidden in args.hidden:
                for dropout in args.dropout:
                    result = run_sequence(phase1_data, architecture, hidden, dropout, args.lr, args.epochs, args.seed)
                    results.append(result)
                    print_result(result)
    if "calibrator" in args.methods:
        for base in args.calibrator_bases:
            for hidden in args.hidden:
                for dropout in args.dropout:
                    for scale in args.correction_scales:
                        result = run_calibrator(args.data_dir, base, hidden, dropout, args.lr, args.epochs, args.seed, scale)
                        results.append(result)
                        print_result(result)
    if "gate" in args.methods:
        for base in args.calibrator_bases:
            for hidden in args.hidden:
                for dropout in args.dropout:
                    for scale in args.correction_scales:
                        result = run_gate(args.data_dir, base, hidden, dropout, args.lr, args.epochs, args.seed, scale)
                        results.append(result)
                        print_result(result)
    results.sort(key=lambda row: row.validation_mape)
    write_results(args.output, results)
    print(f"best_validation_mape={results[0].validation_mape:.6f}")
    print(f"output={args.output}")


def print_result(result: NNResult) -> None:
    print(
        f"method={result.method} arch={result.architecture} detail={result.detail} "
        f"hidden={result.hidden} dropout={result.dropout} lr={result.lr} "
        f"best_epoch={result.best_epoch} internal_mape={result.internal_mape:.6f} "
        f"calibration_mape={result.calibration_mape:.6f} "
        f"validation_mape={result.validation_mape:.6f} pred_mean={result.pred_mean:.3f}"
    )
