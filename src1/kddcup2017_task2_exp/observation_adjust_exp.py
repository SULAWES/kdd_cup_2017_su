from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, TargetRow, block_name, combine_date_time
from kddcup2017_task2.ensemble import apply_scoped_blend, optimize_scoped_blend_weights
from kddcup2017_task2.model import mape
from kddcup2017_task2_exp.trajectory_ensemble_exp import (
    apply_scoped_weights,
    build_candidate_matrices,
    load_inputs,
    optimize_scoped_capped_blend,
)


BASE_CONFIGS = (
    ("hour4", "hour", None),
    ("block4", "block", None),
    ("traj_block_cap010", "block", 0.10),
    ("traj_block_cap015", "block", 0.15),
    ("traj_block_cap020", "block", 0.20),
    ("traj_hour_cap010", "hour", 0.10),
)


def scope_key(row: TargetRow, scope: str) -> tuple[str, ...]:
    if scope == "global":
        return ("global",)
    if scope == "block":
        return (block_name(row.start),)
    if scope == "hour":
        return (f"{row.start.hour:02d}",)
    if scope == "combo_block":
        return row.combo + (block_name(row.start),)
    raise ValueError(f"unknown scope: {scope}")


def obs_sum(aggregate: Mapping, row: TargetRow) -> float:
    block = block_name(row.start)
    return float(
        sum(
            aggregate.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0)
            for clock in OBS_TIMES[block]
        )
    )


def fit_expected_obs(train_agg: Mapping, train_days: Sequence, combos: Sequence[tuple[str, str]]):
    by_combo_block = defaultdict(list)
    by_combo_block_dow = defaultdict(list)
    for combo in combos:
        for day in train_days:
            for block, clocks in OBS_TIMES.items():
                value = float(
                    sum(
                        train_agg.get((combine_date_time(day, clock), combo[0], combo[1]), 0)
                        for clock in clocks
                    )
                )
                by_combo_block[(combo, block)].append(value)
                by_combo_block_dow[(combo, block, day.weekday())].append(value)

    def median(values):
        if not values:
            return 0.0
        return float(np.median(np.asarray(values, dtype=float)))

    combo_block = {key: median(values) for key, values in by_combo_block.items()}
    combo_block_dow = {key: median(values) for key, values in by_combo_block_dow.items()}
    return combo_block, combo_block_dow


def observation_strengths(rows, known_agg, combo_block, combo_block_dow, expected_mode: str, smoothing: float):
    strengths = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        block = block_name(row.start)
        fallback = combo_block.get((row.combo, block), 0.0)
        if expected_mode == "combo_block_dow":
            expected = combo_block_dow.get((row.combo, block, row.start.weekday()), fallback)
        elif expected_mode == "combo_block":
            expected = fallback
        else:
            raise ValueError(f"unknown expected mode: {expected_mode}")
        current = obs_sum(known_agg, row)
        strengths[idx] = np.log((current + smoothing) / (expected + smoothing))
    return np.clip(strengths, -1.0, 1.0)


def fit_apply_adjustment(
    calibration_rows,
    calibration_actual,
    calibration_base,
    calibration_strength,
    validation_rows,
    validation_base,
    validation_strength,
    adjustment_scope: str,
    beta_max: float = 0.8,
):
    beta_steps = max(3, int(round((2.0 * beta_max) / 0.02)) + 1)
    beta_grid = np.linspace(-beta_max, beta_max, beta_steps)
    calibration_predictions = np.asarray(calibration_base, dtype=float).copy()
    validation_predictions = np.asarray(validation_base, dtype=float).copy()
    betas = {}
    cal_keys = [scope_key(row, adjustment_scope) for row in calibration_rows]
    val_keys = [scope_key(row, adjustment_scope) for row in validation_rows]
    for key in sorted(set(cal_keys) | set(val_keys)):
        cal_idx = [idx for idx, item in enumerate(cal_keys) if item == key]
        val_idx = [idx for idx, item in enumerate(val_keys) if item == key]
        if not cal_idx:
            beta = 0.0
        else:
            best_score = float("inf")
            beta = 0.0
            for candidate_beta in beta_grid:
                factor = np.exp(candidate_beta * calibration_strength[cal_idx])
                preds = calibration_base[cal_idx] * factor
                score = mape(calibration_actual[cal_idx], preds)
                if score < best_score:
                    best_score = score
                    beta = float(candidate_beta)
        betas[key] = beta
        if cal_idx:
            calibration_predictions[cal_idx] = calibration_base[cal_idx] * np.exp(
                beta * calibration_strength[cal_idx]
            )
        if val_idx:
            validation_predictions[val_idx] = validation_base[val_idx] * np.exp(
                beta * validation_strength[val_idx]
            )
    return calibration_predictions, validation_predictions, betas


def build_base_predictions(base_name, base_scope, trajectory_cap, calibration_rows, calibration_actual, calibration_matrix, validation_rows, validation_matrix):
    if trajectory_cap is None:
        weights_by_scope, calibration_mape, _ = optimize_scoped_blend_weights(
            calibration_actual,
            calibration_matrix[:, :4],
            calibration_rows,
            base_scope,
        )
        validation_predictions = apply_scoped_blend(
            validation_matrix[:, :4],
            validation_rows,
            weights_by_scope,
            base_scope,
        )
        calibration_predictions = apply_scoped_blend(
            calibration_matrix[:, :4],
            calibration_rows,
            weights_by_scope,
            base_scope,
        )
    else:
        weights_by_scope, calibration_mape = optimize_scoped_capped_blend(
            calibration_actual,
            calibration_matrix,
            calibration_rows,
            base_scope,
            trajectory_cap,
        )
        validation_predictions = apply_scoped_weights(
            validation_matrix,
            validation_rows,
            weights_by_scope,
            base_scope,
        )
        calibration_predictions = apply_scoped_weights(
            calibration_matrix,
            calibration_rows,
            weights_by_scope,
            base_scope,
        )
    return {
        "base_name": base_name,
        "base_scope": base_scope,
        "trajectory_cap": trajectory_cap,
        "calibration_mape": calibration_mape,
        "calibration_predictions": calibration_predictions,
        "validation_predictions": validation_predictions,
    }


def run_experiments(data_dir: Path, beta_max: float):
    data = load_inputs(data_dir)
    (
        calibration_rows,
        calibration_actual,
        calibration_matrix,
        validation_rows,
        validation_actual,
        validation_matrix,
    ) = build_candidate_matrices(data, include_route_means=False)
    train_days = sorted({row.start.date() for row in calibration_rows})
    all_train_days = sorted({start.date() for start, _, _ in data["train1"]})
    calibration_train_days = [day for day in all_train_days if day < min(train_days)]
    combos = sorted({row.combo for row in calibration_rows})
    combo_block, combo_block_dow = fit_expected_obs(data["train1"], calibration_train_days, combos)
    validation_combo_block, validation_combo_block_dow = fit_expected_obs(data["train1"], all_train_days, combos)
    calibration_known = data["train1"]
    validation_known = data["test1_obs"]

    rows = []
    base_predictions = [
        build_base_predictions(*config, calibration_rows, calibration_actual, calibration_matrix, validation_rows, validation_matrix)
        for config in BASE_CONFIGS
    ]
    for base in base_predictions:
        rows.append(
            {
                "base_name": base["base_name"],
                "expected_mode": "none",
                "smoothing": "",
                "adjustment_scope": "none",
                "calibration_mape": base["calibration_mape"],
                "validation_mape": mape(validation_actual, base["validation_predictions"]),
                "pred_mean": float(base["validation_predictions"].mean()),
                "betas": "",
            }
        )
        for expected_mode in ("combo_block", "combo_block_dow"):
            for smoothing in (5.0, 20.0, 50.0):
                calibration_strength = observation_strengths(
                    calibration_rows,
                    calibration_known,
                    combo_block,
                    combo_block_dow,
                    expected_mode,
                    smoothing,
                )
                validation_strength = observation_strengths(
                    validation_rows,
                    validation_known,
                    validation_combo_block,
                    validation_combo_block_dow,
                    expected_mode,
                    smoothing,
                )
                for adjustment_scope in ("global", "block", "hour"):
                    cal_preds, val_preds, betas = fit_apply_adjustment(
                        calibration_rows,
                        calibration_actual,
                        base["calibration_predictions"],
                        calibration_strength,
                        validation_rows,
                        base["validation_predictions"],
                        validation_strength,
                        adjustment_scope,
                        beta_max,
                    )
                    beta_text = " | ".join(
                        f"{'/'.join(key)}:{value:.3f}" for key, value in sorted(betas.items())
                    )
                    rows.append(
                        {
                            "base_name": base["base_name"],
                            "expected_mode": expected_mode,
                            "smoothing": smoothing,
                            "adjustment_scope": adjustment_scope,
                            "calibration_mape": mape(calibration_actual, cal_preds),
                            "validation_mape": mape(validation_actual, val_preds),
                            "pred_mean": float(val_preds.mean()),
                            "betas": beta_text,
                        }
                    )
    return sorted(rows, key=lambda row: float(row["validation_mape"]))


def write_results(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_name",
        "expected_mode",
        "smoothing",
        "adjustment_scope",
        "calibration_mape",
        "validation_mape",
        "pred_mean",
        "betas",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in ("calibration_mape", "validation_mape"):
                output[key] = f"{float(output[key]):.6f}"
            output["pred_mean"] = f"{float(output['pred_mean']):.3f}"
            writer.writerow(output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Observation-window posterior adjustment experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/src1_observation_adjust.csv"))
    parser.add_argument("--beta-max", type=float, default=0.8)
    args = parser.parse_args(argv)
    rows = run_experiments(args.data_dir, args.beta_max)
    write_results(args.output, rows)
    for row in rows[:12]:
        print(
            f"base={row['base_name']} expected={row['expected_mode']} "
            f"smoothing={row['smoothing']} adjustment={row['adjustment_scope']} "
            f"calibration_mape={float(row['calibration_mape']):.6f} "
            f"validation_mape={float(row['validation_mape']):.6f} pred_mean={float(row['pred_mean']):.3f}"
        )
    print(f"best_validation_mape={float(rows[0]['validation_mape']):.6f}")
    print(f"output={args.output}")
