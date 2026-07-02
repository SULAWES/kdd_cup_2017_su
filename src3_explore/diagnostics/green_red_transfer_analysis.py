from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from kddcup2017_task2.data import OBS_TIMES, TargetRow, block_name, combine_date_time, target_volume

from src3_explore.common.metrics import bucket_quantiles, mape_value
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.visibility import VisibilityContext, load_phase1_context, load_train1_latest_fold_context


TARGET_TIMES_BY_BLOCK = {
    "morning": (("08:00", 8, 0), ("08:20", 8, 20), ("08:40", 8, 40), ("09:00", 9, 0), ("09:20", 9, 20), ("09:40", 9, 40)),
    "evening": (("17:00", 17, 0), ("17:20", 17, 20), ("17:40", 17, 40), ("18:00", 18, 0), ("18:20", 18, 20), ("18:40", 18, 40)),
}


@dataclass(frozen=True)
class TransferMatrix:
    weights: np.ndarray
    intercept: np.ndarray

    def predict(self, green: np.ndarray) -> np.ndarray:
        pred = np.asarray(green, dtype=float) @ self.weights + self.intercept
        return np.maximum(pred, 0.0)


def fit_nonnegative_transfer_matrix(green: np.ndarray, red: np.ndarray) -> TransferMatrix:
    x = np.asarray(green, dtype=float)
    y = np.asarray(red, dtype=float)
    x_aug = np.column_stack([x, np.ones(len(x), dtype=float)])
    weights = np.zeros((6, 6), dtype=float)
    intercept = np.zeros(6, dtype=float)
    for target_idx in range(6):
        coef, _ = nnls(x_aug, y[:, target_idx])
        weights[:, target_idx] = coef[:6]
        intercept[target_idx] = coef[-1]
    return TransferMatrix(weights=weights, intercept=intercept)


def block_vectors(context: VisibilityContext, source_agg, label_agg, days: Sequence, combos: Sequence[tuple[str, str]]):
    green_rows = []
    red_rows = []
    meta = []
    for day in days:
        for combo in combos:
            for block, clocks in OBS_TIMES.items():
                green = [float(source_agg.get((combine_date_time(day, clock), combo[0], combo[1]), 0)) for clock in clocks]
                red = [
                    float(label_agg.get((combine_date_time(day, __import__("datetime").time(hour, minute)), combo[0], combo[1]), 0))
                    for _, hour, minute in TARGET_TIMES_BY_BLOCK[block]
                ]
                green_rows.append(green)
                red_rows.append(red)
                meta.append({"date": str(day), "combo": f"{combo[0]}_{combo[1]}", "block": block})
    return np.asarray(green_rows, dtype=float), np.asarray(red_rows, dtype=float), meta


def evaluate_transfer(
    transfer: TransferMatrix,
    context: VisibilityContext,
    source_agg,
    label_agg,
    days: Sequence,
) -> tuple[list[dict[str, object]], float]:
    green, red, meta = block_vectors(context, source_agg, label_agg, days, context.combos)
    pred = transfer.predict(green)
    rows = []
    for block_idx, info in enumerate(meta):
        for slot_idx in range(6):
            rows.append(
                {
                    **info,
                    "red_slot": slot_idx,
                    "actual": f"{red[block_idx, slot_idx]:.6f}",
                    "prediction": f"{pred[block_idx, slot_idx]:.6f}",
                    "green_sum": f"{float(np.sum(green[block_idx])):.6f}",
                }
            )
    return rows, mape_value(red.ravel(), pred.ravel())


def green_shape_clusters(green: np.ndarray, n_clusters: int = 4) -> list[int]:
    from sklearn.cluster import KMeans

    x = np.asarray(green, dtype=float)
    denom = np.maximum(x.sum(axis=1, keepdims=True), 1.0)
    shape = x / denom
    k = max(1, min(n_clusters, len(shape)))
    if k == 1:
        return [0] * len(shape)
    model = KMeans(n_clusters=k, random_state=13, n_init=10)
    return [int(value) for value in model.fit_predict(shape)]


def ratio_surface_rows(green: np.ndarray, red: np.ndarray, meta: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    green_sum = np.asarray(green, dtype=float).sum(axis=1)
    buckets = bucket_quantiles(green_sum, labels=("weak", "normal", "strong"))
    rows = []
    for idx, bucket in enumerate(buckets):
        for slot_idx in range(6):
            rows.append(
                {
                    "green_strength_bucket": bucket,
                    "block": meta[idx]["block"],
                    "red_slot": slot_idx,
                    "green_sum": f"{green_sum[idx]:.6f}",
                    "red": f"{float(red[idx, slot_idx]):.6f}",
                    "red_to_green_ratio": f"{float(red[idx, slot_idx]) / max(float(green_sum[idx]), 1.0):.6f}",
                }
            )
    return rows


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    green_fit, red_fit, fit_meta = block_vectors(
        train_context,
        train_context.train_agg,
        train_context.train_agg,
        train_context.train_days,
        train_context.combos,
    )
    transfer = fit_nonnegative_transfer_matrix(green_fit, red_fit)
    train_rows, train_mape = evaluate_transfer(
        transfer,
        train_context,
        train_context.known_agg,
        train_context.label_agg,
        train_context.eval_days,
    )
    phase1_rows, phase1_mape = evaluate_transfer(
        fit_nonnegative_transfer_matrix(
            *block_vectors(phase1_context, phase1_context.train_agg, phase1_context.train_agg, phase1_context.train_days, phase1_context.combos)[:2]
        ),
        phase1_context,
        phase1_context.known_agg,
        phase1_context.label_agg,
        phase1_context.eval_days,
    )
    clusters = green_shape_clusters(green_fit)
    cluster_rows = [{**meta, "cluster": cluster} for meta, cluster in zip(fit_meta, clusters)]
    ratio_rows = ratio_surface_rows(green_fit, red_fit, fit_meta)
    transfer_rows = [
        {"green_slot": i, "red_slot": j, "weight": f"{float(transfer.weights[i, j]):.8f}"}
        for i in range(6)
        for j in range(6)
    ]
    for j in range(6):
        transfer_rows.append({"green_slot": "intercept", "red_slot": j, "weight": f"{float(transfer.intercept[j]):.8f}"})

    out_dir = output_dir / "diagnostics"
    transfer_csv = out_dir / "green_red_transfer_matrix.csv"
    train_csv = out_dir / "green_red_transfer_train1_fold.csv"
    phase1_csv = out_dir / "green_red_transfer_phase1_observation.csv"
    cluster_csv = out_dir / "green_shape_clusters.csv"
    ratio_csv = out_dir / "green_red_ratio_surface.csv"
    write_csv(transfer_csv, transfer_rows)
    write_csv(train_csv, train_rows)
    write_csv(phase1_csv, phase1_rows)
    write_csv(cluster_csv, cluster_rows)
    write_csv(ratio_csv, ratio_rows)
    card = ExperimentCard(
        name="green_red_transfer_analysis",
        hypothesis="The six green slots carry a constrained shape signal for the six red slots.",
        data_visibility=(
            "Transfer defaults are selected on train1 only. Phase1 labels are used once to observe the fixed "
            "transfer prototype, not to choose matrix rank, clusters, or smoothing."
        ),
        prototype="Fit nonnegative 6x6 green-to-red transfer, cluster green shapes, and tabulate red/green ratios.",
        metrics={"train1_latest_fold_mape": f"{train_mape:.6f}", "phase1_observation_mape": f"{phase1_mape:.6f}"},
        result=f"Wrote transfer matrix and ratio surfaces under {out_dir}.",
        insight="If this simple constrained matrix fails only in specific clusters, those clusters are regime candidates.",
        next_step="Compare transfer residuals with residual_atlas high-error groups before adding nonlinear correction.",
        artifacts=(str(transfer_csv), str(train_csv), str(phase1_csv), str(cluster_csv), str(ratio_csv)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze green-window to red-window transfer")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

