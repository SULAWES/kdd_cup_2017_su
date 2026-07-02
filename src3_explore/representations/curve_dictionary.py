from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

from kddcup2017_task2.data import OBS_TIMES, block_name, combine_date_time, target_volume

from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.visibility import load_phase1_context, load_train1_latest_fold_context


def slot_index(clock: time) -> int:
    return clock.hour * 3 + clock.minute // 20


BLOCK_RED_CLOCKS = {
    "morning": (time(8, 0), time(8, 20), time(8, 40), time(9, 0), time(9, 20), time(9, 40)),
    "evening": (time(17, 0), time(17, 20), time(17, 40), time(18, 0), time(18, 20), time(18, 40)),
}


@dataclass(frozen=True)
class CurveBasis:
    kind: str
    mean: np.ndarray
    components: np.ndarray

    def reconstruct_from_green(self, green_values: np.ndarray, green_indices: Sequence[int]) -> np.ndarray:
        green = np.asarray(green_values, dtype=float)
        basis_green = self.components[:, list(green_indices)].T
        if self.kind == "pca":
            coef, *_ = np.linalg.lstsq(basis_green, green - self.mean[list(green_indices)], rcond=None)
            recon = self.mean + coef @ self.components
        elif self.kind == "nmf":
            coef, _ = nnls(np.maximum(basis_green, 0.0), np.maximum(green, 0.0))
            recon = coef @ np.maximum(self.components, 0.0)
        else:
            coef, *_ = np.linalg.lstsq(basis_green, green - self.mean[list(green_indices)], rcond=None)
            recon = self.mean + coef @ self.components
        return np.maximum(recon, 0.0)


def day_combo_curves(aggregate: Mapping, days: Sequence, combos: Sequence[tuple[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    curves = []
    meta = []
    clocks = [time(hour, minute) for hour in range(24) for minute in (0, 20, 40)]
    for day in days:
        for combo in combos:
            curves.append([float(aggregate.get((combine_date_time(day, clock), combo[0], combo[1]), 0)) for clock in clocks])
            meta.append({"date": str(day), "combo": f"{combo[0]}_{combo[1]}"})
    return np.asarray(curves, dtype=float), meta


def fit_curve_basis(matrix: np.ndarray, kind: str = "pca", n_components: int = 6) -> CurveBasis:
    n = max(1, min(n_components, min(matrix.shape)))
    if kind == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=n, random_state=13)
        model.fit(matrix)
        return CurveBasis(kind=kind, mean=model.mean_, components=model.components_)
    if kind == "nmf":
        from sklearn.decomposition import NMF

        model = NMF(n_components=n, random_state=13, init="nndsvda", max_iter=800)
        model.fit(np.maximum(matrix, 0.0))
        return CurveBasis(kind=kind, mean=np.zeros(matrix.shape[1], dtype=float), components=model.components_)
    if kind == "dictionary":
        from sklearn.decomposition import DictionaryLearning

        model = DictionaryLearning(n_components=n, random_state=13, transform_algorithm="lasso_lars", max_iter=400)
        centered = matrix - matrix.mean(axis=0)
        model.fit(centered)
        return CurveBasis(kind=kind, mean=matrix.mean(axis=0), components=model.components_)
    raise ValueError(f"unknown dictionary kind: {kind}")


def predict_rows_from_basis(context, basis: CurveBasis) -> tuple[list[dict[str, object]], float]:
    rows = []
    actual = []
    pred = []
    for row in context.rows:
        block = block_name(row.start)
        green_indices = [slot_index(clock) for clock in OBS_TIMES[block]]
        red_indices = [slot_index(clock) for clock in BLOCK_RED_CLOCKS[block]]
        green = np.asarray(
            [
                float(context.known_agg.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0))
                for clock in OBS_TIMES[block]
            ],
            dtype=float,
        )
        recon = basis.reconstruct_from_green(green, green_indices)
        slot_pos = red_indices.index(slot_index(row.start))
        p = float(recon[red_indices[slot_pos]])
        y = float(target_volume(context.label_agg, row))
        actual.append(y)
        pred.append(p)
        rows.append(
            {
                "date": str(row.start.date()),
                "combo": f"{row.tollgate_id}_{row.direction}",
                "block": block,
                "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                "actual": f"{y:.6f}",
                "prediction": f"{p:.6f}",
            }
        )
    return rows, mape_value(actual, pred)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    summary = []
    artifacts = []
    for kind in ("pca", "nmf", "dictionary"):
        train_curves, _ = day_combo_curves(train_context.train_agg, train_context.train_days, train_context.combos)
        basis = fit_curve_basis(train_curves, kind=kind, n_components=6)
        train_rows, train_mape = predict_rows_from_basis(train_context, basis)
        phase1_curves, _ = day_combo_curves(phase1_context.train_agg, phase1_context.train_days, phase1_context.combos)
        phase1_basis = fit_curve_basis(phase1_curves, kind=kind, n_components=6)
        phase1_rows, phase1_mape = predict_rows_from_basis(phase1_context, phase1_basis)
        train_csv = output_dir / "representations" / f"curve_dictionary_{kind}_train1_fold.csv"
        phase1_csv = output_dir / "representations" / f"curve_dictionary_{kind}_phase1.csv"
        write_csv(train_csv, train_rows)
        write_csv(phase1_csv, phase1_rows)
        artifacts.extend([str(train_csv), str(phase1_csv)])
        summary.append(
            {
                "basis": kind,
                "components": 6,
                "train1_latest_fold_mape": f"{train_mape:.6f}",
                "phase1_observation_mape": f"{phase1_mape:.6f}",
            }
        )
    summary_csv = output_dir / "representations" / "curve_dictionary_summary.csv"
    write_csv(summary_csv, summary)
    artifacts.append(str(summary_csv))
    best = min(summary, key=lambda row: float(row["train1_latest_fold_mape"]))
    card = ExperimentCard(
        name="curve_dictionary",
        hypothesis="A small dictionary of day-combo curves may recover red-window shape from green-window shape.",
        data_visibility=(
            "Curve bases are fitted on train labels only. Validation days expose only green slots for coefficient "
            "fitting; red labels are used only for scoring the reconstructed slots."
        ),
        prototype="Fit PCA/NMF/dictionary bases over 72-slot day curves and reconstruct target red slots from 6 green slots.",
        metrics={"best_train1_basis": best["basis"], "best_train1_mape": best["train1_latest_fold_mape"]},
        result=f"Wrote dictionary summaries to {summary_csv}.",
        insight="This is a pattern-completion baseline; failure cases identify days whose shape is not in the train dictionary.",
        next_step="Join dictionary residuals with green shape clusters and holiday/post-holiday regimes.",
        artifacts=tuple(artifacts),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Curve dictionary representation experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

