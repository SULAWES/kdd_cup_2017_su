from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

from kddcup2017_task2.data import OBS_TIMES, block_name, combine_date_time, target_volume

from src3_explore.common.metrics import mape_value, summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
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
                "signed_error": f"{p - y:.6f}",
                "abs_pct_error": f"{abs(p - y) / max(abs(y), 1.0):.6f}",
            }
        )
    return rows, mape_value(actual, pred)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    summary = []
    grouped_rows = []
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
        for context_name, rows in (("train1_latest_fold", train_rows), ("phase1_observation", phase1_rows)):
            for fields in (["combo"], ["block"], ["slot"], ["combo", "block"]):
                for item in summarize_errors(rows, fields):
                    item["basis"] = kind
                    item["context"] = context_name
                    item["dimension"] = "/".join(fields)
                    item["value"] = "/".join(str(item.pop(field)) for field in fields)
                    grouped_rows.append(item)
        summary.append(
            {
                "basis": kind,
                "components": 6,
                "train1_latest_fold_mape": f"{train_mape:.6f}",
                "phase1_observation_mape": f"{phase1_mape:.6f}",
            }
        )
    summary_csv = output_dir / "representations" / "curve_dictionary_summary.csv"
    grouped_csv = output_dir / "representations" / "curve_dictionary_grouped_errors.csv"
    chart = output_dir / "representations" / "curve_dictionary_phase1_mape.svg"
    write_csv(summary_csv, summary)
    write_csv(grouped_csv, grouped_rows)
    write_bar_svg(
        chart,
        [
            {"label": row["basis"], "mape": row["phase1_observation_mape"]}
            for row in sorted(summary, key=lambda item: float(item["phase1_observation_mape"]), reverse=True)
        ],
        "label",
        "mape",
        "Curve dictionary phase1 MAPE",
    )
    artifacts.extend([str(summary_csv), str(grouped_csv), str(chart)])
    best = min(summary, key=lambda row: float(row["train1_latest_fold_mape"]))
    card = ExperimentCard(
        name="curve_dictionary",
        hypothesis="day × combo 的 72-slot 日内曲线可能由少量基表示；如果绿色观察窗能拟合 pattern weight，就可能补全红窗形状。",
        data_visibility=(
            "曲线基只用训练标签拟合；验证日只暴露绿色 slot 来拟合系数，红窗标签只在补全后用于评分。"
        ),
        prototype="在 72-slot day curves 上拟合 PCA、NMF、DictionaryLearning，用 6 个 green slot 反推系数并重建红窗。",
        metrics={"best_train1_basis": best["basis"], "best_train1_mape": best["train1_latest_fold_mape"]},
        result=f"最佳 train1 fold 基为 {best['basis']}，MAPE={best['train1_latest_fold_mape']}；整体说明仅靠 6 个 green slot 做全天曲线补全过于欠定。",
        insight="即使分数弱，也能识别哪些日期/组合的日内形状不在训练字典里，并判断 NMF 这类非负形状是否比 PCA 更可解释。",
        next_step="归档为表示诊断。除非与 green shape cluster 或节假日 regime 明确重合，否则不扩展为主预测模型。",
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
