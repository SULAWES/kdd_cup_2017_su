from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, TargetRow, block_name, combine_date_time, make_target_rows, target_volume

from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.visibility import load_phase1_context, load_train1_latest_fold_context


ATTR_GROUPS = {
    "etc": ("0", "1"),
    "model": ("0", "1", "2", "3", "4", "5", "6", "7"),
    "veh_type": ("blank", "0", "1"),
}


def attr_count(attr_agg: Mapping, row: TargetRow, attr_name: str, attr_value: str, red: bool) -> float:
    if red:
        return float(attr_agg.get((row.start, row.tollgate_id, row.direction, attr_name, attr_value), 0))
    total = 0.0
    for clock in OBS_TIMES[block_name(row.start)]:
        obs_start = combine_date_time(row.start.date(), clock)
        total += float(attr_agg.get((obs_start, row.tollgate_id, row.direction, attr_name, attr_value), 0))
    return total


def fit_component_ratios(rows: Sequence[TargetRow], attr_agg: Mapping, label_agg: Mapping, attr_name: str) -> dict[tuple, float]:
    values = defaultdict(list)
    for row in rows:
        for attr_value in ATTR_GROUPS[attr_name]:
            green = attr_count(attr_agg, row, attr_name, attr_value, red=False)
            red = attr_count(attr_agg, row, attr_name, attr_value, red=True)
            if green > 0:
                values[(row.combo, block_name(row.start), attr_value)].append(red / green)
    return {key: float(np.median(items)) for key, items in values.items() if items}


def predict_components(rows, known_attr, ratios, attr_name: str) -> np.ndarray:
    pred = []
    for row in rows:
        total = 0.0
        for attr_value in ATTR_GROUPS[attr_name]:
            green = attr_count(known_attr, row, attr_name, attr_value, red=False)
            ratio = ratios.get((row.combo, block_name(row.start), attr_value), 0.0)
            total += green * ratio
        pred.append(total)
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def evaluate_attr(context, attr_name: str):
    train_rows = make_target_rows(context.train_days, context.combos)
    ratios = fit_component_ratios(train_rows, context.train_attr_agg, context.train_agg, attr_name)
    pred = predict_components(context.rows, context.known_attr_agg, ratios, attr_name)
    actual = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    rows = []
    for row, y, p in zip(context.rows, actual, pred):
        rows.append(
            {
                "date": str(row.start.date()),
                "combo": f"{row.tollgate_id}_{row.direction}",
                "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                "attr_name": attr_name,
                "actual": f"{y:.6f}",
                "component_prediction": f"{float(p):.6f}",
                "signed_error": f"{float(p - y):.6f}",
            }
        )
    return rows, mape_value(actual, pred)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    summary = []
    artifacts = []
    for attr_name in ATTR_GROUPS:
        train_rows, train_mape = evaluate_attr(train_context, attr_name)
        phase1_rows, phase1_mape = evaluate_attr(phase1_context, attr_name)
        train_csv = output_dir / "mechanisms" / f"etc_component_{attr_name}_train1_fold.csv"
        phase1_csv = output_dir / "mechanisms" / f"etc_component_{attr_name}_phase1.csv"
        write_csv(train_csv, train_rows)
        write_csv(phase1_csv, phase1_rows)
        artifacts.extend([str(train_csv), str(phase1_csv)])
        summary.append(
            {
                "component": attr_name,
                "train1_latest_fold_mape": f"{train_mape:.6f}",
                "phase1_observation_mape": f"{phase1_mape:.6f}",
            }
        )
    summary_csv = output_dir / "mechanisms" / "etc_component_model_summary.csv"
    write_csv(summary_csv, summary)
    artifacts.append(str(summary_csv))
    best = min(summary, key=lambda row: float(row["train1_latest_fold_mape"]))
    card = ExperimentCard(
        name="etc_component_model",
        hypothesis="Vehicle components can act as generated sub-flows, not only flat features.",
        data_visibility=(
            "Ratios are fitted on train windows only. Held-out and phase1 red labels are used only to score fixed "
            "component-sum predictions."
        ),
        prototype="Predict red total by summing attribute-level green-to-red component ratios for ETC, vehicle model, and type.",
        metrics={"best_train_component": best["component"], "best_train_mape": best["train1_latest_fold_mape"]},
        result=f"Wrote component reconciliation summary to {summary_csv}.",
        insight="Component failures show whether structure shares are stable enough for a reconciled generator.",
        next_step="Use only components that improve rolling folds, then reconcile them with total-flow predictions.",
        artifacts=tuple(artifacts),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ETC and vehicle component mechanism model")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()


