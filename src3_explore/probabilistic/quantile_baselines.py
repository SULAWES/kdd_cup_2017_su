from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features

from src3_explore.common.metrics import interval_coverage, mape_value
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.visibility import VisibilityContext, load_phase1_context, load_train1_latest_fold_context


def build_xy(context: VisibilityContext):
    train_rows = make_target_rows(context.train_days, context.combos)
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = filter_features(builder.transform(train_rows, context.train_agg, context.train_attr_agg), DEFAULT_DROP_FEATURES)
    pred_features = filter_features(builder.transform(context.rows, context.known_agg, context.known_attr_agg), DEFAULT_DROP_FEATURES)
    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_pred = vectorizer.transform(pred_features)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_actual = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    return x_train, y_train, x_pred, y_actual


def fit_quantile_predictions(context: VisibilityContext, quantiles: Sequence[float] = (0.1, 0.5, 0.9)):
    from sklearn.ensemble import GradientBoostingRegressor

    x_train, y_train, x_pred, y_actual = build_xy(context)
    preds = {}
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=float(q),
            n_estimators=160,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=8,
            random_state=13,
        )
        model.fit(x_train, np.log1p(y_train))
        preds[q] = np.maximum(np.expm1(model.predict(x_pred)), 0.0)
    return preds, y_actual


def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = actual - pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def rows_for_context(context: VisibilityContext, preds: dict[float, np.ndarray], actual: np.ndarray):
    rows = []
    for idx, row in enumerate(context.rows):
        item = {
            "date": str(row.start.date()),
            "combo": f"{row.tollgate_id}_{row.direction}",
            "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
            "actual": f"{actual[idx]:.6f}",
        }
        for q, values in preds.items():
            item[f"q{int(q * 100):02d}"] = f"{float(values[idx]):.6f}"
        rows.append(item)
    return rows


def evaluate_context(context: VisibilityContext):
    preds, actual = fit_quantile_predictions(context)
    lower = preds[0.1]
    median = preds[0.5]
    upper = preds[0.9]
    cov = interval_coverage(actual, lower, upper)
    metrics = {
        "median_mape": f"{mape_value(actual, median):.6f}",
        "p10_pinball": f"{pinball_loss(actual, lower, 0.1):.6f}",
        "p50_pinball": f"{pinball_loss(actual, median, 0.5):.6f}",
        "p90_pinball": f"{pinball_loss(actual, upper, 0.9):.6f}",
        "interval_coverage_10_90": f"{cov['coverage']:.6f}",
        "interval_mean_width": f"{cov['mean_width']:.6f}",
    }
    return rows_for_context(context, preds, actual), metrics


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    train_context = load_train1_latest_fold_context(data_dir)
    phase1_context = load_phase1_context(data_dir)
    train_rows, train_metrics = evaluate_context(train_context)
    phase1_rows, phase1_metrics = evaluate_context(phase1_context)
    train_csv = output_dir / "probabilistic" / "quantile_baselines_train1_fold.csv"
    phase1_csv = output_dir / "probabilistic" / "quantile_baselines_phase1.csv"
    summary_csv = output_dir / "probabilistic" / "quantile_baselines_summary.csv"
    write_csv(train_csv, train_rows)
    write_csv(phase1_csv, phase1_rows)
    summary = [
        {"context": "train1_latest_fold", **train_metrics},
        {"context": "phase1_observation", **phase1_metrics},
    ]
    write_csv(summary_csv, summary)
    card = ExperimentCard(
        name="quantile_baselines",
        hypothesis="Quantile models should widen intervals for difficult regimes and improve failure awareness.",
        data_visibility=(
            "Quantile regressors fit only on visible train labels. Held-out and phase1 labels score fixed quantiles "
            "and coverage after prediction."
        ),
        prototype="GradientBoosting quantile regressors at p10/p50/p90 using the official feature builder.",
        metrics={"train1_coverage": train_metrics["interval_coverage_10_90"], "phase1_coverage": phase1_metrics["interval_coverage_10_90"]},
        result=f"Wrote quantile predictions and summary to {summary_csv}.",
        insight="Coverage and width reveal whether uncertainty tracks actual model errors.",
        next_step="Compare quantile width with model_disagreement and conformal intervals by regime.",
        artifacts=(str(train_csv), str(phase1_csv), str(summary_csv)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quantile prediction baselines")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

