from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features

from src3_explore.common.metrics import bucket_quantiles, interval_coverage, mape_value, summarize_interval_rows
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
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
    lower = preds[0.1]
    median = preds[0.5]
    upper = preds[0.9]
    width_buckets = bucket_quantiles(upper - lower, labels=("narrow", "mid", "wide"))
    for idx, row in enumerate(context.rows):
        item = {
            "date": str(row.start.date()),
            "combo": f"{row.tollgate_id}_{row.direction}",
            "hour": f"{row.start.hour:02d}",
            "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
            "actual": f"{actual[idx]:.6f}",
            "prediction": f"{float(median[idx]):.6f}",
            "lower": f"{float(lower[idx]):.6f}",
            "upper": f"{float(upper[idx]):.6f}",
            "covered": bool(lower[idx] <= actual[idx] <= upper[idx]),
            "interval_width": f"{float(upper[idx] - lower[idx]):.6f}",
            "interval_width_bucket": width_buckets[idx],
            "signed_error": f"{float(median[idx] - actual[idx]):.6f}",
            "abs_pct_error": f"{abs(float(median[idx] - actual[idx])) / max(abs(float(actual[idx])), 1.0):.6f}",
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
    grouped_csv = output_dir / "probabilistic" / "quantile_baselines_grouped_coverage.csv"
    failures_csv = output_dir / "probabilistic" / "quantile_baselines_phase1_failure_cases.csv"
    chart = output_dir / "probabilistic" / "quantile_baselines_coverage_by_width.svg"
    write_csv(train_csv, train_rows)
    write_csv(phase1_csv, phase1_rows)
    summary = [
        {"context": "train1_latest_fold", **train_metrics},
        {"context": "phase1_observation", **phase1_metrics},
    ]
    write_csv(summary_csv, summary)
    grouped = []
    for context_name, rows in (("train1_latest_fold", train_rows), ("phase1_observation", phase1_rows)):
        for fields in (["combo"], ["hour"], ["interval_width_bucket"], ["combo", "hour"]):
            for item in summarize_interval_rows(rows, fields):
                item["context"] = context_name
                item["dimension"] = "/".join(fields)
                item["value"] = "/".join(str(item.pop(field)) for field in fields)
                grouped.append(item)
    write_csv(grouped_csv, grouped)
    failures = sorted(
        [row for row in phase1_rows if row["covered"] is False],
        key=lambda row: float(row["abs_pct_error"]),
        reverse=True,
    )[:25]
    write_csv(failures_csv, failures)
    width_rows = [
        {"label": row["value"], "coverage": row["coverage"]}
        for row in grouped
        if row["context"] == "phase1_observation" and row["dimension"] == "interval_width_bucket"
    ]
    write_bar_svg(chart, width_rows, "label", "coverage", "Quantile phase1 coverage by interval width")
    card = ExperimentCard(
        name="quantile_baselines",
        hypothesis="分位数模型应在困难 regime 上给出更宽区间；如果区间宽度和 miss rate 相关，说明模型知道自己何时不确定。",
        data_visibility=(
            "分位数回归器只使用合法可见的训练标签拟合；held-out 和 phase1 标签只在预测完成后用于计算 pinball loss、MAPE 和 coverage。"
        ),
        prototype="使用官方 FeatureBuilder 特征训练 GradientBoosting p10/p50/p90 分位数回归，输出区间覆盖率、宽度桶和未覆盖失败样本。",
        metrics={"train1_coverage": train_metrics["interval_coverage_10_90"], "phase1_coverage": phase1_metrics["interval_coverage_10_90"]},
        result=(
            f"p10-p90 区间 coverage：train1={train_metrics['interval_coverage_10_90']}，"
            f"phase1={phase1_metrics['interval_coverage_10_90']}；区间有不确定性信号，但仍有校准不足。"
        ),
        insight="即使 median MAPE 不强，coverage、width bucket 和 failure cases 能揭示哪些 regime 的不确定性被低估。",
        next_step="保留。下一步与 model_disagreement 和 conformal interval 按 combo/hour/regime 对齐，检查是否能形成 train1-only 的风险标记。",
        artifacts=(str(train_csv), str(phase1_csv), str(summary_csv), str(grouped_csv), str(failures_csv), str(chart)),
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
