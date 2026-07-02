from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.candidate_cache import build_prediction_payload
from src3_explore.common.metrics import bucket_quantiles, interval_coverage, summarize_interval_rows
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg


def conformal_radius(abs_residuals: np.ndarray, alpha: float = 0.1) -> float:
    residuals = np.asarray(abs_residuals, dtype=float)
    if len(residuals) == 0:
        return 0.0
    q = np.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals)
    q = min(max(float(q), 0.0), 1.0)
    return float(np.quantile(residuals, q, method="higher"))


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    payload = build_prediction_payload(data_dir)
    context = payload["context"]
    cal_actual = payload["calibration_actual"]
    cal_pred = payload["calibration_prediction"]
    val_actual = payload["validation_actual"]
    val_pred = payload["validation_prediction"]
    matrix = payload["validation_matrix"]
    radius = conformal_radius(np.abs(cal_actual - cal_pred), alpha=0.1)
    lower = np.maximum(val_pred - radius, 0.0)
    upper = val_pred + radius
    uncertainty = np.std(matrix, axis=1) / np.maximum(np.mean(matrix, axis=1), 1.0)
    buckets = bucket_quantiles(uncertainty, labels=("low", "mid", "high"))
    rows = []
    for idx, row in enumerate(context.rows):
        rows.append(
            {
                "date": str(row.start.date()),
                "combo": f"{row.tollgate_id}_{row.direction}",
                "hour": f"{row.start.hour:02d}",
                "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                "actual": f"{val_actual[idx]:.6f}",
                "prediction": f"{val_pred[idx]:.6f}",
                "lower": f"{lower[idx]:.6f}",
                "upper": f"{upper[idx]:.6f}",
                "interval_width": f"{float(upper[idx] - lower[idx]):.6f}",
                "covered": bool(lower[idx] <= val_actual[idx] <= upper[idx]),
                "signed_error": f"{float(val_pred[idx] - val_actual[idx]):.6f}",
                "abs_pct_error": f"{abs(float(val_pred[idx] - val_actual[idx])) / max(abs(float(val_actual[idx])), 1.0):.6f}",
                "uncertainty": f"{uncertainty[idx]:.6f}",
                "uncertainty_bucket": buckets[idx],
            }
        )
    summary = []
    for fields in (["uncertainty_bucket"], ["combo"], ["hour"], ["combo", "hour"]):
        for item in summarize_interval_rows(rows, fields):
            item["dimension"] = "/".join(fields)
            item["value"] = "/".join(str(item.pop(field)) for field in fields)
            summary.append(item)
    coverage = interval_coverage(val_actual, lower, upper)
    csv_path = output_dir / "probabilistic" / "conformal_intervals_phase1.csv"
    summary_csv = output_dir / "probabilistic" / "conformal_intervals_by_uncertainty.csv"
    failures_csv = output_dir / "probabilistic" / "conformal_intervals_failure_cases.csv"
    chart = output_dir / "probabilistic" / "conformal_intervals_coverage_by_uncertainty.svg"
    write_csv(csv_path, rows)
    write_csv(summary_csv, summary)
    failures = sorted([row for row in rows if row["covered"] is False], key=lambda row: float(row["abs_pct_error"]), reverse=True)[:25]
    write_csv(failures_csv, failures)
    chart_rows = [
        {"label": row["value"], "coverage": row["coverage"]}
        for row in summary
        if row["dimension"] == "uncertainty_bucket"
    ]
    write_bar_svg(chart, chart_rows, "label", "coverage", "Conformal coverage by ensemble uncertainty")
    card = ExperimentCard(
        name="conformal_intervals",
        hypothesis="train1 calibration residual 和候选模型 spread 应能给出合法的不确定性区间，并暴露预测何时不可靠。",
        data_visibility=(
            "conformal radius 只在最新 train1 calibration fold 上拟合；phase1 标签只用于评估固定区间 coverage。"
        ),
        prototype="围绕官方 hour-weight ensemble 构造对称 split-conformal interval，并用候选模型 spread 划分 uncertainty bucket。",
        metrics={
            "radius": f"{radius:.6f}",
            "coverage": f"{coverage['coverage']:.6f}",
            "mean_width": f"{coverage['mean_width']:.6f}",
        },
        result=f"pooled conformal radius={radius:.6f}，phase1 coverage={coverage['coverage']:.6f}，mean width={coverage['mean_width']:.6f}；覆盖率偏保守，但区间宽度缺少自适应。",
        insight="即使区间过宽，也能量化模型的失败风险；按 spread bucket 的 coverage 可判断 ensemble 是否知道一部分不确定性。",
        next_step="保留并扩展。只有在 train1-only regime 选择稳定时，才尝试按 combo/hour/regime 分组校准 radius。",
        artifacts=(str(csv_path), str(summary_csv), str(failures_csv), str(chart)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Conformal interval and ensemble uncertainty diagnostics")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
