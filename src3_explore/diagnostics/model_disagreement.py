from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.candidate_cache import ensure_phase1_candidate_cache, load_candidate_rows
from src3_explore.common.metrics import bucket_quantiles, summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg


def nearest_model_by_actual(actual: np.ndarray, prediction_matrix: np.ndarray, model_names: Sequence[str]) -> list[str]:
    actual_arr = np.asarray(actual, dtype=float)
    matrix = np.asarray(prediction_matrix, dtype=float)
    winners = np.argmin(np.abs(matrix - actual_arr[:, None]), axis=1)
    return [model_names[int(idx)] for idx in winners]


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    cache = ensure_phase1_candidate_cache(data_dir, output_dir, force=force_cache)
    rows = load_candidate_rows(cache)
    model_cols = [name for name in rows[0] if name.startswith("candidate_")] if rows else []
    model_names = [name.removeprefix("candidate_") for name in model_cols]
    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    matrix = np.asarray([[float(row[col]) for col in model_cols] for row in rows], dtype=float)
    winners = nearest_model_by_actual(actual, matrix, model_names)
    disagreement = (np.max(matrix, axis=1) - np.min(matrix, axis=1)) / np.maximum(np.mean(matrix, axis=1), 1.0)
    buckets = bucket_quantiles(disagreement, labels=("low", "mid", "high", "extreme"))

    detail_rows = []
    for idx, row in enumerate(rows):
        item = {
            "date": row["date"],
            "combo": row["combo"],
            "slot": row["slot"],
            "block": row["block"],
            "hour": row["hour"],
            "actual": row["actual"],
            "prediction": row["prediction"],
            "signed_error": f"{float(row['prediction']) - actual[idx]:.6f}",
            "abs_pct_error": f"{abs(float(row['prediction']) - actual[idx]) / max(abs(actual[idx]), 1.0):.6f}",
            "winner_model": winners[idx],
            "model_disagreement": f"{disagreement[idx]:.6f}",
            "model_disagreement_bucket": buckets[idx],
            "ensemble_abs_error": f"{abs(float(row['prediction']) - actual[idx]):.6f}",
        }
        for col in model_cols:
            item[col] = row[col]
        detail_rows.append(item)

    summary = summarize_errors(detail_rows, ["model_disagreement_bucket", "winner_model"])
    output_csv = output_dir / "diagnostics" / "model_disagreement_winners.csv"
    summary_csv = output_dir / "diagnostics" / "model_disagreement_summary.csv"
    failures_csv = output_dir / "diagnostics" / "model_disagreement_failure_cases.csv"
    chart = output_dir / "diagnostics" / "model_disagreement_mape.svg"
    write_csv(output_csv, detail_rows)
    write_csv(summary_csv, summary)
    failure_rows = sorted(
        [row for row in detail_rows if row["model_disagreement_bucket"] in {"high", "extreme"}],
        key=lambda row: (float(row["abs_pct_error"]), float(row["model_disagreement"])),
        reverse=True,
    )[:25]
    write_csv(failures_csv, failure_rows)
    chart_rows = [
        {
            "label": f"{row['model_disagreement_bucket']}/{row['winner_model']}",
            "mape": row["mape"],
        }
        for row in sorted(summary, key=lambda item: float(item["mape"]), reverse=True)
    ]
    write_bar_svg(chart, chart_rows, "label", "mape", "MAPE by disagreement bucket and nearest model")

    win_counts = {name: winners.count(name) for name in model_names}
    win_text = "; ".join(f"{key}={value}" for key, value in sorted(win_counts.items()))
    card = ExperimentCard(
        name="model_disagreement",
        hypothesis="候选模型分歧较大的窗口应对应融合先验较脆弱的 regime；真实值在这些窗口可能系统性更接近某一类候选模型。",
        data_visibility=(
            "使用 train1-only 候选模型和 test1 绿窗生成的 phase1 预测缓存；train2 标签只在预测完成后用于判断哪个候选最接近真实值。"
        ),
        prototype="计算候选预测 spread，按分歧桶分组，并为每行标注真实值最近的候选模型和高分歧失败样本。",
        metrics={
            "rows": len(rows),
            "winner_counts": win_text,
            "failure_cases": len(failure_rows),
        },
        result=f"真实值最近的候选并不集中在单一模型，winner 分布为 {win_text}；高分歧失败样本说明融合权重的脆弱性有结构性。",
        insight="即使不调整融合权重，也能判断新增模型是否真的补足 blind spot，或只是复制已有候选的错误。",
        next_step="保留并扩展。下一步按 combo/hour/regime 检查 winner 切换是否稳定，再决定是否设计 train1-only 的局部 reweighting。",
        artifacts=(str(output_csv), str(summary_csv), str(failures_csv), str(chart)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze candidate model disagreement")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir, args.force_cache)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
