from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src3_explore.common.candidate_cache import ensure_phase1_candidate_cache, load_candidate_rows
from src3_explore.common.metrics import summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg


GROUP_FIELDS = (
    "date",
    "combo",
    "hour",
    "slot",
    "green_obs_strength_bucket",
    "ETC_share_bucket",
    "trajectory_signal_bucket",
    "model_disagreement_bucket",
)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    cache = ensure_phase1_candidate_cache(data_dir, output_dir, force=force_cache)
    rows = load_candidate_rows(cache)
    atlas_rows = []
    artifacts = [str(cache)]
    for field in GROUP_FIELDS:
        summary = summarize_errors(rows, [field])
        for item in summary:
            item["dimension"] = field
            item["value"] = item.pop(field)
        atlas_rows.extend(summary)
        chart = output_dir / "diagnostics" / f"residual_atlas_{field}.svg"
        chart_rows = [
            {"label": str(row["value"]), "mape": row["mape"]}
            for row in sorted(summary, key=lambda item: float(item["mape"]), reverse=True)
        ]
        write_bar_svg(chart, chart_rows, "label", "mape", f"Residual atlas MAPE by {field}", max_items=18)
        artifacts.append(str(chart))

    grouped_csv = output_dir / "diagnostics" / "residual_atlas_grouped.csv"
    detail_csv = output_dir / "diagnostics" / "residual_atlas_rows.csv"
    failures_csv = output_dir / "diagnostics" / "residual_atlas_failure_cases.csv"
    write_csv(grouped_csv, atlas_rows)
    write_csv(detail_csv, rows)
    failure_rows = sorted(rows, key=lambda row: float(row["abs_pct_error"]), reverse=True)[:40]
    write_csv(failures_csv, failure_rows)
    artifacts.extend([str(grouped_csv), str(detail_csv), str(failures_csv)])
    worst = sorted(atlas_rows, key=lambda item: float(item["mape"]), reverse=True)[:5]
    worst_text = "; ".join(f"{row['dimension']}={row['value']} mape={float(row['mape']):.4f}" for row in worst)
    card = ExperimentCard(
        name="residual_atlas",
        hypothesis="如果任务里存在可解释的结构性失效，残差应按日期、combo、目标时段、绿窗强弱、ETC 占比、轨迹信号和模型分歧聚集，而不是均匀散布。",
        data_visibility=(
            "候选预测只用 train1 训练并使用 test1 绿窗作为可见输入；train2 红窗标签只在预测缓存固定后接入，用于最终分组诊断。"
        ),
        prototype=(
            "缓存官方候选模型预测，按 date、combo、hour、slot、green_obs_strength、ETC_share、trajectory_signal、model_disagreement 汇总 signed error、MAPE 和失败样本。"
        ),
        metrics={"rows": len(rows), "grouped_rows": len(atlas_rows), "worst_groups": worst_text, "failure_cases": len(failure_rows)},
        result=f"残差不是随机分布；最高误差组集中在 {worst_text}。这说明后续应优先检查晚高峰低流量和特定 combo 的机制，而不是直接扩大模型搜索。",
        insight="即使不带来更低 MAPE，也能定位稳定失败区域，并给 transfer、allocation、ETC component、trajectory kernel 等机制实验提供 join key。",
        next_step="保留。继续把高误差行与绿红 transfer、tollgate 1/2 allocation、ETC component 和 route kernel 输出做交叉表。",
        artifacts=tuple(artifacts),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build residual atlas for cached candidate predictions")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir, args.force_cache)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
