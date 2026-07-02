from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, project_paths, target_volume

from src3_explore.common.metrics import safe_corr
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
from src3_explore.common.trajectory import INTERSECTIONS, read_trajectory_aggregate, route_count_at_lag
from src3_explore.common.visibility import load_phase1_context


def kernel_feature_rows(context, trajectory_agg, lags: Sequence[int]) -> list[dict[str, object]]:
    rows = []
    for row in make_target_rows(context.train_days, context.combos):
        y = float(target_volume(context.train_agg, row))
        for intersection in INTERSECTIONS:
            for lag in lags:
                rows.append(
                    {
                        "date": str(row.start.date()),
                        "combo": f"{row.tollgate_id}_{row.direction}",
                        "tollgate_id": row.tollgate_id,
                        "direction": row.direction,
                        "block": "morning" if row.start.hour < 12 else "evening",
                        "slot": f"{row.start.hour:02d}:{row.start.minute:02d}",
                        "intersection": intersection,
                        "lag_minutes": lag,
                        "route_count": route_count_at_lag(row, trajectory_agg, intersection, lag),
                        "actual": y,
                    }
                )
    return rows


def summarize_kernel(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups = {}
    for row in rows:
        key = (row["intersection"], row["tollgate_id"], row["lag_minutes"], row["block"])
        groups.setdefault(key, {"x": [], "y": []})
        groups[key]["x"].append(float(row["route_count"]))
        groups[key]["y"].append(float(row["actual"]))
    out = []
    for key, value in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), int(item[0][2]), str(item[0][3]))):
        x = np.asarray(value["x"], dtype=float)
        y = np.asarray(value["y"], dtype=float)
        corr = safe_corr(x, y)
        out.append(
            {
                "intersection": key[0],
                "tollgate_id": key[1],
                "lag_minutes": key[2],
                "block": key[3],
                "corr": f"{corr:.6f}",
                "route_count_mean": f"{float(x.mean()) if len(x) else 0.0:.6f}",
                "actual_mean": f"{float(y.mean()) if len(y) else 0.0:.6f}",
            }
        )
    return out


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    context = load_phase1_context(data_dir)
    paths = project_paths(data_dir)
    traj = read_trajectory_aggregate([data_dir / "dataSets" / "training" / "trajectories(table 5)_training.csv"])
    lags = (20, 40, 60, 80, 100, 120)
    detail = kernel_feature_rows(context, traj, lags)
    summary = summarize_kernel(detail)
    out_dir = output_dir / "mechanisms"
    detail_csv = out_dir / "route_arrival_kernel_rows.csv"
    summary_csv = out_dir / "route_arrival_kernel_summary.csv"
    chart = out_dir / "route_arrival_kernel_top_corr.svg"
    write_csv(detail_csv, detail)
    write_csv(summary_csv, summary)
    chart_rows = [
        {
            "label": f"{row['intersection']}->{row['tollgate_id']} {row['block']} lag={row['lag_minutes']}",
            "corr": row["corr"],
        }
        for row in sorted(summary, key=lambda item: abs(float(item["corr"])), reverse=True)
    ]
    write_bar_svg(chart, chart_rows, "label", "corr", "Route arrival kernel strongest correlations", max_items=18)
    best = max(summary, key=lambda row: abs(float(row["corr"]))) if summary else {"corr": "0"}
    card = ExperimentCard(
        name="route_arrival_kernel",
        hypothesis="上游 route/trajectory count 经过 travel-time lag 后，应能解释一部分 tollgate volume；这比继续调五节点 GNN 更接近真实机制。",
        data_visibility=(
            "机制分析只使用 train1 trajectory 和 train1 volume 标签；不调参、不重跑五节点 GNN，也不使用 phase1 标签。"
        ),
        prototype="对 intersection→tollgate route count 计算 20-120 分钟 lag，与目标 tollgate volume 做相关性表和最强相关图。",
        metrics={"rows": len(detail), "max_abs_corr": f"{abs(float(best['corr'])):.6f}"},
        result=(
            f"最强 lead-lag raw correlation 的 abs(corr)={abs(float(best['corr'])):.6f}；"
            f"route 信号存在，但强度不足以直接替代现有树模型路线。"
        ),
        insight="即使不提升分数，也能识别哪些上游 route、lag 和 tollgate 组合具有机制解释力，避免继续在弱五节点图上调参。",
        next_step="保留为机制候选。只有当 lag 特征满足绿色窗口可见性并通过 train1 rolling 后，才考虑进入预测候选。",
        artifacts=(str(detail_csv), str(summary_csv), str(chart)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Route arrival lead-lag kernel analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
