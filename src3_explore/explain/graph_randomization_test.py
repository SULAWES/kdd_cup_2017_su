from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.reporting import write_csv
from src3_explore.explain.common import (
    ExplanationCard,
    adjacency_matrix,
    anchor_matrix,
    explain_dir,
    message_pass,
    phase1_candidate_frame,
    pivot_candidate_rows,
    score_prediction,
    train_target_matrix,
    write_explanation_card,
    write_metric_chart,
)
from src3_explore.common.visibility import load_phase1_context


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, _, combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    context = load_phase1_context(data_dir)
    raw, times, train_combos = train_target_matrix(context)
    residual = raw - anchor_matrix(context, times, train_combos)
    signal = residual[:, [train_combos.index(combo) for combo in combos]]

    result_rows = []
    for seed in range(20):
        for mode in ("identity", "topology", "full", "random"):
            adjacency = adjacency_matrix(mode, combos, signal, seed=seed + 17)
            smoothed = message_pass(pred_matrix, adjacency, layers=1)
            score = score_prediction(actual_matrix.reshape(-1), smoothed.reshape(-1))
            result_rows.append(
                {
                    "seed": seed,
                    "edge_type": mode,
                    "mape": f"{score['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                }
            )

    grouped = []
    for mode in sorted({row["edge_type"] for row in result_rows}):
        values = np.asarray([float(row["mape"]) for row in result_rows if row["edge_type"] == mode], dtype=float)
        grouped.append(
            {
                "edge_type": mode,
                "mean_mape": f"{float(values.mean()):.6f}",
                "std_mape": f"{float(values.std()):.6f}",
                "best_mape": f"{float(values.min()):.6f}",
            }
        )

    out_dir = explain_dir(output_dir)
    rows_csv = out_dir / "graph_randomization_test_rows.csv"
    summary_csv = out_dir / "graph_randomization_test_summary.csv"
    chart = out_dir / "graph_randomization_test_mape.svg"
    write_csv(rows_csv, result_rows)
    write_csv(summary_csv, grouped)
    write_metric_chart(chart, sorted(grouped, key=lambda row: float(row["mean_mape"]), reverse=True), "edge_type", "mean_mape", "Mean MAPE by edge type")
    topology = next(float(row["mean_mape"]) for row in grouped if row["edge_type"] == "topology")
    random_mean = next(float(row["mean_mape"]) for row in grouped if row["edge_type"] == "random")
    card = ExplanationCard(
        name="explain_graph_randomization_test",
        hypothesis="如果五节点真实/手工边有交通语义，topology 边应稳定优于随机边和全连接边。",
        method="在固定 ensemble anchor 上对 identity、topology、full、random 边做 20 seed message passing，并比较 MAPE 分布。",
        expected_falsification="若 topology 平均 MAPE 不优于 random，说明当前图边主要是任意平滑器，而非有效交通边。",
        metrics={"topology_mean_mape": f"{topology:.6f}", "random_mean_mape": f"{random_mean:.6f}", "seeds": 20},
        key_result=f"topology mean MAPE={topology:.6f}，random mean MAPE={random_mean:.6f}。",
        interpretation="真实边若无法打败随机边，五节点图定义本身缺少交通语义；这解释了 GNN 在该任务中没有超过结构化 ensemble。",
        next_step="归档五节点静态图。后续只保留 route/intersection/tollgate lead-lag 图作为更有语义的图替代。",
        artifacts=(str(rows_csv), str(summary_csv), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Randomization test for five-node graph edges")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
