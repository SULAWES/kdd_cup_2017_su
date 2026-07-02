from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.explain.common import (
    ExplanationCard,
    adjacency_matrix,
    anchor_matrix,
    explain_dir,
    laplacian_energy,
    pairwise_corr_rows,
    train_target_matrix,
    write_explanation_card,
    write_metric_chart,
)
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context


def rolling_stability_rows(signal: np.ndarray, combos: Sequence[str], chunks: int = 4) -> list[dict[str, object]]:
    parts = np.array_split(signal, chunks)
    rows = []
    for i, left in enumerate(combos):
        for j, right in enumerate(combos):
            if j <= i:
                continue
            corrs = []
            for part in parts:
                x = part[:, i]
                y = part[:, j]
                corr = 0.0 if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
                corrs.append(corr)
            signs = [np.sign(value) for value in corrs if abs(value) > 0.05]
            stable = len(set(signs)) <= 1 if signs else False
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "mean_corr": f"{float(np.mean(corrs)):.6f}",
                    "std_corr": f"{float(np.std(corrs)):.6f}",
                    "stable_sign": stable,
                    "chunk_corrs": ";".join(f"{value:.4f}" for value in corrs),
                }
            )
    return rows


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    del force_cache
    context = load_phase1_context(data_dir)
    raw, times, combos = train_target_matrix(context)
    anchor = anchor_matrix(context, times, combos)
    residual = raw - anchor
    log_residual = np.log1p(raw) - np.log1p(np.maximum(anchor, 0.0))
    matrices = {"raw_volume": raw, "anchor_residual": residual, "log_residual": log_residual}

    corr_rows = []
    lag_rows = []
    stability = []
    smoothness = []
    for name, matrix in matrices.items():
        corr_rows.extend(pairwise_corr_rows(matrix, combos, name, lag=0))
        for lag in (1, 2, 3, 6):
            lag_rows.extend(pairwise_corr_rows(matrix, combos, name, lag=lag))
        if name != "raw_volume":
            stability.extend({"matrix": name, **row} for row in rolling_stability_rows(matrix, combos))
        for mode in ("topology", "full", "corr"):
            adjacency = adjacency_matrix(mode, combos, matrix)
            smoothness.append(
                {
                    "matrix": name,
                    "adjacency": mode,
                    "laplacian_energy": f"{laplacian_energy(matrix, adjacency):.8f}",
                }
            )

    out_dir = explain_dir(output_dir)
    corr_csv = out_dir / "graph_signal_audit_pair_corr.csv"
    lag_csv = out_dir / "graph_signal_audit_lagged_corr.csv"
    stability_csv = out_dir / "graph_signal_audit_rolling_stability.csv"
    smooth_csv = out_dir / "graph_signal_audit_laplacian_energy.csv"
    chart = out_dir / "graph_signal_audit_residual_corr.svg"
    write_csv(corr_csv, corr_rows)
    write_csv(lag_csv, lag_rows)
    write_csv(stability_csv, stability)
    write_csv(smooth_csv, smoothness)
    residual_corr = [row for row in corr_rows if row["matrix"] == "log_residual"]
    write_metric_chart(
        chart,
        [
            {"label": f"{row['left']}/{row['right']}", "corr": row["corr"]}
            for row in sorted(residual_corr, key=lambda item: abs(float(item["corr"])), reverse=True)
        ],
        "label",
        "corr",
        "Log residual node-pair correlation",
    )

    stable_count = sum(1 for row in stability if row["matrix"] == "log_residual" and row["stable_sign"])
    mean_abs_resid_corr = float(np.mean([abs(float(row["corr"])) for row in residual_corr])) if residual_corr else 0.0
    card = ExplanationCard(
        name="explain_graph_signal_audit",
        hypothesis="如果五节点 GNN 有必要，anchor/log residual 在五个 combo 节点之间应有稳定、可复用的图信号。",
        method="构造 raw volume、anchor residual、log residual 的 [time, combo] 矩阵，计算 same-time/lagged correlation、rolling edge stability 和 Laplacian smoothness energy。",
        expected_falsification="若 residual correlation 弱、rolling 符号不稳定、或 topology Laplacian energy 不低于无语义图，则五节点静态图缺少可传递 residual signal。",
        metrics={
            "mean_abs_log_residual_corr": f"{mean_abs_resid_corr:.6f}",
            "stable_log_residual_edges": stable_count,
            "node_pairs": len(residual_corr),
        },
        key_result=f"log residual 平均绝对节点相关为 {mean_abs_resid_corr:.4f}，稳定 residual 边数量为 {stable_count}/{len(residual_corr)}。",
        interpretation="五节点图若没有稳定 residual 边，message passing 很容易传播噪声而不是交通机制；这解释了 GNN 弱于显式结构化 ensemble 的一部分原因。",
        next_step="保留为图信号审计。若要继续图路线，应改用 route/intersection/tollgate lead-lag 图，而不是继续调五节点静态图。",
        artifacts=(str(corr_csv), str(lag_csv), str(stability_csv), str(smooth_csv), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit whether five-node residual graph signal is stable")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
