from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import (
    ExplanationCard,
    adjacency_matrix,
    anchor_matrix,
    explain_dir,
    mean_pairwise_distance,
    message_pass,
    phase1_candidate_frame,
    pivot_candidate_rows,
    score_prediction,
    train_target_matrix,
    write_explanation_card,
    write_metric_chart,
)


def transformed_spaces(pred: np.ndarray, anchor: np.ndarray) -> dict[str, tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]]:
    expected = np.maximum(anchor, 1.0)
    return {
        "raw_prediction": (pred, lambda values: values),
        "log1p_prediction": (np.log1p(np.maximum(pred, 0.0)), lambda values: np.expm1(values)),
        "ratio_to_combo_hour_slot_expected": (pred / expected, lambda values: values * expected),
        "diff_to_combo_hour_slot_expected": (pred - expected, lambda values: values + expected),
        "residual_ratio_relative_to_anchor": ((pred - expected) / expected, lambda values: expected * (1.0 + values)),
    }


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, times, combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    context = load_phase1_context(data_dir)
    expected = anchor_matrix(context, times, combos)
    train_raw, train_times, train_combos = train_target_matrix(context)
    train_anchor_matrix = anchor_matrix(context, train_times, train_combos)
    train_residual = train_raw - train_anchor_matrix

    rows_out: list[dict[str, object]] = []
    for space_name, (values, inverse) in transformed_spaces(pred_matrix, expected).items():
        for mode in ("identity", "topology", "corr", "full", "random"):
            signal = train_residual[:, [train_combos.index(combo) for combo in combos]]
            adjacency = adjacency_matrix(mode, combos, signal, seed=31)
            smoothed = message_pass(values, adjacency, layers=1)
            pred = np.maximum(inverse(smoothed), 0.0)
            score = score_prediction(actual_matrix.reshape(-1), pred.reshape(-1))
            before_distance = mean_pairwise_distance(values)
            after_distance = mean_pairwise_distance(smoothed)
            raw_before = mean_pairwise_distance(pred_matrix)
            raw_after = mean_pairwise_distance(pred)
            rows_out.append(
                {
                    "space": space_name,
                    "adjacency": mode,
                    "layers": 1,
                    "mape": f"{score['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                    "node_distance_before_space": f"{before_distance:.6f}",
                    "node_distance_after_space": f"{after_distance:.6f}",
                    "node_distance_ratio_space": f"{after_distance / max(before_distance, 1e-9):.6f}",
                    "node_distance_before_raw_prediction": f"{raw_before:.6f}",
                    "node_distance_after_raw_prediction": f"{raw_after:.6f}",
                    "node_distance_ratio_raw_prediction": f"{raw_after / max(raw_before, 1e-9):.6f}",
                }
            )

    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "scale_normalized_message_passing.csv"
    chart = out_dir / "scale_normalized_message_passing_mape.svg"
    write_csv(csv_path, rows_out)
    write_metric_chart(
        chart,
        [
            {"label": f"{row['space']}/{row['adjacency']}", "mape": row["mape"]}
            for row in sorted(rows_out, key=lambda item: float(item["mape"]), reverse=True)
        ],
        "label",
        "mape",
        "Scale-normalized message passing MAPE",
        max_items=24,
    )

    spaces = sorted({str(row["space"]) for row in rows_out})
    topology_worse_count = 0
    comparisons = []
    for space in spaces:
        identity = min(float(row["mape"]) for row in rows_out if row["space"] == space and row["adjacency"] == "identity")
        topology = min(float(row["mape"]) for row in rows_out if row["space"] == space and row["adjacency"] == "topology")
        if topology > identity:
            topology_worse_count += 1
        comparisons.append(f"{space}: identity={identity:.6f}, topology={topology:.6f}, delta={topology - identity:+.6f}")
    best_non_identity = min(float(row["mape"]) for row in rows_out if row["adjacency"] != "identity")
    identity_best = min(float(row["mape"]) for row in rows_out if row["adjacency"] == "identity")
    card = ExplanationCard(
        name="explain_scale_normalized_message_passing",
        hypothesis="五节点 GNN 失败可能只是跨 combo 尺度不匹配；若在 log、ratio、diff、anchor residual ratio 空间中 topology 仍差于 identity，则核心问题是图语义而非尺度。",
        method="在 raw/log1p/expected ratio/expected diff/anchor residual ratio 五个空间对固定 ensemble 预测做 identity/topology/corr/full/random 一层 message passing，再逆变换回 raw volume 评分。",
        data_visibility="只使用 train1 统计构造 combo-hour-slot expected，phase1 红窗只用于固定输出后的诊断评分；没有用 phase1 分数选择可晋升参数。",
        expected_falsification="若归一化后 topology 明显接近或优于 identity，说明原始 GNN 有相当部分失败来自尺度混合，而不完全是边语义错误。",
        metrics={
            "best_identity_mape": f"{identity_best:.6f}",
            "best_non_identity_mape": f"{best_non_identity:.6f}",
            "topology_worse_space_count": f"{topology_worse_count}/{len(spaces)}",
        },
        key_result="；".join(comparisons),
        interpretation="topology 若在归一化空间仍系统性更差，说明五节点 label graph 缺少可平滑 residual signal；若某些 ratio 空间改善，则原始 GNN 同时存在尺度混合问题。",
        next_step="保留。若要继续 message passing，应先在 residual/ratio 空间做严格 rolling 约束，并优先替换成 route/process graph。",
        artifacts=(str(csv_path), str(chart)),
        explain_card_filename="scale_normalized_message_passing_summary.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scale-normalized message passing diagnostics")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
