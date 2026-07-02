from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.metrics import summarize_errors
from src3_explore.common.reporting import write_csv
from src3_explore.explain.common import (
    ExplanationCard,
    adjacency_matrix,
    evaluate_detail_rows,
    explain_dir,
    mean_pairwise_distance,
    message_pass,
    ordered_candidate_rows,
    phase1_candidate_frame,
    pivot_candidate_rows,
    score_prediction,
    train_target_matrix,
    anchor_matrix,
    write_explanation_card,
    write_metric_chart,
)
from src3_explore.common.visibility import load_phase1_context


def combo_bias_from_train(data_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    context = load_phase1_context(data_dir)
    raw, times, combos = train_target_matrix(context)
    anchor = anchor_matrix(context, times, combos)
    residual = raw - anchor
    bias = residual.mean(axis=0)
    return residual, combos, bias


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, times, combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    ordered_rows = ordered_candidate_rows(rows, times, combos)
    train_residual, train_combos, train_bias = combo_bias_from_train(data_dir)
    bias_map = {combo: train_bias[idx] for idx, combo in enumerate(train_combos)}
    bias = np.asarray([bias_map.get(combo, 0.0) for combo in combos], dtype=float)

    summary_rows = []
    detail_rows = []
    distance_rows = []
    group_rows = []
    rng_seeds = {"random": 31}
    for mode in ("identity", "topology", "corr", "full", "random"):
        adjacency_signal = train_residual[:, [train_combos.index(combo) for combo in combos]]
        adjacency = adjacency_matrix(mode, combos, adjacency_signal, seed=rng_seeds.get(mode, 13))
        for layers in (1, 2):
            for with_embedding in (False, True):
                smoothed = message_pass(pred_matrix, adjacency, layers=layers)
                if with_embedding:
                    smoothed = np.maximum(smoothed + 0.05 * bias[None, :], 0.0)
                method = f"{mode}_layers{layers}_{'emb' if with_embedding else 'noemb'}"
                flat_pred = smoothed.reshape(-1)
                flat_actual = actual_matrix.reshape(-1)
                score = score_prediction(flat_actual, flat_pred)
                before_distance = mean_pairwise_distance(pred_matrix)
                after_distance = mean_pairwise_distance(smoothed)
                mask_10_evening = np.array(
                    [combo == "1_0" and slot.startswith("18") or combo == "1_0" and slot.startswith("17") for _, slot in times for combo in combos]
                )
                signed_10_evening = float(np.mean(flat_pred[mask_10_evening] - flat_actual[mask_10_evening])) if np.any(mask_10_evening) else 0.0
                summary_rows.append(
                    {
                        "method": method,
                        "adjacency": mode,
                        "layers": layers,
                        "with_node_embedding": with_embedding,
                        "mape": f"{score['mape']:.6f}",
                        "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                        "one_zero_evening_signed_error": f"{signed_10_evening:.6f}",
                        "distance_before": f"{before_distance:.6f}",
                        "distance_after": f"{after_distance:.6f}",
                        "distance_ratio": f"{after_distance / max(before_distance, 1e-9):.6f}",
                    }
                )
                detail = evaluate_detail_rows(ordered_rows, flat_pred, method)
                detail_rows.extend(detail)
                grouped = summarize_errors(detail, ["method", "combo"])
                grouped.extend(summarize_errors(detail, ["method", "hour"]))
                grouped.extend(summarize_errors(detail, ["method", "slot"]))
                group_rows.extend(grouped)
                distance_rows.append({"method": method, "distance_before": before_distance, "distance_after": after_distance})

    out_dir = explain_dir(output_dir)
    summary_csv = out_dir / "gnn_message_passing_damage_summary.csv"
    grouped_csv = out_dir / "gnn_message_passing_damage_grouped.csv"
    detail_csv = out_dir / "gnn_message_passing_damage_rows.csv"
    chart = out_dir / "gnn_message_passing_damage_mape.svg"
    write_csv(summary_csv, summary_rows)
    write_csv(grouped_csv, group_rows)
    write_csv(detail_csv, detail_rows)
    write_metric_chart(
        chart,
        sorted(summary_rows, key=lambda row: float(row["mape"]), reverse=True),
        "method",
        "mape",
        "Message passing MAPE damage",
        max_items=18,
    )
    identity_mape = min(float(row["mape"]) for row in summary_rows if row["adjacency"] == "identity")
    non_identity_mape = min(float(row["mape"]) for row in summary_rows if row["adjacency"] != "identity")
    min_distance_ratio = min(float(row["distance_ratio"]) for row in summary_rows)
    card = ExplanationCard(
        name="explain_gnn_message_passing_damage",
        hypothesis="如果五节点图边有交通语义，message passing 应在不压扁节点差异的情况下改善或至少不伤害 ensemble anchor。",
        method="在固定 phase1 ensemble 预测上模拟 identity/topology/corr/full/random adjacency 的 1/2 层 message passing，并比较 with/without node embedding。",
        expected_falsification="若非 identity 图不优于 identity，且节点表示距离明显下降，则说明 message passing 主要带来过度平滑和低流量负迁移。",
        metrics={
            "best_identity_mape": f"{identity_mape:.6f}",
            "best_non_identity_mape": f"{non_identity_mape:.6f}",
            "min_distance_ratio": f"{min_distance_ratio:.6f}",
        },
        key_result=f"最佳非 identity MAPE={non_identity_mape:.6f}，identity MAPE={identity_mape:.6f}，最小节点距离比例={min_distance_ratio:.4f}。",
        interpretation="五节点 message passing 对小样本低流量 combo 容易把其他节点信息传成偏差；结构化 ensemble 通过显式 combo/hour/low-volume 规则避免了这种负迁移。",
        next_step="归档为 GNN 失效解释。后续图实验应转向 route lead-lag 图，而不是调静态五节点 adjacency。",
        artifacts=(str(summary_csv), str(grouped_csv), str(detail_csv), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Explain GNN message passing damage on five-node graph")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
