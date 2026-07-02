from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.metrics import safe_corr, summarize_errors
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import (
    ExplanationCard,
    anchor_matrix,
    evaluate_detail_rows,
    explain_dir,
    low_volume_mask,
    ordered_candidate_rows,
    phase1_candidate_frame,
    pivot_candidate_rows,
    score_prediction,
    topology_edge_pairs,
    train_target_matrix,
    write_explanation_card,
    write_metric_chart,
)


ALPHAS = (0.02, 0.05, 0.10, 0.20)
WORST_EDGE = ("3_0", "1_0")


def sign_agreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.sign(left) == np.sign(right))) if len(left) else 0.0


def edge_agreement_stats(data_dir: Path, combos: Sequence[str]) -> dict[tuple[str, str], dict[str, float]]:
    context = load_phase1_context(data_dir)
    train_raw, train_times, train_combos = train_target_matrix(context)
    train_anchor = anchor_matrix(context, train_times, train_combos)
    residual = train_raw - train_anchor
    train_index = {combo: idx for idx, combo in enumerate(train_combos)}
    stats = {}
    for source, target in topology_edge_pairs(combos, directed=True):
        left = residual[:, train_index[source]]
        right = residual[:, train_index[target]]
        stats[(source, target)] = {
            "residual_corr": safe_corr(left, right),
            "residual_sign_agreement": sign_agreement(left, right),
        }
    return stats


def smooth_with_edges(pred: np.ndarray, combos: Sequence[str], edges: Sequence[tuple[str, str]], alpha: float) -> np.ndarray:
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    delta = np.zeros_like(pred)
    counts = np.zeros(len(combos), dtype=float)
    for source, target in edges:
        src = combo_index[source]
        dst = combo_index[target]
        delta[:, dst] += pred[:, src] - pred[:, dst]
        counts[dst] += 1.0
    for idx, count in enumerate(counts):
        if count > 0:
            delta[:, idx] /= count
    return np.maximum(pred + alpha * delta, 0.0)


def add_regime_flags(detail: list[dict[str, object]], source_rows: Sequence[dict[str, str]]) -> None:
    low_mask = low_volume_mask(source_rows)
    for idx, item in enumerate(detail):
        row = source_rows[idx]
        item["low_volume"] = "low" if low_mask[idx] else "not_low"
        item["late_1_0_evening"] = (
            "yes"
            if row.get("combo") == "1_0" and row.get("block") == "evening" and row.get("slot") in {"18:20", "18:40"}
            else "no"
        )


def grouped_rows(method: str, detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for fields in (["combo"], ["hour"], ["slot"], ["low_volume"], ["late_1_0_evening"]):
        for item in summarize_errors(detail, fields):
            rows.append(
                {
                    "method": method,
                    "dimension": "/".join(fields),
                    "value": "/".join(str(item[field]) for field in fields),
                    "count": item["count"],
                    "mape": f"{float(item['mape']):.6f}",
                    "signed_error_mean": f"{float(item['signed_error_mean']):.6f}",
                }
            )
    return rows


def edge_variants(combos: Sequence[str], stats: dict[tuple[str, str], dict[str, float]]) -> dict[str, list[tuple[str, str]]]:
    topology = topology_edge_pairs(combos, directed=True)
    return {
        "topology_self_loop_only": [],
        "topology_full": topology,
        "topology_drop_worst_3_0_to_1_0": [edge for edge in topology if edge != WORST_EDGE],
        "topology_keep_positive_residual_agreement_edges": [
            edge for edge in topology if stats[edge]["residual_sign_agreement"] >= 0.5
        ],
    }


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, times, combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    ordered_rows = ordered_candidate_rows(rows, times, combos)
    flat_actual = actual_matrix.reshape(-1)
    flat_pred = pred_matrix.reshape(-1)
    baseline = score_prediction(flat_actual, flat_pred)
    stats = edge_agreement_stats(data_dir, combos)
    variants = edge_variants(combos, stats)

    edge_mask_rows = []
    for variant, edges in variants.items():
        edge_set = set(edges)
        for edge, values in sorted(stats.items()):
            edge_mask_rows.append(
                {
                    "variant": variant,
                    "source": edge[0],
                    "target": edge[1],
                    "edge": f"{edge[0]}->{edge[1]}",
                    "included": edge in edge_set,
                    "residual_corr": f"{values['residual_corr']:.6f}",
                    "residual_sign_agreement": f"{values['residual_sign_agreement']:.6f}",
                }
            )

    summary_rows = []
    regime_rows: list[dict[str, object]] = []
    for variant, edges in variants.items():
        for alpha in ALPHAS:
            pred = smooth_with_edges(pred_matrix, combos, edges, alpha).reshape(-1)
            score = score_prediction(flat_actual, pred)
            method = f"{variant}_alpha{alpha:.2f}"
            summary_rows.append(
                {
                    "variant": variant,
                    "alpha": f"{alpha:.2f}",
                    "edge_count": len(edges),
                    "mape": f"{score['mape']:.6f}",
                    "delta_vs_identity": f"{score['mape'] - baseline['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                }
            )
            detail = evaluate_detail_rows(ordered_rows, pred, method)
            add_regime_flags(detail, ordered_rows)
            regime_rows.extend(grouped_rows(method, detail))

    out_dir = explain_dir(output_dir)
    summary_csv = out_dir / "edge_dropout_sanity_check.csv"
    regime_csv = out_dir / "edge_dropout_sanity_by_regime.csv"
    masks_csv = out_dir / "edge_dropout_sanity_edge_masks.csv"
    chart = out_dir / "edge_dropout_sanity_mape.svg"
    write_csv(summary_csv, summary_rows)
    write_csv(regime_csv, regime_rows)
    write_csv(masks_csv, edge_mask_rows)
    write_metric_chart(
        chart,
        sorted(summary_rows, key=lambda row: float(row["mape"]), reverse=True),
        "variant",
        "mape",
        "Topology edge dropout sanity MAPE",
        max_items=16,
    )

    best_by_variant = {
        variant: min((row for row in summary_rows if row["variant"] == variant), key=lambda row: float(row["mape"]))
        for variant in variants
    }
    keep_positive_edges = len(variants["topology_keep_positive_residual_agreement_edges"])
    card = ExplanationCard(
        name="explain_edge_dropout_sanity_check",
        hypothesis="如果 topology 伤害来自少数坏边，drop worst edge 或只保留 positive residual-agreement edges 应显著接近 identity；若仍弱，说明五节点 topology 整体不稳。",
        method="在固定 ensemble anchor 上比较 self-loop only、full topology、drop `3_0->1_0`、keep only train1 residual sign-agreement >= 0.5 edges 的 unsigned smoothing。",
        data_visibility="edge dropout 规则只用 train1 residual agreement 和前一轮已固定的 worst-edge 诊断；phase1 红窗只用于固定输出后的评分。",
        expected_falsification="如果 drop worst edge 或 positive-agreement 子图明显优于 full topology 并接近 identity，则问题集中在少数 anti-smoothing 边；否则 topology graph 整体不可用。",
        metrics={
            "identity_mape": f"{baseline['mape']:.6f}",
            "full_topology_best_mape": best_by_variant["topology_full"]["mape"],
            "drop_worst_best_mape": best_by_variant["topology_drop_worst_3_0_to_1_0"]["mape"],
            "positive_agreement_edge_count": keep_positive_edges,
            "positive_agreement_best_mape": best_by_variant["topology_keep_positive_residual_agreement_edges"]["mape"],
        },
        key_result=(
            f"full topology best={best_by_variant['topology_full']['mape']}，"
            f"drop 3_0->1_0 best={best_by_variant['topology_drop_worst_3_0_to_1_0']['mape']}，"
            f"positive-agreement best={best_by_variant['topology_keep_positive_residual_agreement_edges']['mape']}，identity={baseline['mape']:.6f}。"
        ),
        interpretation="如果 drop worst edge 只能小幅改善，而 positive-agreement 子图仍不超过 identity，说明五节点 label topology 不是简单删一条边就能修好。",
        next_step="保留为 edge dropout sanity check。后续不要继续围绕五节点 topology 调参，除非有 train1 rolling 支持的 edge selection 协议。",
        artifacts=(str(summary_csv), str(regime_csv), str(masks_csv), str(chart)),
        explain_card_filename="edge_dropout_sanity_check_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Topology edge dropout sanity check")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
