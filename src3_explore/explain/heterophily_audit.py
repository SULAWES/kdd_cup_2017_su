from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.metrics import safe_corr
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import (
    ExplanationCard,
    anchor_matrix,
    explain_dir,
    phase1_candidate_frame,
    pivot_candidate_rows,
    topology_edge_pairs,
    train_target_matrix,
    write_explanation_card,
    write_metric_chart,
)


def sign_agreement(left: np.ndarray, right: np.ndarray) -> float:
    left_sign = np.sign(left)
    right_sign = np.sign(right)
    return float(np.mean(left_sign == right_sign)) if len(left_sign) else 0.0


def flag_agreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left == right)) if len(left) else 0.0


def block_from_slot(slot: str) -> str:
    hour = int(slot.split(":", 1)[0])
    return "morning" if hour < 12 else "evening"


def edge_stats_for_group(
    source: str,
    target: str,
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_residual: np.ndarray,
    target_residual: np.ndarray,
    source_low: np.ndarray,
    target_low: np.ndarray,
    source_error: np.ndarray,
    target_error: np.ndarray,
    meta: dict[str, object],
) -> dict[str, object]:
    return {
        **meta,
        "source": source,
        "target": target,
        "edge": f"{source}-{target}",
        "volume_correlation": f"{safe_corr(source_values, target_values):.6f}",
        "anchor_residual_correlation": f"{safe_corr(source_residual, target_residual):.6f}",
        "residual_sign_agreement": f"{sign_agreement(source_residual, target_residual):.6f}",
        "low_volume_flag_agreement": f"{flag_agreement(source_low, target_low):.6f}",
        "error_sign_agreement": f"{sign_agreement(source_error, target_error):.6f}",
        "rows": len(source_values),
    }


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    context = load_phase1_context(data_dir)
    train_raw, train_times, train_combos = train_target_matrix(context)
    train_anchor = anchor_matrix(context, train_times, train_combos)
    train_residual = train_raw - train_anchor
    train_low_cut = float(np.quantile(train_raw, 0.25))
    train_low = train_raw <= train_low_cut

    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, val_times, val_combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    val_anchor = anchor_matrix(context, val_times, val_combos)
    val_error = pred_matrix - actual_matrix
    val_residual = actual_matrix - val_anchor
    val_low = actual_matrix <= float(np.quantile(actual_matrix, 0.25))

    combo_index_train = {combo: idx for idx, combo in enumerate(train_combos)}
    combo_index_val = {combo: idx for idx, combo in enumerate(val_combos)}
    edges = topology_edge_pairs(val_combos, directed=False)
    stats_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    for source, target in edges:
        si = combo_index_train[source]
        ti = combo_index_train[target]
        vsi = combo_index_val[source]
        vti = combo_index_val[target]
        stats_rows.append(
            edge_stats_for_group(
                source,
                target,
                train_raw[:, si],
                train_raw[:, ti],
                train_residual[:, si],
                train_residual[:, ti],
                train_low[:, si],
                train_low[:, ti],
                val_error[:, vsi],
                val_error[:, vti],
                {"dimension": "overall", "value": "all"},
            )
        )
        for dimension, value, mask in (
            *[
                ("slot", slot, np.asarray([time_slot == slot for _, time_slot in val_times], dtype=bool))
                for slot in sorted({slot for _, slot in val_times})
            ],
            *[
                ("hour", hour, np.asarray([time_slot.startswith(hour) for _, time_slot in val_times], dtype=bool))
                for hour in sorted({slot.split(":", 1)[0] for _, slot in val_times})
            ],
            *[
                ("block", block, np.asarray([block_from_slot(time_slot) == block for _, time_slot in val_times], dtype=bool))
                for block in ("morning", "evening")
            ],
        ):
            if not np.any(mask):
                continue
            regime_rows.append(
                edge_stats_for_group(
                    source,
                    target,
                    actual_matrix[mask, vsi],
                    actual_matrix[mask, vti],
                    val_residual[mask, vsi],
                    val_residual[mask, vti],
                    val_low[mask, vsi],
                    val_low[mask, vti],
                    val_error[mask, vsi],
                    val_error[mask, vti],
                    {"dimension": dimension, "value": value},
                )
            )

    out_dir = explain_dir(output_dir)
    edge_csv = out_dir / "heterophily_edge_stats.csv"
    regime_csv = out_dir / "heterophily_by_regime.csv"
    chart = out_dir / "heterophily_residual_sign_agreement.svg"
    write_csv(edge_csv, stats_rows)
    write_csv(regime_csv, regime_rows)
    write_metric_chart(
        chart,
        sorted(stats_rows, key=lambda row: float(row["residual_sign_agreement"])),
        "edge",
        "residual_sign_agreement",
        "Topology edge residual sign agreement",
        max_items=16,
    )

    anti_residual = [row for row in stats_rows if float(row["residual_sign_agreement"]) < 0.5]
    anti_error = [row for row in stats_rows if float(row["error_sign_agreement"]) < 0.5]
    min_edge = min(stats_rows, key=lambda row: float(row["residual_sign_agreement"]))
    card = ExplanationCard(
        name="explain_heterophily_audit",
        hypothesis="如果 topology 边两端 residual/误差符号经常相反，五节点图是 heterophilic 或 anti-smoothing graph，message passing 会把相反误差信号混合。",
        method="对每条 topology 无向边计算体量相关、anchor residual 相关、residual sign agreement、低流量标记一致率、phase1 error sign agreement，并按 hour/slot/block 分组。",
        data_visibility="train1 标签用于 residual/低流量统计；phase1 红窗只用于固定 ensemble error sign 的事后解释，不参与模型训练或选边。",
        expected_falsification="若 topology 边 residual sign 和 error sign 高度一致，则 message passing 伤害不能由 heterophily 解释，需要转向优化/容量或尺度问题。",
        metrics={
            "topology_edges": len(stats_rows),
            "residual_sign_agreement_lt_0_5": len(anti_residual),
            "error_sign_agreement_lt_0_5": len(anti_error),
            "lowest_residual_agreement_edge": f"{min_edge['edge']}={min_edge['residual_sign_agreement']}",
        },
        key_result=(
            f"{len(anti_residual)}/{len(stats_rows)} 条 topology 边 residual sign agreement < 0.5；"
            f"{len(anti_error)}/{len(stats_rows)} 条边 phase1 error sign agreement < 0.5。"
        ),
        interpretation="若 topology 边两端 residual sign 经常相反，当前五节点图更像 heterophilic / anti-smoothing graph；平滑会把方向相反的误差信号混在一起，造成负迁移。",
        next_step="保留为 GNN 机制解释。后续图结构应基于 route arrival process 或显式 residual compatibility，而不是 tollgate/direction label 相似性。",
        artifacts=(str(edge_csv), str(regime_csv), str(chart)),
        explain_card_filename="heterophily_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit topology edge heterophily")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
