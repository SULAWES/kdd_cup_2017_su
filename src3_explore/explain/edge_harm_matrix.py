from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.metrics import mape_value, summarize_errors
from src3_explore.common.reporting import write_csv
from src3_explore.explain.common import (
    ExplanationCard,
    all_directed_edges,
    evaluate_detail_rows,
    explain_dir,
    late_evening_mask,
    low_volume_mask,
    ordered_candidate_rows,
    phase1_candidate_frame,
    pivot_candidate_rows,
    score_prediction,
    topology_edge_pairs,
    write_explanation_card,
)


ALPHAS = (0.02, 0.05, 0.10, 0.20)


def write_edge_heatmap(path: Path, rows: Sequence[dict[str, object]], combos: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_delta: dict[tuple[str, str], float] = {}
    for row in rows:
        edge = (str(row["source"]), str(row["target"]))
        delta = float(row["delta_mape"])
        if edge not in best_delta or delta > best_delta[edge]:
            best_delta[edge] = delta
    width = 720
    cell = 78
    left = 128
    top = 72
    vmax = max([abs(value) for value in best_delta.values()] + [1e-9])
    height = top + cell * len(combos) + 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Segoe UI, Arial" font-size="18" font-weight="600">Edge harm matrix: worst delta MAPE by directed edge</text>',
    ]
    for j, combo in enumerate(combos):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="56" text-anchor="middle" font-family="Segoe UI, Arial" font-size="12">{escape(combo)}</text>'
        )
    for i, source in enumerate(combos):
        y = top + i * cell + cell / 2
        parts.append(f'<text x="24" y="{y + 4:.1f}" font-family="Segoe UI, Arial" font-size="12">{escape(source)}</text>')
        for j, target in enumerate(combos):
            x = left + j * cell
            y0 = top + i * cell
            if source == target:
                color = "#f3f4f6"
                label = ""
            else:
                value = best_delta.get((source, target), 0.0)
                intensity = min(1.0, abs(value) / vmax)
                if value >= 0:
                    red = 254
                    green = int(230 - 120 * intensity)
                    blue = int(230 - 150 * intensity)
                else:
                    red = int(219 - 120 * intensity)
                    green = int(234 - 75 * intensity)
                    blue = 254
                color = f"#{red:02x}{green:02x}{blue:02x}"
                label = f"{value:+.4f}"
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell - 6}" height="{cell - 6}" fill="{color}" stroke="#d1d5db"/>')
            if label:
                parts.append(
                    f'<text x="{x + (cell - 6) / 2:.1f}" y="{y0 + cell / 2:.1f}" text-anchor="middle" '
                    f'font-family="Segoe UI, Arial" font-size="11">{escape(label)}</text>'
                )
    parts.append('<text x="24" y="{0}" font-family="Segoe UI, Arial" font-size="12" fill="#374151">Positive values mean message passing worsens MAPE.</text>'.format(height - 28))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def add_regime_flags(detail: list[dict[str, object]], source_rows: Sequence[dict[str, str]]) -> None:
    low_mask = low_volume_mask(source_rows)
    for idx, item in enumerate(detail):
        row = source_rows[idx]
        item["low_volume"] = "low" if low_mask[idx] else "not_low"
        item["green_obs_strength_bucket"] = row.get("green_obs_strength_bucket", "")
        item["late_1_0_evening"] = (
            "yes"
            if row.get("combo") == "1_0" and row.get("block") == "evening" and row.get("slot") in {"18:20", "18:40"}
            else "no"
        )


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, times, combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    ordered_rows = ordered_candidate_rows(rows, times, combos)
    flat_actual = actual_matrix.reshape(-1)
    flat_pred = pred_matrix.reshape(-1)
    baseline_mape = mape_value(flat_actual, flat_pred)
    baseline_detail = evaluate_detail_rows(ordered_rows, flat_pred, "baseline_identity")
    add_regime_flags(baseline_detail, ordered_rows)
    group_fields = (["combo"], ["hour"], ["slot"], ["low_volume"], ["green_obs_strength_bucket"], ["late_1_0_evening"])
    baseline_groups = {}
    for fields in group_fields:
        for item in summarize_errors(baseline_detail, fields):
            key = ("/".join(fields), "/".join(str(item[field]) for field in fields))
            baseline_groups[key] = float(item["mape"])

    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    topology_edges = set(topology_edge_pairs(combos, directed=True))
    focus_mask = late_evening_mask(ordered_rows)
    summary_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    for source, target in all_directed_edges(combos):
        src_idx = combo_index[source]
        dst_idx = combo_index[target]
        for alpha in ALPHAS:
            changed = pred_matrix.copy()
            changed[:, dst_idx] = (1.0 - alpha) * pred_matrix[:, dst_idx] + alpha * pred_matrix[:, src_idx]
            flat_changed = np.maximum(changed.reshape(-1), 0.0)
            score = score_prediction(flat_actual, flat_changed)
            focus_mape = mape_value(flat_actual[focus_mask], flat_changed[focus_mask]) if np.any(focus_mask) else 0.0
            focus_signed = float(np.mean(flat_changed[focus_mask] - flat_actual[focus_mask])) if np.any(focus_mask) else 0.0
            method = f"{source}_to_{target}_a{alpha:.2f}"
            summary_rows.append(
                {
                    "source": source,
                    "target": target,
                    "edge": f"{source}->{target}",
                    "topology_edge": (source, target) in topology_edges,
                    "alpha": f"{alpha:.2f}",
                    "overall_mape": f"{score['mape']:.6f}",
                    "delta_mape": f"{score['mape'] - baseline_mape:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                    "one_zero_evening_late_mape": f"{focus_mape:.6f}",
                    "one_zero_evening_late_signed_error": f"{focus_signed:.6f}",
                }
            )
            detail = evaluate_detail_rows(ordered_rows, flat_changed, method)
            add_regime_flags(detail, ordered_rows)
            for fields in group_fields:
                dim = "/".join(fields)
                for item in summarize_errors(detail, fields):
                    value = "/".join(str(item[field]) for field in fields)
                    base = baseline_groups.get((dim, value), 0.0)
                    regime_rows.append(
                        {
                            "source": source,
                            "target": target,
                            "edge": f"{source}->{target}",
                            "topology_edge": (source, target) in topology_edges,
                            "alpha": f"{alpha:.2f}",
                            "dimension": dim,
                            "value": value,
                            "count": item["count"],
                            "mape": f"{float(item['mape']):.6f}",
                            "delta_mape": f"{float(item['mape']) - base:.6f}",
                            "signed_error_mean": f"{float(item['signed_error_mean']):.6f}",
                        }
                    )

    out_dir = explain_dir(output_dir)
    matrix_csv = out_dir / "edge_harm_matrix.csv"
    regime_csv = out_dir / "edge_harm_by_regime.csv"
    heatmap = out_dir / "edge_harm_heatmap.svg"
    write_csv(matrix_csv, summary_rows)
    write_csv(regime_csv, regime_rows)
    write_edge_heatmap(heatmap, summary_rows, combos)

    topology_worst = max((row for row in summary_rows if row["topology_edge"]), key=lambda row: float(row["delta_mape"]))
    late_worst = max(
        (row for row in regime_rows if row["dimension"] == "late_1_0_evening" and row["value"] == "yes"),
        key=lambda row: float(row["delta_mape"]),
    )
    low_worst = max(
        (row for row in regime_rows if row["dimension"] == "low_volume" and row["value"] == "low"),
        key=lambda row: float(row["delta_mape"]),
    )
    card = ExplanationCard(
        name="explain_edge_harm_matrix",
        hypothesis="五节点 message passing 的伤害如果来自错误边语义，单边 u->v 的少量混合也会在特定 combo/hour/low-volume regime 上稳定变坏。",
        method="以固定 phase1 ensemble 预测为 anchor，对每条有向候选边扫描 alpha=0.02/0.05/0.10/0.20，替换目标节点预测为 `(1-alpha)*pred_v+alpha*pred_u`。",
        data_visibility="只使用 phase1 固定候选预测和验证标签做事后诊断；没有用验证红窗参与训练、选边或选择可晋升参数。",
        expected_falsification="如果单边混合多数不伤害，或 topology 边不比非 topology 边更坏，则五节点 GNN 失败不应归因于具体边负迁移。",
        metrics={
            "baseline_mape": f"{baseline_mape:.6f}",
            "worst_topology_edge": f"{topology_worst['edge']} alpha={topology_worst['alpha']}",
            "worst_topology_delta": topology_worst["delta_mape"],
            "worst_late_1_0_delta": late_worst["delta_mape"],
            "worst_low_volume_delta": low_worst["delta_mape"],
        },
        key_result=(
            f"最坏 topology 单边为 {topology_worst['edge']}，alpha={topology_worst['alpha']}，"
            f"overall MAPE delta={topology_worst['delta_mape']}；1_0 evening late-slot 最大 delta={late_worst['delta_mape']}。"
        ),
        interpretation="如果伤害集中在 `1_0`、evening late slot 和低流量行，说明五节点 message passing 把其他节点尺度/误差形态混入小分母 regime，形成负迁移。",
        next_step="保留为 GNN 负迁移证据。后续只应在 route/process graph 或显式 gated residual transfer 上继续，而不是盲目加深五节点 GNN。",
        artifacts=(str(matrix_csv), str(regime_csv), str(heatmap)),
        explain_card_filename="edge_harm_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Per-edge message passing harm matrix")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
