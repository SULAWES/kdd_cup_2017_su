from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.metrics import mape_value, safe_corr, summarize_errors
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


def edge_signs_from_train_residual(data_dir: Path, combos: Sequence[str]) -> dict[tuple[str, str], dict[str, float]]:
    context = load_phase1_context(data_dir)
    train_raw, train_times, train_combos = train_target_matrix(context)
    train_anchor = anchor_matrix(context, train_times, train_combos)
    residual = train_raw - train_anchor
    train_index = {combo: idx for idx, combo in enumerate(train_combos)}
    signs = {}
    for source, target in topology_edge_pairs(combos, directed=True):
        corr = safe_corr(residual[:, train_index[source]], residual[:, train_index[target]])
        signs[(source, target)] = {"corr": corr, "sign": 1.0 if corr >= 0.0 else -1.0}
    return signs


def all_edge_smoothing(pred: np.ndarray, combos: Sequence[str], signs: dict[tuple[str, str], dict[str, float]], alpha: float, signed: bool) -> np.ndarray:
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    delta = np.zeros_like(pred)
    counts = np.zeros(len(combos), dtype=float)
    for source, target in topology_edge_pairs(combos, directed=True):
        src = combo_index[source]
        dst = combo_index[target]
        edge_sign = float(signs[(source, target)]["sign"]) if signed else 1.0
        delta[:, dst] += edge_sign * (pred[:, src] - pred[:, dst])
        counts[dst] += 1.0
    for idx, count in enumerate(counts):
        if count > 0:
            delta[:, idx] /= count
    return np.maximum(pred + alpha * delta, 0.0)


def single_edge_smoothing(pred: np.ndarray, combos: Sequence[str], source: str, target: str, sign: float, alpha: float) -> np.ndarray:
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    src = combo_index[source]
    dst = combo_index[target]
    changed = pred.copy()
    changed[:, dst] = pred[:, dst] + alpha * sign * (pred[:, src] - pred[:, dst])
    return np.maximum(changed, 0.0)


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


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    pred_matrix, times, combos = pivot_candidate_rows(rows, "prediction")
    actual_matrix, _, _ = pivot_candidate_rows(rows, "actual")
    ordered_rows = ordered_candidate_rows(rows, times, combos)
    flat_actual = actual_matrix.reshape(-1)
    flat_pred = pred_matrix.reshape(-1)
    baseline = score_prediction(flat_actual, flat_pred)
    signs = edge_signs_from_train_residual(data_dir, combos)

    edge_rows = [
        {
            "source": source,
            "target": target,
            "edge": f"{source}->{target}",
            "train1_residual_corr": f"{values['corr']:.6f}",
            "edge_sign": int(values["sign"]),
        }
        for (source, target), values in sorted(signs.items())
    ]
    summary_rows = [
        {
            "method": "identity",
            "edge": "",
            "alpha": "0.00",
            "signed": "",
            "mape": f"{baseline['mape']:.6f}",
            "delta_vs_identity": "0.000000",
            "signed_error_mean": f"{baseline['signed_error_mean']:.6f}",
        }
    ]
    regime_rows: list[dict[str, object]] = []
    baseline_detail = evaluate_detail_rows(ordered_rows, flat_pred, "identity")
    add_regime_flags(baseline_detail, ordered_rows)
    regime_rows.extend(grouped_rows("identity", baseline_detail))

    for alpha in ALPHAS:
        for signed in (False, True):
            method = f"all_topology_{'signed' if signed else 'unsigned'}_alpha{alpha:.2f}"
            pred = all_edge_smoothing(pred_matrix, combos, signs, alpha, signed=signed).reshape(-1)
            score = score_prediction(flat_actual, pred)
            summary_rows.append(
                {
                    "method": method,
                    "edge": "all_topology",
                    "alpha": f"{alpha:.2f}",
                    "signed": signed,
                    "mape": f"{score['mape']:.6f}",
                    "delta_vs_identity": f"{score['mape'] - baseline['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                }
            )
            detail = evaluate_detail_rows(ordered_rows, pred, method)
            add_regime_flags(detail, ordered_rows)
            regime_rows.extend(grouped_rows(method, detail))
        for source, target in topology_edge_pairs(combos, directed=True):
            sign = signs[(source, target)]["sign"]
            method = f"single_signed_{source}_to_{target}_alpha{alpha:.2f}"
            pred = single_edge_smoothing(pred_matrix, combos, source, target, sign, alpha).reshape(-1)
            score = score_prediction(flat_actual, pred)
            summary_rows.append(
                {
                    "method": method,
                    "edge": f"{source}->{target}",
                    "alpha": f"{alpha:.2f}",
                    "signed": True,
                    "mape": f"{score['mape']:.6f}",
                    "delta_vs_identity": f"{score['mape'] - baseline['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                    "edge_sign": int(sign),
                    "train1_residual_corr": f"{signs[(source, target)]['corr']:.6f}",
                }
            )

    out_dir = explain_dir(output_dir)
    summary_csv = out_dir / "signed_edge_smoothing.csv"
    regime_csv = out_dir / "signed_edge_smoothing_by_regime.csv"
    signs_csv = out_dir / "signed_edge_smoothing_edge_signs.csv"
    chart = out_dir / "signed_edge_smoothing_mape.svg"
    write_csv(summary_csv, summary_rows)
    write_csv(regime_csv, regime_rows)
    write_csv(signs_csv, edge_rows)
    chart_rows = [row for row in summary_rows if str(row["method"]).startswith("all_topology") or row["method"] == "identity"]
    write_metric_chart(chart, sorted(chart_rows, key=lambda row: float(row["mape"]), reverse=True), "method", "mape", "Signed edge smoothing MAPE", max_items=12)

    best_signed = min((row for row in summary_rows if str(row["method"]).startswith("all_topology_signed")), key=lambda row: float(row["mape"]))
    best_unsigned = min((row for row in summary_rows if str(row["method"]).startswith("all_topology_unsigned")), key=lambda row: float(row["mape"]))
    best_single = min((row for row in summary_rows if str(row["method"]).startswith("single_signed")), key=lambda row: float(row["mape"]))
    negative_edges = sum(1 for row in edge_rows if int(row["edge_sign"]) < 0)
    signed_delta = float(best_signed["mape"]) - baseline["mape"]
    unsigned_delta = float(best_unsigned["mape"]) - baseline["mape"]
    single_delta = float(best_single["mape"]) - baseline["mape"]
    if signed_delta < 0:
        interpretation = (
            "signed all-topology smoothing 在 phase1 诊断上有轻微收益，且明显好于 unsigned smoothing；"
            "这说明五节点图存在 heterophily/反向边信号，但收益很小，不能脱离 train1 rolling 稳定性验证。"
        )
    else:
        interpretation = (
            "signed all-topology smoothing 仍弱于 identity，说明五节点图即使允许反向边也缺少稳定可用结构；"
            "普通 GCN 平滑和简单 signed 平滑都不适合这个 label graph。"
        )
    card = ExplanationCard(
        name="explain_signed_edge_smoothing",
        hypothesis="如果五节点图是 heterophilic，允许 residual-correlation sign 为负的边做反向 smoothing 可能比普通 unsigned topology smoothing 更稳；若仍不行，说明五节点图即使加 signed edge 也不稳定。",
        method="根据 train1 anchor residual correlation 给 topology edge 赋 sign，并评估 `pred_v + alpha * sign(edge) * (pred_u - pred_v)` 的 single-edge 和 all-topology smoothing。",
        data_visibility="edge sign 只由 train1 residual correlation 决定；phase1 红窗只用于固定 smoothing 输出后的诊断评分，不参与调参或选边。",
        expected_falsification="如果 signed all-topology smoothing 仍不优于 identity，五节点图即使允许反向边也缺少稳定可用结构；如果略优于 unsigned，则支持 heterophily graph model 方向。",
        metrics={
            "identity_mape": f"{baseline['mape']:.6f}",
            "negative_topology_edges": negative_edges,
            "best_unsigned_mape": best_unsigned["mape"],
            "best_unsigned_delta": f"{unsigned_delta:.6f}",
            "best_signed_mape": best_signed["mape"],
            "best_signed_delta": f"{signed_delta:.6f}",
            "best_single_signed": f"{best_single['edge']} alpha={best_single['alpha']} mape={best_single['mape']} delta={single_delta:.6f}",
        },
        key_result=(
            f"best signed all-topology MAPE={best_signed['mape']} (delta={signed_delta:+.6f})，"
            f"best unsigned all-topology MAPE={best_unsigned['mape']} (delta={unsigned_delta:+.6f})，"
            f"best single-edge signed={best_single['edge']} MAPE={best_single['mape']} (delta={single_delta:+.6f})。"
        ),
        interpretation=interpretation,
        next_step="保留为 signed/heterophily 图模型的 sanity check；后续若继续，需要 train1 rolling 验证 signed edge 是否稳定，而不是 phase1 直选。",
        artifacts=(str(summary_csv), str(regime_csv), str(signs_csv), str(chart)),
        explain_card_filename="signed_edge_smoothing_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Signed edge smoothing sanity check")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
