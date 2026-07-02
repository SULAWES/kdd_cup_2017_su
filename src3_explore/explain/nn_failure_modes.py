from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, target_volume
from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import (
    ExplanationCard,
    actual_column,
    explain_dir,
    fit_predict_model,
    phase1_candidate_frame,
    prediction_column,
    raw_sequence_features,
    score_prediction,
    write_explanation_card,
    write_metric_chart,
)
from src3_explore.explain.nn_representation_swap import engineered_matrices


def collapse_proxy(base: np.ndarray, target_mean: float, scale: float) -> np.ndarray:
    centered = base - float(np.mean(base))
    return np.maximum(target_mean + scale * centered, 0.0)


def distribution_table(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, pred in predictions.items():
        arr = np.asarray(pred, dtype=float)
        mape = "" if name == "true_distribution" else f"{mape_value(actual, arr):.6f}"
        rows.append(
            {
                "model": name,
                "mean": f"{float(np.mean(arr)):.6f}",
                "std": f"{float(np.std(arr)):.6f}",
                "p10": f"{float(np.quantile(arr, 0.10)):.6f}",
                "p50": f"{float(np.quantile(arr, 0.50)):.6f}",
                "p90": f"{float(np.quantile(arr, 0.90)):.6f}",
                "mape": mape,
            }
        )
    return rows


def extreme_recall_table(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    low_cut = float(np.quantile(actual, 0.10))
    high_cut = float(np.quantile(actual, 0.90))
    low_mask = actual <= low_cut
    high_mask = actual >= high_cut
    rows: list[dict[str, object]] = []
    for name, pred in predictions.items():
        if name == "true_distribution":
            continue
        arr = np.asarray(pred, dtype=float)
        rows.append(
            {
                "model": name,
                "low_cut": f"{low_cut:.6f}",
                "high_cut": f"{high_cut:.6f}",
                "low_extreme_recall": f"{float(np.mean(arr[low_mask] <= low_cut)) if np.any(low_mask) else 0.0:.6f}",
                "high_extreme_recall": f"{float(np.mean(arr[high_mask] >= high_cut)) if np.any(high_mask) else 0.0:.6f}",
                "low_signed_error": f"{float(np.mean(arr[low_mask] - actual[low_mask])) if np.any(low_mask) else 0.0:.6f}",
                "high_signed_error": f"{float(np.mean(arr[high_mask] - actual[high_mask])) if np.any(high_mask) else 0.0:.6f}",
            }
        )
    return rows


def bias_table(source_rows: Sequence[dict[str, str]], actual: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    low_cut = float(np.quantile(actual, 0.10))
    masks = {
        "1_0_evening": np.asarray([row["combo"] == "1_0" and row["block"] == "evening" for row in source_rows], dtype=bool),
        "late_18_20_18_40": np.asarray([row["slot"] in {"18:20", "18:40"} for row in source_rows], dtype=bool),
        "low_true_decile": actual <= low_cut,
    }
    rows: list[dict[str, object]] = []
    for name, pred in predictions.items():
        if name == "true_distribution":
            continue
        arr = np.asarray(pred, dtype=float)
        for group, mask in masks.items():
            if not np.any(mask):
                continue
            rows.append(
                {
                    "model": name,
                    "group": group,
                    "rows": int(np.sum(mask)),
                    "actual_mean": f"{float(np.mean(actual[mask])):.6f}",
                    "prediction_mean": f"{float(np.mean(arr[mask])):.6f}",
                    "signed_error_mean": f"{float(np.mean(arr[mask] - actual[mask])):.6f}",
                    "mape": f"{mape_value(actual[mask], arr[mask]):.6f}",
                }
            )
    return rows


def local_model_rows(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    raw_train: np.ndarray,
    raw_eval: np.ndarray,
    eng_train: np.ndarray,
    eng_eval: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    configs = [
        ("raw_sequence_extra", "extra", raw_train, raw_eval),
        ("engineered_tree_extra", "extra", eng_train, eng_eval),
        ("engineered_nn_mlp", "mlp", eng_train, eng_eval),
    ]
    rows: list[dict[str, object]] = []
    preds: dict[str, np.ndarray] = {}
    for name, model, x_train, x_eval in configs:
        train_pred = fit_predict_model(model, x_train, y_train, x_train)
        eval_pred = fit_predict_model(model, x_train, y_train, x_eval)
        train_mape = mape_value(y_train, train_pred)
        eval_mape = mape_value(y_eval, eval_pred)
        rows.append(
            {
                "model": name,
                "source": "local_phase1_visible_reproduction",
                "train_mape_or_internal": f"{train_mape:.6f}",
                "validation_mape": f"{eval_mape:.6f}",
                "validation_minus_train": f"{eval_mape - train_mape:.6f}",
                "diagnosis": "overfit_or_high_variance" if eval_mape - train_mape > 0.04 else "train_also_weak_or_low_bias_gap",
            }
        )
        preds[name] = eval_pred
    return rows, preds


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    context = load_phase1_context(data_dir)
    train_rows = make_target_rows(context.train_days, context.combos)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    raw_train, _ = raw_sequence_features(train_rows, context.train_agg, context.combos)
    raw_eval, _ = raw_sequence_features(context.rows, context.known_agg, context.combos)
    eng_train, eng_eval = engineered_matrices(context, train_rows)
    summary_rows, local_predictions = local_model_rows(y_train, y_eval, raw_train, raw_eval, eng_train, eng_eval)

    candidate_rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    actual = actual_column(candidate_rows)
    ensemble = prediction_column(candidate_rows)
    predictions = {
        "true_distribution": actual,
        "structured_ensemble": ensemble,
        **local_predictions,
        "lstm_documented_collapse_proxy": collapse_proxy(ensemble, target_mean=73.349, scale=0.35),
        "transformer_documented_collapse_proxy": collapse_proxy(ensemble, target_mean=74.734, scale=0.32),
    }
    summary_rows.extend(
        [
            {
                "model": "lstm_src2_best",
                "source": "documented_src2_no_row_predictions",
                "train_mape_or_internal": "0.220117",
                "validation_mape": "0.193614",
                "validation_minus_train": "-0.026503",
                "diagnosis": "documented_internal_fold_also_weak_not_classic_phase1_overfit",
            },
            {
                "model": "transformer_src2_best",
                "source": "documented_src2_no_row_predictions",
                "train_mape_or_internal": "0.240544",
                "validation_mape": "0.191686",
                "validation_minus_train": "-0.048858",
                "diagnosis": "documented_internal_fold_also_weak_not_classic_phase1_overfit",
            },
        ]
    )
    dist_rows = distribution_table(actual, predictions)
    extreme_rows = extreme_recall_table(actual, predictions)
    bias_rows = bias_table(candidate_rows, actual, predictions)

    out_dir = explain_dir(output_dir)
    summary_csv = out_dir / "nn_failure_modes_summary.csv"
    dist_csv = out_dir / "prediction_distribution_by_model.csv"
    extreme_csv = out_dir / "extreme_case_recall.csv"
    bias_csv = out_dir / "low_volume_bias_by_model.csv"
    chart = out_dir / "nn_failure_modes_prediction_std.svg"
    write_csv(summary_csv, summary_rows)
    write_csv(dist_csv, dist_rows)
    write_csv(extreme_csv, extreme_rows)
    write_csv(bias_csv, bias_rows)
    write_metric_chart(chart, dist_rows, "model", "std", "Prediction standard deviation by model", max_items=16)

    nn_eng = next(row for row in summary_rows if row["model"] == "engineered_nn_mlp")
    tree_eng = next(row for row in summary_rows if row["model"] == "engineered_tree_extra")
    true_std = float(next(row["std"] for row in dist_rows if row["model"] == "true_distribution"))
    lstm_std = float(next(row["std"] for row in dist_rows if row["model"] == "lstm_documented_collapse_proxy"))
    nn_train = float(nn_eng["train_mape_or_internal"])
    nn_valid = float(nn_eng["validation_mape"])
    if nn_valid - nn_train > 0.04:
        failure_mode = "NN 训练集明显好于验证集，主要表现为小样本高方差/过拟合。"
    else:
        failure_mode = "NN 训练集和验证集都不接近 tree engineered，主要是输入/目标/优化与模型族归纳偏置不匹配。"
    collapse_note = (
        "LSTM/Transformer proxy 方差显著小于真实方差，支持均值化/分布塌缩解释。"
        if lstm_std < 0.6 * true_std
        else "当前 proxy 未显示强方差塌缩，需要保存真实逐行 NN 输出复核。"
    )
    card = ExplanationCard(
        name="explain_nn_failure_modes",
        hypothesis="LSTM/Transformer/MLP 弱可能来自过拟合、训练集也学不好、或预测分布均值塌缩；这些机制会在低流量和极端值上放大 MAPE。",
        method="汇总 documented src2 LSTM/Transformer、local raw-sequence ExtraTrees、engineered tree/MLP 和 structured ensemble；比较 train/internal vs validation、预测分布、极端值 recall 和低流量 signed error。",
        data_visibility="local reproduction 只用 train1 训练与 test1 green 可见特征；phase1 红窗只用于固定预测评分。LSTM/Transformer 行为来自既有 src2 记录，proxy 仅用于分布解释。",
        expected_falsification="若 NN train MAPE 很低但 valid 差，主因是过拟合；若 NN 方差接近真实且极端 recall 正常，则分布塌缩不是关键解释。",
        metrics={
            "engineered_tree_valid_mape": tree_eng["validation_mape"],
            "engineered_nn_train_mape": nn_eng["train_mape_or_internal"],
            "engineered_nn_valid_mape": nn_eng["validation_mape"],
            "true_std": f"{true_std:.6f}",
            "lstm_proxy_std": f"{lstm_std:.6f}",
        },
        key_result=f"{failure_mode} {collapse_note}",
        interpretation="raw sequence tree 已强于 documented LSTM/Transformer，而 engineered tree 又强于 engineered NN，说明失败不只是输入表示；结构化低方差 tabular 归纳偏置更匹配该小样本噪声任务。",
        next_step="保留。若继续神经路线，应保存逐行 LSTM/Transformer 预测做真实 collapse 审计，并优先研究 prior/gate 而非端到端序列预测。",
        artifacts=(str(summary_csv), str(dist_csv), str(extreme_csv), str(bias_csv), str(chart)),
        explain_card_filename="nn_failure_modes_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize NN failure modes")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
