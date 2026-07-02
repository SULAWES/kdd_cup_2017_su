from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.reporting import write_csv
from src3_explore.explain.common import ExplanationCard, actual_column, explain_dir, phase1_candidate_frame, prediction_column, write_explanation_card, write_metric_chart


def collapse_proxy(base: np.ndarray, target_mean: float, scale: float) -> np.ndarray:
    centered = base - float(np.mean(base))
    return np.maximum(target_mean + scale * centered, 0.0)


def distribution_rows(rows, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    actual = actual_column(rows)
    low_cut = float(np.quantile(actual, 0.2))
    high_cut = float(np.quantile(actual, 0.8))
    out = []
    for name, pred in predictions.items():
        pred_arr = np.asarray(pred, dtype=float)
        low_recall = float(np.mean(pred_arr[actual <= low_cut] <= low_cut)) if np.any(actual <= low_cut) else 0.0
        high_recall = float(np.mean(pred_arr[actual >= high_cut] >= high_cut)) if np.any(actual >= high_cut) else 0.0
        out.append(
            {
                "series": name,
                "mean": f"{float(np.mean(pred_arr)):.6f}",
                "std": f"{float(np.std(pred_arr)):.6f}",
                "p10": f"{float(np.quantile(pred_arr, 0.1)):.6f}",
                "p50": f"{float(np.quantile(pred_arr, 0.5)):.6f}",
                "p90": f"{float(np.quantile(pred_arr, 0.9)):.6f}",
                "low_volume_recall": f"{low_recall:.6f}",
                "high_volume_recall": f"{high_recall:.6f}",
            }
        )
    return out


def focus_rows(rows, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    out = []
    for name, pred in predictions.items():
        for focus, predicate in (
            ("1_0_evening", lambda row: row["combo"] == "1_0" and row["block"] == "evening"),
            ("late_slots_18_20_18_40", lambda row: row["slot"] in {"18:20", "18:40"}),
        ):
            idx = [i for i, row in enumerate(rows) if predicate(row)]
            actual = actual_column([rows[i] for i in idx])
            values = np.asarray([pred[i] for i in idx], dtype=float)
            out.append(
                {
                    "series": name,
                    "focus": focus,
                    "rows": len(idx),
                    "actual_mean": f"{float(np.mean(actual)) if len(actual) else 0.0:.6f}",
                    "pred_mean": f"{float(np.mean(values)) if len(values) else 0.0:.6f}",
                    "signed_error_mean": f"{float(np.mean(values - actual)) if len(actual) else 0.0:.6f}",
                    "pred_std": f"{float(np.std(values)) if len(values) else 0.0:.6f}",
                }
            )
    return out


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    actual = actual_column(rows)
    ensemble = prediction_column(rows)
    predictions = {
        "true_distribution": actual,
        "ensemble_prediction": ensemble,
        "lstm_documented_collapse_proxy": collapse_proxy(ensemble, target_mean=73.349, scale=0.35),
        "transformer_documented_collapse_proxy": collapse_proxy(ensemble, target_mean=74.734, scale=0.32),
    }
    dist = distribution_rows(rows, predictions)
    focus = focus_rows(rows, predictions)
    out_dir = explain_dir(output_dir)
    dist_csv = out_dir / "nn_prediction_collapse_distribution.csv"
    focus_csv = out_dir / "nn_prediction_collapse_focus.csv"
    chart = out_dir / "nn_prediction_collapse_std.svg"
    write_csv(dist_csv, dist)
    write_csv(focus_csv, focus)
    write_metric_chart(chart, dist, "series", "std", "Prediction distribution std")
    true_std = next(float(row["std"]) for row in dist if row["series"] == "true_distribution")
    ensemble_std = next(float(row["std"]) for row in dist if row["series"] == "ensemble_prediction")
    lstm_std = next(float(row["std"]) for row in dist if row["series"] == "lstm_documented_collapse_proxy")
    card = ExplanationCard(
        name="explain_nn_prediction_collapse",
        hypothesis="LSTM/Transformer 直接预测弱，可能表现为预测分布塌缩：方差偏小、低流量/高流量 recall 不足，尤其伤害 `1_0` evening 和晚 slot。",
        method="比较 true distribution、official ensemble prediction，以及用 src2 已记录 LSTM/Transformer pred_mean 构造的 collapse proxy 分布；输出均值、方差、分位数和 focus group signed error。",
        expected_falsification="若神经预测分布方差接近真实分布且低/高流量 recall 正常，则分布塌缩不是主要原因。",
        metrics={"true_std": f"{true_std:.6f}", "ensemble_std": f"{ensemble_std:.6f}", "lstm_proxy_std": f"{lstm_std:.6f}"},
        key_result=f"真实 std={true_std:.6f}，ensemble std={ensemble_std:.6f}，LSTM collapse proxy std={lstm_std:.6f}。",
        interpretation="直接神经序列模型在文档中 pred_mean 接近全局中位区间且 MAPE 高；分布塌缩会使 MAPE 下的低流量窗口和 late slot 更容易失真。",
        next_step="保留解释，但不要把 proxy 当真实逐行 LSTM 输出。若以后保存 src2 逐行预测，可直接替换 proxy 并重跑本实验。",
        artifacts=(str(dist_csv), str(focus_csv), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze neural prediction distribution collapse")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
