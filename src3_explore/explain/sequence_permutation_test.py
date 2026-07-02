from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, target_volume
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import ExplanationCard, explain_dir, fit_predict_model, raw_sequence_features, score_prediction, write_explanation_card, write_metric_chart


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    del force_cache
    context = load_phase1_context(data_dir)
    train_rows = make_target_rows(context.train_days, context.combos)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    hist_perm = list(reversed(range(7)))
    obs_perm = [2, 0, 5, 1, 4, 3]
    variants = {
        "normal_order": {},
        "shuffled_history_order": {"hist_order": hist_perm},
        "shuffled_obs_order": {"obs_order": obs_perm},
        "summary_only": {"summary_only": True},
    }
    rows = []
    for variant, kwargs in variants.items():
        x_train, _ = raw_sequence_features(train_rows, context.train_agg, context.combos, **kwargs)
        x_eval, _ = raw_sequence_features(context.rows, context.known_agg, context.combos, **kwargs)
        for model in ("ridge", "extra"):
            pred = fit_predict_model(model, x_train, y_train, x_eval)
            score = score_prediction(y_eval, pred)
            rows.append(
                {
                    "method": model,
                    "variant": variant,
                    "source": "local_raw_sequence_baseline",
                    "mape": f"{score['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                    "uses_sequence_order": variant == "normal_order",
                }
            )
    rows.extend(
        [
            {
                "method": "lstm",
                "variant": "normal_order",
                "source": "documented_src2_best",
                "mape": "0.193614",
                "signed_error_mean": "",
                "uses_sequence_order": True,
            },
            {
                "method": "transformer",
                "variant": "normal_order",
                "source": "documented_src2_best",
                "mape": "0.191686",
                "signed_error_mean": "",
                "uses_sequence_order": True,
            },
        ]
    )
    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "sequence_permutation_test.csv"
    chart = out_dir / "sequence_permutation_test_mape.svg"
    write_csv(csv_path, rows)
    chart_rows = [{"label": f"{row['method']}/{row['variant']}", "mape": row["mape"]} for row in rows]
    write_metric_chart(chart, sorted(chart_rows, key=lambda row: float(row["mape"]), reverse=True), "label", "mape", "Sequence permutation MAPE", max_items=18)
    normal_extra = next(float(row["mape"]) for row in rows if row["method"] == "extra" and row["variant"] == "normal_order")
    summary_extra = next(float(row["mape"]) for row in rows if row["method"] == "extra" and row["variant"] == "summary_only")
    card = ExplanationCard(
        name="explain_sequence_permutation_test",
        hypothesis="如果 LSTM/Transformer 的优势来自序列顺序，打乱历史或 obs 顺序应显著伤害 raw-sequence baseline。",
        method="比较 raw-sequence Ridge/ExtraTrees 的正常顺序、历史逆序、obs 固定打乱和 summary-only，并加入 src2 已记录 LSTM/Transformer 最佳分数作为参照。",
        expected_falsification="若打乱顺序或 summary-only 与正常顺序接近，说明任务主要依赖聚合统计和显式上下文，而不是长序列顺序模式。",
        metrics={"extra_normal_mape": f"{normal_extra:.6f}", "extra_summary_mape": f"{summary_extra:.6f}", "documented_lstm_mape": "0.193614"},
        key_result=f"ExtraTrees raw sequence 正常顺序 MAPE={normal_extra:.6f}，summary-only MAPE={summary_extra:.6f}。",
        interpretation="若顺序扰动影响很小，LSTM/Transformer 很难通过序列编码获得优势；结构化 ensemble 的显式统计特征更适合小样本任务。",
        next_step="保留。后续若训练神经序列模型，应先证明它比 summary-only baseline 更多利用顺序。",
        artifacts=(str(csv_path), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sequence order permutation test")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
