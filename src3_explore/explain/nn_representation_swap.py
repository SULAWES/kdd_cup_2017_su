from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import ExplanationCard, explain_dir, fit_predict_model, raw_sequence_features, score_prediction, write_explanation_card, write_metric_chart


def engineered_matrices(context, train_rows):
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = filter_features(builder.transform(train_rows, context.train_agg, context.train_attr_agg), DEFAULT_DROP_FEATURES)
    eval_features = filter_features(builder.transform(context.rows, context.known_agg, context.known_attr_agg), DEFAULT_DROP_FEATURES)
    vectorizer = Vectorizer()
    return vectorizer.fit_transform(train_features), vectorizer.transform(eval_features)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    del force_cache
    context = load_phase1_context(data_dir)
    train_rows = make_target_rows(context.train_days, context.combos)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    raw_train, _ = raw_sequence_features(train_rows, context.train_agg, context.combos)
    raw_eval, _ = raw_sequence_features(context.rows, context.known_agg, context.combos)
    eng_train, eng_eval = engineered_matrices(context, train_rows)
    rows = []
    for model in ("mlp", "extra"):
        for representation, x_train, x_eval in (
            ("raw_sequence", raw_train, raw_eval),
            ("engineered_features", eng_train, eng_eval),
        ):
            pred = fit_predict_model(model, x_train, y_train, x_eval)
            score = score_prediction(y_eval, pred)
            rows.append(
                {
                    "model_family": "NN" if model == "mlp" else "tree",
                    "model": model,
                    "representation": representation,
                    "mape": f"{score['mape']:.6f}",
                    "signed_error_mean": f"{score['signed_error_mean']:.6f}",
                }
            )
    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "nn_representation_swap.csv"
    chart = out_dir / "nn_representation_swap_mape.svg"
    write_csv(csv_path, rows)
    write_metric_chart(
        chart,
        [{"label": f"{row['model']}/{row['representation']}", "mape": row["mape"]} for row in sorted(rows, key=lambda item: float(item["mape"]), reverse=True)],
        "label",
        "mape",
        "Representation swap MAPE",
    )
    tree_engineered = next(float(row["mape"]) for row in rows if row["model"] == "extra" and row["representation"] == "engineered_features")
    nn_engineered = next(float(row["mape"]) for row in rows if row["model"] == "mlp" and row["representation"] == "engineered_features")
    tree_raw = next(float(row["mape"]) for row in rows if row["model"] == "extra" and row["representation"] == "raw_sequence")
    card = ExplanationCard(
        name="explain_nn_representation_swap",
        hypothesis="如果 LSTM/Transformer 弱主要是表示问题，tree on raw sequence 也应弱；如果是模型族问题，NN on engineered features 仍会弱于 tree。",
        method="在同一 phase1 可见性下比较 MLP/ExtraTrees × raw sequence/engineered features 四组组合。",
        expected_falsification="若 NN on engineered features 接近 tree on engineered features，则主要问题是输入表示；若 tree on raw sequence 已强于 NN，则主要问题是模型族和训练方差。",
        metrics={
            "tree_engineered_mape": f"{tree_engineered:.6f}",
            "nn_engineered_mape": f"{nn_engineered:.6f}",
            "tree_raw_mape": f"{tree_raw:.6f}",
        },
        key_result=f"tree engineered MAPE={tree_engineered:.6f}，NN engineered MAPE={nn_engineered:.6f}，tree raw MAPE={tree_raw:.6f}。",
        interpretation="结构化 ensemble 的优势来自显式特征、树模型低方差和 MAPE-aware 处理的组合；神经模型即使用 engineered features 也未必稳定。",
        next_step="保留。后续神经路线应优先做 prior/gate，而不是从 raw sequence 端到端预测。",
        artifacts=(str(csv_path), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Swap model family and input representation")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
