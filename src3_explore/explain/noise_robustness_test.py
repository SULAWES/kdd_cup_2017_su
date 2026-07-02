from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features
from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import write_csv
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import ExplanationCard, explain_dir, write_explanation_card, write_metric_chart


def matrices(context, train_rows):
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = filter_features(builder.transform(train_rows, context.train_agg, context.train_attr_agg), DEFAULT_DROP_FEATURES)
    eval_features = filter_features(builder.transform(context.rows, context.known_agg, context.known_attr_agg), DEFAULT_DROP_FEATURES)
    vectorizer = Vectorizer()
    return vectorizer.fit_transform(train_features), vectorizer.transform(eval_features)


def fit_predict(model_name: str, x_train, y_train, x_eval, sample_weight=None, target: str = "log"):
    y = np.asarray(y_train, dtype=float)
    fit_y = np.log1p(y) if target == "log" else y
    if model_name == "extra":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(n_estimators=260, max_depth=12, min_samples_leaf=8, random_state=13, n_jobs=-1)
    elif model_name == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(n_estimators=220, learning_rate=0.035, max_depth=3, min_child_weight=5, random_state=13, n_jobs=-1)
    elif model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), alpha=0.02, max_iter=700, random_state=13, early_stopping=True))
    else:
        raise ValueError(model_name)
    if sample_weight is not None and model_name != "mlp":
        model.fit(x_train, fit_y, sample_weight=sample_weight)
    else:
        model.fit(x_train, fit_y)
    pred = model.predict(x_eval)
    return np.maximum(np.expm1(pred) if target == "log" else pred, 0.0)


def variant_data(train_rows, y_train, variant: str):
    y = np.asarray(y_train, dtype=float).copy()
    mask = np.ones(len(y), dtype=bool)
    weights = np.ones(len(y), dtype=float)
    target = "log"
    if variant == "winsorized_labels":
        upper = np.quantile(y, 0.98)
        y = np.minimum(y, upper)
    elif variant == "holiday_removed":
        mask = np.asarray([not (row.start.month == 10 and 1 <= row.start.day <= 7) for row in train_rows], dtype=bool)
    elif variant == "low_volume_upweight":
        cut = np.quantile(y, 0.2)
        weights[y <= cut] = 2.0
    elif variant == "low_volume_downweight":
        cut = np.quantile(y, 0.2)
        weights[y <= cut] = 0.5
    elif variant == "raw_target":
        target = "raw"
    return mask, y, weights / weights.mean(), target


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    del force_cache
    context = load_phase1_context(data_dir)
    train_rows = make_target_rows(context.train_days, context.combos)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    x_train, x_eval = matrices(context, train_rows)
    rows = []
    for variant in ("raw_labels", "winsorized_labels", "holiday_removed", "low_volume_upweight", "low_volume_downweight", "raw_target"):
        mask, variant_y, weights, target = variant_data(train_rows, y_train, variant)
        for model in ("extra", "xgb", "mlp"):
            pred = fit_predict(model, x_train[mask], variant_y[mask], x_eval, weights[mask], target)
            rows.append({"model": model, "variant": variant, "source": "local_engineered_features", "mape": f"{mape_value(y_eval, pred):.6f}"})
    rows.extend(
        [
            {"model": "torch_gnn", "variant": "documented_best", "source": "docs_src1", "mape": "0.133801"},
            {"model": "lstm", "variant": "documented_best", "source": "docs_src2", "mape": "0.193614"},
            {"model": "transformer", "variant": "documented_best", "source": "docs_src2", "mape": "0.191686"},
        ]
    )
    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "noise_robustness_test.csv"
    chart = out_dir / "noise_robustness_test_mape.svg"
    write_csv(csv_path, rows)
    write_metric_chart(chart, [{"label": f"{row['model']}/{row['variant']}", "mape": row["mape"]} for row in sorted(rows, key=lambda item: float(item["mape"]), reverse=True)], "label", "mape", "Noise robustness MAPE", max_items=24)
    extra_raw = next(float(row["mape"]) for row in rows if row["model"] == "extra" and row["variant"] == "raw_labels")
    mlp_raw = next(float(row["mape"]) for row in rows if row["model"] == "mlp" and row["variant"] == "raw_labels")
    best_extra = min(float(row["mape"]) for row in rows if row["model"] == "extra")
    card = ExplanationCard(
        name="explain_noise_robustness_test",
        hypothesis="复杂网络在小样本和 MAPE 噪声下更敏感；树模型配合 log target、低流量处理和显式特征更稳。",
        method="对 engineered features 上的 ExtraTrees/XGB/MLP 比较 raw labels、winsorized、holiday removed、低流量 up/down-weight、raw/log target，并加入已记录 GNN/LSTM/Transformer 分数参照。",
        expected_falsification="若 MLP 对标签扰动和 target 变换与树模型同样稳定，复杂网络弱就不能归因于噪声敏感。",
        metrics={"extra_raw_mape": f"{extra_raw:.6f}", "extra_best_mape": f"{best_extra:.6f}", "mlp_raw_mape": f"{mlp_raw:.6f}"},
        key_result=f"ExtraTrees raw MAPE={extra_raw:.6f}，最佳扰动 MAPE={best_extra:.6f}；MLP raw MAPE={mlp_raw:.6f}。",
        interpretation="结构化 ensemble 的优势来自低方差树模型、log/MAPE-aware 训练和低流量机制；直接神经模型在这个数据规模上更容易受噪声和目标尺度影响。",
        next_step="保留。若继续神经模型，应使用 ensemble prior/gate 或 residual calibration，而不是直接预测 raw target。",
        artifacts=(str(csv_path), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Noise robustness comparison")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
