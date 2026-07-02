from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import make_target_rows, project_paths, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features
from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import write_csv
from src3_explore.common.trajectory import merge_trajectory_aggregates, read_trajectory_aggregate, trajectory_green_count
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import ExplanationCard, explain_dir, phase1_candidate_frame, write_explanation_card, write_metric_chart


def feature_dicts(context, train_rows):
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = filter_features(builder.transform(train_rows, context.train_agg, context.train_attr_agg), DEFAULT_DROP_FEATURES)
    eval_features = filter_features(builder.transform(context.rows, context.known_agg, context.known_attr_agg), DEFAULT_DROP_FEATURES)
    return train_features, eval_features


def add_route_signal(data_dir: Path, context, train_rows, train_features, eval_features):
    paths = project_paths(data_dir)
    train_traj = read_trajectory_aggregate([paths["train1_volume"].parents[0] / "trajectories(table 5)_training.csv"])
    test1_traj = read_trajectory_aggregate([data_dir / "dataSets" / "testing_phase1" / "trajectories(table 5)_test1.csv"])
    known_traj = merge_trajectory_aggregates(train_traj, test1_traj)
    for row, features in zip(train_rows, train_features):
        features["route_green_count"] = trajectory_green_count(row, train_traj)
    for row, features in zip(context.rows, eval_features):
        features["route_green_count"] = trajectory_green_count(row, known_traj)


def select_features(rows, allowed):
    return [{key: value for key, value in row.items() if any(key.startswith(prefix) or key == prefix for prefix in allowed)} for row in rows]


def fit_stage(train_features, eval_features, y_train, y_eval):
    from sklearn.ensemble import ExtraTreesRegressor

    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_eval = vectorizer.transform(eval_features)
    model = ExtraTreesRegressor(n_estimators=260, max_depth=12, min_samples_leaf=8, random_state=13, n_jobs=-1)
    model.fit(x_train, np.log1p(y_train))
    pred = np.maximum(np.expm1(model.predict(x_eval)), 0.0)
    return mape_value(y_eval, pred), pred


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    context = load_phase1_context(data_dir)
    train_rows = make_target_rows(context.train_days, context.combos)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    train_features, eval_features = feature_dicts(context, train_rows)
    add_route_signal(data_dir, context, train_rows, train_features, eval_features)
    stages = [
        ("combo_hour_slot", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend"]),
        ("plus_lag7", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_7"]),
        ("plus_rolling_stats", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean"]),
        ("plus_green_obs", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_"]),
        ("plus_attr_etc", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_", "etc_", "model_", "veh_type_"]),
        ("plus_route_signal", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_", "etc_", "model_", "veh_type_", "route_green_count"]),
    ]
    rows = []
    previous = None
    for stage, allowed in stages:
        mape, _ = fit_stage(select_features(train_features, allowed), select_features(eval_features, allowed), y_train, y_eval)
        rows.append(
            {
                "stage": stage,
                "mape": f"{mape:.6f}",
                "delta_vs_previous": "" if previous is None else f"{mape - previous:.6f}",
            }
        )
        previous = mape
    candidate_rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    disagreement_values = np.asarray([float(row["model_disagreement"]) for row in candidate_rows], dtype=float)
    ape = np.asarray([float(row["abs_pct_error"]) for row in candidate_rows], dtype=float)
    disagreement_corr = 0.0 if np.std(disagreement_values) == 0 or np.std(ape) == 0 else float(np.corrcoef(disagreement_values, ape)[0, 1])
    rows.append({"stage": "disagreement_diagnostic", "mape": "", "delta_vs_previous": "", "ape_corr": f"{disagreement_corr:.6f}"})

    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "information_decomposition.csv"
    chart = out_dir / "information_decomposition_mape.svg"
    write_csv(csv_path, rows)
    write_metric_chart(chart, [row for row in rows if row.get("mape")], "stage", "mape", "MAPE by information stage", max_items=12)
    first_mape = float(rows[0]["mape"])
    best_mape = min(float(row["mape"]) for row in rows if row.get("mape"))
    card = ExplanationCard(
        name="explain_information_decomposition",
        hypothesis="当前 ensemble 更强，是因为显式组合了 combo/hour/slot、lag、rolling stats、green obs、attr/ETC、low-volume/route/disagreement 等结构信息。",
        method="用同一个 ExtraTrees 原型递增加入信息源，比较 phase1 诊断 MAPE；disagreement 作为固定候选输出后的误差相关诊断。",
        expected_falsification="若只用 raw time identity 就接近完整特征，结构化特征并不是 ensemble 优势来源。",
        metrics={"base_mape": f"{first_mape:.6f}", "best_stage_mape": f"{best_mape:.6f}", "disagreement_ape_corr": f"{disagreement_corr:.6f}"},
        key_result=f"基础 combo/hour/slot MAPE={first_mape:.6f}，最佳阶段 MAPE={best_mape:.6f}，disagreement 与 APE 相关={disagreement_corr:.6f}。",
        interpretation="误差下降若主要来自显式信息源而非序列模型，说明结构化 ensemble 更适合这个小样本、规则强、噪声重的任务。",
        next_step="保留。下一步可把最有贡献的信息源映射到 residual_atlas 的 failure cases，形成解释优先的 ablation 报告。",
        artifacts=(str(csv_path), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Information decomposition for structured ensemble advantage")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
