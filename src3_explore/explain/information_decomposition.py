from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, block_name, combine_date_time, make_target_rows, project_paths, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features
from src3_explore.common.metrics import mape_value, summarize_errors
from src3_explore.common.reporting import write_csv
from src3_explore.common.trajectory import merge_trajectory_aggregates, read_trajectory_aggregate, trajectory_green_count
from src3_explore.common.visibility import load_phase1_context
from src3_explore.explain.common import (
    ExplanationCard,
    combo_name,
    combo_slot_anchor,
    explain_dir,
    phase1_candidate_frame,
    write_explanation_card,
    write_metric_chart,
)


def obs_sum(row, known_agg) -> float:
    return float(
        sum(
            known_agg.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0)
            for clock in OBS_TIMES[block_name(row.start)]
        )
    )


def feature_dicts(context, train_rows):
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    builder.fit_stats(train_rows)
    train_features = filter_features(builder.transform(train_rows, context.train_agg, context.train_attr_agg), DEFAULT_DROP_FEATURES)
    eval_features = filter_features(builder.transform(context.rows, context.known_agg, context.known_attr_agg), DEFAULT_DROP_FEATURES)
    return train_features, eval_features


def add_low_volume_features(context, train_rows, train_features, eval_features) -> None:
    anchor = combo_slot_anchor(context)
    anchor_values = list(anchor.values())
    low_cut = float(np.quantile(anchor_values, 0.25)) if anchor_values else 0.0
    for rows, features_list, known in (
        (train_rows, train_features, context.train_agg),
        (context.rows, eval_features, context.known_agg),
    ):
        for row, features in zip(rows, features_list):
            expected = float(anchor.get((combo_name(row.combo), f"{row.start.hour:02d}:{row.start.minute:02d}"), 0.0))
            features["combo_slot_expected"] = expected
            features["low_volume_regime"] = 1.0 if expected <= low_cut else 0.0
            features["is_1_0_evening_late"] = 1.0 if combo_name(row.combo) == "1_0" and row.start.strftime("%H:%M") in {"18:20", "18:40"} else 0.0
            features["obs_to_expected_ratio"] = obs_sum(row, known) / max(expected, 1.0)


def add_route_signal(data_dir: Path, context, train_rows, train_features, eval_features) -> None:
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


def fit_stage(train_features, eval_features, y_train):
    from sklearn.ensemble import ExtraTreesRegressor

    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_eval = vectorizer.transform(eval_features)
    model = ExtraTreesRegressor(n_estimators=260, max_depth=12, min_samples_leaf=8, random_state=13, n_jobs=-1)
    model.fit(x_train, np.log1p(y_train))
    return np.maximum(np.expm1(model.predict(x_eval)), 0.0)


def anchor_prediction(context, rows) -> np.ndarray:
    anchor = combo_slot_anchor(context)
    return np.asarray(
        [anchor.get((combo_name(row.combo), f"{row.start.hour:02d}:{row.start.minute:02d}"), 0.0) for row in rows],
        dtype=float,
    )


def detail_rows(stage: str, meta_rows: Sequence[dict[str, str]], actual: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    low_cut = float(np.quantile(actual, 0.25))
    details = []
    for row, y, value in zip(meta_rows, actual, pred):
        details.append(
            {
                "stage": stage,
                "date": row["date"],
                "combo": row["combo"],
                "hour": row["hour"],
                "slot": row["slot"],
                "block": row["block"],
                "low_volume": "low" if float(y) <= low_cut else "not_low",
                "green_obs_strength_bucket": row.get("green_obs_strength_bucket", ""),
                "actual": f"{float(y):.6f}",
                "prediction": f"{float(value):.6f}",
                "signed_error": f"{float(value - y):.6f}",
                "abs_pct_error": f"{abs(float(value - y)) / max(abs(float(y)), 1.0):.6f}",
            }
        )
    return details


def grouped_stage_rows(details: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fields in (["stage", "combo"], ["stage", "hour"], ["stage", "slot"], ["stage", "low_volume"], ["stage", "green_obs_strength_bucket"]):
        for item in summarize_errors(details, fields):
            stage = item.pop("stage")
            dim_fields = fields[1:]
            value = "/".join(str(item.pop(field)) for field in dim_fields)
            item["stage"] = stage
            item["dimension"] = "/".join(dim_fields)
            item["value"] = value
            rows.append(item)
    return rows


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    context = load_phase1_context(data_dir)
    train_rows = make_target_rows(context.train_days, context.combos)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in context.rows], dtype=float)
    train_features, eval_features = feature_dicts(context, train_rows)
    add_low_volume_features(context, train_rows, train_features, eval_features)
    add_route_signal(data_dir, context, train_rows, train_features, eval_features)
    candidate_rows = phase1_candidate_frame(data_dir, output_dir, force_cache)

    stages = [
        ("M1_combo_hour_slot", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend"]),
        ("M2_plus_lag7", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_7"]),
        ("M3_plus_rolling_stats", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean"]),
        ("M4_plus_green_obs", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_"]),
        ("M5_plus_attr_ETC", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_", "etc_", "model_", "veh_type_"]),
        ("M6_plus_low_volume_regime", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_", "etc_", "model_", "veh_type_", "combo_slot_expected", "low_volume_regime", "is_1_0_evening_late", "obs_to_expected_ratio"]),
        ("M7_plus_route_legal_signal", ["bias", "tollgate=", "direction=", "combo=", "slot=", "dow=", "target_hour", "target_minute", "is_weekend", "lag_", "hist_", "combo_mean", "combo_slot_mean", "obs_", "etc_", "model_", "veh_type_", "combo_slot_expected", "low_volume_regime", "is_1_0_evening_late", "obs_to_expected_ratio", "route_green_count"]),
    ]

    step_rows: list[dict[str, object]] = []
    all_details: list[dict[str, object]] = []
    previous_mape: float | None = None

    anchor_pred = anchor_prediction(context, context.rows)
    m0 = mape_value(y_eval, anchor_pred)
    step_rows.append({"stage": "M0_historical_expected_baseline", "mape": f"{m0:.6f}", "delta_vs_previous": "", "signed_error_mean": f"{float(np.mean(anchor_pred - y_eval)):.6f}"})
    all_details.extend(detail_rows("M0_historical_expected_baseline", candidate_rows, y_eval, anchor_pred))
    previous_mape = m0

    for stage, allowed in stages:
        pred = fit_stage(select_features(train_features, allowed), select_features(eval_features, allowed), y_train)
        mape = mape_value(y_eval, pred)
        step_rows.append(
            {
                "stage": stage,
                "mape": f"{mape:.6f}",
                "delta_vs_previous": f"{mape - previous_mape:.6f}" if previous_mape is not None else "",
                "signed_error_mean": f"{float(np.mean(pred - y_eval)):.6f}",
            }
        )
        all_details.extend(detail_rows(stage, candidate_rows, y_eval, pred))
        previous_mape = mape

    ensemble_pred = np.asarray([float(row["prediction"]) for row in candidate_rows], dtype=float)
    ensemble_mape = mape_value(y_eval, ensemble_pred)
    disagreement = np.asarray([float(row["model_disagreement"]) for row in candidate_rows], dtype=float)
    ape = np.abs(ensemble_pred - y_eval) / np.maximum(np.abs(y_eval), 1.0)
    disagreement_corr = 0.0 if np.std(disagreement) == 0 or np.std(ape) == 0 else float(np.corrcoef(disagreement, ape)[0, 1])
    step_rows.append(
        {
            "stage": "M8_structured_ensemble_plus_disagreement_diagnostic",
            "mape": f"{ensemble_mape:.6f}",
            "delta_vs_previous": f"{ensemble_mape - previous_mape:.6f}" if previous_mape is not None else "",
            "signed_error_mean": f"{float(np.mean(ensemble_pred - y_eval)):.6f}",
            "disagreement_ape_corr": f"{disagreement_corr:.6f}",
        }
    )
    all_details.extend(detail_rows("M8_structured_ensemble_plus_disagreement_diagnostic", candidate_rows, y_eval, ensemble_pred))
    regime_rows = grouped_stage_rows(all_details)

    out_dir = explain_dir(output_dir)
    steps_csv = out_dir / "information_decomposition_steps.csv"
    regime_csv = out_dir / "information_decomposition_by_regime.csv"
    legacy_csv = out_dir / "information_decomposition.csv"
    chart = out_dir / "information_decomposition_mape.svg"
    write_csv(steps_csv, step_rows)
    write_csv(regime_csv, regime_rows)
    write_csv(legacy_csv, step_rows)
    write_metric_chart(chart, step_rows, "stage", "mape", "MAPE by information stage", max_items=12)

    best_stage = min((row for row in step_rows if row.get("mape")), key=lambda row: float(row["mape"]))
    card = ExplanationCard(
        name="explain_information_decomposition",
        hypothesis="当前复杂方案强在结构化归纳偏置：显式编码时间、周周期、rolling、green demand state、ETC/车型、低流量保护、route legal signal 和模型分歧。",
        method="从 M0 historical expected baseline 开始，用同一 ExtraTrees 原型递增加入 M1-M7 信息源；M8 使用固定结构化 ensemble 并把 disagreement 作为 uncertainty diagnostic。",
        data_visibility="训练只使用 train1 标签；phase1 eval 只使用 test1 green/attr/trajectory 可见窗口；phase1 红窗真实值只用于固定输出后的诊断评分。",
        expected_falsification="若 M0/M1 已接近完整模型，或加入 green/low-volume/route 后没有 regime-specific 改善，则 ensemble 的结构化解释不足。",
        metrics={
            "M0_mape": step_rows[0]["mape"],
            "best_stage": best_stage["stage"],
            "best_stage_mape": best_stage["mape"],
            "M8_ensemble_mape": f"{ensemble_mape:.6f}",
            "disagreement_ape_corr": f"{disagreement_corr:.6f}",
        },
        key_result=f"M0 MAPE={step_rows[0]['mape']}，最佳阶段 {best_stage['stage']} MAPE={best_stage['mape']}，M8 ensemble MAPE={ensemble_mape:.6f}。",
        interpretation="当前 ensemble 可解释为结构化 mixture-of-experts：low_volume_block 保护小分母，ratio_lag_7 编码周周期，hour weights 编码早晚误差结构，tree models 适合小样本 tabular 非线性，MLP 提供不同误差形态，obs adjustment 捕捉同日 demand state。",
        next_step="保留并继续细化到 failure case 报告。若扩展 gate，先用 grouped rolling 证明 M6/M8 regime 规则稳定，再做 selective soft reweight。",
        artifacts=(str(steps_csv), str(regime_csv), str(chart)),
        explain_card_filename="information_decomposition_card.md",
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
