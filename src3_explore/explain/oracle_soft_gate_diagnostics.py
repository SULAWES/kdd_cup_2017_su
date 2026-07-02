from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.features import Vectorizer
from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import write_csv
from src3_explore.explain.common import ExplanationCard, explain_dir, phase1_candidate_frame, write_explanation_card, write_metric_chart


def candidate_columns(rows: Sequence[dict[str, str]]) -> list[str]:
    return [key for key in rows[0] if key.startswith("candidate_")]


def feature_matrix(rows: Sequence[dict[str, str]], model_cols: Sequence[str]) -> np.ndarray:
    features = []
    for row in rows:
        ensemble = float(row["prediction"])
        item = {
            f"combo={row['combo']}": 1.0,
            f"hour={row['hour']}": 1.0,
            f"slot={row['slot']}": 1.0,
            f"block={row['block']}": 1.0,
            f"green_bucket={row['green_obs_strength_bucket']}": 1.0,
            f"etc_bucket={row['ETC_share_bucket']}": 1.0,
            f"traj_bucket={row['trajectory_signal_bucket']}": 1.0,
            f"disagreement_bucket={row['model_disagreement_bucket']}": 1.0,
            "green_obs_strength": float(row["green_obs_strength"]),
            "ETC_share": float(row["ETC_share"]),
            "trajectory_signal": float(row["trajectory_signal"]),
            "model_disagreement": float(row["model_disagreement"]),
            "ensemble_prediction_log": float(np.log1p(max(ensemble, 0.0))),
        }
        for col in model_cols:
            value = float(row[col])
            name = col.removeprefix("candidate_")
            item[f"{name}_relative_to_ensemble"] = (value - ensemble) / max(ensemble, 1.0)
        features.append(item)
    vectorizer = Vectorizer()
    return vectorizer.fit_transform(features)


def day_level_probabilities(rows: Sequence[dict[str, str]], x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    from sklearn.ensemble import RandomForestClassifier

    classes = sorted(set(labels))
    class_index = {label: idx for idx, label in enumerate(classes)}
    dates = np.asarray([row["date"] for row in rows])
    probs = np.zeros((len(rows), len(classes)), dtype=float)
    for date in sorted(set(dates)):
        train_idx = np.where(dates != date)[0]
        test_idx = np.where(dates == date)[0]
        train_labels = labels[train_idx]
        if len(set(train_labels)) < 2:
            only = train_labels[0]
            probs[test_idx, class_index[only]] = 1.0
            continue
        clf = RandomForestClassifier(n_estimators=220, max_depth=5, min_samples_leaf=8, random_state=17, n_jobs=-1)
        clf.fit(x[train_idx], train_labels)
        fold_prob = clf.predict_proba(x[test_idx])
        for local_idx, label in enumerate(clf.classes_):
            probs[test_idx, class_index[label]] = fold_prob[:, local_idx]
    return probs, classes


def topk_contains(prob: np.ndarray, classes: Sequence[str], label: str, k: int) -> bool:
    order = np.argsort(prob)[::-1][: min(k, len(classes))]
    return label in {classes[idx] for idx in order}


def regime_support(
    train_rows: Sequence[dict[str, str]],
    model_cols: Sequence[str],
    winner: str,
    combo: str,
    hour: str,
) -> bool:
    model_col = f"candidate_{winner}"
    groups = (
        [row for row in train_rows if row["combo"] == combo and row["hour"] == hour],
        [row for row in train_rows if row["combo"] == combo],
        list(train_rows),
    )
    for group in groups:
        if len(group) < 8:
            continue
        actual = np.asarray([float(row["actual"]) for row in group], dtype=float)
        ensemble = np.asarray([float(row["prediction"]) for row in group], dtype=float)
        candidate = np.asarray([float(row[model_col]) for row in group], dtype=float)
        cand_mape = mape_value(actual, candidate)
        ensemble_mape = mape_value(actual, ensemble)
        return cand_mape <= ensemble_mape - 0.002
    return False


def margin_bucket_masks(margin: np.ndarray) -> dict[str, np.ndarray]:
    masks = {"all": np.ones(len(margin), dtype=bool)}
    for share in (0.50, 0.25, 0.10):
        threshold = float(np.quantile(margin, 1.0 - share))
        masks[f"top_{int(share * 100)}pct_margin"] = margin >= threshold
    return masks


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    model_cols = candidate_columns(rows)
    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    ensemble = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    candidates = np.asarray([[float(row[col]) for col in model_cols] for row in rows], dtype=float)
    abs_err = np.abs(candidates - actual[:, None])
    order = np.argsort(abs_err, axis=1)
    winner_idx = order[:, 0]
    second_idx = order[:, 1]
    labels = np.asarray([model_cols[idx].removeprefix("candidate_") for idx in winner_idx], dtype=object)
    winner_pred = candidates[np.arange(len(rows)), winner_idx]
    second_err = abs_err[np.arange(len(rows)), second_idx]
    best_err = abs_err[np.arange(len(rows)), winner_idx]
    margin = second_err - best_err
    normalized_margin = margin / np.maximum(actual, 1.0)

    x = feature_matrix(rows, model_cols)
    probs, classes = day_level_probabilities(rows, x, labels)
    pred_idx = np.argmax(probs, axis=1)
    pred_winner = np.asarray([classes[idx] for idx in pred_idx], dtype=object)
    confidence = np.max(probs, axis=1)

    ensemble_mape = mape_value(actual, ensemble)
    oracle_mape = mape_value(actual, winner_pred)
    margin_rows: list[dict[str, object]] = []
    topk_rows: list[dict[str, object]] = []
    for bucket, mask in margin_bucket_masks(normalized_margin).items():
        top1 = float(np.mean(pred_winner[mask] == labels[mask])) if np.any(mask) else 0.0
        row = {
            "bucket": bucket,
            "rows": int(np.sum(mask)),
            "mean_normalized_margin": f"{float(np.mean(normalized_margin[mask])) if np.any(mask) else 0.0:.6f}",
            "winner_top1_recall": f"{top1:.6f}",
            "oracle_mape": f"{mape_value(actual[mask], winner_pred[mask]) if np.any(mask) else 0.0:.6f}",
            "ensemble_mape": f"{mape_value(actual[mask], ensemble[mask]) if np.any(mask) else 0.0:.6f}",
            "oracle_gap": f"{mape_value(actual[mask], ensemble[mask]) - mape_value(actual[mask], winner_pred[mask]) if np.any(mask) else 0.0:.6f}",
        }
        margin_rows.append(row)
        for k in (1, 2, 3):
            recall = float(np.mean([topk_contains(probs[i], classes, str(labels[i]), k) for i in np.where(mask)[0]])) if np.any(mask) else 0.0
            topk_rows.append({"bucket": bucket, "k": k, "winner_recall": f"{recall:.6f}", "rows": int(np.sum(mask))})

    dates = np.asarray([row["date"] for row in rows])
    disagreement = np.asarray([float(row["model_disagreement"]) for row in rows], dtype=float)
    override_rows: list[dict[str, object]] = []
    for conf_threshold in (0.45, 0.55, 0.65):
        override_pred = ensemble.copy()
        selected = np.zeros(len(rows), dtype=bool)
        false_costs = []
        for date in sorted(set(dates)):
            train_idx = np.where(dates != date)[0]
            test_idx = np.where(dates == date)[0]
            disagreement_cut = float(np.quantile(disagreement[train_idx], 0.67))
            train_rows = [rows[idx] for idx in train_idx]
            for idx in test_idx:
                winner = str(pred_winner[idx])
                supported = regime_support(train_rows, model_cols, winner, rows[idx]["combo"], rows[idx]["hour"])
                if confidence[idx] >= conf_threshold and disagreement[idx] >= disagreement_cut and supported:
                    selected[idx] = True
                    candidate_value = float(rows[idx][f"candidate_{winner}"])
                    override_pred[idx] = candidate_value
                    candidate_abs = abs(candidate_value - actual[idx])
                    ensemble_abs = abs(ensemble[idx] - actual[idx])
                    if candidate_abs > ensemble_abs:
                        false_costs.append(candidate_abs - ensemble_abs)
        override_mape = mape_value(actual, override_pred)
        override_rows.append(
            {
                "protocol": "day_level_split_conf_disagreement_regime_support",
                "confidence_threshold": f"{conf_threshold:.2f}",
                "coverage": f"{float(np.mean(selected)):.6f}",
                "selected_rows": int(np.sum(selected)),
                "baseline_ensemble_mape": f"{ensemble_mape:.6f}",
                "override_mape": f"{override_mape:.6f}",
                "override_gain": f"{ensemble_mape - override_mape:.6f}",
                "false_override_cost_abs_mean": f"{float(np.mean(false_costs)) if false_costs else 0.0:.6f}",
            }
        )

    out_dir = explain_dir(output_dir)
    margin_csv = out_dir / "oracle_margin_analysis.csv"
    topk_csv = out_dir / "topk_winner_recall.csv"
    override_csv = out_dir / "selective_override_simulation.csv"
    chart = out_dir / "oracle_soft_gate_topk_recall.svg"
    write_csv(margin_csv, margin_rows)
    write_csv(topk_csv, topk_rows)
    write_csv(override_csv, override_rows)
    write_metric_chart(
        chart,
        [{"label": f"{row['bucket']}/top{row['k']}", "recall": row["winner_recall"]} for row in topk_rows],
        "label",
        "recall",
        "Day-level top-k oracle winner recall",
        max_items=16,
    )

    all_top1 = next(float(row["winner_recall"]) for row in topk_rows if row["bucket"] == "all" and int(row["k"]) == 1)
    all_top2 = next(float(row["winner_recall"]) for row in topk_rows if row["bucket"] == "all" and int(row["k"]) == 2)
    top10_top1 = next(float(row["winner_recall"]) for row in topk_rows if row["bucket"] == "top_10pct_margin" and int(row["k"]) == 1)
    best_override = max(override_rows, key=lambda row: float(row["override_gain"]))
    card = ExplanationCard(
        name="explain_oracle_soft_gate_diagnostics",
        hypothesis="candidate oracle 很低说明模型互补性真实存在，但 row-level hard winner gate 不可靠；更合理的是 margin-aware、top-k、regret-based 的 selective soft gate。",
        method="基于固定候选预测计算 oracle gap 和 winner margin；用 day-level split 预测 oracle winner top-k；仅在高置信、高分歧、训练天同 regime 支持时模拟 selective override。",
        data_visibility="phase1 行只作为诊断观察；gate 预测和 override 评估使用 day-level grouped split，不用随机 row KFold 作主结论，也不把 phase1 sweep 当正式可晋升选择。",
        expected_falsification="若 margin 大样本 top-k recall 仍不升高，或 selective override 覆盖后收益为负，则 hard/soft gate 都不应继续。",
        metrics={
            "ensemble_mape": f"{ensemble_mape:.6f}",
            "oracle_mape": f"{oracle_mape:.6f}",
            "all_top1_recall": f"{all_top1:.6f}",
            "all_top2_recall": f"{all_top2:.6f}",
            "top10_margin_top1_recall": f"{top10_top1:.6f}",
            "best_override_gain": best_override["override_gain"],
            "best_override_coverage": best_override["coverage"],
        },
        key_result=(
            f"ensemble MAPE={ensemble_mape:.6f}, oracle MAPE={oracle_mape:.6f}; "
            f"day-level all top1={all_top1:.3f}, top2={all_top2:.3f}, top10 margin top1={top10_top1:.3f}。"
        ),
        interpretation="oracle gap 大代表候选互补性真实；winner accuracy 低代表逐行 hard gate 不可靠。下一步应减少覆盖、按 margin/top-k/regret 做 soft reweight，而不是直接 winner classifier。",
        next_step="保留并扩展。后续若做 gate，必须转到 train1-only grouped rolling 协议，先优化 regret/coverage 而非 winner accuracy。",
        artifacts=(str(margin_csv), str(topk_csv), str(override_csv), str(chart)),
        explain_card_filename="oracle_soft_gate_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Oracle soft-gate diagnostics")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
