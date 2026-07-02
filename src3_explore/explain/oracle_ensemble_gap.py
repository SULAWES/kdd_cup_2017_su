from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import write_csv
from src3_explore.explain.common import ExplanationCard, explain_dir, phase1_candidate_frame, write_explanation_card, write_metric_chart


def winner_features(rows):
    features = []
    labels = []
    for row in rows:
        item = {
            f"combo={row['combo']}": 1.0,
            f"hour={row['hour']}": 1.0,
            "green_obs_strength": float(row["green_obs_strength"]),
            "ETC_share": float(row["ETC_share"]),
            "trajectory_signal": float(row["trajectory_signal"]),
            "model_disagreement": float(row["model_disagreement"]),
            "is_1_0": 1.0 if row["combo"] == "1_0" else 0.0,
            "is_evening": 1.0 if row["block"] == "evening" else 0.0,
        }
        features.append(item)
        labels.append(row["oracle_winner"])
    from kddcup2017_task2.features import Vectorizer

    vectorizer = Vectorizer()
    return vectorizer.fit_transform(features), np.asarray(labels), vectorizer.names


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    rows = phase1_candidate_frame(data_dir, output_dir, force_cache)
    model_cols = [key for key in rows[0] if key.startswith("candidate_")]
    detail = []
    actual = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    ensemble = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    candidate_matrix = np.asarray([[float(row[col]) for col in model_cols] for row in rows], dtype=float)
    winner_idx = np.argmin(np.abs(candidate_matrix - actual[:, None]), axis=1)
    oracle_pred = candidate_matrix[np.arange(len(rows)), winner_idx]
    for idx, row in enumerate(rows):
        winner = model_cols[int(winner_idx[idx])].removeprefix("candidate_")
        regret = abs(float(ensemble[idx] - actual[idx])) - abs(float(oracle_pred[idx] - actual[idx]))
        detail.append({**row, "oracle_prediction": f"{oracle_pred[idx]:.6f}", "oracle_winner": winner, "ensemble_regret_abs": f"{regret:.6f}"})
    x, y, feature_names = winner_features(detail)
    accuracy = 0.0
    importances = []
    if len(set(y)) > 1:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        min_class = min(np.bincount(np.unique(y, return_inverse=True)[1]))
        folds = min(5, int(min_class))
        if folds >= 2:
            clf = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=8, random_state=13, n_jobs=-1)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=13)
            accuracy = float(np.mean(cross_val_score(clf, x, y, cv=cv, scoring="accuracy")))
            clf.fit(x, y)
            importances = [
                {"feature": name, "importance": f"{float(value):.8f}"}
                for name, value in sorted(zip(feature_names, clf.feature_importances_), key=lambda item: float(item[1]), reverse=True)
            ][:25]
    ensemble_mape = mape_value(actual, ensemble)
    oracle_mape = mape_value(actual, oracle_pred)
    regret = ensemble_mape - oracle_mape
    counts = {}
    for item in detail:
        counts[item["oracle_winner"]] = counts.get(item["oracle_winner"], 0) + 1
    summary = [
        {
            "ensemble_mape": f"{ensemble_mape:.6f}",
            "oracle_mape": f"{oracle_mape:.6f}",
            "oracle_regret": f"{regret:.6f}",
            "winner_accuracy_cv": f"{accuracy:.6f}",
            "winner_counts": "; ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        }
    ]
    out_dir = explain_dir(output_dir)
    detail_csv = out_dir / "oracle_ensemble_gap_rows.csv"
    summary_csv = out_dir / "oracle_ensemble_gap_summary.csv"
    importance_csv = out_dir / "oracle_ensemble_gap_winner_features.csv"
    chart = out_dir / "oracle_ensemble_gap_winner_counts.svg"
    write_csv(detail_csv, detail)
    write_csv(summary_csv, summary)
    write_csv(importance_csv, importances)
    write_metric_chart(chart, [{"label": key, "count": value} for key, value in sorted(counts.items())], "label", "count", "Oracle winner counts")
    card = ExplanationCard(
        name="explain_oracle_ensemble_gap",
        hypothesis="当前 ensemble 强但仍有 oracle gap；如果 oracle winner 可由 green strength、combo/hour、disagreement、ETC、route 等信号预测，就存在可解释 reweighting 空间。",
        method="计算每行候选 oracle MAPE、ensemble regret，并用合法上下文信号做 oracle winner 交叉验证分类诊断。",
        expected_falsification="若 oracle gap 很小或 winner 不可预测，继续做复杂 gating 的收益有限。",
        metrics={"ensemble_mape": f"{ensemble_mape:.6f}", "oracle_mape": f"{oracle_mape:.6f}", "winner_accuracy_cv": f"{accuracy:.6f}"},
        key_result=f"ensemble MAPE={ensemble_mape:.6f}，candidate oracle MAPE={oracle_mape:.6f}，oracle regret={regret:.6f}。",
        interpretation="oracle gap 量化了模型多样性的上界；winner 可预测性决定 neural prior gate 或规则 reweighting 是否有解释基础。",
        next_step="保留并扩展。只允许用 train1-only rolling 训练 winner/gate，phase1 只能作为最终诊断观察。",
        artifacts=(str(detail_csv), str(summary_csv), str(importance_csv), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Oracle ensemble gap analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
