from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import infer_dates, make_target_rows
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features

from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg
from src3_explore.common.visibility import load_phase1_context, load_phase2_visible_rows


def feature_matrix(context, rows):
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    builder.fit_stats(make_target_rows(context.train_days, context.combos))
    features = builder.transform(rows, context.known_agg, context.known_attr_agg)
    return filter_features(features, DEFAULT_DROP_FEATURES)


def domain_auc(left_features, right_features) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    vectorizer = Vectorizer()
    x = vectorizer.fit_transform(left_features + right_features)
    y = np.asarray([0] * len(left_features) + [1] * len(right_features), dtype=int)
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        return 0.5
    clf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=8, random_state=13, n_jobs=-1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=13)
    prob = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def domain_diagnostics(comparison: str, left_features, right_features) -> tuple[dict[str, object], list[dict[str, object]]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    vectorizer = Vectorizer()
    x = vectorizer.fit_transform(left_features + right_features)
    y = np.asarray([0] * len(left_features) + [1] * len(right_features), dtype=int)
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        auc = 0.5
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=8, random_state=13, n_jobs=-1)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=13)
        prob = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, prob))
    fitted = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=8, random_state=13, n_jobs=-1)
    fitted.fit(x, y)
    importances = [
        {
            "comparison": comparison,
            "feature": name,
            "importance": f"{float(value):.8f}",
        }
        for name, value in zip(vectorizer.names, fitted.feature_importances_)
        if float(value) > 0.0
    ]
    importances.sort(key=lambda row: float(row["importance"]), reverse=True)
    return (
        {
            "comparison": comparison,
            "auc": f"{auc:.6f}",
            "left_rows": len(left_features),
            "right_rows": len(right_features),
            "top_feature": importances[0]["feature"] if importances else "",
        },
        importances[:30],
    )


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    phase1 = load_phase1_context(data_dir)
    phase2 = load_phase2_visible_rows(data_dir)
    train_days = list(phase1.train_days)
    mid = len(train_days) // 2
    early_rows = make_target_rows(train_days[:mid], phase1.combos)
    late_rows = make_target_rows(train_days[mid:], phase1.combos)
    phase1_train_rows = make_target_rows(phase1.train_days, phase1.combos)
    phase1_features = feature_matrix(phase1, phase1.rows)
    phase2_features = feature_matrix(phase2, phase2.rows)
    train_features = feature_matrix(phase1, phase1_train_rows)
    early_features = feature_matrix(phase1, early_rows)
    late_features = feature_matrix(phase1, late_rows)
    comparisons = (
        ("train1_early_vs_train1_late", early_features, late_features),
        ("train1_targets_vs_phase1_visible", train_features, phase1_features),
        ("phase1_visible_vs_phase2_visible", phase1_features, phase2_features),
    )
    rows = []
    importance_rows = []
    for comparison, left, right in comparisons:
        row, importances = domain_diagnostics(comparison, left, right)
        rows.append(row)
        importance_rows.extend(importances)
    csv_path = output_dir / "diagnostics" / "adversarial_validation.csv"
    importances_csv = output_dir / "diagnostics" / "adversarial_validation_feature_importance.csv"
    chart = output_dir / "diagnostics" / "adversarial_validation_auc.svg"
    write_csv(csv_path, rows)
    write_csv(importances_csv, importance_rows)
    write_bar_svg(chart, rows, "comparison", "auc", "Adversarial validation AUC")
    max_auc = max(float(row["auc"]) for row in rows)
    card = ExperimentCard(
        name="adversarial_validation",
        hypothesis="如果 train1、phase1 可见输入、phase2 可见输入之间存在分布偏移，域分类器应能把这些时期区分开。",
        data_visibility=(
            "只使用 visibility context 生成的特征行；域分类不需要目标标签，也不会读取 phase2 红窗标签。"
        ),
        prototype="用 RandomForest 做 train/phase split 域分类，交叉验证输出 AUC，并保存特征重要性。",
        metrics={"max_auc": f"{max_auc:.6f}", "comparisons": len(rows)},
        result=f"最大域分类 AUC={max_auc:.6f}，说明时期之间高度可分；当前 top feature 主要受绝对日期影响，不能直接解释为因果 traffic shift。",
        insight="即使不用于建模，也能提醒 pooled MAPE 会掩盖时期偏移；特征重要性可定位偏移来自日期、绿窗强度、车辆结构还是历史统计。",
        next_step="扩展。下一版应增加去除 day_of_month 的 adversarial validation，以隔离真实输入分布偏移。",
        artifacts=(str(csv_path), str(importances_csv), str(chart)),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial validation across Task 2 periods")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()
