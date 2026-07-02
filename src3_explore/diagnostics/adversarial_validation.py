from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import infer_dates, make_target_rows
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features

from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
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
    rows = [
        {
            "comparison": "train1_early_vs_train1_late",
            "auc": f"{domain_auc(early_features, late_features):.6f}",
            "left_rows": len(early_features),
            "right_rows": len(late_features),
        },
        {
            "comparison": "train1_targets_vs_phase1_visible",
            "auc": f"{domain_auc(train_features, phase1_features):.6f}",
            "left_rows": len(train_features),
            "right_rows": len(phase1_features),
        },
        {
            "comparison": "phase1_visible_vs_phase2_visible",
            "auc": f"{domain_auc(phase1_features, phase2_features):.6f}",
            "left_rows": len(phase1_features),
            "right_rows": len(phase2_features),
        },
    ]
    csv_path = output_dir / "diagnostics" / "adversarial_validation.csv"
    write_csv(csv_path, rows)
    max_auc = max(float(row["auc"]) for row in rows)
    card = ExperimentCard(
        name="adversarial_validation",
        hypothesis="Domain classifiers should detect whether phase1/phase2 visible rows differ from train1.",
        data_visibility=(
            "Uses only feature rows generated through visibility contexts. No target labels are required for domain "
            "classification, and phase2 red labels are unavailable."
        ),
        prototype="RandomForest domain classifier with cross-validated AUC for train/phase splits.",
        metrics={"max_auc": f"{max_auc:.6f}", "comparisons": len(rows)},
        result=f"Wrote adversarial validation summary to {csv_path}.",
        insight="AUC far above 0.5 means validation error should be read by regime, not only as a pooled MAPE.",
        next_step="Join high-AUC feature importances with day/regime clustering to identify the shifted dimensions.",
        artifacts=(str(csv_path),),
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

