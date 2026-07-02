from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, block_name, combine_date_time, make_target_rows, target_volume

from src3_explore.common.metrics import robust_z_scores
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.visibility import load_phase1_context


def day_total_attr_share(attr_agg, day, attr_name: str, positive_value: str = "1") -> float:
    total = 0.0
    positive = 0.0
    for key, value in attr_agg.items():
        start, _, _, name, attr_value = key
        if start.date() != day or name != attr_name:
            continue
        total += float(value)
        if attr_value == positive_value:
            positive += float(value)
    return positive / total if total > 0 else 0.0


def allocation_ratio(aggregate, day) -> float:
    y1 = 0.0
    y2 = 0.0
    for key, value in aggregate.items():
        start, tollgate, direction = key
        if start.date() != day or direction != "0":
            continue
        if tollgate == "1":
            y1 += float(value)
        elif tollgate == "2":
            y2 += float(value)
    return y2 / max(y1 + y2, 1.0)


def build_day_embeddings(context) -> tuple[np.ndarray, list[dict[str, object]]]:
    rows = []
    vectors = []
    for day in context.train_days:
        target_rows = make_target_rows([day], context.combos)
        combo_means = []
        block_means = []
        for combo in context.combos:
            combo_values = [target_volume(context.train_agg, row) for row in target_rows if row.combo == combo]
            combo_means.append(float(np.mean(combo_values)) if combo_values else 0.0)
        for block, clocks in OBS_TIMES.items():
            values = []
            for combo in context.combos:
                values.append(
                    sum(
                        context.train_agg.get((combine_date_time(day, clock), combo[0], combo[1]), 0)
                        for clock in clocks
                    )
                )
            block_means.append(float(np.mean(values)) if values else 0.0)
        total = float(sum(combo_means))
        etc_share = day_total_attr_share(context.train_attr_agg, day, "etc", "1")
        r2 = allocation_ratio(context.train_agg, day)
        vectors.append(combo_means + block_means + [total, etc_share, r2, float(day.weekday())])
        rows.append(
            {
                "date": str(day),
                "weekday": day.weekday(),
                "is_weekend": day.weekday() >= 5,
                "is_national_day": day.month == 10 and 1 <= day.day <= 7,
                "is_post_holiday": day.month == 10 and 8 <= day.day <= 14,
                "total_target_volume": f"{total:.6f}",
                "ETC_share": f"{etc_share:.6f}",
                "r2_allocation": f"{r2:.6f}",
            }
        )
    return np.asarray(vectors, dtype=float), rows


def regime_labels(rows: list[dict[str, object]]) -> list[str]:
    totals = np.asarray([float(row["total_target_volume"]) for row in rows], dtype=float)
    etc = np.asarray([float(row["ETC_share"]) for row in rows], dtype=float)
    r2 = np.asarray([float(row["r2_allocation"]) for row in rows], dtype=float)
    low_cut = float(np.quantile(totals, 0.2)) if len(totals) else 0.0
    etc_z = robust_z_scores(etc)
    r2_z = robust_z_scores(r2)
    labels = []
    for idx, row in enumerate(rows):
        parts = []
        if row["is_weekend"]:
            parts.append("weekend")
        else:
            parts.append("weekday")
        if row["is_national_day"]:
            parts.append("holiday")
        if row["is_post_holiday"]:
            parts.append("post_holiday")
        if float(row["total_target_volume"]) <= low_cut:
            parts.append("low_volume")
        if abs(float(etc_z[idx])) >= 2.0:
            parts.append("ETC_anomaly")
        if abs(float(r2_z[idx])) >= 2.0:
            parts.append("tollgate_allocation_anomaly")
        labels.append(";".join(parts))
    return labels


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    context = load_phase1_context(data_dir)
    x, rows = build_day_embeddings(context)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    k = min(6, max(1, len(rows)))
    if k > 1:
        x_scaled = StandardScaler().fit_transform(x)
        clusters = KMeans(n_clusters=k, random_state=13, n_init=10).fit_predict(x_scaled)
    else:
        clusters = np.zeros(len(rows), dtype=int)
    labels = regime_labels(rows)
    for row, cluster, label in zip(rows, clusters, labels):
        row["cluster"] = int(cluster)
        row["regime_labels"] = label
    csv_path = output_dir / "representations" / "day_embedding_regimes.csv"
    write_csv(csv_path, rows)
    counts = {}
    for row in rows:
        counts[row["cluster"]] = counts.get(row["cluster"], 0) + 1
    card = ExperimentCard(
        name="day_embedding_clustering",
        hypothesis="Day-level embeddings should separate normal weekdays, weekends, holiday/post-holiday, and anomaly regimes.",
        data_visibility="Uses train1 labels and attributes only; no phase1 labels or phase2 labels are needed.",
        prototype="KMeans over day embeddings with heuristic labels for low volume, ETC, and tollgate allocation anomalies.",
        metrics={"days": len(rows), "clusters": k, "cluster_counts": "; ".join(f"{key}={value}" for key, value in sorted(counts.items()))},
        result=f"Wrote regime table to {csv_path}.",
        insight="Regime labels provide join keys for residual_atlas and uncertainty analysis.",
        next_step="Inspect high-error phase1 rows by nearest train1 regime before adding model complexity.",
        artifacts=(str(csv_path),),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Day embedding clustering")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

