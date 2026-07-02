from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src3_explore.common.candidate_cache import ensure_phase1_candidate_cache, load_candidate_rows
from src3_explore.common.metrics import summarize_errors
from src3_explore.common.reporting import ExperimentCard, write_card, write_csv
from src3_explore.common.svg import write_bar_svg


GROUP_FIELDS = (
    "date",
    "combo",
    "hour",
    "slot",
    "green_obs_strength_bucket",
    "ETC_share_bucket",
    "trajectory_signal_bucket",
    "model_disagreement_bucket",
)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExperimentCard:
    cache = ensure_phase1_candidate_cache(data_dir, output_dir, force=force_cache)
    rows = load_candidate_rows(cache)
    atlas_rows = []
    artifacts = [str(cache)]
    for field in GROUP_FIELDS:
        summary = summarize_errors(rows, [field])
        for item in summary:
            item["dimension"] = field
            item["value"] = item.pop(field)
        atlas_rows.extend(summary)
        chart = output_dir / "diagnostics" / f"residual_atlas_{field}.svg"
        chart_rows = [
            {"label": str(row["value"]), "mape": row["mape"]}
            for row in sorted(summary, key=lambda item: float(item["mape"]), reverse=True)
        ]
        write_bar_svg(chart, chart_rows, "label", "mape", f"Residual atlas MAPE by {field}", max_items=18)
        artifacts.append(str(chart))

    grouped_csv = output_dir / "diagnostics" / "residual_atlas_grouped.csv"
    detail_csv = output_dir / "diagnostics" / "residual_atlas_rows.csv"
    write_csv(grouped_csv, atlas_rows)
    write_csv(detail_csv, rows)
    artifacts.extend([str(grouped_csv), str(detail_csv)])
    worst = sorted(atlas_rows, key=lambda item: float(item["mape"]), reverse=True)[:5]
    worst_text = "; ".join(f"{row['dimension']}={row['value']} mape={float(row['mape']):.4f}" for row in worst)
    card = ExperimentCard(
        name="residual_atlas",
        hypothesis="Residuals should cluster by interpretable traffic regimes rather than scatter uniformly.",
        data_visibility=(
            "Candidate predictions are trained without train2 labels; phase1 labels are joined only after fixed "
            "prediction cache creation to compute grouped residual diagnostics."
        ),
        prototype=(
            "Cache official candidate predictions and summarize signed error/MAPE by date, combo, hour, slot, "
            "green strength, ETC share, trajectory signal, and model disagreement."
        ),
        metrics={"rows": len(rows), "grouped_rows": len(atlas_rows), "worst_groups": worst_text},
        result=f"Wrote residual atlas CSV to {grouped_csv}.",
        insight="Worst groups are the starting point for mechanism-specific analysis, not direct hyperparameter choices.",
        next_step="Review recurring high-MAPE groups against transfer, allocation, ETC, and trajectory mechanism outputs.",
        artifacts=tuple(artifacts),
    )
    write_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build residual atlas for cached candidate predictions")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    card = run(args.data_dir, args.output_dir, args.force_cache)
    print(card.to_markdown())


if __name__ == "__main__":
    main()

