from __future__ import annotations

from datetime import time, timedelta
from html import escape
from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import TargetRow, block_name, combine_date_time, make_target_rows, project_paths, target_volume
from src3_explore.common.metrics import safe_corr
from src3_explore.common.reporting import write_csv
from src3_explore.common.trajectory import INTERSECTIONS, TrajectoryStats, read_trajectory_aggregate, route_count_at_lag
from src3_explore.common.visibility import load_train1_latest_fold_context
from src3_explore.explain.common import ExplanationCard, combo_name, combo_slot_anchor, explain_dir, write_explanation_card


LAGS = tuple(range(20, 241, 20))


def block_bounds(row: TargetRow) -> tuple:
    day = row.start.date()
    if block_name(row.start) == "morning":
        return combine_date_time(day, time(8, 0)), combine_date_time(day, time(10, 0))
    return combine_date_time(day, time(17, 0)), combine_date_time(day, time(19, 0))


def include_source(row: TargetRow, source_time, mode: str) -> bool:
    start, end = block_bounds(row)
    if mode == "strictly_legal_before_block":
        return source_time < start
    if mode == "within_red_window_illegal":
        return start <= source_time < end
    if mode == "raw_all_lags":
        return source_time < row.start
    raise ValueError(f"unknown route visibility mode: {mode}")


def red_window_total(row: TargetRow, aggregate, intersection: str) -> float:
    start, end = block_bounds(row)
    current = start
    total = 0.0
    while current < end:
        total += float(aggregate.get((current, intersection, row.tollgate_id), TrajectoryStats()).count)
        current += timedelta(minutes=20)
    return total


def route_feature_matrix(rows: Sequence[TargetRow], aggregate, mode: str) -> tuple[np.ndarray, list[str]]:
    features = []
    names = []
    for row in rows:
        values = []
        row_names = []
        for intersection in INTERSECTIONS:
            for lag in LAGS:
                source_time = row.start - timedelta(minutes=lag)
                value = route_count_at_lag(row, aggregate, intersection, lag) if include_source(row, source_time, mode) else 0.0
                values.append(value)
                row_names.append(f"{intersection}_lag{lag}")
            if mode == "within_red_window_illegal":
                values.append(red_window_total(row, aggregate, intersection))
                row_names.append(f"{intersection}_red_window_total_illegal")
        features.append(values)
        names = row_names
    return np.asarray(features, dtype=float), names


def anchor_for_rows(context, rows: Sequence[TargetRow]) -> np.ndarray:
    anchors = combo_slot_anchor(context)
    return np.asarray(
        [anchors.get((combo_name(row.combo), f"{row.start.hour:02d}:{row.start.minute:02d}"), 0.0) for row in rows],
        dtype=float,
    )


def fit_residual_r2(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> tuple[float, np.ndarray]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    denom = float(np.sum((y_eval - np.mean(y_eval)) ** 2))
    r2 = 0.0 if denom <= 0 else float(1.0 - np.sum((y_eval - pred) ** 2) / denom)
    return r2, pred


def write_lag_heatmap(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = ["strictly_legal_before_block", "within_red_window_illegal", "raw_all_lags"]
    lags = [str(lag) for lag in LAGS]
    lookup = {(str(row["mode"]), str(row["lag_minutes"])): float(row["correlation"]) for row in rows if row.get("row_type") == "lag"}
    width = 980
    cell_w = 66
    cell_h = 46
    left = 230
    top = 72
    values = [abs(v) for v in lookup.values()]
    vmax = max(values) if values else 1.0
    height = top + cell_h * len(modes) + 68
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Segoe UI, Arial" font-size="18" font-weight="600">Route lag correlation by visibility mode</text>',
    ]
    for j, lag in enumerate(lags):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="56" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11">{lag}</text>')
    for i, mode in enumerate(modes):
        y = top + i * cell_h + 27
        parts.append(f'<text x="24" y="{y}" font-family="Segoe UI, Arial" font-size="12">{escape(mode)}</text>')
        for j, lag in enumerate(lags):
            value = lookup.get((mode, lag), 0.0)
            intensity = min(1.0, abs(value) / vmax)
            if value >= 0:
                color = f"#{int(220 - 80 * intensity):02x}{int(238 - 80 * intensity):02x}ff"
            else:
                color = f"#ff{int(230 - 110 * intensity):02x}{int(230 - 130 * intensity):02x}"
            x = left + j * cell_w
            y0 = top + i * cell_h
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w - 5}" height="{cell_h - 6}" fill="{color}" stroke="#d1d5db"/>')
            parts.append(
                f'<text x="{x + (cell_w - 5) / 2:.1f}" y="{y0 + 26}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial" font-size="10">{value:+.2f}</text>'
            )
    parts.append('<text x="24" y="{0}" font-family="Segoe UI, Arial" font-size="12" fill="#374151">Strict legal means source_time is before the target block start.</text>'.format(height - 24))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def group_corr_rows(mode: str, eval_rows: Sequence[TargetRow], signal: np.ndarray, residual: np.ndarray) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dimension, values in (
        ("combo", [combo_name(row.combo) for row in eval_rows]),
        ("hour", [f"{row.start.hour:02d}" for row in eval_rows]),
        ("block", [block_name(row.start) for row in eval_rows]),
    ):
        for value in sorted(set(values)):
            mask = np.asarray([item == value for item in values], dtype=bool)
            output.append(
                {
                    "row_type": "group",
                    "mode": mode,
                    "dimension": dimension,
                    "value": value,
                    "rows": int(np.sum(mask)),
                    "correlation": f"{safe_corr(signal[mask], residual[mask]):.6f}",
                }
            )
    return output


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    del force_cache
    context = load_train1_latest_fold_context(data_dir)
    paths = project_paths(data_dir)
    trajectory = read_trajectory_aggregate([paths["train1_volume"].parents[0] / "trajectories(table 5)_training.csv"])
    train_rows = make_target_rows(context.train_days, context.combos)
    eval_rows = list(context.rows)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in eval_rows], dtype=float)
    train_residual = y_train - anchor_for_rows(context, train_rows)
    eval_residual = y_eval - anchor_for_rows(context, eval_rows)

    rows_out: list[dict[str, object]] = []
    for mode in ("strictly_legal_before_block", "within_red_window_illegal", "raw_all_lags"):
        x_train, names = route_feature_matrix(train_rows, trajectory, mode)
        x_eval, _ = route_feature_matrix(eval_rows, trajectory, mode)
        signal = np.sum(x_eval, axis=1)
        r2, residual_pred = fit_residual_r2(x_train, train_residual, x_eval, eval_residual)
        rows_out.append(
            {
                "row_type": "summary",
                "mode": mode,
                "correlation": f"{safe_corr(signal, eval_residual):.6f}",
                "linear_residual_r2": f"{r2:.6f}",
                "predicted_residual_correlation": f"{safe_corr(residual_pred, eval_residual):.6f}",
                "mean_signal": f"{float(np.mean(signal)):.6f}",
                "features": len(names),
            }
        )
        rows_out.extend(group_corr_rows(mode, eval_rows, signal, eval_residual))
        for lag in LAGS:
            idx = [i for i, name in enumerate(names) if name.endswith(f"lag{lag}")]
            lag_signal = np.sum(x_eval[:, idx], axis=1) if idx else np.zeros(len(eval_rows))
            rows_out.append(
                {
                    "row_type": "lag",
                    "mode": mode,
                    "lag_minutes": lag,
                    "correlation": f"{safe_corr(lag_signal, eval_residual):.6f}",
                    "mean_signal": f"{float(np.mean(lag_signal)):.6f}",
                }
            )

    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "route_legal_decomposition.csv"
    heatmap = out_dir / "route_lag_corr_heatmap.svg"
    write_csv(csv_path, rows_out)
    write_lag_heatmap(heatmap, rows_out)
    summary = {row["mode"]: row for row in rows_out if row["row_type"] == "summary"}
    strict_corr = float(summary["strictly_legal_before_block"]["correlation"])
    illegal_corr = float(summary["within_red_window_illegal"]["correlation"])
    strict_r2 = float(summary["strictly_legal_before_block"]["linear_residual_r2"])
    card = ExplanationCard(
        name="explain_route_legal_decomposition",
        hypothesis="route lead-lag 的解释力如果来自预测前合法 source_time，process graph 有研究价值；如果主要来自红窗内部 source_time，则只能作为机制解释，不能进预测。",
        method="在 train1 最新 rolling fold 中把 trajectory route signal 分成 strictly legal before block、within-red-window illegal、raw all lags，比较 residual correlation、线性 residual R2、分组相关和 by-lag 相关。",
        data_visibility="主结论使用 train1 rolling fold；strict legal 特征只允许 source_time 早于 08:00/17:00 block start。red-window internal signal 明确标为 illegal diagnostic。",
        expected_falsification="若 strict legal signal 相关性和 R2 近零，而 illegal signal 明显更强，则 route 信息主要解释同步机制，不能作为合法预测增量。",
        metrics={
            "strict_legal_corr": f"{strict_corr:.6f}",
            "strict_legal_r2": f"{strict_r2:.6f}",
            "illegal_corr": f"{illegal_corr:.6f}",
        },
        key_result=f"strict legal signal residual corr={strict_corr:.6f}, R2={strict_r2:.6f}; red-window illegal corr={illegal_corr:.6f}。",
        interpretation="strict legal 若仍有相关性，说明不是所有图思想都失败，失败的是五节点 label graph；illegal 更强则说明很多 route 解释来自目标窗同步到达机制，不能直接用来冲分。",
        next_step="保留 route/process graph 方向，但后续预测实验必须只用 strict legal lag，并用 train1 rolling 选择是否进入受限融合。",
        artifacts=(str(csv_path), str(heatmap)),
        explain_card_filename="route_legal_card.md",
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Route signal legal/illegal decomposition")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir, args.force_cache).to_markdown())


if __name__ == "__main__":
    main()
