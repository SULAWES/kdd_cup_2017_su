from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, block_name, combine_date_time, make_target_rows, project_paths, target_volume
from src3_explore.common.metrics import mape_value
from src3_explore.common.reporting import write_csv
from src3_explore.common.trajectory import INTERSECTIONS, read_trajectory_aggregate, route_count_at_lag
from src3_explore.common.visibility import load_train1_latest_fold_context
from src3_explore.explain.common import ExplanationCard, combo_name, combo_slot_anchor, explain_dir, write_explanation_card, write_metric_chart


def anchor_for_rows(context, rows):
    anchor = combo_slot_anchor(context)
    values = []
    for row in rows:
        values.append(anchor.get((combo_name(row.combo), f"{row.start.hour:02d}:{row.start.minute:02d}"), 0.0))
    return np.asarray(values, dtype=float)


def static_graph_features(context, rows):
    combos = [combo_name(combo) for combo in context.combos]
    parsed = {combo_name(combo): combo for combo in context.combos}
    matrix = []
    for row in rows:
        own = combo_name(row.combo)
        values = []
        for other_name in combos:
            other = parsed[other_name]
            is_neighbor = other != row.combo and (other[0] == row.tollgate_id or other[1] == row.direction)
            total = 0.0
            if is_neighbor:
                for clock in OBS_TIMES[block_name(row.start)]:
                    total += context.known_agg.get((combine_date_time(row.start.date(), clock), other[0], other[1]), 0)
            values.append(float(total))
        values.append(float(own == "1_0"))
        values.append(float(row.start.hour))
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def route_features(rows, trajectory_agg):
    matrix = []
    for row in rows:
        values = []
        for intersection in INTERSECTIONS:
            for lag in (20, 40, 60, 80, 100, 120):
                values.append(route_count_at_lag(row, trajectory_agg, intersection, lag))
        values.append(float(row.start.hour))
        values.append(float(row.start.weekday()))
        return_values = values
        matrix.append(return_values)
    return np.asarray(matrix, dtype=float)


def fit_residual_model(x_train, residual_train, x_eval):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_train, residual_train)
    return model.predict(x_eval)


def run(data_dir: Path, output_dir: Path, force_cache: bool = False) -> ExplanationCard:
    del force_cache
    context = load_train1_latest_fold_context(data_dir)
    paths = project_paths(data_dir)
    trajectory = read_trajectory_aggregate([paths["train1_volume"].parents[0] / "trajectories(table 5)_training.csv"])
    train_rows = make_target_rows(context.train_days, context.combos)
    eval_rows = list(context.rows)
    y_train = np.asarray([target_volume(context.train_agg, row) for row in train_rows], dtype=float)
    y_eval = np.asarray([target_volume(context.label_agg, row) for row in eval_rows], dtype=float)
    train_anchor = anchor_for_rows(context, train_rows)
    eval_anchor = anchor_for_rows(context, eval_rows)
    residual_train = y_train - train_anchor

    experiments = []
    for name, train_x, eval_x in (
        ("five_node_static_green", static_graph_features(context, train_rows), static_graph_features(context, eval_rows)),
        ("route_arrival_kernel", route_features(train_rows, trajectory), route_features(eval_rows, trajectory)),
    ):
        residual_pred = fit_residual_model(train_x, residual_train, eval_x)
        pred = np.maximum(eval_anchor + residual_pred, 0.0)
        base_mape = mape_value(y_eval, eval_anchor)
        mape = mape_value(y_eval, pred)
        residual_corr = 0.0 if np.std(residual_pred) == 0 or np.std(y_eval - eval_anchor) == 0 else float(np.corrcoef(residual_pred, y_eval - eval_anchor)[0, 1])
        experiments.append(
            {
                "method": name,
                "base_anchor_mape": f"{base_mape:.6f}",
                "residual_adjusted_mape": f"{mape:.6f}",
                "mape_delta_vs_anchor": f"{mape - base_mape:.6f}",
                "residual_corr": f"{residual_corr:.6f}",
            }
        )

    out_dir = explain_dir(output_dir)
    csv_path = out_dir / "route_graph_replacement.csv"
    chart = out_dir / "route_graph_replacement_mape.svg"
    write_csv(csv_path, experiments)
    write_metric_chart(chart, experiments, "method", "residual_adjusted_mape", "Residual explanation MAPE")
    route = next(row for row in experiments if row["method"] == "route_arrival_kernel")
    static = next(row for row in experiments if row["method"] == "five_node_static_green")
    card = ExplanationCard(
        name="explain_route_graph_replacement",
        hypothesis="如果五节点图定义错误，route/intersection/tollgate lead-lag 图应比静态五节点邻居更能解释 residual。",
        method="在 train1 最新 fold 上比较五节点静态邻居 green 特征与 route arrival kernel lag 特征的 residual correction。",
        expected_falsification="若 route kernel 不能优于静态邻居特征，说明轨迹 lead-lag 在当前可见性下也不足以解释 residual。",
        metrics={
            "static_adjusted_mape": static["residual_adjusted_mape"],
            "route_adjusted_mape": route["residual_adjusted_mape"],
            "route_residual_corr": route["residual_corr"],
        },
        key_result=f"静态五节点 residual MAPE={static['residual_adjusted_mape']}，route kernel residual MAPE={route['residual_adjusted_mape']}。",
        interpretation="route kernel 如果更好，说明失败点在五节点图语义，而不是所有图方法；如果仍弱，则轨迹信号更适合受限融合或解释而非主模型。",
        next_step="保留 route kernel 作为可解释候选；继续前必须通过 train1 rolling 且特征只使用目标前可见 lag。",
        artifacts=(str(csv_path), str(chart)),
    )
    write_explanation_card(output_dir, card)
    return card


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Replace static five-node graph with route lead-lag graph")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
