from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kddcup2017_task2.data import OBS_TIMES, TARGET_TIMES, TargetRow, block_name, combine_date_time, make_target_rows, target_volume
from kddcup2017_task2.features import FeatureBuilder, Vectorizer
from kddcup2017_task2.pipeline import DEFAULT_DROP_FEATURES, filter_features

from src3_explore.common.candidate_cache import ensure_phase1_candidate_cache, load_candidate_rows
from src3_explore.common.metrics import mape_value, summarize_errors
from src3_explore.common.reporting import safe_slug, write_csv
from src3_explore.common.svg import write_bar_svg
from src3_explore.common.visibility import VisibilityContext, load_phase1_context


COMBO_ORDER = (("1", "0"), ("1", "1"), ("2", "0"), ("3", "0"), ("3", "1"))


@dataclass(frozen=True)
class ExplanationCard:
    name: str
    hypothesis: str
    method: str
    expected_falsification: str
    metrics: Mapping[str, object]
    key_result: str
    interpretation: str
    next_step: str
    artifacts: Sequence[str] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        metric_lines = "\n".join(f"- `{key}`: {value}" for key, value in self.metrics.items()) or "- 无"
        artifact_lines = "\n".join(f"- `{item}`" for item in self.artifacts) or "- 无"
        return "\n".join(
            [
                f"## {self.name}",
                "",
                f"**假设（Hypothesis）:** {self.hypothesis}",
                "",
                f"**方法（Method）:** {self.method}",
                "",
                f"**可证伪预期（Expected falsification）:** {self.expected_falsification}",
                "",
                "**指标（Metrics）:**",
                metric_lines,
                "",
                f"**关键结果（Key result）:** {self.key_result}",
                "",
                f"**解释（Interpretation）:** {self.interpretation}",
                "",
                f"**下一步（Next step）:** {self.next_step}",
                "",
                "**产物（Artifacts）:**",
                artifact_lines,
                "",
            ]
        )


def write_explanation_card(output_dir: Path, card: ExplanationCard) -> Path:
    card_dir = output_dir / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    path = card_dir / f"{safe_slug(card.name)}.md"
    path.write_text(card.to_markdown(), encoding="utf-8")
    return path


def explain_dir(output_dir: Path) -> Path:
    return output_dir / "explain"


def combo_name(combo: tuple[str, str]) -> str:
    return f"{combo[0]}_{combo[1]}"


def combo_tuple(name: str) -> tuple[str, str]:
    tollgate, direction = name.split("_", 1)
    return tollgate, direction


def phase1_candidate_frame(data_dir: Path, output_dir: Path, force_cache: bool = False) -> list[dict[str, str]]:
    cache = ensure_phase1_candidate_cache(data_dir, output_dir, force=force_cache)
    return load_candidate_rows(cache)


def prediction_column(rows: Sequence[Mapping[str, str]], preferred: str = "prediction") -> np.ndarray:
    return np.asarray([float(row[preferred]) for row in rows], dtype=float)


def actual_column(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    return np.asarray([float(row["actual"]) for row in rows], dtype=float)


def row_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return str(row["date"]), str(row["slot"]), str(row["combo"])


def pivot_candidate_rows(rows: Sequence[Mapping[str, str]], value_field: str) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    times = sorted({(str(row["date"]), str(row["slot"])) for row in rows})
    combos = sorted({str(row["combo"]) for row in rows})
    time_index = {key: idx for idx, key in enumerate(times)}
    combo_index = {key: idx for idx, key in enumerate(combos)}
    matrix = np.zeros((len(times), len(combos)), dtype=float)
    for row in rows:
        matrix[time_index[(str(row["date"]), str(row["slot"]))], combo_index[str(row["combo"])]] = float(row[value_field])
    return matrix, times, combos


def ordered_candidate_rows(
    rows: Sequence[Mapping[str, str]],
    times: Sequence[tuple[str, str]],
    combos: Sequence[str],
) -> list[Mapping[str, str]]:
    lookup = {(str(row["date"]), str(row["slot"]), str(row["combo"])): row for row in rows}
    return [lookup[(date, slot, combo)] for date, slot in times for combo in combos]


def train_target_matrix(context: VisibilityContext) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    combos = [combo_name(combo) for combo in context.combos]
    times = [(str(day), f"{clock.hour:02d}:{clock.minute:02d}") for day in context.train_days for clock in TARGET_TIMES]
    matrix = np.zeros((len(times), len(combos)), dtype=float)
    combo_index = {combo_name(combo): idx for idx, combo in enumerate(context.combos)}
    row_idx = 0
    for day in context.train_days:
        for clock in TARGET_TIMES:
            start = combine_date_time(day, clock)
            for combo in context.combos:
                matrix[row_idx, combo_index[combo_name(combo)]] = float(
                    context.train_agg.get((start, combo[0], combo[1]), 0)
                )
            row_idx += 1
    return matrix, times, combos


def combo_slot_anchor(context: VisibilityContext) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = {}
    for row in make_target_rows(context.train_days, context.combos):
        key = (combo_name(row.combo), f"{row.start.hour:02d}:{row.start.minute:02d}")
        values.setdefault(key, []).append(float(target_volume(context.train_agg, row)))
    return {key: float(np.median(items)) if items else 0.0 for key, items in values.items()}


def anchor_matrix(context: VisibilityContext, times: Sequence[tuple[str, str]], combos: Sequence[str]) -> np.ndarray:
    anchor = combo_slot_anchor(context)
    matrix = np.zeros((len(times), len(combos)), dtype=float)
    for i, (_, slot) in enumerate(times):
        for j, combo in enumerate(combos):
            matrix[i, j] = anchor.get((combo, slot), 0.0)
    return matrix


def pairwise_corr_rows(matrix: np.ndarray, combos: Sequence[str], label: str, lag: int = 0) -> list[dict[str, object]]:
    rows = []
    for i, left in enumerate(combos):
        for j, right in enumerate(combos):
            if j <= i:
                continue
            x = matrix[:-lag, i] if lag > 0 else matrix[:, i]
            y = matrix[lag:, j] if lag > 0 else matrix[:, j]
            corr = 0.0 if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
            rows.append({"matrix": label, "lag": lag, "left": left, "right": right, "corr": f"{corr:.6f}"})
    return rows


def adjacency_matrix(mode: str, combos: Sequence[str], signal: np.ndarray | None = None, seed: int = 13) -> np.ndarray:
    n = len(combos)
    adjacency = np.eye(n, dtype=float)
    if mode == "identity":
        return adjacency
    if mode == "full":
        return np.ones((n, n), dtype=float) / n
    if mode == "topology":
        adjacency = np.eye(n, dtype=float)
        parsed = [combo_tuple(combo) for combo in combos]
        for i, left in enumerate(parsed):
            for j, right in enumerate(parsed):
                if i == j:
                    continue
                if left[0] == right[0] or left[1] == right[1]:
                    adjacency[i, j] = 1.0
        return row_normalize(adjacency)
    if mode == "corr":
        if signal is None:
            raise ValueError("corr adjacency requires a signal matrix")
        corr = np.corrcoef(signal.T)
        corr = np.nan_to_num(corr, nan=0.0)
        adjacency = np.maximum(corr, 0.0)
        np.fill_diagonal(adjacency, 1.0)
        return row_normalize(adjacency)
    if mode == "random":
        rng = np.random.default_rng(seed)
        adjacency = rng.random((n, n))
        adjacency = (adjacency + adjacency.T) / 2.0
        np.fill_diagonal(adjacency, 1.0)
        return row_normalize(adjacency)
    raise ValueError(f"unknown adjacency mode: {mode}")


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    denom = arr.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return arr / denom


def message_pass(values: np.ndarray, adjacency: np.ndarray, layers: int, self_weight: float = 0.65) -> np.ndarray:
    h = np.asarray(values, dtype=float)
    for _ in range(layers):
        h = self_weight * h + (1.0 - self_weight) * (h @ adjacency.T)
    return h


def mean_pairwise_distance(matrix: np.ndarray) -> float:
    if matrix.shape[1] < 2:
        return 0.0
    distances = []
    for row in matrix:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                distances.append(abs(float(row[i] - row[j])))
    return float(np.mean(distances)) if distances else 0.0


def laplacian_energy(signal: np.ndarray, adjacency: np.ndarray) -> float:
    weights = np.asarray(adjacency, dtype=float)
    degrees = np.diag(weights.sum(axis=1))
    laplacian = degrees - weights
    values = np.asarray(signal, dtype=float)
    numerator = np.trace(values @ laplacian @ values.T)
    denominator = float(np.sum(values * values)) + 1e-9
    return float(numerator / denominator)


def raw_sequence_features(
    rows: Sequence[TargetRow],
    known_agg: Mapping,
    combos: Sequence[tuple[str, str]],
    hist_days: int = 7,
    hist_order: Sequence[int] | None = None,
    obs_order: Sequence[int] | None = None,
    summary_only: bool = False,
) -> tuple[np.ndarray, list[str]]:
    combo_index = {combo: idx for idx, combo in enumerate(combos)}
    names: list[str] = []
    records = []
    for row in rows:
        hist = []
        for days_back in range(hist_days, 0, -1):
            start = row.start - timedelta(days=days_back)
            hist.append(float(known_agg.get((start, row.tollgate_id, row.direction), 0)))
        obs = [
            float(known_agg.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0))
            for clock in OBS_TIMES[block_name(row.start)]
        ]
        if hist_order is not None:
            hist = [hist[idx] for idx in hist_order]
        if obs_order is not None:
            obs = [obs[idx] for idx in obs_order]
        if summary_only:
            numeric = [
                float(np.mean(hist)) if hist else 0.0,
                float(np.median(hist)) if hist else 0.0,
                float(hist[-1]) if hist else 0.0,
                float(np.mean(obs)) if obs else 0.0,
                float(np.sum(obs)),
                float(obs[-1] - obs[0]) if obs else 0.0,
            ]
            row_names = ["hist_mean", "hist_median", "hist_last", "obs_mean", "obs_sum", "obs_trend"]
        else:
            numeric = hist + obs
            row_names = [f"hist_{idx}" for idx in range(len(hist))] + [f"obs_{idx}" for idx in range(len(obs))]
        extras = [
            float(combo_index[row.combo]),
            float(row.start.hour),
            float(row.start.minute),
            float(row.start.weekday()),
            1.0 if row.start.weekday() >= 5 else 0.0,
        ]
        records.append(numeric + extras)
        names = row_names + ["combo_index", "hour", "minute", "weekday", "is_weekend"]
    return np.asarray(records, dtype=float), names


def engineered_features(context: VisibilityContext, rows: Sequence[TargetRow], known_agg: Mapping, known_attr_agg: Mapping):
    builder = FeatureBuilder(context.train_agg, context.weather, include_weather=False)
    train_rows = make_target_rows(context.train_days, context.combos)
    builder.fit_stats(train_rows)
    features = filter_features(builder.transform(rows, known_agg, known_attr_agg), DEFAULT_DROP_FEATURES)
    vectorizer = Vectorizer()
    if rows == train_rows:
        return vectorizer.fit_transform(features), vectorizer
    train_features = filter_features(builder.transform(train_rows, context.train_agg, context.train_attr_agg), DEFAULT_DROP_FEATURES)
    vectorizer.fit_transform(train_features)
    return vectorizer.transform(features), vectorizer


def evaluate_detail_rows(rows: Sequence[Mapping[str, str]], prediction: Sequence[float], method: str) -> list[dict[str, object]]:
    output = []
    for row, pred in zip(rows, prediction):
        actual = float(row["actual"])
        value = float(pred)
        output.append(
            {
                "method": method,
                "date": row["date"],
                "combo": row["combo"],
                "hour": row["hour"],
                "slot": row["slot"],
                "block": row["block"],
                "actual": f"{actual:.6f}",
                "prediction": f"{value:.6f}",
                "signed_error": f"{value - actual:.6f}",
                "abs_pct_error": f"{abs(value - actual) / max(abs(actual), 1.0):.6f}",
            }
        )
    return output


def grouped_error_rows(detail_rows: Sequence[Mapping[str, object]], groupings: Sequence[Sequence[str]]) -> list[dict[str, object]]:
    grouped = []
    for fields in groupings:
        for item in summarize_errors(detail_rows, fields):
            item["dimension"] = "/".join(fields)
            item["value"] = "/".join(str(item.pop(field)) for field in fields)
            grouped.append(item)
    return grouped


def write_metric_chart(path: Path, rows: Sequence[Mapping[str, object]], label: str, metric: str, title: str, max_items: int = 20) -> Path:
    return write_bar_svg(path, rows, label, metric, title, max_items=max_items)


def score_prediction(actual: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    return {
        "mape": mape_value(actual_arr, pred_arr),
        "signed_error_mean": float(np.mean(pred_arr - actual_arr)) if len(actual_arr) else 0.0,
        "mae": float(np.mean(np.abs(pred_arr - actual_arr))) if len(actual_arr) else 0.0,
    }


def fit_predict_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    elif model_name == "extra":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(n_estimators=240, max_depth=12, min_samples_leaf=8, random_state=13, n_jobs=-1)
    elif model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(48, 24), alpha=0.02, max_iter=700, random_state=13, early_stopping=True),
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(x_train, np.log1p(np.asarray(y_train, dtype=float)))
    return np.maximum(np.expm1(model.predict(x_pred)), 0.0)
