from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def mape_value(actual: Sequence[float], prediction: Sequence[float], eps: float = 1.0) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    denom = np.maximum(np.abs(actual_arr), eps)
    return float(np.mean(np.abs(actual_arr - pred_arr) / denom))


def signed_error(actual: float, prediction: float) -> float:
    return float(prediction) - float(actual)


def pct_error(actual: float, prediction: float, eps: float = 1.0) -> float:
    return abs(float(actual) - float(prediction)) / max(abs(float(actual)), eps)


def summarize_errors(
    rows: Iterable[Mapping[str, object]],
    group_fields: Sequence[str],
    actual_field: str = "actual",
    prediction_field: str = "prediction",
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        actual = np.asarray([float(item[actual_field]) for item in items], dtype=float)
        pred = np.asarray([float(item[prediction_field]) for item in items], dtype=float)
        errors = pred - actual
        summary: dict[str, object] = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "count": len(items),
                "actual_mean": float(actual.mean()) if len(actual) else 0.0,
                "prediction_mean": float(pred.mean()) if len(pred) else 0.0,
                "signed_error_mean": float(errors.mean()) if len(errors) else 0.0,
                "signed_error_median": float(np.median(errors)) if len(errors) else 0.0,
                "mape": mape_value(actual, pred) if len(actual) else 0.0,
                "mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
            }
        )
        output.append(summary)
    return output


def bucket_quantiles(values: Sequence[float], labels: Sequence[str] = ("low", "mid", "high")) -> list[str]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0 or float(np.nanmax(finite)) == float(np.nanmin(finite)):
        return [labels[len(labels) // 2]] * len(arr)
    cuts = np.quantile(finite, np.linspace(0, 1, len(labels) + 1)[1:-1])
    buckets = []
    for value in arr:
        idx = int(np.searchsorted(cuts, value, side="right"))
        buckets.append(labels[min(idx, len(labels) - 1)])
    return buckets


def robust_z_scores(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(arr))
    if scale <= 0 or not math.isfinite(scale):
        return np.zeros_like(arr)
    return (arr - med) / scale


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2 or float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def interval_coverage(
    actual: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    inside = (actual_arr >= lower_arr) & (actual_arr <= upper_arr)
    return {
        "coverage": float(np.mean(inside)) if len(inside) else 0.0,
        "mean_width": float(np.mean(upper_arr - lower_arr)) if len(inside) else 0.0,
        "median_width": float(np.median(upper_arr - lower_arr)) if len(inside) else 0.0,
    }


def summarize_interval_rows(
    rows: Iterable[Mapping[str, object]],
    group_fields: Sequence[str],
    actual_field: str = "actual",
    prediction_field: str = "prediction",
    lower_field: str = "lower",
    upper_field: str = "upper",
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        actual = np.asarray([float(item[actual_field]) for item in items], dtype=float)
        pred = np.asarray([float(item[prediction_field]) for item in items], dtype=float)
        lower = np.asarray([float(item[lower_field]) for item in items], dtype=float)
        upper = np.asarray([float(item[upper_field]) for item in items], dtype=float)
        inside = (actual >= lower) & (actual <= upper)
        errors = pred - actual
        summary: dict[str, object] = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "count": len(items),
                "coverage": float(np.mean(inside)) if len(inside) else 0.0,
                "mean_width": float(np.mean(upper - lower)) if len(items) else 0.0,
                "median_width": float(np.median(upper - lower)) if len(items) else 0.0,
                "mape": mape_value(actual, pred) if len(actual) else 0.0,
                "signed_error_mean": float(errors.mean()) if len(errors) else 0.0,
                "miss_count": int(np.sum(~inside)) if len(inside) else 0,
            }
        )
        output.append(summary)
    return output
