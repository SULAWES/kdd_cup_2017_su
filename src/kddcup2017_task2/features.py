from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Mapping, Sequence

from .data import OBS_TIMES, AttrKey, TargetRow, WindowKey, block_name, combine_date_time, target_volume, weather_at


ATTR_VALUES = {
    "model": ("0", "1", "2", "3", "4", "5", "6", "7"),
    "etc": ("0", "1"),
    "veh_type": ("blank", "0", "1"),
}


class FeatureBuilder:
    def __init__(self, train_aggregate: Mapping[WindowKey, int], weather: Mapping, include_weather: bool = False):
        self.train_aggregate = train_aggregate
        self.weather = weather
        self.include_weather = include_weather
        self.global_mean = 1.0
        self.combo_mean: Dict[tuple, float] = {}
        self.combo_slot_mean: Dict[tuple, float] = {}

    def fit_stats(self, rows: Sequence[TargetRow]) -> None:
        values = []
        by_combo = defaultdict(list)
        by_combo_slot = defaultdict(list)
        for row in rows:
            y = target_volume(self.train_aggregate, row)
            values.append(y)
            slot = self._target_slot(row.start)
            by_combo[row.combo].append(y)
            by_combo_slot[(row.combo, slot)].append(y)
        self.global_mean = sum(values) / len(values) if values else 1.0
        self.combo_mean = {key: sum(items) / len(items) for key, items in by_combo.items()}
        self.combo_slot_mean = {key: sum(items) / len(items) for key, items in by_combo_slot.items()}

    def transform_row(
        self,
        row: TargetRow,
        known_aggregate: Mapping[WindowKey, int],
        attr_aggregate: Mapping[AttrKey, int] | None = None,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        combo = row.combo
        slot = self._target_slot(row.start)
        block = block_name(row.start)

        features["bias"] = 1.0
        features[f"tollgate={row.tollgate_id}"] = 1.0
        features[f"direction={row.direction}"] = 1.0
        features[f"combo={row.tollgate_id}_{row.direction}"] = 1.0
        features[f"slot={slot}"] = 1.0
        features[f"dow={row.start.weekday()}"] = 1.0
        features["is_weekend"] = 1.0 if row.start.weekday() >= 5 else 0.0
        features["day_of_month"] = float(row.start.day)
        features["target_hour"] = float(row.start.hour)
        features["target_minute"] = float(row.start.minute)
        features["time_sin"] = math.sin(2 * math.pi * (row.start.hour * 60 + row.start.minute) / 1440.0)
        features["time_cos"] = math.cos(2 * math.pi * (row.start.hour * 60 + row.start.minute) / 1440.0)

        obs_values = []
        for idx, clock in enumerate(OBS_TIMES[block]):
            obs_start = combine_date_time(row.start.date(), clock)
            value = float(known_aggregate.get((obs_start, row.tollgate_id, row.direction), 0))
            features[f"obs_{block}_{idx}"] = value
            obs_values.append(value)
        features["obs_sum"] = sum(obs_values)
        features["obs_mean"] = sum(obs_values) / len(obs_values)
        features["obs_max"] = max(obs_values) if obs_values else 0.0
        features["obs_last"] = obs_values[-1] if obs_values else 0.0
        features["obs_trend"] = (obs_values[-1] - obs_values[0]) if obs_values else 0.0
        if attr_aggregate:
            self._add_attr_features(features, row, attr_aggregate)

        lag_1 = self._history_value(known_aggregate, row, 1)
        lag_7 = self._history_value(known_aggregate, row, 7)
        combo_mean = self.combo_mean.get(combo, self.global_mean)
        combo_slot_mean = self.combo_slot_mean.get((combo, slot), combo_mean)
        features["lag_1"] = lag_1 if lag_1 is not None else combo_slot_mean
        features["lag_7"] = lag_7 if lag_7 is not None else combo_slot_mean
        features["combo_mean"] = combo_mean
        features["combo_slot_mean"] = combo_slot_mean

        if self.include_weather:
            for key, value in weather_at(self.weather, row.start).items():
                features[f"weather_{key}"] = float(value)

        return features

    def transform(
        self,
        rows: Sequence[TargetRow],
        known_aggregate: Mapping[WindowKey, int],
        attr_aggregate: Mapping[AttrKey, int] | None = None,
    ) -> List[Dict[str, float]]:
        return [self.transform_row(row, known_aggregate, attr_aggregate) for row in rows]

    @staticmethod
    def _target_slot(start) -> str:
        return f"{start.hour:02d}:{start.minute:02d}"

    @staticmethod
    def _history_value(known_aggregate: Mapping[WindowKey, int], row: TargetRow, days_back: int):
        start = row.start - timedelta(days=days_back)
        key = (start, row.tollgate_id, row.direction)
        if key in known_aggregate:
            return float(known_aggregate[key])
        return None

    @staticmethod
    def _add_attr_features(
        features: Dict[str, float],
        row: TargetRow,
        attr_aggregate: Mapping[AttrKey, int],
    ) -> None:
        block = block_name(row.start)
        for clock in OBS_TIMES[block]:
            obs_start = combine_date_time(row.start.date(), clock)
            for attr_name, values in ATTR_VALUES.items():
                for attr_value in values:
                    feature_name = f"{attr_name}_{attr_value}_obs_sum"
                    features[feature_name] = features.get(feature_name, 0.0) + float(
                        attr_aggregate.get(
                            (obs_start, row.tollgate_id, row.direction, attr_name, attr_value),
                            0,
                        )
                    )

        for attr_name, values in ATTR_VALUES.items():
            total = sum(features.get(f"{attr_name}_{attr_value}_obs_sum", 0.0) for attr_value in values)
            if total <= 0:
                continue
            for attr_value in values:
                sum_name = f"{attr_name}_{attr_value}_obs_sum"
                features[f"{attr_name}_{attr_value}_obs_share"] = features.get(sum_name, 0.0) / total


class Vectorizer:
    def __init__(self):
        self.names: List[str] = []

    def fit_transform(self, rows: Sequence[Mapping[str, float]]):
        names = sorted({name for row in rows for name in row.keys()})
        self.names = names
        return self.transform(rows)

    def transform(self, rows: Sequence[Mapping[str, float]]):
        import numpy as np

        matrix = np.zeros((len(rows), len(self.names)), dtype=float)
        for i, row in enumerate(rows):
            for j, name in enumerate(self.names):
                matrix[i, j] = float(row.get(name, 0.0))
        return matrix
