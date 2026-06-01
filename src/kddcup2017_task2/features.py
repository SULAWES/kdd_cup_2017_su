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
        self.combo_slot_median: Dict[tuple, float] = {}
        self.combo_slot_dow_mean: Dict[tuple, float] = {}
        self.combo_block_mean: Dict[tuple, float] = {}
        self.combo_slot_obs_ratio: Dict[tuple, float] = {}

    def fit_stats(self, rows: Sequence[TargetRow]) -> None:
        values = []
        by_combo = defaultdict(list)
        by_combo_slot = defaultdict(list)
        by_combo_slot_dow = defaultdict(list)
        by_combo_block = defaultdict(list)
        by_combo_slot_ratio = defaultdict(list)
        for row in rows:
            y = target_volume(self.train_aggregate, row)
            values.append(y)
            slot = self._target_slot(row.start)
            block = block_name(row.start)
            by_combo[row.combo].append(y)
            by_combo_slot[(row.combo, slot)].append(y)
            by_combo_slot_dow[(row.combo, slot, row.start.weekday())].append(y)
            by_combo_block[(row.combo, block)].append(y)
            obs_sum = self._obs_sum(self.train_aggregate, row)
            if obs_sum > 0:
                by_combo_slot_ratio[(row.combo, slot)].append(y / obs_sum)
        self.global_mean = sum(values) / len(values) if values else 1.0
        self.combo_mean = {key: sum(items) / len(items) for key, items in by_combo.items()}
        self.combo_slot_mean = {key: sum(items) / len(items) for key, items in by_combo_slot.items()}
        self.combo_slot_median = {key: self._median(items) for key, items in by_combo_slot.items()}
        self.combo_slot_dow_mean = {key: sum(items) / len(items) for key, items in by_combo_slot_dow.items()}
        self.combo_block_mean = {key: sum(items) / len(items) for key, items in by_combo_block.items()}
        self.combo_slot_obs_ratio = {key: self._median(items) for key, items in by_combo_slot_ratio.items()}

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
        features["obs_first"] = obs_values[0] if obs_values else 0.0
        features["obs_std"] = self._std(obs_values)
        features["obs_last_first_ratio"] = (
            (obs_values[-1] + 1.0) / (obs_values[0] + 1.0) if obs_values else 1.0
        )
        if attr_aggregate:
            self._add_attr_features(features, row, attr_aggregate)

        lag_1 = self._history_value(known_aggregate, row, 1)
        lag_7 = self._history_value(known_aggregate, row, 7)
        combo_mean = self.combo_mean.get(combo, self.global_mean)
        combo_slot_mean = self.combo_slot_mean.get((combo, slot), combo_mean)
        combo_slot_median = self.combo_slot_median.get((combo, slot), combo_slot_mean)
        combo_slot_dow_mean = self.combo_slot_dow_mean.get((combo, slot, row.start.weekday()), combo_slot_mean)
        combo_block_mean = self.combo_block_mean.get((combo, block), combo_mean)
        obs_ratio = self.combo_slot_obs_ratio.get(
            (combo, slot),
            combo_slot_mean / max(self._obs_sum(self.train_aggregate, row), 1.0),
        )
        features["lag_1"] = lag_1 if lag_1 is not None else combo_slot_mean
        features["lag_7"] = lag_7 if lag_7 is not None else combo_slot_mean
        features["combo_mean"] = combo_mean
        features["combo_slot_mean"] = combo_slot_mean
        features["combo_slot_median"] = combo_slot_median
        features["combo_slot_dow_mean"] = combo_slot_dow_mean
        features["combo_block_mean"] = combo_block_mean
        features["obs_ratio_pred"] = features["obs_sum"] * obs_ratio
        self._add_rolling_features(features, row, known_aggregate, combo_slot_mean)
        self._add_holiday_features(features, row)

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

    def _add_rolling_features(
        self,
        features: Dict[str, float],
        row: TargetRow,
        known_aggregate: Mapping[WindowKey, int],
        fallback: float,
    ) -> None:
        for window in (3, 7, 14):
            values = []
            for days_back in range(1, window + 1):
                value = self._history_value(known_aggregate, row, days_back)
                if value is not None:
                    values.append(value)
            features[f"hist_mean_{window}"] = sum(values) / len(values) if values else fallback
            features[f"hist_median_{window}"] = self._median(values) if values else fallback

    @staticmethod
    def _add_holiday_features(features: Dict[str, float], row: TargetRow) -> None:
        # China National Day holiday period in the 2016 KDD Cup data.
        day = row.start.date()
        features["is_national_day"] = 1.0 if day.month == 10 and 1 <= day.day <= 7 else 0.0
        features["is_post_holiday"] = 1.0 if day.month == 10 and 8 <= day.day <= 14 else 0.0
        features["days_since_national_day"] = (
            float(max(-7, min(21, (day - day.replace(month=10, day=7)).days)))
            if day.month in (9, 10)
            else 0.0
        )

    @staticmethod
    def _obs_sum(aggregate: Mapping[WindowKey, int], row: TargetRow) -> float:
        block = block_name(row.start)
        return float(
            sum(
                aggregate.get((combine_date_time(row.start.date(), clock), row.tollgate_id, row.direction), 0)
                for clock in OBS_TIMES[block]
            )
        )

    @staticmethod
    def _median(values) -> float:
        if not values:
            return 0.0
        items = sorted(float(value) for value in values)
        mid = len(items) // 2
        if len(items) % 2:
            return items[mid]
        return (items[mid - 1] + items[mid]) / 2.0

    @staticmethod
    def _std(values) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

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
