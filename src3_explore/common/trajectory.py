from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

from kddcup2017_task2.data import OBS_TIMES, TargetRow, floor_20min, parse_dt


INTERSECTIONS = ("A", "B", "C")


@dataclass(frozen=True)
class TrajectoryStats:
    count: int = 0
    travel_sum: float = 0.0

    @property
    def travel_mean(self) -> float:
        return self.travel_sum / max(self.count, 1)


def read_trajectory_aggregate(paths: Sequence[Path]) -> dict[tuple, TrajectoryStats]:
    totals: dict[tuple, list[float]] = defaultdict(lambda: [0, 0.0])
    for path in paths:
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = floor_20min(parse_dt(row["starting_time"]))
                key = (start, str(row["intersection_id"]), str(row["tollgate_id"]))
                totals[key][0] += 1
                totals[key][1] += float(row["travel_time"])
    return {key: TrajectoryStats(count=int(value[0]), travel_sum=float(value[1])) for key, value in totals.items()}


def merge_trajectory_aggregates(*aggregates: Mapping[tuple, TrajectoryStats]) -> dict[tuple, TrajectoryStats]:
    totals: dict[tuple, list[float]] = defaultdict(lambda: [0, 0.0])
    for aggregate in aggregates:
        for key, value in aggregate.items():
            totals[key][0] += value.count
            totals[key][1] += value.travel_sum
    return {key: TrajectoryStats(count=int(value[0]), travel_sum=float(value[1])) for key, value in totals.items()}


def trajectory_observation_windows_only(
    aggregate: Mapping[tuple, TrajectoryStats],
    days,
) -> dict[tuple, TrajectoryStats]:
    day_set = set(days)
    clocks = set(OBS_TIMES["morning"] + OBS_TIMES["evening"])
    return {
        key: value
        for key, value in aggregate.items()
        if key[0].date() in day_set and key[0].time() in clocks
    }


def trajectory_green_count(row: TargetRow, aggregate: Mapping[tuple, TrajectoryStats]) -> float:
    from kddcup2017_task2.data import block_name, combine_date_time

    total = 0
    for clock in OBS_TIMES[block_name(row.start)]:
        obs_start = combine_date_time(row.start.date(), clock)
        for intersection in INTERSECTIONS:
            total += aggregate.get((obs_start, intersection, row.tollgate_id), TrajectoryStats()).count
    return float(total)


def route_count_at_lag(
    row: TargetRow,
    aggregate: Mapping[tuple, TrajectoryStats],
    intersection: str,
    lag_minutes: int,
) -> float:
    start = row.start - timedelta(minutes=lag_minutes)
    if start >= row.start:
        return 0.0
    return float(aggregate.get((start, intersection, row.tollgate_id), TrajectoryStats()).count)

