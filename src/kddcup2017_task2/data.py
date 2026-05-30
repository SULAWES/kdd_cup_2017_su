from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Combo = Tuple[str, str]
WindowKey = Tuple[datetime, str, str]

TARGET_TIMES = (
    time(8, 0),
    time(8, 20),
    time(8, 40),
    time(9, 0),
    time(9, 20),
    time(9, 40),
    time(17, 0),
    time(17, 20),
    time(17, 40),
    time(18, 0),
    time(18, 20),
    time(18, 40),
)
OBS_TIMES = {
    "morning": (time(6, 0), time(6, 20), time(6, 40), time(7, 0), time(7, 20), time(7, 40)),
    "evening": (time(15, 0), time(15, 20), time(15, 40), time(16, 0), time(16, 20), time(16, 40)),
}
WINDOW_RE = re.compile(r"^\[(.*),(.*)\)$")


@dataclass(frozen=True)
class TargetRow:
    tollgate_id: str
    direction: str
    start: datetime

    @property
    def combo(self) -> Combo:
        return self.tollgate_id, self.direction


def project_paths(data_dir: Path) -> Mapping[str, Path]:
    return {
        "train1_volume": data_dir / "dataSets" / "training" / "volume(table 6)_training.csv",
        "train2_volume": data_dir / "dataSet_phase2" / "volume(table 6)_training2.csv",
        "test1_volume": data_dir / "dataSets" / "testing_phase1" / "volume(table 6)_test1.csv",
        "test2_volume": data_dir / "dataSet_phase2" / "volume(table 6)_test2.csv",
        "sample_volume": data_dir / "submission_sample_volume.csv",
        "weather_train": data_dir / "weather (table 7)_training_update.csv",
        "weather_train_orig": data_dir / "dataSets" / "training" / "weather (table 7)_training.csv",
        "weather_phase1": data_dir / "dataSets" / "testing_phase1" / "weather (table 7)_test1.csv",
        "weather_phase2": data_dir / "dataSet_phase2" / "weather (table 7)_2.csv",
    }


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def floor_20min(dt: datetime) -> datetime:
    minute = int(math.floor(dt.minute / 20) * 20)
    return datetime(dt.year, dt.month, dt.day, dt.hour, minute, 0)


def combine_date_time(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock)


def block_name(start: datetime) -> str:
    if start.hour < 12:
        return "morning"
    return "evening"


def format_window(start: datetime) -> str:
    end = start + timedelta(minutes=20)
    return f"[{start:%Y-%m-%d %H:%M:%S},{end:%Y-%m-%d %H:%M:%S})"


def parse_window_start(value: str) -> datetime:
    match = WINDOW_RE.match(value)
    if not match:
        raise ValueError(f"invalid time_window: {value!r}")
    return parse_dt(match.group(1))


def read_volume_aggregate(paths: Iterable[Path]) -> Dict[WindowKey, int]:
    volumes: Dict[WindowKey, int] = defaultdict(int)
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt_value = row.get("time") or row.get("date_time")
                tollgate = row.get("tollgate_id") or row.get("tollgate")
                direction = row["direction"]
                start = floor_20min(parse_dt(dt_value))
                volumes[(start, str(tollgate), str(direction))] += 1
    return dict(volumes)


def merge_aggregates(*aggregates: Mapping[WindowKey, int]) -> Dict[WindowKey, int]:
    merged: Dict[WindowKey, int] = defaultdict(int)
    for aggregate in aggregates:
        for key, value in aggregate.items():
            merged[key] += value
    return dict(merged)


def infer_combos(aggregate: Mapping[WindowKey, int]) -> List[Combo]:
    combos = sorted({(tollgate, direction) for _, tollgate, direction in aggregate})
    return combos


def infer_dates(aggregate: Mapping[WindowKey, int]) -> List[date]:
    return sorted({start.date() for start, _, _ in aggregate})


def make_target_rows(days: Sequence[date], combos: Sequence[Combo]) -> List[TargetRow]:
    rows: List[TargetRow] = []
    for combo in combos:
        for clock in TARGET_TIMES:
            for day in days:
                rows.append(TargetRow(combo[0], combo[1], combine_date_time(day, clock)))
    return rows


def make_target_rows_like_sample(sample_path: Path, first_day: date) -> List[TargetRow]:
    sample_days, _ = read_sample_shape(sample_path)
    if not sample_days:
        return []
    day_shift = first_day - sample_days[0]
    rows: List[TargetRow] = []
    with sample_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_start = parse_window_start(row["time_window"])
            rows.append(
                TargetRow(
                    str(row["tollgate_id"]),
                    str(row["direction"]),
                    sample_start + day_shift,
                )
            )
    return rows


def target_volume(aggregate: Mapping[WindowKey, int], row: TargetRow) -> int:
    return int(aggregate.get((row.start, row.tollgate_id, row.direction), 0))


def read_sample_shape(path: Path) -> Tuple[List[date], List[Combo]]:
    days = set()
    combos = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.add(parse_window_start(row["time_window"]).date())
            combos.add((str(row["tollgate_id"]), str(row["direction"])))
    return sorted(days), sorted(combos)


def shifted_days(sample_path: Path, first_day: date) -> List[date]:
    sample_days, _ = read_sample_shape(sample_path)
    return [first_day + timedelta(days=i) for i in range(len(sample_days))]


def write_submission(path: Path, rows: Sequence[TargetRow], predictions: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tollgate_id", "time_window", "direction", "volume"])
        for row, pred in zip(rows, predictions):
            writer.writerow(
                [
                    row.tollgate_id,
                    format_window(row.start),
                    row.direction,
                    f"{max(0.0, pred):.6f}".rstrip("0").rstrip("."),
                ]
            )


def load_weather(paths: Iterable[Path]) -> Dict[datetime, Dict[str, float]]:
    weather: Dict[datetime, Dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = datetime.combine(
                    datetime.strptime(row["date"], "%Y-%m-%d").date(),
                    time(int(row["hour"]), 0),
                )
                weather[dt] = {
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"date", "hour"} and value != ""
                }
    return weather


def weather_at(weather: Mapping[datetime, Mapping[str, float]], start: datetime) -> Mapping[str, float]:
    weather_hour = start.hour - (start.hour % 3)
    key = datetime.combine(start.date(), time(weather_hour, 0))
    return weather.get(key, {})
