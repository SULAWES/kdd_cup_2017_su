from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

from kddcup2017_task2.data import (
    OBS_TIMES,
    TARGET_TIMES,
    AttrKey,
    TargetRow,
    WindowKey,
    infer_combos,
    infer_dates,
    load_weather,
    make_target_rows,
    make_target_rows_like_sample,
    merge_aggregates,
    merge_attr_aggregates,
    project_paths,
    read_volume_aggregate,
    read_volume_attr_aggregate,
)
from kddcup2017_task2.ensemble import latest_training_fold_split


OBS_CLOCKS = frozenset(OBS_TIMES["morning"] + OBS_TIMES["evening"])
TARGET_CLOCKS = frozenset(TARGET_TIMES)


@dataclass(frozen=True)
class VisibilityContext:
    name: str
    data_dir: Path
    train_agg: Mapping[WindowKey, int]
    known_agg: Mapping[WindowKey, int]
    train_attr_agg: Mapping[AttrKey, int]
    known_attr_agg: Mapping[AttrKey, int]
    label_agg: Mapping[WindowKey, int]
    weather: Mapping
    train_days: Sequence[date]
    eval_days: Sequence[date]
    rows: Sequence[TargetRow]
    combos: Sequence[tuple[str, str]]
    data_visibility: str


def build_known_for_eval_days(
    aggregate: Mapping[WindowKey, int],
    train_days: Sequence[date],
    eval_days: Sequence[date],
) -> dict[WindowKey, int]:
    train_day_set = set(train_days)
    eval_day_set = set(eval_days)
    known = {
        key: value
        for key, value in aggregate.items()
        if key[0].date() in train_day_set or (key[0].date() in eval_day_set and key[0].time() in OBS_CLOCKS)
    }
    assert_no_eval_target_windows(known, eval_days)
    return known


def build_attr_known_for_eval_days(
    aggregate: Mapping[AttrKey, int],
    train_days: Sequence[date],
    eval_days: Sequence[date],
) -> dict[AttrKey, int]:
    train_day_set = set(train_days)
    eval_day_set = set(eval_days)
    known = {
        key: value
        for key, value in aggregate.items()
        if key[0].date() in train_day_set or (key[0].date() in eval_day_set and key[0].time() in OBS_CLOCKS)
    }
    assert_no_eval_target_windows(known, eval_days)
    return known


def assert_no_eval_target_windows(known: Mapping, eval_days: Sequence[date]) -> None:
    eval_day_set = set(eval_days)
    leaked = [key for key in known if key[0].date() in eval_day_set and key[0].time() in TARGET_CLOCKS]
    if leaked:
        sample = leaked[0]
        raise ValueError(f"visibility leak: target window {sample[0]} is present in known aggregate")


def load_phase1_context(data_dir: Path) -> VisibilityContext:
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test1_obs = read_volume_aggregate([paths["test1_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    test1_attr = read_volume_attr_aggregate([paths["test1_volume"]])
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase1"]])
    combos = infer_combos(train1)
    train_days = infer_dates(train1)
    eval_days = infer_dates(train2)
    known = merge_aggregates(train1, test1_obs)
    known_attr = merge_attr_aggregates(train1_attr, test1_attr)
    assert_no_eval_target_windows(known, eval_days)
    rows = make_target_rows(eval_days, combos)
    return VisibilityContext(
        name="phase1_final_observation",
        data_dir=data_dir,
        train_agg=train1,
        known_agg=known,
        train_attr_agg=train1_attr,
        known_attr_agg=known_attr,
        label_agg=train2,
        weather=weather,
        train_days=train_days,
        eval_days=eval_days,
        rows=rows,
        combos=combos,
        data_visibility=(
            "Phase1 no-leak observation: train1 labels train models/statistics; test1 green windows are "
            "visible inputs; train2 red labels are available only to score fixed outputs."
        ),
    )


def load_train1_latest_fold_context(data_dir: Path) -> VisibilityContext:
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"]])
    combos = infer_combos(train1)
    all_days = infer_dates(train1)
    fit_days, eval_days = latest_training_fold_split(all_days)
    train_agg = {key: value for key, value in train1.items() if key[0].date() in set(fit_days)}
    train_attr = {key: value for key, value in train1_attr.items() if key[0].date() in set(fit_days)}
    known = build_known_for_eval_days(train1, fit_days, eval_days)
    known_attr = build_attr_known_for_eval_days(train1_attr, fit_days, eval_days)
    rows = make_target_rows(eval_days, combos)
    return VisibilityContext(
        name="train1_latest_fold",
        data_dir=data_dir,
        train_agg=train_agg,
        known_agg=known,
        train_attr_agg=train_attr,
        known_attr_agg=known_attr,
        label_agg=train1,
        weather=weather,
        train_days=fit_days,
        eval_days=eval_days,
        rows=rows,
        combos=combos,
        data_visibility=(
            "Train1 rolling fold: fit days train models/statistics; held-out train1 days expose only "
            "same-day green windows before their red labels are used for scoring."
        ),
    )


def load_phase2_visible_rows(data_dir: Path) -> VisibilityContext:
    paths = project_paths(data_dir)
    train1 = read_volume_aggregate([paths["train1_volume"]])
    train2 = read_volume_aggregate([paths["train2_volume"]])
    test2_obs = read_volume_aggregate([paths["test2_volume"]])
    train1_attr = read_volume_attr_aggregate([paths["train1_volume"]])
    train2_attr = read_volume_attr_aggregate([paths["train2_volume"]])
    test2_attr = read_volume_attr_aggregate([paths["test2_volume"]])
    train_all = merge_aggregates(train1, train2)
    train_attr = merge_attr_aggregates(train1_attr, train2_attr)
    known = merge_aggregates(train_all, test2_obs)
    known_attr = merge_attr_aggregates(train_attr, test2_attr)
    weather = load_weather([paths["weather_train"], paths["weather_train_orig"], paths["weather_phase2"]])
    pred_days = infer_dates(test2_obs)
    first_pred_day = min(pred_days)
    rows = make_target_rows_like_sample(paths["sample_volume"], first_pred_day)
    combos = sorted({row.combo for row in rows}) or infer_combos(train_all)
    assert_no_eval_target_windows(known, pred_days)
    return VisibilityContext(
        name="phase2_visible_unlabeled",
        data_dir=data_dir,
        train_agg=train_all,
        known_agg=known,
        train_attr_agg=train_attr,
        known_attr_agg=known_attr,
        label_agg={},
        weather=weather,
        train_days=infer_dates(train_all),
        eval_days=pred_days,
        rows=rows,
        combos=combos,
        data_visibility=(
            "Phase2 visible unlabeled rows: train1+train2 labels may train models; test2 green windows "
            "are visible inputs; target red labels are unavailable."
        ),
    )

