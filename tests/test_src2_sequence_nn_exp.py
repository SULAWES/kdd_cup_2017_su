from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src2"))

from kddcup2017_task2.data import TargetRow
from kddcup2017_task2_exp2.sequence_nn_exp import build_holdout_known, fit_array_normalizer, row_obs_values


class SequenceNNExpTests(unittest.TestCase):
    def test_build_holdout_known_keeps_only_green_windows_from_holdout_days(self) -> None:
        fit_day = date(2016, 10, 1)
        holdout_day = date(2016, 10, 2)
        fit_target = (datetime.combine(fit_day, time(8, 0)), "1", "0")
        holdout_green = (datetime.combine(holdout_day, time(6, 0)), "1", "0")
        holdout_red = (datetime.combine(holdout_day, time(8, 0)), "1", "0")
        aggregate = {
            fit_target: 11,
            holdout_green: 7,
            holdout_red: 999,
        }

        known = build_holdout_known(aggregate, [fit_day], [holdout_day])

        self.assertEqual(known[fit_target], 11)
        self.assertEqual(known[holdout_green], 7)
        self.assertNotIn(holdout_red, known)

    def test_row_obs_values_uses_same_day_observation_block_only(self) -> None:
        day = date(2016, 10, 3)
        row = TargetRow("1", "0", datetime.combine(day, time(8, 40)))
        aggregate = {
            (datetime.combine(day, time(6, 0)), "1", "0"): 1,
            (datetime.combine(day, time(6, 20)), "1", "0"): 2,
            (datetime.combine(day, time(6, 40)), "1", "0"): 3,
            (datetime.combine(day, time(7, 0)), "1", "0"): 4,
            (datetime.combine(day, time(7, 20)), "1", "0"): 5,
            (datetime.combine(day, time(7, 40)), "1", "0"): 6,
            (datetime.combine(day, time(8, 40)), "1", "0"): 999,
        }

        self.assertEqual(row_obs_values(aggregate, row), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_fit_array_normalizer_handles_constant_columns(self) -> None:
        mean, std = fit_array_normalizer([[3.0, 4.0], [3.0, 6.0]])

        self.assertEqual(mean.tolist(), [3.0, 5.0])
        self.assertEqual(std.tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
