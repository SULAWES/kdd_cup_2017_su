from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kddcup2017_task2.data import TargetRow
from src3_explore.common.metrics import summarize_errors
from src3_explore.common.reporting import ExperimentCard
from src3_explore.common.candidate_cache import calibration_actual_values
from src3_explore.common.visibility import build_known_for_eval_days
from src3_explore.diagnostics.green_red_transfer_analysis import fit_nonnegative_transfer_matrix
from src3_explore.diagnostics.model_disagreement import nearest_model_by_actual
from src3_explore.explain.common import ExplanationCard


class Src3ExploreCoreTests(unittest.TestCase):
    def test_build_known_for_eval_days_exposes_only_eval_green_windows(self) -> None:
        train_day = date(2016, 10, 1)
        eval_day = date(2016, 10, 2)
        train_red = (datetime.combine(train_day, time(8, 0)), "1", "0")
        eval_green = (datetime.combine(eval_day, time(6, 0)), "1", "0")
        eval_red = (datetime.combine(eval_day, time(8, 0)), "1", "0")
        aggregate = {train_red: 11, eval_green: 7, eval_red: 999}

        known = build_known_for_eval_days(aggregate, [train_day], [eval_day])

        self.assertEqual(known[train_red], 11)
        self.assertEqual(known[eval_green], 7)
        self.assertNotIn(eval_red, known)

    def test_experiment_card_markdown_contains_required_sections(self) -> None:
        card = ExperimentCard(
            name="demo",
            hypothesis="h",
            data_visibility="visible",
            prototype="proto",
            metrics={"mape": "0.123"},
            result="result",
            insight="insight",
            next_step="next",
        )

        markdown = card.to_markdown()

        for section in (
            "## demo",
            "**假设（Hypothesis）:** h",
            "**数据可见性（Data visibility）:** visible",
            "**预期洞察（Expected insight）:** insight",
            "**下一步（Next）:** next",
        ):
            self.assertIn(section, markdown)

    def test_explanation_card_markdown_contains_required_sections(self) -> None:
        card = ExplanationCard(
            name="explain_demo",
            hypothesis="h",
            method="m",
            expected_falsification="f",
            metrics={"mape": "0.1"},
            key_result="r",
            interpretation="i",
            next_step="n",
        )

        markdown = card.to_markdown()

        for section in (
            "## explain_demo",
            "**假设（Hypothesis）:** h",
            "**方法（Method）:** m",
            "**可证伪预期（Expected falsification）:** f",
            "**关键结果（Key result）:** r",
            "**解释（Interpretation）:** i",
            "**下一步（Next step）:** n",
        ):
            self.assertIn(section, markdown)

    def test_summarize_errors_reports_group_mape_and_signed_error(self) -> None:
        rows = [
            {"combo": "1_0", "actual": 10.0, "prediction": 12.0},
            {"combo": "1_0", "actual": 20.0, "prediction": 18.0},
            {"combo": "2_0", "actual": 5.0, "prediction": 10.0},
        ]

        summary = summarize_errors(rows, ["combo"])

        by_combo = {row["combo"]: row for row in summary}
        self.assertAlmostEqual(by_combo["1_0"]["signed_error_mean"], 0.0)
        self.assertAlmostEqual(by_combo["1_0"]["mape"], 0.15)
        self.assertEqual(by_combo["2_0"]["count"], 1)

    def test_fit_nonnegative_transfer_matrix_is_nonnegative(self) -> None:
        green = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8, 9],
            ],
            dtype=float,
        )
        red = green * 2.0

        transfer = fit_nonnegative_transfer_matrix(green, red)

        self.assertEqual(transfer.weights.shape, (6, 6))
        self.assertTrue(np.all(transfer.weights >= -1e-12))
        self.assertTrue(np.all(transfer.predict(green) >= 0.0))

    def test_nearest_model_by_actual_returns_lowest_absolute_error(self) -> None:
        winners = nearest_model_by_actual(
            actual=np.array([10.0, 10.0]),
            prediction_matrix=np.array([[8.0, 12.0, 11.0], [20.0, 9.0, 12.0]]),
            model_names=["a", "b", "c"],
        )

        self.assertEqual(winners, ["c", "b"])

    def test_calibration_actual_values_use_training_labels(self) -> None:
        day = date(2016, 10, 1)
        row = TargetRow("1", "0", datetime.combine(day, time(8, 0)))
        train_agg = {(row.start, row.tollgate_id, row.direction): 17}

        actual = calibration_actual_values(train_agg, [row])

        self.assertEqual(actual.tolist(), [17.0])


if __name__ == "__main__":
    unittest.main()
