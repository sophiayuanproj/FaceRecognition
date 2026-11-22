"""Synthetic tests for report-level steady-average diagnostics (no hardware claim)."""
import math
import random
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import steady_average as sa


class SteadyAverageTests(unittest.TestCase):
    def test_four_value_report_center_resists_one_isolated_spike(self):
        samples = [
            {"voltage_v": 5.20, "current_a": 0.047},
            {"voltage_v": 5.20, "current_a": 0.047},
            {"voltage_v": 99.0, "current_a": 100.0},
            {"voltage_v": 5.20, "current_a": 0.047},
        ]
        center = sa.report_center(samples)
        self.assertAlmostEqual(center["voltage_center_v"], 5.20)
        self.assertAlmostEqual(center["current_center_a"], 0.047)
        self.assertAlmostEqual(center["power_center_w"], 5.20 * 0.047)
        self.assertEqual(center["voltage_sample_max_v"], 99.0)

    def test_ten_distinct_synthetic_roots_keep_voltage_and_power_exclusions_visible(self):
        for root_seed in range(5101, 5111):
            with self.subTest(root_seed=root_seed):
                rng = random.Random(root_seed)
                rows = []
                for index in range(80):
                    power = 0.245 + rng.uniform(-0.0012, 0.0012)
                    rows.append({
                        "report_arrival_index": index,
                        "voltage_center_v": 5.20 + rng.uniform(-0.01, 0.01),
                        "current_center_a": power / 5.20,
                        "power_center_w": power,
                        "accepted_for_average": None,
                        "rejection_reason": "",
                    })
                rows[3]["power_center_w"] = 0.40
                rows[3]["current_center_a"] = 0.40 / rows[3]["voltage_center_v"]
                rows[7]["voltage_center_v"] = 0.0
                result = sa.classify_report_centers(rows)
                self.assertEqual(result["voltage_rejected_count"], 1)
                self.assertGreaterEqual(result["power_rejected_count"], 1)
                self.assertEqual(len(result["accepted_rows"]), 78)
                self.assertFalse(rows[3]["accepted_for_average"])
                self.assertFalse(rows[7]["accepted_for_average"])
                self.assertAlmostEqual(
                    sum(row["power_center_w"] for row in result["accepted_rows"])
                    / len(result["accepted_rows"]),
                    0.245,
                    delta=0.0002,
                )

    def test_bootstrap_is_reproducible_and_constant_input_is_exact(self):
        values = [0.247] * 120
        first = sa.circular_block_bootstrap_means(values, 17, 7101, 1000)
        second = sa.circular_block_bootstrap_means(values, 17, 7101, 1000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1000)
        self.assertTrue(all(value == 0.247 for value in first))

    def test_bootstrap_rejects_empty_or_invalid_parameters(self):
        with self.assertRaises(ValueError):
            sa.circular_block_bootstrap_means([], 1, 1)
        with self.assertRaises(ValueError):
            sa.circular_block_bootstrap_means([1.0], 0, 1)

    def test_quantile_interpolates(self):
        self.assertEqual(sa.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(sa.quantile([1.0, 2.0, 3.0, 4.0], 0.0), 1.0)
        self.assertEqual(sa.quantile([1.0, 2.0, 3.0, 4.0], 1.0), 4.0)

    def test_zero_mad_retains_only_exact_center(self):
        rows = [
            {"voltage_center_v": 5.2, "current_center_a": p / 5.2, "power_center_w": p,
             "accepted_for_average": None, "rejection_reason": ""}
            for p in [0.2, 0.2, 0.2, 0.3]
        ]
        result = sa.classify_report_centers(rows)
        self.assertEqual(result["accepted_rows"], rows[:3])
        self.assertEqual(result["power_rejected_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
