"""Conservative duplicate-peak handling shared by saved replay and firmware."""

import math
from pathlib import Path
import sys
import unittest

FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))

from identity_contracts import merge_near_duplicate_fomo_peaks


class FomoPeakMergeV6Tests(unittest.TestCase):
    @staticmethod
    def detection(x, y, score):
        return {"center": (x, y), "score": score, "rect": [x - 13.5, y - 13.5, 27, 27]}

    def test_empty_and_single_inputs_are_preserved(self):
        self.assertEqual(merge_near_duplicate_fomo_peaks([]), ([], False, None))
        one = self.detection(100, 100, 0.9)
        self.assertEqual(merge_near_duplicate_fomo_peaks([one]), ([one], False, None))

    def test_close_pair_keeps_highest_scoring_candidate(self):
        low = self.detection(100, 100, 0.6)
        high = self.detection(120, 100, 0.9)
        resolved, merged, span = merge_near_duplicate_fomo_peaks([low, high], enabled=True)
        self.assertTrue(merged)
        self.assertEqual(resolved, [high])
        self.assertEqual(span, 20.0)

    def test_48px_boundary_is_inclusive(self):
        a, b = self.detection(10, 10, 0.8), self.detection(58, 10, 0.7)
        resolved, merged, span = merge_near_duplicate_fomo_peaks([a, b], 48.0, enabled=True)
        self.assertTrue(merged)
        self.assertEqual(span, 48.0)
        self.assertEqual(resolved, [a])

    def test_separated_pair_remains_ambiguous(self):
        a, b = self.detection(10, 10, 0.9), self.detection(59, 10, 0.8)
        resolved, merged, span = merge_near_duplicate_fomo_peaks([a, b], 48.0)
        self.assertFalse(merged)
        self.assertEqual(resolved, [a, b])
        self.assertAlmostEqual(span, 49.0)

    def test_three_peaks_are_never_collapsed(self):
        detections = [self.detection(0, 0, .9), self.detection(10, 0, .8),
                      self.detection(20, 0, .7)]
        self.assertEqual(merge_near_duplicate_fomo_peaks(detections), (detections, False, None))

    def test_equal_scores_keep_first_peak_deterministically(self):
        a, b = self.detection(0, 0, .8), self.detection(1, 0, .8)
        resolved, merged, _span = merge_near_duplicate_fomo_peaks([a, b], enabled=True)
        self.assertTrue(merged)
        self.assertEqual(resolved, [a])

    def test_nonfinite_geometry_and_scores_fail_closed(self):
        with self.assertRaises(ValueError):
            merge_near_duplicate_fomo_peaks([self.detection(math.nan, 0, .8),
                                              self.detection(1, 0, .9)])
        with self.assertRaises(ValueError):
            merge_near_duplicate_fomo_peaks([self.detection(0, 0, math.inf),
                                              self.detection(1, 0, .9)])

    def test_invalid_limits_fail_closed(self):
        pair = [self.detection(0, 0, .8), self.detection(1, 0, .9)]
        for limit in (0, -1, math.inf, math.nan):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                merge_near_duplicate_fomo_peaks(pair, limit)

    def test_seeded_geometry_cases(self):
        for seed in range(10):
            x, y = seed * 9, seed * 7
            distance = 40.0 if seed % 2 == 0 else 49.0
            a, b = self.detection(x, y, .9), self.detection(x + distance, y, .8)
            resolved, merged, span = merge_near_duplicate_fomo_peaks([a, b], enabled=True)
            self.assertEqual(merged, distance <= 48.0)
            self.assertEqual(len(resolved), 1 if merged else 2)
            self.assertAlmostEqual(span, distance)

    def test_close_pair_stays_ambiguous_by_default(self):
        pair = [self.detection(10, 10, .9), self.detection(30, 10, .8)]
        self.assertEqual(merge_near_duplicate_fomo_peaks(pair), (pair, False, 20.0))


if __name__ == "__main__":
    unittest.main()
