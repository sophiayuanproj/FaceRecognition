"""Host replay for the optional Nicla scene-change trigger."""

import math
from pathlib import Path
import random
import sys
import unittest


APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "firmware"))

from runtime_support import (SCENE_SIGNATURE_LENGTH, SimpleSceneTrigger,
                             sample_gray_scene_signature)  # noqa: E402


CONFIG = {"min_detector_interval_ms": 500, "detector_fallback_ms": 2000}
SPATIAL_CONFIG = dict(CONFIG, spatial={"cell_delta_u8": 12, "onset_cells": 3, "hysteresis_cells": 1})
FRAME_SIDE = 240


def ticks_diff(now, earlier):
    return ((now - earlier + 2048) % 4096) - 2048


def translated_subject(center_x, center_y, background, foreground):
    """Render a face-sized silhouette with constant histogram under translation."""
    radius_x, radius_y = 22, 30
    pixels = bytearray([background]) * (FRAME_SIDE * FRAME_SIDE)
    for y_offset in range(-radius_y, radius_y + 1):
        half_width = math.isqrt(radius_x ** 2 * (radius_y ** 2 - y_offset ** 2) // radius_y ** 2)
        left = (center_y + y_offset) * FRAME_SIDE + center_x - half_width
        span = 2 * half_width + 1
        pixels[left:left + span] = bytes([foreground]) * span
    return pixels


def same_stats_motion_sequence(seed):
    rng = random.Random(seed)
    start_x = rng.randrange(66, 75)
    center_y = rng.randrange(97, 114)
    background = rng.randrange(65, 85)
    foreground = rng.randrange(131, 156)
    for step in range(9):
        yield translated_subject(start_x + 8 * step, center_y, background, foreground)


class SpatialSceneTriggerTest(unittest.TestCase):
    def test_equal_histogram_motion_triggers_before_periodic_fallback_for_ten_seeds(self):
        for seed in range(421700, 421710):
            with self.subTest(seed=seed):
                spatial = SimpleSceneTrigger(SPATIAL_CONFIG, ticks_diff)
                global_only = SimpleSceneTrigger(CONFIG, ticks_diff)
                scratch = bytearray(SCENE_SIGNATURE_LENGTH)
                motion_checks = []
                global_checks = []
                first_sum = None
                for index, frame in enumerate(same_stats_motion_sequence(seed)):
                    total = sum(frame)
                    if first_sum is None:
                        first_sum = total
                    self.assertEqual(total, first_sum)
                    # The image uses only two fixed luma values, so identical
                    # sums also imply an identical histogram and variance.
                    mean = total / len(frame)
                    background = min(frame)
                    foreground = max(frame)
                    foreground_fraction = (mean - background) / (foreground - background)
                    stdev = math.sqrt(foreground_fraction * (1 - foreground_fraction)) * (foreground - background)
                    now = 250 * index
                    signature = sample_gray_scene_signature(frame, FRAME_SIDE, FRAME_SIDE, scratch)
                    self.assertIs(signature, scratch)
                    result = spatial.decide(mean, stdev, now, signature)
                    if result["run_detector"]:
                        motion_checks.append((now, result))
                    if global_only.decide(mean, stdev, now)["run_detector"]:
                        global_checks.append(now)

                self.assertEqual(global_checks, [0, 2000])
                early_check, evidence = motion_checks[1]
                self.assertGreaterEqual(early_check, 500)
                self.assertLess(early_check, 2000)
                self.assertFalse(evidence["periodic_fallback"])
                self.assertEqual(evidence["delta_mean_u8"], 0)
                self.assertEqual(evidence["delta_stdev_u8"], 0)
                self.assertGreaterEqual(evidence["spatial_changed_cells"], 3)
                self.assertEqual(spatial.detector_calls + spatial.skipped_frames, 9)
                self.assertLess(spatial.detector_calls, 9)

    def test_still_scene_uses_periodic_detector_checks_across_tick_wrap(self):
        trigger = SimpleSceneTrigger(SPATIAL_CONFIG, ticks_diff)
        calls = []
        for elapsed in range(0, 6001, 250):
            # Small coherent brightness changes model exposure/readout noise.
            signature = bytes([80 + (elapsed // 250) % 4] * SCENE_SIGNATURE_LENGTH)
            result = trigger.decide(signature[0], 8, (3700 + elapsed) % 4096, signature)
            if result["run_detector"]:
                calls.append((elapsed, result["periodic_fallback"]))
        self.assertEqual(calls, [(0, False), (2000, True), (4000, True), (6000, True)])
        self.assertEqual(trigger.detector_calls, 4)
        self.assertEqual(trigger.skipped_frames, 21)

    def test_cooldown_hysteresis_and_reference_are_bound_to_detector_checks(self):
        trigger = SimpleSceneTrigger(SPATIAL_CONFIG, ticks_diff)

        def at(ticks, count):
            signature = bytes([100] * count + [80] * (SCENE_SIGNATURE_LENGTH - count))
            return trigger.decide(80, 8, ticks, signature)

        self.assertTrue(at(0, 0)["run_detector"])
        self.assertEqual(at(100, 3)["reason"], "scene_change_waiting_for_cooldown")
        self.assertEqual(at(200, 2)["reason"], "scene_change_waiting_for_cooldown")
        self.assertEqual(at(300, 1)["reason"], "scene_stable")
        self.assertEqual(at(500, 0)["reason"], "scene_stable")
        self.assertTrue(at(600, 3)["run_detector"])
        self.assertEqual(at(700, 3)["spatial_changed_cells"], 0)
        self.assertFalse(at(800, 4)["run_detector"])

    def test_subthreshold_frame_to_frame_changes_accumulate_from_last_detector_check(self):
        trigger = SimpleSceneTrigger(SPATIAL_CONFIG, ticks_diff)
        calls = []
        for step in range(6):
            # Consecutive signatures differ by only three luma levels.
            shift = 3 * step
            signature = bytes([80 + shift] * 3 + [80 - shift] * 3
                              + [80] * (SCENE_SIGNATURE_LENGTH - 6))
            result = trigger.decide(80, 8, 250 * step, signature)
            if result["run_detector"]:
                calls.append((step, result["spatial_changed_cells"]))
        self.assertEqual(calls, [(0, None), (4, 6)])

    def test_missing_signature_and_camera_discontinuity_clear_reference(self):
        trigger = SimpleSceneTrigger(SPATIAL_CONFIG, ticks_diff)
        initial = bytes([80] * SCENE_SIGNATURE_LENGTH)
        changed = bytes([100] * SCENE_SIGNATURE_LENGTH)
        self.assertTrue(trigger.decide(80, 8, 0, initial)["run_detector"])
        self.assertFalse(trigger.decide(80, 8, 100, changed)["run_detector"])
        invalid = trigger.decide(80, 8, 200, changed[:3])
        self.assertEqual(invalid["reason"], "invalid_spatial_signature_fail_open")
        self.assertTrue(invalid["run_detector"])
        self.assertIsNone(invalid["spatial_changed_cells"])
        fresh = trigger.decide(80, 8, 300, changed)
        self.assertEqual(fresh["reason"], "initial_scene_check")
        self.assertFalse(trigger.decide(80, 8, 400, changed)["run_detector"])
        trigger.reset_reference()
        self.assertTrue(trigger.decide(80, 8, 450, initial)["run_detector"])
        self.assertEqual(trigger.detector_calls, 4)

    def test_legacy_scene_trigger_ignores_optional_signature_and_keeps_result_shape(self):
        trigger = SimpleSceneTrigger(CONFIG, ticks_diff)
        initial = trigger.decide(80, 8, 0)
        self.assertEqual(initial, {"run_detector": True, "reason": "initial_scene_check",
            "delta_mean_u8": None, "delta_stdev_u8": None, "periodic_fallback": False})
        stable = trigger.decide(80, 8, 500, bytes([100] * SCENE_SIGNATURE_LENGTH))
        self.assertEqual(stable, {"run_detector": False, "reason": "scene_stable",
            "delta_mean_u8": 0, "delta_stdev_u8": 0, "periodic_fallback": False})

    def test_signature_samples_four_pixel_quarters_in_each_cell(self):
        ramp = bytearray(x // 2 + y // 2 for y in range(FRAME_SIDE) for x in range(FRAME_SIDE))
        scratch = bytearray(SCENE_SIGNATURE_LENGTH)
        sample_gray_scene_signature(ramp, FRAME_SIDE, FRAME_SIDE, scratch)
        self.assertEqual(scratch[0], 14)
        self.assertEqual(scratch[1], 29)
        self.assertEqual(scratch[8], 29)
        self.assertEqual(scratch[-1], 224)

        rgb565 = bytearray(2 * FRAME_SIDE * FRAME_SIDE)
        self.assertIsNone(sample_gray_scene_signature(rgb565, FRAME_SIDE, FRAME_SIDE, scratch))


if __name__ == "__main__":
    unittest.main()
