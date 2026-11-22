"""Corrected camera-pixel geometry tests; no board or study outputs are touched."""

import random
import sys
from pathlib import Path
import unittest


FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))

from camera_geometry import roi_clip_plan
from identity_contracts import (
    centered_roi_clamped,
    crop_for_boundary_policy,
    crop_xywh,
    fomo_center_camera_px,
)


class CameraGeometryV6Tests(unittest.TestCase):
    def test_saved_fomo_rect_is_already_in_camera_pixels(self):
        center = fomo_center_camera_px((115, 75, 27, 27), 240, 240)
        self.assertEqual(center, (128.5, 88.5))
        roi, reason = crop_for_boundary_policy(
            center, 120, 240, 240, "slide_inward_fixed_size_no_padding")
        self.assertIsNone(reason)
        self.assertEqual(roi, (68, 28, 120, 120))
        self.assertEqual(roi_clip_plan(roi, 240, 240)["black_padding_fraction"], 0.0)

    def test_corrected_host_module_uses_the_firmware_geometry_contract(self):
        import camera_geometry

        self.assertIs(camera_geometry.fomo_center_camera_px, fomo_center_camera_px)
        self.assertIs(camera_geometry.centered_roi_clamped, centered_roi_clamped)

    def test_boundary_policy_is_explicit_and_preserves_historical_rejection(self):
        center = (11.0, 88.5)
        old_roi, old_reason = crop_for_boundary_policy(
            center, 96, 240, 240, "reject_without_clamp_or_padding")
        self.assertEqual(old_roi, crop_xywh(11.0, 88.5, 96, 240, 240)[0])
        self.assertEqual(old_reason, "crop_outside_frame")

        new_roi, new_reason = crop_for_boundary_policy(
            center, 96, 240, 240, "slide_inward_fixed_size_no_padding")
        self.assertIsNone(new_reason)
        self.assertEqual(new_roi, (0, 40, 96, 96))
        self.assertEqual(roi_clip_plan(new_roi, 240, 240)["black_padding_fraction"], 0.0)

    def test_unknown_and_invalid_geometry_fail_closed(self):
        self.assertEqual(
            crop_for_boundary_policy((20, 20), 96, 240, 240, "padding_unspecified"),
            (None, "unknown_crop_boundary_policy"),
        )
        with self.assertRaises(ValueError):
            fomo_center_camera_px((220, 20, 30, 30), 240, 240)
        with self.assertRaises(ValueError):
            centered_roi_clamped((float("nan"), 20), 96, 240, 240)

    def test_ten_seeded_rect_and_roi_cases_stay_bounded(self):
        for seed in range(10):
            rng = random.Random(seed)
            x = rng.randrange(0, 180)
            y = rng.randrange(0, 180)
            width = rng.randrange(8, 60)
            height = rng.randrange(8, 60)
            rect = (x, y, width, height)
            center = fomo_center_camera_px(rect, 240, 240)
            self.assertEqual(center, (x + width / 2.0, y + height / 2.0))

            side = (48, 64, 80, 96, 120)[seed % 5]
            roi = centered_roi_clamped(center, side, 240, 240)
            rx, ry, rw, rh = roi
            self.assertEqual((rw, rh), (side, side))
            self.assertGreaterEqual(rx, 0)
            self.assertGreaterEqual(ry, 0)
            self.assertLessEqual(rx + rw, 240)
            self.assertLessEqual(ry + rh, 240)
            self.assertEqual(roi_clip_plan(roi, 240, 240)["black_padding_fraction"], 0.0)

    def test_xy_and_array_row_column_order_are_named(self):
        import camera_geometry

        self.assertEqual(camera_geometry.frame_row_col(17, 29), (29, 17))


if __name__ == "__main__":
    unittest.main()
