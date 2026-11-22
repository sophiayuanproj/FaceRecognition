import math
import sys
import unittest
from pathlib import Path

FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))

from policy_controller import (  # noqa: E402
    MAX_MODEL_ESCALATIONS_PER_MINUTE,
    MAX_SAMPLE_GAP_MS,
    PolicyController,
    reason_coded_quality,
)


class PolicyControllerV6Tests(unittest.TestCase):
    def test_fixed_and_adaptive_cadence_bounds(self):
        expected = {"H": 250, "G": 250, "S": 250, "A-acquire": 250, "L": 2000, "A-model": 1000}
        for policy, interval in expected.items():
            with self.subTest(policy=policy):
                self.assertEqual(PolicyController(policy).next_interval(False, 0)[0], interval)
        for policy in ("A-sense", "A-full"):
            with self.subTest(policy=policy):
                controller = PolicyController(policy)
                interval, reason = controller.next_interval(True, 0)
                self.assertEqual(reason, "periodic_stationary_check")
                self.assertEqual(interval, 1500)
                controller.update_motion(160)
                controller.update_motion(160)
                self.assertEqual(controller.next_interval(True, 0)[0], 250)
                self.assertEqual(controller.next_interval(True, 1)[0], 250)
                self.assertLessEqual(interval, MAX_SAMPLE_GAP_MS)
        scene = PolicyController("A-sense")
        self.assertEqual(scene.next_interval(True, 0), (1500, "periodic_stationary_check"))
        self.assertEqual(scene.next_interval(True, 1), (500, "face_candidate_check"))
        self.assertEqual(scene.next_interval(False, 0), (250, "quality_recapture"))

    def test_a_sense_gates_identity_during_wearer_motion_but_full_policy_does_not(self):
        quality_gate = PolicyController("A-sense")
        self.assertEqual(quality_gate.should_run_identity(True), (True, "identity_enabled"))
        quality_gate.update_motion(160)
        quality_gate.update_motion(160)
        self.assertEqual(quality_gate.should_run_identity(True), (False, "wearer_motion_identity_gate"))
        self.assertEqual(quality_gate.should_run_identity(False), (False, "missing_or_low_quality_evidence"))
        joint = PolicyController("A-full")
        joint.update_motion(160)
        joint.update_motion(160)
        self.assertEqual(joint.should_run_identity(True), (True, "identity_enabled"))

    def test_acquisition_only_skips_detector_identity_and_model_calls(self):
        controller = PolicyController("A-acquire")
        self.assertEqual(controller.next_interval(True, 0)[0], 250)
        self.assertEqual(controller.should_run_identity(True), (False, "acquisition_only_policy"))
        self.assertEqual(controller.select_models(True), ([], "acquisition_only_policy"))
        controller.note_capture(0)
        controller.note_result(5, "acquisition_only")
        self.assertEqual(controller.counters["captures"], 1)
        self.assertEqual(controller.counters["decisions"], 1)

    def test_motion_hysteresis_and_missing_or_nonfinite_imu(self):
        controller = PolicyController("G")
        self.assertIsNone(controller.motion_score([float("nan"), 0, 0], [0, 0, 0]))
        self.assertEqual(controller.update_motion(float("nan"))["reason"], "imu_missing")
        self.assertFalse(controller.update_motion(160)["motion_active"])
        self.assertTrue(controller.update_motion(160)["motion_active"])
        self.assertTrue(controller.update_motion(None)["motion_active"])
        self.assertTrue(controller.update_motion(75)["motion_active"])
        self.assertTrue(controller.update_motion(75)["motion_active"])
        self.assertFalse(controller.update_motion(75)["motion_active"])

    def test_quality_rejects_missing_blank_extreme_and_nonfinite(self):
        self.assertEqual(reason_coded_quality(None, None), (False, "quality_statistics_missing"))
        self.assertEqual(reason_coded_quality(2, 30), (False, "too_dim"))
        self.assertEqual(reason_coded_quality(250, 30), (False, "saturated"))
        self.assertEqual(reason_coded_quality(80, 1), (False, "low_contrast_or_blank"))
        self.assertEqual(reason_coded_quality(float("nan"), 30),
                         (False, "quality_statistics_nonfinite"))
        self.assertTrue(reason_coded_quality(80, 30)[0])

    def test_camera_sleep_disabled_and_quality_unknown_never_sleeps(self):
        default = PolicyController("A-full")
        default.next_interval(True, 0)
        for _ in range(5):
            self.assertFalse(default.should_sleep_camera(True, 0)[0])
        enabled = PolicyController("A-full", allow_camera_sleep=True)
        enabled.next_interval(True, 0)
        self.assertFalse(enabled.should_sleep_camera(None, 0)[0])
        self.assertFalse(enabled.should_sleep_camera(False, 0)[0])
        self.assertEqual(enabled.no_face_streak, 0)
        self.assertFalse(enabled.should_sleep_camera(True, 1)[0])
        self.assertEqual(enabled.no_face_streak, 0)
        self.assertFalse(enabled.should_sleep_camera(True, 0)[0])
        self.assertFalse(enabled.should_sleep_camera(True, 0)[0])
        self.assertTrue(enabled.should_sleep_camera(True, 0)[0])
        self.assertEqual(enabled.interval_ms, 1500)
        enabled.note_wake(1500)
        self.assertEqual(enabled.counters["camera_wakes"], 1)
        enabled.update_motion(160)
        enabled.update_motion(160)
        self.assertFalse(enabled.should_sleep_camera(True, 0)[0])

    def test_model_escalation_is_quality_gated_bounded_and_finite(self):
        controller = PolicyController("A-model")
        self.assertEqual(controller.select_models(False, 0.8, 0.8)[0], [])
        invalid, reason = controller.select_models(True, float("nan"), 0.8, now_ms=0)
        self.assertEqual(invalid, ["small"])
        self.assertEqual(reason, "invalid_small_model_score_no_escalation")
        escalated = []
        for second in range(8):
            calls, _ = controller.select_models(True, 0.8, 0.8, now_ms=second * 1000)
            escalated.append(calls == ["small", "large"])
        self.assertEqual(sum(escalated), MAX_MODEL_ESCALATIONS_PER_MINUTE)
        self.assertEqual(controller.counters["model_escalations"], MAX_MODEL_ESCALATIONS_PER_MINUTE)
        calls, _ = controller.select_models(True, 0.2, 0.8, now_ms=61000)
        self.assertEqual(calls, ["small"])

    def test_temporal_confirmation_identity_unknown_and_invalid_evidence(self):
        controller = PolicyController("G")
        self.assertEqual(controller.temporal_decision("known", "K01", now_ms=1000)[0], "uncertain")
        self.assertEqual(controller.temporal_decision("known", "K01", now_ms=1250)[0], "known")
        self.assertEqual(controller.temporal_decision("unknown", now_ms=1500)[0], "known")
        self.assertEqual(controller.temporal_decision("unknown", now_ms=1750)[0], "unknown")
        status, reason = controller.temporal_decision("known", "K01", False, now_ms=2000)
        self.assertEqual((status, reason), ("uncertain", "evidence_not_valid_for_temporal_vote"))
        self.assertEqual(controller.temporal, [])

    def test_temporal_votes_expire_after_stale_gap_and_tick_wrap_is_supported(self):
        controller = PolicyController("A-full", ticks_diff=lambda end, start: ((end - start + 128) % 256) - 128)
        controller.temporal_decision("known", "K02", now_ms=250)
        status, reason = controller.temporal_decision("known", "K02", now_ms=10)
        self.assertEqual(status, "known")  # elapsed 16ms through wrap
        self.assertEqual(reason, "two_of_three_confirmation")

        stale = PolicyController("G")
        stale.temporal_decision("known", "K02", now_ms=20)
        status, reason = stale.temporal_decision("known", "K02", now_ms=20 + MAX_SAMPLE_GAP_MS + 1)
        self.assertEqual(status, "uncertain")
        self.assertEqual(reason, "awaiting_two_of_three_confirmation")
        self.assertEqual(len(stale.temporal), 1)
        legacy = PolicyController("G")
        legacy.temporal_decision("known", "K02")
        self.assertEqual(legacy.temporal_decision("known", "K02", now_ms=100)[0], "uncertain")

    def test_heartbeat_clears_stale_votes_and_counts_one_timeout_per_episode(self):
        controller = PolicyController("G")
        controller.note_wake(0)
        controller.temporal_decision("known", "K03", now_ms=0)
        first = controller.heartbeat(MAX_SAMPLE_GAP_MS + 1)
        second = controller.heartbeat(MAX_SAMPLE_GAP_MS + 20)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(controller.temporal, [])
        self.assertEqual(controller.counters["timeouts"], 1)
        controller.note_result(MAX_SAMPLE_GAP_MS + 30, "uncertain")
        controller.heartbeat(2 * MAX_SAMPLE_GAP_MS + 31)
        self.assertEqual(controller.counters["timeouts"], 2)

    def test_late_identity_is_suppressed(self):
        controller = PolicyController("L")
        controller.note_wake(0)
        result, timeout = controller.apply_response_deadline(
            {"status": "known", "identity_id": "K04", "cosine_similarity": 0.99},
            MAX_SAMPLE_GAP_MS + 1)
        self.assertEqual(timeout["reason"], "max_response_interval_timeout")
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["identity_id"])
        self.assertNotIn("cosine_similarity", result)


if __name__ == "__main__":
    unittest.main()
