"""Offline QA for the separate host RGB YuNet/SFace reference mode."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from control_center import server
from control_center.host_rgb_reference import (
    EXPECTED_KNOWN_THRESHOLD,
    EXPECTED_MARGIN_THRESHOLD,
    EXPECTED_NUMPY_VERSION,
    EXPECTED_OPENCV_VERSION,
    analyze_jpeg,
    load_models,
)
from control_center.host_rgb_reference_runner import (
    HostRgbReferenceWorker,
    HostRgbReferenceWorkerError,
)
from control_center.server import ControlCenter


WORKSPACE = Path(__file__).resolve().parents[3]
STUDY = WORKSPACE / "work" / "project-research-v5"
BUSH_FIXTURE = (
    STUDY
    / "private"
    / "camera_optical_role_capture_v1"
    / "frames"
    / "000_K01_lfw_George_W_Bush_George_W_Bush_0044.jpg_rep02.jpg"
)
BLANK_FIXTURE = (
    STUDY / "private" / "camera_identity_survey_v2" / "scene_frames_cursor_clear" / "BLANK_preview.jpg"
)
BUSH_FIXTURE_SHA256 = "4f315270835b510f7ccca6d32bc0f86fab8476a01e5a292c827590adebf9fc9f"
BLANK_FIXTURE_SHA256 = "8ad37acc35d29c17666233c8bfda3dd7b6414a76a3ba09f1f2d7c4de2f5c0087"


class HostRgbReferenceFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models()

    def test_pinned_run_gallery_and_unknown_only_thresholds_reproduce(self):
        self.assertEqual(self.models["opencv_version"], EXPECTED_OPENCV_VERSION)
        self.assertEqual(self.models["numpy_version"], EXPECTED_NUMPY_VERSION)
        self.assertEqual(self.models["threshold_known_max_cosine"], EXPECTED_KNOWN_THRESHOLD)
        self.assertEqual(self.models["threshold_known_other_margin"], EXPECTED_MARGIN_THRESHOLD)
        self.assertEqual(self.models["unknown_calibration_people"], 28)
        self.assertEqual(self.models["gallery_source_groups"], {
            "K01": 11,
            "K02": 12,
            "K03": 12,
            "K04": 11,
            "K05": 12,
            "OTHER_train_identities": 19,
        })
        self.assertEqual(self.models["result"]["readiness"]["recognition_qualified"], False)
        self.assertIn("pretraining identities/data exposure", self.models["result"]["pretraining_caveat"])

    def test_unpinned_numpy_or_opencv_runtime_fails_closed(self):
        import cv2
        from control_center import host_rgb_reference

        with patch.object(host_rgb_reference.np, "__version__", "99.0"):
            with self.assertRaisesRegex(RuntimeError, "numpy_version_unpinned"):
                load_models()
        with patch.object(cv2, "__version__", "99.0"):
            with self.assertRaisesRegex(RuntimeError, "opencv_version_unpinned"):
                load_models()

    def test_saved_bush_k01_fixture_is_detected_but_calibrated_gates_reject(self):
        jpeg = BUSH_FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(jpeg).hexdigest(), BUSH_FIXTURE_SHA256)
        result = analyze_jpeg(jpeg, self.models)
        self.assertEqual(result["source"]["jpeg_sha256"], BUSH_FIXTURE_SHA256)
        self.assertEqual(result["detector"]["face_count"], 1)
        self.assertEqual(result["detector"]["status"], "one_face")
        self.assertEqual(len(result["detector"]["bbox_xywh_240_px"]), 4)
        self.assertGreater(result["detector"]["score"], 0.50)
        self.assertEqual(result["recognition"]["raw_top1_known_class"], "K01")
        self.assertFalse(result["recognition"]["known_cosine_gate"]["accepted"])
        self.assertFalse(result["recognition"]["known_vs_other_margin_gate"]["accepted"])
        self.assertEqual(result["recognition"]["known_cosine_gate"]["reason"], "max_known_cosine_below_threshold")
        self.assertEqual(result["recognition"]["known_vs_other_margin_gate"]["reason"], "known_minus_other_margin_below_threshold")
        self.assertIsNone(result["qualified_identity"])
        self.assertFalse(result["recognition_qualified"])
        crop = result["recognition"]["aligned_crop"]
        self.assertEqual(crop["shape"], [112, 112, 3])
        self.assertEqual(crop["color_order"], "BGR")
        self.assertTrue(crop["bytes_base64"])
        self.assertEqual(len(crop["sha256"]), 64)
        self.assertEqual(result["execution_location"], "host_reference")
        self.assertIn("NOT ON NICLA", result["evidence_scope"])

    def test_saved_blank_fixture_suppresses_identity_scores(self):
        jpeg = BLANK_FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(jpeg).hexdigest(), BLANK_FIXTURE_SHA256)
        result = analyze_jpeg(jpeg, self.models)
        self.assertEqual(result["detector"]["face_count"], 0)
        self.assertEqual(result["detector"]["status"], "no_face")
        self.assertEqual(result["recognition"]["status"], "not_run")
        self.assertEqual(result["recognition"]["reason"], "yunet_detected_no_face_at_score_threshold_0.50")
        self.assertIsNone(result["recognition"]["raw_top1_known_class"])
        self.assertIsNone(result["qualified_identity"])

    def test_wrong_geometry_and_invalid_jpeg_fail_closed(self):
        for payload, expected in (
            (BUSH_FIXTURE.read_bytes()[:120], "jpeg_decode_failed"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                analyze_jpeg(payload, self.models)
        import cv2
        import numpy as np

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        with self.assertRaisesRegex(ValueError, "240x240"):
            analyze_jpeg(encoded.tobytes(), self.models)

    def test_frontend_mode_is_separate_opt_in_and_never_badges_on_nicla(self):
        web_root = WORKSPACE / "FaceRecognition" / "control_center" / "web"
        html = (web_root / "index.html").read_text(encoding="utf-8")
        app_js = (web_root / "app.js").read_text(encoding="utf-8")

        class IdCollector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.attrs_by_id = {}

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                element_id = attrs.get("id")
                if element_id:
                    self.ids.append(element_id)
                    self.attrs_by_id[element_id] = attrs

        parser = IdCollector()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "HTML element IDs must be unique")
        required = {
            "hostRgbReferenceToggle", "hostRgbReferencePanel", "hostRgbReferenceResult",
            "hostRgbReferenceScores", "hostRgbReferenceCrop", "hostRgbReferenceTimings",
            "rgbReferenceOverlay", "hostRgbReferenceJpegHash", "hostRgbReferenceBbox",
        }
        self.assertTrue(required.issubset(parser.attrs_by_id), required - set(parser.attrs_by_id))
        self.assertIn("disabled", parser.attrs_by_id["hostRgbReferenceToggle"])
        self.assertIn("hidden", parser.attrs_by_id["hostRgbReferenceResult"])
        self.assertNotIn("ON NICLA · PREVIEW ONLY", app_js)
        self.assertIn("/api/v1/live/rgb-reference/enable", app_js)
        self.assertIn("/api/v1/live/rgb-reference/disable", app_js)
        self.assertIn("/api/v1/live/rgb-reference/run", app_js)
        self.assertIn("renderRgbReferenceOverlay", app_js)
        self.assertIn("no identity is emitted", html.lower())

    def test_separate_worker_process_loads_and_runs_pinned_positive_fixture(self):
        worker = HostRgbReferenceWorker(WORKSPACE)
        try:
            ready = worker.start()
            self.assertEqual(ready["status"], "ready")
            self.assertEqual(ready["ready"]["threshold_known_max_cosine"], EXPECTED_KNOWN_THRESHOLD)
            result = worker.infer(BUSH_FIXTURE.read_bytes())
            self.assertEqual(result["source"]["jpeg_sha256"], BUSH_FIXTURE_SHA256)
            self.assertEqual(result["recognition"]["raw_top1_known_class"], "K01")
            self.assertIsNone(result["qualified_identity"])
        finally:
            worker.stop()
        self.assertEqual(worker.status, "off")


class HostRgbReferenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.app = ControlCenter()
        self.client_id = "rgb_reference_offline"
        self.client = self.app.client(self.client_id)
        self.client["live_attached"] = True
        self.client["mode"] = "live"
        self.client["host_rgb_reference_enabled"] = True
        self.client["observation_id"] = 27
        self.jpeg = b"\xff\xd8" + b"fixture-bytes" * 12 + b"\xff\xd9"
        self.client["live_snapshot"] = self.jpeg
        self.client["live_snapshot_received"] = time.monotonic()

        class Worker:
            status = "ready"

            def __init__(inner, override_hash=None, during=None):
                inner.override_hash = override_hash
                inner.during = during
                inner.received = []

            def infer(inner, jpeg):
                inner.received.append(jpeg)
                if inner.during:
                    inner.during()
                return {
                    "source": {
                        "jpeg_sha256": inner.override_hash or hashlib.sha256(jpeg).hexdigest()
                    },
                    "recognition": {},
                }

        self.worker_type = Worker
        self.app.host_rgb_reference_worker = Worker()
        self.app.refresh_live_status = lambda: {"connected": True, "age_s": 0.1}

    def test_result_is_bound_to_host_observation_and_cached_jpeg_sha(self):
        result = self.app.run_host_rgb_reference(self.client_id, 27)
        self.assertEqual(result["host_observation_index"], 27)
        self.assertEqual(result["source"]["host_observation_index"], 27)
        self.assertEqual(result["source"]["jpeg_sha256"], hashlib.sha256(self.jpeg).hexdigest())
        self.assertIn("no device frame ID", result["source"]["binding"])
        self.assertIn("NOT ON NICLA", result["qualification"])

    def test_default_mode_does_not_start_rgb_worker(self):
        app = ControlCenter()
        self.assertEqual(app.host_rgb_reference_status("rgb_default")["status"], "off")
        self.assertIsNone(app.host_rgb_reference_worker)
        with self.assertRaisesRegex(RuntimeError, "attach to the existing preview"):
            app.enable_host_rgb_reference("rgb_default")
        self.assertIsNone(app.host_rgb_reference_worker)

    def test_wrong_host_observation_suppresses_inference(self):
        with self.assertRaisesRegex(RuntimeError, "host_observation_superseded"):
            self.app.run_host_rgb_reference(self.client_id, 26)
        self.assertEqual(self.app.host_rgb_reference_worker.received, [])

    def test_stale_preview_before_inference_suppresses_worker(self):
        self.app.refresh_live_status = lambda: {"connected": True, "age_s": 2.1}
        with self.assertRaisesRegex(RuntimeError, "stale_before_host_rgb_reference"):
            self.app.run_host_rgb_reference(self.client_id, 27)
        self.assertEqual(self.app.host_rgb_reference_worker.received, [])

    def test_stale_transition_after_inference_suppresses_result(self):
        statuses = iter((
            {"connected": True, "age_s": 0.1},
            {"connected": True, "age_s": 2.1},
        ))
        self.app.refresh_live_status = lambda: next(statuses)
        with self.assertRaisesRegex(RuntimeError, "stale_after_host_rgb_reference"):
            self.app.run_host_rgb_reference(self.client_id, 27)

    def test_observation_superseded_during_inference_suppresses_result(self):
        self.app.host_rgb_reference_worker = self.worker_type(
            during=lambda: self.client.update(observation_id=28)
        )
        with self.assertRaisesRegex(RuntimeError, "superseded_during_rgb_reference"):
            self.app.run_host_rgb_reference(self.client_id, 27)

    def test_opt_out_during_inference_suppresses_result(self):
        self.app.host_rgb_reference_worker = self.worker_type(
            during=lambda: self.client.update(host_rgb_reference_enabled=False)
        )
        with self.assertRaisesRegex(RuntimeError, "disabled_during_inference"):
            self.app.run_host_rgb_reference(self.client_id, 27)

    def test_worker_jpeg_hash_mismatch_suppresses_result(self):
        self.app.host_rgb_reference_worker = self.worker_type(override_hash="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "jpeg_binding_mismatch"):
            self.app.run_host_rgb_reference(self.client_id, 27)

    def test_diagnostic_requires_opt_in(self):
        self.client["host_rgb_reference_enabled"] = False
        with self.assertRaisesRegex(RuntimeError, "host_rgb_reference_not_opted_in"):
            self.app.run_host_rgb_reference(self.client_id, 27)

    def test_concurrent_worker_call_is_rejected_instead_of_queued(self):
        worker = HostRgbReferenceWorker(WORKSPACE)
        worker.status = "ready"
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock():
            with worker.lock:
                acquired.set()
                release.wait(timeout=3)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(acquired.wait(timeout=1))
        try:
            self.assertEqual(worker.status_record()["status"], "busy")
            with self.assertRaisesRegex(HostRgbReferenceWorkerError, "worker_busy"):
                worker.infer(b"unused")
        finally:
            release.set()
            holder.join(timeout=1)
            self.assertFalse(holder.is_alive())


if __name__ == "__main__":
    unittest.main()
