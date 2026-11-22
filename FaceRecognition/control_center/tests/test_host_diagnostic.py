"""Offline tests for the opt-in host-only JPEG reconstruction path."""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from control_center import server
from control_center.diagnostic_runner import HostDiagnosticWorker, HostWorkerError
from control_center.host_diagnostic import (
    CLASSIFIER_SHA256,
    DETECTOR_SHA256,
    MANIFEST_SHA256,
    analyze_jpeg,
    load_models,
    pillow_bilinear_grayscale_u8,
)
from control_center.server import ControlCenter


class FakeInterpreter:
    def __init__(self, output_detail, output, error=None):
        self.output_detail = output_detail
        self.output = np.asarray(output, dtype=np.int8)
        self.error = error
        self.input = None
        self.invoked = False

    def set_tensor(self, index, tensor):
        self.input = np.asarray(tensor).copy()

    def invoke(self):
        self.invoked = True
        if self.error:
            raise self.error

    def get_tensor(self, index):
        return self.output.copy()


def synthetic_jpeg(*, size=(240, 240), brightness=True) -> bytes:
    width, height = size
    if brightness:
        yy, xx = np.mgrid[:height, :width]
        pixels = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
    else:
        pixels = np.zeros((height, width), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, mode="L").convert("RGB").save(output, format="JPEG", quality=91)
    return output.getvalue()


def fake_models(cells=((6, 6),), *, classifier_scores=(80, -10, -10, -10, -10, -10)):
    detector_output = np.full((1, 12, 12, 2), -128, dtype=np.int8)
    for gy, gx in cells:
        detector_output[0, gy, gx, 1] = 75
    classifier_output = np.asarray([classifier_scores], dtype=np.int8)
    return {
        "detector": FakeInterpreter({"quantization": (1 / 256, -128)}, detector_output),
        "classifier": FakeInterpreter({"quantization": (0.026111828163266182, -10)}, classifier_output),
        "detector_input": {"index": 0},
        "detector_output": {"index": 1, "quantization": (1 / 256, -128)},
        "classifier_input": {"index": 0},
        "classifier_output": {"index": 1, "quantization": (0.026111828163266182, -10)},
        "detector_threshold": 0.7,
        "classifier_threshold": 0.505630741905405,
        "uncertainty_band": 0.025,
        "class_order": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
        "hashes": {"detector_gray_fomo_int8.tflite": "d" * 64, "identity_classifier_a025_int8.tflite": "c" * 64},
        "runtime": "offline-test-double",
        "runtime_config_sha256": "r" * 64,
        "manifest_runtime_config_sha256": "m" * 64,
        "integration_manifest_sha256": "h" * 64,
        "manifest_runtime_config_matches": False,
    }


def decode_b64(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


class HostReconstructionContractTests(unittest.TestCase):
    def test_one_region_reconstructs_jpeg_bound_tensor_crop_scores_and_timings(self):
        jpeg = synthetic_jpeg()
        models = fake_models()
        result = analyze_jpeg(jpeg, models)

        self.assertEqual(result["evidence_scope"], "HOST RECONSTRUCTION FROM LOSSY PREVIEW JPEG; NOT FRAME-IDENTICAL OR DEVICE-EQUIVALENT")
        self.assertTrue(result["experimental"])
        self.assertIsNone(result["qualified_identity"])
        self.assertNotIn("identity_id", result["identity"])
        self.assertEqual(result["source"]["jpeg_sha256"], hashlib.sha256(jpeg).hexdigest())
        self.assertFalse(result["manifest_runtime_config_matches"])
        self.assertTrue(result["warnings"])

        detector_tensor = decode_b64(result["detector"]["input_tensor"]["bytes_base64"])
        self.assertEqual(len(detector_tensor), 96 * 96)
        self.assertEqual(hashlib.sha256(detector_tensor).hexdigest(), result["detector"]["input_tensor"]["sha256"])
        with Image.open(io.BytesIO(jpeg)) as image:
            gray = image.convert("L").tobytes()
        expected_u8 = bytearray(96 * 96)
        pillow_bilinear_grayscale_u8(gray, 240, 240, (0, 0, 240, 240), expected_u8)
        expected_detector = (np.frombuffer(expected_u8, dtype=np.uint8).astype(np.int16) - 128).astype(np.int8).tobytes()
        self.assertEqual(detector_tensor, expected_detector)

        detector = result["detector"]
        self.assertEqual(detector["status"], "regions_found")
        self.assertEqual(len(detector["regions"]), 1)
        detector_output_bytes = decode_b64(detector["raw_output_tensor"]["bytes_base64"])
        self.assertEqual(len(detector_output_bytes), 12 * 12 * 2)
        self.assertEqual(hashlib.sha256(detector_output_bytes).hexdigest(), detector["raw_output_tensor"]["sha256"])
        region = detector["regions"][0]
        x, y, w, h = region["region_xywh_window_px"]
        self.assertTrue(0 <= x < 240 and 0 <= y < 240 and w > 0 and h > 0)
        cx, cy = region["center_xy_window_px"]
        expected_crop_x = int(np.floor(cx - 48 + 0.5))
        expected_crop_y = int(np.floor(cy - 48 + 0.5))
        crop = result["identity"]["crop"]
        self.assertEqual(crop["xywh_window_px"], [expected_crop_x, expected_crop_y, 96, 96])
        crop_bytes = decode_b64(crop["bytes_base64"])
        self.assertEqual(len(crop_bytes), 96 * 96)
        self.assertEqual(hashlib.sha256(crop_bytes).hexdigest(), crop["sha256"])

        scores = result["identity"]["raw_scores_int8"]
        self.assertEqual(scores, [80, -10, -10, -10, -10, -10])
        classifier_tensor = decode_b64(result["identity"]["input_tensor"]["bytes_base64"])
        self.assertEqual(len(classifier_tensor), 96 * 96)
        self.assertEqual(hashlib.sha256(classifier_tensor).hexdigest(), result["identity"]["input_tensor"]["sha256"])
        classifier_output = decode_b64(result["identity"]["raw_output_tensor"]["bytes_base64"])
        self.assertEqual(len(classifier_output), 6)
        self.assertEqual(hashlib.sha256(classifier_output).hexdigest(), result["identity"]["raw_output_tensor"]["sha256"])
        np.testing.assert_array_equal(np.frombuffer(classifier_output, dtype=np.int8), scores)
        self.assertEqual(len(result["identity"]["raw_logits"]), 6)
        self.assertEqual(result["identity"]["class_order"], ["K01", "K02", "K03", "K04", "K05", "OTHER"])
        self.assertEqual(result["identity"]["candidate_class"], "K01")
        self.assertIn("experimental", result["identity"]["threshold_comparison"])
        self.assertIn("threshold", result["identity"])
        self.assertIn("fomo_postprocess", result["timings_ms"])
        self.assertIn("classifier_inference", result["timings_ms"])
        self.assertIn("total", result["timings_ms"])
        np.testing.assert_array_equal(models["detector"].input.shape, (1, 96, 96, 1))
        np.testing.assert_array_equal(models["classifier"].input.shape, (1, 96, 96, 1))

    def test_no_regions_suppresses_classifier_and_candidate(self):
        models = fake_models(cells=())
        result = analyze_jpeg(synthetic_jpeg(), models)
        self.assertEqual(result["detector"]["status"], "no_face_regions")
        self.assertEqual(result["identity"]["status"], "not_run")
        self.assertIsNone(result["identity"]["candidate_class"])
        self.assertFalse(models["classifier"].invoked)

    def test_multiple_regions_suppress_classifier(self):
        models = fake_models(cells=((2, 2), (9, 9)))
        result = analyze_jpeg(synthetic_jpeg(), models)
        self.assertEqual(len(result["detector"]["regions"]), 2)
        self.assertEqual(result["identity"]["reason"], "multiple_fomo_regions")
        self.assertFalse(models["classifier"].invoked)

    def test_out_of_frame_crop_is_rejected_without_clamping(self):
        models = fake_models(cells=((0, 0),))
        result = analyze_jpeg(synthetic_jpeg(), models)
        crop = result["identity"]["crop"]
        self.assertEqual(result["identity"]["reason"], "crop_outside_frame")
        self.assertIsNone(crop["bytes_base64"])
        self.assertFalse(models["classifier"].invoked)

    def test_quality_failure_suppresses_classifier(self):
        models = fake_models(cells=((6, 6),))
        result = analyze_jpeg(synthetic_jpeg(brightness=False), models)
        self.assertEqual(result["identity"]["status"], "rejected")
        self.assertIn("crop_quality_", result["identity"]["reason"])
        self.assertEqual(result["identity"]["crop"]["quality_gate"], "reject")
        self.assertFalse(models["classifier"].invoked)

    def test_historical_threshold_can_reject_without_qualified_identity(self):
        models = fake_models(cells=((6, 6),), classifier_scores=(-10, -10, -10, -10, -10, 80))
        result = analyze_jpeg(synthetic_jpeg(), models)
        self.assertEqual(result["identity"]["status"], "experimental_reject")
        self.assertEqual(result["identity"]["reason"], "other_logit_argmax")
        self.assertEqual(len(result["identity"]["raw_scores_int8"]), 6)
        self.assertIsNone(result["qualified_identity"])
        self.assertNotIn("identity_id", result["identity"])

    def test_hash_pinned_workspace_models_run_offline_on_synthetic_jpeg(self):
        models = load_models()
        self.assertEqual(models["hashes"]["detector_gray_fomo_int8.tflite"], DETECTOR_SHA256)
        self.assertEqual(models["hashes"]["identity_classifier_a025_int8.tflite"], CLASSIFIER_SHA256)
        self.assertTrue(models["manifest_runtime_config_matches"])
        self.assertEqual(models["integration_manifest_sha256"], MANIFEST_SHA256)
        result = analyze_jpeg(synthetic_jpeg(), models)
        self.assertEqual(result["source"]["width"], 240)
        self.assertEqual(result["source"]["height"], 240)
        self.assertEqual(result["evidence_scope"], "HOST RECONSTRUCTION FROM LOSSY PREVIEW JPEG; NOT FRAME-IDENTICAL OR DEVICE-EQUIVALENT")
        self.assertIsNone(result["qualified_identity"])
        self.assertEqual(result["warnings"], [])
        self.assertIn("no third-party training-data license is asserted", result["model_use_scope"])

    def test_host_manifest_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            with patch("control_center.host_diagnostic.MANIFEST_PATH", manifest_path):
                with self.assertRaisesRegex(ValueError, "host_manifest_hash_mismatch"):
                    load_models()

    def test_invalid_jpeg_and_geometry_fail_closed(self):
        for payload, reason in ((b"not jpeg" * 20, "jpeg_decode_failed"),
                                 (synthetic_jpeg(size=(320, 240)), "geometry_mismatch")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                analyze_jpeg(payload, fake_models())

    def test_invalid_model_output_geometry_fails_closed(self):
        models = fake_models()
        models["detector"].output = np.zeros((1, 11, 12, 2), dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "detector_output_geometry_mismatch"):
            analyze_jpeg(synthetic_jpeg(), models)


class HostDiagnosticBindingTests(unittest.TestCase):
    def setUp(self):
        self.app = ControlCenter()
        self.client_id = "offline_host_diagnostic"
        self.client = self.app.client(self.client_id)
        self.client["live_attached"] = True
        self.client["mode"] = "live"
        self.client["host_diagnostic_enabled"] = True
        self.client["observation_id"] = 14
        self.jpeg = synthetic_jpeg()
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
                digest = hashlib.sha256(jpeg).hexdigest()
                return {"source": {"jpeg_sha256": inner.override_hash or digest}, "identity": {}}

        self.worker_type = Worker
        self.app.host_diagnostic_worker = Worker()
        self.app.refresh_live_status = lambda: {"connected": True, "age_s": 0.1}

    def test_default_mode_does_not_create_or_start_worker(self):
        app = ControlCenter()
        client = app.client("default_preview_only")
        self.assertEqual(app.host_diagnostic_status("default_preview_only")["status"], "off")
        self.assertFalse(client["host_diagnostic_enabled"])
        self.assertIsNone(app.host_diagnostic_worker)
        with self.assertRaisesRegex(RuntimeError, "attach to the existing preview"):
            app.enable_host_diagnostic("default_preview_only")
        self.assertIsNone(app.host_diagnostic_worker)

    def test_result_is_bound_to_one_host_observation_and_exact_jpeg_hash(self):
        result = self.app.run_host_diagnostic(self.client_id, 14)
        self.assertEqual(result["host_observation_index"], 14)
        self.assertEqual(result["source"]["host_observation_index"], 14)
        self.assertEqual(result["source"]["jpeg_sha256"], hashlib.sha256(self.jpeg).hexdigest())
        self.assertIn("no device frame ID", result["source"]["binding"])
        self.assertIn("UNQUALIFIED", result["qualification"])

    def test_wrong_or_superseded_observation_is_rejected_before_inference(self):
        with self.assertRaisesRegex(RuntimeError, "host_observation_superseded"):
            self.app.run_host_diagnostic(self.client_id, 13)
        self.assertEqual(self.app.host_diagnostic_worker.received, [])

    def test_preview_age_stale_before_worker_suppresses_inference(self):
        self.app.refresh_live_status = lambda: {"connected": True, "age_s": 2.1}
        with self.assertRaisesRegex(RuntimeError, "stale_before_host_inference"):
            self.app.run_host_diagnostic(self.client_id, 14)
        self.assertEqual(self.app.host_diagnostic_worker.received, [])

    def test_preview_age_stale_after_worker_suppresses_result(self):
        statuses = iter(({"connected": True, "age_s": 0.1}, {"connected": True, "age_s": 2.1}))
        self.app.refresh_live_status = lambda: next(statuses)
        with self.assertRaisesRegex(RuntimeError, "stale_after_host_inference"):
            self.app.run_host_diagnostic(self.client_id, 14)

    def test_snapshot_superseded_during_worker_suppresses_result(self):
        self.app.host_diagnostic_worker = self.worker_type(during=lambda: self.client.update(observation_id=15))
        with self.assertRaisesRegex(RuntimeError, "superseded_during_inference"):
            self.app.run_host_diagnostic(self.client_id, 14)

    def test_opt_out_during_worker_suppresses_result(self):
        self.app.host_diagnostic_worker = self.worker_type(during=lambda: self.client.update(host_diagnostic_enabled=False))
        with self.assertRaisesRegex(RuntimeError, "disabled_during_inference"):
            self.app.run_host_diagnostic(self.client_id, 14)

    def test_worker_jpeg_hash_mismatch_suppresses_result(self):
        self.app.host_diagnostic_worker = self.worker_type(override_hash="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "jpeg_binding_mismatch"):
            self.app.run_host_diagnostic(self.client_id, 14)

    def test_diagnostic_requires_explicit_opt_in(self):
        self.client["host_diagnostic_enabled"] = False
        with self.assertRaisesRegex(RuntimeError, "not_opted_in"):
            self.app.run_host_diagnostic(self.client_id, 14)


class HostDiagnosticCapacityTests(unittest.TestCase):
    def test_concurrent_worker_request_is_rejected_without_queueing(self):
        worker = HostDiagnosticWorker(Path.cwd())
        worker.status = "ready"
        worker.ready = {"runtime": "test"}
        acquired = threading.Event()
        release = threading.Event()

        def hold_worker_lock():
            with worker.lock:
                acquired.set()
                release.wait(timeout=3)

        holder = threading.Thread(target=hold_worker_lock)
        holder.start()
        self.assertTrue(acquired.wait(timeout=1))
        try:
            self.assertEqual(worker.status_record()["status"], "busy")
            with self.assertRaisesRegex(HostWorkerError, "worker_busy"):
                worker.infer(b"unused")
        finally:
            release.set()
            holder.join(timeout=1)
            self.assertFalse(holder.is_alive())

    def test_concurrent_preview_capture_is_rejected_without_network_read(self):
        app = ControlCenter()
        app.capture_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "preview_capture_busy"):
                app.capture_live_snapshot("bounded_capture")
        finally:
            app.capture_lock.release()


if __name__ == "__main__":
    unittest.main()
