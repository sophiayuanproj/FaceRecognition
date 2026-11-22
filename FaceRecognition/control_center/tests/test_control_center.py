import hashlib
import json
import unittest
import io
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from PIL import Image

from control_center import server
from control_center.server import ControlCenter, _input_dtype_label, _loopback_preview_url


class LoopbackAdapterTests(unittest.TestCase):
    def test_accepts_existing_preview_root(self):
        self.assertEqual(_loopback_preview_url("http://127.0.0.1:8766/"), "http://127.0.0.1:8766")

    def test_rejects_remote_preview(self):
        with self.assertRaises(ValueError):
            _loopback_preview_url("http://192.0.2.10:8766/")

    def test_rejects_unrelated_local_service(self):
        with self.assertRaises(ValueError):
            _loopback_preview_url("http://127.0.0.1:8767/")

    def test_rejects_url_with_credentials_or_path(self):
        for value in ("http://user@127.0.0.1:8766/", "http://127.0.0.1:8766/admin"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _loopback_preview_url(value)


class TensorHeaderTests(unittest.TestCase):
    def test_openmv_typecodes_are_reported_as_readable_tensor_dtypes(self):
        for raw, expected in (("98", "int8"), ("b", "int8"), ("66", "uint8"),
                              ("104", "int16"), ("102", "float32"), ("int8", "int8")):
            with self.subTest(raw=raw):
                self.assertEqual(_input_dtype_label(raw), expected)

    def test_unknown_tensor_dtype_is_not_silently_relabelled(self):
        self.assertEqual(_input_dtype_label("complex"), "complex")


class DeviceMacReferenceBindingTests(unittest.TestCase):
    def test_worker_result_must_match_the_broker_jpeg(self):
        app = ControlCenter()
        self.addCleanup(app.close)
        frame = b"camera frame"
        digest = hashlib.sha256(frame).hexdigest()
        app.device_broker = Mock()
        app.device_broker.get_frame.return_value = (
            frame, {"boot_id": "boot-a", "frame_id": 9, "sha256": digest}
        )
        worker = Mock()
        app.device_reference_worker = worker
        worker.infer.return_value = {"source": {"jpeg_sha256": "wrong"}}
        with self.assertRaisesRegex(RuntimeError, "Mac result frame hash mismatch"):
            app.run_device_host_reference("boot-a", 9)
        worker.infer.return_value = {"source": {"jpeg_sha256": digest}}
        result = app.run_device_host_reference("boot-a", 9)
        self.assertEqual(result["source"]["device_boot_id"], "boot-a")
        self.assertEqual(result["source"]["device_frame_id"], 9)
        worker.infer.assert_called_with(frame)
        app.device_broker.command.assert_not_called()


class LiveAttachFailureTests(unittest.TestCase):
    def test_failed_attach_returns_client_to_clean_replay_state(self):
        app = ControlCenter()
        client = app.client("failed_attach")
        client.update({
            "mode": "live",
            "live_attached": True,
            "live_snapshot": b"stale cached JPEG",
            "live_snapshot_received": 123.0,
            "host_diagnostic_enabled": True,
            "host_rgb_reference_enabled": True,
        })
        app.refresh_live_status = lambda: {"connected": False, "reason": "preview refused connection"}

        attached, result = app.connect_live("failed_attach")

        self.assertFalse(attached)
        self.assertIn("no serial connection was opened", result["error"])
        self.assertFalse(client["live_attached"])
        self.assertEqual(client["mode"], "replay")
        self.assertIsNone(client["live_snapshot"])
        self.assertIsNone(client["live_snapshot_received"])
        self.assertFalse(client["host_diagnostic_enabled"])
        self.assertFalse(client["host_rgb_reference_enabled"])


class ReplayContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nicla-synthetic-replay-")
        self.addCleanup(temporary.cleanup)
        self.fixture_dir = Path(temporary.name)
        result_path = self.fixture_dir / "synthetic-result.json"
        cases = {}
        rows = []
        for case in server.CASE_ORDER:
            frame_path = self.fixture_dir / f"{case}.jpg"
            Image.new("L", (8, 8), 128).save(frame_path, format="JPEG")
            cases[case] = frame_path
            rows.append({
                "case": case,
                "fomo_accepted_clusters": 0 if case == "BLANK" else 1,
                "fomo_raw_clusters": 0 if case == "BLANK" else 1,
                "decision": "NO_FACE" if case == "BLANK" else "OTHER",
                "predicted_class": None if case == "BLANK" else "OTHER",
                "elapsed_ms": 100,
            })
        result_path.write_text(json.dumps({
            "schema": "nicla-live-camera-identity-verified-set-v2",
            "candidate": {"encoder_sha256": "a" * 64, "head_sha256": "b" * 64},
            "per_condition": rows,
        }), encoding="utf-8")
        for key, value in (("RUN_RESULT", result_path), ("CASE_IMAGES", cases)):
            patcher = patch.object(server, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = ControlCenter()
        self.addCleanup(self.app.close)

    def test_replay_has_explicit_unavailable_input(self):
        event = self.app.replay_case("K01")
        self.assertEqual(event["connection"]["execution_location"], "recorded_replay")
        self.assertIsNone(event["model_input"]["sha256"])
        self.assertEqual(event["model_input"]["source"], "none")
        self.assertTrue(event["model_input"]["warnings"])

    def test_replay_keeps_decision_and_case_frame_identity(self):
        for index, case in enumerate(("K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK"), 1):
            with self.subTest(case=case):
                event = self.app.replay_case(case)
                self.assertEqual(event["frame_id"], index)
                self.assertEqual(event["frame"]["frame_id"], index)
                self.assertEqual(event["decision"]["frame_id"], index)

    def test_replay_does_not_invent_scores_or_power(self):
        event = self.app.replay_case("K05")
        self.assertEqual(event["decision"]["scores"], {})
        self.assertEqual(event["decision"]["score_semantics"], "none")
        self.assertNotIn("power_w", event["telemetry"])
        self.assertNotEqual(event["telemetry"].get("power"), 0)

    def test_replay_scene_image_uses_the_configured_fixture_set(self):
        event = self.app.replay_case("OTHER")
        path = server.CASE_IMAGES[event["case"]]
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent, self.fixture_dir)

    def test_live_without_attachment_stays_no_decision(self):
        event = self.app.live_state()
        self.assertEqual(event["connection"]["state"], "disconnected")
        self.assertEqual(event["decision"]["status"], "not_run")
        self.assertEqual(event["detection"]["status"], "not_run")
        self.assertIsNone(event["telemetry"]["inference_hz"])

    def test_viewer_modes_are_isolated(self):
        one = self.app.client("viewer_one")
        two = self.app.client("viewer_two")
        one["mode"] = "live"
        one["live_attached"] = True
        one["case"] = "K05"
        self.assertEqual(self.app.client("viewer_two")["mode"], "replay")
        self.assertFalse(self.app.client("viewer_two")["live_attached"])
        self.assertEqual(self.app.replay_case(client_id="viewer_two")["case"], "K01")
        self.assertEqual(self.app.client("viewer_one")["case"], "K05")

    def test_bad_replay_case_is_not_silently_replaced(self):
        with self.assertRaises(KeyError):
            self.app.replay_case("K99")


class OptionalReplayStartupTests(unittest.TestCase):
    def test_missing_private_replay_does_not_block_live_service_startup(self):
        with patch.object(server, "RUN_RESULT", server.PROJECT_ROOT / "missing-private-replay-result.json"), patch.object(
            server, "CASE_IMAGES", {case: server.PROJECT_ROOT / "missing-private" / f"{case}.jpg" for case in server.CASE_ORDER}
        ):
            app = ControlCenter()
        try:
            health = app.health()
            self.assertTrue(health["ok"])
            self.assertFalse(health["replay_available"])
            self.assertIn("FileNotFoundError", health["replay_error"])
            with self.assertRaisesRegex(RuntimeError, "reviewed replay unavailable"):
                app.replay_case("K01")
        finally:
            app.close()


class LiveAgeFreshnessTests(unittest.TestCase):
    def event_with_age(self, age_s_marker="missing"):
        app = ControlCenter()
        client = app.client("age_regression")
        client["mode"] = "live"
        client["live_attached"] = True
        client["observation_id"] = 1
        status = {
            "connected": True,
            "frames": 42,
            "fps": 7.5,
            "avg_frame_time_ms": 133.3,
        }
        if age_s_marker != "missing":
            status["age_s"] = age_s_marker
        app._preview_json = lambda endpoint, timeout=1.0: dict(status)
        return app.live_state("age_regression")

    def test_missing_age_is_unknown_and_stale(self):
        event = self.event_with_age()
        self.assertIsNone(event["frame"]["age_ms"])
        self.assertTrue(event["frame"]["stale"])
        self.assertIsNone(event["frame"]["frame_ref"])

    def test_non_numeric_age_is_unknown_and_stale(self):
        event = self.event_with_age("n/a")
        self.assertIsNone(event["frame"]["age_ms"])
        self.assertTrue(event["frame"]["stale"])
        self.assertIsNone(event["frame"]["frame_ref"])

    def test_nan_age_is_unknown_and_stale(self):
        event = self.event_with_age(float("nan"))
        self.assertIsNone(event["frame"]["age_ms"])
        self.assertTrue(event["frame"]["stale"])
        self.assertIsNone(event["frame"]["frame_ref"])

    def test_negative_age_is_unknown_and_stale(self):
        event = self.event_with_age(-0.01)
        self.assertIsNone(event["frame"]["age_ms"])
        self.assertTrue(event["frame"]["stale"])
        self.assertIsNone(event["frame"]["frame_ref"])

    def test_age_over_two_seconds_is_stale(self):
        event = self.event_with_age(2.01)
        self.assertEqual(event["frame"]["age_ms"], 2010.0)
        self.assertTrue(event["frame"]["stale"])
        self.assertIsNone(event["frame"]["frame_ref"])

    def _capture_app(self, ages):
        app = ControlCenter()
        client = app.client("capture_age_regression")
        client["mode"] = "live"
        client["live_attached"] = True
        statuses = iter(ages)

        def preview_json(endpoint, timeout=1.0):
            status = next(statuses)
            return {"connected": True, "frames": 7, "fps": 7.0, **status}

        app._preview_json = preview_json
        return app, client

    def test_stale_capture_is_suppressed_before_jpeg_read(self):
        app, client = self._capture_app([{"age_s": 2.1}])
        with patch.object(server.LOCAL_HTTP, "open") as open_preview:
            with self.assertRaisesRegex(RuntimeError, "age is missing, invalid, or stale"):
                app.capture_live_snapshot("capture_age_regression")
        open_preview.assert_not_called()
        self.assertEqual(client["observation_id"], 0)
        self.assertIsNone(app.live_snapshot)

    def test_stale_transition_during_jpeg_read_suppresses_capture(self):
        app, client = self._capture_app([{"age_s": 0.1}, {"age_s": 2.1}])
        jpeg = b"\xff\xd8" + b"x" * 120 + b"\xff\xd9"
        with patch.object(server.LOCAL_HTTP, "open", return_value=io.BytesIO(jpeg)) as open_preview:
            with self.assertRaisesRegex(RuntimeError, "became stale while reading"):
                app.capture_live_snapshot("capture_age_regression")
        open_preview.assert_called_once()
        self.assertEqual(client["observation_id"], 0)
        self.assertIsNone(app.live_snapshot)

    def test_valid_recent_age_can_be_fresh(self):
        event = self.event_with_age(0.1)
        self.assertEqual(event["frame"]["age_ms"], 100.0)
        self.assertFalse(event["frame"]["stale"])
        self.assertIsNotNone(event["frame"]["frame_ref"])


if __name__ == "__main__":
    unittest.main()
