import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from host.nicla_broker import NiclaBroker
from test_nicla_broker_ready_state import _Serial


class NiclaVideoCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.broker = NiclaBroker(lock_path=Path(self.temp.name) / "serial.lock")
        self.addCleanup(self.broker.close)
        self.broker._serial = _Serial()
        self.broker._connected = self.broker._device_ready = True
        self.broker._latest_boot_id = "video-test-boot"
        self.broker._applied.update(running=False, policy_id="W")
        self.broker._capabilities["measurement_stage_ids"] = ["camera_capture_only"]
        self.controller = self.broker.claim_controller("operator")["controller_id"]
        image = io.BytesIO()
        Image.new("L", (240, 240), color=100).save(image, "JPEG")
        self.jpeg = image.getvalue()
        self.sha = hashlib.sha256(self.jpeg).hexdigest()

    def record(self, event, **fields):
        self.broker._handle_stream_item({"type": "record", "record": {
            "event": event, "boot_id": "video-test-boot", **fields}},
            receive_ns=1_000_000, receive_utc="2026-11-11T18:00:00Z")

    def packet(self, frame_id):
        self.broker._handle_stream_item({"type": "packet", "header": {
            "kind": "frame_jpeg", "boot_id": "video-test-boot", "frame_id": frame_id,
            "width": 240, "height": 240, "sha256": self.sha},
            "data": self.jpeg}, receive_ns=frame_id * 256_000_000,
            receive_utc="2026-11-11T18:00:01Z")

    def test_recording_outlasts_ring_cache_and_excludes_frames_outside_trial(self):
        with self.assertRaises(PermissionError):
            self.broker.start_video(self.controller, "other-viewer")
        status = self.broker.start_video(self.controller, "operator")
        self.assertEqual(status["state"], "starting")
        clip_id = status["id"]
        command = json.loads(self.broker._serial.writes[-1])
        self.assertEqual((command["cmd"], command["trial_id"], command["duration_ms"],
                          command["measurement_stage_id"]),
                         ("trial", clip_id, 12000, "camera_capture_only"))
        self.packet(0)
        self.record("control_ack", command="trial", command_id=clip_id,
                    control_status="completed", trial_id=clip_id, running=True)
        for frame_id in range(1, 43):
            self.record("capture_start", frame_index=frame_id,
                        ticks_ms=frame_id * 256, ticks_modulus_ms=1 << 30)
            self.packet(frame_id)
        self.assertEqual(self.broker.video_status()["frames"], 42)
        with self.assertRaises(KeyError):
            self.broker.get_frame("video-test-boot", 1)

        self.broker.stop_video(self.controller, "operator")
        stop = json.loads(self.broker._serial.writes[-1])
        self.assertEqual(stop["cmd"], "stop")
        self.record("trial_finished", trial_id=clip_id, reason_code="stopped", running=False)
        self.packet(43)
        self.broker.close()

        status = self.broker.video_status()
        self.assertEqual((status["state"], status["frames"], status["missed_frames"]), ("ready", 42, 0))
        clip = self.broker.video_root / clip_id
        manifest = json.loads((clip / "manifest.json").read_text())
        self.assertEqual([row["frame_id"] for row in manifest["frames"]], list(range(1, 43)))
        self.assertEqual(manifest["frames"][0]["capture_ticks_ms"], 256)
        self.assertEqual((clip / manifest["frames"][0]["path"]).read_bytes(), self.jpeg)
        self.assertEqual(self.broker.get_video_file(clip_id)[:4], b"RIFF")

    def test_capture_rejection_leaves_no_video(self):
        status = self.broker.start_video(self.controller, "operator")
        self.record("control_ack", command="trial", command_id=status["id"],
                    control_status="rejected", reason_code="late_command")
        self.assertEqual(self.broker.video_status()["state"], "error")
        self.assertFalse(self.broker.video_root.exists())

    def test_rejected_recording_keeps_previous_clip_available(self):
        first = self.broker.start_video(self.controller, "operator")
        clip_id = first["id"]
        self.record("control_ack", command="trial", command_id=clip_id,
                    control_status="completed", trial_id=clip_id, running=True)
        self.record("capture_start", frame_index=1, ticks_ms=256, ticks_modulus_ms=1 << 30)
        self.packet(1)
        self.record("trial_finished", trial_id=clip_id, reason_code="stopped", running=False)
        self.broker._video["writer"].join(timeout=5)
        ready = self.broker.video_status()
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["last_clip"]["manifest_url"], ready["manifest_url"])

        attempt = self.broker.start_video(self.controller, "operator")
        self.assertEqual(attempt["last_clip"]["id"], clip_id)
        self.record("control_ack", command="trial", command_id=attempt["id"],
                    control_status="rejected", reason_code="late_command")
        failed = self.broker.video_status()
        self.assertEqual(failed["state"], "error")
        self.assertEqual(failed["last_clip"], ready["last_clip"])
        self.assertEqual(self.broker.get_video_file(clip_id)[:4], b"RIFF")

    def test_inference_recording_on_running_loop_joins_same_frame_and_never_stops_board(self):
        self.broker._capabilities["measurement_stage_ids"].append("full_pipeline")
        self.broker._applied.update(running=True, measurement_stage_id="full_pipeline")
        wire_before = list(self.broker._serial.writes)
        clip_id = self.broker.start_video(self.controller, "operator", "inference")["id"]
        self.packet(0)

        def observation(frame_id, sha):
            self.record("observation_complete", frame_id=frame_id,
                frame={"boot_id": "video-test-boot", "frame_id": frame_id, "sha256": sha},
                applied_state={"measurement_stage_id": "full_pipeline", "running": True},
                detection={"status": "face"}, identity={"status": "ran"}, decision={"status": "OTHER"})

        for frame_id in range(1, 42):
            self.record("capture_start", frame_index=frame_id, ticks_ms=frame_id * 250,
                        ticks_modulus_ms=1 << 30)
            self.packet(frame_id)
            observation(frame_id, self.sha if frame_id != 1 else "wrong-hash")
        self.record("capture_start", frame_index=42, ticks_ms=10500, ticks_modulus_ms=1 << 30)
        stop = self.broker.stop_video(self.controller, "operator")
        self.assertTrue(stop["draining"])
        self.record("capture_start", frame_index=43, ticks_ms=10750, ticks_modulus_ms=1 << 30)
        self.packet(43)
        observation(43, self.sha)
        observation(42, self.sha)
        self.packet(42)
        self.assertTrue(self.broker._applied["running"])
        self.assertEqual(self.broker._serial.writes, wire_before)
        self.broker._finish_video("stopped")
        self.broker.close()

        manifest = json.loads(self.broker.get_video_file(clip_id, "manifest.json"))
        self.assertEqual(manifest["recording_mode"], "inference")
        self.assertEqual(manifest["matched_observations"], 41)
        self.assertEqual([f["frame_id"] for f in manifest["frames"]], list(range(1, 43)))
        self.assertNotIn("observation", manifest["frames"][0])
        self.assertEqual(manifest["frames"][-1]["observation"]["decision"]["status"], "OTHER")
        self.assertEqual(self.broker.get_video_file(clip_id, "frames/frame-000042.jpg"), self.jpeg)
        with self.assertRaises(KeyError):
            self.broker.get_video_file(clip_id, "../serial.lock")

    def test_stopped_inference_recording_requests_the_full_firmware_trial(self):
        self.broker._capabilities["measurement_stage_ids"].append("full_pipeline")
        self.broker.start_video(self.controller, "operator", "inference")
        trial = json.loads(self.broker._serial.writes[-1])
        self.assertEqual(trial["measurement_stage_id"], "full_pipeline")
        self.assertEqual(trial["model_slot"], "small")


if __name__ == "__main__":
    unittest.main()
