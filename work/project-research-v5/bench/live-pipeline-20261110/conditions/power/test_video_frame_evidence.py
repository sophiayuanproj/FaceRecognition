"""Synthetic frame/linkage fixtures with no attached Nicla."""

from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from video_frame_evidence import PrivateSamples, coverage_counts, fetch_verified_jpeg, link_frames


BOOT = "fixture-boot"
JPEG = b"\xff\xd8synthetic-fixture\xff\xd9"
DIGEST = hashlib.sha256(JPEG).hexdigest()
PROJECTION = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
VIDEO = {"sha256": "video-fixture", "fps": 20, "episodes": [
    {"id": "quiet", "start_ms": 0, "end_ms_exclusive": 2000, "expected_face": False},
    {"id": "face", "start_ms": 2000, "end_ms_exclusive": 4000, "expected_face": True,
     "face_box_xywh_video_px": [10, 20, 30, 40]},
]}
RENDER = [{"event": event, "media_time_s": secs, "host_receive_monotonic_ns": (10 + secs) * 10**9}
          for event, secs in (("video_started", 0), ("video_tick", 2), ("video_ended", 4))]
START = {"command_id": "fixture-start", "policy_id": "F", "measurement_stage_id": "full_pipeline"}


def event(frame_id, ms, seq, **fields):
    return dict(boot_id=BOOT, frame_id=frame_id, event_seq=seq,
                host_received_monotonic_ns=10**10 + ms * 10**6, **fields)


def packet(frame_id, ms, seq):
    return event(frame_id, ms, seq, type="frame_payload", kind="frame_jpeg", sha256=DIGEST)


def observation(frame_id, ms, seq, *, face, sha=DIGEST):
    rect = [10, 20, 30, 40]
    return event(frame_id, ms, seq, type="observation_complete",
                 frame={"boot_id": BOOT, "frame_id": frame_id, "sha256": sha},
                 detection={"status": "face" if face else "no_face", "regions_xywh": [rect] if face else []},
                 identity={"status": "ran" if face else "skipped"},
                 crop={"status": "valid", "rect_px": rect, "frame_id": frame_id} if face else None,
                 applied_state={"running": True, "policy_id": "F", "measurement_stage_id": "full_pipeline"},
                 decision={"known_acceptance_enabled": False}, timings_ms={"total": 0})


class FixtureResponse(io.BytesIO):
    def __init__(self, jpeg=JPEG, *, boot=BOOT):
        super().__init__(jpeg)
        self.headers = Message()
        self.headers["Content-Type"] = "image/jpeg"
        self.headers["X-Device-Boot-ID"] = boot
        self.headers["X-Device-Frame-ID"] = "4"
        self.headers["X-Frame-SHA256"] = DIGEST


class FrameEvidenceTest(unittest.TestCase):
    def test_linkage_counts_preview_only_errors_hash_mismatch_and_event_replay(self):
        quiet = packet(1, 900, 1)
        events = [quiet, quiet, observation(1, 900, 2, face=False), packet(2, 2600, 3),
                  event(2, 2600, 4, event="telemetry_encode_error", record={"failed_event": "observation_complete"}),
                  packet(3, 3000, 5), observation(3, 3000, 6, face=True, sha="0" * 64),
                  packet(4, 3200, 7), observation(4, 3200, 8, face=True)]
        frames, errors = link_frames(events, RENDER, BOOT)
        counts = coverage_counts(frames, errors, VIDEO)
        face = counts["face_present_episode_interiors"]
        self.assertEqual(face["preview_jpegs"], 3)
        self.assertEqual(face["observation_complete"], 2)
        self.assertEqual(face["matched_observations"], 1)
        self.assertEqual(face["observation_encode_errors"], 1)
        self.assertEqual(face["mismatched_or_duplicate_frame_ids"], [3])
        self.assertEqual(counts["quiet_episode_interiors"]["matched_observations"], 1)
        self.assertEqual(counts["episodes"]["face"]["max_linked_observation_gap_ms"], 850)

    def test_private_samples_preserve_positive_crop_quiet_and_command_receipt(self):
        events = [packet(1, 900, 1), observation(1, 900, 2, face=False),
                  packet(4, 3200, 7), observation(4, 3200, 8, face=True)]
        frames, _ = link_frames(events, RENDER, BOOT)
        with tempfile.TemporaryDirectory() as temp, patch(
            "video_frame_evidence.fetch_verified_jpeg", return_value=JPEG,
        ) as fetch:
            root = Path(temp) / "private"
            samples = PrivateSamples(root, "http://127.0.0.1:8780", BOOT, START, VIDEO, PROJECTION)
            samples.collect(frames)
            samples.collect(frames)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(samples.write_receipt(), {"face_positive": 1, "quiet_negative": 1})
            rows = json.loads((root / "LINKED_FRAMES.json").read_text())["samples"]
            positive = rows[0]
            self.assertEqual(positive["start_command_id"], START["command_id"])
            self.assertEqual(positive["projected_face_crop_coverage"], 1)
            self.assertEqual(positive["identity"]["status"], "ran")
            self.assertEqual((root / positive["file"]).read_bytes(), JPEG)

    def test_jpeg_download_requires_header_boot_frame_and_exact_bytes(self):
        with patch("urllib.request.urlopen", return_value=FixtureResponse()):
            self.assertEqual(fetch_verified_jpeg("http://127.0.0.1:8780", BOOT, 4, DIGEST), JPEG)
        with patch("urllib.request.urlopen", return_value=FixtureResponse(boot="different-boot")):
            with self.assertRaisesRegex(ValueError, "provenance"):
                fetch_verified_jpeg("http://127.0.0.1:8780", BOOT, 4, DIGEST)
        with patch("urllib.request.urlopen", return_value=FixtureResponse(b"\xff\xd8wrong\xff\xd9")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                fetch_verified_jpeg("http://127.0.0.1:8780", BOOT, 4, DIGEST)


if __name__ == "__main__":
    unittest.main()
