from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from host import nicla_clip  # noqa: E402


JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkI"
    "BgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAACAAMBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQ"
    "FBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2Jyg"
    "gkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTl"
    "JWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9"
    "oACAEBAAA/AOLr/9k="
)


def frame(frame_id, ticks=None, host_ns=0, boot_id="test-boot", data=JPEG):
    return {"data": data, "boot_id": boot_id, "frame_id": frame_id, "width": 3, "height": 2,
            "sha256": hashlib.sha256(data).hexdigest(), "capture_ticks_ms": ticks,
            "ticks_modulus_ms": 1000, "host_received_monotonic_ns": host_ns,
            "host_received_utc": "2026-11-11T00:00:00.000+00:00"}


class NiclaClipTests(unittest.TestCase):
    def test_preserves_jpegs_timing_and_avi_index(self):
        # Insert a legal one-byte JPEG comment to exercise RIFF chunk padding.
        odd = JPEG[:2] + b"\xff\xfe\x00\x03x" + JPEG[2:]
        frames = [frame(19, 970, 0, data=odd), frame(20, 10, 90_000_000),
                  frame(21, 50, 190_000_000)]
        with tempfile.TemporaryDirectory() as temporary:
            info = nicla_clip.write_clip(Path(temporary), "wrap", frames, reason="fixture")
            manifest_path = Path(info["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(info["frame_count"], 3)
            self.assertEqual(info["playback_fps"], 25.0)
            self.assertEqual(manifest["timing"]["board_capture_intervals"], 2)
            self.assertEqual(manifest["timing"]["host_receipt_intervals"], 0)
            self.assertEqual(manifest["reason"], "fixture")
            video = Path(info["video_path"]).read_bytes()
            self.assertEqual(struct.unpack_from("<4sI4s", video), (b"RIFF", len(video) - 8, b"AVI "))
            movi = video.index(b"movi")
            position = movi + 4
            index_entries = []
            for original, saved in zip(frames, manifest["frames"]):
                jpeg = original["data"]
                self.assertEqual((manifest_path.parent / saved["path"]).read_bytes(), jpeg)
                for key in ("boot_id", "frame_id", "sha256", "capture_ticks_ms", "ticks_modulus_ms",
                            "host_received_monotonic_ns", "host_received_utc"):
                    self.assertEqual(saved[key], original[key])
                self.assertEqual(struct.unpack_from("<4sI", video, position), (b"00dc", len(jpeg)))
                self.assertEqual(video[position + 8:position + 8 + len(jpeg)], jpeg)
                index_entries.append((b"00dc", 0x10, position - movi, len(jpeg)))
                position += 8 + len(jpeg) + len(jpeg) % 2
            self.assertEqual(struct.unpack_from("<4sI", video, position), (b"idx1", 16 * len(frames)))
            self.assertEqual(list(struct.iter_unpack("<4sIII", video[position + 8:])), index_entries)

    def test_host_time_fallback_and_single_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = nicla_clip.write_clip(root, "reset", [frame(19, 990),
                frame(1, 1, 250_000_000, boot_id="new-boot"),
                frame(2, None, 500_000_000, boot_id="new-boot")], reason="fixture")
            manifest = json.loads(Path(info["manifest_path"]).read_text())
            self.assertEqual(manifest["timing"]["board_capture_intervals"], 0)
            self.assertEqual(manifest["timing"]["host_receipt_intervals"], 2)
            self.assertEqual(info["playback_fps"], 4.0)
            single = nicla_clip.write_clip(root, "single", [frame(1)], reason="fixture")
            manifest = json.loads(Path(single["manifest_path"]).read_text())
            self.assertTrue(manifest["timing"]["default_used"])
            self.assertEqual(single["playback_fps"], 1.0)

    def test_invalid_data_never_publishes_a_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for frames, message in (
                ([], "at least one frame"),
                ([frame(1), frame(1)], "duplicate frame identity"),
                ([dict(frame(1), width=4)], "dimension mismatch"),
                ([dict(frame(1), sha256="0" * 64)], "SHA-256 mismatch"),
                ([frame(1, data=JPEG[:-2])], "JPEG start or end marker"),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    nicla_clip.write_clip(root, "bad", frames, reason="fixture")
                self.assertEqual(list(root.iterdir()), [])
            with self.assertRaisesRegex(ValueError, "invalid clip ID"):
                nicla_clip.write_clip(root, "../outside", [frame(1)], reason="fixture")

    def test_disk_error_cleans_stage_and_existing_clip_is_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(nicla_clip, "_write_avi", side_effect=OSError("fixture disk error")):
                with self.assertRaisesRegex(OSError, "fixture disk error"):
                    nicla_clip.write_clip(root, "atomic", [frame(1)], reason="fixture")
            self.assertEqual(list(root.iterdir()), [])
            info = nicla_clip.write_clip(root, "atomic", [frame(1)], reason="fixture")
            previous = Path(info["manifest_path"]).read_bytes()
            with self.assertRaises(FileExistsError):
                nicla_clip.write_clip(root, "atomic", [frame(2)], reason="fixture")
            self.assertEqual(Path(info["manifest_path"]).read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
