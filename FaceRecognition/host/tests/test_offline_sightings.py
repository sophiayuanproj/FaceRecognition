import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "firmware"))
import offline_sightings  # noqa: E402


def jpeg(size=64):
    return b"\xff\xd8" + b"x" * (size - 4) + b"\xff\xd9"


class OfflineSightingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "spool"
        self.store = offline_sightings.OfflineSightings(self.root)
        sync = patch.object(offline_sightings, "_sync")
        sync.start()
        self.addCleanup(sync.stop)

    def save(self, frame_id, data=None, **metadata):
        return self.store.save(jpeg() if data is None else data, boot_id="boot-7",
                               frame_id=frame_id, ticks_ms=100 + frame_id,
                               ticks_modulus_ms=1024, **metadata)

    def test_lazy_save_duplicate_and_chunk_read(self):
        self.assertEqual(self.store.status()["record_count"], 0)
        self.assertEqual(self.store.clear()["record_count"], 0)
        self.assertFalse(self.root.exists())
        data = jpeg(5000)
        row = self.save(1, data, face_count=1, decision={"status": "KNOWN_IDENTITY",
            "known_acceptance_enabled": True, "candidate": "K03"})
        self.assertEqual(self.save(1, data), row)
        self.assertEqual(row["identity_id"], "K03")
        self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
        assembled = b"".join(self.store.read(row["record_id"], offset, 2048)
                             for offset in range(0, len(data), 2048))
        self.assertEqual(assembled, data)
        self.assertEqual(self.store.read(row["record_id"], len(data)), b"")
        with self.assertRaises(ValueError):
            self.store.read(row["record_id"], size=2049)
        self.assertEqual(self.store.status()["jpeg_bytes"], len(data))

    def test_retains_newest_within_both_caps(self):
        for frame_id in range(1, 31):
            self.save(frame_id)
        self.assertEqual(self.store.status()["record_count"], 24)
        self.assertEqual([r["frame_id"] for r in self.store.list_records()], list(range(30, 6, -1)))
        for frame_id in range(31, 45):
            self.save(frame_id, jpeg(8192))
        self.assertEqual(self.store.status()["record_count"], 12)
        self.assertEqual(self.store.status()["jpeg_bytes"], 96 * 1024)
        self.assertEqual([r["frame_id"] for r in self.store.list_records()], list(range(44, 32, -1)))

    def test_power_cut_and_header_corruption_are_hidden_and_recovered(self):
        first = self.save(1)
        with patch.object(offline_sightings.os, "rename", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.save(2)
        self.assertEqual(self.store.list_records(), [first])
        self.assertEqual(self.store.status()["invalid_count"], 1)
        damaged = next(self.root.glob("*.rec"))
        data = damaged.read_bytes()
        damaged.write_bytes(data[:38] + bytes((data[38] ^ 1,)) + data[39:])
        self.assertEqual(self.store.list_records(), [])
        self.assertEqual(self.store.status()["invalid_count"], 2)
        valid = self.save(3, decision={"status": "UNCERTAIN", "known_acceptance_enabled": True,
                                       "candidate": "K03"})
        self.assertEqual(valid["classification"], "unknown")
        self.assertIsNone(valid["identity_id"])
        self.assertEqual(self.store.list_records(), [valid])
        self.assertEqual(self.store.status()["invalid_count"], 0)

    def test_clear_preserves_unrelated_files(self):
        self.save(1)
        unrelated = self.root / "notes.txt"
        unrelated.write_text("keep")
        self.assertEqual(self.store.clear()["record_count"], 0)
        self.assertEqual(unrelated.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
