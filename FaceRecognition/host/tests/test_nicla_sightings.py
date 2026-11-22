from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from host.nicla_sightings import NiclaSightingsStore  # noqa: E402


JPEG = b"\xff\xd8sighting-test-frame\xff\xd9"


def photo(boot_id, frame_id, *, status="UNCERTAIN", enabled=False, candidate="K01", jpeg=JPEG):
    return {"boot_id": boot_id, "frame_id": frame_id, "jpeg": jpeg,
            "metadata": {"sha256": hashlib.sha256(jpeg).hexdigest(),
                         "host_received_utc": "2026-11-11T00:00:00.000+00:00",
                         "decision": {"status": status, "known_acceptance_enabled": enabled,
                                      "candidate": candidate, "raw_candidate": "unaccepted"}}}


class NiclaSightingsTests(unittest.TestCase):
    def test_construction_and_empty_reads_do_not_create_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private"
            store = NiclaSightingsStore(directory / "sightings.sqlite3")
            self.assertEqual(store.summary()["total_count"], 0)
            self.assertEqual(store.summary()["retained_count"], 0)
            with self.assertRaises(KeyError):
                store.get_jpeg(1)
            self.assertFalse(directory.exists())

    def test_known_requires_exact_status_boolean_gate_and_roster(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sightings.sqlite3"
            store = NiclaSightingsStore(path)
            decisions = [("OTHER", True, "K01"), ("KNOWN_IDENTITY", False, "K01"),
                         ("KNOWN_IDENTITY", 1, "K01"), ("KNOWN_IDENTITY", True, "K06"),
                         ("KNOWN_IDENTITY", True, "K02")]
            for frame_id, (status, enabled, candidate) in enumerate(decisions):
                row = photo("boot-1", frame_id, status=status, enabled=enabled, candidate=candidate)
                row["metadata"]["face_count"] = 2 if frame_id == 4 else 9
                store.add(**row)
            summary = NiclaSightingsStore(path).summary()
            self.assertEqual(summary["total_count"], 5)
            self.assertEqual(summary["identity_counts"], {"K01": 0, "K02": 1, "K03": 0, "K04": 0, "K05": 0})
            self.assertEqual(summary["items"][0]["classification"], "known")
            self.assertEqual(summary["items"][0]["identity_id"], "K02")
            self.assertEqual(summary["items"][0]["person_id"], "K02")
            self.assertEqual(summary["items"][0]["person_assignment"], "device")
            self.assertEqual(summary["items"][0]["face_count"], 2)
            for item in summary["items"][1:]:
                self.assertEqual(item["classification"], "unknown")
                self.assertIsNone(item["identity_id"])
                self.assertIsNone(item["person_id"])
                self.assertEqual(item["person_assignment"], "unassigned")
                self.assertNotIn("candidate", item)
                self.assertNotIn("raw_candidate", item)
                self.assertNotIn("face_count", item)

    def test_retention_counts_and_boot_watermark_survive_reopening(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sightings.sqlite3"
            store = NiclaSightingsStore(path, max_items=2, max_jpeg_bytes=2 * len(JPEG))
            for frame_id in (1, 2, 3):
                store.add(**photo("boot-1", frame_id, status="KNOWN_IDENTITY", enabled=True))
            store = NiclaSightingsStore(path, max_items=2, max_jpeg_bytes=2 * len(JPEG))
            self.assertEqual(store.add(**photo("boot-1", 1)),
                             {"id": None, "added": False, "updated": False, "retained": False})
            self.assertEqual(store.add(**photo("boot-1", 2))["id"], 2)
            summary = store.summary(limit=1)
            self.assertEqual(summary["total_count"], 3)
            self.assertEqual(summary["retained_count"], 2)
            self.assertEqual(summary["retained_jpeg_bytes"], 2 * len(JPEG))
            self.assertEqual(summary["identity_counts"]["K01"], 3)
            self.assertEqual([item["id"] for item in summary["items"]], [3])
            with self.assertRaises(KeyError):
                store.get_jpeg(1)
            self.assertEqual(store.get_jpeg(3)[0], JPEG)
            self.assertEqual(store.add(**photo("boot-2", 1))["id"], 4)

    def test_byte_budget_removes_oldest_even_below_row_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = NiclaSightingsStore(Path(temporary) / "sightings.sqlite3", max_items=10,
                                        max_jpeg_bytes=2 * len(JPEG) - 1)
            store.add(**photo("boot-1", 1))
            store.add(**photo("boot-1", 2))
            self.assertEqual(store.summary()["total_count"], 2)
            self.assertEqual(store.summary()["retained_count"], 1)
            self.assertEqual(store.summary()["retained_jpeg_bytes"], len(JPEG))
            self.assertEqual(store.summary()["items"][0]["id"], 2)

    def test_upgrade_keeps_encounter_id_and_uses_accepted_frame_photo(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = NiclaSightingsStore(Path(temporary) / "sightings.sqlite3")
            first = store.add(**photo("boot-1", 17))
            raw = b"\xff\xd8later-image\xff\xd9"
            known = photo("boot-1", 20, status="KNOWN_IDENTITY", enabled=True, candidate="K04", jpeg=raw)
            upgrade = store.add(**known, encounter_frame_id=17)
            self.assertEqual(upgrade, {"id": first["id"], "added": False, "updated": True, "retained": True})
            self.assertFalse(store.add(**known, encounter_frame_id=17)["updated"])
            self.assertFalse(store.add(**photo("boot-1", 21, status="KNOWN_IDENTITY", enabled=True),
                                       encounter_frame_id=17)["updated"])
            self.assertFalse(store.add(**photo("boot-2", 20, status="KNOWN_IDENTITY", enabled=True),
                                       encounter_frame_id=17)["added"])
            jpeg, item = store.get_jpeg(first["id"])
            self.assertEqual(jpeg, raw)
            self.assertEqual((item["encounter_frame_id"], item["frame_id"], item["identity_id"]), (17, 20, "K04"))
            self.assertEqual(store.summary()["total_count"], 1)
            self.assertEqual(store.summary()["identity_counts"]["K04"], 1)

    def test_manual_person_survives_reopen_and_a_later_device_accept(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sightings.sqlite3"
            store = NiclaSightingsStore(path)
            sighting_id = store.add(**photo("boot-1", 17))["id"]
            original_jpeg, original = store.get_jpeg(sighting_id)
            self.assertEqual(store.assign_person(sighting_id, "K05"),
                             {"sighting_id": sighting_id, "person_id": "K05", "person_assignment": "manual"})
            store = NiclaSightingsStore(path)
            first_summary = store.summary()
            self.assertEqual(first_summary["items"][0]["person_id"], "K05")
            self.assertEqual(first_summary["identity_counts"]["K05"], 0)
            self.assertEqual(first_summary["items"][0]["sha256"], original["sha256"])
            self.assertEqual(store.get_jpeg(sighting_id)[0], original_jpeg)

            store.add(**photo("boot-1", 20, status="KNOWN_IDENTITY", enabled=True, candidate="K04"),
                      encounter_frame_id=17)
            item = store.summary()["items"][0]
            self.assertEqual((item["person_id"], item["person_assignment"], item["identity_id"]),
                             ("K05", "manual", "K04"))
            store.assign_person(sighting_id, None)
            summary = store.summary()
            item = summary["items"][0]
            self.assertEqual((item["person_id"], item["person_assignment"], item["identity_id"]),
                             (None, "manual", "K04"))
            self.assertEqual(summary["identity_counts"]["K04"], 1)
            self.assertEqual(summary["total_count"], 1)

    def test_manual_assignment_rejects_invalid_ids_and_evicted_photos(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = NiclaSightingsStore(Path(temporary) / "sightings.sqlite3", max_items=1)
            old_id = store.add(**photo("boot-1", 17))["id"]
            new_id = store.add(**photo("boot-1", 19))["id"]
            for person in ("K06", "", True, ["K01"]):
                with self.subTest(person=person), self.assertRaises(ValueError):
                    store.assign_person(new_id, person)
            with self.assertRaises(ValueError):
                store.assign_person(True, "K01")
            with self.assertRaises(KeyError):
                store.assign_person(old_id, "K01")
            self.assertEqual(store.summary()["items"][0]["person_assignment"], "unassigned")

    def test_concurrent_calls_count_once_and_keep_bounded_recent_boots(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sightings.sqlite3"
            store = NiclaSightingsStore(path, max_items=3)
            with ThreadPoolExecutor(max_workers=8) as executor:
                same = list(executor.map(lambda _: store.add(**photo("boot-one", 1)), range(12)))
                self.assertEqual(sum(row["added"] for row in same), 1)
                new = list(executor.map(lambda index: store.add(**photo(f"boot-{index}", 1)), range(40)))
                self.assertEqual(sum(row["added"] for row in new), 40)
            self.assertEqual(store.summary()["total_count"], 41)
            self.assertEqual(store.summary()["retained_count"], 3)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM recent_boots").fetchone()[0], 32)

    def test_invalid_input_does_not_create_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "private"
            store = NiclaSightingsStore(directory / "sightings.sqlite3")
            wrong = photo("boot-1", 1)
            wrong["metadata"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                store.add(**wrong)
            with self.assertRaisesRegex(ValueError, "bounded JPEG"):
                store.add(**photo("boot-1", 1, jpeg=b"invalid"))
            for item_id in (True, -1, 0, 1 << 63, "1"):
                with self.subTest(item_id=item_id), self.assertRaises(KeyError):
                    store.get_jpeg(item_id)
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
