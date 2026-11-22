import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from host.nicla_broker import NiclaBroker


class NiclaSightingBrokerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.lock_path = Path(temp.name) / "private/nicla.lock"
        self.broker = NiclaBroker(lock_path=self.lock_path)
        self.addCleanup(self.broker.close)
        self.controller = self.broker.claim_controller("viewer")['controller_id']
        stream = io.BytesIO()
        Image.new("L", (240, 240), 128).save(stream, format="JPEG")
        self.jpeg = stream.getvalue()
        self.digest = hashlib.sha256(self.jpeg).hexdigest()

    def packet(self, frame_id, digest=None):
        self.broker._handle_stream_item({"type": "packet", "header": {
            "kind": "frame_jpeg", "boot_id": "sight-test", "frame_id": frame_id,
            "width": 240, "height": 240, "sha256": digest or self.digest}, "data": self.jpeg},
            100_000_000, "2026-11-11T21:00:00Z")

    def observation(self, frame_id, seconds, *, detection="face", stage="full_pipeline",
                    accepted=False, status="OTHER", digest=None):
        self.broker._handle_stream_item({"type": "record", "record": {
            "event": "observation_complete", "boot_id": "sight-test", "source": "on_nicla",
            "frame_id": frame_id, "applied_state": {"measurement_stage_id": stage},
            "frame": {"boot_id": "sight-test", "frame_id": frame_id, "sha256": digest or self.digest},
            "detection": {"status": detection, "regions_xywh": [{"rect_px": [30, 30, 40, 40]}]},
            "decision": {"status": status, "candidate": "K01", "known_acceptance_enabled": accepted},
        }}, seconds * 1_000_000_000, "2026-11-11T21:00:00Z")

    def test_toggle_matching_frame_and_debounce(self):
        self.packet(1)
        self.observation(1, 1)
        self.assertEqual(self.broker.sightings_status()["total_encounters"], 0)
        self.assertFalse(self.lock_path.parent.exists())
        with self.assertRaises(PermissionError):
            self.broker.set_sightings_enabled(True, self.controller, "reader")
        self.broker.set_sightings_enabled(True, self.controller, "viewer")
        self.packet(2)
        self.observation(2, 2, stage="camera_capture_only")
        self.packet(3, "b" * 64)
        self.observation(3, 3)
        self.observation(4, 4, status="KNOWN_IDENTITY")  # disabled acceptance cannot name a person
        self.packet(4)
        self.broker._sightings_queue.join()
        summary = self.broker.sightings_status()
        self.assertEqual(summary["total_encounters"], 1)
        self.assertIsNone(summary["items"][0]["identity_id"])
        self.assertEqual(self.broker.get_sighting_jpeg(summary["items"][0]["id"]), self.jpeg)

        self.observation(5, 5, detection="no_face")
        self.packet(6)
        self.observation(6, 6, accepted=True, status="KNOWN_IDENTITY")
        self.broker._sightings_queue.join()
        summary = self.broker.sightings_status()
        self.assertEqual(summary["total_encounters"], 1)
        self.assertEqual(summary["items"][0]["identity_id"], "K01")
        self.assertEqual(summary["items"][0]["frame_id"], 6)
        self.assertEqual(summary["identity_counts"]["K01"], 1)

        self.observation(7, 7, detection="no_face")
        self.observation(8, 9, detection="no_face")
        self.packet(9)
        self.observation(9, 10)
        self.broker._sightings_queue.join()
        self.assertEqual(self.broker.sightings_status()["total_encounters"], 1)
        self.packet(10)
        self.observation(10, 11)
        self.broker._sightings_queue.join()
        self.assertEqual(self.broker.sightings_status()["total_encounters"], 2)
        restarted = NiclaBroker(lock_path=self.lock_path)
        self.addCleanup(restarted.close)
        self.assertFalse(restarted.sightings_status()["enabled"])
        self.assertEqual(restarted.sightings_status()["total_encounters"], 2)


if __name__ == "__main__":
    unittest.main()
