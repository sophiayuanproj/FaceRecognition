"""Synthetic firmware ACKs exercise battery handling without opening a serial port."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from control_center.server import make_handler
from host.nicla_battery import BATTERY_COMMANDS
from host.nicla_broker import NiclaBroker
from host.nicla_sightings import NiclaSightingsStore


NOW = "2026-11-11T21:02:03.000+00:00"
USB_BOOT = "boot-usb"
JPEG = b"\xff\xd8" + b"x" * 1400 + b"\xff\xd9"


class SerialFixture:
    def __init__(self, broker):
        self.broker = broker
        self.writes = []
        self.ack_lengths = []
        self.armed = False
        self.running = False
        self.record_count = 1
        self.available = True
        self.invalid_digest = False
        self.bad_offset = False
        self.drop_current_boot = False
        self.photo = {"record_id": 7, "boot_id": "boot-offline", "frame_id": 12, "ticks_ms": 75,
                      "ticks_modulus_ms": 1 << 30, "byte_count": len(JPEG), "face_count": 1,
                      "sha256": hashlib.sha256(JPEG).hexdigest(), "classification": "unknown", "identity_id": None}

    def emit(self, row):
        wire = json.dumps({"schema_id": "nicla-face-telemetry-1", "schema_version": 1, "sequence": 4,
                           "schema_sha256": "1" * 64, "firmware_sha256": "2" * 64, "model_config_sha256": "3" * 64,
                           "ticks_ms": 405, "ticks_modulus_ms": 1 << 30, **row}, separators=(",", ":")).encode() + b"\n"
        self.ack_lengths.append(len(wire))
        for item in self.broker._parser.feed(wire):
            self.broker._handle_stream_item(item, 100, NOW)

    def write(self, raw):
        assert len(raw) <= 384
        req = json.loads(raw)
        assert req.get("expected_boot_id") == USB_BOOT
        assert "expected_frame_id" not in req
        self.writes.append(req)
        cmd = req["cmd"]
        if cmd == "battery_arm":
            self.armed = True
            self.running = True
        elif cmd == "battery_disarm":
            self.armed = False
        if cmd in ("battery_arm", "battery_disarm", "battery_status"):
            if self.available:
                payload = {"armed": self.armed, "available": True, "running": self.running,
                           "record_count": self.record_count, "max_records": 24, "max_jpeg_bytes": 98304,
                           "jpeg_bytes": len(JPEG) * self.record_count, "invalid_count": 0}
            else:
                payload = {"armed": False, "available": False, "reason": "serial_only_usb_required"}
        elif cmd == "battery_list":
            payload = {"index": req["index"], "record_count": 1, "record": dict(self.photo)}
            if self.invalid_digest:
                payload["record"]["sha256"] = "0" * 64
        else:
            assert cmd == "battery_read" and 1 <= req["size"] <= 512 and req["record_id"] == self.photo["record_id"]
            offset = req["offset"]
            payload = {"record_id": req["record_id"], "offset": offset + int(self.bad_offset),
                       "chunk_hex": JPEG[offset:offset + req["size"]].hex()}
        ack = {"event": "control_ack", "command": cmd, "command_id": req["command_id"],
               "source": "usb", "control_status": "completed", **payload}
        self.emit(dict(ack, boot_id="wrong-boot"))
        if not self.drop_current_boot:
            self.emit(dict(ack, boot_id=USB_BOOT))
        return len(raw)

    def flush(self):
        pass

    def close(self):
        pass


class BatteryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.broker = NiclaBroker(lock_path=Path(self.tmp.name) / "serial.lock", utc_now=lambda: NOW)
        self.fixture = SerialFixture(self.broker)
        self.broker._connected = self.broker._device_ready = True
        self.broker._serial = self.fixture
        self.broker._latest_boot_id = USB_BOOT
        self.broker._applied["running"] = False
        self.controller = self.broker.claim_controller("viewer")["controller_id"]

    def tearDown(self):
        self.broker.close()
        self.tmp.cleanup()

    def configure(self, commands=BATTERY_COMMANDS):
        self.fixture.emit({"event": "control_ack", "command": "config", "command_id": "config-fixture",
                           "boot_id": USB_BOOT, "control_status": "completed", "supported_commands": list(commands)})

    def complete_job(self):
        self.broker._battery.close()
        result = self.broker.battery_status("viewer")
        self.assertFalse(result["busy"])
        return result

    def action(self, action):
        self.broker.battery_action(action, self.controller, "viewer")
        return self.complete_job()

    def stop_processing(self):
        self.fixture.running = False
        self.fixture.emit({"event": "control_ack", "command": "stop", "command_id": "stop-fixture",
                           "boot_id": USB_BOOT, "control_status": "completed", "running": False})
        self.broker.battery_status("viewer")
        return self.complete_job()

    def test_import_review_and_deletion_survive_repeat_without_leaking_chunks(self):
        self.configure()
        self.broker.battery_status("viewer")
        initial = self.complete_job()
        self.assertEqual(initial["actions"], {"arm": True, "disarm": False, "import": True})
        with self.assertRaises(PermissionError):
            self.broker.battery_action("arm", self.controller, "other-viewer")
        self.assertFalse(any(self.broker.battery_status("other-viewer")["actions"].values()))
        self.fixture.record_count = 0
        self.assertTrue(self.action("arm")["armed"])
        self.assertTrue(self.broker.status()["running"])
        self.assertFalse(any(self.broker.battery_status("viewer")["actions"].values()))
        self.fixture.record_count = 1
        self.assertTrue(self.stop_processing()["actions"]["import"])
        self.assertFalse(self.action("disarm")["armed"])
        imported = self.action("import")
        self.assertEqual((imported["pending_photos"], imported["imported_count"], imported["last_import_count"]), (1, 1, 1))
        self.assertEqual(imported["unimported_count"], 0)
        [sighting] = self.broker.sightings_status()["items"]
        self.assertEqual((sighting["source"], sighting["imported_utc"], sighting["boot_id"], sighting["frame_id"],
                          sighting["capture_ticks_ms"], sighting["identity_id"]),
                         ("on_nicla_battery", NOW, "boot-offline", 12, 75, None))
        self.assertIsNone(sighting["utc"])
        self.assertEqual(self.broker.get_sighting_jpeg(sighting["id"]), JPEG)
        self.assertEqual([row["size"] for row in self.fixture.writes if row["cmd"] == "battery_read"], [512, 512, 380])
        self.assertLess(max(self.fixture.ack_lengths), 2048)
        self.assertNotIn("chunk_hex", json.dumps(self.broker.events()))
        self.assertFalse(self.broker._frames)
        repeat = self.action("import")
        self.assertEqual((repeat["last_import_count"], repeat["unimported_count"]), (0, 0))
        self.assertTrue(self.broker.delete_battery_sighting(sighting["id"], self.controller, "viewer"))
        self.broker._battery.sightings = NiclaSightingsStore(self.broker._sightings.path)
        self.assertEqual(self.action("import")["imported_count"], 1)
        self.assertEqual(self.broker.sightings_status()["retained_count"], 0)
        self.assertEqual(sum(row["cmd"] == "battery_read" for row in self.fixture.writes), 3)

    def test_corrupt_transfer_and_wrong_offset_do_not_create_a_photo(self):
        self.configure()
        self.broker.battery_status("viewer")
        self.complete_job()
        self.fixture.bad_offset = True
        self.assertIn("offset", self.action("import")["reason"])
        self.fixture.bad_offset = False
        self.fixture.invalid_digest = True
        self.assertIn("checksum mismatch", self.action("import")["reason"])
        self.assertEqual(self.broker.sightings_status()["total_count"], 0)
        self.assertFalse(self.broker._sightings.path.exists())

    def test_old_firmware_and_wrong_boot_acks_are_ignored(self):
        self.configure(commands=["status"])
        self.assertFalse(self.broker.battery_status("viewer")["supported"])
        self.assertEqual(self.fixture.writes, [])
        self.configure()
        self.fixture.drop_current_boot = True
        with patch("host.nicla_battery._ACK_TIMEOUT", .03):
            self.broker.battery_status("viewer")
            status = self.complete_job()
        self.assertFalse(status["available"])
        self.assertIn("No reply", status["reason"])
        self.assertFalse(any(status["actions"].values()))

    def test_firmware_unavailable_ack_does_not_require_a_storage_count(self):
        self.configure()
        self.fixture.available = False
        self.broker.battery_status("viewer")
        status = self.complete_job()
        self.assertTrue(status["supported"])
        self.assertFalse(status["available"])
        self.assertIsNone(status["pending_photos"])
        self.assertIn("serial_only_usb_required", status["reason"])

    def test_http_route_requires_viewer_match_and_returns_direct_status(self):
        cls = make_handler(SimpleNamespace(device_broker=self.broker), {})
        handler = cls.__new__(cls)
        handler.viewer_id = lambda: "viewer"
        handler.send_json = lambda status, value: (status, value)
        handler.path = "/api/v2/device/battery"
        code, status = handler.do_GET()
        self.assertEqual(code, 200)
        self.assertIn("actions", status)
        handler.read_body = lambda: {"viewer_id": "other", "action": "arm", "controller_id": self.controller}
        self.assertEqual(handler.do_POST()[0], 403)
        handler.read_body = lambda: {"viewer_id": "viewer", "action": "invalid", "controller_id": self.controller}
        self.assertEqual(handler.do_POST()[0], 400)
        self.configure()
        self.broker.battery_status("viewer")
        self.complete_job()
        handler.read_body = lambda: {"viewer_id": "viewer", "action": "arm", "controller_id": self.controller}
        code, status = handler.do_POST()
        self.assertEqual(code, 202)
        self.assertIn("pending_photos", status)
        self.assertTrue(self.complete_job()["armed"])

    def test_gallery_http_assignment_requires_controller_but_not_usb(self):
        self.configure()
        self.broker.battery_status("viewer")
        self.complete_job()
        self.action("import")
        [sighting] = self.broker.sightings_status()["items"]
        self.broker.disconnect()
        cls = make_handler(SimpleNamespace(device_broker=self.broker), {})
        handler = cls.__new__(cls)
        handler.viewer_id = lambda: "viewer"
        handler.send_json = lambda status, value: (status, value)
        handler.path = "/api/v2/device/sightings/assign"
        body = {"viewer_id": "viewer", "controller_id": self.controller,
                "sighting_id": sighting["id"], "person_id": "K03"}
        for wrong in (dict(body, viewer_id="other"), dict(body, controller_id="wrong")):
            handler.read_body = lambda: wrong
            self.assertEqual(handler.do_POST()[0], 403)
        handler.read_body = lambda: dict(body, sighting_id=sighting["id"] + 1)
        self.assertEqual(handler.do_POST()[0], 404)
        handler.read_body = lambda: dict(body, person_id="K06")
        self.assertEqual(handler.do_POST()[0], 400)
        handler.read_body = lambda: body
        self.assertEqual(handler.do_POST(), (200, {"sighting_id": sighting["id"],
                                                 "person_id": "K03", "person_assignment": "manual"}))
        summary = self.broker.sightings_status()
        self.assertEqual(summary["items"][0]["person_id"], "K03")
        self.assertEqual(summary["identity_counts"]["K03"], 0)
        self.assertFalse(self.broker.status()["connected"])
