from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from host import nicla_broker
from host.nicla_broker import NiclaBroker, NiclaPortBusy, NiclaPortNotFound


class _Serial:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _QuietSerial(_Serial):
    latest = None

    def __init__(self, **_kwargs):
        super().__init__()
        type(self).latest = self
        self.dtr = False
        self.rts = False
        self.port = None
        self.in_waiting = 0

    def open(self):
        return None

    def read(self, _size):
        time.sleep(0.002)
        return b""


class _SerialModule:
    Serial = _QuietSerial


class NiclaBrokerReadyStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.broker = NiclaBroker(lock_path=Path(self.temp.name) / "serial.lock")
        self.serial = _Serial()
        self.broker._descriptor = {
            "device": "/dev/fake-nicla", "serial_number": "FAKE001",
            "vid": 0x2341, "pid": 0x045F,
        }
        self.broker._serial = self.serial
        self.broker._connected = True
        self.broker._device_ready = False

    def tearDown(self):
        self.broker.close()
        self.temp.cleanup()

    def test_windows_usbser_descriptor_can_omit_product_name(self):
        nicla = SimpleNamespace(device="COM9", vid=0x2341, pid=0x045F,
                               serial_number="board-serial", product=None)
        meter = SimpleNamespace(device="COM3", vid=0x2E3C, pid=0x5558,
                               serial_number="meter-serial", product=None)
        with patch.object(self.broker, "_list_ports", return_value=[meter, nicla]):
            with patch.object(nicla_broker, "os", SimpleNamespace(name="nt")):
                self.assertEqual(self.broker.discover()["device"], "COM9")
            with patch.object(nicla_broker, "os", SimpleNamespace(name="posix")):
                with self.assertRaises(NiclaPortNotFound):
                    self.broker.discover()

    def test_second_broker_cannot_lock_existing_serial_lock_file(self):
        first = NiclaBroker(lock_path=self.broker.lock_path)
        second = NiclaBroker(lock_path=self.broker.lock_path)
        try:
            first._acquire_port_lock("fixture")
            with self.assertRaises(NiclaPortBusy):
                second._acquire_port_lock("fixture")
            first._release_port_lock()
            second._acquire_port_lock("fixture")
        finally:
            first._release_port_lock()
            second._release_port_lock()

    def test_open_serial_is_not_a_ready_device_and_commands_fail_closed(self):
        status = self.broker.status()
        self.assertTrue(status["serial_open"])
        self.assertFalse(status["device_ready"])
        self.assertFalse(status["connected"])
        self.assertEqual(status["mode"], "DISCONNECTED")
        response = self.broker.command({"cmd": "status", "command_id": "wait_for_boot"})
        self.assertEqual(response["status"], "rejected")
        self.assertEqual(response["message"], "device_not_ready_waiting_for_boot_or_frame")
        self.assertEqual(self.serial.writes, [])

    def test_boot_hello_makes_device_ready_and_clears_stale_warning(self):
        self.broker._connection_warning = "nicla_hello_or_frame_timeout_after_usb_dtr_assert"
        self.broker._handle_stream_item(
            {"type": "record", "record": {
                "event": "boot", "boot_id": "boot-test-1", "run_id": "boot-test-1",
                "reset_cause": 0, "model_config_sha256": "a" * 64,
                "firmware_sha256": "b" * 64,
            }},
            receive_ns=10_000_000,
            receive_utc="2026-11-02T23:00:00+00:00",
        )
        status = self.broker.status()
        self.assertTrue(status["connected"])
        self.assertTrue(status["device_ready"])
        self.assertEqual(status["boot_id"], "boot-test-1")
        self.assertIsNone(status["connection_warning"])
        response = self.broker.command({"cmd": "status", "command_id": "after_boot"})
        self.assertEqual(response["status"], "accepted")
        self.assertEqual(len(self.serial.writes), 1)

    def test_observation_can_confirm_a_device_when_boot_record_was_dropped(self):
        self.broker._handle_stream_item(
            {"type": "record", "record": {
                "event": "observation_complete", "boot_id": "boot-test-2",
                "run_id": "boot-test-2", "frame_id": 4,
                "camera": {"frame_shape": [240, 240]},
            }},
            receive_ns=20_000_000,
            receive_utc="2026-11-02T23:00:01+00:00",
        )
        status = self.broker.status()
        self.assertTrue(status["device_ready"])
        self.assertEqual(status["last_observation"]["frame_id"], 4)

    def test_versioned_control_status_is_a_hello_when_loop_is_stopped(self):
        self.broker._connection_warning = "nicla_hello_or_frame_timeout_after_usb_dtr_assert"
        self.broker._handle_stream_item(
            {"type": "record", "record": {
                "event": "control_input_status",
                "schema_id": "nicla-face-telemetry-1",
                "schema_version": 1,
                "boot_id": "boot-stopped-1", "run_id": "boot-stopped-1",
                "firmware_sha256": "b" * 64,
                "model_config_sha256": "c" * 64,
                "schema_sha256": "d" * 64,
                "usb_connected": True,
            }},
            receive_ns=30_000_000,
            receive_utc="2026-11-02T23:00:01+00:00",
        )
        status = self.broker.status()
        self.assertTrue(status["device_ready"])
        self.assertTrue(status["connected"])
        self.assertEqual(status["boot_id"], "boot-stopped-1")
        self.assertIsNone(status["connection_warning"])
        response = self.broker.command({"cmd": "status", "command_id": "stopped_hello"})
        self.assertEqual(response["status"], "accepted")
        self.assertEqual(len(self.serial.writes), 1)
        first = self.broker.events(after_seq=0, limit=10)["events"][-2]
        self.assertEqual(first["type"], "device_hello")
        self.broker._handle_stream_item(
            {"type": "record", "record": {
                "event": "control_input_status",
                "schema_id": "nicla-face-telemetry-1",
                "boot_id": "boot-stopped-1", "run_id": "boot-stopped-1",
                "firmware_sha256": "b" * 64,
                "model_config_sha256": "c" * 64,
                "schema_sha256": "d" * 64,
            }},
            receive_ns=31_000_000,
            receive_utc="2026-11-02T23:00:02+00:00",
        )
        last = self.broker.events(after_seq=0, limit=10)["events"][-1]
        self.assertEqual(last["type"], "telemetry")

    def test_malformed_control_status_does_not_make_device_ready(self):
        self.broker._handle_stream_item(
            {"type": "record", "record": {
                "event": "control_input_status",
                "schema_id": "nicla-face-telemetry-1",
                "boot_id": "boot-stopped-1",
                "firmware_sha256": "bad",
                "model_config_sha256": "c" * 64,
                "schema_sha256": "d" * 64,
            }},
            receive_ns=30_000_000,
            receive_utc="2026-11-02T23:00:02+00:00",
        )
        self.assertFalse(self.broker.status()["device_ready"])

    def test_disconnect_clears_device_ready_state(self):
        self.broker._device_ready = True
        status = self.broker.disconnect()
        self.assertFalse(status["serial_open"])
        self.assertFalse(status["device_ready"])
        self.assertFalse(status["connected"])

    def test_usb_disconnect_finishes_pending_capture(self):
        self.broker._device_ready = True
        self.broker._latest_boot_id = "boot-test-1"
        lease = self.broker.claim_controller("operator")
        command = {"cmd": "one_shot", "command_id": "capture-while-unplugged",
                   "controller_id": lease["controller_id"]}
        self.assertEqual(self.broker.command(command, viewer_id="operator")["status"], "accepted")

        self.broker.disconnect()

        self.assertFalse(self.broker._pending)
        acks = [event for event in self.broker.events()["events"]
                if event.get("command_id") == command["command_id"]]
        self.assertEqual(acks[-1]["control_status"], "error")
        self.assertEqual(acks[-1]["reason_code"], "device_disconnected")

    def test_stop_targets_the_boot_without_a_synthetic_frame_deadline(self):
        self.broker._device_ready = True
        self.broker._latest_observation = {"boot_id": "boot-fast", "frame_id": 120}
        controller = self.broker.claim_controller("viewer")["controller_id"]
        request = {"controller_id": controller, "cmd": "stop", "command_id": "stop_fast"}
        self.assertEqual(self.broker.command(request, viewer_id="viewer")["status"], "accepted")
        wire = json.loads(self.serial.writes[-1])
        self.assertEqual(wire["expected_boot_id"], "boot-fast")
        self.assertNotIn("expected_frame_id", wire)
        self.assertNotIn("max_frame_lag", wire)

    def test_cleared_model_override_reports_policy_route_not_stale_observation_model(self):
        self.broker._applied.update({"model": "large", "model_override": None,
                                     "policy_id": "S"})
        status = self.broker.status()
        self.assertEqual(status["applied_state"]["model_override"], None)
        self.assertEqual(status["applied_state"]["model"], "policy")

    def test_event_backlog_returns_recent_tail_and_reports_skips(self):
        for index in range(5):
            self.broker._publish({"type": "fixture_record", "index": index})
        result = self.broker.events(after_seq=0, limit=2)
        self.assertEqual([row["index"] for row in result["events"]], [3, 4])
        self.assertEqual(result["skipped_count"], 3)
        self.assertTrue(result["truncated"])

    def test_config_control_ack_populates_firmware_choices(self):
        self.broker._connected = True
        self.broker._handle_stream_item(
            {"type": "record", "record": {
                "event": "control_ack", "command": "config", "control_status": "completed",
                "protocol": "nicla-face-control-2", "policy_ids": ["H", "L", "S", "A-full"],
                "supported_commands": ["one_shot", "start", "stop", "trial"],
                "measurement_stage_ids": ["full_pipeline", "camera_capture_only"],
                "max_trial_frames": 12, "max_trial_duration_ms": 9000,
                "camera_sleep_supported": True,
                "models": {
                    "small": {"model_id": "small-model", "status": "experimental_unqualified"},
                    "large": {"model_id": "large-model", "status": "experimental_unqualified"},
                },
            }},
            receive_ns=30_000_000,
            receive_utc="2026-11-02T23:00:02+00:00",
        )
        capabilities = self.broker.capabilities()
        self.assertEqual(capabilities["policies"], ["H", "L", "S", "A-full"])
        self.assertEqual([row["id"] for row in capabilities["models"]], ["small", "large"])
        self.assertIn("trial", capabilities["commands"])
        self.assertEqual(capabilities["measurement_stage_ids"], ["full_pipeline", "camera_capture_only"])
        self.assertEqual(capabilities["max_trial_frames"], 12)
        self.assertEqual(capabilities["max_trial_duration_ms"], 9000)
        self.assertTrue(capabilities["camera_sleep_supported"])

    def test_trial_command_is_bounded_validated_and_forwarded_once(self):
        self.broker._device_ready = True
        self.broker._capabilities.update({
            "policies": ["S", "L"],
            "measurement_stage_ids": ["full_pipeline", "camera_capture_only"],
            "max_trial_frames": 12,
            "max_trial_duration_ms": 9000,
        })
        lease = self.broker.claim_controller("viewer_a")
        base = {"cmd": "trial", "frames": 5, "duration_ms": 5000,
                "trial_id": "trial_001", "measurement_stage_id": "full_pipeline",
                "model_slot": "small", "policy_id": "S",
                "controller_id": lease["controller_id"]}
        for suffix, update, reason in (
            ("bool-frames", {"frames": True}, "trial_frames_out_of_range"),
            ("too-many", {"frames": 13}, "trial_frames_out_of_range"),
            ("too-long", {"duration_ms": 9001}, "trial_duration_out_of_range"),
            ("bad-stage", {"measurement_stage_id": "identity_hopefully"}, "unsupported_measurement_stage"),
            ("bad-slot", {"model_slot": "medium"}, "unsupported_model_slot"),
            ("bad-policy", {"policy_id": "UNKNOWN"}, "unsupported_policy"),
            ("bad-id", {"trial_id": "spaces are invalid"}, "invalid_trial_id"),
        ):
            command = {**base, **update, "command_id": suffix}
            result = self.broker.command(command, viewer_id="viewer_a")
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["message"], reason)
        self.assertEqual(self.serial.writes, [])

        command = {**base, "command_id": "bounded_trial_1"}
        result = self.broker.command(command, viewer_id="viewer_a")
        self.assertEqual(result["status"], "accepted")
        wire = json.loads(self.serial.writes[-1].decode("utf-8"))
        self.assertEqual(wire["cmd"], "trial")
        self.assertEqual(wire["trial_id"], "trial_001")
        self.assertEqual(wire["measurement_stage_id"], "full_pipeline")
        self.assertEqual(wire["model_slot"], "small")
        write_count = len(self.serial.writes)
        duplicate = self.broker.command(command, viewer_id="viewer_a")
        self.assertEqual(duplicate["status"], "accepted")
        self.assertEqual(len(self.serial.writes), write_count)

    def test_trial_requires_unique_id_and_only_advertised_firmware_capabilities(self):
        self.broker._device_ready = True
        self.broker._capabilities.update({
            "commands": ["status", "start", "stop"],
            "measurement_stage_ids": ["camera_capture_only"],
            "max_trial_frames": 5,
            "max_trial_duration_ms": 1000,
        })
        capabilities = self.broker.capabilities()
        self.assertNotIn("trial", capabilities["commands"])
        lease = self.broker.claim_controller("viewer_a")
        result = self.broker.command({
            "cmd": "trial", "command_id": "trial_no_id", "frames": 1,
            "duration_ms": 500, "measurement_stage_id": "camera_capture_only",
            "trial_id": "trial_001", "controller_id": lease["controller_id"],
        }, viewer_id="viewer_a")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["message"], "unsupported_command")
        self.assertEqual(self.serial.writes, [])

        self.broker._capabilities["commands"] = [*self.broker.capabilities()["commands"], "trial"]
        missing_id = self.broker.command({
            "cmd": "trial", "command_id": "trial_missing_id", "frames": 1,
            "duration_ms": 500, "measurement_stage_id": "camera_capture_only",
            "controller_id": lease["controller_id"],
        }, viewer_id="viewer_a")
        self.assertEqual(missing_id["status"], "rejected")
        self.assertEqual(missing_id["message"], "invalid_trial_id")
        self.assertEqual(self.serial.writes, [])

    def test_trial_finished_updates_applied_state_to_stopped(self):
        self.broker._applied.update({"running": True, "trial_active": True})
        self.broker._handle_stream_item(
            {"type": "record", "record": {"event": "trial_finished",
                "boot_id": "boot-trial", "trial_id": "trial_001",
                "stage_id": "camera_capture_only", "reason_code": "frame_limit",
                "frame_index": 5, "running": False}},
            receive_ns=50_000_000,
            receive_utc="2026-11-03T13:00:00+00:00",
        )
        applied = self.broker.status()["applied_state"]
        self.assertFalse(applied["running"])
        self.assertFalse(applied["trial_active"])
        event = self.broker.events(after_seq=0, limit=10)["events"][-1]
        self.assertEqual(event["event"], "trial_finished")

    def test_packet_openmv_int8_dtype_code_is_canonicalized_for_api(self):
        payload = bytes([0, 1, 127, 255])
        self.broker._handle_stream_item(
            {"type": "packet", "header": {
                "schema_id": "nicla-packet-v1", "packet_version": 1,
                "kind": "identity_input", "boot_id": "boot-dtype-test",
                "frame_id": 8, "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [1, 1, 2, 2], "dtype": 98,
                "scale": 0.00392157, "zero_point": -128,
            }, "data": payload},
            receive_ns=40_000_000,
            receive_utc="2026-11-03T11:00:00+00:00",
        )
        actual, metadata = self.broker.get_input("boot-dtype-test", 8, "identity")
        self.assertEqual(actual, payload)
        self.assertEqual(metadata["dtype"], "int8")
        self.assertEqual(metadata["dtype_wire_value"], 98)
        event = self.broker.events(after_seq=0, limit=10)["events"][-1]
        self.assertEqual(event["metadata"]["dtype"], "int8")

    def test_failed_boot_handshake_releases_open_serial(self):
        self.broker._connected = False
        self.broker._serial = None
        self.broker._device_ready = False
        self.broker.serial_module = _SerialModule()
        self.broker.hello_timeout_s = 0.05
        self.broker.discover = lambda: {
            "device": "/dev/fake-nicla", "serial_number": "FAKE001",
            "vid": 0x2341, "pid": 0x045F,
        }
        status = self.broker.connect()
        self.assertFalse(status["connected"])
        self.assertFalse(status["serial_open"])
        self.assertEqual(status["connection_warning"], "nicla_hello_or_frame_timeout_after_usb_dtr_assert")
        self.assertTrue(_QuietSerial.latest.closed)


if __name__ == "__main__":
    unittest.main()
