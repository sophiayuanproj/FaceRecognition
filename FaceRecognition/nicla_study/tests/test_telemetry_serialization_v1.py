import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import hashlib

FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))

from telemetry import Telemetry, _json_safe


class NumericWrapper:
    def __float__(self):
        return 3.25


class UnsupportedDiagnostic:
    pass


class FakeUsb:
    def __init__(self):
        self.writes = []

    def isconnected(self):
        return True

    def write(self, data):
        self.writes.append(data)
        return len(data)


class TelemetrySerializationTests(unittest.TestCase):
    def test_json_safe_normalizes_nested_openmv_scalars_and_objects(self):
        value = _json_safe({
            "numeric": NumericWrapper(),
            "unsupported": UnsupportedDiagnostic(),
            7: (1, 2.0),
            "bytes": b"private tensor bytes are not copied into JSON",
        })
        self.assertEqual(value["numeric"], 3.25)
        self.assertEqual(value["unsupported"], "<UnsupportedDiagnostic>")
        self.assertEqual(value["7"], [1, 2.0])
        self.assertEqual(value["bytes"], {
            "_type": "bytes", "byte_count": len(b"private tensor bytes are not copied into JSON")})

    def test_event_survives_non_json_diagnostic_value(self):
        config = {"telemetry": {"max_queue_bytes_per_sink": 4096,
                                 "max_record_bytes": 2048}}
        telemetry = Telemetry(None, None, config, lambda: 10, "boot-test", 0,
                              "f" * 64, "c" * 64, "s" * 64)
        record = telemetry.emit("decision", {
            "frame_id": 8,
            "status": "UNCERTAIN",
            "diagnostic": UnsupportedDiagnostic(),
        })
        self.assertEqual(record["event"], "decision")
        self.assertEqual(record["sequence"], 0)
        self.assertEqual(record["diagnostic"], "<UnsupportedDiagnostic>")

    def test_encode_failure_compacts_observation_and_keeps_release_hashes(self):
        config = {"telemetry": {"max_queue_bytes_per_sink": 4096,
                                 "max_record_bytes": 2048}}
        telemetry = Telemetry(None, None, config, lambda: 10, "boot-test", 0,
                              "f" * 64, "c" * 64, "s" * 64)
        original_dumps = __import__("json").dumps
        attempts = 0
        def fail_once_then_encode(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise MemoryError()
            return original_dumps(*args, **kwargs)
        with patch("telemetry.json.dumps", side_effect=fail_once_then_encode):
            record = telemetry.emit("observation_complete", {"frame_id": 2})
        self.assertEqual(record["event"], "observation_complete")
        self.assertTrue(record["record_truncated"])
        self.assertEqual(record["record_truncated_reason"], "observation_memory_compaction")
        self.assertEqual(record["sequence"], 0)
        self.assertEqual(record["firmware_sha256"], "f" * 64)
        self.assertEqual(record["model_config_sha256"], "c" * 64)

    def test_packet_sends_model_owned_bytearray_without_full_copy(self):
        config = {"telemetry": {"max_queue_bytes_per_sink": 4096,
                                 "max_record_bytes": 2048,
                                 "flush_bytes_per_loop": 4096}}
        usb = FakeUsb()
        telemetry = Telemetry(usb, None, config, lambda: 10, "boot-test", 0,
                              "f" * 64, "c" * 64, "s" * 64)
        payload = bytearray(range(256)) * 36
        with patch("telemetry.time.ticks_ms", return_value=0, create=True):
            result = telemetry.send_packet("identity_input", {"frame_id": 7}, payload)
        self.assertTrue(result["sent"])
        self.assertEqual(result["byte_count"], len(payload))
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertIsInstance(usb.writes[0], bytes)
        self.assertIsInstance(usb.writes[1], bytearray)


if __name__ == "__main__":
    unittest.main()
