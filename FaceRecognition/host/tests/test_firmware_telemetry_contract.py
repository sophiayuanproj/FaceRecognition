from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))
import telemetry  # noqa: E402


class _USB:
    def __init__(self):
        self.writes = []

    def isconnected(self):
        return True

    def write(self, data):
        self.writes.append(data)
        return len(data)


class FirmwareTelemetryContractTests(unittest.TestCase):
    def setUp(self):
        self.writer = telemetry.Telemetry(
            usb=_USB(), uart=None,
            config={"telemetry": {"max_queue_bytes_per_sink": 24_576,
                                  "max_record_bytes": 8192,
                                  "flush_bytes_per_loop": 4096}},
            ticks_ms=lambda: 25, run_id="boot-test", reset_cause=0,
            firmware_sha256="a" * 64, config_sha256="b" * 64,
            schema_sha256="c" * 64, ticks_modulus_ms=1 << 30,
        )

    @staticmethod
    def _observation():
        return {
            "type": "observation_complete", "source": "on_nicla",
            "boot_id": "boot-test", "frame_id": 41,
            "command_id": "one-shot-41",
            "frame": {"boot_id": "boot-test", "frame_id": 41,
                      "frame_ref": "/api/v2/device/frame?boot_id=boot-test&frame_id=41"},
            "detection": {"boot_id": "boot-test", "frame_id": 41, "status": "no_face"},
            "detector_input": {"boot_id": "boot-test", "frame_id": 41,
                               "sha256": "d" * 64, "shape": [1, 96, 96, 1]},
            "crop": None, "identity": {"status": "skipped"},
            "decision": {"frame_id": 41, "status": "NO_FACE",
                         "reason": "no_face_detected"},
            "trace": ["detector returned no face"],
            "qualification": {"statistically_qualified": False},
            "imu": None, "requested_state": None,
        }

    def test_observation_reuses_json_native_nested_fields(self):
        fields = self._observation()
        with mock.patch.object(telemetry, "_json_safe", side_effect=AssertionError("unexpected copy")):
            emitted = self.writer.emit("observation_complete", fields)
        self.assertEqual(emitted["schema_id"], "nicla-observation-v2")
        self.assertIs(emitted["frame"], fields["frame"])
        self.assertEqual(emitted["decision"]["frame_id"], emitted["frame_id"])

    def test_memory_pressure_compacts_but_keeps_same_frame_result(self):
        original = telemetry.json.dumps
        raised = {"once": False}

        def fail_first_observation(value, *args, **kwargs):
            if value.get("event") == "observation_complete" and not raised["once"]:
                raised["once"] = True
                raise MemoryError("injected constrained heap")
            return original(value, *args, **kwargs)

        with mock.patch.object(telemetry.json, "dumps", side_effect=fail_first_observation):
            emitted = self.writer.emit("observation_complete", self._observation())
        self.assertTrue(emitted["record_truncated"])
        self.assertEqual(emitted["record_truncated_reason"], "observation_memory_compaction")
        self.assertEqual(emitted["frame"]["frame_id"], 41)
        self.assertEqual(emitted["detector_input"]["frame_id"], 41)
        self.assertEqual(emitted["decision"]["frame_id"], 41)
        self.assertEqual(emitted["decision"]["status"], "NO_FACE")

    def test_live_face_record_fits_bounded_encoding_and_preserves_provenance(self):
        runtime_path = FIRMWARE / "main.py"
        runtime = next(node for node in ast.parse(runtime_path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == "NiclaRuntime")
        runtime.body = [node for node in runtime.body if isinstance(node, ast.FunctionDef)
                        and node.name in ("_observation_summary", "_frame_payloads")]
        scope = {}
        exec(compile(ast.Module(body=[runtime], type_ignores=[]), str(runtime_path), "exec"), scope)
        app = scope["NiclaRuntime"]()
        app.telemetry = self.writer
        app.running, app.sleeping, app.model_override = True, False, None
        app.policy_id, app.measurement_stage_id = "F", "full_pipeline"
        app.policy = {"identity_backend": "small"}
        app.config = {"identity": {"allow_known_identity_acceptance": False}}
        app.current_observation_command_id = None
        fields = {"frame_index": 43, "frame_shape": [240, 240], "face_count": 1,
                  "detector_status": "ONE_FACE", "detector_backend": "gray_haar_box",
                  "detector_regions": [{"rect_xywh_px": [67, 50, 65, 70], "score": 0.8}],
                  "crop_xywh": [57, 39, 90, 90], "crop_status": "valid", "model_id": "small_int8",
                  "input_sha256": "d" * 64, "input_shape": [1, 96, 96, 1],
                  "input_dtype": "int8", "input_byte_count": 9216,
                  "input_packet": {"status": "not_exported_live"}, "model_sha256": "e" * 64,
                  "payloads": [{"kind": "frame_jpeg", "sha256": "f" * 64}],
                  "status": "OTHER", "decision_reason": "candidate_below_acceptance_cutoff",
                  "candidate_identity_id": "K01", "candidate_score": 0.44,
                  "inference_us": 23_110, "controller_response_latency_ms": 177,
                  "python_heap_free_bytes": 15_000}
        compact = app._observation_summary(fields, compact=True)
        rich = app._observation_summary(fields)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(rich)) * 0.66)
        original = telemetry._encode_line

        def enforce_budget(record):
            encoded = original(record)
            if len(encoded) > 3000:
                raise MemoryError("injected contiguous allocation limit")
            return encoded

        with mock.patch.object(telemetry, "_encode_line", side_effect=enforce_budget):
            result = self.writer.emit("observation_complete", compact)
        self.writer.flush()
        wire = json.loads(self.writer.usb.writes[0])
        self.assertEqual(wire, json.loads(json.dumps(result)))
        self.assertEqual(wire["frame"]["sha256"], "f" * 64)
        self.assertEqual(wire["frame"]["boot_id"], wire["boot_id"])
        for stage in ("frame", "detection", "crop", "identity_input", "decision"):
            self.assertEqual(wire[stage]["frame_id"], 43)
        self.assertEqual(wire["detection"]["status"], "face")
        self.assertEqual(wire["identity"]["status"], "ran")
        self.assertEqual(wire["decision"]["status"], "OTHER")
        self.assertFalse(wire["decision"]["known_acceptance_enabled"])
        self.assertEqual(wire["timings_ms"], {"identity_predict": 23.11, "total": 177})
        self.assertEqual(wire["applied_state"]["measurement_stage_id"], "full_pipeline")

        def fail_to_cache(frame_fields):
            frame_fields["frame_encoding_error"] = "MemoryError:allocation failed"
            return False

        app._cache_current_frame_jpeg = fail_to_cache
        self.assertEqual(app._frame_payloads(fields), [])
        missing_image = app._observation_summary(fields, compact=True)
        self.assertIsNone(missing_image["frame"]["sha256"])
        self.assertEqual(missing_image["frame"]["encoding_error"], "MemoryError:allocation failed")

    def test_send_packet_streams_memoryview_without_tensor_sized_conversion(self):
        tensor = bytearray((index * 17) & 255 for index in range(9216))
        payload = memoryview(tensor)
        expected = hashlib.sha256(tensor).hexdigest()
        with mock.patch.object(telemetry.time, "ticks_ms", return_value=25, create=True):
            result = self.writer.send_packet("detector_input", {
                "boot_id": "boot-test", "frame_id": 42,
                "shape": [1, 96, 96, 1], "dtype": "int8"}, payload)
        self.assertTrue(result["sent"])
        self.assertEqual(result["sha256"], expected)
        self.assertTrue(all(isinstance(item, memoryview)
                            for item in self.writer.usb.writes[1:]))


if __name__ == "__main__":
    unittest.main()
