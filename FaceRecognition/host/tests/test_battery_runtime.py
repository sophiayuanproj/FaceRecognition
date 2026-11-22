import ast
import binascii
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class BatteryRuntimeTests(unittest.TestCase):
    def setUp(self):
        firmware = Path(__file__).resolve().parents[2] / "firmware/main.py"
        tree = ast.parse(firmware.read_text())
        runtime = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "NiclaRuntime")
        runtime.body = [node for node in runtime.body if isinstance(node, ast.FunctionDef)
                        and (node.name.startswith("_battery_") or node.name in (
                            "_record_battery_sighting", "_handle_battery_command"))]
        self.clock = SimpleNamespace(now=100, ticks_diff=lambda a, b: a - b)
        self.clock.ticks_ms = lambda: self.clock.now
        self.usb_mode = Mock(return_value="VCP")
        self.fs = Mock()
        scope = dict(time=self.clock, pyb=SimpleNamespace(usb_mode=self.usb_mode),
                     os=self.fs, binascii=binascii, BATTERY_ARM_PATH="/flash/sightings.armed",
                     BATTERY_CONFIRM_MS=1800, BATTERY_GAP_MS=2400)
        exec(compile(ast.Module(body=[runtime], type_ignores=[]), str(firmware), "exec"), scope)
        self.app = scope["NiclaRuntime"]()
        self.app.config = {"control": {"offline_sightings_enabled": True},
                           "default_policy": "F", "identity": {}}
        self.app.battery_armed = False
        self.app.battery_last_face = None
        self.app.battery_encounter_saved = False
        self.app.battery_store = Mock()
        self.app.battery_list_snapshot = None
        self.app.battery_error = None
        self.app.measurement_stage_id = "full_pipeline"
        self.app.current_observation_command_id = None
        self.app.telemetry = SimpleNamespace(run_id="boot-123", ticks_modulus_ms=1 << 30)
        self.app.current_frame_jpeg_bytes = b"fixture"
        self.app._cache_current_frame_jpeg = Mock(return_value=True)
        self.app._control_ack = Mock()
        self.app.sleeping = False
        self.app.set_policy = Mock()

    def record(self, frame, tick, faces):
        self.clock.now = tick
        self.app._record_battery_sighting({"status": "OTHER"}, {
            "frame_index": frame, "face_count": faces})

    def test_mounted_usb_cannot_arm_or_write(self):
        self.usb_mode.return_value = "VCP+MSC"
        self.app._handle_battery_command("usb", "battery_arm", {})
        self.assertEqual(self.app._control_ack.call_args.args[-1],
                         {"reason_code": "serial_only_usb_required"})
        self.fs.mkdir.assert_not_called()
        self.app.battery_armed = True
        self.record(1, 100, 1)
        self.record(2, 200, 1)
        self.app.battery_store.save.assert_not_called()

    def test_battery_arm_saves_once_per_encounter_before_the_face_leaves(self):
        self.app._battery_set_armed(True)
        self.fs.mkdir.assert_called_once_with("/flash/sightings.armed")
        self.assertTrue(self.app.running)
        self.app.set_policy.assert_called_once_with("F")
        self.record(1, 100, 1)
        self.app.battery_store.save.assert_not_called()
        self.record(2, 400, 1)
        self.record(3, 700, 1)
        self.record(4, 800, 0)
        self.record(5, 3150, 1)
        self.record(6, 3400, 1)
        self.assertEqual([row.kwargs["frame_id"] for row in
                          self.app.battery_store.save.call_args_list], [2, 6])
        self.app._battery_set_armed(False)
        self.fs.rmdir.assert_called_once_with("/flash/sightings.armed")

    def test_binary_chunk_roundtrip_is_bounded_and_usb_only(self):
        self.app.battery_store.read.return_value = b"\x00\xab\xff"
        request = {"record_id": 2, "offset": 12, "size": 3}
        self.app._handle_battery_command("uart", "battery_read", request)
        self.app._handle_battery_command("usb", "battery_read", dict(request, size=513))
        self.app.battery_store.read.assert_not_called()
        self.app._handle_battery_command("usb", "battery_read", request)
        self.app.battery_store.read.assert_called_once_with(2, 12, 3)
        self.assertEqual(self.app._control_ack.call_args.kwargs["fields"], {
            "record_id": 2, "offset": 12, "chunk_hex": "00abff"})
