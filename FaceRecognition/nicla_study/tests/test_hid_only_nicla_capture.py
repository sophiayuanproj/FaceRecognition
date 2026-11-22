"""Regression check for release-bound HID capture beside a dashboard."""

from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nicla_study.instruments.windows_fnirsi import capture_windows
from nicla_study.tests.test_offline_fnirsi_audit import make_report


FIRMWARE = "1" * 64
BOOT = "boot-fixture"
URL = "http://127.0.0.1:8780"


def dashboard_response(config_sha, **overrides):
    status = {"schema_id": "nicla-device-status-v2", "connected": True, "serial_open": True,
              "device_ready": True, "running": False, "sleeping": False, "boot_id": BOOT,
              "host_last_receive_age_ms": 1200, "device": {"firmware_sha256": FIRMWARE,
              "config_sha256": config_sha}, **overrides}
    response = Mock(status=200)
    response.read.return_value = json.dumps(status).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    return connection


class HidOnlyNiclaCaptureTests(unittest.TestCase):
    def test_meter_capture_binds_local_release_and_live_dashboard_without_serial(self):
        with tempfile.TemporaryDirectory(prefix="Nicla HID-only Ω ") as temp:
            root = Path(temp)
            out = root / "run"
            hashes = {}
            for name in ("source.py", "release.mpy", "runtime_config.json"):
                payload = name.encode()
                (root / name).write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
            config_sha = hashes["runtime_config.json"]
            cli = ["--project-root", str(root), "--output-dir", str(out), "--run-id", "unit-test",
                   "--duration", ".02", "--warmup-seconds", "0", "--interface-number", "2",
                   "--board-type", "nicla_vision", "--board-label", "Nicla Vision", "--device-reset-id", BOOT,
                   "--firmware-source", "source.py", "--release-payload", "release.mpy",
                   "--board-configuration", "runtime_config.json", "--nicla-dashboard-url", URL,
                   "--expected-device-firmware-sha256", FIRMWARE, "--expected-device-config-sha256", config_sha,
                   "--expected-firmware-source-sha256", hashes["source.py"],
                   "--expected-release-payload-sha256", hashes["release.mpy"]]
            connection = dashboard_response(config_sha)
            meter = Mock()
            meter.write.return_value = 65
            meter.read.return_value = make_report([(500000, 4700, 0)] * 4)
            hid = SimpleNamespace(device=Mock(return_value=meter), enumerate=Mock(return_value=[{
                "vendor_id": 0x2E3C, "product_id": 0x5558, "interface_number": 2, "path": b"fixture-hid"}]))
            with (patch.dict(sys.modules, hid=hid),
                  patch.object(capture_windows, "HardwareLock", return_value=nullcontext()),
                  patch.object(capture_windows.http.client, "HTTPConnection", return_value=connection),
                  patch.object(capture_windows.importlib.metadata, "version", return_value="fixture"),
                  patch.object(capture_windows, "NiclaRawSerialCapture") as serial_capture,
                  patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(capture_windows.main(cli), 0)
                serial_capture.assert_not_called()
                meter.open_path.assert_called_once_with(b"fixture-hid")
                connection.request.assert_called_once_with("GET", "/api/v2/device/status")
            metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["selected_board"]["configuration_sha256"], config_sha)
            self.assertEqual(metadata["selected_board"]["firmware_source_sha256"], hashes["source.py"])
            receipt = json.loads((out / "nicla_dashboard_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["nicla_dashboard_preflight"], receipt)
            self.assertEqual(receipt["boot_id"], BOOT)
            self.assertFalse(receipt["serial_opened_by_logger"])
            self.assertTrue((out / "decoded_samples.csv").is_file())

    def test_preflight_refuses_wrong_boot_stale_running_or_wrong_config(self):
        config_sha = "2" * 64
        cases = [(dict(boot_id="other"), "boot ID"), (dict(host_last_receive_age_ms=9000), "recent"),
                 (dict(running=True), "stopped"), (dict(device={}), "hashes")]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), patch.object(capture_windows.http.client, "HTTPConnection",
                    return_value=dashboard_response(config_sha, **overrides)):
                with self.assertRaisesRegex(ValueError, message):
                    capture_windows.read_dashboard_preflight(URL, BOOT, FIRMWARE, config_sha)
        with self.assertRaisesRegex(ValueError, "local URL"):
            capture_windows.read_dashboard_preflight("http://example.com:8780", BOOT, FIRMWARE, config_sha)


if __name__ == "__main__":
    unittest.main()
