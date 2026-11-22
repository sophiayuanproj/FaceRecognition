"""Offline acceptance checks for launch, port discovery and clean stop/restart."""

from __future__ import annotations

import json
import errno
import io
import os
import re
import select
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import urllib.request
from pathlib import Path

from control_center import server as server_module
from host.nicla_broker import NiclaBroker


WORKSPACE = Path(__file__).resolve().parents[3]
LAUNCHER = WORKSPACE / "FaceRecognition" / "bin" / "nicla-control-center.command"


class LauncherAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.cwd = tempfile.TemporaryDirectory(prefix="nicla-control-center-launch-")
        self.processes: list[subprocess.Popen[str]] = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        self.cwd.cleanup()

    def _spawn(self, *args: str) -> tuple[subprocess.Popen[str], str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.Popen(
            [str(LAUNCHER), *args],
            cwd=self.cwd.name,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.processes.append(process)
        assert process.stdout is not None
        ready_line = None
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            readable, _, _ = select.select([process.stdout], [], [], 0.25)
            if readable:
                line = process.stdout.readline()
                if line.startswith("NICLA_CONTROL_CENTER "):
                    ready_line = line.strip()
                    break
                if process.poll() is not None:
                    break
        if ready_line is None:
            if process.poll() is None:
                process.terminate()
            output, _ = process.communicate(timeout=5)
            self.fail(f"launcher did not print the selected URL; exit={process.returncode}; output={output}")
        match = re.fullmatch(r"NICLA_CONTROL_CENTER (http://[^\s]+/)", ready_line)
        self.assertIsNotNone(match, ready_line)
        return process, match.group(1)

    def _doctor(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [str(LAUNCHER), "--doctor", *args],
            cwd=self.cwd.name,
            env=env,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )

    def test_doctor_reports_os_assigned_port_and_rejects_occupied_port(self):
        ready = self._doctor("--port", "0")
        self.assertEqual(ready.returncode, 0, ready.stdout)
        self.assertIn("port discovery: OS selects an available port when launched", ready.stdout)
        self.assertIn("result: ready", ready.stdout)
        self.assertIn("explicit Connect", ready.stdout)
        self.assertIn("pyserial", ready.stdout)

        stream = io.StringIO()
        with patch("control_center.server.socket.socket") as make_socket:
            fake_socket = make_socket.return_value.__enter__.return_value
            fake_socket.bind.side_effect = OSError(errno.EADDRINUSE, "address in use")
            with redirect_stdout(stream):
                exit_code = server_module.doctor("127.0.0.1", 8780, "http://127.0.0.1:8766")
        self.assertEqual(exit_code, 1)
        self.assertIn("port 8780 cannot be bound", stream.getvalue())

    def test_cli_formats_actual_bound_url_and_handles_sigterm_without_binding(self):
        class StubApp:
            def __init__(self, preview_url):
                self.preview_url = preview_url
                self.stop_requested = False
                self.closed = False

            def close(self):
                self.closed = True

        class StubHTTPServer:
            def __init__(self, address, handler):
                self.server_address = ("127.0.0.1", 54321)
                self._shutdown_lock = __import__("threading").Lock()
                self._shutdown_requested = False
                self.shutdown_complete = __import__("threading").Event()
                self.closed = False

            def serve_forever(self, poll_interval):
                signal_handler = signal.getsignal(signal.SIGTERM)
                signal_handler(signal.SIGTERM, None)
                if not self.shutdown_complete.wait(timeout=2):
                    raise AssertionError("SIGTERM did not request background shutdown")

            def shutdown(self):
                self.shutdown_complete.set()

            def server_close(self):
                self.closed = True

        app_ref = {}
        server_ref = {}
        stream = io.StringIO()

        class TrackingApp(StubApp):
            def __init__(self, preview_url):
                super().__init__(preview_url)
                app_ref["app"] = self

        def fake_server(address, handler):
            instance = StubHTTPServer(address, handler)
            server_ref["server"] = instance
            return instance

        with patch.object(server_module, "ControlCenter", TrackingApp), patch.object(
            server_module, "ControlCenterHTTPServer", side_effect=fake_server
        ), redirect_stdout(stream):
            exit_code = server_module.main(["--port", "0"])
        self.assertEqual(exit_code, 0)
        self.assertIn("NICLA_CONTROL_CENTER http://127.0.0.1:54321/", stream.getvalue())
        self.assertIn("Stopping control center (SIGTERM)", stream.getvalue())
        self.assertIn("Control center stopped.", stream.getvalue())
        self.assertTrue(app_ref["app"].closed)
        self.assertTrue(app_ref["app"].stop_requested)
        self.assertTrue(server_ref["server"].closed)

    def test_main_reports_port_zero_recovery_when_bind_is_occupied(self):
        class StubApp:
            def __init__(self, preview_url):
                self.closed = False

            def close(self):
                self.closed = True

        stream = io.StringIO()
        with patch.object(server_module, "ControlCenter", StubApp), patch.object(
            server_module,
            "ControlCenterHTTPServer",
            side_effect=OSError(errno.EADDRINUSE, "address in use"),
        ), patch("sys.stderr", stream):
            exit_code = server_module.main(["--port", "8780"])
        self.assertEqual(exit_code, 2)
        self.assertIn("port 8780", stream.getvalue())
        self.assertIn("--port 0", stream.getvalue())

    def test_live_and_replay_ui_labels_cover_stale_disconnect_and_attach_failure(self):
        app_js = (WORKSPACE / "FaceRecognition" / "control_center" / "web" / "app.js").read_text(encoding="utf-8")
        for label in (
            "RECORDED REPLAY",
            "LEGACY PREVIEW · READ ONLY",
            "LEGACY PREVIEW · STALE",
            "PREVIEW OFFLINE",
            "STALE / AGE UNKNOWN",
            "Preview attachment failed; no live image or result is shown.",
        ):
            with self.subTest(label=label):
                self.assertIn(label, app_js)

    def test_one_action_url_shutdown_sigterm_and_same_port_restart(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as error:
            self.skipTest(f"sandbox forbids local loopback bind: {error}")
        process, url = self._spawn("--port", "0")
        self.assertRegex(url, r"^http://127\.0\.0\.1:[1-9][0-9]*/$")
        with urllib.request.urlopen(url + "api/v1/health", timeout=3) as response:
            health = json.loads(response.read())
        self.assertEqual(health["mode"], "replay")
        self.assertFalse(health["live_attached"])
        self.assertIn("NiclaBroker", health["serial_owner"])
        self.assertFalse(health["serial_opened_by_service"])
        self.assertFalse(health["device_connected"])
        self.assertEqual(health["version"], "0.2.0")

        stop_request = urllib.request.Request(
            url + "api/v1/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(stop_request, timeout=3) as response:
            acknowledgement = json.loads(response.read())
        self.assertTrue(acknowledgement["ok"])
        self.assertIn("Preview owner is unchanged", acknowledgement["message"])
        first_output, _ = process.communicate(timeout=8)
        self.assertEqual(process.returncode, 0, first_output)
        self.assertIn("Stopping control center (dashboard request)", first_output)
        self.assertIn("Control center stopped.", first_output)

        port = int(url.rsplit(":", 1)[1].rstrip("/"))
        restarted, restarted_url = self._spawn("--port", str(port))
        self.assertEqual(restarted_url, url)
        restarted.send_signal(signal.SIGTERM)
        second_output, _ = restarted.communicate(timeout=8)
        self.assertEqual(restarted.returncode, 0, second_output)
        self.assertIn("Stopping control center (SIGTERM)", second_output)
        self.assertIn("Control center stopped.", second_output)


class DeviceConnectAuthorityTests(unittest.TestCase):
    def _request_handler(self, path, body, broker):
        handler_class = server_module.make_handler(
            type("App", (), {"device_broker": broker, "device_broker_import_error": None})(),
            {},
        )
        handler = object.__new__(handler_class)
        handler.path = path
        handler.headers = {"X-Viewer-ID": "viewer_a"}
        handler.read_body = lambda: body
        handler.viewer_id = lambda: "viewer_a"
        handler.require_device_broker = lambda: broker
        handler.send_json = lambda status, payload: (status, payload)
        return handler

    def test_expired_controller_cannot_change_usb_connection_and_can_claim_again(self):
        now = [100.0]
        with tempfile.TemporaryDirectory() as directory:
            broker = NiclaBroker(lock_path=Path(directory) / "serial.lock", monotonic=lambda: now[0])
            self.addCleanup(broker.close)
            controller = broker.claim_controller("viewer_a")["controller_id"]
            self.assertIsNone(broker.status("viewer_b")["controller_authority"]["controller_id"])
            now[0] += broker.controller_lease_s
            self.assertFalse(broker.status("viewer_a")["controller_authority"]["owned_by_viewer"])
            self.assertIsNone(broker.status("viewer_a")["controller_authority"]["controller_id"])

            with patch.object(broker, "connect") as connect, patch.object(broker, "disconnect") as disconnect:
                for route in ("connect", "disconnect", "controller/heartbeat", "controller/release"):
                    with self.subTest(route=route):
                        handler = self._request_handler("/api/v2/device/" + route, {
                            "viewer_id": "viewer_a", "controller_id": controller}, broker)
                        code, payload = handler.do_POST()
                        self.assertEqual(code, 409)
                        self.assertIn("expired", payload["error"])
                connect.assert_not_called()
                disconnect.assert_not_called()

                current = broker.claim_controller("viewer_a")["controller_id"]
                self.assertNotEqual(controller, current)
                for route, expected in (("controller/heartbeat", "renewed"), ("controller/release", "released")):
                    handler = self._request_handler("/api/v2/device/" + route, {
                        "viewer_id": "viewer_a", "controller_id": current}, broker)
                    code, payload = handler.do_POST()
                    self.assertEqual((code, payload["status"]), (200, expected))
                self.assertFalse(broker.status("viewer_a")["controller_authority"]["owned_by_viewer"])

    def test_read_only_viewer_cannot_open_or_close_serial(self):
        class Broker:
            def __init__(self):
                self.connect_calls = 0
                self.disconnect_calls = 0

            def status(self, viewer_id=None):
                return {"controller_authority": {"viewer_id": "owner_b", "owned_by_viewer": False, "lease_remaining_ms": 12000}}

            def heartbeat_controller(self, controller_id, viewer_id):
                raise PermissionError("another viewer holds the controller lease")

            def connect(self):
                self.connect_calls += 1
                return {"connected": True}

            def disconnect(self):
                self.disconnect_calls += 1
                return {"connected": False}

        broker = Broker()
        connect = self._request_handler("/api/v2/device/connect", {"schema": "nicla-connect-v2", "viewer_id": "viewer_a", "controller_id": "stale-lease"}, broker)
        status, payload = type(connect).do_POST(connect)
        self.assertEqual(status, 409)
        self.assertEqual(payload["status"], "read_only")
        self.assertEqual(broker.connect_calls, 0)

        disconnect = self._request_handler("/api/v2/device/disconnect", {"schema": "nicla-disconnect-v2", "viewer_id": "viewer_a", "controller_id": "stale-lease"}, broker)
        status, payload = type(disconnect).do_POST(disconnect)
        self.assertEqual(status, 409)
        self.assertEqual(payload["status"], "read_only")
        self.assertEqual(broker.disconnect_calls, 0)

    def test_connect_route_requires_matching_active_controller_lease(self):
        class Broker:
            def __init__(self):
                self.connect_calls = 0
                self.lease_checks = 0

            def status(self, viewer_id=None):
                return {"controller_authority": {"viewer_id": viewer_id, "owned_by_viewer": True, "lease_remaining_ms": 12000}}

            def heartbeat_controller(self, controller_id, viewer_id):
                self.lease_checks += 1
                if controller_id != "lease-a" or viewer_id != "viewer_a":
                    raise PermissionError("lease mismatch")
                return {"status": "renewed", "controller_id": controller_id, "viewer_id": viewer_id}

            def connect(self):
                self.connect_calls += 1
                return {"connected": True}

        broker = Broker()
        handler = self._request_handler("/api/v2/device/connect", {"schema": "nicla-connect-v2", "viewer_id": "viewer_a", "controller_id": "lease-a"}, broker)
        status, payload = type(handler).do_POST(handler)
        self.assertEqual(status, 200)
        self.assertTrue(payload["connected"])
        self.assertEqual(broker.lease_checks, 1)
        self.assertEqual(broker.connect_calls, 1)


if __name__ == "__main__":
    unittest.main()
