import json
from http.client import HTTPConnection
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from control_center.server import ControlCenterHTTPServer, make_handler


class PowerRouteTests(unittest.TestCase):
    def setUp(self):
        self.meter = Mock()
        self.meter.status.return_value = {"state": "unsupported"}
        self.meter.samples.return_value = {"samples": [], "next_cursor": 7}
        self.meter.connect.return_value = {"state": "connecting"}
        self.app = SimpleNamespace(power_meter=self.meter, stop_requested=False)
        self.server = ControlCenterHTTPServer(("127.0.0.1", 0), make_handler(self.app, {}))
        thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, method, route, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method, f"/api/v2/power/{route}", body, headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_status_and_sample_reads_do_not_open_hardware(self):
        self.assertEqual(self.request("GET", "status"), (200, {"state": "unsupported"}))
        self.assertEqual(self.request("GET", "samples?after=5&limit=2")[0], 200)
        self.meter.samples.assert_called_once_with(5, 2)
        self.meter.scan.assert_not_called()
        self.meter.connect.assert_not_called()
        self.assertEqual(self.request("GET", "samples?after=invalid")[0], 400)

    def test_connect_requires_a_local_json_action(self):
        body = '{"interface_id":"fresh-interface"}'
        self.assertEqual(self.request("POST", "connect", body)[0], 403)
        headers = {"X-Control-Center-Client": "test-viewer", "Content-Type": "application/json"}
        for extra in ({"Origin": "https://example.org"}, {"Host": "example.org"},
                      {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(extra=extra):
                self.assertEqual(self.request("POST", "connect", body, headers | extra)[0], 403)
        self.meter.connect.assert_not_called()
        self.assertEqual(self.request("POST", "connect", body, headers | {"Origin": self.origin}),
                         (200, {"state": "connecting"}))
        self.meter.connect.assert_called_once_with("fresh-interface")

    def test_start_rejects_conflict_and_shutdown(self):
        headers = {"X-Control-Center-Client": "test-viewer", "Content-Type": "application/json"}
        self.meter.start_recording.side_effect = RuntimeError("Meter is busy")
        self.assertEqual(self.request("POST", "start-recording", '{"topology_confirmed":true}', headers),
                         (409, {"error": "Meter is busy"}))
        self.meter.start_recording.assert_called_once_with(True)
        self.app.stop_requested = True
        self.assertEqual(self.request("POST", "scan", '{}', headers)[0], 503)
        self.meter.scan.assert_not_called()
