"""Deterministic HID transport fixtures. No physical device is opened."""

from collections import Counter
import csv
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from nicla_study.instruments.windows_fnirsi import dashboard_meter, receipt_clock
from nicla_study.instruments.windows_fnirsi.offline_audit import audit_run
from nicla_study.instruments.windows_fnirsi.upstream.power.fnirsi_protocol import (
    PID, VID, crc8, decode_report, make_command, summarize_report,
)
from nicla_study.instruments.windows_fnirsi.upstream.scripts.hardware_lock_windows import HardwareLock


def interface(path=b"fixture-report-path", number=7, serial="test-serial"):
    return {
        "path": path, "interface_number": number, "serial_number": serial,
        "product_string": "Fixture FNIRSI HID", "vendor_id": VID, "product_id": PID,
        "usage_page": 0xFF00, "usage": 1,
    }


def report():
    packet = bytearray(64)
    packet[:2] = b"\xaa\x04"
    for slot, (voltage, current) in enumerate(zip(
        (480000, 490000, 510000, 520000), (5000, 8000, 12000, 15000),
    )):
        offset = 2 + slot * 15
        packet[offset:offset + 4] = voltage.to_bytes(4, "little")
        packet[offset + 4:offset + 8] = current.to_bytes(4, "little")
    packet[-1] = crc8(packet[1:-1])
    return bytes(packet)


class FixtureHid:
    def __init__(self):
        self.interfaces = [interface()]
        self.log = []
        self.responses = Queue()

    def enumerate(self, vid, pid):
        self.log.append(("enumerate", threading.get_ident(), (vid, pid)))
        return list(self.interfaces)

    def device(self):
        return self

    def open_path(self, path):
        self.log.append(("open", threading.get_ident(), path))

    def write(self, packet):
        self.log.append(("write", threading.get_ident(), bytes(packet)))
        return len(packet)

    def read(self, size, timeout_ms):
        self.log.append(("read", threading.get_ident(), (size, timeout_ms)))
        try:
            result = self.responses.get(timeout=0.025)
        except Empty:
            return []
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.log.append(("close", threading.get_ident(), None))


class FixtureLock:
    def __init__(self, trace):
        self.trace = trace

    def __enter__(self):
        self.trace.append(("lock", threading.get_ident(), None))

    def __exit__(self, *_):
        self.trace.append(("unlock", threading.get_ident(), None))


class DashboardPowerMeterTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nicla dashboard meter fixture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hid = FixtureHid()
        self.lock_paths = []

    def make_meter(self, **options):
        def factory(path, session_id, timeout_s):
            self.lock_paths.append(path)
            return FixtureLock(self.hid.log)

        arguments = dict(platform_name="nt", hid_loader=lambda: self.hid, lock_factory=factory)
        arguments.update(options)
        meter = dashboard_meter.PowerMeter(self.root, **arguments)
        self.addCleanup(meter.close)
        return meter

    def wait_for(self, meter, condition):
        deadline = time.monotonic() + 4
        status = meter.status()
        while not condition(status) and time.monotonic() < deadline:
            time.sleep(0.01)
            status = meter.status()
        self.assertTrue(condition(status), status)
        return status

    def connect_fixture(self, meter):
        selected = meter.scan()["interfaces"][0]["id"]
        meter.connect(selected)
        return self.wait_for(meter, lambda status: status["connected"])

    def test_lazy_platform_fallback_missing_dependency_and_absent_meter(self):
        def no_hid():
            raise ImportError("fixture missing dependency")

        meter = self.make_meter(platform_name="posix", hid_loader=no_hid)
        self.assertEqual(meter.status()["state"], "unsupported")
        self.assertEqual(meter.scan()["state"], "unsupported")
        meter = self.make_meter(hid_loader=no_hid)
        self.assertEqual(meter.status()["state"], "unknown")
        self.assertEqual(meter.samples()["samples"], [])
        self.assertEqual(meter.scan()["state"], "dependency_missing")
        meter = self.make_meter()
        self.hid.interfaces = []
        self.assertEqual(meter.scan()["state"], "absent")
        self.assertEqual([event[0] for event in self.hid.log], ["enumerate"])

    def test_passive_scan_is_bounded_and_never_acquires_hardware(self):
        unblock = threading.Event()
        self.addCleanup(unblock.set)

        def slow_hid():
            unblock.wait(3)
            return self.hid

        meter = self.make_meter(hid_loader=slow_hid)
        with patch.object(dashboard_meter, "SCAN_WAIT_SECONDS", 0.02):
            self.assertTrue(meter.scan()["scan_in_progress"])
        self.assertEqual(self.lock_paths, [])
        self.assertEqual(meter.samples()["samples"], [])
        unblock.set()
        status = self.wait_for(meter, lambda status: not status["scan_in_progress"])
        self.assertEqual(status["state"], "available")
        self.assertEqual([event[0] for event in self.hid.log], ["enumerate"])

    def test_multiple_candidates_and_changed_descriptor_need_explicit_selection(self):
        meter = self.make_meter()
        self.hid.interfaces.append(interface(b"other fixture path", number=3))
        scanned = meter.scan()
        self.assertEqual(scanned["state"], "ambiguous")
        with self.assertRaises(ValueError):
            meter.connect("")
        first_id, second_id = [item["id"] for item in scanned["interfaces"]]
        self.hid.interfaces[0] = interface(serial="different meter at the same HID path")
        meter.connect(first_id)
        self.wait_for(meter, lambda status: status["state"] == "ambiguous")
        self.assertNotIn("open", [event[0] for event in self.hid.log])
        meter.connect(second_id)
        self.wait_for(meter, lambda status: status["connected"])
        meter.disconnect()
        opened = [event[2] for event in self.hid.log if event[0] == "open"]
        self.assertEqual(opened, [b"other fixture path"])

    def test_recording_preserves_raw_receipts_and_matches_offline_four_slot_values(self):
        meter = self.make_meter()
        self.connect_fixture(meter)
        started = meter.start_recording(True)
        self.assertEqual(started["recording"]["state"], "running")
        self.assertTrue(started["topology_confirmed"])
        self.wait_for(meter, lambda status: status["timeout_count"] > 0)

        good = report()
        invalid = good[:-1] + bytes([good[-1] ^ 1])
        self.hid.responses.put(good)
        self.wait_for(meter, lambda status: status["latest"] is not None)
        # Windows Python 3.12's monotonic counter can tick at 15.6 ms.
        # Leave time for a counter tick without triggering the fixture's timeout.
        time.sleep(0.02)
        self.hid.responses.put(invalid)
        self.hid.responses.put(good)
        self.hid.responses.put(good)
        state = self.wait_for(meter, lambda status: (status["latest"] or {}).get("cursor", 0) >= 3)
        self.assertEqual(state["state"], "receiving")
        self.assertEqual(state["invalid_count"], 1)
        self.assertEqual(state["crc_fail_count"], 1)
        self.assertEqual(state["duplicate_count"], 1)
        self.assertGreater(state["report_rate_hz"], 0)
        points = meter.samples()["samples"]
        self.assertTrue(points[1]["gap_before"])
        self.assertFalse(points[2]["gap_before"])
        with self.assertRaises(ValueError):
            meter.samples(limit=121)
        page = meter.samples(after=points[0]["cursor"], limit=1)
        self.assertEqual(page["next_cursor"], points[1]["cursor"])
        stopped = meter.stop_recording()["recording"]
        self.assertEqual(stopped["state"], "complete")

        run = Path(stopped["output_dir"])
        metadata = json.loads((run / "run_metadata.json").read_text())
        self.assertEqual(metadata["topology_id"], dashboard_meter.TOPOLOGY_ID)
        self.assertEqual(metadata["meter_hid_descriptor"]["serial_number"], "test-serial")
        self.assertEqual(metadata["host_utc_clock"], receipt_clock.UTC_SOURCE)
        raw = [json.loads(line) for line in (run / "raw_packets.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["record_type"] == "timeout" for row in raw))
        self.assertEqual([row["raw_report_hex"] for row in raw if row["record_type"] == "invalid_report"],
                         [invalid.hex()])
        valid = [row for row in raw if row["record_type"] == "measurement_report"]
        self.assertEqual([row["host_receive_utc"] for row in valid], [p["host_receive_utc"] for p in points])
        with (run / "decoded_samples.csv").open(newline="") as stream:
            decoded = list(csv.DictReader(stream))
        self.assertEqual(len(decoded), 12)
        self.assertEqual(Counter(row["host_receive_monotonic_ns"] for row in decoded),
                         Counter(str(row["host_receive_monotonic_ns"]) for row in valid for _ in range(4)))
        expected = summarize_report(decode_report(good))
        self.assertAlmostEqual(points[0]["voltage_v"], expected["voltage_mean_v"])
        self.assertAlmostEqual(points[0]["current_ma"], expected["current_mean_a"] * 1000)
        self.assertAlmostEqual(points[0]["power_w"], expected["power_mean_w"])
        self.assertNotAlmostEqual(points[0]["power_w"], points[0]["voltage_v"] * points[0]["current_ma"] / 1000)
        audit = audit_run(run)
        self.assertEqual(audit["counts"]["decoded_crc_valid_reports"], 3)
        self.assertEqual(audit["counts"]["crc_rejections"], 1)
        self.assertEqual(audit["counts"]["logged_raw_sample_field_mismatch_reports_or_fields"], 0)

        meter.disconnect()
        owner = [entry for entry in self.hid.log if entry[0] != "enumerate"]
        self.assertEqual(len({entry[1] for entry in owner}), 1)
        self.assertNotEqual(owner[0][1], threading.get_ident())
        self.assertEqual([entry[0] for entry in owner[:2]], ["lock", "open"])
        self.assertEqual([entry[0] for entry in owner[-2:]], ["close", "unlock"])
        self.assertEqual(self.lock_paths, [(self.root / "logs" / "hardware.lock").resolve()])
        self.assertEqual(owner[2][2], b"\0" + make_command(0x81))
        self.assertEqual(owner[3][2], b"\0" + make_command(0x82))
        self.assertEqual(owner[4][2], (64, 250))

    def test_stale_unplug_and_explicit_reconnect_preserve_partial_run(self):
        meter = self.make_meter()
        first = self.connect_fixture(meter)
        self.hid.responses.put(report())
        self.wait_for(meter, lambda status: status["state"] == "receiving")
        with patch.object(dashboard_meter, "STALE_NS", 25_000_000):
            self.wait_for(meter, lambda status: status["state"] == "stale")
        recording = meter.start_recording()["recording"]
        self.hid.responses.put(report())
        self.hid.responses.put(OSError("fixture unplug"))
        failed = self.wait_for(meter, lambda status: status["state"] == "error")
        self.assertEqual(failed["recording"]["state"], "failed")
        self.assertFalse(failed["connected"])
        self.assertIsNone(failed["latest"])
        self.assertEqual(audit_run(Path(recording["output_dir"]))["counts"]["decoded_samples"], 4)
        self.assertEqual([entry[0] for entry in self.hid.log].count("open"), 1)
        second = self.connect_fixture(meter)
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertFalse(second["topology_confirmed"])
        self.hid.responses.put(report())
        latest = self.wait_for(meter, lambda status: status["state"] == "receiving")["latest"]
        self.assertEqual(latest["cursor"], 3)
        self.assertTrue(latest["gap_before"])
        run = meter.start_recording()["recording"]
        meter.close()
        self.assertEqual(meter.status()["recording"]["state"], "interrupted")
        metadata = json.loads((Path(run["output_dir"]) / "run_metadata.json").read_text())
        self.assertEqual(metadata["status"], "interrupted")

    def test_stop_without_valid_reports_preserves_failed_capture(self):
        meter = self.make_meter()
        self.connect_fixture(meter)
        timeouts = meter.start_recording()["timeout_count"]
        self.wait_for(meter, lambda status: status["timeout_count"] > timeouts)
        finished = meter.stop_recording()["recording"]
        self.assertEqual(finished["state"], "failed")
        raw = audit_run(Path(finished["output_dir"]))
        self.assertGreater(raw["counts"]["record_type_counts"]["timeout"], 0)
        self.assertEqual(raw["counts"]["decoded_crc_valid_reports"], 0)

    def test_recording_uses_run_local_arrival_indices_after_preview(self):
        meter = self.make_meter()
        self.connect_fixture(meter)
        self.hid.responses.put(report())
        self.wait_for(meter, lambda status: status["latest"] is not None)
        started = meter.start_recording()
        self.wait_for(meter, lambda status: status["timeout_count"] > started["timeout_count"])
        self.hid.responses.put(report())
        self.wait_for(meter, lambda status: status["latest"]["cursor"] == 2)
        recording = meter.stop_recording()["recording"]
        raw_path = Path(recording["output_dir"]) / "raw_packets.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        receipts = [(row["record_type"], row["report_arrival_index"])
                    for row in rows if "report_arrival_index" in row]
        self.assertEqual(receipts[0], ("timeout", 0))
        self.assertEqual([row for row in receipts if row[0] == "measurement_report"],
                         [("measurement_report", 0)])

    @unittest.skipUnless(os.name == "nt", "Windows named mutex")
    def test_windows_mutex_blocks_cli_process_and_transfers_after_disconnect(self):
        meter = self.make_meter(lock_factory=HardwareLock)
        interface_id = meter.scan()["interfaces"][0]["id"]
        with HardwareLock(meter.lock_path, "fixture-cli-held"):
            meter.connect(interface_id)
            busy = self.wait_for(meter, lambda status: status["state"] == "busy")
            self.assertFalse(busy["connected"])
            self.assertNotIn("open", [entry[0] for entry in self.hid.log])
        self.connect_fixture(meter)
        contender = (
            "import sys\n"
            "from nicla_study.instruments.windows_fnirsi.upstream.scripts.hardware_lock_windows "
            "import HardwareLock\n"
            "try:\n"
            "    with HardwareLock(sys.argv[1], 'fixture-cli-child'):\n"
            "        pass\n"
            "except TimeoutError:\n"
            "    sys.exit(6)\n"
        )
        command = [sys.executable, "-c", contender, str(meter.lock_path)]
        self.assertEqual(subprocess.run(command, cwd=APP, timeout=10).returncode, 6)
        meter.disconnect()
        self.assertEqual(subprocess.run(command, cwd=APP, timeout=10).returncode, 0)
        lock_receipt = json.loads(meter.lock_path.read_text())
        self.assertEqual(lock_receipt["state"], "released")
        self.assertEqual(lock_receipt["run_id"], "fixture-cli-child")


class ReceiptClockTest(unittest.TestCase):
    def test_utc_format_preserves_microseconds_with_no_float_rounding(self):
        self.assertEqual(receipt_clock.iso_utc_ns(1_735_689_600_123_456_789),
                         "2025-01-01T00:00:00.123456+00:00")

    @unittest.skipUnless(os.name == "nt", "Windows native UTC clock")
    def test_native_utc_has_sub_tick_values_and_the_unix_utc_epoch(self):
        coarse_before = time.time_ns()
        stamps = [receipt_clock.host_utc_ns() for _ in range(128)]
        coarse_after = time.time_ns()
        self.assertEqual(receipt_clock.UTC_SOURCE, "GetSystemTimePreciseAsFileTime")
        self.assertGreater(len(set(stamps)), 5)
        self.assertLess(abs(stamps[0] - coarse_before), 500_000_000)
        self.assertLess(abs(stamps[-1] - coarse_after), 500_000_000)


if __name__ == "__main__":
    unittest.main()
