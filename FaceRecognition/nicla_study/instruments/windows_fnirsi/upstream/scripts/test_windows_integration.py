"""Ten-root Windows fake-interface/path/interrupt/reconnect integration suite.

The fake HID transport exercises host code only. It is not hardware or
instrument qualification and is kept in a separate report from real captures.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "fake_hid"))

from device_selector_windows import choose_device
from nano_serial_capture_windows import NanoSerialCapture
import hid as fake_hid


def run_logger(project_root, logger, output_dir, run_id, generation, failure=None, with_serial=False, seed=None,
               serial_dtr=False, serial_rts=False):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "fake_hid")
    env["FAKE_HID_GENERATION"] = generation
    env.pop("FAKE_HID_FAIL_AFTER_ONE", None)
    env.pop("FAKE_HID_INTERRUPT_AFTER_ONE", None)
    if failure == "disconnect":
        env["FAKE_HID_FAIL_AFTER_ONE"] = "1"
    elif failure == "interrupt":
        env["FAKE_HID_INTERRUPT_AFTER_ONE"] = "1"
    command = [sys.executable, str(logger), "--project-root", str(project_root),
               "--run-id", run_id, "--duration", "0.10", "--interface-number", "2",
               "--output-dir", str(output_dir), "--topology", "fake-test-only",
               "--load-description", "deterministic fake HID report"]
    if with_serial:
        command += ["--serial-port", "COM7", "--root-seed", str(seed)]
        if serial_dtr:
            command.append("--serial-dtr")
        if serial_rts:
            command.append("--serial-rts")
    proc = subprocess.run(
        command, text=True, capture_output=True, env=env, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    expected = {"disconnect": 5, "interrupt": 130}.get(failure, 0)
    if proc.returncode != expected:
        raise AssertionError("%s returned %d expected %d: %s" %
                             (run_id, proc.returncode, expected, proc.stderr))
    meta_path = output_dir / "run_metadata.json"
    if not meta_path.exists():
        raise AssertionError("%s did not retain metadata" % run_id)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (output_dir / "raw_packets.jsonl").read_text(encoding="utf-8").splitlines()]
    raw_reports = [row for row in records if row.get("record_type") == "measurement_report"]
    if not raw_reports:
        raise AssertionError("%s did not preserve a valid raw report" % run_id)
    return proc, metadata, records



def test_fake_nano_serial(root, seed):
    class FakeSerial:
        latest = None

        def __init__(self):
            self.port = None
            self.baudrate = 0
            self.timeout = None
            self.write_timeout = None
            self.dtr = True
            self.rts = True
            self.is_open = False
            self.rx = bytearray()
            self.tx = []
            self.fail_on_waiting = False
            FakeSerial.latest = self

        def open(self):
            self.is_open = True

        @property
        def in_waiting(self):
            if self.fail_on_waiting:
                raise OSError("simulated COM device removal")
            return len(self.rx)

        def read(self, count):
            chunk = bytes(self.rx[:count])
            del self.rx[:count]
            return chunk

        def write(self, payload):
            self.tx.append(bytes(payload))
            return len(payload)

        def close(self):
            self.is_open = False

        def feed(self, text):
            self.rx.extend(text.encode("ascii"))

    class FakeSerialModule:
        Serial = FakeSerial

    port_info = SimpleNamespace(device="COM7", description="Arduino Nano fake",
                                hwid="USB VID:PID=2341:0043 SER=private",
                                vid=0x2341, pid=0x0043, manufacturer="Arduino",
                                product="Nano", serial_number="private")
    out = root / "fake-serial-capture"
    out.mkdir()
    capture = NanoSerialCapture(out, "COM7", "fake-%04d" % seed, seed,
                                port_info=port_info, serial_module=FakeSerialModule)
    fake = FakeSerial.latest
    if fake.dtr or fake.rts:
        raise AssertionError("default DTR/RTS state was not low")
    fake.feed("EV,10,4294967290,ready,auto-start enabled\n")
    capture.poll()
    fake.feed("EV,11,4294967295,seed_set,%d\n" % seed)
    fake.feed("PONG,12,1,win-fake-%04d\n" % seed)
    fake.feed("EV,14,2,phase_start,seed=%d,phase=idle_led_off\n" % seed)
    capture.poll()
    capture.close("complete")
    sent = b"".join(fake.tx).decode("ascii")
    if "SEED %d\n" % seed not in sent or "PING win-fake-%04d\n" % seed not in sent or "START\n" not in sent:
        raise AssertionError("fake Nano did not receive seed, ping, and start commands")
    rows = [json.loads(line) for line in (out / "nano_serial.jsonl").read_text(encoding="utf-8").splitlines()]
    if not any(row.get("record_type") == "sequence_gap" and row.get("missing_sequence_count") == 1 for row in rows):
        raise AssertionError("Nano sequence gap was not represented")
    phase = next(row for row in rows if row.get("decoded_line", "").startswith("EV,14,"))
    if phase.get("device_millis_wrap_count") != 1 or phase.get("device_millis_unwrapped") != (1 << 32) + 2:
        raise AssertionError("Nano millis rollover was not unwrapped")

    fallback_dir = root / "fake-serial-banner-timeout"
    fallback_dir.mkdir()
    fallback = NanoSerialCapture(fallback_dir, "COM7", "fallback-%04d" % seed, seed,
                                 port_info=port_info, serial_module=FakeSerialModule)
    fallback_fake = FakeSerial.latest
    if fallback_fake.dtr or fallback_fake.rts:
        raise AssertionError("default DTR/RTS low state was not requested before serial open")
    fallback.opened_monotonic_ns = time.monotonic_ns() - 2_000_000_000
    fallback.poll()
    fallback_fake.feed("EV,50,200,seed_set,%d\n" % seed)
    fallback.poll()
    fallback.close("complete")
    fallback_tx = b"".join(fallback_fake.tx).decode("ascii")
    if "SEED %d\n" % seed not in fallback_tx or "START\n" not in fallback_tx:
        raise AssertionError("banner-missed serial fallback did not set seed and start")

    asserted_dir = root / "fake-serial-dtr-rts-asserted"
    asserted_dir.mkdir()
    asserted = NanoSerialCapture(asserted_dir, "COM7", "asserted-%04d" % seed, seed,
                                 dtr=True, rts=True, port_info=port_info,
                                 serial_module=FakeSerialModule)
    asserted_fake = FakeSerial.latest
    if not asserted_fake.dtr or not asserted_fake.rts:
        raise AssertionError("requested CDC DTR/RTS high state was not applied")
    asserted_fake.feed("EV,70,300,ready,auto-start enabled\n")
    asserted.poll()
    asserted_fake.feed("EV,71,301,seed_set,%d\n" % seed)
    asserted.poll()
    asserted.close("complete")
    asserted_rows=[json.loads(line) for line in (asserted_dir/"nano_serial.jsonl").read_text(encoding="utf-8").splitlines()]
    open_attempt=next(row for row in asserted_rows if row.get("event")=="serial_open_attempt")
    if not open_attempt["dtr_requested_before_open"] or not open_attempt["rts_requested_before_open"]:
        raise AssertionError("asserted CDC line state was not recorded")

    reconnect_dir = root / "fake-serial-reset-reconnect"
    reconnect_dir.mkdir()
    reenumerated_port = SimpleNamespace(device="COM8", description="ESP32 USB Serial/JTAG (COM8)",
                                        hwid="USB VID:PID=303A:1001 SER=pr:iv-at e",
                                        vid=0x303A, pid=0x1001, manufacturer="Microsoft",
                                        product=None, serial_number="pr:iv-at e")

    class FakeListPortsModule:
        ports = [port_info]

        @classmethod
        def comports(cls):
            return list(cls.ports)

    reconnect = NanoSerialCapture(reconnect_dir, "COM7", "reconnect-%04d" % seed, seed,
                                  port_info=port_info, serial_module=FakeSerialModule,
                                  list_ports_module=FakeListPortsModule, reconnect_interval_s=0)
    first_connection = FakeSerial.latest
    first_connection.fail_on_waiting = True
    reconnect.poll()
    if reconnect.connected or first_connection.is_open:
        raise AssertionError("Nano removal did not close the lost serial handle")
    FakeListPortsModule.ports = [reenumerated_port]
    reconnect.poll()
    second_connection = FakeSerial.latest
    if not reconnect.connected or reconnect.port_name != "COM8" or reconnect.reconnect_count != 1:
        raise AssertionError("Nano COM re-enumeration did not reconnect to the same board")
    second_connection.feed("EV,1,10,ready,auto-start enabled\n")
    reconnect.poll()
    second_connection.feed("EV,2,11,seed_set,%d\n" % seed)
    reconnect.poll()
    second_connection.feed("EV,3,12,experiment_start,idle_led_off\n")
    second_connection.feed("EV,4,13,experiment_end,four phases complete\n")
    reconnect.poll()
    reconnect.close("complete")
    reconnect_rows = [json.loads(line) for line in
                      (reconnect_dir / "nano_serial.jsonl").read_text(encoding="utf-8").splitlines()]
    reconnect_events = [row.get("event") for row in reconnect_rows]
    if "serial_disconnect" not in reconnect_events or "serial_reconnected" not in reconnect_events:
        raise AssertionError("serial disconnect/reconnect was not retained in the raw ledger")
    reconnect_text = (reconnect_dir / "nano_serial.jsonl").read_text(encoding="utf-8")
    if "SER=private" in reconnect_text or "pr:iv-at e" in reconnect_text:
        raise AssertionError("private USB serial identity leaked into the re-enumeration ledger")
    reconnect_tx = [row.get("command_ascii") for row in reconnect_rows if row.get("record_type") == "tx"]
    if "SEED %d" % seed not in reconnect_tx or "START" not in reconnect_tx:
        raise AssertionError("reconnected Nano did not receive the seed and start handshake")
    reconnect_close = next(row for row in reconnect_rows if row.get("event") == "serial_close")
    if not reconnect_close["experiment_started"] or not reconnect_close["experiment_completed"]:
        raise AssertionError("Nano experiment completeness was not recorded in the serial close event")

    meter_port = SimpleNamespace(device="COM3", description="USB Serial Device (COM3)",
                                 hwid="USB VID:PID=2E3C:5558", vid=0x2E3C, pid=0x5558,
                                 manufacturer="FNIRSI", product="FNB-C2", serial_number=None)
    try:
        NanoSerialCapture(out, "COM3", "should-reject-meter-com", seed,
                          port_info=meter_port, serial_module=FakeSerialModule)
        raise AssertionError("meter's own COM interface was accepted as Nano")
    except ValueError as exc:
        if "refusing meter interface" not in str(exc):
            raise


def run_ten_roots(project_root):
    project_root = Path(project_root).resolve()
    logger = Path(__file__).resolve().parent / "log_fnirsi_windows.py"
    base = project_root / "reports" / ("windows-integration-ten-roots-" + str(time.time_ns()))
    base.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(1, 11):
        seed = 5100 + index
        root = base / ("root seed %04d - space λ" % seed)
        root.mkdir()
        fake_devices = fake_hid.enumerate(0x2E3C, 0x5558)
        try:
            choose_device(fake_devices, None)
            raise AssertionError("missing explicit interface selection was accepted")
        except ValueError:
            pass
        try:
            choose_device(fake_devices, 9)
            raise AssertionError("nonexistent interface was accepted")
        except ValueError:
            pass
        selected = choose_device(fake_devices, 2)
        if selected["interface_number"] != 2:
            raise AssertionError("selected wrong interface")
        first = root / "output with spaces λ" / "cycle-a"
        second = root / "output with spaces λ" / "cycle-b"
        first.parent.mkdir()
        _, first_meta, _ = run_logger(project_root, logger, first, "fake-%04d-a" % seed,
                                      "a", with_serial=True, seed=seed,
                                      serial_dtr=True, serial_rts=True)
        _, second_meta, _ = run_logger(project_root, logger, second, "fake-%04d-b" % seed, "b")
        if first_meta.get("nano_serial_log_file") != "nano_serial.jsonl":
            raise AssertionError("actual logger did not capture Nano serial in the same run")
        serial_rows = [json.loads(line) for line in (first / "nano_serial.jsonl").read_text(encoding="utf-8").splitlines()]
        tx_commands = [row.get("command_ascii") for row in serial_rows if row.get("record_type") == "tx"]
        if not all(command in tx_commands for command in ("SEED %d" % seed, "PING win-fake-%04d-a" % seed, "START")):
            raise AssertionError("combined logger serial handshake failed: %r" % tx_commands)
        if first_meta["meter_hid_descriptor"]["interface_number"] != 2:
            raise AssertionError("logger did not select interface 2")
        if first_meta["meter_hid_descriptor"]["enumerated_path_sha256"] == second_meta["meter_hid_descriptor"]["enumerated_path_sha256"]:
            raise AssertionError("fresh enumeration did not observe changed fake path")
        failed_dir = root / "disconnect-recovery"
        _, failed_meta, _ = run_logger(project_root, logger, failed_dir, "fake-%04d-fail" % seed, "c", "disconnect")
        if failed_meta["status"] != "failed":
            raise AssertionError("disconnect was falsely complete")
        reopen_dir = root / "reopen-after-disconnect"
        _, reopen_meta, _ = run_logger(project_root, logger, reopen_dir, "fake-%04d-reopen" % seed, "d")
        if reopen_meta["status"] != "complete":
            raise AssertionError("reader did not reopen after disconnect")
        interrupted_dir = root / "interrupted-session"
        _, interrupted_meta, _ = run_logger(project_root, logger, interrupted_dir,
                                            "fake-%04d-interrupt" % seed, "e", "interrupt")
        if interrupted_meta["status"] != "interrupted":
            raise AssertionError("interrupt was falsely complete")
        test_fake_nano_serial(root, seed)
        rows.append({"root_seed": seed, "status": "pass",
                     "checks": ["explicit_interface", "unicode_space_paths", "fresh_reenumeration",
                                "disconnect_raw_recovery", "reader_reopen", "interrupt_partial_not_complete",
                                "nano_serial_seed_ping_start", "nano_seed_fallback_when_banner_missed",
                                "nano_serial_disconnect_reconnect_and_reseed", "nano_experiment_completion_state",
                                "nano_sequence_gap_and_millis_rollover",
                                "meter_com3_rejected_as_nano"],
                     "evidence_root": str(root)})
    report = {
        "schema_version": "windows-fake-hid-integration-1.0",
        "roots_expected": 10,
        "roots_passed": len(rows),
        "transport": "synthetic fake HID and serial modules; no physical device communication",
        "physical_hardware_claim": False,
        "rows": rows,
    }
    report_path = base / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "roots_passed": len(rows),
                      "transport": "fake HID and serial; software only"}, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    run_ten_roots(args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



