#!/usr/bin/env python3
"""Windows/macOS USB-CDC or external TTL UART control client for Nicla firmware."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


class NiclaClient:
    def __init__(self, port, baudrate=115200, timeout=0.2):
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("Install pyserial on the host running this adapter") from error
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout,
                                    write_timeout=timeout, dsrdtr=False, rtscts=False)

    def close(self):
        self.serial.close()

    def send(self, command, wait_seconds=2.0):
        payload = (json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8")
        self.serial.write(payload)
        self.serial.flush()
        deadline = time.monotonic() + max(0.1, float(wait_seconds))
        records = []
        while time.monotonic() < deadline:
            line = self.serial.readline()
            if not line:
                continue
            try:
                row = json.loads(line.decode("utf-8", errors="replace").strip())
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                records.append(row)
                if row.get("event") == "control_ack" and row.get("command") == command.get("cmd"):
                    return records
        return records


class NiclaRecordValidator:
    """Validate raw JSONL records while keeping rejected evidence inspectable."""

    def __init__(self, expected_run_id=None, firmware_sha256=None, config_sha256=None):
        self.expected_run_id = expected_run_id
        self.firmware_sha256 = firmware_sha256.lower() if isinstance(firmware_sha256, str) else firmware_sha256
        self.config_sha256 = config_sha256.lower() if isinstance(config_sha256, str) else config_sha256
        self.last_sequence_by_run = {}
        self.last_ticks_by_run = {}

    def validate(self, raw_line):
        raw = bytes(raw_line)
        result = {"raw_line_hex": raw.hex(), "raw_line_sha256": hashlib.sha256(raw).hexdigest(),
                  "status": "rejected", "errors": []}
        try:
            row = json.loads(raw.decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError) as error:
            result["errors"].append("invalid_jsonl:%s" % type(error).__name__)
            return result
        if not isinstance(row, dict):
            result["errors"].append("record_not_object")
            return result
        result["record"] = row
        if row.get("schema_id") not in ("nicla-face-telemetry-1", "nicla-observation-v2"):
            result["errors"].append("unsupported_schema_id")
        run_id = row.get("run_id")
        sequence = row.get("sequence")
        ticks = row.get("ticks_ms")
        sequence_acceptable = True
        if not isinstance(run_id, str) or not run_id:
            result["errors"].append("missing_run_id")
            sequence_acceptable = False
        if self.expected_run_id and run_id != self.expected_run_id:
            result["errors"].append("wrong_run_id")
        if type(sequence) is not int or sequence < 0:
            result["errors"].append("invalid_sequence")
            sequence_acceptable = False
        elif run_id:
            prior = self.last_sequence_by_run.get(run_id)
            if prior is not None:
                if sequence == prior:
                    result["errors"].append("duplicate_sequence")
                    sequence_acceptable = False
                elif sequence < prior:
                    result["errors"].append("stale_or_inverted_sequence")
                    sequence_acceptable = False
                elif sequence > prior + 1:
                    result["sequence_gap"] = sequence - prior - 1
            if sequence_acceptable:
                self.last_sequence_by_run[run_id] = sequence
        if type(ticks) is not int or ticks < 0:
            result["errors"].append("invalid_device_ticks")
        elif run_id and sequence_acceptable:
            modulus = row.get("ticks_modulus_ms")
            if type(modulus) is int and modulus > 1 and ticks >= modulus:
                result["errors"].append("device_ticks_outside_modulus")
            prior_ticks = self.last_ticks_by_run.get(run_id)
            if prior_ticks is not None and ticks < prior_ticks:
                if type(modulus) is int and modulus > 1 and prior_ticks - ticks > modulus // 2:
                    result["device_clock_event"] = "wrap"
                else:
                    result["device_clock_event"] = "reset_or_discontinuity"
            self.last_ticks_by_run[run_id] = ticks
        if self.firmware_sha256 and str(row.get("firmware_sha256", "")).lower() != self.firmware_sha256:
            result["errors"].append("firmware_hash_mismatch")
        if self.config_sha256 and str(row.get("model_config_sha256", "")).lower() != self.config_sha256:
            result["errors"].append("configuration_hash_mismatch")
        result["status"] = "accepted" if not result["errors"] else "rejected"
        return result


class NiclaRawSerialCapture:
    """Raw-first, bounded serial capture for actual Nicla USB CDC or UART data."""

    def __init__(self, port_name, run_dir, run_id, *, baudrate=115200,
                 dtr=False, rts=False, expected_firmware_sha256=None,
                 expected_config_sha256=None, monotonic_ns=time.monotonic_ns,
                 utc_now=None, serial_module=None):
        if serial_module is None:
            try:
                import serial as serial_module
            except ImportError as error:
                raise RuntimeError("pyserial is required by the Windows instrument environment") from error
        import serial.tools.list_ports as list_ports
        descriptors = [p for p in list_ports.comports()
                       if str(getattr(p, "device", "")).casefold() == str(port_name).casefold()]
        if len(descriptors) != 1:
            raise ValueError("Nicla serial port must match exactly one fresh descriptor")
        descriptor = descriptors[0]
        vid = getattr(descriptor, "vid", None)
        pid = getattr(descriptor, "pid", None)
        if (vid, pid) == (0x2E3C, 0x5558):
            raise ValueError("meter COM interface cannot be used as Nicla serial telemetry")
        self.descriptor = {"device": str(getattr(descriptor, "device", "")),
                           "vid": vid, "pid": pid,
                           "product": getattr(descriptor, "product", None),
                           "manufacturer": getattr(descriptor, "manufacturer", None)}
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.monotonic_ns = monotonic_ns
        self.utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat(timespec="microseconds"))
        self.validator = NiclaRecordValidator(None, expected_firmware_sha256,
                                               expected_config_sha256)
        self.raw_path = self.run_dir / "nicla_serial_raw.bin"
        self.chunk_path = self.run_dir / "nicla_serial_chunks.jsonl"
        self.event_path = self.run_dir / "nicla_serial_events.jsonl"
        self.command_path = self.run_dir / "nicla_serial_commands.jsonl"
        self.raw_stream = self.raw_path.open("xb")
        self.chunk_stream = self.chunk_path.open("x", encoding="utf-8", buffering=1)
        self.event_stream = self.event_path.open("x", encoding="utf-8", buffering=1)
        self.command_stream = self.command_path.open("x", encoding="utf-8", buffering=1)
        self.buffer = bytearray()
        self.offset = 0
        self.ack_records = []
        self.valid_records = []
        self.read_errors = []
        self.closed = False
        self.serial = serial_module.Serial(port=None, baudrate=baudrate, timeout=0,
                                           write_timeout=1, dsrdtr=False, rtscts=False)
        try:
            self.serial.dtr = bool(dtr)
            self.serial.rts = bool(rts)
            self.serial.port = port_name
            self.serial.open()
        except Exception:
            for stream in (self.raw_stream, self.chunk_stream, self.event_stream, self.command_stream):
                stream.close()
            raise
        self.metadata = {"port_descriptor": self.descriptor, "baudrate": int(baudrate),
                         "dtr_requested": bool(dtr), "rts_requested": bool(rts),
                         "dtr_may_reset_board": bool(dtr), "raw_log_file": self.raw_path.name,
                         "event_log_file": self.event_path.name,
                         "chunk_log_file": self.chunk_path.name,
                         "command_log_file": self.command_path.name}

    def _append_event(self, row):
        self.event_stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        self.event_stream.flush()

    def poll(self, maximum_bytes=4096):
        try:
            waiting = int(getattr(self.serial, "in_waiting", 0))
            if waiting <= 0:
                return 0
            data = bytes(self.serial.read(min(maximum_bytes, waiting)))
        except Exception as error:
            failure = {"utc": self.utc_now(), "host_monotonic_ns": self.monotonic_ns(),
                       "event": "serial_read_error", "error_type": type(error).__name__,
                       "message": str(error)[:160]}
            self.read_errors.append(failure)
            self._append_event(failure)
            return 0
        if not data:
            return 0
        receive_ns = self.monotonic_ns()
        receive_utc = self.utc_now()
        chunk_hash = hashlib.sha256(data).hexdigest()
        self.raw_stream.write(data)
        self.raw_stream.flush()
        self.chunk_stream.write(json.dumps({"host_received_utc": receive_utc,
            "host_received_monotonic_ns": receive_ns, "offset": self.offset,
            "bytes": len(data), "sha256": chunk_hash, "raw_hex": data.hex()},
            sort_keys=True) + "\n")
        self.chunk_stream.flush()
        self.offset += len(data)
        self.buffer.extend(data)
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(self.buffer[:newline + 1])
            del self.buffer[:newline + 1]
            validation = self.validator.validate(raw_line)
            row = {"host_received_utc": receive_utc,
                   "host_received_monotonic_ns": receive_ns,
                   "validation": validation}
            self._append_event(row)
            record = validation.get("record")
            if isinstance(record, dict) and record.get("event") == "control_ack":
                self.ack_records.append(record)
            if isinstance(record, dict) and validation.get("status") == "accepted":
                self.valid_records.append(record)
        return len(data)

    def send(self, command):
        payload = (json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8")
        started_ns = self.monotonic_ns()
        written = self.serial.write(payload)
        self.serial.flush()
        self.command_stream.write(json.dumps({"host_utc": self.utc_now(),
            "host_monotonic_ns": started_ns, "command": command,
            "raw_hex": payload.hex(), "bytes_requested": len(payload),
            "bytes_written": int(written)}, sort_keys=True) + "\n")
        self.command_stream.flush()
        if int(written) != len(payload):
            raise IOError("partial Nicla command write")

    def command(self, command, wait_seconds=2.0):
        self.send(command)
        deadline = time.monotonic() + max(0.1, float(wait_seconds))
        while time.monotonic() < deadline:
            self.poll()
            for index, row in enumerate(self.ack_records):
                if row.get("command") == command.get("cmd"):
                    return self.ack_records.pop(index)
            time.sleep(0.01)
        raise TimeoutError("no Nicla control_ack before bounded timeout for %s" % command.get("cmd"))

    def close(self, outcome="complete"):
        if self.closed:
            return
        self.poll()
        if self.buffer:
            raw_fragment = bytes(self.buffer)
            self._append_event({"utc": self.utc_now(), "host_monotonic_ns": self.monotonic_ns(),
                "event": "unterminated_serial_fragment", "raw_fragment_hex": raw_fragment.hex(),
                "raw_fragment_sha256": hashlib.sha256(raw_fragment).hexdigest(),
                "byte_count": len(raw_fragment)})
            self.buffer.clear()
        self._append_event({"utc": self.utc_now(), "host_monotonic_ns": self.monotonic_ns(),
            "event": "nicla_serial_capture_end", "outcome": outcome,
            "raw_bytes": self.offset, "read_error_count": len(self.read_errors)})
        self.serial.close()
        for stream in (self.raw_stream, self.chunk_stream, self.event_stream, self.command_stream):
            stream.close()
        self.closed = True


def build_command(args):
    if args.command in ("identity", "discover", "config", "ping", "stop", "status"):
        return {"cmd": args.command}
    if args.command == "seed":
        return {"cmd": "seed", "seed_id": args.seed_id, "host_marker": args.host_marker}
    if args.command == "start":
        return {"cmd": "start", "policy_id": args.policy}
    if args.command == "trial":
        return {"cmd": "trial", "policy_id": args.policy,
                "frames": args.frames, "duration_ms": args.duration_ms}
    raise ValueError("unsupported host command")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Windows COM port or serial device path")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--wait", type=float, default=2.0)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("identity", "discover", "config", "ping", "stop", "status"):
        sub.add_parser(name)
    p_seed = sub.add_parser("seed")
    p_seed.add_argument("seed_id")
    p_seed.add_argument("--host-marker", default="")
    p_start = sub.add_parser("start")
    p_start.add_argument("--policy", default="G")
    p_trial = sub.add_parser("trial")
    p_trial.add_argument("--policy", default="G")
    p_trial.add_argument("--frames", type=int, required=True)
    p_trial.add_argument("--duration-ms", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "trial" and (args.frames < 1 or args.frames > 50 or
                                     args.duration_ms < 1 or args.duration_ms > 30000):
        parser.error("trial is bounded to 1..50 frames and 1..30000 ms")
    client = NiclaClient(args.port, args.baudrate)
    try:
        rows = client.send(build_command(args), args.wait)
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        if not rows:
            print("No JSON response before timeout; check selected USB CDC/UART port and wiring.", file=sys.stderr)
            return 2
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
