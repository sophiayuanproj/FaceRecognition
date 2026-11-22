#!/usr/bin/env python3
"""Execute the bounded H host-control queue and save every received JSONL record."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class QueueRunner:
    def __init__(self, port, baudrate, output):
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("Install pyserial for the H control queue") from error
        # Keep DTR asserted for USB_VCP.isconnected(), and assert RTS so the
        # device-side VCP read gate is open on firmware that enables RTS flow.
        self.serial = serial.Serial(port, baudrate, timeout=0.15, write_timeout=1,
                                    dsrdtr=False, rtscts=False)
        self.serial.rts = True
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output.open("w", encoding="utf-8", newline="\n")
        self.records = []
        self._rx_buffer = bytearray()
        self._pending_records = []

    def close(self):
        self.stream.close()
        self.serial.close()

    def _read_record(self):
        if not self._pending_records:
            data = self.serial.read(256)
            if data:
                self._rx_buffer.extend(data)
            while True:
                newline = self._rx_buffer.find(b"\n")
                if newline < 0:
                    break
                line = bytes(self._rx_buffer[:newline]).strip()
                del self._rx_buffer[:newline + 1]
                if not line:
                    continue
                try:
                    record = json.loads(line.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(record, dict):
                    self._pending_records.append(record)
            if len(self._rx_buffer) > 8192:
                # Runtime records are capped below 1 KB; discard a malformed
                # or unterminated stream before it can grow without bound.
                self._rx_buffer.clear()
        if not self._pending_records:
            return None
        record = self._pending_records.pop(0)
        self.records.append(record)
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.stream.flush()
        print(json.dumps(record, sort_keys=True))
        return record

    def command(self, payload, timeout=10.0):
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        command = str(payload.get("cmd"))
        self.serial.write(encoded)
        self.serial.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self._read_record()
            if row and row.get("event") == "control_ack" and row.get("command") == command:
                if row.get("control_status") not in ("ok",):
                    raise RuntimeError("control rejected: " + json.dumps(row))
                return row
        raise RuntimeError("control ACK timeout for " + command)

    def wait_trial(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self._read_record()
            if row and row.get("event") == "trial_finished":
                return row
        raise RuntimeError("trial did not finish before host timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="COM port for Nicla USB CDC or external UART adapter")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--seed-id", default="H-20261030-01")
    parser.add_argument("--policy", default="G", choices=("H", "L", "S", "G", "A-sense", "A-model", "A-full", "T", "U"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--duration-ms", type=int, default=25000)
    parser.add_argument("--output", type=Path,
        default=ROOT.parent / "work/project-finish-v4/reports/H/latest_smoke.jsonl")
    args = parser.parse_args()
    if args.frames < 1 or args.frames > 50 or args.duration_ms < 1 or args.duration_ms > 30000:
        parser.error("H queue limits are 1..50 frames and 1..30000 ms")
    runner = QueueRunner(args.port, args.baudrate, args.output)
    try:
        # Let the standalone loader and camera settle before the first USB
        # command; a command sent during CDC enumeration can be lost.
        time.sleep(2.0)
        runner.command({"cmd": "identity"})
        runner.command({"cmd": "config"})
        runner.command({"cmd": "ping"})
        runner.command({"cmd": "seed", "seed_id": args.seed_id,
                        "host_marker": datetime.now(timezone.utc).isoformat(timespec="milliseconds")})
        runner.command({"cmd": "status"})
        trial_id = "h-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        runner.command({"cmd": "trial", "policy_id": args.policy,
                        "frames": args.frames, "duration_ms": args.duration_ms,
                        "trial_id": trial_id})
        finish = runner.wait_trial(args.duration_ms / 1000.0 + 5.0)
        runner.command({"cmd": "status"})
        # Startup may predate the host opening USB CDC; standalone cold boot
        # is verified separately. This bounded host queue requires trial
        # decisions and its completion marker after the control ACK.
        required = {"decision", "trial_finished"}
        seen = {row.get("event") for row in runner.records}
        summary = {"status": "host_queue_complete" if required.issubset(seen) else "host_queue_incomplete",
            "port": args.port, "baudrate": args.baudrate, "seed_id": args.seed_id,
            "policy_id": args.policy, "frames_requested": args.frames,
            "duration_ms": args.duration_ms, "records": len(runner.records),
            "trial_finished": finish, "required_events_seen": sorted(required & seen),
            "output": str(args.output)}
        runner.stream.write(json.dumps({"host_summary": summary}, sort_keys=True) + "\n")
        runner.stream.flush()
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "host_queue_complete" else 3
    finally:
        runner.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print("H queue stopped:", error, file=sys.stderr)
        raise SystemExit(2)
