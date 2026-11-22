#!/usr/bin/env python3
"""Run the changed-policy board smoke check over a fresh Nicla serial port.

This is a software/action smoke check, not a recognition accuracy, standalone,
or power-qualification run. It writes raw USB-CDC bytes and all validated JSONL
records to a new output directory before summarizing observed actions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
from nicla_protocol import NiclaRawSerialCapture  # noqa: E402

DEFAULT_POLICIES = ("H", "L", "S", "G", "A-sense", "A-model", "A-risk", "A-cost", "A-full", "T", "U")
DEFAULT_STAGES = (
    ("idle", "U", "small"),
    ("camera_capture_only", "U", "small"),
    ("detector_only", "U", "small"),
    ("identity_small_only", "S", "small"),
    ("identity_large_only", "H", "large"),
    ("cold_model_load", "S", "small"),
    ("cold_model_load", "H", "large"),
    ("warm_inference", "S", "small"),
    ("warm_inference", "H", "large"),
    ("model_swap_and_gc", "A-model", "small"),
    ("full_pipeline", "H", "large"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value))


def summarize_policy(policy_id: str, records: list[dict], elapsed_s: float) -> dict:
    events = Counter(str(row.get("event", "")) for row in records)
    statuses = Counter(str(row.get("status", "")) for row in records if row.get("event") == "decision")
    captures = [row for row in records if row.get("event") == "capture_start"]
    model_loads = [row.get("model_id") for row in records if row.get("event") == "model_load"]
    heap = [row.get("python_heap_free_before_bytes") for row in records
            if isinstance(row.get("python_heap_free_before_bytes"), (int, float))]
    tick_deltas = []
    previous = None
    for row in captures:
        tick = row.get("ticks_ms")
        modulus = row.get("ticks_modulus_ms")
        if not isinstance(tick, int):
            continue
        if previous is not None:
            prior_tick, prior_modulus = previous
            delta = tick - prior_tick
            if delta < 0 and isinstance(modulus, int) and modulus > 1 and prior_tick - tick > modulus // 2:
                delta += modulus
            if delta >= 0:
                tick_deltas.append(delta)
        previous = tick, modulus
    return {
        "policy_id": policy_id,
        "duration_s": round(elapsed_s, 3),
        "telemetry_event_counts": dict(sorted(events.items())),
        "decision_status_counts": dict(sorted(statuses.items())),
        "capture_start_count": len(captures),
        "median_capture_interval_ms": sorted(tick_deltas)[len(tick_deltas) // 2] if tick_deltas else None,
        "observed_capture_rate_hz": round(len(captures) / elapsed_s, 3) if elapsed_s > 0 else None,
        "model_load_ids": model_loads,
        "minimum_reported_python_heap_free_bytes": min(heap) if heap else None,
        "camera_sleep_events": events.get("camera_sleep", 0),
        "camera_wake_events": events.get("camera_wake", 0),
        "camera_wake_failures": events.get("camera_wake_failed", 0),
        "runtime_fault_events": sum(events.get(name, 0) for name in
                                     ("runtime_error", "detect_failed", "camera_sleep_failed", "camera_wake_failed")),
        "action_transition_observed": events.get("policy_changed", 0) > 0 and len(captures) > 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="freshly discovered USB-CDC serial device/COM port")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; must not exist")
    parser.add_argument("--expected-firmware-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True,
                        help="config hash from the selected deployment receipt, not a guessed source hash")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES), choices=DEFAULT_POLICIES)
    parser.add_argument("--seconds-per-policy", type=float, default=8.0)
    parser.add_argument("--seconds-per-stage", type=float, default=2.0)
    parser.add_argument("--only-stage", choices=sorted({row[0] for row in DEFAULT_STAGES} | {
        "identity_center_crop_diagnostic"}),
        help="run one measurement stage and skip the policy sweep")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--dtr", action="store_true", help="assert DTR only for the documented board mode; may reset")
    parser.add_argument("--rts", action="store_true", help="assert RTS only for the documented board mode")
    parser.add_argument("--run-id", default="nicla-smoke-" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    args = parser.parse_args(argv)
    for name, value in (("firmware", args.expected_firmware_sha256), ("config", args.expected_config_sha256)):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            parser.error("expected %s SHA-256 must be 64 hexadecimal characters" % name)
    if (not safe_id(args.run_id) or args.seconds_per_policy <= 0 or args.seconds_per_policy > 120 or
            args.seconds_per_stage <= 0 or args.seconds_per_stage > 15):
        parser.error("run ID or bounded policy/stage dwell is invalid")
    if len(set(args.policies)) != len(args.policies):
        parser.error("policy list must not contain duplicates")
    requested_stages = ([(args.only_stage, "S", "small")] if args.only_stage else list(DEFAULT_STAGES))

    try:
        import serial
        from serial.tools import list_ports
    except ImportError as error:
        print("Install pyserial in the isolated instrument environment: %s" % error, file=sys.stderr)
        return 2
    matches = [port for port in list_ports.comports()
               if str(getattr(port, "device", "")).casefold() == args.port.casefold()]
    if len(matches) != 1:
        print("Port must match exactly one current serial descriptor", file=sys.stderr)
        return 3
    descriptor = matches[0]
    if (getattr(descriptor, "vid", None), getattr(descriptor, "pid", None)) == (0x2E3C, 0x5558):
        print("The FNIRSI meter COM interface cannot serve as Nicla telemetry", file=sys.stderr)
        return 3

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    capture = None
    outcome = "failed"
    error_text = None
    identity = config = ping = None
    per_policy = []
    per_stage = []
    try:
        capture = NiclaRawSerialCapture(args.port, output, args.run_id, baudrate=args.baudrate,
            dtr=args.dtr, rts=args.rts,
            expected_firmware_sha256=args.expected_firmware_sha256,
            expected_config_sha256=args.expected_config_sha256, serial_module=serial)
        identity = capture.command({"cmd": "identity"}, wait_seconds=5)
        config = capture.command({"cmd": "config"}, wait_seconds=5)
        ping = capture.command({"cmd": "ping"}, wait_seconds=5)
        if identity.get("device") != "Arduino Nicla Vision / OpenMV CSI":
            raise ValueError("selected port did not identify as Nicla Vision")
        if identity.get("firmware_sha256", "").lower() != args.expected_firmware_sha256.lower():
            raise ValueError("live firmware hash differs from deployment receipt")
        if identity.get("config_sha256", "").lower() != args.expected_config_sha256.lower():
            raise ValueError("live runtime config hash differs from deployment receipt")
        if ping.get("pong") is not True:
            raise ValueError("Nicla ping failed")
        advertised = set(config.get("policy_ids", []))
        missing = [policy for policy in args.policies if policy not in advertised]
        if missing:
            raise ValueError("live firmware does not advertise requested policies: " + ", ".join(missing))
        advertised_stages = set(config.get("measurement_stage_ids", []))
        missing_stages = sorted({row[0] for row in requested_stages} - advertised_stages)
        if missing_stages:
            raise ValueError("live firmware does not advertise measurement stages: " + ", ".join(missing_stages))

        seed_id = "SMOKE-" + uuid.uuid4().hex[:16]
        seed_ack = capture.command({"cmd": "seed", "seed_id": seed_id, "host_marker": args.run_id}, wait_seconds=5)
        if seed_ack.get("control_status", "ok") != "ok":
            raise ValueError("Nicla rejected smoke seed")
        for index, (stage_id, policy_id, model_slot) in enumerate(requested_stages):
            start_index = len(capture.valid_records)
            start_ack = capture.command({"cmd": "start", "policy_id": policy_id,
                "measurement_stage_id": stage_id, "model_slot": model_slot}, wait_seconds=5)
            if (start_ack.get("control_status", "ok") != "ok" or
                    start_ack.get("measurement_stage_id") != stage_id or
                    start_ack.get("policy_id") != policy_id):
                raise ValueError("Nicla did not accept measurement stage %s" % stage_id)
            dwell_start = time.monotonic()
            deadline = dwell_start + args.seconds_per_stage
            while time.monotonic() < deadline:
                capture.poll()
                time.sleep(0.01)
            elapsed = time.monotonic() - dwell_start
            stop_ack = capture.command({"cmd": "stop"}, wait_seconds=5)
            if stop_ack.get("control_status", "ok") != "ok":
                raise ValueError("Nicla did not stop measurement stage %s" % stage_id)
            records = capture.valid_records[start_index:]
            events = Counter(str(row.get("event", "")) for row in records)
            stage_events = sum(1 for row in records if row.get("stage_id") == stage_id and
                               row.get("event") == "measurement_stage_result")
            idle_events = events.get("measurement_stage_idle", 0) if stage_id == "idle" else 0
            per_stage.append({"stage_id": stage_id, "policy_id": policy_id,
                "model_slot": model_slot, "duration_s": round(elapsed, 3),
                "start_ack": start_ack, "stop_ack": stop_ack,
                "capture_start_count": events.get("capture_start", 0),
                "decision_status_counts": dict(Counter(str(row.get("status", "")) for row in records
                    if row.get("event") == "decision")),
                "measurement_stage_result_count": stage_events,
                "idle_sample_count": idle_events,
                "camera_sleep_events": events.get("camera_sleep", 0),
                "camera_wake_events": events.get("camera_wake", 0),
                "model_load_events": events.get("model_load", 0),
                "model_swap_events": events.get("model_swap", 0),
                "smoke_block_seed": index})
        for index, policy_id in enumerate(() if args.only_stage else args.policies):
            start_index = len(capture.valid_records)
            start_ack = capture.command({"cmd": "start", "policy_id": policy_id}, wait_seconds=5)
            if start_ack.get("control_status", "ok") != "ok" or start_ack.get("policy_id") != policy_id:
                raise ValueError("Nicla did not start requested policy %s" % policy_id)
            dwell_start = time.monotonic()
            deadline = dwell_start + args.seconds_per_policy
            while time.monotonic() < deadline:
                capture.poll()
                time.sleep(0.01)
            elapsed = time.monotonic() - dwell_start
            stop_ack = capture.command({"cmd": "stop"}, wait_seconds=5)
            if stop_ack.get("control_status", "ok") != "ok":
                raise ValueError("Nicla did not stop policy %s" % policy_id)
            records = capture.valid_records[start_index:]
            summary = summarize_policy(policy_id, records, elapsed)
            summary["requested_policy_id"] = policy_id
            summary["start_ack"] = start_ack
            summary["stop_ack"] = stop_ack
            summary["smoke_block_seed"] = index
            per_policy.append(summary)
        outcome = "software_action_smoke_captured"
    except Exception as error:
        error_text = "%s: %s" % (type(error).__name__, str(error))
        if capture is not None:
            try:
                capture.command({"cmd": "stop"}, wait_seconds=3)
            except Exception:
                pass
    finally:
        if capture is not None:
            capture.close(outcome)

    validation_rows = []
    event_path = output / "nicla_serial_events.jsonl"
    if event_path.is_file():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            try:
                validation_rows.append(json.loads(line))
            except ValueError:
                validation_rows.append({"status": "malformed_event_log_line"})
    report = {
        "schema": "nicla-board-smoke-1",
        "run_id": args.run_id,
        "status": outcome,
        "evidence_tier": "software_action_smoke_only",
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "port_descriptor": {"device": descriptor.device, "vid": getattr(descriptor, "vid", None),
                             "pid": getattr(descriptor, "pid", None),
                             "product": getattr(descriptor, "product", None),
                             "manufacturer": getattr(descriptor, "manufacturer", None)},
        "expected_firmware_sha256": args.expected_firmware_sha256.lower(),
        "expected_config_sha256": args.expected_config_sha256.lower(),
        "observed_identity_ack": identity,
        "observed_config_ack": config,
        "observed_ping_ack": ping,
        "measurement_stages_requested": [row[0] for row in requested_stages],
        "measurement_stages": per_stage,
        "policies_requested": [] if args.only_stage else args.policies,
        "policies": per_policy,
        "serial_validation_event_count": len(validation_rows),
        "serial_rejected_line_count": sum(1 for row in validation_rows
                                           if row.get("validation", {}).get("status") == "rejected"),
        "error": error_text,
        "limits": [
            "No people or identity stimuli are supplied; a NO_FACE result does not validate recognition outputs.",
            "USB serial operation does not qualify battery-only cold start or standalone acceptance.",
            "No FNIRSI meter data, power estimate, or G/I energy result is produced.",
            "Stage dwell uses the live ambient camera scene; identity/model-load stages may produce no inference when no face is visible."
        ],
    }
    report_path = output / "board_smoke_result.json"
    temp_path = report_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(report_path)
    print(json.dumps({"status": outcome, "output_dir": str(output),
                      "report_sha256": sha256(report_path), "error": error_text}, indent=2))
    return 0 if outcome == "software_action_smoke_captured" else 4


if __name__ == "__main__":
    raise SystemExit(main())
