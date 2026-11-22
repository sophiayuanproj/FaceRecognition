"""Execute or preview one of the finite, resumable Nicla physical schedules."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from .fs import atomic_write_json

WORKSPACE = Path(__file__).resolve().parents[2]
STUDY = WORKSPACE / "work/project-finish-v4"
APP = WORKSPACE / "FaceRecognition"
CAPTURE = APP / "nicla_study/instruments/windows_fnirsi/capture_windows.py"
PHASE_DEFAULTS = {
    "JDEV": (STUDY / "config/J_CONTROLLER_PLAN.json", "cells", "development"),
    "JPILOT": (STUDY / "config/J_CONTROLLER_PLAN.json", "pilot.cells", "development"),
    "KFINAL": (STUDY / "config/K_FINAL_CAPTURE_PLAN.json", "cells", "final"),
}
CONDITION_MAP = {
    "known_frontal_stationary_normal": ("known", "normal", "stationary", 50.0),
    "known_offcenter_range": ("known", "normal", "stationary", 150.0),
    "known_motion_blur": ("known", "normal", "operator_moves_camera", 50.0),
    "known_low_light": ("known", "low_not_lux_metered", "stationary", 50.0),
    "unknown_entry_stationary": ("unknown", "normal", "stationary", 50.0),
    "empty_periodic_check": ("empty", "normal", "stationary", 50.0),
    "ambiguous_multiple_faces": ("ambiguous", "normal", "stationary", 50.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_rows(plan: dict, phase: str) -> list[dict]:
    if phase == "JPILOT":
        return plan.get("pilot", {}).get("cells", [])
    return plan.get("cells", [])


def _model_id(config: dict, policy_id: str) -> str:
    small = config["identity"]["small"]["model_id"]
    large = config["identity"]["large"]["model_id"]
    if policy_id in {"H", "L", "G", "A-sense"}:
        return large
    if policy_id in {"S", "T", "U"}:
        return small
    return small + "+" + large


def _receipt_gate(path: Path | None, name: str) -> dict:
    if path is None or not path.is_file():
        raise ValueError("%s qualification receipt is required before physical capture" % name)
    data = _read(path)
    if data.get("status") != "complete_verified":
        raise ValueError("%s receipt is not complete_verified" % name)
    return data


def _hash_present(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_hash_present(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_hash_present(item, expected) for item in value)
    return isinstance(value, str) and value.lower() == expected.lower()


def _validate_final_receipt(path: Path, config_sha: str, firmware_sha: str,
                            runtime_config_path: Path):
    receipt = _read(path)
    if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False:
        raise ValueError("K requires a new frozen receipt with final_role_opened=false")
    for field in ("model_sha256", "config_sha256", "calibration_sha256", "protocol_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))):
            raise ValueError("frozen receipt is missing valid " + field)
    if receipt["config_sha256"].lower() != config_sha.lower():
        raise ValueError("frozen configuration hash differs from selected deployment")
    if receipt["firmware_sha256"].lower() != firmware_sha.lower():
        raise ValueError("frozen firmware hash differs from selected deployment")
    from .freeze_selection import verify_receipt_sources
    verified_sources = verify_receipt_sources(path)
    source_files = verified_sources["resolved_source_files"]
    expected_sources = {
        "runtime_config": runtime_config_path.resolve(),
        "controller_plan": (STUDY / "config/J_CONTROLLER_PLAN.json").resolve(),
        "protocol": (STUDY / "config/J_PILOT_SELECTION_PROTOCOL.json").resolve(),
        "roster": (STUDY / "data/roster.json").resolve(),
    }
    for field, expected_path in expected_sources.items():
        observed = source_files.get(field)
        if not observed or Path(observed).resolve() != expected_path:
            raise ValueError("frozen receipt does not reference the current %s source" % field)
    return receipt


def _build_command(args, cell, run_id: str, out_dir: Path, rerun_of: str | None,
                   runtime_config: dict, phase: str, freeze_path: Path | None) -> list[str]:
    condition = str(cell["condition"])
    if condition not in CONDITION_MAP:
        raise ValueError("condition is not present in the frozen stimulus map: " + condition)
    player_condition, lighting, motion, range_cm = CONDITION_MAP[condition]
    event_log = out_dir / "stimulus-events.jsonl"
    stimulus_seed = int(cell["physical_block_seed"])
    command = [sys.executable, str(CAPTURE),
        "--project-root", str(WORKSPACE), "--duration", str(args.duration),
        "--warmup-seconds", str(args.warmup_seconds), "--run-id", run_id,
        "--experiment-id", "JIK-%s" % phase,
        "--scenario-id", condition,
        "--interface-number", str(args.meter_interface),
        "--board-type", "nicla_vision", "--board-label", args.board_label,
        "--device-reset-id", args.device_reset_id,
        "--firmware-source", str(args.firmware_source),
        "--firmware-binary", str(args.firmware_binary),
        "--board-configuration", str(args.board_configuration),
        "--firmware-version", args.firmware_version,
        "--serial-port", args.serial_port, "--serial-kind", "nicla",
        "--policy-id", cell["policy_id"], "--model-id", _model_id(runtime_config, cell["policy_id"]),
        "--physical-block-seed", str(cell["physical_block_seed"]),
        "--training-root-seed", str(cell["training_root_seed"]),
        "--session-id", cell["session_id"], "--condition", condition,
        "--input-manifest-sha256", args.input_manifest_sha256,
        "--expected-device-firmware-sha256", args.firmware_sha256,
        "--expected-device-config-sha256", args.config_sha256,
        "--meter-label", args.meter_label, "--meter-firmware", args.meter_firmware,
        "--topology", args.topology, "--load-description", args.load_description,
        "--stimulus-block", "final" if phase == "KFINAL" else "development",
        "--stimulus-condition", player_condition, "--stimulus-count", str(args.stimulus_count),
        "--stimulus-seed", str(stimulus_seed), "--stimulus-session-id", cell["session_id"],
        "--stimulus-event-log", str(event_log), "--stimulus-lighting-state", lighting,
        "--stimulus-motion-state", motion, "--stimulus-range-cm", str(range_cm),
        "--stimulus-hold-seconds", str(args.stimulus_hold_seconds),
        "--stimulus-blank-seconds", str(args.stimulus_blank_seconds),
        "--output-dir", str(out_dir)]
    if args.meter_path_fragment:
        command.extend(["--path-fragment", args.meter_path_fragment])
    if args.nicla_dtr:
        command.append("--nicla-dtr")
    if args.nicla_rts:
        command.append("--nicla-rts")
    if rerun_of:
        command.extend(["--rerun-of", rerun_of])
    if freeze_path:
        command.extend(["--stimulus-freeze-receipt", str(freeze_path)])
    return command


def _validate_capture(out_dir: Path, process_return: int, g_receipt: dict,
                      h_receipt: dict, plan_sha: str, cell: dict, run_id: str,
                      rerun_of: str | None, command: list[str]) -> dict:
    metadata_path = out_dir / "run_metadata.json"
    serial_events = out_dir / "nicla_serial_events.jsonl"
    raw_power = out_dir / "raw_packets.jsonl"
    raw_nicla = out_dir / "nicla_serial_raw.bin"
    stimulus_path = out_dir / "stimulus-events.jsonl"
    metadata = _read(metadata_path) if metadata_path.is_file() else {}
    validations = []
    malformed_serial_rows = 0
    if serial_events.is_file():
        for line in serial_events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                validations.append(json.loads(line))
            except ValueError:
                malformed_serial_rows += 1
    accepted = sum(1 for row in validations if row.get("validation", {}).get("status") == "accepted")
    rejected = sum(1 for row in validations if row.get("validation", {}).get("status") == "rejected")
    gaps = [row.get("validation", {}).get("sequence_gap", 0) for row in validations
            if row.get("validation", {}).get("sequence_gap")]
    stimulus = metadata.get("stimulus_player", {})
    startup = metadata.get("nicla_startup_receipt", {})
    checks = {
        "capture_process_exit_zero": process_return == 0,
        "run_metadata_complete": metadata.get("status") == "complete",
        "nicla_start_ack_present": startup.get("start_ack", {}).get("command") == "start",
        "nicla_expected_firmware_hash_matches": startup.get("observed_firmware_sha256") == metadata.get("expected_device_firmware_sha256"),
        "nicla_expected_config_hash_matches": startup.get("observed_config_sha256") == metadata.get("expected_device_config_sha256"),
        "nicla_capture_status_complete": metadata.get("nicla_serial_outcome") == "complete",
        "nicla_raw_serial_present": raw_nicla.is_file(),
        "nicla_serial_events_present": serial_events.is_file(),
        "nicla_valid_records_present": accepted > 0,
        "nicla_telemetry_rejections_absent": rejected == 0 and malformed_serial_rows == 0,
        "power_raw_packets_present": raw_power.is_file() and int(metadata.get("measurement_report_count", 0)) > 0,
        "power_invalid_report_count_zero": int(metadata.get("invalid_packet_count", 0)) == 0,
        "stimulus_log_hash_bound": stimulus.get("event_log_sha256") == (sha256(stimulus_path) if stimulus_path.is_file() else None),
        "stimulus_player_completed": stimulus.get("status") == "complete",
        "qualified_G_receipt_present": g_receipt.get("status") == "complete_verified",
        "qualified_H_receipt_present": h_receipt.get("status") == "complete_verified",
    }
    status = "complete_verified" if all(checks.values()) else "failed_qc_or_incomplete"
    return {
        "schema": "JIK-physical-capture-result-1",
        "status": status,
        "evidence_tier": "physical_capture_pending_formal_aggregate_analysis",
        "phase": cell.get("phase"),
        "planned_cell_run_id": cell["run_id"],
        "actual_run_id": run_id,
        "rerun_of": rerun_of,
        "condition": cell["condition"],
        "policy_id": cell["policy_id"],
        "physical_block_seed": cell["physical_block_seed"],
        "training_root_seed": cell["training_root_seed"],
        "session_id": cell["session_id"],
        "randomized_block_order_index": cell.get("randomized_block_order_index"),
        "randomized_policy_order_index": cell.get("randomized_order_index"),
        "plan_sha256": plan_sha,
        "run_metadata_sha256": sha256(metadata_path) if metadata_path.is_file() else None,
        "raw_power_sha256": sha256(raw_power) if raw_power.is_file() else None,
        "raw_nicla_serial_sha256": sha256(raw_nicla) if raw_nicla.is_file() else None,
        "nicla_serial_events_sha256": sha256(serial_events) if serial_events.is_file() else None,
        "stimulus_event_log_sha256": sha256(stimulus_path) if stimulus_path.is_file() else None,
        "nicla_accepted_record_count": accepted,
        "nicla_rejected_record_count": rejected,
        "nicla_sequence_gap_count": len(gaps),
        "power_report_count": metadata.get("measurement_report_count"),
        "power_decoded_sample_count": metadata.get("decoded_sample_count"),
        "checks": checks,
        "physical_claim_scope": "raw capture only; inferential I/J/K claims require qualified timing/QC and block-level analysis",
        "command": command,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=tuple(PHASE_DEFAULTS), required=True)
    p.add_argument("--plan", type=Path, help="frozen plan; defaults to study config")
    p.add_argument("--cell-run-id", help="preview or run one planned cell; omit with --execute to run all remaining cells")
    p.add_argument("--execute", action="store_true", help="perform physical captures; omission means dry-run")
    p.add_argument("--resume", action="store_true", help="skip only cells with complete_verified QC receipts")
    p.add_argument("--serial-port")
    p.add_argument("--meter-interface", type=int)
    p.add_argument("--meter-path-fragment")
    p.add_argument("--board-label")
    p.add_argument("--device-reset-id")
    p.add_argument("--firmware-source", type=Path)
    p.add_argument("--firmware-binary", type=Path)
    p.add_argument("--board-configuration", type=Path)
    p.add_argument("--firmware-version")
    p.add_argument("--runtime-config", type=Path)
    p.add_argument("--firmware-sha256")
    p.add_argument("--config-sha256")
    p.add_argument("--input-manifest-sha256")
    p.add_argument("--g-qualification-receipt", type=Path)
    p.add_argument("--h-qualification-receipt", type=Path)
    p.add_argument("--freeze-receipt", type=Path)
    p.add_argument("--pilot-protocol", type=Path,
                   help="hash-bound J pilot selection protocol, required for JPILOT execution")
    p.add_argument("--pilot-attempt-log", type=Path,
                   help="append-only preregistered settings log, required for JPILOT execution")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup-seconds", type=float, default=5.0)
    p.add_argument("--stimulus-count", type=int, default=10)
    p.add_argument("--stimulus-hold-seconds", type=float, default=1.5)
    p.add_argument("--stimulus-blank-seconds", type=float, default=1.0)
    p.add_argument("--meter-label", default="FNIRSI FNB-C2; verify against live descriptor")
    p.add_argument("--meter-firmware", required=False)
    p.add_argument("--topology", required=False)
    p.add_argument("--load-description", required=False)
    p.add_argument("--nicla-dtr", action="store_true")
    p.add_argument("--nicla-rts", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    plan_path, selector, stimulus_block = PHASE_DEFAULTS[args.phase]
    plan_path = args.plan or plan_path
    if not plan_path.is_file():
        p.error("plan file does not exist: " + str(plan_path))
    plan = _read(plan_path)
    rows = _cell_rows(plan, args.phase)
    if not rows:
        p.error("frozen schedule has no cells for " + args.phase)
    if args.cell_run_id:
        rows = [row for row in rows if row.get("run_id") == args.cell_run_id]
        if not rows:
            p.error("planned cell not found: " + args.cell_run_id)
    plan_hash = sha256(plan_path)
    if not args.execute:
        preview = rows[0]
        try:
            config_path = args.runtime_config or args.board_configuration or (APP / "firmware/runtime_config.json")
            runtime = _read(config_path)
            preview_dir = STUDY / preview.get("result_path", "runs/preview/capture_result.json")
            preview_out = preview_dir.parent
            cmd = _build_command(args, preview, preview["run_id"], preview_out, None,
                                 runtime, args.phase, args.freeze_receipt)
        except Exception as error:
            cmd = ["<supply Windows fresh ports, receipts, board paths, and hashes to preview capture command>"]
            preview_error = str(error)
        else:
            preview_error = None
        print(json.dumps({"status": "dry_run_only", "phase": args.phase,
            "plan_sha256": plan_hash, "schedule_cells": len(_cell_rows(plan, args.phase)),
            "preview_cell": preview, "stimulus_block": stimulus_block,
            "command": cmd, "preview_error": preview_error,
            "physical_measurement_performed": False}, indent=2))
        return 0

    required_values = {
        "--serial-port": args.serial_port, "--meter-interface": args.meter_interface,
        "--board-label": args.board_label, "--device-reset-id": args.device_reset_id,
        "--firmware-source": args.firmware_source, "--firmware-binary": args.firmware_binary,
        "--board-configuration": args.board_configuration, "--firmware-version": args.firmware_version,
        "--firmware-sha256": args.firmware_sha256, "--config-sha256": args.config_sha256,
        "--input-manifest-sha256": args.input_manifest_sha256,
        "--meter-firmware": args.meter_firmware, "--topology": args.topology,
        "--load-description": args.load_description,
    }
    missing = [key for key, value in required_values.items() if value is None]
    if missing:
        p.error("physical execution requires: " + ", ".join(missing))
    for key, value in (("firmware", args.firmware_sha256), ("config", args.config_sha256),
                       ("input manifest", args.input_manifest_sha256)):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)):
            p.error("%s must be a 64-character SHA-256" % key)
    g_receipt = _receipt_gate(args.g_qualification_receipt, "G meter/instrument")
    h_receipt = _receipt_gate(args.h_qualification_receipt, "H Nicla")
    if not _hash_present(h_receipt, args.firmware_sha256) or not _hash_present(h_receipt, args.config_sha256):
        p.error("H receipt does not qualify the exact selected firmware/config hashes")
    freeze_path = args.freeze_receipt
    runtime_path = args.runtime_config or args.board_configuration
    if args.phase == "KFINAL":
        if freeze_path is None or not freeze_path.is_file():
            p.error("K final capture requires --freeze-receipt from the development-only selector")
        if runtime_path is None:
            p.error("K final capture requires --runtime-config for the frozen deployment")
        _validate_final_receipt(freeze_path, args.config_sha256, args.firmware_sha256, runtime_path)
    if runtime_path is None or not runtime_path.is_file():
        p.error("an exact local runtime config file is required")
    runtime = _read(runtime_path)

    if args.execute and args.phase == "JPILOT":
        if args.pilot_protocol is None or not args.pilot_protocol.is_file():
            p.error("JPILOT execution requires --pilot-protocol generated before the pilot")
        if args.pilot_attempt_log is None or not args.pilot_attempt_log.is_file():
            p.error("JPILOT execution requires all settings preregistered in --pilot-attempt-log")
        try:
            from .j_pilot_workflow import validate_physical_pilot_inputs
            preflight = validate_physical_pilot_inputs(args.pilot_protocol, args.pilot_attempt_log,
                runtime_path.resolve(), plan_path.resolve())
        except (ValueError, FileNotFoundError, PermissionError) as error:
            p.error("JPILOT preflight failed: " + str(error))
        print(json.dumps({"status": "pilot_source_preflight_pass", "preflight": preflight}, indent=2))

    for cell in rows:
        phase_root = STUDY / "runs" / args.phase
        run_id = cell["run_id"]
        original_dir = phase_root / run_id
        result_path = original_dir / "capture_result.json"
        if args.resume and result_path.is_file():
            result = _read(result_path)
            if result.get("status") == "complete_verified" and result.get("plan_sha256") == plan_hash:
                continue
        rerun_of = None
        out_dir = original_dir
        if out_dir.exists():
            rerun_of = run_id
            rerun_index = 2
            while (phase_root / (run_id + "-R%02d" % rerun_index)).exists():
                rerun_index += 1
            run_id = run_id + "-R%02d" % rerun_index
            out_dir = phase_root / run_id
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = _build_command(args, cell, run_id, out_dir, rerun_of, runtime,
                             args.phase, freeze_path)
        proc = subprocess.run(cmd, cwd=WORKSPACE, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
        result = _validate_capture(out_dir, proc.returncode, g_receipt, h_receipt,
                                   plan_hash, cell, run_id, rerun_of, cmd)
        result["phase"] = args.phase
        result["capture_stdout_tail"] = (proc.stdout or "")[-4000:]
        result["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result_file = out_dir / "capture_result.json"
        atomic_write_json(result_file, result)
        ledger = STUDY / "runs" / args.phase / "CAMPAIGN_EVENTS.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"cell_run_id": cell["run_id"], "actual_run_id": run_id,
                "rerun_of": rerun_of, "status": result["status"], "result_path": str(result_file),
                "result_sha256": sha256(result_file), "utc": result["created_utc"]}, sort_keys=True) + "\n")
            stream.flush()
        print(json.dumps({"cell_run_id": cell["run_id"], "actual_run_id": run_id,
                          "status": result["status"], "output_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
