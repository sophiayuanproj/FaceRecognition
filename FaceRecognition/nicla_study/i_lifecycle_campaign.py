"""Run the finite, resumable selected-runtime Nicla lifecycle stage schedule.

The ten-root E model/compression matrix is preserved separately in
I_PHYSICAL_MATRIX.json. This runner measures lifecycle stages for one exact
currently deployed small/large pair; each execution is blocked on exact G/H
receipts and live board hash readback. It never accesses final-role inputs.
"""
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
PLAN = STUDY / "config/I_LIFECYCLE_PLAN.json"
CAPTURE = APP / "nicla_study/instruments/windows_fnirsi/capture_windows.py"
SOURCE_NAMES = ("wearable_app.py", "identity_contracts.py", "runtime_support.py", "telemetry.py")
SOURCE_PATHS = {
    "wearable_app.py": APP / "firmware/main.py",
    "identity_contracts.py": APP / "firmware/identity_contracts.py",
    "runtime_support.py": APP / "firmware/runtime_support.py",
    "telemetry.py": APP / "firmware/telemetry.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _firmware_hash() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_NAMES:
        digest.update((name + "\t" + sha256(SOURCE_PATHS[name]) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _hash_present(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_hash_present(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_hash_present(item, expected) for item in value)
    return isinstance(value, str) and value.lower() == expected.lower()


def _receipt(path: Path | None, label: str) -> dict:
    if path is None or not path.is_file():
        raise ValueError("%s qualification receipt is required" % label)
    result = _read(path)
    if result.get("status") != "complete_verified":
        raise ValueError("%s receipt is not complete_verified" % label)
    return result


def _stage_model_id(runtime: dict, cell: dict) -> str:
    stage = cell["stage_id"]
    if stage == "model_swap_and_gc":
        return runtime["identity"]["small"]["model_id"] + "+" + runtime["identity"]["large"]["model_id"]
    if stage in ("identity_small_only", "identity_large_only", "cold_model_load", "warm_inference"):
        return runtime["identity"][cell["model_slot"]]["model_id"]
    if cell["policy_id"] in ("H", "L", "G", "A-sense"):
        return runtime["identity"]["large"]["model_id"]
    return runtime["identity"]["small"]["model_id"]


def _build_command(args, runtime: dict, cell: dict, out_dir: Path,
                   firmware_sha: str, config_sha: str, input_sha: str) -> list[str]:
    stimulus_log = out_dir / "stimulus-events.jsonl"
    command = [sys.executable, str(CAPTURE),
        "--project-root", str(WORKSPACE), "--duration", str(args.duration),
        "--warmup-seconds", str(args.warmup_seconds), "--run-id", cell["run_id"],
        "--experiment-id", "I-LIFECYCLE", "--scenario-id", cell["stage_id"],
        "--interface-number", str(args.meter_interface),
        "--board-type", "nicla_vision", "--board-label", args.board_label,
        "--device-reset-id", args.device_reset_id,
        "--firmware-source", str(args.firmware_source),
        "--firmware-binary", str(args.firmware_binary),
        "--board-configuration", str(args.runtime_config),
        "--firmware-version", args.firmware_version,
        "--serial-port", args.serial_port, "--serial-kind", "nicla",
        "--policy-id", cell["policy_id"], "--model-id", _stage_model_id(runtime, cell),
        "--nicla-stage-id", cell["stage_id"], "--nicla-model-slot", cell["model_slot"],
        "--physical-block-seed", str(cell["physical_block_seed"]),
        "--session-id", cell["session_id"], "--condition", cell["condition"],
        "--input-manifest-sha256", input_sha,
        "--expected-device-firmware-sha256", firmware_sha,
        "--expected-device-config-sha256", config_sha,
        "--meter-label", args.meter_label, "--meter-firmware", args.meter_firmware,
        "--topology", args.topology, "--load-description", args.load_description,
        "--stimulus-block", "development", "--stimulus-condition", "mixed",
        "--stimulus-count", str(args.stimulus_count),
        "--stimulus-seed", str(cell["paired_stimulus_seed"]),
        "--stimulus-session-id", cell["session_id"], "--stimulus-event-log", str(stimulus_log),
        "--stimulus-lighting-state", "normal", "--stimulus-motion-state", "stationary",
        "--stimulus-range-cm", "50", "--stimulus-hold-seconds", str(args.stimulus_hold_seconds),
        "--stimulus-blank-seconds", str(args.stimulus_blank_seconds),
        "--output-dir", str(out_dir)]
    if cell.get("preload_model"):
        command.extend(["--nicla-preload-model", cell["preload_model"]])
    if args.meter_path_fragment:
        command.extend(["--path-fragment", args.meter_path_fragment])
    if args.nicla_dtr:
        command.append("--nicla-dtr")
    if args.nicla_rts:
        command.append("--nicla-rts")
    return command


def _result(out_dir: Path, code: int, cell: dict, command: list[str], plan_sha: str,
            g_receipt: dict, h_receipt: dict, firmware_sha: str, config_sha: str) -> dict:
    metadata_path = out_dir / "run_metadata.json"
    raw_power = out_dir / "raw_packets.jsonl"
    raw_nicla = out_dir / "nicla_serial_raw.bin"
    serial_events = out_dir / "nicla_serial_events.jsonl"
    stimulus_log = out_dir / "stimulus-events.jsonl"
    metadata = _read(metadata_path) if metadata_path.is_file() else {}
    serial_rows = []
    malformed = 0
    if serial_events.is_file():
        for line in serial_events.read_text(encoding="utf-8").splitlines():
            try:
                serial_rows.append(json.loads(line))
            except ValueError:
                malformed += 1
    records = [row.get("record", {}) for row in serial_rows
               if row.get("validation", {}).get("status") == "accepted"]
    accepted = len(records)
    rejected = sum(row.get("validation", {}).get("status") == "rejected" for row in serial_rows)
    stage = cell["stage_id"]
    if stage == "idle":
        stage_observations = sum(row.get("event") == "measurement_stage_idle" for row in records)
    elif stage == "full_pipeline":
        stage_observations = sum(row.get("event") == "decision" for row in records)
    else:
        stage_observations = sum(row.get("event") == "measurement_stage_result" and
            row.get("stage_id") == stage for row in records)
    warm_records = [row for row in records if row.get("event") == "measurement_stage_result" and
                    row.get("stage_id") == "warm_inference"]
    warm_ok = (stage != "warm_inference" or
               (len(warm_records) > 0 and all(row.get("model_warm") is True for row in warm_records)))
    idle_camera_sleep_ok = (stage != "idle" or any(row.get("event") == "measurement_stage_idle" and
        row.get("camera_sleeping") is True for row in records))
    checks = {
        "capture_process_exit_zero": code == 0,
        "run_metadata_complete": metadata.get("status") == "complete",
        "nicla_start_ack_matches_stage": metadata.get("nicla_startup_receipt", {}).get("start_ack", {}).get(
            "measurement_stage_id") == stage,
        "nicla_live_firmware_hash_matches": metadata.get("nicla_startup_receipt", {}).get(
            "observed_firmware_sha256") == firmware_sha,
        "nicla_live_config_hash_matches": metadata.get("nicla_startup_receipt", {}).get(
            "observed_config_sha256") == config_sha,
        "nicla_stop_ack_ok": metadata.get("nicla_stop_ack", {}).get("control_status", "ok") == "ok",
        "nicla_raw_serial_present": raw_nicla.is_file(),
        "nicla_serial_events_present": serial_events.is_file(),
        "nicla_valid_records_present": accepted > 0,
        "nicla_rejections_absent": rejected == 0 and malformed == 0,
        "lifecycle_stage_observed": stage_observations > 0,
        "idle_stage_camera_sleep_confirmed": idle_camera_sleep_ok,
        "warm_inference_preloaded_and_retained": warm_ok,
        "power_packets_present": raw_power.is_file() and int(metadata.get("measurement_report_count", 0)) > 0,
        "power_invalid_count_zero": int(metadata.get("invalid_packet_count", 0)) == 0,
        "balanced_stimulus_completed": metadata.get("stimulus_player", {}).get("status") == "complete",
        "stimulus_log_hash_bound": metadata.get("stimulus_player", {}).get("event_log_sha256") ==
            (sha256(stimulus_log) if stimulus_log.is_file() else None),
        "G_receipt_complete": g_receipt.get("status") == "complete_verified",
        "H_receipt_complete": h_receipt.get("status") == "complete_verified",
    }
    return {"schema": "I-lifecycle-capture-result-1",
        "status": "complete_verified" if all(checks.values()) else "failed_qc_or_incomplete",
        "evidence_tier": "physical_capture_pending_formal_aggregate_analysis",
        "run_id": cell["run_id"], "block_id": cell["block_id"],
        "stage_id": stage, "policy_id": cell["policy_id"], "model_slot": cell["model_slot"],
        "training_root_seed": cell.get("training_root_seed"),
        "runtime_model_root_seeds": cell.get("runtime_model_root_seeds"),
        "physical_block_seed": cell["physical_block_seed"],
        "paired_stimulus_seed": cell["paired_stimulus_seed"],
        "session_id": cell["session_id"], "plan_sha256": plan_sha,
        "accepted_serial_records": accepted, "rejected_serial_records": rejected,
        "stage_observation_count": stage_observations,
        "raw_power_sha256": sha256(raw_power) if raw_power.is_file() else None,
        "raw_serial_sha256": sha256(raw_nicla) if raw_nicla.is_file() else None,
        "serial_event_log_sha256": sha256(serial_events) if serial_events.is_file() else None,
        "stimulus_event_log_sha256": sha256(stimulus_log) if stimulus_log.is_file() else None,
        "checks": checks,
        "scope": "This runner captures lifecycle stages for the exact currently deployed pair. It does not execute the separate 544-comparison E model matrix or claim energy/recognition acceptance before aggregate analysis.",
        "command": command}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--execute", action="store_true", help="perform gated Windows physical captures; default is preview")
    p.add_argument("--dry-run", action="store_true", help="explicitly request the default non-physical preview")
    p.add_argument("--resume", action="store_true", help="skip only complete_verified, plan-bound captures")
    p.add_argument("--only-run-id", help="execute or preview one planned lifecycle cell")
    p.add_argument("--serial-port")
    p.add_argument("--meter-interface", type=int)
    p.add_argument("--meter-path-fragment")
    p.add_argument("--board-label")
    p.add_argument("--device-reset-id")
    p.add_argument("--firmware-source", type=Path)
    p.add_argument("--firmware-binary", type=Path)
    p.add_argument("--runtime-config", type=Path, default=APP / "firmware/runtime_config.json")
    p.add_argument("--firmware-version")
    p.add_argument("--firmware-sha256")
    p.add_argument("--config-sha256")
    p.add_argument("--g-qualification-receipt", type=Path)
    p.add_argument("--h-qualification-receipt", type=Path)
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--warmup-seconds", type=float, default=5.0)
    p.add_argument("--stimulus-count", type=int, default=12)
    p.add_argument("--stimulus-hold-seconds", type=float, default=1.5)
    p.add_argument("--stimulus-blank-seconds", type=float, default=1.0)
    p.add_argument("--meter-label", default="FNIRSI FNB-C2; verify against live descriptor")
    p.add_argument("--meter-firmware")
    p.add_argument("--topology")
    p.add_argument("--load-description")
    p.add_argument("--nicla-dtr", action="store_true")
    p.add_argument("--nicla-rts", action="store_true")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.plan.is_file():
        parser().error("I lifecycle plan is missing; run nicla_study.prepare_jik first")
    plan = _read(args.plan)
    cells = plan.get("cells", [])
    if args.only_run_id:
        cells = [row for row in cells if row.get("run_id") == args.only_run_id]
        if not cells:
            parser().error("planned lifecycle run ID not found")
    plan_sha = sha256(args.plan)
    if not args.execute:
        print(json.dumps({"status": "dry_run_only", "plan_sha256": plan_sha,
            "lifecycle_cells_in_view": len(cells), "plan_cell_count": len(plan.get("cells", [])),
            "paired_blocks": plan.get("paired_block_count"),
            "stage_ids": sorted({row.get("stage_id") for row in plan.get("cells", [])}),
            "current_model_ids": plan.get("current_runtime_model_ids"),
            "physical_measurement_performed": False,
            "runner_scope": "selected deployed model pair lifecycle only; E comparison matrix is separate"}, indent=2))
        return 0

    required = {"--serial-port": args.serial_port, "--meter-interface": args.meter_interface,
        "--board-label": args.board_label, "--device-reset-id": args.device_reset_id,
        "--firmware-source": args.firmware_source, "--firmware-binary": args.firmware_binary,
        "--firmware-version": args.firmware_version, "--firmware-sha256": args.firmware_sha256,
        "--config-sha256": args.config_sha256, "--G receipt": args.g_qualification_receipt,
        "--H receipt": args.h_qualification_receipt, "--meter firmware": args.meter_firmware,
        "--topology": args.topology, "--load description": args.load_description}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser().error("physical execution requires " + ", ".join(missing))
    for label, value in (("firmware", args.firmware_sha256), ("config", args.config_sha256)):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)):
            parser().error(label + " hash must be 64 hexadecimal characters")
    if not args.runtime_config.is_file():
        parser().error("exact runtime config file does not exist")
    runtime = _read(args.runtime_config)
    if sha256(args.runtime_config).lower() != args.config_sha256.lower():
        parser().error("runtime config file does not match --config-sha256")
    if _firmware_hash().lower() != args.firmware_sha256.lower():
        parser().error("current four-file firmware source hash does not match --firmware-sha256")
    required_stage_ids = {cell["stage_id"] for cell in cells}
    advertised = set(runtime.get("control", {}).get("measurement_stage_ids", []))
    if not required_stage_ids.issubset(advertised):
        parser().error("runtime config lacks stage IDs: " + ", ".join(sorted(required_stage_ids - advertised)))
    g_receipt = _receipt(args.g_qualification_receipt, "G meter/instrument")
    h_receipt = _receipt(args.h_qualification_receipt, "H Nicla")
    if not _hash_present(h_receipt, args.firmware_sha256) or not _hash_present(h_receipt, args.config_sha256):
        parser().error("H receipt does not qualify the exact source/config hashes")
    input_manifest = WORKSPACE / "work/project-finish-v4/data/primary_split_manifest.jsonl"
    if not input_manifest.is_file():
        parser().error("development presentation manifest is missing")
    input_sha = sha256(input_manifest)
    for label, path in (("firmware source", args.firmware_source), ("firmware binary", args.firmware_binary)):
        if not path.is_file():
            parser().error(label + " path does not exist: " + str(path))
    if args.duration <= 0 or args.duration > 600 or args.warmup_seconds < 0:
        parser().error("bounded capture duration or warmup is invalid")
    stimulus_time = args.stimulus_count * (args.stimulus_hold_seconds + args.stimulus_blank_seconds)
    if args.stimulus_count != 12 or stimulus_time > args.duration:
        parser().error("balanced mixed lifecycle capture requires exactly 12 stimuli covered by measurement duration")

    run_root = STUDY / "runs/I/lifecycle"
    for cell in cells:
        if cell.get("preload_model") and int(runtime.get("models", {}).get("max_resident_models", 1)) < 2:
            ledger_path = run_root / "CAMPAIGN_EVENTS.jsonl"
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"planned_run_id": cell["run_id"],
                    "block_id": cell["block_id"], "status": "unsupported_preflight",
                    "reason": "warm_inference_requires_two_resident_models; current exact runtime config sets max_resident_models<2",
                    "plan_sha256": plan_sha, "config_sha256": args.config_sha256.lower(),
                    "physical_measurement_performed": False,
                    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, sort_keys=True) + "\n")
            print(json.dumps({"planned_run_id": cell["run_id"], "status": "unsupported_preflight",
                "reason": "warm_inference_requires_two_resident_models"}, indent=2))
            continue
        original_dir = run_root / cell["run_id"]
        result_path = original_dir / "capture_result.json"
        if args.resume and result_path.is_file():
            prior = _read(result_path)
            if prior.get("status") == "complete_verified" and prior.get("plan_sha256") == plan_sha:
                continue
        run_id = cell["run_id"]
        rerun_of = None
        out_dir = original_dir
        if out_dir.exists():
            rerun_of = run_id
            rerun_index = 2
            while (run_root / (run_id + "-R%02d" % rerun_index)).exists():
                rerun_index += 1
            run_id += "-R%02d" % rerun_index
            out_dir = run_root / run_id
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        command_cell = dict(cell, run_id=run_id)
        command = _build_command(args, runtime, command_cell, out_dir,
            args.firmware_sha256, args.config_sha256, input_sha)
        process = subprocess.run(command, cwd=WORKSPACE, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        result = _result(out_dir, process.returncode, command_cell, command,
            plan_sha, g_receipt, h_receipt, args.firmware_sha256.lower(), args.config_sha256.lower())
        result["rerun_of"] = rerun_of
        result["capture_stdout_tail"] = (process.stdout or "")[-4000:]
        result["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(out_dir / "capture_result.json", result)
        ledger_path = run_root / "CAMPAIGN_EVENTS.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"planned_run_id": cell["run_id"], "actual_run_id": run_id,
                "block_id": cell["block_id"], "status": result["status"],
                "result_path": str(out_dir / "capture_result.json"),
                "result_sha256": sha256(out_dir / "capture_result.json"),
                "utc": result["created_utc"]}, sort_keys=True) + "\n")
            stream.flush()
        print(json.dumps({"planned_run_id": cell["run_id"], "actual_run_id": run_id,
            "status": result["status"], "output_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
