"""Phase-aware Mac/Windows Nicla study entry point.

This module intentionally has no eager TensorFlow, NumPy, HID, or serial imports.
Windows discovery/capture can run in its separate instrument environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import zipfile

from .artifacts import verify_artifact_manifest
from .fs import atomic_write_json, sha256_file
from .inventory import fresh_device_discovery, make_inventory, probe_interpreter
from .paths import PathResolutionError, resolve_registry
from .plan import validate_experiment_plan, validate_phase_plan
from .roles import assert_role_access, validate_manifest
from .seeds import validate_roots
from .state import (complete_phase_from_receipt, dispatch_analysis, initialize_state,
                    mark_stale_running_phases_interrupted, reconcile_stale_resource_leases,
                    resume_phase, start_phase,
                    state_paths, transact)


PROTOCOL_ID = "nicla-five-known-study-v4"
SUCCESS_STATUSES = {"complete_verified"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    atomic_write_json(path, value)


def _workspace(args) -> Path:
    if getattr(args, "workspace", None):
        return Path(args.workspace).expanduser().resolve()
    if os.name == "nt":
        raise PathResolutionError("Windows mode requires --workspace for the transferred root")
    return Path(__file__).resolve().parents[2]


def _paths(workspace: Path) -> tuple[Path, dict]:
    study = workspace / "work" / "project-finish-v4"
    registry = _read_json(study / "config" / "paths.json")
    return study, registry


def _phase_plan(study: Path) -> dict:
    return _read_json(study / "config" / "PLATFORM_PHASE_PLAN.json")


def _experiment_plan(study: Path) -> dict:
    return _read_json(study / "config" / "EXPERIMENT_PLAN.json")


def command_init(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    state = initialize_state(study, _phase_plan(study),
                             "work/project-finish-v4/config/EXPERIMENT_PLAN.json", PROTOCOL_ID)
    print(json.dumps({"status": "initialized_or_resumed", "study_root": str(study),
                      "state": state_paths(study)[0].relative_to(workspace).as_posix(),
                      "phase_count": len(state["phases"])}, indent=2))
    return 0


def command_discover(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    discovery = fresh_device_discovery()
    output = study / "reports" / "fresh-device-discovery.json"
    _write_json(output, discovery)
    state_file = state_paths(study)[0]
    if state_file.is_file():
        found = discovery
        def mutate(state):
            state["devices"]["last_fresh_discovery_utc"] = found["enumerated_utc"]
            if os.name == "nt":
                devices = found.get("hid_devices", [])
                serials = found.get("serial_ports", [])
                state["devices"]["fnirsi"] = "observed" if any(d.get("vendor_id") == 0x2E3C and d.get("product_id") == 0x5558 for d in devices) else "not_observed"
                state["devices"]["nano"] = "observed" if serials else "not_observed"
                state["devices"]["nicla"] = "observed_only_if_user_identified" if any("nicla" in str(p).lower() for p in serials) else "not_observed"
            else:
                has_usb_serial = bool(found.get("serial_ports"))
                state["devices"]["nicla"] = "not_observed"
                state["devices"]["fnirsi"] = "not_observed"
                state["devices"]["nano"] = "not_observed"
                state["devices"]["usb_uart"] = "observed" if has_usb_serial else "not_observed"
            return None
        transact(study, mutate, event_type="fresh_device_discovery",
                 event_details={"report": output.relative_to(workspace).as_posix(),
                                "observed_count": discovery.get("attached_device_count")})
    print(json.dumps({"report": str(output), "attached_device_count": discovery.get("attached_device_count"),
                      "enumerated_utc": discovery.get("enumerated_utc"),
                      "errors": discovery.get("errors", [])}, indent=2))
    return 0


def command_resolve_paths(args) -> int:
    workspace = _workspace(args)
    study, registry = _paths(workspace)
    resolved = resolve_registry(registry, workspace=workspace)
    output = study / "reports" / "resolved-path-registry.json"
    _write_json(output, resolved)
    print(json.dumps({"report": str(output), **resolved}, indent=2))
    return 0


def command_doctor(args) -> int:
    workspace = _workspace(args)
    study, registry = _paths(workspace)
    discovery = fresh_device_discovery()
    output = study / "reports" / ("doctor-" + args.mode + ".json")
    training_python = workspace / registry["paths"]["training_interpreter"]["relative"]
    instrument_python = workspace / registry["windows"]["instrument_interpreter_relative"]
    runtime = probe_interpreter(training_python) if args.mode == "host" else {
        "path": str(instrument_python), "status": "not_required_for_host_mode"}
    evidence_root = workspace / "work/finish-line-agent-pack-2026-10-28/references/windows-evidence"
    subset_receipt_path = workspace / "work/finish-line-agent-pack-2026-10-28/references/WINDOWS_EVIDENCE_SUBSET.json"
    subset_receipt = _read_json(subset_receipt_path) if subset_receipt_path.is_file() else {}
    evidence_files = sum(1 for path in evidence_root.rglob("*") if path.is_file()) if evidence_root.is_dir() else 0
    checks = {
        "workspace": workspace.is_dir(),
        "canonical_repository": (workspace / "FaceRecognition").is_dir(),
        "study_configuration": (study / "config/EXPERIMENT_PLAN.json").is_file(),
        "curated_windows_evidence_subset_available": evidence_files == 126,
        "training_interpreter": training_python.is_file() if args.mode == "host" else True,
        "windows_instrument_interpreter": instrument_python.is_file() if args.mode == "windows" and os.name == "nt" else args.mode != "windows",
        "tensorflow_not_imported_in_runner": "tensorflow" not in sys.modules,
    }
    checks["curated_windows_evidence_provenance_valid"] = (
        subset_receipt.get("schema") == "windows-evidence-subset-1"
        and subset_receipt.get("source_zip_sha256") == "d85593caadeeebe9841db3338870cefe484eb63f738599d15d30f882f0758fe4"
        and subset_receipt.get("files_copied") == evidence_files == 126)
    status = "ready_host" if args.mode == "host" and all(checks.values()) and runtime.get("status") == "pass" else "host_pending"
    if args.mode == "windows":
        status = "ready_windows_tools" if os.name == "nt" and checks["windows_instrument_interpreter"] else "windows_host_required"
    report = {
        "schema": "nicla-study-doctor-1", "protocol_id": PROTOCOL_ID,
        "mode": args.mode, "captured_utc": _now(), "workspace": str(workspace),
        "checks": checks, "training_environment": runtime,
        "fresh_device_discovery": discovery,
        "device_claim_policy": "current enumeration only; no historical attached_devices or COM-port value is copied",
        "status": status,
    }
    _write_json(output, report)
    print(json.dumps({"status": status, "report": str(output), "checks": checks,
                      "training_environment": runtime, "fresh_devices": discovery.get("attached_device_count")}, indent=2))
    return 0 if status in {"ready_host", "ready_windows_tools"} else 2


def command_inventory(args) -> int:
    workspace = _workspace(args)
    study, registry = _paths(workspace)
    report = make_inventory(workspace, study)
    training_python = workspace / registry["paths"]["training_interpreter"]["relative"]
    report["training_environment"] = probe_interpreter(training_python)
    report["windows_instrument_environment"] = {
        "target_python": registry["windows"]["instrument_python_target"],
        "runtime_relative": registry["windows"]["instrument_interpreter_relative"],
        "available_on_current_host": (workspace / registry["windows"]["instrument_interpreter_relative"]).is_file()
                                     if os.name == "nt" else False,
        "tensorflow_required": False,
    }
    report["input_source_versions"] = {
        "v3_protocol": "model-system-2026-10-23-v3",
        "v3_training_environment_documented": "TensorFlow 2.10.1 / Python 3.9.6",
        "v3_run_manifests_also_record": ["TensorFlow 2.10.1", "TensorFlow 2.13.0-rc2"],
        "v4_selected_training_environment": "existing frozen Python 3.9.6 / TensorFlow 2.10.1 CPU environment",
    }
    output = study / "reports" / "inventory-initial.json"
    _write_json(output, report)
    print(json.dumps({"report": str(output), "system": report["system"],
                      "v3_model_inventory": report.get("existing_v3_models"),
                      "training_environment": report["training_environment"]}, indent=2))
    return 0


def command_validate(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    phase_plan = _phase_plan(study)
    experiment_plan = _experiment_plan(study)
    phase_result = validate_phase_plan(phase_plan)
    experiment_result = validate_experiment_plan(experiment_plan)
    phase_ids = {row["id"] for row in phase_plan.get("phases", [])}
    experiment_ids = {row.get("id") for row in experiment_plan.get("experiments", [])}
    unknown = []
    for row in experiment_plan.get("experiments", []):
        for prerequisite in row.get("prerequisites", []):
            if prerequisite not in phase_ids and prerequisite not in experiment_ids:
                unknown.append("%s references %s" % (row.get("id"), prerequisite))
    roots = _read_json(study / "config/seed_registry.json")["training_roots"]
    try:
        validate_roots([row["root_seed"] for row in roots], minimum=10)
        seeds_ok = True
    except Exception as exc:
        seeds_ok = False
        unknown.append("training root registry invalid: " + str(exc))
    report = {"schema": "nicla-plan-validation-1", "captured_utc": _now(),
              "phase_plan": phase_result, "experiment_plan": experiment_result,
              "unknown_experiment_prerequisites": unknown,
              "training_root_count": len(roots), "training_roots_distinct": seeds_ok,
              "status": "pass" if phase_result["status"] == "pass" and experiment_result["status"] == "pass" and not unknown else "invalid"}
    output = study / "reports" / "plan-validation.json"
    _write_json(output, report)
    print(json.dumps({**report, "report": str(output)}, indent=2))
    return 0 if report["status"] == "pass" else 1


def command_status(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    state_file = state_paths(study)[0]
    if not state_file.is_file():
        print(json.dumps({"status": "not_initialized", "study_root": str(study)}, indent=2))
        return 2
    state = _read_json(state_file)
    if args.job:
        output = {"job": args.job, "job_state": state.get("jobs", {}).get(args.job),
                  "phases": {key: value for key, value in state.get("phases", {}).items()
                             if value.get("job") == args.job}}
    else:
        output = {"protocol_id": state.get("protocol_id"), "active_job": state.get("active_job"),
                  "active_host": state.get("active_host"), "jobs": state.get("jobs"),
                  "phases": state.get("phases"), "devices": state.get("devices"),
                  "pending_external_actions": state.get("pending_external_actions")}
    print(json.dumps(output, indent=2))
    return 0


def command_resume(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    if args.phase:
        row = resume_phase(study, args.phase)
        print(json.dumps({"phase": args.phase, "status": row["status"], "resumable": True}, indent=2))
    else:
        changed = mark_stale_running_phases_interrupted(study)
        stale_leases = reconcile_stale_resource_leases(study)
        print(json.dumps({"interrupted_phases": changed, "reconciled_stale_leases": stale_leases,
                          "next_action": "resume --phase <phase-id> after checking its receipt and inputs"}, indent=2))
    return 0


def command_phase_start(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    row = start_phase(study, args.phase)
    print(json.dumps({"phase": args.phase, "status": row["status"],
                      "owner": row["owner"], "resources": row["resources"]}, indent=2))
    return 0


def command_phase_complete(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    receipt = Path(args.receipt)
    if not receipt.is_absolute():
        receipt = workspace / receipt
    row = complete_phase_from_receipt(study, args.phase, receipt)
    def update_job(state):
        job = row["job"]
        state["jobs"].setdefault(job, {"subphases": {}})
        state["jobs"][job]["status"] = "complete_verified"
        state["active_job"] = None
    transact(study, update_job, event_type="job_completed", event_details={"job": row["job"]})
    print(json.dumps({"phase": args.phase, "status": row["status"], "receipt": str(receipt)}, indent=2))
    return 0


def command_dispatch_analysis(args) -> int:
    workspace = _workspace(args)
    study, _ = _paths(workspace)
    result = dispatch_analysis(study, args.job, adapters={})
    print(json.dumps(result, indent=2))
    return int(result["exit_code"])


def command_verify_artifacts(args) -> int:
    workspace = _workspace(args)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = workspace / manifest_path
    result = verify_artifact_manifest(workspace, _read_json(manifest_path))
    result["manifest"] = str(manifest_path)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


def command_package_return(args) -> int:
    workspace = _workspace(args)
    source = Path(args.source)
    source = source if source.is_absolute() else workspace / source
    source = source.resolve()
    try:
        relative = source.relative_to(workspace)
    except ValueError:
        raise SystemExit("return package source must stay within the verified workspace")
    if any(part.lower() in {".venv", "venv", "private", "participant", "templates"} for part in relative.parts):
        raise SystemExit("return source includes a protected environment or private-data path")
    destination = Path(args.destination)
    destination = destination if destination.is_absolute() else workspace / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit("refusing to overwrite existing return package: " + str(destination))
    files = []
    for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        if any(part.lower() in {".venv", "venv", "private", "participant", "templates"} for part in Path(rel).parts):
            continue
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema": "partial-return-package-1", "created_utc": _now(),
                "status": args.status, "files": files,
                "note": "Partial evidence packaging preserves failure status and does not complete experiments."}
    if destination.suffix.lower() == ".zip":
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for row in files:
                archive.write(workspace / row["path"], row["path"])
            archive.writestr("PARTIAL_RETURN_MANIFEST.json", json.dumps(manifest, indent=2))
    else:
        with tarfile.open(destination, "x:gz") as archive:
            for row in files:
                archive.add(workspace / row["path"], arcname=row["path"], recursive=False)
            import io
            data = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("PARTIAL_RETURN_MANIFEST.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    print(json.dumps({"destination": str(destination), "status": args.status,
                      "files": len(files), "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def with_workspace(child):
        child.add_argument("--workspace", type=Path, help="transferred workspace root; required on Windows")
    p = sub.add_parser("init", help="initialize or resume the shared state files"); with_workspace(p); p.set_defaults(fn=command_init)
    p = sub.add_parser("discover", help="freshly enumerate current USB/serial interfaces"); with_workspace(p); p.set_defaults(fn=command_discover)
    p = sub.add_parser("resolve-paths", help="resolve every portable path against this host workspace"); with_workspace(p); p.set_defaults(fn=command_resolve_paths)
    p = sub.add_parser("doctor", help="check host mode without opening data or devices"); with_workspace(p); p.add_argument("--mode", choices=["host","windows"], default="host"); p.set_defaults(fn=command_doctor)
    p = sub.add_parser("inventory", help="write current source/artifact/disk/compute inventory"); with_workspace(p); p.set_defaults(fn=command_inventory)
    p = sub.add_parser("validate-plan", help="validate phase DAG, finite experiment matrix and roots"); with_workspace(p); p.set_defaults(fn=command_validate)
    p = sub.add_parser("status", help="read the single machine progress state"); with_workspace(p); p.add_argument("--job"); p.set_defaults(fn=command_status)
    p = sub.add_parser("resume", help="recover stale phases or mark one phase resumable"); with_workspace(p); p.add_argument("--phase"); p.set_defaults(fn=command_resume)
    p = sub.add_parser("phase-start", help="start a phase only after prerequisites pass"); with_workspace(p); p.add_argument("phase"); p.set_defaults(fn=command_phase_start)
    p = sub.add_parser("phase-complete", help="complete from a verified receipt only"); with_workspace(p); p.add_argument("phase"); p.add_argument("--receipt", required=True); p.set_defaults(fn=command_phase_complete)
    p = sub.add_parser("dispatch-analysis", help="dispatch only a registered implemented analysis action"); with_workspace(p); p.add_argument("job"); p.set_defaults(fn=command_dispatch_analysis)
    p = sub.add_parser("verify-artifacts", help="verify every artifact-manifest file by size and SHA-256"); with_workspace(p); p.add_argument("--manifest", default="work/project-finish-v4/handoffs/A/artifact_manifest.json"); p.set_defaults(fn=command_verify_artifacts)
    p = sub.add_parser("package-return", help="create a partial evidence archive after success or failure"); with_workspace(p); p.add_argument("--source", required=True); p.add_argument("--destination", required=True); p.add_argument("--status", choices=["partial","complete"], default="partial"); p.set_defaults(fn=command_package_return)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError, RuntimeError, PathResolutionError, TimeoutError) as exc:
        print(json.dumps({"status": "failed_closed", "error": "%s: %s" % (type(exc).__name__, exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
