"""Create and verify the one-way development-only freeze gate for K."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_receipt_sources(receipt_path: Path, workspace: Path | None = None) -> dict:
    """Resolve portable v2 paths; leave immutable legacy absolute paths intact."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    relative = receipt.get("source_files_relative")
    if relative is None:
        return {key: Path(value).resolve() for key, value in receipt.get("source_files", {}).items()}
    if not isinstance(relative, dict):
        raise ValueError("portable source mapping is not an object")
    base = (workspace.resolve() if workspace is not None else
            (receipt_path.resolve().parent / receipt["workspace_from_receipt"]).resolve())
    paths = {}
    for key, value in relative.items():
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("portable source path escapes workspace")
        path = (base / value).resolve()
        path.relative_to(base)
        paths[key] = path
    return paths


def freeze(selection_path: Path, runtime_config: Path, model_file: Path,
           calibration_file: Path, roster_file: Path, controller_plan: Path,
           protocol_file: Path, firmware_sha256: str, output: Path,
           workspace: Path | None = None) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidate = selection.get("selected_candidate")
    if selection.get("status") != "selected_development_candidate" or not selection.get("freeze_allowed"):
        raise ValueError("selector has no qualified development candidate")
    if selection.get("data_role") != "development_only" or selection.get("used_final_data") is not False:
        raise PermissionError("selection is not explicitly development-only")
    if not isinstance(candidate, dict) or candidate.get("data_role") != "development" or candidate.get("used_final_data") is not False:
        raise PermissionError("selected candidate is not explicitly development-only")
    hashes = {
        "model_sha256": sha256(model_file),
        "config_sha256": sha256(runtime_config),
        "calibration_sha256": sha256(calibration_file),
        "roster_sha256": sha256(roster_file),
        "controller_sha256": sha256(controller_plan),
        "protocol_sha256": sha256(protocol_file),
        "firmware_sha256": firmware_sha256.lower(),
    }
    candidate_hash_fields = {
        "model_sha256": candidate.get("model_sha256"),
        "config_sha256": candidate.get("config_sha256", candidate.get("runtime_config_sha256")),
        "calibration_sha256": candidate.get("calibration_sha256"),
        "roster_sha256": candidate.get("roster_sha256"),
        "controller_sha256": candidate.get("controller_sha256"),
    }
    for key, value in candidate_hash_fields.items():
        if str(value or "").lower() != hashes[key]:
            raise ValueError("selected candidate %s does not match supplied file" % key)
    if not HASH_RE.fullmatch(hashes["firmware_sha256"]):
        raise ValueError("firmware SHA-256 must be exact")
    if output.exists():
        raise FileExistsError("freeze receipt already exists; create a new versioned protocol receipt")
    sources = {"runtime_config": runtime_config, "model": model_file,
               "calibration": calibration_file, "roster": roster_file,
               "controller_plan": controller_plan, "protocol": protocol_file,
               "selection_report": selection_path}
    base = (workspace.resolve() if workspace is not None else
            Path(os.path.commonpath([str(path.resolve().parent) for path in sources.values()])))
    relative = {key: path.resolve().relative_to(base).as_posix() for key, path in sources.items()}
    receipt = {
        "schema": "J-frozen-selection-2",
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_role": "development_only",
        "used_final_data": False,
        "final_role_opened": False,
        "selected_model_id": candidate.get("model_id"),
        **hashes,
        "selection_report_sha256": sha256(selection_path),
        "selection_rule_sha256": sha256(Path(__file__).resolve()),
        "selected_candidate_metrics": candidate.get("metrics"),
        "independent_paired_blocks": candidate.get("independent_paired_blocks"),
        "source_files": {"runtime_config": str(runtime_config), "model": str(model_file),
            "calibration": str(calibration_file), "roster": str(roster_file),
            "controller_plan": str(controller_plan), "protocol": str(protocol_file),
            "selection_report": str(selection_path)},
        "source_files_relative": relative,
        "workspace_from_receipt": os.path.relpath(base, output.resolve().parent),
        "final_access_policy": "this receipt permits evaluation only; K player must still enforce the data-role gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    return receipt


def verify(receipt_path: Path, runtime_config: Path, model_file: Path,
           calibration_file: Path, roster_file: Path, controller_plan: Path,
           protocol_file: Path, firmware_sha256: str) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False:
        raise PermissionError("receipt is not a valid sealed freeze")
    actual = {"model_sha256": sha256(model_file), "config_sha256": sha256(runtime_config),
        "calibration_sha256": sha256(calibration_file), "roster_sha256": sha256(roster_file),
        "controller_sha256": sha256(controller_plan), "protocol_sha256": sha256(protocol_file),
        "firmware_sha256": firmware_sha256.lower()}
    mismatches = [key for key, value in actual.items() if receipt.get(key) != value]
    if mismatches:
        raise ValueError("frozen selection hashes changed: " + ", ".join(mismatches))
    return {"status": "pass", "final_role_opened": False, "verified_hashes": actual,
        "receipt_sha256": sha256(receipt_path)}


def verify_receipt_sources(receipt_path: Path) -> dict:
    """Verify every local source referenced by a freeze before a final-role read."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False:
        raise PermissionError("receipt is not a sealed frozen selection")
    if receipt.get("used_final_data") is not False or receipt.get("data_role") != "development_only":
        raise PermissionError("freeze must be derived from development-only evidence")
    sources = receipt.get("source_files")
    required = ("runtime_config", "model", "calibration", "roster", "controller_plan", "protocol", "selection_report")
    if not isinstance(sources, dict) or any(not sources.get(key) for key in required):
        raise ValueError("freeze receipt lacks source paths needed for independent verification")
    paths = resolve_receipt_sources(receipt_path)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("freeze source files are unavailable on this host: " + ", ".join(missing))
    selection = json.loads(paths["selection_report"].read_text(encoding="utf-8"))
    if (sha256(paths["selection_report"]) != receipt.get("selection_report_sha256") or
            selection.get("status") != "selected_development_candidate" or
            selection.get("data_role") != "development_only" or selection.get("used_final_data") is not False):
        raise ValueError("frozen selector report is missing, changed, or not development-only")
    candidate = selection.get("selected_candidate")
    if not isinstance(candidate, dict) or candidate.get("model_id") != receipt.get("selected_model_id"):
        raise ValueError("selected model does not match the development selector report")
    result = verify(receipt_path, paths["runtime_config"], paths["model"], paths["calibration"],
        paths["roster"], paths["controller_plan"], paths["protocol"], receipt.get("firmware_sha256", ""))
    config = json.loads(paths["runtime_config"].read_text(encoding="utf-8"))
    configured_models = {str(row.get("model_sha256", "")).lower()
        for row in config.get("identity", {}).values() if isinstance(row, dict)}
    if receipt.get("model_sha256", "").lower() not in configured_models:
        raise ValueError("frozen model hash is not present in the verified runtime config")
    return {**result, "selection_report_sha256": receipt["selection_report_sha256"],
            "selected_model_id": receipt.get("selected_model_id"), "source_file_count": len(paths),
            "resolved_source_files": {key: str(value) for key, value in paths.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("freeze", "verify"):
        cmd = sub.add_parser(name)
        if name == "freeze":
            cmd.add_argument("--selection", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
        else:
            cmd.add_argument("--receipt", type=Path, required=True)
        cmd.add_argument("--runtime-config", type=Path, required=True)
        cmd.add_argument("--model-file", type=Path, required=True)
        cmd.add_argument("--calibration-file", type=Path, required=True)
        cmd.add_argument("--roster-file", type=Path, required=True)
        cmd.add_argument("--controller-plan", type=Path, required=True)
        cmd.add_argument("--protocol-file", type=Path, required=True)
        cmd.add_argument("--firmware-sha256", required=True)
        cmd.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    if not HASH_RE.fullmatch(args.firmware_sha256.lower()):
        parser.error("--firmware-sha256 must be 64 hexadecimal characters")
    if args.action == "freeze":
        result = freeze(args.selection, args.runtime_config, args.model_file,
            args.calibration_file, args.roster_file, args.controller_plan,
            args.protocol_file, args.firmware_sha256, args.output, workspace=args.workspace)
    else:
        result = verify(args.receipt, args.runtime_config, args.model_file,
            args.calibration_file, args.roster_file, args.controller_plan,
            args.protocol_file, args.firmware_sha256)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
