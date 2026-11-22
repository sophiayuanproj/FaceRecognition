"""Create and validate independent optical-encounter annotations for K."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SCHEMA = "K-encounter-annotation-1"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("invalid JSON at line %d: %s" % (line_number, error)) from error
        if not isinstance(value, dict):
            raise ValueError("JSONL row %d must be an object" % line_number)
        rows.append(value)
    return rows


def _validate_freeze(receipt: dict | None, receipt_path: Path | None = None) -> None:
    if not isinstance(receipt, dict) or receipt.get("status") != "frozen":
        raise PermissionError("final annotations require the frozen J selection receipt")
    if receipt.get("final_role_opened") is not False or receipt.get("used_final_data") is not False:
        raise PermissionError("final annotation gate requires a sealed, development-only freeze")
    for field in ("model_sha256", "config_sha256", "calibration_sha256", "protocol_sha256",
                  "roster_sha256", "controller_sha256", "firmware_sha256"):
        if not isinstance(receipt.get(field), str) or not HASH_RE.fullmatch(receipt[field]):
            raise PermissionError("freeze receipt lacks valid " + field)
    if receipt_path is not None:
        from .freeze_selection import verify_receipt_sources
        verify_receipt_sources(receipt_path)


def _trials(event_rows: list[dict]) -> tuple[dict, dict, dict]:
    sessions = {row.get("run_id"): row for row in event_rows if row.get("event") == "session_start"}
    starts = {row.get("trial_id"): row for row in event_rows if row.get("event") == "stimulus_start"}
    ends = {row.get("trial_id"): row for row in event_rows if row.get("event") == "stimulus_end"}
    if len(sessions) != 1:
        raise ValueError("one presentation event log must contain exactly one session_start")
    if not starts:
        raise ValueError("presentation event log has no stimulus_start events")
    if set(starts) != set(ends):
        raise ValueError("each stimulus_start must have one stimulus_end")
    return sessions, starts, ends


def create_annotation_template(event_log: Path, output_jsonl: Path,
                               freeze_receipt_path: Path | None = None) -> dict:
    event_rows = _read_jsonl(event_log)
    sessions, starts, ends = _trials(event_rows)
    session = next(iter(sessions.values()))
    if session.get("block") == "final":
        _validate_freeze(json.loads(freeze_receipt_path.read_text(encoding="utf-8"))
                         if freeze_receipt_path else None, freeze_receipt_path)
    templates = []
    for trial_id, start in sorted(starts.items(), key=lambda item: int(item[1].get("trial_index", -1))):
        end = ends[trial_id]
        templates.append({
            "schema": SCHEMA,
            "run_id": start.get("run_id"),
            "session_id": session.get("session_id"),
            "trial_id": trial_id,
            "stimulus_event_condition": start.get("condition"),
            "stimulus_record_ids": start.get("record_ids", []),
            "stimulus_identity_ids": start.get("identity_ids", []),
            "stimulus_roles": start.get("roles", []),
            "stimulus_start_monotonic_ns": start.get("monotonic_ns"),
            "stimulus_end_monotonic_ns": end.get("monotonic_ns"),
            "observed_face_count": None,
            "observed_target_visible": None,
            "appearance_monotonic_ns": None,
            "usable_interval_start_monotonic_ns": None,
            "usable_interval_end_monotonic_ns": None,
            "disappearance_monotonic_ns": None,
            "appearance_time_source": None,
            "clock_offset_to_controller_ms": None,
            "clock_uncertainty_ms": None,
            "camera_to_target_range_cm": None,
            "range_measurement_method": None,
            "camera_orientation_deg": None,
            "lighting_description": None,
            "wearer_motion": None,
            "independent_observer_id": None,
            "observer_role": None,
            "controller_outputs_seen_before_annotation": None,
            "visibility_failure_reason": None,
            "notes": "",
        })
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in templates), encoding="utf-8")
    tmp.replace(output_jsonl)
    return {"schema": "K-annotation-template-receipt-1", "status": "template_created",
            "block": session.get("block"), "run_id": session.get("run_id"),
            "trial_count": len(templates), "event_log_sha256": _hash(event_log),
            "output_jsonl": str(output_jsonl.resolve())}


def _hash(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_annotations(event_log: Path, annotations_jsonl: Path,
                          freeze_receipt_path: Path | None = None) -> dict:
    event_rows = _read_jsonl(event_log)
    sessions, starts, ends = _trials(event_rows)
    session = next(iter(sessions.values()))
    block = session.get("block")
    if block == "final":
        _validate_freeze(json.loads(freeze_receipt_path.read_text(encoding="utf-8"))
                         if freeze_receipt_path else None, freeze_receipt_path)
    rows = _read_jsonl(annotations_jsonl)
    by_trial = {}
    errors = []
    allowed_sources = {"synchronized_optical_video", "witnessed_live_marker"}
    for index, row in enumerate(rows):
        prefix = "annotation[%d]" % index
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in by_trial:
            errors.append(prefix + ": missing or duplicate trial_id")
            continue
        by_trial[trial_id] = row
        if row.get("schema") != SCHEMA:
            errors.append(prefix + ": unsupported schema")
        if row.get("run_id") != session.get("run_id") or row.get("session_id") != session.get("session_id"):
            errors.append(prefix + ": run/session binding mismatch")
        if row.get("independent_observer_id") in (None, "") or row.get("observer_role") != "independent_observer":
            errors.append(prefix + ": independent observer identity/role missing")
        if row.get("controller_outputs_seen_before_annotation") is not False:
            errors.append(prefix + ": annotation must be made without seeing controller outputs")
        if row.get("appearance_time_source") not in allowed_sources:
            errors.append(prefix + ": appearance_time_source must use video synchronization or a witnessed live marker")
        for name in ("clock_offset_to_controller_ms", "clock_uncertainty_ms", "camera_to_target_range_cm",
                     "camera_orientation_deg"):
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(prefix + ": missing numeric " + name)
        if isinstance(row.get("clock_uncertainty_ms"), (int, float)) and row["clock_uncertainty_ms"] < 0:
            errors.append(prefix + ": clock uncertainty must be nonnegative")
        if not row.get("range_measurement_method") or not row.get("lighting_description") or not row.get("wearer_motion"):
            errors.append(prefix + ": range method, visible lighting description and wearer motion are required")
        if row.get("stimulus_record_ids") != starts.get(trial_id, {}).get("record_ids"):
            errors.append(prefix + ": source record IDs differ from presentation event")

    expected_trials = set(starts)
    if set(by_trial) != expected_trials:
        missing = sorted(expected_trials - set(by_trial))
        extra = sorted(set(by_trial) - expected_trials)
        if missing:
            errors.append("annotations missing trials: " + ", ".join(missing[:20]))
        if extra:
            errors.append("annotations contain trials absent from event log: " + ", ".join(extra[:20]))

    for trial_id, start in starts.items():
        row = by_trial.get(trial_id)
        if row is None:
            continue
        condition = start.get("condition")
        count = row.get("observed_face_count")
        visible = row.get("observed_target_visible")
        if type(count) is not int or count < 0 or type(visible) is not bool:
            errors.append(trial_id + ": observed_face_count and observed_target_visible must be explicit")
            continue
        if condition == "empty" and (count != 0 or visible):
            errors.append(trial_id + ": empty stimulus must have no visible target")
        elif condition == "known" and (count != 1 or not visible):
            if not row.get("visibility_failure_reason"):
                errors.append(trial_id + ": known presentation must be visible or have a visibility-failure reason")
        elif condition == "unknown" and (count != 1 or not visible):
            if not row.get("visibility_failure_reason"):
                errors.append(trial_id + ": unknown presentation must be visible or have a visibility-failure reason")
        elif condition == "ambiguous" and count < 2:
            if not row.get("visibility_failure_reason"):
                errors.append(trial_id + ": ambiguous presentation needs two visible faces or a visibility-failure reason")
        if visible:
            start_ns = start.get("monotonic_ns")
            end_ns = ends[trial_id].get("monotonic_ns")
            times = [row.get("appearance_monotonic_ns"), row.get("usable_interval_start_monotonic_ns"),
                     row.get("usable_interval_end_monotonic_ns"), row.get("disappearance_monotonic_ns")]
            if any(type(value) is not int for value in times):
                errors.append(trial_id + ": visible target needs appearance, usable interval, and disappearance times")
            elif not (start_ns <= times[0] <= times[1] < times[2] <= times[3] <= end_ns):
                errors.append(trial_id + ": annotated visibility interval falls outside presentation markers or is inverted")
        elif any(row.get(key) is not None for key in ("appearance_monotonic_ns", "usable_interval_start_monotonic_ns",
                                                        "usable_interval_end_monotonic_ns", "disappearance_monotonic_ns")):
            errors.append(trial_id + ": invisible target must not receive fabricated appearance times")
    return {
        "schema": "K-annotation-validation-1",
        "status": "pass" if not errors else "failed_annotation_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_tier": "offline_annotation_contract_validation",
        "final_role_opened": False,
        "block": block,
        "run_id": session.get("run_id"),
        "session_id": session.get("session_id"),
        "event_log_sha256": _hash(event_log),
        "annotations_sha256": _hash(annotations_jsonl),
        "trial_count": len(starts),
        "annotation_count": len(rows),
        "errors": errors,
        "scope": "checks independent annotation structure and clock bounds; does not prove optical visibility or validate the physical camera",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("template", "validate"):
        command = sub.add_parser(action)
        command.add_argument("--event-log", type=Path, required=True)
        command.add_argument("--freeze-receipt", type=Path)
        if action == "template":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--annotations", type=Path, required=True)
            command.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "template":
        result = create_annotation_template(args.event_log, args.output, args.freeze_receipt)
    else:
        result = validate_annotations(args.event_log, args.annotations, args.freeze_receipt)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(args.output_json)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"pass", "template_created"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
