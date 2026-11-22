"""Fail-closed dataset role manifests with identity-group leakage checks."""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Iterable

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLE_ACTIONS = {
    "train": {"train", "stopping", "unknown_train", "scene_train"},
    "quantize": {"train", "scene_train"},
    "threshold_fit": {"threshold_calibration", "unknown_calibration"},
    "select": {"development", "unknown_development"},
    "evaluate_final": {"final", "unknown_final"},
    "scene_train": {"scene_train"},
}
KNOWN_ROLES = {
    "train", "stopping", "threshold_calibration", "development", "final",
    "unknown_train", "unknown_calibration", "unknown_development", "unknown_final",
    "scene_train", "scene_development", "scene_final",
}


def validate_manifest(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    errors: list[str] = []
    seen_records = set()
    known_seen = set()
    unknown_partitions: dict[str, str] = {}
    allowed_unknown_roles = {
        "unknown_train": "train", "unknown_calibration": "calibration",
        "unknown_development": "development", "unknown_final": "final",
    }
    for index, row in enumerate(rows):
        prefix = "row[%d]" % index
        if not isinstance(row, dict):
            errors.append(prefix + ": expected object")
            continue
        record_id = row.get("record_id")
        identity = row.get("identity_id")
        role = row.get("role")
        digest = row.get("sha256")
        rel = row.get("relative_path")
        if not isinstance(record_id, str) or not record_id:
            errors.append(prefix + ": missing record_id")
        elif record_id in seen_records:
            errors.append(prefix + ": duplicate record_id " + record_id)
        else:
            seen_records.add(record_id)
        if not isinstance(identity, str) or not identity:
            errors.append(prefix + ": missing identity_id")
        if role not in KNOWN_ROLES:
            errors.append(prefix + ": invalid role")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(prefix + ": sha256 must be lowercase 64-hex")
        if not isinstance(rel, str):
            errors.append(prefix + ": missing relative_path")
        else:
            path = PurePosixPath(rel.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                errors.append(prefix + ": path must be relative and may not traverse upward")
        if isinstance(identity, str) and identity:
            if isinstance(role, str) and role.startswith("unknown_"):
                partition = allowed_unknown_roles.get(role)
                prior = unknown_partitions.setdefault(identity, partition)
                if prior != partition:
                    errors.append(prefix + ": unknown identity crosses train/calibration/development/final")
                if identity in known_seen:
                    errors.append(prefix + ": identity appears in both known and unknown partitions")
            elif role in {"train", "stopping", "threshold_calibration", "development", "final"}:
                known_seen.add(identity)
                if identity in unknown_partitions:
                    errors.append(prefix + ": identity appears in both known and unknown partitions")
    return {"schema": "dataset-role-manifest-check-1", "rows": len(rows),
            "errors": errors, "status": "pass" if not errors else "invalid"}


def assert_role_access(rows: Iterable[dict], action: str, *, frozen_receipt: dict | None = None) -> None:
    if action not in ROLE_ACTIONS:
        raise ValueError("unknown data action: " + action)
    rows = list(rows)
    invalid = [row.get("role") for row in rows if row.get("role") not in ROLE_ACTIONS[action]]
    if invalid:
        raise PermissionError("%s cannot access roles: %s" % (action, sorted(set(map(str, invalid)))))
    if action == "evaluate_final":
        if not frozen_receipt or frozen_receipt.get("status") != "frozen":
            raise PermissionError("final evaluation requires a frozen model/config/calibration receipt")
        if frozen_receipt.get("final_role_opened") is True:
            raise PermissionError("final role has already been opened; require a new protocol amendment")
        for key in ("model_sha256", "config_sha256", "calibration_sha256", "protocol_sha256"):
            if not isinstance(frozen_receipt.get(key), str) or not SHA256_RE.fullmatch(frozen_receipt[key]):
                raise PermissionError("freeze receipt missing valid " + key)
