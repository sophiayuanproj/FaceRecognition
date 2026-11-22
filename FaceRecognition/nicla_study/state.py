"""Transactional shared study state, append-only event history and resumable phases."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import os
import uuid
from typing import Any, Callable

from .fs import FileLock, ResourceLease, atomic_write_bytes, atomic_write_json

TERMINAL_SUCCESS = {"complete_verified"}
PREREQUISITE_SUCCESS = {"complete_verified"}
PHASE_STATUSES = {"pending", "ready", "in_progress", "interrupted", "partial_external_dependency",
                  "complete_verified", "failed_needs_repair", "unsupported"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_paths(study_root: Path | str) -> tuple[Path, Path, Path]:
    root = Path(study_root)
    return root / "RUN_STATE.json", root / ".state.lock", root / "RUN_EVENTS.jsonl"


def initial_state(phase_plan: dict, experiment_plan_path: str, protocol_id: str) -> dict:
    phases = {}
    jobs = {}
    for row in phase_plan["phases"]:
        phase_id = row["id"]
        phases[phase_id] = {
            "status": "pending", "host": row["host"], "job": row["job"],
            "owner": row.get("owner", row["host"]), "requires": row.get("requires", []),
            "requires_any": row.get("requires_any", []), "resources": row.get("resources", []),
            "receipt": None, "updated_utc": utc_now(),
        }
        jobs.setdefault(row["job"], {"status": "pending",
                                      "subphases": {"preparation": "pending", "capture": "pending",
                                                    "pilot": "pending", "analysis": "pending"}})
    jobs.setdefault("A", {"status": "pending", "subphases": {}})
    return {
        "schema": "nicla-study-run-state-1", "protocol_id": protocol_id,
        "created_utc": utc_now(), "updated_utc": utc_now(),
        "shared_state_writer": "mac_coordinator",
        "active_job": "A", "active_host": "mac",
        "jobs": jobs, "phases": phases,
        "leases": {"heavy_training_capacity": 1, "active": []},
        "devices": {"last_fresh_discovery_utc": None, "nicla": "not_observed",
                    "fnirsi": "not_observed", "nano": "not_observed", "usb_uart": "not_observed"},
        "experiment_plan": experiment_plan_path, "pending_external_actions": [],
        "history": [],
        "claim_evidence": {"schema": "claim-evidence-ledger-1", "claims": [], "history": []},
    }


def render_current_status(state: dict) -> bytes:
    complete = [name for name, row in state.get("phases", {}).items()
                if row.get("status") == "complete_verified"]
    pending = [name for name, row in state.get("phases", {}).items()
               if row.get("status") not in {"complete_verified", "unsupported"}]
    lines = [
        "# Current study status",
        "",
        "**Protocol:** %s" % state.get("protocol_id", "uninitialized"),
        "**Active job:** %s" % (state.get("active_job") or "none"),
        "**Active host:** %s" % state.get("active_host", "unassigned"),
        "**Updated:** %s" % state.get("updated_utc", utc_now()),
        "",
        "## Completed phases",
        "",
        ", ".join(complete) if complete else "None verified yet.",
        "",
        "## Pending phases",
        "",
        ", ".join(pending[:12]) + (" …" if len(pending) > 12 else "") if pending else "None.",
        "",
        "## Device availability",
        "",
        "Fresh discovery: %s. Nicla: %s. FNIRSI: %s. Nano: %s." % (
            state.get("devices", {}).get("last_fresh_discovery_utc", "not run"),
            state.get("devices", {}).get("nicla", "not observed"),
            state.get("devices", {}).get("fnirsi", "not observed"),
            state.get("devices", {}).get("nano", "not observed")),
        "",
        "## Next action",
        "",
        state.get("next_action", "Run the next phase only after its declared prerequisites and owner are verified."),
        "",
        "Machine state: RUN_STATE.json. Event history: RUN_EVENTS.jsonl. Claim evidence: CLAIM_EVIDENCE_LEDGER.json.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def _write_shared_files(study_root: Path, state: dict) -> None:
    state_file, _, _ = state_paths(study_root)
    atomic_write_json(state_file, state)
    ledger = dict(state.get("claim_evidence", {"schema":"claim-evidence-ledger-1","claims":[],"history":[]}))
    ledger["protocol_id"] = state.get("protocol_id")
    ledger["updated_utc"] = state.get("updated_utc")
    atomic_write_json(Path(study_root) / "CLAIM_EVIDENCE_LEDGER.json", ledger)
    atomic_write_bytes(Path(study_root) / "CURRENT_STATUS.md", render_current_status(state))


def record_claim_evidence(study_root: Path | str, claim_id: str, description: str,
                          status: str, evidence_paths: list[str], *, evidence_tier: str,
                          supported_claim: str | None = None) -> dict:
    allowed = {"pending", "software_verified", "host_development_only",
               "hardware_pending", "qualified", "unsupported"}
    if status not in allowed:
        raise ValueError("invalid claim evidence status")
    def update(state):
        ledger = state.setdefault("claim_evidence", {"schema":"claim-evidence-ledger-1","claims":[],"history":[]})
        claims = {row["claim_id"]: row for row in ledger["claims"]}
        old = claims.get(claim_id)
        revision = 1 if old is None else int(old.get("revision", 1)) + 1
        row = {"claim_id":claim_id,"description":description,"status":status,
               "evidence_tier":evidence_tier,"evidence_paths":list(evidence_paths),
               "supported_claim":supported_claim,"revision":revision,"updated_utc":utc_now()}
        claims[claim_id] = row
        ledger["claims"] = list(claims.values())
        ledger.setdefault("history",[]).append({"claim_id":claim_id,"revision":revision,
                                                 "status":status,"updated_utc":row["updated_utc"],
                                                 "evidence_paths":list(evidence_paths)})
        return row
    state, row = transact(study_root, update, event_type="claim_evidence_updated",
                          event_details={"claim_id":claim_id,"status":status})
    return row


def _event(events_file: Path, event: dict) -> None:
    events_file.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("event_utc", utc_now())
    with events_file.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def transact(study_root: Path | str, mutator: Callable[[dict], Any], *, event_type: str,
             event_details: dict | None = None) -> tuple[dict, Any]:
    state_file, lock_file, events_file = state_paths(study_root)
    Path(study_root).mkdir(parents=True, exist_ok=True)
    with FileLock(lock_file, timeout_s=30):
        if not state_file.is_file():
            raise FileNotFoundError("RUN_STATE.json is not initialized")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        result = mutator(state)
        state["updated_utc"] = utc_now()
        state["history"].append({"event_type": event_type, "event_utc": state["updated_utc"],
                                 "details": event_details or {}})
        _write_shared_files(Path(study_root), state)
        _event(events_file, {"event_type": event_type, "details": event_details or {}})
        return state, result


@contextmanager
def resource_lease(study_root: Path | str, resource: str, *, owner: str,
                   timeout_s: float = 0.0):
    """Hold a cross-process resource lock and reflect ownership in RUN_STATE."""
    lease_id = uuid.uuid4().hex
    with ResourceLease(study_root, resource, timeout_s=timeout_s):
        record = {"lease_id": lease_id, "resource": resource, "owner": owner,
                  "pid": os.getpid(), "acquired_utc": utc_now()}
        def add(state):
            active = state.setdefault("leases", {}).setdefault("active", [])
            if any(row.get("resource") == resource for row in active):
                raise RuntimeError("resource already appears active in shared state: " + resource)
            active.append(record)
            return record
        transact(study_root, add, event_type="resource_lease_acquired",
                 event_details={"resource": resource, "owner": owner, "lease_id": lease_id})
        try:
            yield record
        finally:
            def remove(state):
                active = state.setdefault("leases", {}).setdefault("active", [])
                state["leases"]["active"] = [row for row in active if row.get("lease_id") != lease_id]
            transact(study_root, remove, event_type="resource_lease_released",
                     event_details={"resource": resource, "owner": owner, "lease_id": lease_id})


def reconcile_stale_resource_leases(study_root: Path | str) -> list[dict]:
    """Remove state rows whose OS lock is no longer held by any process."""
    state_file = state_paths(study_root)[0]
    if not state_file.is_file():
        return []
    snapshot = json.loads(state_file.read_text(encoding="utf-8"))
    stale = []
    for row in snapshot.get("leases", {}).get("active", []):
        probe = ResourceLease(study_root, row["resource"], timeout_s=0)
        try:
            probe.acquire()
        except TimeoutError:
            continue
        else:
            probe.release()
            stale.append(row)
    if not stale:
        return []
    stale_ids = {row.get("lease_id") for row in stale}
    def remove(state):
        active = state.setdefault("leases", {}).setdefault("active", [])
        state["leases"]["active"] = [row for row in active if row.get("lease_id") not in stale_ids]
    transact(study_root, remove, event_type="stale_resource_leases_reconciled",
             event_details={"lease_ids": sorted(str(value) for value in stale_ids)})
    return stale


def initialize_state(study_root: Path | str, phase_plan: dict, experiment_plan_path: str,
                     protocol_id: str = "nicla-five-known-study-v4") -> dict:
    state_file, lock_file, _ = state_paths(study_root)
    Path(study_root).mkdir(parents=True, exist_ok=True)
    with FileLock(lock_file, timeout_s=30):
        if state_file.exists():
            return json.loads(state_file.read_text(encoding="utf-8"))
        state = initial_state(phase_plan, experiment_plan_path, protocol_id)
        _write_shared_files(Path(study_root), state)
        _event(state_paths(study_root)[2], {"event_type": "state_initialized",
                                            "protocol_id": protocol_id})
        return state


def phase_prerequisites_complete(state: dict, phase: dict) -> bool:
    phases = state.get("phases", {})
    for prerequisite in phase.get("requires", []):
        if phases.get(prerequisite, {}).get("status") not in PREREQUISITE_SUCCESS:
            return False
    alternatives = phase.get("requires_any", [])
    for group in alternatives:
        if not any(phases.get(choice, {}).get("status") in PREREQUISITE_SUCCESS for choice in group):
            return False
    return True


def start_phase(study_root: Path | str, phase_id: str) -> dict:
    def update(state):
        phase = state["phases"].get(phase_id)
        if phase is None:
            raise KeyError("unknown phase " + phase_id)
        if phase["status"] not in {"ready", "pending", "interrupted"}:
            raise RuntimeError("phase cannot start from status " + phase["status"])
        if not phase_prerequisites_complete(state, phase):
            raise RuntimeError("phase prerequisites are not complete")
        phase["status"] = "in_progress"
        phase["started_utc"] = utc_now()
        phase["updated_utc"] = phase["started_utc"]
        phase["receipt"] = None
        state["active_job"] = phase.get("job")
        state["active_host"] = phase.get("host")
        state["next_action"] = "Complete %s only after its acceptance receipt passes." % phase_id
        return phase
    state, _ = transact(study_root, update, event_type="phase_started", event_details={"phase_id": phase_id})
    return state["phases"][phase_id]


def complete_phase_from_receipt(study_root: Path | str, phase_id: str, receipt_path: Path | str) -> dict:
    receipt_file = Path(receipt_path).resolve()
    if not receipt_file.is_file():
        raise FileNotFoundError(str(receipt_file))
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete_verified":
        raise RuntimeError("only a complete_verified receipt can complete a phase")
    if receipt.get("acceptance_checks") is None:
        raise RuntimeError("receipt must contain acceptance_checks")
    failed = [check for check in receipt["acceptance_checks"] if check.get("status") != "pass"]
    if failed:
        raise RuntimeError("receipt has failing or pending acceptance checks")
    def update(state):
        phase = state["phases"].get(phase_id)
        if phase is None:
            raise KeyError("unknown phase " + phase_id)
        phase["status"] = "complete_verified"
        phase["updated_utc"] = utc_now()
        phase["receipt"] = str(receipt_file)
        state["next_action"] = "Review the next declared phase and start it only after its prerequisites pass."
        return phase
    state, _ = transact(study_root, update, event_type="phase_completed",
                        event_details={"phase_id": phase_id, "receipt": str(receipt_file)})
    return state["phases"][phase_id]


def resume_phase(study_root: Path | str, phase_id: str) -> dict:
    def update(state):
        phase = state["phases"].get(phase_id)
        if phase is None:
            raise KeyError("unknown phase " + phase_id)
        if not phase_prerequisites_complete(state, phase):
            raise RuntimeError("phase prerequisites are not complete")
        if phase["status"] not in {"interrupted", "failed_needs_repair", "pending", "ready"}:
            raise RuntimeError("phase cannot resume from status " + phase["status"])
        phase["status"] = "ready"
        phase["resumed_utc"] = utc_now()
        phase["updated_utc"] = phase["resumed_utc"]
        return phase
    state, _ = transact(study_root, update, event_type="phase_resumable",
                        event_details={"phase_id": phase_id})
    return state["phases"][phase_id]


def mark_stale_running_phases_interrupted(study_root: Path | str) -> list[str]:
    def update(state):
        changed = []
        for phase_id, phase in state["phases"].items():
            if phase.get("status") == "in_progress":
                phase["status"] = "interrupted"
                phase["interrupted_utc"] = utc_now()
                phase["updated_utc"] = phase["interrupted_utc"]
                changed.append(phase_id)
        return changed
    state, changed = transact(study_root, update, event_type="resume_scan",
                              event_details={"action": "in_progress_to_interrupted"})
    return changed


def dispatch_analysis(study_root: Path | str, job: str, adapters: dict[str, dict] | None = None) -> dict:
    adapter = (adapters or {}).get(job)
    if not adapter or adapter.get("status") != "implemented":
        return {"job": job, "action_status": "pending_not_implemented",
                "status": "pending", "exit_code": 3,
                "reason": "no implemented analysis adapter is registered for this job"}
    return {"job": job, "action_status": "implemented", "status": "pending",
            "exit_code": 3, "reason": "adapter exists but has no verified source receipt"}
