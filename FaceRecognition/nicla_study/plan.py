"""Validation for the shared Mac/Windows phase DAG and finite study matrix."""
from __future__ import annotations

from collections import defaultdict, deque


def validate_phase_plan(plan: dict) -> dict:
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        return {"status": "invalid", "errors": ["phases must be a nonempty list"], "phase_count": 0}
    errors = []
    by_id = {}
    for index, phase in enumerate(phases):
        phase_id = phase.get("id")
        if not isinstance(phase_id, str) or not phase_id:
            errors.append("phase[%d] has no id" % index)
            continue
        if phase_id in by_id:
            errors.append("duplicate phase id: " + phase_id)
        by_id[phase_id] = phase
        if not phase.get("host") or not phase.get("job"):
            errors.append(phase_id + " requires host and job")
        if not isinstance(phase.get("requires", []), list):
            errors.append(phase_id + " requires must be a list")
        if not isinstance(phase.get("resources", []), list):
            errors.append(phase_id + " resources must be a list")
    graph = defaultdict(list)
    indegree = {phase_id: 0 for phase_id in by_id}
    for phase_id, phase in by_id.items():
        for prerequisite in phase.get("requires", []):
            if prerequisite not in by_id:
                errors.append("%s references missing prerequisite %s" % (phase_id, prerequisite))
            else:
                graph[prerequisite].append(phase_id)
                indegree[phase_id] += 1
        for group in phase.get("requires_any", []):
            if not isinstance(group, list) or not group:
                errors.append(phase_id + " has an empty requires_any group")
            elif any(choice not in by_id for choice in group):
                errors.append(phase_id + " requires_any references missing phase")
    ready = deque([phase_id for phase_id, degree in indegree.items() if degree == 0])
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for child in graph[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(by_id):
        errors.append("phase dependency graph contains a cycle")
    owners = {}
    for phase in phases:
        host_owner = phase.get("owner", phase.get("host"))
        owners[phase["id"]] = host_owner
    return {"schema": "phase-plan-validation-1", "phase_count": len(phases),
            "owners": owners, "errors": errors, "status": "pass" if not errors else "invalid"}


def validate_experiment_plan(plan: dict) -> dict:
    rows = plan.get("experiments")
    errors = []
    if not isinstance(rows, list) or not rows:
        return {"status": "invalid", "experiment_count": 0, "errors": ["experiments must be nonempty"]}
    ids = set()
    for index, row in enumerate(rows):
        experiment_id = row.get("id")
        if not experiment_id or experiment_id in ids:
            errors.append("experiment[%d] missing or duplicate id" % index)
        ids.add(experiment_id)
        if not row.get("owner") or not row.get("phase"):
            errors.append(str(experiment_id) + " must name an owner and phase")
        if row.get("status") not in {"pending", "prepared", "running", "complete_verified",
                                     "partial_external_dependency", "failed_needs_repair",
                                     "unsupported"}:
            errors.append(str(experiment_id) + " has invalid status")
        if not row.get("action"):
            errors.append(str(experiment_id) + " lacks an executable action key")
        if not isinstance(row.get("prerequisites", []), list):
            errors.append(str(experiment_id) + " prerequisites must be a list")
    return {"schema": "finite-experiment-plan-validation-1", "experiment_count": len(rows),
            "errors": errors, "status": "pass" if not errors else "invalid"}
