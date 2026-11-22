"""Bounded, development-only J pilot registration, fitting, and selection.

The module consumes only explicit development encounter/cost summaries. It does
not connect to hardware, open dataset manifests, or inspect final-role inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .development_selector import fit_controller_calibration, read_jsonl, select_development_candidates
from .fs import atomic_write_json

HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PROTOCOL_SCHEMA = "J-pilot-selection-protocol-1"
PILOT_POLICY_LIMIT = 12
ENERGY_VALUE_MJ_PER_CORRECT = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def build_protocol(workspace: Path, runtime_config: Path | None = None) -> dict:
    workspace = workspace.resolve()
    study = workspace / "work/project-finish-v4"
    app = workspace / "FaceRecognition"
    plan_path = study / "config/J_CONTROLLER_PLAN.json"
    config_path = (runtime_config or app / "firmware/runtime_config.json").resolve()
    target_path = study / "config/ENGINEERING_TARGETS.json"
    selector_path = app / "nicla_study/development_selector.py"
    plan = _read_json(plan_path)
    runtime = _read_json(config_path)
    targets = _read_json(target_path)
    pilot = plan.get("pilot", {})
    policy_ids = list(plan.get("policy_ids", []))
    if not policy_ids or len(policy_ids) > PILOT_POLICY_LIMIT:
        raise ValueError("pilot must contain between one and twelve predeclared policy settings")
    cells = pilot.get("cells", [])
    blocks = pilot.get("blocks", [])
    conditions = sorted({row.get("condition") for row in blocks})
    if len(cells) != len(blocks) * len(policy_ids):
        raise ValueError("pilot schedule is not a complete policy-by-block design")
    if len({row.get("physical_block_seed") for row in blocks}) != len(blocks):
        raise ValueError("pilot physical block seeds must be unique")
    if not set(policy_ids).issubset(runtime.get("policies", {})):
        raise ValueError("runtime config is missing a preregistered policy")
    planned_policy_definitions = plan.get("policy_definitions", {})
    if any(runtime["policies"][policy_id] != planned_policy_definitions.get(policy_id)
           for policy_id in policy_ids):
        raise ValueError("runtime policy settings differ from the frozen controller plan")
    if plan.get("candidate_decision_rules", {}).get("tuning_budget", {}).get("candidate_settings_max") != PILOT_POLICY_LIMIT:
        raise ValueError("J plan and pilot protocol candidate limits differ")

    setting_rows = []
    for policy_id in policy_ids:
        settings = runtime["policies"][policy_id]
        setting_rows.append({
            "candidate_setting_id": policy_id,
            "policy_id": policy_id,
            "policy_definition": settings,
            "policy_definition_sha256": _canonical_hash(settings),
            "reason": "predeclared named baseline, mechanism ablation, or adaptive controller in the frozen J plan",
        })
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "status": "predeclared_source_protocol_no_outcomes",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace_relative_paths": {
            "controller_plan": "work/project-finish-v4/config/J_CONTROLLER_PLAN.json",
            "runtime_config": str(config_path.relative_to(workspace)) if config_path.is_relative_to(workspace) else None,
            "engineering_targets": "work/project-finish-v4/config/ENGINEERING_TARGETS.json",
            "selector_source": "FaceRecognition/nicla_study/development_selector.py",
        },
        "source_sha256": {
            "controller_plan": sha256_file(plan_path),
            "runtime_config": sha256_file(config_path),
            "engineering_targets": sha256_file(target_path),
            "selector_source": sha256_file(selector_path),
            "pilot_workflow_source": sha256_file(Path(__file__).resolve()),
        },
        "scope": "J development pilot and pre-final selection only; final data are not opened or read",
        "candidate_budget": {
            "maximum_candidate_settings": PILOT_POLICY_LIMIT,
            "predeclared_candidate_settings": len(setting_rows),
            "settings": setting_rows,
            "unallocated_slots_require_versioned_protocol_amendment_before_data_collection": True,
            "candidate_outcomes_may_not_be_added_after_pilot_results_are_seen": True,
        },
        "pilot_schedule": {
            "plan_phase": "JPILOT",
            "independently_started_paired_blocks": len(blocks),
            "conditions": conditions,
            "policy_settings_per_block": len(policy_ids),
            "planned_capture_count": len(cells),
            "training_roots": sorted({row.get("training_root_seed") for row in blocks}),
            "randomized_block_and_policy_order": True,
            "physical_block_seed_is_distinct_from_training_root": True,
            "all_predeclared_policy_attempts_required_for_selection": True,
        },
        "cost_calibration": {
            "source": "I development-only measured incremental small-to-large cost receipt",
            "required_fields": ["status", "source_role", "incremental_energy_mj", "incremental_latency_ms"],
            "energy_value_mj_per_correct": ENERGY_VALUE_MJ_PER_CORRECT,
            "energy_value_source": "fixed prospective engineering exchange rate; not an observed measurement",
            "risk_threshold": 0.1,
            "fit_score": "small-model candidate cosine score, in [0,1]",
            "fit_bin_width": 0.1,
            "minimum_encounters_per_bin": 30,
            "minimum_session_identity_clusters_per_bin": 10,
            "training_row_unit": "one independently annotated encounter, never adjacent frames",
        },
        "candidate_selection": {
            "metrics": ["known_macro_top1", "known_macro_accepted_coverage", "response_p95_ms",
                        "energy_per_accepted_correct_known_vs_fixed_high", "unknown false-identification counts by identity"],
            "minimum_independent_paired_blocks": 10,
            "thresholds": targets,
            "unknown_bound": "maximum of identity-cluster bootstrap and exact one-sided encounter Clopper-Pearson bound",
            "rule": "filter every candidate through every quality, unknown, response, measurement, role and hash gate; select lowest energy per accepted correct known encounter; tie-break by known macro top-1 then policy ID",
            "reject_all_is_not_success": True,
        },
        "attempt_log": {
            "path": "work/project-finish-v4/runs/JPILOT/ATTEMPTS.jsonl",
            "registration": "register all ten policy settings before physical capture; append-only",
            "required_fields": ["attempt_id", "candidate_setting_id", "policy_id", "registered_utc",
                                "protocol_sha256", "policy_definition_sha256", "registered_before_capture"],
        },
        "sequence": [
            "Generate the protocol against the fitted runtime config before the pilot.",
            "Register every preregistered policy setting in the append-only attempt log.",
            "Run the JPILOT schedule only after I cost and exact-current-firmware H gates pass.",
            "Prepare one development-only encounter row per annotated encounter and a complete candidate result row per policy.",
            "Fit calibration from development encounters and I's measured cost; stage a new runtime config and rerun affected H checks.",
            "Score candidates only after all preregistered policy rows are present; then use the separate one-way freeze tool before K.",
        ],
        "physical_outcomes": "pending; this source protocol contains no measurements or selected candidate",
    }
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    return protocol


def verify_protocol(protocol: dict) -> str:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "predeclared_source_protocol_no_outcomes":
        raise ValueError("unsupported or already scored J pilot protocol")
    expected = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    actual = _canonical_hash(unsigned)
    if expected != actual:
        raise ValueError("pilot protocol hash mismatch")
    rows = protocol.get("candidate_budget", {}).get("settings", [])
    ids = [row.get("candidate_setting_id") for row in rows]
    if len(rows) > PILOT_POLICY_LIMIT or len(ids) != len(set(ids)):
        raise ValueError("candidate settings exceed the frozen budget or contain duplicates")
    return actual


def register_all(protocol_path: Path, attempt_log: Path) -> dict:
    protocol = _read_json(protocol_path)
    protocol_hash = verify_protocol(protocol)
    settings = protocol["candidate_budget"]["settings"]
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(attempt_log) if attempt_log.exists() else []
    expected_ids = {row["candidate_setting_id"] for row in settings}
    existing_ids = [row.get("candidate_setting_id") for row in existing]
    if len(existing_ids) != len(set(existing_ids)) or not set(existing_ids).issubset(expected_ids):
        raise ValueError("attempt log contains duplicate or unregistered settings")
    missing = [row for row in settings if row["candidate_setting_id"] not in set(existing_ids)]
    with attempt_log.open("a", encoding="utf-8", newline="\n") as stream:
        for setting in missing:
            attempt_id = "JPILOT-" + setting["candidate_setting_id"]
            event = {"schema": "J-pilot-attempt-1", "attempt_id": attempt_id,
                "candidate_setting_id": setting["candidate_setting_id"], "policy_id": setting["policy_id"],
                "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_sha256": protocol_hash,
                "policy_definition_sha256": setting["policy_definition_sha256"],
                "registered_before_capture": True, "data_role": "development_only",
                "reason": setting["reason"], "status": "preregistered_not_yet_run"}
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
        import os
        os.fsync(stream.fileno())
    return {"schema": "J-pilot-attempt-registration-1", "status": "pass_preregistered",
            "protocol_sha256": protocol_hash, "registered_candidate_settings": len(settings),
            "attempt_log_sha256": sha256_file(attempt_log), "attempt_log": str(attempt_log.resolve())}


def validate_attempt_log(protocol: dict, attempts: list[dict]) -> dict:
    protocol_hash = verify_protocol(protocol)
    settings = protocol["candidate_budget"]["settings"]
    expected = {row["candidate_setting_id"]: row for row in settings}
    seen = set()
    errors = []
    for row in attempts:
        setting_id = row.get("candidate_setting_id")
        if setting_id not in expected or setting_id in seen:
            errors.append("attempt has unknown or duplicate candidate_setting_id: " + str(setting_id))
            continue
        seen.add(setting_id)
        if row.get("protocol_sha256") != protocol_hash:
            errors.append("attempt protocol hash mismatch: " + str(setting_id))
        if row.get("policy_definition_sha256") != expected[setting_id]["policy_definition_sha256"]:
            errors.append("attempt policy definition hash mismatch: " + str(setting_id))
        if row.get("registered_before_capture") is not True or row.get("data_role") != "development_only":
            errors.append("attempt was not pre-registered as development-only: " + str(setting_id))
    if seen != set(expected):
        errors.append("attempt log does not preregister every frozen policy setting")
    return {"status": "pass" if not errors else "failed", "candidate_setting_count": len(expected),
            "registered_count": len(seen), "errors": errors, "protocol_sha256": protocol_hash}


def validate_physical_pilot_inputs(protocol_path: Path, attempt_log: Path,
                                   runtime_config_path: Path, controller_plan_path: Path) -> dict:
    """Fail closed before a JPILOT capture if preregistration or calibration is incomplete."""
    protocol = _read_json(protocol_path)
    protocol_hash = verify_protocol(protocol)
    if protocol["source_sha256"]["runtime_config"] != sha256_file(runtime_config_path):
        raise ValueError("JPILOT runtime config differs from the pre-registered protocol")
    if protocol["source_sha256"]["controller_plan"] != sha256_file(controller_plan_path):
        raise ValueError("JPILOT plan differs from the pre-registered protocol")
    attempts = read_jsonl(attempt_log)
    attempt_check = validate_attempt_log(protocol, attempts)
    if attempt_check["status"] != "pass":
        raise ValueError("JPILOT attempt log failed validation: " + "; ".join(attempt_check["errors"]))
    config = _read_json(runtime_config_path)
    calibration = config.get("controller_calibration", {})
    if (calibration.get("status") not in {"calibrated_development_pending_H_requalification", "calibrated"} or
            calibration.get("source_role") != "development_only" or
            calibration.get("used_final_data") is not False or not calibration.get("risk_bins") or
            not isinstance(calibration.get("energy_value_mj_per_correct"), (int, float)) or
            calibration.get("energy_value_mj_per_correct", 0) <= 0):
        raise ValueError("JPILOT requires development-fitted risk bins and measured-cost calibration in runtime config")
    return {"status": "pass_preflight_only", "protocol_sha256": protocol_hash,
            "attempt_registry_check": attempt_check, "runtime_config_sha256": sha256_file(runtime_config_path),
            "controller_plan_sha256": sha256_file(controller_plan_path),
            "physical_measurement_performed": False}


def score_candidates(protocol_path: Path, attempt_log: Path, candidate_results: Path,
                     output: Path) -> dict:
    protocol = _read_json(protocol_path)
    attempts = read_jsonl(attempt_log)
    attempt_check = validate_attempt_log(protocol, attempts)
    if attempt_check["status"] != "pass":
        raise ValueError("pilot attempt registry failed: " + "; ".join(attempt_check["errors"]))
    result_rows = read_jsonl(candidate_results)
    settings = {row["candidate_setting_id"]: row for row in protocol["candidate_budget"]["settings"]}
    attempt_by_policy = {row["policy_id"]: row for row in attempts}
    policy_ids = set(settings)
    result_by_policy = {}
    for row in result_rows:
        policy_id = row.get("policy_id")
        if policy_id not in policy_ids or policy_id in result_by_policy:
            raise ValueError("candidate results contain unknown or duplicate policy_id: " + str(policy_id))
        if row.get("candidate_setting_id") != policy_id:
            raise ValueError("candidate result policy/setting IDs differ: " + str(policy_id))
        if row.get("attempt_id") != attempt_by_policy[policy_id].get("attempt_id"):
            raise ValueError("candidate result is not linked to its preregistered attempt: " + str(policy_id))
        if row.get("data_role") != "development" or row.get("used_final_data") is not False:
            raise PermissionError("selector accepts only explicitly development-only candidate results")
        if row.get("runtime_config_sha256") != protocol["source_sha256"]["runtime_config"]:
            raise ValueError("candidate runtime-config hash differs from the preregistered configuration")
        if row.get("controller_sha256") != protocol["source_sha256"]["controller_plan"]:
            raise ValueError("candidate controller-plan hash differs from the preregistered plan")
        result_by_policy[policy_id] = dict(row, controller_setting=policy_id)
    if set(result_by_policy) != policy_ids:
        raise ValueError("all frozen policy settings need a result row, including failed attempts")
    selection = select_development_candidates(list(result_by_policy.values()))
    selection.update({"pilot_protocol_sha256": verify_protocol(protocol),
        "attempt_log_sha256": sha256_file(attempt_log),
        "candidate_results_sha256": sha256_file(candidate_results),
        "attempt_registry_check": attempt_check,
        "selection_evidence_tier": "development_only; physical metrics must carry their own verified capture/QC receipts"})
    atomic_write_json(output, selection)
    return selection


def fit_calibration_cli(encounters_path: Path, measured_cost_path: Path,
                        protocol_path: Path, energy_value: float | None, output: Path) -> dict:
    protocol = _read_json(protocol_path)
    protocol_hash = verify_protocol(protocol)
    frozen_value = float(protocol["cost_calibration"]["energy_value_mj_per_correct"])
    if energy_value is not None and float(energy_value) != frozen_value:
        raise ValueError("energy value differs from the preregistered pilot protocol")
    encounter_rows = read_jsonl(encounters_path)
    cost = _read_json(measured_cost_path)
    if cost.get("evidence_tier") not in ("measured_physical_development", "qualified_physical_measurement"):
        raise ValueError("measured-cost receipt must identify qualified physical development evidence")
    fitted = fit_controller_calibration(encounter_rows, cost, frozen_value)
    fitted["encounter_rows_sha256"] = sha256_file(encounters_path)
    fitted["measured_cost_receipt_sha256"] = sha256_file(measured_cost_path)
    fitted["pilot_protocol_sha256"] = protocol_hash
    fitted["energy_value_source"] = "fixed prospective engineering exchange rate; not measured"
    fitted["calibration_sha256"] = _canonical_hash(fitted)
    atomic_write_json(output, fitted)
    return fitted


def apply_calibration(base_config_path: Path, fit_path: Path, output_path: Path) -> dict:
    base_config_path = base_config_path.resolve()
    output_path = output_path.resolve()
    if output_path == base_config_path:
        raise ValueError("write a staged versioned runtime config; do not overwrite the F-owned base config")
    config = _read_json(base_config_path)
    fitted = _read_json(fit_path)
    if (fitted.get("status") != "ready_for_hash_bound_runtime_update" or
            fitted.get("source_role") != "development_only" or fitted.get("used_final_data") is not False):
        raise ValueError("calibration fit is not a complete development-only fit")
    if not fitted.get("risk_bins"):
        raise ValueError("no calibrated risk bins; the adaptive risk/cost routes must remain fail-closed")
    value = fitted.get("energy_value_mj_per_correct")
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("fit is missing the predeclared positive energy value")
    config["controller_calibration"] = {
        "status": "calibrated_development_pending_H_requalification",
        "source_role": "development_only", "used_final_data": False,
        "risk_threshold": fitted["risk_threshold"],
        "energy_value_mj_per_correct": value,
        "risk_bins": fitted["risk_bins"],
        "calibration_source_sha256": fitted["calibration_sha256"],
        "measured_cost_source": fitted["measured_cost_source"],
    }
    atomic_write_json(output_path, config)
    return {"schema": "J-staged-runtime-config-1", "status": "staged_requires_exact_H_requalification",
        "base_runtime_config_sha256": sha256_file(base_config_path),
        "calibration_sha256": fitted["calibration_sha256"],
        "staged_runtime_config_sha256": sha256_file(output_path),
        "staged_runtime_config": str(output_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    prepare.add_argument("--runtime-config", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    register = sub.add_parser("register-all")
    register.add_argument("--protocol", type=Path, required=True)
    register.add_argument("--attempt-log", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--attempt-log", type=Path, required=True)
    score.add_argument("--candidate-results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("fit-calibration")
    fit.add_argument("--encounters", type=Path, required=True)
    fit.add_argument("--measured-cost", type=Path, required=True)
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--energy-value-mj-per-correct", type=float,
                     help="optional assertion; must match the preregistered value")
    fit.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("stage-calibration")
    apply.add_argument("--base-runtime-config", type=Path, required=True)
    apply.add_argument("--fit", type=Path, required=True)
    apply.add_argument("--output-runtime-config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = build_protocol(args.workspace, args.runtime_config)
        atomic_write_json(args.output, result)
    elif args.action == "register-all":
        result = register_all(args.protocol, args.attempt_log)
    elif args.action == "score":
        result = score_candidates(args.protocol, args.attempt_log, args.candidate_results, args.output)
    elif args.action == "fit-calibration":
        result = fit_calibration_cli(args.encounters, args.measured_cost, args.protocol,
                                     args.energy_value_mj_per_correct, args.output)
    else:
        result = apply_calibration(args.base_runtime_config, args.fit, args.output_runtime_config)
    print(json.dumps(result, indent=2))
    status = result.get("status", "")
    return 0 if status in {"pass_preregistered", "selected_development_candidate", "no_qualified_candidate",
                           "ready_for_hash_bound_runtime_update", "insufficient_development_encounters",
                           "staged_requires_exact_H_requalification", "predeclared_source_protocol_no_outcomes"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
