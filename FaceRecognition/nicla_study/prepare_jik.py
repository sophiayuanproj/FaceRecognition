"""Build prospective, resumable I/J/K plans without opening final-role data."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from .fs import atomic_write_json, sha256_file

WORKSPACE = Path(__file__).resolve().parents[2]
STUDY = WORKSPACE / "work/project-finish-v4"
APP = WORKSPACE / "FaceRecognition"
PLAN_SEED = 20261401
POLICIES = ("H", "L", "T", "G", "A-sense", "A-model", "A-risk", "A-cost", "A-full", "U")
CONDITIONS = (
    "known_frontal_stationary_normal",
    "known_offcenter_range",
    "known_motion_blur",
    "known_low_light",
    "unknown_entry_stationary",
    "empty_periodic_check",
    "ambiguous_multiple_faces",
)
FORMATS = (
    "float32",
    "fp16_weights_float_io",
    "dynamic_range_int8_weights_float_io",
    "full_int8_ptq_train_only_rep",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path_text: str) -> str:
    path = Path(path_text)
    try:
        return path.resolve().relative_to(WORKSPACE).as_posix()
    except (OSError, ValueError):
        return "external_unavailable:" + path.name


def _hash(path: Path) -> str:
    return sha256_file(path)


def _format_alias(name: str) -> str:
    return "full_int8" if name == "full_int8_ptq_train_only_rep" else name


def build_model_matrix() -> dict:
    artifact_path = STUDY / "handoffs/E/FHI_ARTIFACT_ROOTS.json"
    h_path = STUDY / "reports/H/nicla_selected_artifact_matrix.jsonl"
    e = _json(artifact_path)
    h_rows = [json.loads(line) for line in h_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    h_by_hash = {str(row.get("artifact_sha256", "")).lower(): row for row in h_rows}
    e_rows = []
    for root in e.get("all_capacity_and_optimization_roots", []):
        arm = root.get("arm_id")
        seed = int(root.get("seed"))
        for export_format in FORMATS:
            item = root.get("formats", {}).get(export_format)
            if not isinstance(item, dict):
                e_rows.append({"arm_id": arm, "training_root_seed": seed,
                    "export_format": export_format, "artifact_status": "missing_from_E_manifest",
                    "hardware_probe_status": "not_probed"})
                continue
            artifact = item.get("artifact") or {}
            digest = str(artifact.get("sha256", "")).lower()
            probe = h_by_hash.get(digest)
            e_rows.append({
                "family": root.get("family"),
                "arm_id": arm,
                "training_root_seed": seed,
                "source_parent_root": root.get("source_parent_root"),
                "source_parent_weights_sha256": root.get("source_parent_weights_sha256"),
                "root_config_sha256": root.get("config", {}).get("sha256"),
                "training_status": root.get("status"),
                "training_seconds": root.get("training_seconds"),
                "export_format": export_format,
                "artifact_status": item.get("status"),
                "artifact_bytes": artifact.get("bytes"),
                "artifact_sha256": digest or None,
                "artifact_path_relative": _relative(artifact.get("path", "")) if artifact.get("path") else None,
                "host_inference_status": item.get("host_inference_status"),
                "development_metrics": item.get("development"),
                "hardware_probe_status": probe.get("probe_status") if probe else "not_probed",
                "hardware_probe_error_type": probe.get("error_type") if probe else None,
                "hardware_readback_matches": probe.get("readback_matches_source") if probe else None,
                "hardware_probe_source": "H report by exact artifact SHA-256" if probe else None,
                "full_pipeline_eligible": False,
                "eligibility_note": "A passing model-only probe is not a live-camera or full-pipeline qualification.",
            })
    probe_counts = {}
    for row in e_rows:
        key = row["hardware_probe_status"]
        probe_counts[key] = probe_counts.get(key, 0) + 1
    arm_rows = e.get("all_capacity_and_optimization_roots", [])
    by_arm = {}
    for root in arm_rows:
        by_arm.setdefault(str(root.get("arm_id")), set()).add(int(root.get("seed")))
    arms = sorted(by_arm)
    variants_by_key = {(str(row.get("arm_id")), int(row.get("training_root_seed")),
                        str(row.get("export_format"))): row for row in e_rows
                       if row.get("training_root_seed") is not None}
    comparisons = []
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1:]:
            matched_roots = sorted(by_arm[left].intersection(by_arm[right]))
            for export_format in FORMATS:
                if len(matched_roots) < 10:
                    continue
                comparison_id = "%s-vs-%s-%s" % (left, right, export_format)
                comparisons.append({"comparison_id": comparison_id,
                    "arm_a": left, "arm_b": right, "export_format": export_format,
                    "matched_training_root_seeds": matched_roots,
                    "paired_artifacts": [{
                        "training_root_seed": seed,
                        "arm_a_artifact_sha256": variants_by_key.get((left, seed, export_format), {}).get("artifact_sha256"),
                        "arm_a_artifact_status": variants_by_key.get((left, seed, export_format), {}).get("artifact_status"),
                        "arm_a_hardware_probe_status": variants_by_key.get((left, seed, export_format), {}).get("hardware_probe_status"),
                        "arm_b_artifact_sha256": variants_by_key.get((right, seed, export_format), {}).get("artifact_sha256"),
                        "arm_b_artifact_status": variants_by_key.get((right, seed, export_format), {}).get("artifact_status"),
                        "arm_b_hardware_probe_status": variants_by_key.get((right, seed, export_format), {}).get("hardware_probe_status"),
                    } for seed in matched_roots],
                    "minimum_independently_started_paired_blocks": 10,
                    "blocks_per_root": 1,
                    "randomized_order_seed_derivation": "SHA-256(comparison_id|training_root_seed|block_index), first 32 bits",
                    "execution_status": "conditional_pending_exact_H_and_live_camera_eligibility",
                    "individual_artifacts_preserved_in_matrix": True})
    pairwise_capture_count = sum(row["minimum_independently_started_paired_blocks"] * 2 for row in comparisons)
    return {
        "schema": "I-full-model-compression-lifecycle-matrix-1",
        "status": "prospective_matrix_preserves_all_E_roots_and_H_outcomes",
        "created_utc": "2026-10-30",
        "source_artifact_manifest": _relative(str(artifact_path)),
        "source_artifact_manifest_sha256": _hash(artifact_path),
        "source_H_probe_manifest": _relative(str(h_path)),
        "source_H_probe_manifest_sha256": _hash(h_path),
        "training_root_record_count": len(arm_rows),
        "training_root_count": len({int(root["seed"]) for root in arm_rows}),
        "training_root_seeds": sorted({int(root["seed"]) for root in e.get("all_capacity_and_optimization_roots", [])}),
        "format_count": len(FORMATS),
        "artifact_variant_count": len(e_rows),
        "unique_arm_count": len(arms),
        "hardware_probe_counts_by_exact_artifact_hash": probe_counts,
        "full_pipeline_measurement_status": "blocked_until_changed_firmware_H_parity_resource_and_standalone_gates_pass",
        "paired_physical_block_rule": {
            "minimum_independently_started_randomized_blocks_per_reported_comparison": 10,
            "training_roots_per_comparison": "all ten matched training roots; pair same root, input, and condition within block",
            "nested_repeats": "repeat within each block only after meter bandwidth and settling are qualified",
            "randomization": "derive a per-comparison/per-root order seed from SHA-256(comparison_id|root_seed|block_index); persist before capture",
            "unsupported_arm_handling": "retain every variant row and exact failure/probe status; do not enter it in a powered comparison unless exact H and live camera gates pass",
        },
        "all_pairwise_arm_format_comparisons": comparisons,
        "prospective_pairwise_comparison_count": len(comparisons),
        "prospective_capture_count_if_every_pair_qualified": pairwise_capture_count,
        "prospective_hours_if_every_pair_qualified": round(pairwise_capture_count * 50 / 3600, 2),
        "comparison_execution_note": "All E arms/formats remain enumerated. This is a conditional full matrix, not a command to run unsupported or unqualified variants; only exact H plus live-camera gates may shortlist physical arms. Each reported comparison still requires ten started paired blocks.",
        "analysis_matrix": {
            "required_fields": ["model_bytes", "actual_arena_bytes", "peak_heap_bytes", "warm_latency_ms",
                "cold_load_latency_ms", "swap_cost_ms", "quality", "accepted_correct_known", "unknown_false_identity",
                "accepted_correct_encounter_energy", "paired_block_id", "training_root_seed"],
            "timestamp_basis": "device sample times only after G qualification; otherwise report supported long-window relative power, not per-inference energy",
            "selection_role": "development_only",
        },
        "variants": e_rows,
    }


def _stable_block_seed(phase: str, condition: str, replicate: int) -> int:
    raw = ("JIK-v1|%s|%d|%s|%d" % (phase, PLAN_SEED, condition, replicate)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _paired_schedule(phase: str, conditions=CONDITIONS, policies=POLICIES, replicates=10):
    seeds = [int(row["root_seed"]) for row in _json(STUDY / "config/seed_registry.json")["training_roots"]]
    blocks = []
    cells = []
    for condition_index, condition in enumerate(conditions):
        for replicate in range(1, replicates + 1):
            seed = _stable_block_seed(phase, condition, replicate)
            order = list(policies)
            random.Random(seed).shuffle(order)
            block_id = "%s-%s-B%02d" % (phase, condition.upper(), replicate)
            training_root = seeds[(replicate - 1) % len(seeds)]
            block = {"block_id": block_id, "condition": condition,
                "replicate": replicate, "physical_block_seed": seed,
                "training_root_seed": training_root,
                "presentation_session_id": "%s-SESSION-%s-B%02d" % (phase, condition.upper(), replicate),
                "policy_order": order,
                "block_interpretation": "one independently started physical paired block; policies share the declared scene/input schedule",
                "data_role": "controlled_presentation" if phase == "JDEV" else "sealed_final_after_freeze"}
            blocks.append(block)
    block_order_seed = int.from_bytes(hashlib.sha256((phase + "|block-order|" + str(PLAN_SEED)).encode()).digest()[:4], "big")
    random.Random(block_order_seed).shuffle(blocks)
    for block_order_index, block in enumerate(blocks, start=1):
        block["randomized_block_order_index"] = block_order_index
        for order_index, policy in enumerate(block["policy_order"], start=1):
            cells.append({
                    "run_id": "%s-%s-B%02d-%s" % (phase, block["condition"].upper(), block["replicate"], policy.replace("-", "")),
                    "block_id": block["block_id"],
                    "policy_id": policy,
                    "condition": block["condition"],
                    "physical_block_seed": block["physical_block_seed"],
                    "training_root_seed": block["training_root_seed"],
                    "session_id": block["presentation_session_id"],
                    "randomized_block_order_index": block_order_index,
                    "randomized_order_index": order_index,
                    "status": "not_started",
                    "result_path": "runs/%s/%s/capture_result.json" % (phase,
                        "%s-%s-B%02d-%s" % (phase, block["condition"].upper(), block["replicate"], policy.replace("-", ""))),
                })
    return blocks, cells


def build_j_plan() -> dict:
    runtime_path = APP / "firmware/runtime_config.json"
    runtime = _json(runtime_path)
    if not set(POLICIES).issubset(runtime.get("policies", {})):
        raise ValueError("runtime config does not contain the complete planned J policy set")
    blocks, cells = _paired_schedule("JDEV")
    pilot_blocks, pilot_cells = _paired_schedule("JPILOT", conditions=CONDITIONS[:5], policies=POLICIES, replicates=10)
    return {
        "schema": "J-prospective-controller-plan-1",
        "status": "prospective_plan_not_controller_result",
        "created_utc": "2026-10-30",
        "runtime_config_relative": _relative(str(runtime_path)),
        "runtime_config_sha256": _hash(runtime_path),
        "policy_ids": list(POLICIES),
        "policy_definitions": {key: runtime["policies"][key] for key in POLICIES},
        "candidate_decision_rules": {
            "simple_threshold": {"policy": "A-model", "rule": "large model only when small embedding score falls inside the predeclared uncertainty band"},
            "risk_aware": {"policy": "A-risk", "rule": "look up development-calibrated small-model error risk by score bin; no calibration means do not escalate"},
            "measured_cost_aware": {"policy": "A-cost", "rule": "escalate only when calibrated expected accuracy gain times frozen energy value exceeds measured incremental energy and remaining latency budget"},
            "calibration_status": runtime.get("controller_calibration", {}).get("status"),
            "tuning_budget": {"candidate_settings_max": 12, "seeded_blocks_per_policy_comparison": 10,
                "role": "development_only", "attempt_log": "append every setting and reason before scoring", "final_data_access": False},
        },
        "conditions": list(CONDITIONS),
        "paired_block_count": len(blocks),
        "planned_policy_captures": len(cells),
        "minimum_paired_blocks_per_policy_comparison_and_condition": 10,
        "training_root_schedule": sorted({block["training_root_seed"] for block in blocks}),
        "blocks": blocks,
        "cells": cells,
        "pilot": {"seeded_block_count": len(pilot_blocks), "planned_policy_captures": len(pilot_cells),
            "blocks": pilot_blocks, "cells": pilot_cells,
            "scope": "development-only action/cadence and cost-fitting pilot; no physical results are implied"},
        "runtime_estimate": {"measurement_seconds_per_capture": 30, "warmup_seconds_per_capture": 5,
            "setup_and_stop_seconds_per_capture": 15,
            "planned_capture_count": len(cells),
            "estimated_total_hours": round(len(cells) * 50 / 3600, 2),
            "estimated_raw_disk_upper_bound_mb": round(len(cells) * 0.5, 1),
            "assumptions": "planning envelope only; replace duration after G bandwidth qualification; includes no failed-run reruns"},
        "physical_gate": "requires actual Nicla, updated firmware readback, F/H exact parity/resource/standalone checks, and G-qualified meter path",
        "final_selection_rule": "selection uses development-only data and measured resource evidence; no final K data, no reject-all selection",
    }


def build_k_plan() -> dict:
    blocks, cells = _paired_schedule("KFINAL")
    return {
        "schema": "K-prospective-final-capture-plan-1",
        "status": "sealed_final_plan_no_final_rows_opened",
        "created_utc": "2026-10-30",
        "policy_ids": list(POLICIES),
        "conditions": list(CONDITIONS),
        "paired_block_count": len(blocks),
        "planned_policy_captures": len(cells),
        "minimum_paired_blocks_per_reported_comparison_and_condition": 10,
        "blocks": blocks,
        "cells": cells,
        "final_role_access": {"authorized_now": False, "required_frozen_receipt": "handoffs/J/FROZEN_SELECTION.json",
            "runtime_gate": "presentation_player --block final requires a hash-bound frozen receipt; no final rows were enumerated during plan generation",
            "stimulus_source_role": "sealed final after freeze only", "unknowns": "disjoint unknown_final identities under B role registry"},
        "ground_truth": {"source": "randomized presentation-player event log joined by run_id/trial_id and source record IDs",
            "independent_of_controller_output": True, "empty_and_ambiguous_scenes": "explicit planned conditions"},
        "runtime_estimate": {"measurement_seconds_per_capture": 30, "warmup_seconds_per_capture": 5,
            "setup_and_stop_seconds_per_capture": 15, "planned_capture_count": len(cells),
            "estimated_total_hours": round(len(cells) * 50 / 3600, 2),
            "estimated_raw_disk_upper_bound_mb": round(len(cells) * 0.5, 1)},
        "resume_rule": "each cell has a unique run_id and immutable raw directory; only a complete, hash-valid receipt satisfies it; interrupted/failed originals stay in place and reruns use a new run_id linked by rerun_of",
    }


def build_i_lifecycle_plan(model_matrix: dict) -> dict:
    runtime_config = _json(APP / "firmware/runtime_config.json")
    runtime_models = runtime_config["identity"]
    model_root_seeds = {}
    for slot in ("small", "large"):
        match = re.search(r"seed-(\d+)$", str(runtime_models[slot].get("model_id", "")))
        model_root_seeds[slot] = int(match.group(1)) if match else None
    stage_specs = [
        {"stage_id": "idle", "policy_id": "U", "model_slot": "small", "preload_model": None,
         "input_source": "actual powered Nicla with camera asleep; USB telemetry remains enabled"},
        {"stage_id": "camera_capture_only", "policy_id": "U", "model_slot": "small", "preload_model": None,
         "input_source": "actual CSI camera and balanced mixed development presentation"},
        {"stage_id": "detector_only", "policy_id": "U", "model_slot": "small", "preload_model": None,
         "input_source": "actual camera frames; no identity inference"},
        {"stage_id": "identity_small_only", "policy_id": "S", "model_slot": "small", "preload_model": None,
         "input_source": "actual camera crop; output vector length and latency only; no identity decision"},
        {"stage_id": "identity_large_only", "policy_id": "H", "model_slot": "large", "preload_model": None,
         "input_source": "actual camera crop; output vector length and latency only; no identity decision"},
        {"stage_id": "cold_model_load", "policy_id": "S", "model_slot": "small", "preload_model": None,
         "input_source": "actual camera crop after bounded cache clear; not a supply-cycle cold start"},
        {"stage_id": "cold_model_load", "policy_id": "H", "model_slot": "large", "preload_model": None,
         "input_source": "actual camera crop after bounded cache clear; not a supply-cycle cold start"},
        {"stage_id": "warm_inference", "policy_id": "S", "model_slot": "small", "preload_model": "small",
         "input_source": "actual camera crop; preloaded model; requires two-resident cache and H memory gate"},
        {"stage_id": "warm_inference", "policy_id": "H", "model_slot": "large", "preload_model": "large",
         "input_source": "actual camera crop; preloaded model; requires two-resident cache and H memory gate"},
        {"stage_id": "model_swap_and_gc", "policy_id": "A-model", "model_slot": "small", "preload_model": None,
         "input_source": "same actual camera crop through small/large/small; record cache, GC and memory"},
        {"stage_id": "full_pipeline", "policy_id": "H", "model_slot": "large", "preload_model": None,
         "input_source": "actual camera with balanced mixed development presentation and independent event log"},
    ]
    stage_ids = [row["stage_id"] for row in stage_specs]
    blocks = []
    cells = []
    for replicate in range(1, 11):
        block_seed = _stable_block_seed("I_LIFECYCLE", "mixed", replicate)
        order = list(range(len(stage_specs)))
        random.Random(block_seed).shuffle(order)
        block_id = "I-LIFECYCLE-MIXED-B%02d" % replicate
        session_id = "I-LIFECYCLE-SESSION-B%02d" % replicate
        blocks.append({"block_id": block_id, "replicate": replicate,
            "paired_stimulus_seed": block_seed,
            "runtime_model_root_seeds": model_root_seeds,
            "session_id": session_id,
            "randomized_stage_order": [stage_specs[index]["stage_id"] + "-%02d" % index for index in order],
            "presentation_condition": "mixed",
            "presentation_count": 12,
            "data_role": "development_only"})
        for order_index, stage_index in enumerate(order, start=1):
            spec = stage_specs[stage_index]
            cell_seed = _stable_block_seed("I_LIFECYCLE_CELL", spec["stage_id"] + "-%s" % spec["model_slot"], replicate)
            model_used = spec["stage_id"] in ("identity_small_only", "identity_large_only",
                "cold_model_load", "warm_inference", "full_pipeline")
            training_root_seed = model_root_seeds.get(spec["model_slot"]) if model_used else None
            cells.append({"run_id": "%s-%s-%s-B%02d" % ("I-LIFECYCLE", spec["stage_id"],
                    spec["model_slot"], replicate), "block_id": block_id,
                "paired_stimulus_seed": block_seed, "physical_block_seed": cell_seed,
                "training_root_seed": training_root_seed,
                "runtime_model_root_seeds": model_root_seeds,
                "session_id": session_id, "policy_id": spec["policy_id"],
                "stage_id": spec["stage_id"], "model_slot": spec["model_slot"],
                "preload_model": spec["preload_model"], "condition": "mixed",
                "randomized_order_index": order_index,
                "status": "not_started",
                "result_path": "runs/I/lifecycle/%s/capture_result.json" %
                    ("%s-%s-%s-B%02d" % ("I-LIFECYCLE", spec["stage_id"], spec["model_slot"], replicate))})
    return {
        "schema": "I-model-lifecycle-execution-plan-1",
        "status": "prospective_runner_contract_physical_execution_blocked_by_H_and_G",
        "created_utc": "2026-10-30",
        "artifact_variant_count": model_matrix["artifact_variant_count"],
        "all_arm_format_comparison_count": model_matrix["prospective_pairwise_comparison_count"],
        "training_root_seeds": model_matrix["training_root_seeds"],
        "required_independent_paired_blocks_per_reported_comparison": 10,
        "lifecycle_stages": stage_specs,
        "stage_order": "stage order is deterministically randomized within ten paired mixed-presentation blocks; stages share each block's exact public-image order and seed",
        "paired_block_count": len(blocks),
        "planned_lifecycle_capture_count": len(cells),
        "blocks": blocks,
        "cells": cells,
        "stimulus": {"block": "development", "condition": "mixed",
            "presentation_manifest": "work/project-finish-v4/data/primary_split_manifest.jsonl",
            "presentation_manifest_sha256": _hash(STUDY / "data/primary_split_manifest.jsonl"),
            "per_cell_trials": 12, "balanced_categories": ["known", "unknown", "empty", "ambiguous"],
            "duration_per_cell_s": 40, "warmup_s": 5, "hold_s": 1.5, "blank_s": 1.0},
        "current_runtime_model_ids": {"small": runtime_config["identity"]["small"]["model_id"],
            "large": runtime_config["identity"]["large"]["model_id"]},
        "current_runtime_model_roots": {"small": runtime_config["identity"]["small"]["model_sha256"],
            "large": runtime_config["identity"]["large"]["model_sha256"]},
        "scope_limit": "This lifecycle schedule measures the exact currently deployed small/large pair. The separate 544-comparison E matrix remains conditional until each variant has a runtime bundle and exact H qualification receipt.",
        "order_rule": "SHA-256(comparison_id|training_root_seed|block_index) determines arm order; record actual cold/warm cache and supply-reset status separately",
        "analysis_outputs": ["raw HID packets", "raw Nicla serial bytes and chunk hashes", "decoded meter report table",
            "device stage timings", "model bytes", "actual model arena and peak heap", "root-level quality and calibration",
            "paired block summaries", "unsupported/invalid arm ledger", "Pareto table"],
        "selection": {"role": "development_only", "selector_module": "nicla_study.development_selector",
            "quality_targets": "config/ENGINEERING_TARGETS.json", "final_role_access": False,
            "large_model_rule": "must improve relevant development recognition cases; model size alone does not qualify"},
        "execution_gates": ["updated exact firmware/model/config readback", "C and D exact parity and full-pipeline resource gates",
            "G raw decoder, clock/timing and power boundary qualification", "exclusive Windows meter and board leases"],
        "execution_note": "The matrix lists all 170 E training roots and four exports each (680 rows) plus exact H outcomes. Variants are not removed for being unsupported. A stage is measured only after its exact deployment and camera/memory gate pass.",
        "runner_entrypoint": "PYTHONPATH=FaceRecognition python -m nicla_study.i_lifecycle_campaign --dry-run",
        "runtime_estimate": {"capture_seconds_each": 45, "setup_seconds_each": 20,
            "planned_capture_count": len(cells),
            "estimated_total_hours": round(len(cells) * 65 / 3600, 2),
            "raw_upper_bound_mb": round(len(cells) * 0.5, 1)},
        "scope_limit_physical": "No board or meter trials were run by this preparation receipt.",
    }


def build_all() -> dict:
    model_matrix = build_model_matrix()
    lifecycle_plan = build_i_lifecycle_plan(model_matrix)
    j_plan = build_j_plan()
    k_plan = build_k_plan()
    outputs = {
        "config/I_PHYSICAL_MATRIX.json": model_matrix,
        "config/I_LIFECYCLE_PLAN.json": lifecycle_plan,
        "config/J_CONTROLLER_PLAN.json": j_plan,
        "config/K_FINAL_CAPTURE_PLAN.json": k_plan,
    }
    for relative, value in outputs.items():
        atomic_write_json(STUDY / relative, value)
    validation = {
        "schema": "JIK-plan-validation-1",
        "status": "pass",
        "final_role_opened": False,
        "matrix_variant_count": model_matrix["artifact_variant_count"],
        "I_pairwise_comparison_count": model_matrix["prospective_pairwise_comparison_count"],
        "training_roots": model_matrix["training_root_count"],
        "controller_policies": j_plan["policy_ids"],
        "J_paired_blocks_per_condition": 10,
        "J_captures": j_plan["planned_policy_captures"],
        "K_paired_blocks_per_condition": 10,
        "K_captures": k_plan["planned_policy_captures"],
        "outputs": {path: _hash(STUDY / path) for path in outputs},
        "physical_measurements_performed": False,
        "selection_or_freeze_performed": False,
    }
    atomic_write_json(STUDY / "reports/J/JIK_PLAN_VALIDATION.json", validation)
    return validation


def main(argv=None):
    global WORKSPACE, STUDY, APP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args(argv)
    WORKSPACE = args.workspace.resolve()
    STUDY = WORKSPACE / "work/project-finish-v4"
    APP = WORKSPACE / "FaceRecognition"
    result = build_all()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
