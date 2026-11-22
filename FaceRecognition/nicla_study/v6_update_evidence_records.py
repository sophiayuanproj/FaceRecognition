"""Amend current project ledgers with the versioned camera-geometry repair."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(rel, kind=None):
    path = ROOT / rel
    row = {"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)}
    if kind:
        row["kind"] = kind
    return row


def load(name):
    path = STUDY / name
    return path, json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_once(rows, row):
    if not any(existing.get("path") == row.get("path") for existing in rows):
        rows.append(row)


def main():
    ledger_path, ledger = load("BRANCH_LEDGER.json")
    ledger["updated_utc"] = NOW
    ledger["closure_note"] = (
        "Prior v5 Mac-close finding is superseded by corrected camera geometry v1. "
        "Historical runs remain preserved; the current branch is open and not ready for transfer or final build."
    )
    branches = {row["id"]: row for row in ledger["branches"]}

    camera = branches["camera"]
    camera["execution_status"] = "host_geometry_policy_checks_passed_device_parity_pending"
    camera["current_corrected_geometry_independent_blocks"] = 0
    camera["additional_single_frame_attempts"] = 1
    append_once(camera["evidence"], {
        "fact": "A physical K01 development source was visible in the saved 240x240 RGB565 frame (hash verified). FOMO allocation failed before detection; no device ROI/tensor parity.",
        "path": "work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/FRAME_RECOVERY.json",
        "unit": "one software-reset single-frame attempt; not an independent optical block",
    })
    append_once(camera["evidence"], {
        "fact": "Shared camera geometry and policy-controller host checks pass; the runtime config still selects the legacy edge-reject policy and the board input path remains unqualified.",
        "path": "work/project-research-v5/runs/controller_policy_v1/TEST_RESULT.json",
        "unit": "17 Mac unittest methods, 10 seeded geometry cases, 25 decision-contract cases; no hardware exercised",
    })

    identity = branches["identity"]
    identity["execution_status"] = "historical_v5_runs_invalidated_for_geometry_corrected_v6_pilot_not_useful"
    identity["historical_v5_roots_invalid_for_camera_geometry"] = 10
    identity["corrected_v6_pilot_roots"] = 1
    identity["corrected_v6_full_campaign_roots"] = 0
    identity["corrected_v6_route_pilots"] = {
        "v6_camera_crop_features48_seed56001": "not_useful",
        "v6_camera_mobilenet_distill_seed56101": "teacher_similarity_fit_no_identity_gain",
        "v6_camera_supervised_mobilenet_seed56102": "not_useful",
        "reported_ten_root_comparisons": 0,
    }
    append_once(identity["evidence"], {
        "fact": "Corrected camera-space 120px pilot seed56001 reached 20% raw top-1, 6% correct accepted and 10% coverage on 50 known development rows; 2 wrong known accepts and 1 false unknown group/30. Not useful; no nine-root padding.",
        "path": "work/project-research-v5/runs/camera_corrected_recognition_v1/roi_120_camera_px/crop_120/56001/COMPLETE.json",
        "unit": "one gradient root; frozen screen-source recaptures; no future-final data",
    })
    append_once(identity["evidence"], {
        "fact": "Aligned SFace is a useful host diagnostic teacher (64% end-to-end correct acceptance/coverage, 0/30 unknown false groups) but the 43-fit-row MobileNet distillation pilot did not transfer identity accuracy; no MCU path is qualified.",
        "path": "work/project-research-v5/CAMERA_CORRECTED_MODEL_DIAGNOSTICS_V1.md",
        "unit": "host diagnostics and one student gradient root; frozen development/calibration roles; no future-final data",
    })
    append_once(identity["evidence"], evidence(
        "work/project-research-v5/runs/camera_corrected_deep_student_v1/RESULT.json", "single_gradient_pilot_result"))
    append_once(identity["evidence"], evidence(
        "work/project-research-v5/runs/camera_supervised_mobilenet_v1/RESULT.json", "single_gradient_pilot_result"))
    append_once(identity["evidence"], evidence(
        "work/project-research-v5/runs/camera_corrected_teacher_v1/RESULT.json", "host_only_teacher_diagnostic"))

    control = branches["control"]
    control["execution_status"] = "host_policy_controller_implemented_tests_passed_hardware_pending"
    append_once(control["evidence"], {
        "fact": "A-acquire records camera captures without detector/model calls; A-sense gates identity during wearer motion; A-full retains 1.5-second periodic stationary camera checks with sleep disabled by default. Stale votes, invalid quality/motion, escalation limits and late identity suppression are covered by passing Mac tests.",
        "path": "work/project-research-v5/runs/controller_policy_v1/TEST_RESULT.json",
        "unit": "17 Mac unittest methods; no board or power measurement",
    })

    numerics = branches["numerics"]
    append_once(numerics["evidence"], {
        "fact": "Updated binary32/open-set decision contract: 25/25 cases passed, including threshold adjacency, ties, quantized thresholds, invalid probabilities, row/group count faults, and NO_FACE/UNCERTAIN/OTHER decisions.",
        "path": "work/project-research-v5/runs/camera_corrected_input_v1/DECISION_CONTRACT.json",
        "unit": "25 deterministic decision-contract cases",
    })

    prototype = branches["prototype"]
    prototype["execution_status"] = "not_qualified_actual_fomo_allocation_failed_before_detector_output"
    append_once(prototype["evidence"], {
        "fact": "The one software-reset K01 source-present attempt captured raw RGB565, then raised ValueError: Failed to allocate tensors while constructing the ROM FOMO model; no detector, ROI or identity output.",
        "path": "work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/RESULT.json",
        "unit": "one failed device attempt; not a cold start or no-face outcome",
    })

    handoff = branches["handoff"]
    handoff["execution_status"] = "historical_v5_instrument_transfer_verified_current_repair_not_packaged"
    append_once(handoff["evidence"], {
        "fact": "The prior v5 instrument-only ZIP still matches its hash sidecar, but current corrected sources are not in that archive. Current readiness is regenerated as not-ready and the old JSON is preserved as a snapshot.",
        "path": "work/project-research-v5/READY_FOR_WINDOWS.json",
        "unit": "current branch readiness; previous immutable transfer preserved",
    })
    save(ledger_path, ledger)

    claims_path, claims = load("CLAIM_EVIDENCE.json")
    claims["generated_utc"] = NOW
    by_id = {row["id"]: row for row in claims["claims"]}

    identity_claim = by_id["camera_matched_identity_usefulness"]
    identity_claim["claim"] = (
        "Historical v5 camera-recognition results are invalid because the cropper multiplied camera-pixel FOMO centers by 2.5. "
        "Corrected v6 learned pilots remain not useful for deployment: the best learned route was 24% raw top-1 with 0% correct acceptance and coverage; aligned SFace is a host-only diagnostic teacher."
    )
    identity_claim["status"] = "superseded_invalid_geometry_corrected_v6_pilot_not_useful"
    identity_claim["metrics"] = {
        "historical_v5_metrics_valid_for_camera_recognition": False,
        "corrected_v6_training_roots": [56001],
        "additional_single_pilot_seeds": [56101, 56102],
        "ten_root_learned_comparisons": 0,
        "known_development_rows": 50,
        "known_raw_top1": 0.20,
        "known_correct_acceptance": 0.06,
        "known_coverage": 0.10,
        "known_wrong_accepted": 2,
        "unknown_development_false_groups": 1,
        "unknown_development_groups": 30,
        "unknown_calibration_false_groups": 0,
        "unknown_calibration_groups": 30,
        "zero_event_upper95_if_no_false_accepts_of_30": 0.09503385285530419,
        "future_final_used": False,
    }
    identity_claim["metrics"]["corrected_route_pilots"] = {
        "aligned_sface_teacher_host_only": {
            "known_raw_top1": 0.64, "known_correct_acceptance": 0.64,
            "known_coverage": 0.64, "known_fomo_valid_rows": 32,
            "unknown_false_groups": 0, "unknown_groups": 30,
            "deployment_eligible": False,
        },
        "fomo_crop_sface_host_only": {
            "known_raw_top1": 0.48, "known_correct_acceptance": 0.22,
            "known_coverage": 0.30, "known_wrong_accepted": 4,
            "unknown_false_groups": 1,
        },
        "linear_teacher_projection_seed56001": {
            "base_known_raw_top1": 0.22, "projected_known_raw_top1": 0.18,
            "projected_correct_acceptance": 0.0, "projected_coverage": 0.0,
        },
        "mobilefacenet_float_112": {
            "known_raw_top1": 0.28, "known_correct_acceptance": 0.06,
            "known_coverage": 0.10, "known_wrong_accepted": 2,
            "unknown_false_groups": 1,
        },
        "mobilefacenet_int8_112": {
            "known_raw_top1": 0.28, "known_correct_acceptance": 0.08,
            "known_coverage": 0.12, "known_wrong_accepted": 2,
            "unknown_false_groups": 2,
        },
        "aligned_teacher_student_seed56101": {
            "known_raw_top1": 0.24, "known_correct_acceptance": 0.0,
            "known_coverage": 0.0, "known_fomo_valid_rows": 32,
            "unknown_false_groups": 0, "fit_rows": 43, "stopping_rows": 8,
        },
        "supervised_mobilenet_seed56102": {
            "known_raw_top1": 0.14, "known_correct_acceptance": 0.02,
            "known_coverage": 0.06, "known_wrong_accepted": 2,
            "unknown_false_groups": 1, "fit_rows": 43, "stopping_rows": 8,
        },
        "ten_root_learned_comparisons": 0,
    }
    identity_claim["evidence"] = [
        row for row in identity_claim["evidence"]
        if row.get("path") != "work/project-research-v5/READY_FOR_WINDOWS.json"
    ]
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/READY_FOR_WINDOWS_V5_TRANSFER_SNAPSHOT.json", "historical_v5_readiness_snapshot"))
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/runs/camera_corrected_input_v1/INPUTS.json", "corrected_input_inventory"))
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/runs/camera_corrected_recognition_v1/roi_120_camera_px/crop_120/56001/COMPLETE.json", "corrected_pilot_metrics"))
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/CAMERA_CORRECTED_MODEL_DIAGNOSTICS_V1.md", "current_model_diagnostics"))
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/runs/camera_corrected_deep_student_v1/RESULT.json", "teacher_student_pilot"))
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/runs/camera_supervised_mobilenet_v1/RESULT.json", "direct_supervised_pilot"))
    append_once(identity_claim["evidence"], evidence("work/project-research-v5/runs/camera_corrected_teacher_v1/RESULT.json", "host_only_teacher_diagnostic"))

    unknown_claim = by_id["unknown_identity_precision"]
    unknown_claim["claim"] = (
        "The unchanged 1% one-sided 95% unknown-group target is not established. Corrected v6 had one false-accepted "
        "development group among 30 and zero unknown calibration groups among 30; zero events still has a 9.503% upper95 bound."
    )
    unknown_claim["status"] = "not_established_corrected_v6"
    unknown_claim["team_gate_unchanged"] = True
    append_once(unknown_claim["evidence"], evidence("work/project-research-v5/runs/camera_corrected_recognition_v1/roi_120_camera_px/crop_120/56001/COMPLETE.json", "corrected_pilot_metrics"))

    additional = [
        {
            "id": "corrected_camera_coordinate_contract",
            "claim": "Corrected replay uses direct camera_xywh_px FOMO rectangles and a fixed clamped 120px camera ROI; ten seeded geometry fixtures and visual crop review passed.",
            "status": "supported_limited_scope",
            "future_final_used": False,
            "evidence": [
                evidence("work/project-research-v5/runs/camera_corrected_input_v1/INPUTS.json", "derived_input_inventory"),
                evidence("work/project-research-v5/private/camera_corrected_input_v1/CROP_GEOMETRY_CONTACT_SHEET.json", "visual_qc_report"),
                evidence("FaceRecognition/firmware/camera_geometry.py", "shared_geometry_source"),
            ],
        },
        {
            "id": "actual_camera_frame_fomo_allocation",
            "claim": "One K01 source-present physical capture saved and hash-verified a 240x240 RGB565 frame; the board FOMO model failed allocation before producing detector output or downstream tensors.",
            "status": "supported_limited_scope",
            "future_final_used": False,
            "evidence": [
                evidence("work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/FRAME_RECOVERY.json", "raw_frame_recovery_record"),
                evidence("work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/RESULT.json", "device_run_transcript_record"),
                evidence("work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/STIMULUS.json", "development_stimulus_context"),
            ],
        },
        {
            "id": "float32_decision_contract_v2",
            "claim": "The shared binary32/open-set decision contract passed 25 deterministic cases; it does not establish camera or recognition qualification.",
            "status": "supported_limited_scope",
            "future_final_used": False,
            "evidence": [
                evidence("work/project-research-v5/runs/camera_corrected_input_v1/DECISION_CONTRACT.json", "decision_contract_report"),
                evidence("FaceRecognition/firmware/open_set_decision.py", "shared_decision_source"),
            ],
        },
        {
            "id": "camera_corrected_model_routes_v1",
            "claim": "Corrected host diagnostics include a conditional aligned SFace teacher and three single-root learned pilots; no learned candidate is useful or qualified, and no ten-root comparison was reported.",
            "status": "host_diagnostics_complete_candidate_not_useful",
            "future_final_used": False,
            "evidence": [
                evidence("work/project-research-v5/CAMERA_CORRECTED_MODEL_DIAGNOSTICS_V1.md", "human_readable_model_findings"),
                evidence("work/project-research-v5/runs/camera_corrected_teacher_v1/RESULT.json", "host_teacher_metrics"),
                evidence("work/project-research-v5/runs/camera_corrected_teacher_student_v1/RESULT.json", "linear_projection_metrics"),
                evidence("work/project-research-v5/runs/camera_corrected_face_encoder_v1/RESULT.json", "float_int8_encoder_metrics"),
                evidence("work/project-research-v5/runs/camera_corrected_deep_student_v1/RESULT.json", "teacher_student_metrics"),
                evidence("work/project-research-v5/runs/camera_supervised_mobilenet_v1/RESULT.json", "six_way_classifier_metrics"),
            ],
        },
        {
            "id": "mac_geometry_policy_control_tests_v1",
            "claim": "Shared camera geometry, float32 decision contracts and policy-controller safety rules pass Mac host tests; no board behavior, deployment, startup, or power claim is supported.",
            "status": "host_software_verified_hardware_pending",
            "future_final_used": False,
            "evidence": [
                evidence("work/project-research-v5/runs/controller_policy_v1/TEST_RESULT.json", "test_receipt"),
                evidence("FaceRecognition/firmware/identity_contracts.py", "shared_geometry_and_decision_source"),
                evidence("FaceRecognition/firmware/policy_controller.py", "policy_source"),
                evidence("FaceRecognition/firmware/recognize_controlled.py", "runtime_controller_integration_source"),
            ],
        },
    ]
    claim_ids = {row["id"] for row in claims["claims"]}
    claims["claims"].extend(row for row in additional if row["id"] not in claim_ids)
    save(claims_path, claims)

    registry_path = STUDY / "EXPERIMENT_REGISTRY.jsonl"
    rows = [json.loads(line) for line in registry_path.read_text().splitlines() if line.strip()]
    ids = {row.get("experiment_id") for row in rows}
    additions = [
        {
            "schema": "nicla-v6-experiment-1",
            "experiment_id": "v6_camera_coordinate_corrected_input_120",
            "status": "host_geometry_visual_qc_passed_device_fomo_allocation_failed",
            "scope": "Corrected direct camera-coordinate FOMO replay, fixed 120px ROI, RGB48/int8 inputs; no device tensor parity.",
            "future_final_used": False,
            "frames": 255,
            "quality_counts": {"single_cluster": 142, "multiple_clusters": 28, "no_face": 85},
            "geometry_fixtures_passed": 10,
            "decision_contract_cases_passed": 25,
            "evidence": [
                evidence("work/project-research-v5/runs/camera_corrected_input_v1/INPUTS.json", "input_inventory"),
                evidence("work/project-research-v5/private/camera_corrected_input_v1/CROP_GEOMETRY_CONTACT_SHEET.json", "visual_qc"),
            ],
        },
        {
            "schema": "nicla-v6-experiment-1",
            "experiment_id": "v6_corrected_camera_task48_seed56001",
            "status": "pilot_negative",
            "scope": "One corrected-geometry camera task classifier pilot; ten-root comparison not started because the route was not useful.",
            "future_final_used": False,
            "gradient_roots": [56001],
            "known_correct_acceptance": 0.06,
            "known_coverage": 0.10,
            "known_raw_top1": 0.20,
            "known_wrong_accepted": 2,
            "unknown_development_false_groups": 1,
            "evidence": [
                evidence("work/project-research-v5/runs/camera_corrected_recognition_v1/roi_120_camera_px/crop_120/56001/COMPLETE.json", "pilot_metrics"),
                evidence("work/project-research-v5/runs/camera_corrected_recognition_v1/LEARNING_CONTROL.json", "fit_control"),
            ],
        },
        {
            "schema": "nicla-v6-experiment-1",
            "experiment_id": "v6_actual_codex_side_window_frame_attempt",
            "status": "raw_frame_captured_fomo_allocation_failed",
            "scope": "One software-reset physical single-frame attempt with a K01 development source displayed in the Codex side window; no detection, crop, tensor or embedding result.",
            "future_final_used": False,
            "software_reset_blocks": 1,
            "cold_starts": 0,
            "evidence": [
                evidence("work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/FRAME_RECOVERY.json", "frame_recovery"),
                evidence("work/project-research-v5/reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01/RESULT.json", "failed_device_attempt"),
            ],
        },
        {
            "schema": "nicla-v6-experiment-1",
            "experiment_id": "v6_camera_corrected_model_diagnostics_v1",
            "status": "teacher_diagnostic_positive_small_student_pilots_not_useful",
            "scope": "Host-only SFace/YuNet diagnostic, one aligned-teacher student gradient pilot, one direct six-way classifier pilot, and corrected existing encoder comparisons over the frozen 255-frame set.",
            "future_final_used": False,
            "gradient_pilot_seeds": [56001, 56101, 56102],
            "ten_root_learned_comparison": False,
            "evidence": [
                evidence("work/project-research-v5/CAMERA_CORRECTED_MODEL_DIAGNOSTICS_V1.md", "findings"),
                evidence("work/project-research-v5/runs/camera_corrected_teacher_v1/RESULT.json", "host_teacher"),
                evidence("work/project-research-v5/runs/camera_corrected_teacher_student_v1/RESULT.json", "linear_projection"),
                evidence("work/project-research-v5/runs/camera_corrected_face_encoder_v1/RESULT.json", "float_int8_encoder"),
                evidence("work/project-research-v5/runs/camera_corrected_deep_student_v1/RESULT.json", "teacher_student"),
                evidence("work/project-research-v5/runs/camera_supervised_mobilenet_v1/RESULT.json", "supervised_student"),
            ],
        },
        {
            "schema": "nicla-v6-experiment-1",
            "experiment_id": "v6_mac_geometry_policy_tests",
            "status": "host_tests_passed_board_actions_pending",
            "scope": "Shared FOMO camera geometry, threshold precision, bounded policy scheduling, stale evidence and safe identity fallback host tests.",
            "future_final_used": False,
            "test_methods": 17,
            "geometry_fixture_cases": 10,
            "decision_contract_cases": 25,
            "hardware_exercised": False,
            "evidence": [
                evidence("work/project-research-v5/runs/controller_policy_v1/TEST_RESULT.json", "test_receipt"),
                evidence("FaceRecognition/nicla_study/tests/test_policy_controller_v6.py", "policy_test_source"),
            ],
        },
    ]
    rows.extend(row for row in additions if row["experiment_id"] not in ids)
    registry_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")

    state_path, state = load("STATE.json")
    state["updated_utc"] = NOW
    state["completion_status"]["mac_research"] = "open_corrected_camera_models_geometry_and_device_parity"
    state["completion_status"]["recognition_application"] = "not_qualified_no_useful_small_candidate"
    state["completion_status"]["ready_for_final_build"] = False
    state["completion_status"]["windows_transfer"] = "historical_v5_instrument_transfer_verified_current_repair_not_packaged"
    state["execution_status"] = "Mac_research_reopened_corrected_geometry_models_and_controls_incomplete"
    state["ready_for_windows"] = False
    state["ready_for_final_build"] = False
    state["milestones"]["M4"]["corrected_v6_model_routes"] = {
        "status": "no_useful_small_candidate; host_teacher_is_diagnostic_only",
        "gradient_pilot_seeds": [56001, 56101, 56102],
        "ten_root_learned_comparisons": 0,
        "aligned_sface_teacher": {"known_correct_acceptance": 0.64,
                                   "known_coverage": 0.64,
                                   "unknown_false_groups": 0,
                                   "unknown_groups": 30,
                                   "host_only": True},
        "deep_teacher_student": {"known_raw_top1": 0.24,
                                  "known_correct_acceptance": 0.0,
                                  "known_coverage": 0.0,
                                  "fit_rows": 43,
                                  "stopping_rows": 8},
        "direct_supervised_six_way": {"known_raw_top1": 0.14,
                                      "known_correct_acceptance": 0.02,
                                      "known_coverage": 0.06,
                                      "unknown_false_groups": 1,
                                      "unknown_groups": 30,
                                      "fit_rows": 43,
                                      "stopping_rows": 8},
        "future_final_used": False,
    }
    state["milestones"]["M5"]["geometry_and_policy_host_tests"] = {
        "status": "passed_host_only",
        "shared_fomo_geometry_contract": True,
        "geometry_seed_cases": 10,
        "policy_and_geometry_unittest_methods": 17,
        "float32_decision_cases": 25,
        "hardware_exercised": False,
    }
    state["pending_actions"] = [
        "The v2 actual-frame FOMO probe is prepared but was not rerun in this continuation; on next approved board access, diagnose the allocation failure without treating the prior failure as no-face, and preserve a separate result namespace.",
        "After FOMO allocates, verify the same physical frame through detector coordinates, corrected clamped ROI, RGB565 conversion, exact model tensor and embedding before claiming device parity.",
        "Corrected host SFace and MobileNet routes were tested; no small candidate is useful. Do not start a ten-root comparison unless a new one-root route first demonstrates train-only fitting and useful development behavior.",
        "Host geometry/controller tests pass, but exercise A-sense, periodic A-full wake, camera sleep recovery, fixed/low cadence, model reuse/swap, standalone output, and the full camera-to-decision path on the Nicla after the exact input/runtime bundle is available.",
        "Keep the old instrument-only Windows transfer unchanged; create a new transfer only after current implementation and its commands are ready for review.",
        "Windows: identify FNIRSI meter/HID and Nano; verify supply topology and alternate-power isolation; run ten independently started randomized commissioning blocks; preserve raw failures and return a verified archive.",
        "Future research: collect permissioned independent target-camera sessions if wearable generalization is required; the 30-person unknown sample cannot certify the unchanged 1% upper-95% target.",
        "No cold-start, battery-only, motion/wearable, final-confirmatory, or publication claim is made.",
    ]
    amend = {
        "revision": "corrected-camera-model-and-policy-tests-v1",
        "updated_utc": NOW,
        "model_findings": "Aligned host SFace is useful conditionally; corrected learned pilots remain not useful.",
        "test_receipt": "work/project-research-v5/runs/controller_policy_v1/TEST_RESULT.json",
        "board_reset_performed": False,
        "ready_for_final_build": False,
    }
    revisions = state.setdefault("current_repair_amendments", [])
    if not any(row.get("revision") == amend["revision"] for row in revisions):
        revisions.append(amend)
    else:
        revisions[:] = [amend if row.get("revision") == amend["revision"] else row for row in revisions]
    save(state_path, state)
    print(json.dumps({"claims": len(claims["claims"]), "registry_rows": len(rows), "ledger_branches": len(ledger["branches"])}))


if __name__ == "__main__":
    main()
