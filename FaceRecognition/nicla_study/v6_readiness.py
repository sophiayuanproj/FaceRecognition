"""Derive current Mac repair readiness without overwriting the v5 handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
OUTPUT = STUDY / "READY_FOR_WINDOWS.json"
V5_SNAPSHOT = STUDY / "READY_FOR_WINDOWS_V5_TRANSFER_SNAPSHOT.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path, kind: str) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build() -> dict:
    if OUTPUT.exists() and not V5_SNAPSHOT.exists():
        previous = read_json(OUTPUT)
        if previous.get("schema") != "nicla-v5-ready-for-windows-1":
            raise ValueError("Refusing to snapshot an unknown readiness schema")
        shutil.copyfile(OUTPUT, V5_SNAPSHOT)
    if not V5_SNAPSHOT.exists():
        raise FileNotFoundError("The immutable v5 readiness snapshot is required")

    old = read_json(V5_SNAPSHOT)
    input_path = STUDY / "runs/camera_corrected_input_v1/INPUTS.json"
    decision_path = STUDY / "runs/camera_corrected_input_v1/DECISION_CONTRACT.json"
    pilot_path = STUDY / "runs/camera_corrected_recognition_v1/roi_120_camera_px/crop_120/56001/COMPLETE.json"
    learning_path = STUDY / "runs/camera_corrected_recognition_v1/LEARNING_CONTROL.json"
    teacher_path = STUDY / "runs/camera_corrected_teacher_v1/RESULT.json"
    linear_student_path = STUDY / "runs/camera_corrected_teacher_student_v1/RESULT.json"
    face_encoder_path = STUDY / "runs/camera_corrected_face_encoder_v1/RESULT.json"
    deep_student_path = STUDY / "runs/camera_corrected_deep_student_v1/RESULT.json"
    supervised_path = STUDY / "runs/camera_supervised_mobilenet_v1/RESULT.json"
    controller_test_path = STUDY / "runs/controller_policy_v1/TEST_RESULT.json"
    model_findings_path = STUDY / "CAMERA_CORRECTED_MODEL_DIAGNOSTICS_V1.md"
    parity_dir = STUDY / "reports/M6/corrected_saved_frame_parity_v1/device_run_codex_side_target_K01"
    device_attempt_path = parity_dir / "RESULT.json"
    frame_record_path = parity_dir / "FRAME_RECOVERY.json"
    stimulus_path = parity_dir / "STIMULUS.json"
    contact_sheet_path = STUDY / "private/camera_corrected_input_v1/CROP_GEOMETRY_CONTACT_SHEET.png"
    contact_report_path = STUDY / "private/camera_corrected_input_v1/CROP_GEOMETRY_CONTACT_SHEET.json"

    inputs = read_json(input_path)
    decision = read_json(decision_path)
    pilot = read_json(pilot_path)
    learning = read_json(learning_path)
    teacher = read_json(teacher_path)
    linear_student = read_json(linear_student_path)
    face_encoder = read_json(face_encoder_path)
    deep_student = read_json(deep_student_path)
    supervised = read_json(supervised_path)
    controller_tests = read_json(controller_test_path)
    attempt = read_json(device_attempt_path)
    frame_record = read_json(frame_record_path)
    stimulus = read_json(stimulus_path)
    program_path = parity_dir.parent / "DEVICE_PROGRAM.py"
    capture = (ROOT / frame_record["raw_frame_path"]).resolve()

    expected_counts = {"fomo_single_cluster": 142, "multiple_clusters": 28, "no_face": 85}
    if inputs.get("capture_rows") != 255 or inputs.get("quality_counts") != expected_counts:
        raise ValueError("Corrected camera input row counts differ from the frozen 255-frame evidence")
    if inputs.get("coordinate_contract") != "OpenMV FOMO postprocessed rectangles are camera_xywh_px; no 96-to-240 center scaling":
        raise ValueError("Corrected FOMO coordinate contract is missing")
    if decision.get("passed") != 25 or decision.get("failed") != 0:
        raise ValueError("The current float32/open-set decision contract is not fully recorded")
    if pilot.get("actual_gradient_training") is not True or pilot.get("future_final_used") is not False:
        raise ValueError("Corrected v6 pilot is incomplete or has invalid data-role provenance")
    if learning.get("finite_gradients") is not True or learning.get("class_balanced") is not True:
        raise ValueError("Corrected v6 learning-control evidence is incomplete")
    for report in (teacher, linear_student, face_encoder, deep_student, supervised):
        if report.get("future_final_used") is not False or report.get("ready_for_final_build") is not False:
            raise ValueError("A corrected camera diagnostic has invalid future-final or readiness provenance")
    if (deep_student.get("training", {}).get("fit_rows") != 43 or
            deep_student.get("training", {}).get("internal_train_role_group_stopping_rows") != 8 or
            deep_student.get("training", {}).get("train_only_control", {}).get("fit_stop_source_group_overlap") != 0):
        raise ValueError("Teacher-student pilot lacks its group-disjoint train-only stopping control")
    if (supervised.get("training", {}).get("fit_rows") != 43 or
            supervised.get("training", {}).get("stopping_rows") != 8 or
            supervised.get("training", {}).get("fit_stop_group_overlap") != 0):
        raise ValueError("Six-way classifier pilot lacks its group-disjoint train-only stopping control")
    if controller_tests.get("status") != "passed" or controller_tests.get("hardware_exercised") is not False:
        raise ValueError("Mac geometry/controller test receipt is missing or overstates hardware scope")
    if controller_tests.get("unittest", {}).get("test_methods") != 17 or \
            controller_tests.get("decision_contract", {}).get("cases") != 25:
        raise ValueError("Mac geometry/controller test counts differ from the recorded passing suite")
    if frame_record.get("frame_bytes") != 240 * 240 * 2 or sha256(capture) != frame_record.get("frame_sha256"):
        raise ValueError("Recovered physical camera frame failed size/hash verification")
    if stimulus.get("source", {}).get("role") != "development" or stimulus.get("source", {}).get("label") != "K01":
        raise ValueError("The actual camera stimulus is not a frozen development K01 source")
    first_attempt = attempt[0] if attempt else {}
    if first_attempt.get("status") != "error" or "Failed to allocate tensors" not in first_attempt.get("stderr", ""):
        raise ValueError("The actual device FOMO allocation failure is not present in the preserved run")
    if sha256(program_path) != first_attempt.get("program_sha256"):
        raise ValueError("The failed v1 device program does not match its preserved execution hash")
    retry_source = (ROOT / "FaceRecognition/nicla_study/v6_actual_tensor_parity.py").read_text()
    if "memoryview(data)" not in retry_source or "+2047" not in retry_source or "--output" not in retry_source:
        raise ValueError("The versioned v2 probe generator is missing bounded streaming/output safeguards")

    archive = STUDY / "handoffs/nicla-camera-research-v5-windows.zip"
    sidecar = STUDY / "handoffs/nicla-camera-research-v5-windows.zip.sha256"
    archive_digest = sha256(archive)
    sidecar_digest = sidecar.read_text().split()[0]
    if archive_digest != sidecar_digest:
        raise ValueError("The prior immutable instrument handoff no longer matches its SHA-256 sidecar")

    source_paths = [
        ROOT / "FaceRecognition/firmware/camera_geometry.py",
        ROOT / "FaceRecognition/firmware/identity_contracts.py",
        ROOT / "FaceRecognition/firmware/open_set_decision.py",
        ROOT / "FaceRecognition/firmware/policy_controller.py",
        ROOT / "FaceRecognition/firmware/recognize_controlled.py",
        ROOT / "FaceRecognition/firmware/main.py",
        ROOT / "FaceRecognition/firmware/research_lab.py",
        ROOT / "FaceRecognition/nicla_study/v5_camera_task48.py",
        ROOT / "FaceRecognition/nicla_study/v5_evidence.py",
        ROOT / "FaceRecognition/nicla_study/v5_task_classifier.py",
        ROOT / "FaceRecognition/nicla_study/v6_actual_tensor_parity.py",
        ROOT / "FaceRecognition/nicla_study/v6_camera_crop_features48.py",
        ROOT / "FaceRecognition/nicla_study/v6_camera_task48.py",
        ROOT / "FaceRecognition/nicla_study/v6_crop_qc.py",
        ROOT / "FaceRecognition/nicla_study/v6_learning_control.py",
        ROOT / "FaceRecognition/nicla_study/v6_sface_camera_diagnostic.py",
        ROOT / "FaceRecognition/nicla_study/v6_linear_teacher_distill.py",
        ROOT / "FaceRecognition/nicla_study/v6_face_encoder_camera_diagnostic.py",
        ROOT / "FaceRecognition/nicla_study/v6_camera_mobilenet_distill.py",
        ROOT / "FaceRecognition/nicla_study/v6_camera_supervised_mobilenet.py",
        ROOT / "FaceRecognition/nicla_study/v6_update_evidence_records.py",
        ROOT / "FaceRecognition/nicla_study/tests/test_camera_geometry_v6.py",
        ROOT / "FaceRecognition/nicla_study/tests/test_policy_controller_v6.py",
        ROOT / "FaceRecognition/nicla_study/v6_readiness.py",
    ]
    evidence_paths = [
        input_path,
        decision_path,
        pilot_path,
        learning_path,
        contact_report_path,
        contact_sheet_path,
        device_attempt_path,
        frame_record_path,
        stimulus_path,
        program_path,
        capture,
        teacher_path,
        STUDY / "private/camera_corrected_teacher_v1/RECORDS.jsonl",
        STUDY / "private/camera_corrected_teacher_v1/sface_features.npz",
        linear_student_path,
        STUDY / "private/camera_corrected_teacher_student_v1/linear_projection.npz",
        STUDY / "private/camera_corrected_teacher_student_v1/PREDICTIONS.jsonl",
        face_encoder_path,
        STUDY / "private/camera_corrected_face_encoder_v1/FLOAT_RECORDS.jsonl",
        STUDY / "private/camera_corrected_face_encoder_v1/INT8_RECORDS.jsonl",
        deep_student_path,
        STUDY / "private/camera_corrected_deep_student_v1/mobilenet_student_weights.npz",
        STUDY / "private/camera_corrected_deep_student_v1/PREDICTIONS.jsonl",
        supervised_path,
        STUDY / "private/camera_supervised_mobilenet_v1/mobilenet_classifier_weights.npz",
        STUDY / "private/camera_supervised_mobilenet_v1/PREDICTIONS.jsonl",
        controller_test_path,
        model_findings_path,
        ROOT / "FaceRecognition/firmware/runtime_config.json",
        ROOT / "FaceRecognition/firmware/policy_config.json",
    ]
    sources = [binding(path, "source") for path in source_paths]
    evidence = [binding(path, "private_current_repair_evidence") for path in evidence_paths]
    metrics = pilot["results"]
    record = {
        "schema": "nicla-v6-current-mac-repair-readiness-1",
        "generated_by": "FaceRecognition/nicla_study/v6_readiness.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "mac_research_reopened_corrected_camera_path_incomplete",
        "ready_for_windows": False,
        "ready_for_windows_scope": [],
        "recognition_qualified": False,
        "candidate_application_deployable": False,
        "ready_for_final_build": False,
        "final_role_opened": False,
        "historical_v5_transfer": {
            "status": "previous_instrument_only_transfer_remains_hash_verified",
            "readiness_snapshot": V5_SNAPSHOT.relative_to(ROOT).as_posix(),
            "readiness_snapshot_sha256": sha256(V5_SNAPSHOT),
            "archive": archive.relative_to(ROOT).as_posix(),
            "archive_sha256": archive_digest,
            "qa_receipt": "work/project-research-v5/handoffs/TRANSFER_QA.json",
            "scope": "Windows instrument commissioning only; it excludes this corrected research branch",
        },
        "legacy_v5_crop_campaign": {
            "status": "preserved_but_invalid_for_recognition_conclusions",
            "reason": "FOMO postprocessed rectangles were already in camera-pixel coordinates; the historical crop code multiplied centers by 2.5.",
            "historical_roots": 10,
            "historical_runs": 40,
            "do_not_reuse_features_or_candidate_selection": True,
        },
        "corrected_v6_inputs": {
            "capture_rows": inputs["capture_rows"],
            "quality_counts": inputs["quality_counts"],
            "roi_side_camera_px": inputs["roi_side_camera_px"],
            "tensor_hwc": inputs["encoder_tensor_hwc"],
            "seeded_geometry_fixtures_passed": inputs["geometry_fixture_passed"],
            "visual_qc_contact_sheet_present": contact_sheet_path.is_file(),
            "board_input_parity": "pending_actual_fomo_allocation_failed_before_device_tensor_generation",
            "future_final_used": False,
        },
        "corrected_v6_pilot": {
            "training_roots": [pilot["seed"]],
            "known_development_n": metrics["known_development_n"],
            "known_raw_top1": metrics["known_raw_top1_full"],
            "known_correct_acceptance": metrics["known_correct_acceptance"],
            "known_coverage": metrics["known_coverage"],
            "known_wrong_accepted": metrics["known_wrong_accepted"],
            "unknown_development_false_groups": metrics["unknown_development_group_risk"]["false_groups"],
            "unknown_calibration_false_groups": metrics["unknown_calibration_group_risk"]["false_groups"],
            "unknown_zero_event_upper95_30": metrics["unknown_calibration_group_risk"]["zero_event_upper95"],
            "pilot_useful": False,
            "ten_root_campaign_started": False,
            "future_final_used": False,
        },
        "corrected_host_model_routes": {
            "teacher_diagnostic": {
                "status": "host_only_reference_not_deployable",
                "aligned_known_correct_acceptance": teacher["routes"]["yu_net_aligned_embedding"]["known_correct_acceptance_all_rows"],
                "aligned_known_coverage": teacher["routes"]["yu_net_aligned_embedding"]["known_coverage_all_rows"],
                "aligned_unknown_false_groups": teacher["routes"]["yu_net_aligned_embedding"]["unknown_false_accepted_groups"],
                "unknown_groups": teacher["routes"]["yu_net_aligned_embedding"]["unknown_development_groups"],
                "fomo_valid_known_rows": teacher["routes"]["yu_net_aligned_embedding"]["known_fomo_valid_rows"],
            },
            "deep_teacher_student": {
                "seed": deep_student["training"]["seed"],
                "fit_rows": deep_student["training"]["fit_rows"],
                "stopping_rows": deep_student["training"]["internal_train_role_group_stopping_rows"],
                "known_raw_top1": deep_student["evaluation"]["known_raw_top1_all_rows"],
                "known_correct_acceptance": deep_student["evaluation"]["known_correct_acceptance_all_rows"],
                "known_coverage": deep_student["evaluation"]["known_coverage_all_rows"],
                "unknown_false_groups": deep_student["evaluation"]["unknown_false_accepted_groups"],
                "pilot_useful": False,
            },
            "direct_supervised_six_way": {
                "seed": supervised["training"]["seed"],
                "fit_rows": supervised["training"]["fit_rows"],
                "stopping_rows": supervised["training"]["stopping_rows"],
                "known_raw_top1": supervised["evaluation"]["known_raw_top1_all_rows"],
                "known_correct_acceptance": supervised["evaluation"]["known_correct_acceptance_all_rows"],
                "known_coverage": supervised["evaluation"]["known_coverage_all_rows"],
                "unknown_false_groups": supervised["evaluation"]["unknown_false_accepted_groups"],
                "pilot_useful": False,
            },
            "linear_projection": {
                "base": linear_student["base_gallery_route"],
                "teacher_projected": linear_student["teacher_projected_gallery_route"],
                "pilot_useful": False,
            },
            "mobilefacenet_float_int8": {
                "float": face_encoder["models"]["float"]["metrics"],
                "int8": face_encoder["models"]["int8"]["metrics"],
                "pilot_useful": False,
            },
            "ten_root_learned_comparisons": 0,
            "future_final_used": False,
        },
        "mac_geometry_policy_tests": {
            "test_receipt": controller_test_path.relative_to(ROOT).as_posix(),
            "unittest_methods_passed": controller_tests["unittest"]["test_methods"],
            "seeded_geometry_cases_passed": controller_tests["unittest"]["seeded_geometry_cases"],
            "decision_contract_cases_passed": controller_tests["decision_contract"]["cases"],
            "hardware_exercised": False,
        },
        "latest_actual_camera_attempt": {
            "stimulus_label": stimulus["source"]["label"],
            "display_surface": stimulus["display_surface"],
            "frame_sha256": frame_record["frame_sha256"],
            "frame_size_bytes": frame_record["frame_bytes"],
            "device_status": frame_record["capture_status"],
            "device_error": first_attempt["stderr"].splitlines()[-1],
            "detector_input_or_embedding_parity": "not_observed",
            "cold_start_claimed": False,
        },
        "probe_failure_diagnosis": {
            "observed_v1_raw_dump_chunk_bytes": 768,
            "known_working_optical_lab_chunk_bytes": 2048,
            "likely_issue": "The failed v1 probe emitted 150 small base64 chunks before constructing FOMO; the existing optical_lab streams 2048-byte memoryview chunks and runs the same ROM model.",
            "cause_status": "plausible_probe_memory_or_fragmentation_hypothesis_not_proven_no_heap_telemetry_in_failed_transcript",
            "versioned_v2_source_prepared": True,
            "v2_hardware_retry_performed": False,
        },
        "bindings": sources + evidence,
        "open_gates": [
            "Diagnose the FOMO allocation failure. A versioned v2 probe now uses the existing optical_lab 2048-byte memoryview streaming pattern and a separate result path; do not call the hypothesis confirmed until the device run verifies it.",
            "Run exact FOMO rectangle, corrected ROI, RGB48/int8 input, and encoder-output parity on the same physical frame.",
            "A corrected one-root camera recognition pilot was not useful; establish a useful single-root route before any ten-root comparison.",
            "Host geometry/controller tests pass but no recognition route is useful; qualify detector, identity, OTHER/UNCERTAIN/NO_FACE, controller, release, startup, reset, and recovery only on an exact board input path.",
            "Rebuild a fresh transfer before sending current repair sources to Windows; the prior immutable archive does not contain this branch.",
        ],
        "all_v6_evidence_future_final_used": False,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    return record


if __name__ == "__main__":
    value = build()
    print(json.dumps({
        "status": value["status"],
        "ready_for_windows": value["ready_for_windows"],
        "ready_for_final_build": value["ready_for_final_build"],
        "v6_known_correct_acceptance": value["corrected_v6_pilot"]["known_correct_acceptance"],
        "device_status": value["latest_actual_camera_attempt"]["device_status"],
    }, sort_keys=True))
