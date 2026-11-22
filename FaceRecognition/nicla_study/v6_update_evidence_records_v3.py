"""Append the volatile full-path prototype and live-screen check to v6 records."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(rel, kind=None):
    path = ROOT / rel
    row = {"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)}
    if kind:
        row["kind"] = kind
    return row


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def add_fact(rows, fact, rel, unit):
    if not any(row.get("path") == rel and row.get("fact") == fact for row in rows):
        rows.append({"fact": fact, "path": rel, "unit": unit})


def add_hashed(rows, rel, kind):
    row = record(rel, kind)
    if not any(existing.get("path") == rel for existing in rows):
        rows.append(row)


def main():
    aggregate_path = "work/project-research-v5/runs/camera_identity_prototype_v1/DEVICE_AGGREGATE.json"
    gallery_path = "work/project-research-v5/runs/camera_prototype_gallery_v1/RESULT.json"
    release_path = "work/project-research-v5/runs/camera_identity_prototype_v1/BOARD_RELEASE_INSPECTION.json"
    alignment_path = "work/project-research-v5/CAMERA_LIVE_SCENE_ALIGNMENT_V1.md"
    device_aggregate = load(ROOT / aggregate_path)
    gallery = load(ROOT / gallery_path)
    assert device_aggregate["aggregate"]["identity_decision"] == "UNCERTAIN"
    assert device_aggregate["aggregate"]["identity_acceptances"] == 0
    assert device_aggregate["runs"]["device_run_multiface"]["fomo_accepted_cluster_count"] == 2

    # Update branch facts while preserving historical rows and dispositions.
    branch_path = STUDY / "BRANCH_LEDGER.json"
    branches_doc = load(branch_path)
    branches_doc["updated_utc"] = NOW
    branches = {row["id"]: row for row in branches_doc["branches"]}
    camera = branches["camera"]
    camera["execution_status"] = "one_frame_input_encoder_parity_and_volatile_decision_pass; live_target_alignment_unverified"
    camera["additional_single_frame_attempts"] = 2
    camera["current_corrected_geometry_independent_blocks"] = 0
    camera["device_parity_exposures"] = 1
    camera["live_target_snapshot_count"] = 2
    camera["live_target_visible_in_camera"] = False
    add_fact(camera["evidence"],
             "A volatile raw-REPL run completed one development frame through FOMO, corrected ROI, INT8 tensor, fused48, six-way head, and open-set decision; it returned UNCERTAIN in 1.900s with zero identities accepted. A separate two-cluster frame was rejected before identity inference.",
             aggregate_path, "two software-reset attempts; one full single-face call, not an independent optical block")
    add_fact(camera["evidence"],
             "The Codex side-window target rendered, but two fresh camera JPEGs, including one after a 100-vmin enlargement, showed a different dark screen scene with no target face. No recognition denominator was updated.",
             "work/project-research-v5/private/camera_live_scene_v3/RECEIPT_100VMIN.json",
             "two frames from the existing live stream; scene-alignment check only")

    identity = branches["identity"]
    identity["execution_status"] = "ten_root_host_head_and_frozen_gallery_complete_no_useful_identity_route; live_device_head_uncertain"
    identity["corrected_v6_head_campaign_roots"] = 10
    identity["frozen_gallery_gradient_roots"] = 0
    identity["live_device_identity_acceptances"] = 0
    add_fact(identity["evidence"],
             "The 10-root host-only 112px MobileFaceNet plus 774-parameter head averaged 49.6% raw top-1 and 21.0% correct acceptance at the zero-calibration-false-group point; it is not useful or device-deployable.",
             "work/project-research-v5/CAMERA_CORRECTED_MODEL_DIAGNOSTICS_V2.md",
             "10 learned roots; 50 known development rows and 30 unknown groups per root")
    add_fact(identity["evidence"],
             "The frozen corrected 48px cosine gallery reached 0/50 correct accepted at calibration cap zero (4% coverage, two wrong known accepts) and 4/50 at cap three (22% coverage, seven wrong accepts, four false unknown groups).",
             gallery_path, "fixed train-group gallery; no gradient roots; 50 known and 30 unknown development groups")

    prototype = branches["prototype"]
    prototype["execution_status"] = "volatile_full_camera_to_decision_ran_uncertain_no_identity_accepted; no_standalone_release"
    prototype["live_single_face_elapsed_ms"] = device_aggregate["aggregate"]["elapsed_ms"]
    prototype["live_identity_decision"] = device_aggregate["aggregate"]["identity_decision"]
    prototype["live_identity_acceptances"] = device_aggregate["aggregate"]["identity_acceptances"]
    add_fact(prototype["evidence"],
             "Actual sensor→FOMO→corrected crop→INT8→fused encoder→six-way head→decision ran once on a single-face development frame and returned UNCERTAIN; no name was accepted, no board files or startup paths changed, and the full call took 1.900s.",
             aggregate_path, "one volatile software-reset execution; not startup, cold power, or p95")
    add_fact(prototype["evidence"],
             "Read-only inspection confirmed the active pointer and startup hash, but runtime_config.json returned EIO; deployment, recovery, and rollback were not attempted.",
             release_path, "read-only inspection; no board writes")

    handoff = branches["handoff"]
    handoff["execution_status"] = "historic_v5_instrument_transfer_verified_current_v2_v3_not_packaged"
    add_fact(handoff["evidence"],
             "The corrected v2/v3 work remains outside the unchanged historical v5 instrument-only archive; no current Windows package is claimed.",
             "work/project-research-v5/READY_FOR_WINDOWS.json",
             "current readiness binding; historic archive handled separately")
    save(branch_path, branches_doc)

    # Amend claims without deleting earlier evidence or changing the historic claim rows.
    claims_path = STUDY / "CLAIM_EVIDENCE.json"
    claims_doc = load(claims_path)
    claims_doc["generated_utc"] = NOW
    claims = {row["id"]: row for row in claims_doc["claims"]}
    identity_claim = claims["camera_matched_identity_usefulness"]
    identity_claim["claim"] = (
        "The historical v5 camera-recognition results are invalid because camera-space FOMO centers were multiplied by 2.5. "
        "Corrected routes remain non-useful: the 10-root 112px host head averaged 49.6% raw top-1 and 21.0% correct acceptance at the zero-calibration-false-group point; the corrected 48px gallery accepted 0/50 correctly there. "
        "A volatile Nicla camera-to-decision call returned UNCERTAIN once and accepted no identity."
    )
    identity_claim["status"] = "corrected_routes_not_useful_live_device_uncertain"
    metrics = identity_claim.setdefault("metrics", {})
    metrics.update({
        "historical_v5_metrics_valid_for_camera_recognition": False,
        "face_pretrained_112px_head_roots": 10,
        "face_pretrained_112px_raw_top1_mean": 0.496,
        "face_pretrained_112px_correct_acceptance_cap0_mean": 0.21,
        "face_pretrained_112px_coverage_cap0_mean": 0.22,
        "face_pretrained_112px_false_unknown_groups_per_root": 1,
        "corrected_48px_gallery_cap0": {"raw_top1": 0.24, "correct_acceptance": 0.0, "coverage": 0.04, "wrong_known_accepts": 2, "unknown_false_groups": 0},
        "corrected_48px_gallery_cap3": {"correct_acceptance": 0.08, "coverage": 0.22, "wrong_known_accepts": 7, "unknown_false_groups": 4},
        "live_device_single_face_decision": "UNCERTAIN",
        "live_device_single_face_elapsed_ms": 1900.105,
        "live_device_identity_acceptances": 0,
        "live_device_multicluster_decision": "UNCERTAIN_before_identity_inference",
        "future_final_used": False,
    })
    for rel, kind in [
        (aggregate_path, "live_device_aggregate"),
        (gallery_path, "corrected_gallery_metrics"),
        ("work/project-research-v5/CAMERA_IDENTITY_PROTOTYPE_V1.md", "human_findings"),
        ("work/project-research-v5/CAMERA_PROTOTYPE_GALLERY_V1.md", "human_findings"),
    ]:
        add_hashed(identity_claim["evidence"], rel, kind)

    unknown = claims["unknown_identity_precision"]
    unknown["claim"] = (
        "The unchanged 1% one-sided 95% unknown-group target remains unestablished. Corrected host routes either have no useful known acceptance or false development unknown groups; zero events among 30 still has a 9.503% upper95 bound."
    )
    unknown["status"] = "not_established_corrected_v6"
    add_hashed(unknown["evidence"], gallery_path, "corrected_gallery_metrics")

    new_claims = [
        {
            "id": "live_nicla_camera_to_decision_v3",
            "claim": "A volatile on-device camera→FOMO→corrected ROI→48px INT8→fused encoder→six-way head→decision path completed once on a single-face development frame and returned UNCERTAIN in 1.900s; a multi-cluster frame was rejected before identity inference.",
            "status": "observed_limited_scope_not_qualified",
            "future_final_used": False,
            "metrics": {"single_face_decision": "UNCERTAIN", "identity_acceptances": 0, "elapsed_ms": 1900.105, "multicluster_decision": "UNCERTAIN", "software_reset_blocks": 2, "cold_power_cycles": 0},
            "evidence": [record(aggregate_path, "device_aggregate"), record("work/project-research-v5/runs/camera_identity_prototype_v1/BOARD_RELEASE_INSPECTION.json", "read_only_board_inspection")],
        },
        {
            "id": "physical_screen_target_alignment_v1",
            "claim": "The Codex target page rendered the expected development photo, but the live Nicla camera stream showed a different dark screen scene in two fresh 240×240 JPEGs, including after enlarging the target. Physical target visibility is unverified; these frames are not identity or detector misses.",
            "status": "user_alignment_action_pending",
            "future_final_used": False,
            "metrics": {"camera_snapshots": 2, "target_visible_in_camera": False, "new_optical_rows": 0, "target_render_physical_onset_verified": False},
            "evidence": [record("work/project-research-v5/private/camera_live_scene_v3/RECEIPT.json", "private_scene_receipt"), record("work/project-research-v5/private/camera_live_scene_v3/RECEIPT_100VMIN.json", "private_scene_receipt"), record(alignment_path, "human_findings")],
        },
    ]
    present_claims = {row["id"] for row in claims_doc["claims"]}
    claims_doc["claims"].extend(row for row in new_claims if row["id"] not in present_claims)
    save(claims_path, claims_doc)

    # Append versioned experiment summaries only; raw probabilities and camera bytes stay private.
    registry_path = STUDY / "EXPERIMENT_REGISTRY.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {row.get("experiment_id") for row in registry}
    additions = [
        {
            "schema": "nicla-v6-experiment-1", "experiment_id": "v6_live_camera_to_decision_probe_v1",
            "status": "completed_uncertain_no_identity_accepted", "future_final_used": False,
            "scope": "Volatile raw-REPL actual camera→FOMO→corrected 120px ROI→48px INT8→fused48→six-way head→open-set decision. One single-face development call returned UNCERTAIN; a separate two-cluster frame was conservatively rejected.",
            "software_reset_blocks": 2, "cold_power_cycles": 0, "development_single_face_calls": 1,
            "identity_acceptances": 0, "elapsed_ms": 1900.105, "response_p95_claimed": False,
            "board_files_written": False, "startup_path_changed": False,
            "evidence": [record(aggregate_path, "device_aggregate"), record("work/project-research-v5/runs/camera_identity_prototype_v1/DEVICE_PROGRAM.py", "executed_program")],
        },
        {
            "schema": "nicla-v6-experiment-1", "experiment_id": "v6_camera_prototype_gallery_v1",
            "status": "host_gallery_not_useful", "future_final_used": False, "gradient_roots": 0,
            "scope": "Corrected 48px cosine prototype gallery fit from train/other_train source groups; thresholds from unknown_calibration; evaluation on all 50 known and 30 unknown development groups.",
            "caps": {cap: {"raw_known_top1": gallery["development_operating_curve"][cap]["known_raw_six_way_top1_correct"] / 50.0, "correct_known_accepts": gallery["development_operating_curve"][cap]["known_correct_accepted"], "coverage": gallery["development_operating_curve"][cap]["known_coverage_all_rows"], "wrong_known_accepts": gallery["development_operating_curve"][cap]["known_wrong_accepted"], "unknown_false_groups": gallery["development_operating_curve"][cap]["unknown_false_accepted_groups"]} for cap in ("0", "3")},
            "unknown_calibration_usable_groups": gallery["calibration"]["unknown_groups_usable"], "unknown_calibration_groups": gallery["calibration"]["unknown_source_groups_total"],
            "evidence": [record(gallery_path, "result_aggregate"), record("work/project-research-v5/CAMERA_PROTOTYPE_GALLERY_V1.md", "human_findings")],
        },
        {
            "schema": "nicla-v6-experiment-1", "experiment_id": "v6_live_scene_alignment_check_v1",
            "status": "target_not_visible_in_camera_frame", "future_final_used": False,
            "scope": "Read-only Nicla preview snapshots compared with the Codex in-app development target; target rendered at 70 and 100 vmin, but fresh 240×240 sensor images showed a different dark screen scene.",
            "camera_snapshots": 2, "new_optical_identity_rows": 0, "physical_onset_verified": False,
            "serial_owner_pid": 1596, "target_server_pid": 1915,
            "evidence": [record("work/project-research-v5/private/camera_live_scene_v3/RECEIPT.json", "private_camera_snapshot"), record("work/project-research-v5/private/camera_live_scene_v3/RECEIPT_100VMIN.json", "private_camera_snapshot"), record(alignment_path, "human_findings")],
        },
    ]
    registry.extend(row for row in additions if row["experiment_id"] not in ids)
    registry_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in registry), encoding="utf-8")

    # Preserve the old readiness snapshot and ZIP while rebinding every current path.
    readiness_path = STUDY / "READY_FOR_WINDOWS.json"
    readiness = load(readiness_path)
    readiness["previous_generated_by_v3"] = "FaceRecognition/nicla_study/v6_readiness.py"
    readiness["generated_by"] = "FaceRecognition/nicla_study/v6_update_evidence_records_v3.py"
    readiness["generated_utc"] = NOW
    readiness["status"] = "mac_research_open_identity_candidate_not_useful_physical_target_alignment_unverified"
    readiness["ready_for_windows"] = False
    readiness["ready_for_windows_scope"] = ["The historical v5 ZIP remains instrument-only; the corrected current branch is not packaged."]
    readiness["ready_for_final_build"] = False
    readiness["recognition_qualified"] = False
    readiness["candidate_application_deployable"] = False
    readiness["final_role_opened"] = False
    readiness["live_camera_to_decision_v3"] = {
        "device_aggregate": aggregate_path,
        "single_face_decision": "UNCERTAIN",
        "identity_acceptances": 0,
        "elapsed_ms": 1900.105,
        "multicluster_policy": "UNCERTAIN_before_identity_inference",
        "software_reset_blocks": 2,
        "board_files_written": False,
        "startup_path_changed": False,
        "full_response_p95_claimed": False,
    }
    readiness["camera_target_alignment_v1"] = {
        "status": "not_verified_camera_saw_a_different_screen_scene",
        "browser_target_token": "b56ac1de32d8311b4bc8ffa6",
        "browser_render_physical_onset_verified": False,
        "camera_snapshots": 2,
        "camera_target_visible": False,
        "new_identity_rows": 0,
        "user_action_pending": True,
        "evidence": alignment_path,
    }
    readiness["corrected_host_model_routes"]["frozen_48px_cosine_gallery"] = {
        "gradient_roots": 0,
        "cap0_correct_acceptance": 0.0,
        "cap0_coverage": 0.04,
        "cap0_wrong_known_accepts": 2,
        "cap0_unknown_false_groups": 0,
        "cap3_correct_acceptance": 0.08,
        "cap3_coverage": 0.22,
        "cap3_wrong_known_accepts": 7,
        "cap3_unknown_false_groups": 4,
        "unknown_calibration_usable_groups": 24,
        "unknown_calibration_groups": 30,
        "pilot_useful": False,
    }
    readiness["open_gates"] = [
        "The live Nicla camera did not show the target photo rendered in the Codex side window; verify physical target visibility before another optical row.",
        "No useful corrected small identity model exists. The ten-root 112px host head has low coverage; the frozen 48px gallery accepts no known correctly at its zero-false-calibration-group point; the one actual Nicla call returned UNCERTAIN.",
        "The volatile one-shot is not a standalone app. Identity LED legend, board controller arms, full lifecycle, deploy/readback/recovery/rollback, startup ACKs, model reuse/swap soak, and bounded wake recovery remain unqualified.",
        "The active release pointer is known, but reading its runtime_config.json returned EIO. No deploy, recovery, rollback, or startup-path change was attempted.",
        "Ten GC2145 sleep/wake repetitions were MicroPython soft resets only. No cold power cycle, battery-only operation, independent setup, power/energy, or full-response p95 claim exists.",
        "The corrected current branch is not packaged for Windows. The old v5 archive remains instrument-only; Windows meter/Nano identification and physical commissioning remain pending.",
        "Thirty unknown identities cannot establish the unchanged 1% one-sided 95% risk target; zero errors still has a 9.503% upper bound. Future-final data remain sealed.",
    ]
    additional_bindings = [
        ("FaceRecognition/nicla_study/v6_live_camera_identity_probe_v1.py", "source"),
        ("FaceRecognition/nicla_study/v6_camera_prototype_gallery_v1.py", "source"),
        ("FaceRecognition/nicla_study/v6_update_evidence_records_v3.py", "source"),
        (aggregate_path, "private_current_repair_aggregate"),
        (release_path, "private_current_repair_read_only_board_receipt"),
        (gallery_path, "private_current_repair_aggregate"),
        ("work/project-research-v5/runs/camera_identity_prototype_v1/DEVICE_PROGRAM.py", "generated_volatile_device_program"),
        ("work/project-research-v5/runs/camera_identity_prototype_v1/DEVICE_PROGRAM.json", "generated_program_receipt"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/SELECTED_STIMULUS_STATE.json", "private_stimulus_receipt"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/STIMULUS_CHANGE.json", "private_stimulus_receipt"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/STIMULUS_RESTORE_RECEIPT.json", "private_stimulus_receipt"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/device_run/RESULT.json", "private_device_run_result"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/device_run/block_00.stdout.bin", "private_device_transcript"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/device_run/block_00.stderr.bin", "private_device_transcript"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/device_run_multiface/RESULT.json", "private_device_run_result"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/device_run_multiface/block_00.stdout.bin", "private_device_transcript"),
        ("work/project-research-v5/private/camera_identity_prototype_v1/device_run_multiface/block_00.stderr.bin", "private_device_transcript"),
        ("work/project-research-v5/private/camera_prototype_gallery_v1/PREDICTIONS.jsonl", "private_gallery_predictions"),
        ("work/project-research-v5/private/camera_prototype_gallery_v1/prototypes.npz", "private_gallery_prototypes"),
        ("work/project-research-v5/CAMERA_IDENTITY_PROTOTYPE_V1.md", "human_findings"),
        ("work/project-research-v5/CAMERA_PROTOTYPE_GALLERY_V1.md", "human_findings"),
        (alignment_path, "human_findings"),
        ("work/project-research-v5/private/camera_live_scene_v3/RECEIPT.json", "private_camera_scene_receipt"),
        ("work/project-research-v5/private/camera_live_scene_v3/RECEIPT_100VMIN.json", "private_camera_scene_receipt"),
        ("work/project-research-v5/private/camera_live_scene_v3/snapshot.jpg", "private_camera_snapshot"),
        ("work/project-research-v5/private/camera_live_scene_v3/snapshot_100vmin.jpg", "private_camera_snapshot"),
        ("work/project-research-v5/runs/optical/CONTROL.json", "development_target_control"),
        ("work/project-research-v5/runs/optical/RENDER_EVENTS.jsonl", "browser_render_receipts"),
        ("work/project-research-v5/STATUS.md", "status"),
        ("work/project-research-v5/MAC_FINDINGS.md", "findings"),
        ("work/project-research-v5/FINAL_BUILD_GAPS.md", "readiness_gaps"),
        ("work/project-research-v5/DECISIONS.md", "decisions"),
        ("work/project-research-v5/02_WINDOWS_RESEARCH_AND_RETURN.md", "windows_entrypoint"),
        ("work/project-research-v5/CURRENT_SESSION_CHECKPOINT_V4.md", "resume_checkpoint"),
        ("work/project-research-v5/BRANCH_LEDGER.json", "branch_ledger"),
        ("work/project-research-v5/CLAIM_EVIDENCE.json", "claim_ledger"),
        ("work/project-research-v5/EXPERIMENT_REGISTRY.jsonl", "experiment_registry"),
    ]
    binding_rows = {row["path"]: row for row in readiness.get("bindings", [])}
    for rel, kind in additional_bindings:
        binding_rows.setdefault(rel, {"path": rel, "kind": kind})
    for rel, row in binding_rows.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("Readiness binding missing: " + rel)
        row["bytes"] = path.stat().st_size
        row["sha256"] = digest(path)
    readiness["bindings"] = list(binding_rows.values())
    readiness["current_repair_v3"] = {
        "revision": "volatile_camera_to_decision_and_live_scene_alignment_v3",
        "updated_utc": NOW,
        "device_probe": record(aggregate_path, "device_aggregate"),
        "gallery": record(gallery_path, "corrected_gallery_metrics"),
        "screen_alignment_report": record(alignment_path, "human_findings"),
        "ready_for_windows": False,
        "ready_for_final_build": False,
        "future_final_used": False,
    }

    # The historic transfer remains an immutable, independently verified instrument archive.
    archive = STUDY / "handoffs/nicla-camera-research-v5-windows.zip"
    sidecar = STUDY / "handoffs/nicla-camera-research-v5-windows.zip.sha256"
    assert digest(archive) == sidecar.read_text(encoding="utf-8").split()[0]
    readiness["historical_v5_transfer"]["archive_sha256"] = digest(archive)
    readiness["historical_v5_transfer"]["readiness_snapshot_sha256"] = digest(STUDY / "READY_FOR_WINDOWS_V5_TRANSFER_SNAPSHOT.json")
    readiness["historical_v5_transfer"]["status"] = "unchanged_instrument_only_archive_hash_verified"
    save(readiness_path, readiness)

    # Reopen research explicitly, preserving the prior v1/v2 amendment history.
    state_path = STUDY / "STATE.json"
    state = load(state_path)
    state["updated_utc"] = NOW
    status = state.setdefault("completion_status", {})
    status["mac_research"] = "open_corrected_identity_candidate_and_physical_target_alignment"
    status["recognition_application"] = "not_qualified_no_useful_small_candidate"
    status["ready_for_final_build"] = False
    status["windows_transfer"] = "historical_v5_instrument_transfer_verified_corrected_v2_v3_not_packaged"
    state["execution_status"] = "corrected_v3_camera_to_decision_ran_uncertain_physical_target_alignment_unverified_active_release_read_eio"
    state["ready_for_windows"] = False
    state["ready_for_final_build"] = False
    state["live_preview"] = {
        "exclusive_serial_owner": False,
        "serial_owner_pid": 1596,
        "serial_owner_script": "FaceRecognition/tools/research_v5/live_camera.py",
        "serial_port": "/dev/cu.usbmodem3270395F35322",
        "target_display_pid": 1915,
        "target_display_script": "FaceRecognition/tools/research_v5/optical_display.py",
        "status": "target tab renders original development photo at 100 vmin; two live camera snapshots showed a different dark screen scene; user alignment action pending",
        "url": "http://127.0.0.1:8767/",
        "physical_onset_verified": False,
    }
    state["current_v3_camera_decision"] = {
        "aggregate": aggregate_path,
        "single_face_decision": "UNCERTAIN",
        "identity_acceptances": 0,
        "elapsed_ms": 1900.105,
        "multicluster_policy": "UNCERTAIN_before_identity_inference",
        "board_files_written": False,
        "startup_path_changed": False,
        "software_reset_blocks": 2,
        "cold_power_cycles": 0,
        "future_final_used": False,
    }
    state["current_v3_screen_alignment"] = {
        "report": alignment_path,
        "camera_target_visible": False,
        "camera_snapshots": 2,
        "new_optical_identity_rows": 0,
        "user_action_pending": True,
    }
    state["pending_actions"] = [
        "Confirm the live development target is physically visible in the Nicla camera frame; the last two fresh sensor JPEGs showed a different screen scene.",
        "After target alignment, capture and inspect one frame before resuming known/OTHER/blank/off-center optical evaluation; do not count a mismatched scene as a detector miss.",
        "No current identity route is useful enough to promote. Preserve the actual one-shot UNCERTAIN decision and the negative ten-root head and 48px gallery results.",
        "The pointed active release runtime_config.json returned EIO during a read-only query. Keep the release and backups untouched until safe recovery evidence exists.",
        "Standalone app, identity LED legend, board controller arms, positive deploy/readback/recovery/rollback, startup ACKs, model reuse/swap soak, and ten cold-start blocks remain unqualified.",
        "The corrected branch is not in a current Windows transfer. Keep the historic instrument ZIP unchanged and Windows Nano/meter physical visit pending.",
        "No cold power, battery-only, independent scene/setup, energy, full-response p95, or final-confirmatory claim is supported; future-final remains sealed.",
    ]
    amend = {
        "revision": "volatile-camera-to-decision-and-screen-alignment-v3",
        "updated_utc": NOW,
        "device_aggregate": aggregate_path,
        "gallery_result": gallery_path,
        "single_face_decision": "UNCERTAIN",
        "identity_acceptances": 0,
        "elapsed_ms": 1900.105,
        "multicluster_decision": "UNCERTAIN_before_identity_inference",
        "screen_target_visible_in_camera": False,
        "cold_power_cycles": 0,
        "ready_for_windows": False,
        "ready_for_final_build": False,
    }
    amendments = state.setdefault("current_repair_amendments", [])
    amendments[:] = [row for row in amendments if row.get("revision") != amend["revision"]]
    amendments.append(amend)
    save(state_path, state)
    print(json.dumps({"claims": len(claims_doc["claims"]), "registry_rows": len(registry), "ledger_branches": len(branches_doc["branches"]), "readiness_bindings": len(readiness["bindings"]), "ready_for_windows": readiness["ready_for_windows"], "ready_for_final_build": readiness["ready_for_final_build"]}, sort_keys=True))


if __name__ == "__main__":
    main()
