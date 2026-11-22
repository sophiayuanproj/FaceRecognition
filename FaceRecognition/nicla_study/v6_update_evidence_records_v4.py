"""Append verified single-scene camera evidence without rewriting v1 history."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(rel: str, kind: str) -> dict:
    path = ROOT / rel
    return {"path": rel, "kind": kind, "bytes": path.stat().st_size, "sha256": digest(path)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    result_rel = "work/project-research-v5/runs/camera_identity_survey_v2/RESULT.json"
    frame_rel = "work/project-research-v5/private/camera_identity_survey_v1/K02_after_2_5s.jpg"
    raw_rel = "work/project-research-v5/private/camera_identity_survey_v2/device_K02_dwell2p5/RESULT.json"
    report_rel = "work/project-research-v5/CAMERA_IDENTITY_SURVEY_V2.md"
    correction_rel = "work/project-research-v5/private/camera_identity_survey_v1/ALIGNMENT_CORRECTION_V1.json"
    review_rel = "work/project-research-v5/private/camera_identity_survey_v2/K02_VISUAL_REVIEW.json"
    result = load(ROOT / result_rel)
    device_runs = load(ROOT / raw_rel)
    assert result["sample_count"] == 1
    assert result["sample"]["label"] == "K02"
    assert result["sample"]["decision"] == "NO_FACE"
    assert result["sample"]["fomo_accepted_clusters"] == 0
    assert device_runs[0]["status"] == "executed"
    assert device_runs[0]["program_sha256"] == result["candidate"]["probe_program_sha256"]
    assert device_runs[0]["records"][0]["board_files_written"] is False

    branch_path = STUDY / "BRANCH_LEDGER.json"
    branches_doc = load(branch_path)
    branches_doc["updated_utc"] = NOW
    branches = {row["id"]: row for row in branches_doc["branches"]}
    camera = branches["camera"]
    camera["execution_status"] = "k02_camera_target_verified; one_linked_device_probe_no_face; broader_scene_survey_pending"
    camera["live_target_visible_in_camera"] = True
    camera["verified_target_preview_count_v2"] = 1
    camera["verified_device_probe_count_v2"] = 1
    camera["rapid_v1_label_metrics_usable"] = False
    camera["current_corrected_geometry_independent_blocks"] = 0
    camera.setdefault("evidence", []).extend([
        {"fact": "A fresh preview after a 2.5-second dwell visibly matched the K02 Colin Powell side-window target; an immediately following volatile board run with the target unchanged returned NO_FACE with zero FOMO clusters in 1506.597ms.", "path": result_rel, "unit": "one visually linked development sample; not an independent block"},
        {"fact": "Rapid v1 survey labels are one scene behind the selected display control and are excluded from per-label recognition metrics; raw transcripts are preserved.", "path": correction_rel, "unit": "alignment correction for seven one-shot diagnostics"},
    ])
    prototype = branches["prototype"]
    prototype["execution_status"] = "prior_single_face_uncertain_plus_verified_k02_no_face; no_standalone_release"
    prototype["latest_verified_scene_decision"] = "NO_FACE"
    prototype["latest_verified_scene_label"] = "K02"
    prototype["latest_verified_scene_elapsed_ms"] = 1506.597
    prototype.setdefault("evidence", []).append({
        "fact": "One K02 target was verified in the live preview, then the unchanged scene produced zero FOMO clusters and NO_FACE on the Nicla; the board captured frame was not retained for visual review.",
        "path": result_rel,
        "unit": "one volatile software-reset execution; no identity inference, startup, cold power, or p95 claim",
    })
    identity = branches["identity"]
    identity["rapid_v1_per_label_metrics_usable"] = False
    identity.setdefault("evidence", []).append({
        "fact": "The rapid v1 optical survey is unscorable by expected label because the side-window stimulus advanced faster than the physical camera scene; its raw outputs remain unlabelled diagnostics.",
        "path": correction_rel,
        "unit": "label-linkage correction; no recognition accuracy estimate",
    })
    handoff = branches["handoff"]
    handoff["execution_status"] = "historic_v5_instrument_transfer_verified_current_v2_v4_not_packaged"
    branch_path.write_text(json.dumps(branches_doc, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    claims_path = STUDY / "CLAIM_EVIDENCE.json"
    claims_doc = load(claims_path)
    claims_doc["generated_utc"] = NOW
    claims = {row["id"]: row for row in claims_doc["claims"]}
    align_v1 = claims.get("physical_screen_target_alignment_v1")
    if align_v1:
        align_v1["status"] = "historical_mismatch_superseded_for_k02_by_v2"
        align_v1["superseded_by"] = "physical_screen_target_alignment_v2"
    usefulness = claims["camera_matched_identity_usefulness"]
    usefulness["status"] = "corrected_routes_not_useful_v1_label_alignment_invalid_v2_k02_no_face"
    usefulness["claim"] = (
        "The historical v5 recognition result is invalid because FOMO camera-pixel centers were multiplied by 2.5. "
        "Corrected host routes remain non-useful, and the prior single-face Nicla call was UNCERTAIN. "
        "A later K02 sample was visually matched to the side-window target but produced NO_FACE; it therefore contributes a detector miss, not an identity prediction. "
        "The rapid v1 live survey label assignments are unverified and excluded from per-label metrics."
    )
    usefulness.setdefault("evidence", []).extend([
        record(result_rel, "verified_scene_device_result"),
        record(correction_rel, "survey_label_alignment_correction"),
    ])
    new_claims = [
        {
            "id": "physical_screen_target_alignment_v2",
            "claim": "After the user aligned the setup, a fresh 240x240 Nicla preview after a 2.5-second dwell visibly matched the K02 Colin Powell target. This verifies one preview-to-display pairing only.",
            "status": "one_development_target_visually_verified",
            "future_final_used": False,
            "metrics": {"label": "K02", "preview_frames": 1, "target_visible": True, "dwell_s": 2.5, "raw_repl_frame_archived": False},
            "evidence": [record(frame_rel, "private_camera_preview"), record(review_rel, "visual_review"), record(report_rel, "human_findings")],
        },
        {
            "id": "live_nicla_camera_to_decision_v4",
            "claim": "With a visually matched K02 development target left unchanged after the preview, one volatile Nicla run returned NO_FACE with zero FOMO clusters in 1506.597ms. It did not run identity inference and is not a qualified application result.",
            "status": "observed_single_verified_scene_detector_miss_not_qualified",
            "future_final_used": False,
            "metrics": {"label": "K02", "decision": "NO_FACE", "fomo_raw_clusters": 0, "fomo_accepted_clusters": 0, "elapsed_ms": 1506.597, "cold_power_cycles": 0, "board_files_written": False, "startup_path_changed": False},
            "evidence": [record(result_rel, "device_result"), record(raw_rel, "raw_device_summary"), record(report_rel, "human_findings")],
        },
    ]
    present = {row["id"] for row in claims_doc["claims"]}
    claims_doc["claims"].extend(row for row in new_claims if row["id"] not in present)
    save(claims_path, claims_doc)

    registry_path = STUDY / "EXPERIMENT_REGISTRY.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    experiment_id = "v6_live_camera_identity_survey_v2"
    if experiment_id not in {row.get("experiment_id") for row in registry}:
        registry.append({
            "schema": "nicla-v6-experiment-1",
            "experiment_id": experiment_id,
            "status": "one_verified_k02_scene_no_face; v1_label_mapping_invalid",
            "future_final_used": False,
            "scope": "One K02 development target verified in the live Nicla preview after a 2.5-second dwell; a volatile raw-REPL run with the target unchanged returned NO_FACE before identity inference.",
            "samples": 1,
            "independent_physical_blocks": 0,
            "cold_power_cycles": 0,
            "decision": "NO_FACE",
            "fomo_accepted_clusters": 0,
            "elapsed_ms": 1506.597,
            "board_files_written": False,
            "startup_path_changed": False,
            "label_alignment_correction": correction_rel,
            "evidence": [record(result_rel, "result"), record(report_rel, "human_findings"), record(review_rel, "private_visual_review")],
        })
    registry_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in registry), encoding="utf-8")

    readiness_path = STUDY / "READY_FOR_WINDOWS.json"
    readiness = load(readiness_path)
    readiness["generated_utc"] = NOW
    readiness["generated_by"] = "FaceRecognition/nicla_study/v6_update_evidence_records_v4.py"
    readiness["status"] = "mac_research_open_verified_k02_detector_miss_no_useful_identity_application"
    readiness["ready_for_windows"] = False
    readiness["ready_for_final_build"] = False
    readiness["recognition_qualified"] = False
    readiness["candidate_application_deployable"] = False
    readiness["camera_target_alignment_v2"] = {
        "status": "one_development_target_verified_in_preview",
        "label": "K02",
        "browser_target_token": "469571835963515a4171d1f4",
        "preview_visible": True,
        "preview_dwell_s": 2.5,
        "linked_device_decision": "NO_FACE",
        "fomo_accepted_clusters": 0,
        "new_identity_rows": 0,
        "user_action_pending": False,
        "evidence": report_rel,
    }
    readiness["live_camera_to_decision_v4"] = {
        "result": result_rel,
        "verified_label": "K02",
        "decision": "NO_FACE",
        "identity_inference_ran": False,
        "elapsed_ms": 1506.597,
        "independent_physical_blocks": 0,
        "cold_power_cycles": 0,
        "board_files_written": False,
        "startup_path_changed": False,
    }
    readiness["open_gates"] = [
        "The rapid v1 on-device survey has invalid per-label linkage because the physical sensor scene lagged the selected side-window stimulus; preserve raw outputs but exclude v1 label metrics.",
        "Only one target (K02) has a verified preview-to-display pairing; that Nicla run returned NO_FACE. Repeat a dwell-and-frame-verified survey across K01-K05, OTHER, blank and scene controls before drawing detector or identity conclusions.",
        "No useful corrected small identity model exists. Host head/gallery routes remain weak; the prior correctly linked K02 device run never reached identity inference.",
        "The volatile one-shot is not a standalone app. Identity LED legend, board controller arms, deploy/readback/recovery/rollback, startup ACKs, model reuse/swap soak, and bounded wake recovery remain unqualified.",
        "The active release pointer is known, but reading its runtime_config.json returned EIO. No deploy, recovery, rollback, or startup-path change was attempted.",
        "Ten GC2145 sleep/wake repetitions were MicroPython soft resets only. No cold power cycle, battery-only operation, independent setup, power/energy, or full-response p95 claim exists.",
        "The corrected current branch is not packaged for Windows. The old v5 archive remains instrument-only; Windows meter/Nano identification and physical commissioning remain pending.",
        "Thirty unknown identities cannot establish the unchanged 1% one-sided 95% risk target; zero errors still has a 9.503% upper bound. Future-final data remain sealed.",
    ]
    readiness["current_repair_v4"] = {
        "revision": "verified_k02_camera_pairing_and_v1_label_alignment_correction_v4",
        "updated_utc": NOW,
        "result": record(result_rel, "device_result"),
        "alignment_correction": record(correction_rel, "survey_correction"),
        "ready_for_windows": False,
        "ready_for_final_build": False,
        "recognition_qualified": False,
        "future_final_used": False,
    }

    state_path = STUDY / "STATE.json"
    state = load(state_path)
    state["updated_utc"] = NOW
    state["completion_status"]["mac_research"] = "open_verified_k02_no_face_model_and_application_integration"
    state["completion_status"]["recognition_application"] = "not_qualified_no_face_on_verified_k02_scene"
    state["completion_status"]["ready_for_final_build"] = False
    state["completion_status"]["windows_transfer"] = "historical_v5_instrument_transfer_verified_current_v2_v4_not_packaged"
    state["execution_status"] = "verified_k02_preview_pairing_one_volatile_device_no_face_v1_survey_labels_invalid_application_integration_open"
    state["ready_for_windows"] = False
    state["ready_for_final_build"] = False
    state["live_preview"] = {
        "exclusive_serial_owner": True,
        "serial_owner_pid": int((STUDY / "private/live_camera.pid").read_text().strip()),
        "serial_owner_script": "FaceRecognition/tools/research_v5/live_camera.py",
        "serial_port": "/dev/cu.usbmodem3270395F35322",
        "target_display_pid": int((STUDY / "runs/optical/display.pid").read_text().strip()),
        "target_display_script": "FaceRecognition/tools/research_v5/optical_display.py",
        "status": "K02 target visible in a fresh camera preview after a 2.5-second dwell; subsequent volatile device probe returned NO_FACE; preview restored after probe",
        "url": "http://127.0.0.1:8767/",
        "preview_url": "http://127.0.0.1:8766/",
        "physical_onset_verified": True,
    }
    state["current_v4_verified_camera_attempt"] = {
        "result": result_rel,
        "label": "K02",
        "decision": "NO_FACE",
        "fomo_accepted_clusters": 0,
        "elapsed_ms": 1506.597,
        "future_final_used": False,
        "board_files_written": False,
        "startup_path_changed": False,
    }
    state["rapid_v1_camera_survey_label_alignment"] = "invalid_for_per_label_metrics; preserve raw rows as unlabelled diagnostics"
    state["pending_actions"] = [
        "The rapid v1 optical survey label linkage is invalid; use stable target dwell and a fresh camera-preview match before each newly labelled board row.",
        "Only one correctly linked K02 board row exists and it returned NO_FACE; establish whether detector misses persist across K01-K05, OTHER and controls.",
        "Build and qualify a useful five-person/OTHER/UNCERTAIN pipeline before considering deployment or promotion.",
        "Standalone app, LED legend, board controller arms, supported deploy/readback/recovery/rollback, startup ACKs, model reuse/swap soak and bounded wake recovery remain unqualified.",
        "The pointed active release runtime_config.json returned EIO during a read-only query; preserve the release and backups until safe recovery evidence exists.",
        "The corrected branch is not in a current Windows transfer; keep the historical instrument ZIP unchanged and the Windows physical visit pending.",
        "No cold power, battery-only, independent scene/setup, energy, full-response p95, or final-confirmatory claim is supported; future-final remains sealed.",
    ]
    amendment = {
        "revision": "verified-k02-camera-pairing-and-v1-label-correction-v4",
        "updated_utc": NOW,
        "result": result_rel,
        "verified_label": "K02",
        "decision": "NO_FACE",
        "elapsed_ms": 1506.597,
        "rapid_v1_per_label_metrics_usable": False,
        "ready_for_windows": False,
        "ready_for_final_build": False,
    }
    amendments = state.setdefault("current_repair_amendments", [])
    amendments[:] = [row for row in amendments if row.get("revision") != amendment["revision"]]
    amendments.append(amendment)
    save(state_path, state)

    # Rebind every prior readiness path, then add the new private and public evidence.
    binding_rows = {row["path"]: row for row in readiness.get("bindings", [])}
    new_paths = [
        ("FaceRecognition/nicla_study/v6_update_evidence_records_v4.py", "source"),
        ("work/project-research-v5/CAMERA_IDENTITY_SURVEY_V2.md", "human_findings"),
        ("work/project-research-v5/CURRENT_SESSION_CHECKPOINT_V5.md", "resume_checkpoint"),
        ("work/project-research-v5/runs/camera_identity_survey_v2/RESULT.json", "result"),
        ("work/project-research-v5/private/camera_identity_survey_v1/ALIGNMENT_CORRECTION_V1.json", "private_correction"),
        ("work/project-research-v5/private/camera_identity_survey_v2/K02_VISUAL_REVIEW.json", "private_visual_review"),
        ("work/project-research-v5/private/camera_identity_survey_v2/device_K02_dwell2p5/RESULT.json", "private_device_result"),
        ("work/project-research-v5/private/camera_identity_survey_v2/device_K02_dwell2p5/block_00.stdout.bin", "private_device_transcript"),
        ("work/project-research-v5/private/camera_identity_survey_v2/device_K02_dwell2p5/block_00.stderr.bin", "private_device_transcript"),
        ("work/project-research-v5/private/camera_identity_survey_v1/K02_after_2_5s.jpg", "private_camera_snapshot"),
        ("work/project-research-v5/STATUS.md", "status"),
        ("work/project-research-v5/CAMERA_LIVE_SCENE_ALIGNMENT_V1.md", "historical_alignment_with_v2_amendment"),
        ("work/project-research-v5/BRANCH_LEDGER.json", "branch_ledger"),
        ("work/project-research-v5/CLAIM_EVIDENCE.json", "claim_ledger"),
        ("work/project-research-v5/EXPERIMENT_REGISTRY.jsonl", "experiment_registry"),
        ("work/project-research-v5/STATE.json", "state"),
    ]
    for rel, kind in new_paths:
        binding_rows.setdefault(rel, {"path": rel, "kind": kind})
    for rel, row in binding_rows.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("Readiness binding missing: " + rel)
        row["bytes"] = path.stat().st_size
        row["sha256"] = digest(path)
    readiness["bindings"] = list(binding_rows.values())

    archive = STUDY / "handoffs/nicla-camera-research-v5-windows.zip"
    sidecar = STUDY / "handoffs/nicla-camera-research-v5-windows.zip.sha256"
    assert digest(archive) == sidecar.read_text(encoding="utf-8").split()[0]
    readiness["historical_v5_transfer"]["archive_sha256"] = digest(archive)
    readiness["historical_v5_transfer"]["status"] = "unchanged_instrument_only_archive_hash_verified"
    save(readiness_path, readiness)
    print(json.dumps({
        "updated_utc": NOW,
        "readiness_bindings": len(binding_rows),
        "ready_for_windows": readiness["ready_for_windows"],
        "ready_for_final_build": readiness["ready_for_final_build"],
        "recognition_qualified": readiness["recognition_qualified"],
        "archive_sha256": digest(archive),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
