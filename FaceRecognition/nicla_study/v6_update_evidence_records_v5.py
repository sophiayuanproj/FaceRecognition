"""Bind the cursor-clear seven-scene Nicla diagnostic into current records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
RESULT_REL = "work/project-research-v5/runs/camera_identity_survey_v2/VERIFIED_SET_RESULT.json"
PRIVATE_REL = "work/project-research-v5/private/camera_identity_survey_v2/VERIFIED_SET_PRIVATE.json"
REPORT_REL = "work/project-research-v5/CAMERA_IDENTITY_SURVEY_V2_VERIFIED_SET.md"
CORRECTION_REL = "work/project-research-v5/private/camera_identity_survey_v1/ALIGNMENT_CORRECTION_V1.json"
OBSTRUCTION_REL = "work/project-research-v5/runs/camera_identity_survey_v2/INITIAL_K02_OBSTRUCTION_CORRECTION.json"
REVIEW_REL = "work/project-research-v5/private/camera_identity_survey_v2/device_previews/VISUAL_REVIEW.json"
CONTACT_REL = "work/project-research-v5/private/camera_identity_survey_v2/VERIFIED_SET_CONTACT_SHEET.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(rel: str, kind: str) -> dict:
    path = ROOT / rel
    return {"path": rel, "kind": kind, "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def add_fact(rows: list, fact: str, rel: str, unit: str) -> None:
    if not any(row.get("fact") == fact and row.get("path") == rel for row in rows):
        rows.append({"fact": fact, "path": rel, "unit": unit})


def add_evidence(rows: list, rel: str, kind: str) -> None:
    if not any(row.get("path") == rel for row in rows):
        rows.append(record(rel, kind))


def main() -> None:
    result = load(ROOT / RESULT_REL)
    private = load(ROOT / PRIVATE_REL)
    expected = {"K01": ("UNCERTAIN", "K01"), "K02": ("NO_FACE", None), "K03": ("UNCERTAIN", "K04"), "K04": ("UNCERTAIN", "K05"), "K05": ("K03", "K03"), "OTHER": ("OTHER", "OTHER"), "BLANK": ("NO_FACE", None)}
    assert len(private["rows"]) == 7 and len(result["per_condition"]) == 7
    for row in result["per_condition"]:
        assert (row["decision"], row["predicted_class"]) == expected[row["case"]]
        assert row["pointer_over_face"] is False
    assert result["known_metrics_all_rows"]["correct_accepted"] == 0
    assert result["known_metrics_all_rows"]["wrong_known_accepts"] == 1
    assert result["latency_ms"]["p95_claimed"] is False
    assert all(row["device_output"]["board_files_written"] is False for row in private["rows"])

    branch_path = STUDY / "BRANCH_LEDGER.json"
    branches_doc = load(branch_path)
    branches_doc["updated_utc"] = NOW
    branches = {row["id"]: row for row in branches_doc["branches"]}
    camera = branches["camera"]
    camera["execution_status"] = "seven_cursor_clear_scene_to_device_runs_complete; weak_one_root_head; labels_verified_after_dwell"
    camera["live_target_visible_in_camera"] = True
    camera["verified_target_labels_v2"] = ["K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK"]
    camera["verified_device_probe_count_v2"] = 7
    camera["rapid_v1_label_metrics_usable"] = False
    camera["initial_k02_pointer_obstruction_excluded"] = True
    camera["scene_preview_dwell_seconds"] = 2.5
    add_fact(camera.setdefault("evidence", []), "Seven cursor-clear development/control scenes were acknowledged in the side window, held 2.5 seconds, visually matched in fresh Nicla previews, and followed by one volatile device run each. The prior rapid v1 label assignments remain excluded.", RESULT_REL, "seven one-shot rows from one setup; no independent physical blocks")
    add_fact(camera["evidence"], "The initial K02 no-face trial had the pointer over the center of the face; it is preserved but excluded from unobstructed detection conclusions. A clean K02 repeat also returned NO_FACE.", OBSTRUCTION_REL, "one obstructed run plus one clean run; only the clean run is in the seven-row summary")

    identity = branches["identity"]
    identity["execution_status"] = "verified_live_v2_head_not_useful_1_of_5_raw_top1_zero_correct_accepts_one_wrong_accept"
    identity["verified_v2_known_rows"] = 5
    identity["verified_v2_raw_top1_correct"] = 1
    identity["verified_v2_correct_accepted"] = 0
    identity["verified_v2_wrong_known_accepts"] = 1
    identity["verified_v2_accepted_coverage"] = 0.2
    identity["rapid_v1_per_label_metrics_usable"] = False
    add_fact(identity.setdefault("evidence", []), "On the five correctly linked known development scenes, the one-root device head produced 1/5 raw top-1, 0/5 correct accepted, one wrong known accept (K05 as K03), and 20% accepted coverage; K02 was a detector miss.", RESULT_REL, "one volatile board run per known identity; not a learned comparison or rate estimate")

    prototype = branches["prototype"]
    prototype["execution_status"] = "seven_scene_camera_to_decision_diagnostics_complete_no_deployable_application"
    prototype["verified_v2_known_raw_top1"] = "1/5"
    prototype["verified_v2_correct_known_accepts"] = "0/5"
    prototype["verified_v2_wrong_known_accepts"] = 1
    prototype["verified_v2_unknown_decision"] = "OTHER_on_one_group"
    prototype["verified_v2_blank_decision"] = "NO_FACE"
    prototype["verified_v2_latency_median_ms"] = 1881.384
    prototype["verified_v2_latency_p95_claimed"] = False
    add_fact(prototype.setdefault("evidence", []), "The seven cursor-clear raw-REPL calls took 1495.757–1901.214ms (median 1881.384ms); the encoder arena was 243856 bytes after detector use. No application-level memory or p95 claim is supported.", RESULT_REL, "seven non-independent one-shot calls; one physical setup")
    handoff = branches["handoff"]
    handoff["execution_status"] = "historic_v5_instrument_transfer_verified_current_v2_v5_not_packaged"
    save(branch_path, branches_doc)

    claims_path = STUDY / "CLAIM_EVIDENCE.json"
    claims_doc = load(claims_path)
    claims_doc["generated_utc"] = NOW
    claims = {row["id"]: row for row in claims_doc["claims"]}
    usefulness = claims["camera_matched_identity_usefulness"]
    usefulness["status"] = "corrected_routes_not_useful_verified_live_v2_zero_correct_accepts_one_wrong_known_accept"
    usefulness["claim"] = (
        "The historical v5 camera-recognition metrics are invalid because camera-space FOMO centers were multiplied by 2.5. "
        "Corrected host routes and the one-root device head remain non-useful. On five correctly linked cursor-clear known scenes, the live head had 1/5 raw top-1, 0/5 correct accepted, and one wrong known acceptance; K02 returned NO_FACE. "
        "The rapid v1 label mapping and an initial cursor-obstructed K02 run are excluded from clean metrics."
    )
    metrics = usefulness.setdefault("metrics", {})
    metrics["verified_live_v2"] = {
        "known_rows": 5,
        "raw_top1_correct": 1,
        "correct_accepted": 0,
        "wrong_known_accepts": 1,
        "coverage": 0.2,
        "no_face": 1,
        "uncertain": 3,
        "other_groups": 1,
        "other_decision": "OTHER",
        "other_false_known_accept": False,
        "blank_decision": "NO_FACE",
        "latency_ms": {"n": 7, "median": 1881.384, "min": 1495.757, "max": 1901.214, "p95_claimed": False},
        "encoder_arena_bytes_after_fomo": 243856,
        "physical_blocks": 0,
        "future_final_used": False,
    }
    for rel, kind in [(RESULT_REL, "verified_scene_device_result"), (REPORT_REL, "human_findings"), (PRIVATE_REL, "private_predictions"), (CORRECTION_REL, "alignment_correction"), (OBSTRUCTION_REL, "obstructed_trial_correction"), (CONTACT_REL, "private_contact_sheet")]:
        add_evidence(usefulness.setdefault("evidence", []), rel, kind)

    align_v2 = claims.get("physical_screen_target_alignment_v2")
    if align_v2:
        align_v2["claim"] = "Seven cursor-clear development/control targets were each acknowledged in the Codex side window, held for 2.5 seconds, and visibly matched in a fresh Nicla preview before a volatile board probe. This verifies target-to-preview pairing for the seven saved scenes only."
        align_v2["status"] = "seven_development_control_scenes_visually_verified"
        align_v2["metrics"].update({"preview_frames": 7, "labels": ["K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK"], "pointer_over_face": False, "dwell_s": 2.5, "raw_repl_frame_archived": False})
        add_evidence(align_v2["evidence"], REVIEW_REL, "private_visual_review")
        add_evidence(align_v2["evidence"], CONTACT_REL, "private_contact_sheet")

    additions = [
        {
            "id": "physical_scene_sequence_alignment_v3",
            "claim": "The seven v2 camera previews match their selected development/control sources after a browser render acknowledgement and 2.5-second dwell; the rapid v1 label sequence was one target behind. The initial K02 frame also had a pointer over the face center and is excluded from clean detection interpretation.",
            "status": "seven_scenes_verified_rapid_v1_unscorable_initial_k02_obstructed",
            "future_final_used": False,
            "metrics": {"verified_scenes": 7, "pointer_over_face": False, "dwell_s": 2.5, "rapid_v1_per_label_metrics_usable": False},
            "evidence": [record(REVIEW_REL, "private_visual_review"), record(CORRECTION_REL, "v1_alignment_correction"), record(OBSTRUCTION_REL, "obstructed_trial_correction"), record(CONTACT_REL, "private_contact_sheet")],
        },
        {
            "id": "live_nicla_camera_to_decision_v5",
            "claim": "Seven cursor-clear one-shot raw-REPL rows completed on correctly linked K01–K05, OTHER and blank scenes. The current one-root head had 1/5 raw known top-1, 0/5 correct accepts, one wrong known accept, an OTHER decision on one unknown, NO_FACE on blank, and 1.881s median elapsed time.",
            "status": "observed_seven_scene_diagnostic_not_useful_not_qualified",
            "future_final_used": False,
            "metrics": result["known_metrics_all_rows"] | {"unknown_decision": "OTHER", "blank_decision": "NO_FACE", "latency_ms": result["latency_ms"], "encoder_arena_bytes": 243856, "cold_power_cycles": 0, "board_files_written": False, "startup_path_changed": False},
            "evidence": [record(RESULT_REL, "result_aggregate"), record(PRIVATE_REL, "private_rows"), record(REPORT_REL, "human_findings")],
        },
    ]
    present = {row["id"] for row in claims_doc["claims"]}
    claims_doc["claims"].extend(row for row in additions if row["id"] not in present)
    save(claims_path, claims_doc)

    registry_path = STUDY / "EXPERIMENT_REGISTRY.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {row.get("experiment_id") for row in registry}
    if "v6_verified_live_scene_survey_v2" not in ids:
        registry.append({
            "schema": "nicla-v6-experiment-1",
            "experiment_id": "v6_verified_live_scene_survey_v2",
            "status": "seven_scene_one_root_diagnostic_wrong_known_accept_no_deployment",
            "future_final_used": False,
            "scope": "Dwell- and camera-preview-verified K01-K05, OTHER and blank stimuli, each followed by one bounded volatile raw-REPL camera-to-decision run with pointer outside the face.",
            "rows": 7,
            "physical_setups": 1,
            "independent_physical_blocks": 0,
            "known_raw_top1_correct": 1,
            "known_correct_accepted": 0,
            "wrong_known_accepts": 1,
            "known_coverage": 0.2,
            "unknown_decision": "OTHER",
            "blank_decision": "NO_FACE",
            "latency_ms": result["latency_ms"],
            "evidence": [record(RESULT_REL, "aggregate_result"), record(REPORT_REL, "human_findings"), record(CONTACT_REL, "private_contact_sheet"), record(REVIEW_REL, "private_visual_review")],
        })
    registry_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in registry), encoding="utf-8")

    readiness_path = STUDY / "READY_FOR_WINDOWS.json"
    readiness = load(readiness_path)
    readiness["generated_utc"] = NOW
    readiness["generated_by"] = "FaceRecognition/nicla_study/v6_update_evidence_records_v5.py"
    readiness["status"] = "mac_research_open_live_v2_wrong_known_accept_no_useful_application"
    readiness["ready_for_windows"] = False
    readiness["ready_for_final_build"] = False
    readiness["recognition_qualified"] = False
    readiness["candidate_application_deployable"] = False
    readiness["camera_target_alignment_v2"] = {
        "status": "seven_cursor_clear_development_control_scenes_visually_verified",
        "labels": ["K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK"],
        "preview_count": 7,
        "pointer_over_face": False,
        "dwell_s": 2.5,
        "rapid_v1_label_metrics_usable": False,
        "initial_k02_obstructed_and_excluded": True,
        "user_action_pending": False,
        "evidence": REPORT_REL,
    }
    readiness["live_camera_to_decision_v5"] = {
        "result": RESULT_REL,
        "known_raw_top1": "1/5",
        "known_correct_accepts": "0/5",
        "wrong_known_accepts": 1,
        "known_coverage": 0.2,
        "known_no_face": 1,
        "unknown": "OTHER_once",
        "blank": "NO_FACE",
        "latency_ms": result["latency_ms"],
        "encoder_arena_bytes_after_fomo": 243856,
        "independent_physical_blocks": 0,
        "cold_power_cycles": 0,
        "board_files_written": False,
        "startup_path_changed": False,
    }
    readiness["open_gates"] = [
        "The one-root live model is not useful: one raw top-1 of five known scenes, zero correct accepted knowns, and one wrong known acceptance (K05 as K03). Do not deploy or package it.",
        "K02 remains a cleanly verified NO_FACE detector miss; three other known rows abstained, so detection misses and identity errors must remain separate.",
        "Every one-shot call took 1495.757–1901.214ms; no p95 or warm standalone latency claim exists. The 243856-byte arena is encoder-only after detector use, not full-application memory.",
        "No standalone application, LED legend, board controller arms, supported deploy/readback/recovery/rollback, startup ACK blocks, model reuse/swap soak, or bounded wake recovery is qualified.",
        "The active release pointer is known, but reading its runtime_config.json returned EIO. Do not alter the release or recovery backups without safe recovery evidence.",
        "Seven scene rows share one physical setup and are not independent blocks. No cold power cycle, battery-only operation, energy, or final statistical validation exists.",
        "The corrected current branch is not packaged for Windows. The historical v5 ZIP remains instrument-only; Windows meter/Nano identification and physical commissioning remain pending.",
        "Thirty unknown identities cannot establish the unchanged 1% one-sided 95% risk target; zero errors still has a 9.503% upper bound. Future-final data remain sealed.",
    ]
    readiness["current_repair_v5"] = {
        "revision": "cursor_clear_verified_seven_scene_camera_survey_v5",
        "updated_utc": NOW,
        "result": record(RESULT_REL, "aggregate_device_result"),
        "human_findings": record(REPORT_REL, "human_findings"),
        "rapid_v1_labels_valid": False,
        "initial_k02_clean_face_valid": False,
        "ready_for_windows": False,
        "ready_for_final_build": False,
        "recognition_qualified": False,
        "future_final_used": False,
    }

    state_path = STUDY / "STATE.json"
    state = load(state_path)
    state["updated_utc"] = NOW
    state["completion_status"]["mac_research"] = "open_no_useful_live_model_standalone_application_and_lifecycle"
    state["completion_status"]["recognition_application"] = "not_qualified_one_wrong_known_accept_and_zero_correct_accepts"
    state["completion_status"]["ready_for_final_build"] = False
    state["completion_status"]["windows_transfer"] = "historical_v5_instrument_transfer_verified_current_v2_v5_not_packaged"
    state["execution_status"] = "seven_cursor_clear_verified_scene_calls_wrong_known_accept_no_useful_candidate_application_work_open"
    state["ready_for_windows"] = False
    state["ready_for_final_build"] = False
    state["live_preview"] = {
        "exclusive_serial_owner": True,
        "serial_owner_pid": int((STUDY / "private/live_camera.pid").read_text().strip()),
        "serial_owner_script": "FaceRecognition/tools/research_v5/live_camera.py",
        "serial_port": "/dev/cu.usbmodem3270395F35322",
        "target_display_pid": int((STUDY / "runs/optical/display.pid").read_text().strip()),
        "target_display_script": "FaceRecognition/tools/research_v5/optical_display.py",
        "status": "K01 restored at 100 vmin; seven prior target scenes were dwell-verified and preview-matched; latest one-root device survey had one wrong known acceptance",
        "url": "http://127.0.0.1:8767/",
        "preview_url": "http://127.0.0.1:8766/",
        "physical_onset_verified": True,
        "cursor_policy": "target page requests cursor:none; pointer remains in upper-left margin, outside faces",
    }
    state["current_v5_verified_scene_survey"] = {
        "result": RESULT_REL,
        "known_raw_top1": "1/5",
        "known_correct_accepted": "0/5",
        "wrong_known_accepts": 1,
        "known_coverage": 0.2,
        "unknown_decision": "OTHER_once",
        "blank_decision": "NO_FACE",
        "elapsed_median_ms": 1881.384,
        "elapsed_p95_claimed": False,
        "future_final_used": False,
    }
    state["rapid_v1_camera_survey_label_alignment"] = "invalid_for_per_label_metrics; preserve raw rows as unlabelled diagnostics"
    state["pending_actions"] = [
        "Investigate the actual wrong-known K05-to-K03 acceptance and K02 detector miss on corrected inputs; do not deploy the current one-root head.",
        "Implement a useful compact identity route with correct calibration precision and at least ten learned roots for a reported learned comparison; keep detector misses in end-to-end denominators.",
        "Implement a standalone camera/model/controller application with clear LED outputs and measure full application memory and response latency.",
        "Exercise doctor/deploy/readback/configure/demo/recovery/rollback positively only after a viable candidate; the active release runtime_config.json read returned EIO, so preserve it and backups.",
        "Run startup ACK/cold-start, model reuse/swap/soak, wake recovery and actual policy arms with one serial owner; no cold power or energy result exists.",
        "Prepare a new current Windows transfer only after it contains the required source/models/configuration and supported commands; keep the historical v5 ZIP instrument-only.",
        "No final-role data, paper, proposal edit, publication, participant communication, or final statistical claim is authorized by these development diagnostics.",
    ]
    amendment = {
        "revision": "cursor-clear-verified-seven-scene-camera-survey-v5",
        "updated_utc": NOW,
        "result": RESULT_REL,
        "known_raw_top1": "1/5",
        "known_correct_accepted": "0/5",
        "wrong_known_accepts": 1,
        "elapsed_median_ms": 1881.384,
        "rapid_v1_label_metrics_usable": False,
        "ready_for_windows": False,
        "ready_for_final_build": False,
    }
    amendments = state.setdefault("current_repair_amendments", [])
    amendments[:] = [row for row in amendments if row.get("revision") != amendment["revision"]]
    amendments.append(amendment)
    save(state_path, state)

    # Refresh every prior hash binding, then bind all v5 source and data artifacts.
    binding_rows = {row["path"]: row for row in readiness.get("bindings", [])}
    extra = [
        ("FaceRecognition/nicla_study/v6_capture_verified_live_targets_v2.py", "source"),
        ("FaceRecognition/nicla_study/v6_run_verified_live_scene_survey_v2.py", "source"),
        ("FaceRecognition/nicla_study/v6_make_verified_camera_contact_sheet_v1.py", "source"),
        ("FaceRecognition/nicla_study/v6_update_evidence_records_v5.py", "source"),
        ("FaceRecognition/tools/research_v5/optical_display.py", "source"),
        (RESULT_REL, "public_aggregate"), (PRIVATE_REL, "private_predictions"),
        (REPORT_REL, "human_findings"),
        ("work/project-research-v5/CAMERA_IDENTITY_SURVEY_V2.md", "human_findings"),
        ("work/project-research-v5/CURRENT_SESSION_CHECKPOINT_V6.md", "resume_checkpoint"),
        (CORRECTION_REL, "private_label_correction"), (OBSTRUCTION_REL, "obstructed_trial_correction"),
        (REVIEW_REL, "private_visual_review"), (CONTACT_REL, "private_contact_sheet"),
        ("work/project-research-v5/private/camera_identity_survey_v2/scene_frames_cursor_clear/CAPTURE_MANIFEST.json", "private_capture_manifest"),
        ("work/project-research-v5/private/camera_identity_survey_v2/scene_frames_cursor_clear/VISUAL_REVIEW.json", "private_visual_review"),
        ("work/project-research-v5/private/camera_identity_survey_v2/scene_frames_cursor_clear/RESTORE.json", "private_restore_receipt"),
        ("work/project-research-v5/runs/camera_identity_survey_v2/INITIAL_K02_OBSTRUCTION_CORRECTION.json", "correction"),
        ("work/project-research-v5/STATUS.md", "status"),
        ("work/project-research-v5/FINAL_BUILD_GAPS.md", "readiness_gaps"),
        ("work/project-research-v5/MAC_FINDINGS.md", "findings"),
        ("work/project-research-v5/DECISIONS.md", "decisions"),
        ("work/project-research-v5/02_WINDOWS_RESEARCH_AND_RETURN.md", "windows_entrypoint"),
        ("work/project-research-v5/CAMERA_LIVE_SCENE_ALIGNMENT_V1.md", "alignment_history_with_v2_followup"),
        ("work/project-research-v5/BRANCH_LEDGER.json", "branch_ledger"),
        ("work/project-research-v5/CLAIM_EVIDENCE.json", "claim_ledger"),
        ("work/project-research-v5/EXPERIMENT_REGISTRY.jsonl", "experiment_registry"),
        ("work/project-research-v5/STATE.json", "state"),
        ("work/project-research-v5/runs/optical/CONTROL.json", "development_target_control"),
        ("work/project-research-v5/runs/optical/RENDER_EVENTS.jsonl", "render_acknowledgements"),
    ]
    for rel, kind in extra:
        binding_rows.setdefault(rel, {"path": rel, "kind": kind})
    for folder in [
        STUDY / "private/camera_identity_survey_v2/scene_frames_cursor_clear",
        STUDY / "private/camera_identity_survey_v2/device_previews",
        STUDY / "private/camera_identity_survey_v2/device_verified_labelled",
    ]:
        for path in folder.rglob("*"):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                kind = "private_camera_snapshot" if path.suffix.lower() in {".jpg", ".png"} else "private_device_receipt_or_transcript"
                binding_rows.setdefault(rel, {"path": rel, "kind": kind})
    for rel, row in binding_rows.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("Readiness binding missing: " + rel)
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha(path)
    readiness["bindings"] = list(binding_rows.values())

    archive = STUDY / "handoffs/nicla-camera-research-v5-windows.zip"
    sidecar = STUDY / "handoffs/nicla-camera-research-v5-windows.zip.sha256"
    assert sha(archive) == sidecar.read_text(encoding="utf-8").split()[0]
    readiness["historical_v5_transfer"]["archive_sha256"] = sha(archive)
    readiness["historical_v5_transfer"]["status"] = "unchanged_instrument_only_archive_hash_verified"
    save(readiness_path, readiness)
    print(json.dumps({
        "updated_utc": NOW,
        "known_metrics": result["known_metrics_all_rows"],
        "unknown": result["unknown_result"],
        "blank": result["blank_result"],
        "latency_ms": result["latency_ms"],
        "readiness_bindings": len(binding_rows),
        "ready_for_windows": readiness["ready_for_windows"],
        "ready_for_final_build": readiness["ready_for_final_build"],
        "recognition_qualified": readiness["recognition_qualified"],
        "archive_sha256": sha(archive),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
