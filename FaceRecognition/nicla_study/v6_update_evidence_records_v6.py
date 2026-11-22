"""Bind the threshold comparison and actual A-full device control diagnostic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

THRESHOLD_REL = "work/project-research-v5/runs/camera_identity_threshold04_v1/THRESHOLD_COMPARISON.json"
THRESHOLD_PUBLIC_REL = "work/project-research-v5/runs/camera_identity_threshold04_v1/RESULT.json"
THRESHOLD_PRIVATE_REL = "work/project-research-v5/private/camera_identity_threshold04_v1/PRIVATE_RESULT.json"
THRESHOLD_REPORT_REL = "work/project-research-v5/CAMERA_DETECTOR_THRESHOLD_V1.md"
THRESHOLD_VISUAL_REL = "work/project-research-v5/private/camera_identity_threshold04_v1/FRESH_FRAME_VISUAL_REVIEW.json"
THRESHOLD_CONTACT_REL = "work/project-research-v5/private/camera_identity_threshold04_v1/THRESHOLD04_CAMERA_PREVIEW_CONTACT_SHEET.png"
POLICY_REL = "work/project-research-v5/runs/camera_policy_sleep_wake_device_v3/RESULT.json"
POLICY_MANIFEST_REL = "work/project-research-v5/runs/camera_policy_sleep_wake_device_v3/PROGRAM_MANIFEST.json"
POLICY_REPORT_REL = "work/project-research-v5/CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md"
POLICY_REVIEW_REL = "work/project-research-v5/private/camera_policy_sleep_wake_device_v3/K02_CAMERA_VISUAL_REVIEW.json"
TEST_REL = "work/project-research-v5/reports/camera_control_contract_tests_v2/TEST_RESULT.json"
TEST_REPORT_REL = "work/project-research-v5/CAMERA_CONTROL_CONTRACT_TESTS_V2.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(rel: str, kind: str) -> dict:
    path = ROOT / rel
    return {"path": rel, "kind": kind, "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def add_claim_evidence(rows: list, rel: str, kind: str) -> None:
    if not any(row.get("path") == rel for row in rows):
        rows.append(record(rel, kind))


def add_branch_fact(rows: list, fact: str, rel: str, unit: str) -> None:
    if not any(row.get("path") == rel and row.get("fact") == fact for row in rows):
        rows.append({"fact": fact, "path": rel, "unit": unit})


def append_markdown(path: Path, section: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if section.strip() not in old:
        path.write_text(old.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def main() -> None:
    comparison = load(ROOT / THRESHOLD_REL)
    policy = load(ROOT / POLICY_REL)
    tests = load(ROOT / TEST_REL)
    if comparison["known_metrics"]["threshold_0_4"]["correct_accepted"] != 0:
        raise ValueError("unexpected 0.4 known-accept result")
    if comparison["known_metrics"]["threshold_0_4"]["wrong_known_accepts"] != 1:
        raise ValueError("unexpected 0.4 wrong-accept result")
    if policy["device_result"]["identity_model_calls"] != 0 or not policy["device_result"]["wake_frame_verified"]:
        raise ValueError("unexpected A-full device scope/result")
    if tests["focused_suite"]["tests"] != 31 or tests["focused_suite"]["status"] != "pass":
        raise ValueError("focused test receipt is not the expected passing result")

    branch_path = STUDY / "BRANCH_LEDGER.json"
    branches_doc = load(branch_path)
    branches_doc["updated_utc"] = NOW
    branches = {row["id"]: row for row in branches_doc["branches"]}
    camera = branches["camera"]
    camera["threshold04_known_correct_accepts"] = "0/5"
    camera["threshold04_wrong_known_accepts"] = 1
    camera["threshold04_known_single_cluster_detections"] = "3/5"
    camera["threshold04_vs_05_status"] = "0.4_did_not_recover_k02_and_increased_ambiguous_clusters"
    add_branch_fact(camera.setdefault("evidence", []), "A fresh seven-scene comparison of FOMO 0.4 versus 0.5 used the same five known, one unknown and blank conditions with the same one-root identity head; 0.4 kept zero correct accepts and one wrong accept while known single-cluster detections fell from four to three.", THRESHOLD_REL, "seven one-shot pairs on one setup; zero independent blocks")
    detection = branches["detection"]
    detection["threshold04_diagnostic_status"] = "not_selected_for_miss_recovery; additional_ambiguous_clusters"
    add_branch_fact(detection.setdefault("evidence", []), "The pre-existing FOMO 0.4 setting failed to recover K02 and introduced multi-cluster outputs on K03, K05 and OTHER; this small diagnostic does not qualify either threshold.", THRESHOLD_REPORT_REL, "one call per scene per threshold; development diagnostic")
    control = branches["control"]
    control["execution_status"] = "one_actual_a_full_quality_fomo_sleep_wake_block_passed; broader_policy_arms_and_startup_unqualified"
    control["actual_a_full_device_blocks"] = 1
    control["device_sleep_trigger"] = "three_consecutive_quality_passing_no_face_samples"
    control["device_wake_to_frame_ms"] = policy["device_result"]["sleep_and_wake"]["wake_to_usable_frame_ms"]
    control["device_max_gap_ms"] = policy["device_result"]["sleep_and_wake"]["elapsed_since_previous_capture_ms"]
    control["identity_model_calls_in_device_control_test"] = 0
    add_branch_fact(control.setdefault("evidence", []), "The maintained PolicyController A-full code requested sensor sleep after three quality-passing no-face FOMO cycles on a verified K02 screen frame, then recovered a usable 240x240 frame in 659.136ms with a 2297ms sample gap and no timeout.", POLICY_REL, "one volatile software-reset device block, one taped setup, no power measurement")
    prototype = branches["prototype"]
    prototype["standalone_identity_application_status"] = "still_not_implemented_as_a_qualified_release; actual_device_control_demo_is_detector_only"
    add_branch_fact(prototype.setdefault("evidence", []), "The A-full device action exercised capture, quality, FOMO and CSI sleep/wake only; it made zero identity-model calls and changed no board files or startup paths.", POLICY_REPORT_REL, "one volatile block; not startup or deployment qualification")
    handoff = branches["handoff"]
    handoff["execution_status"] = "historical_v5_instrument_transfer_unchanged; current_threshold_and_control_records_remain_mac_local"
    save(branch_path, branches_doc)

    claims_path = STUDY / "CLAIM_EVIDENCE.json"
    claims_doc = load(claims_path)
    claims_doc["generated_utc"] = NOW
    existing = {row["id"] for row in claims_doc["claims"]}
    additions = [
        {
            "id": "fomo_threshold04_same_scene_diagnostic_v1",
            "claim": "The pre-existing FOMO detector threshold 0.4 did not improve the seven-scene one-shot identity diagnostic versus 0.5. Both had 1/5 known raw top-1, 0/5 correct accepted, and one wrong known accept. The 0.4 arm failed to detect K02 and reduced known single-cluster detections from four to three by adding multi-cluster outputs.",
            "status": "threshold04_no_useful_improvement_not_qualified",
            "future_final_used": False,
            "metrics": {"threshold_0_5": comparison["known_metrics"]["threshold_0_5"], "threshold_0_4": comparison["known_metrics"]["threshold_0_4"], "unknown_0_5": comparison["unknown_and_blank"]["threshold_0_5_unknown"], "unknown_0_4": comparison["unknown_and_blank"]["threshold_0_4_unknown"], "latency_ms": comparison["latency_ms"], "rows_per_arm": 7, "physical_setups": 1, "independent_physical_blocks": 0, "p95_claimed": False, "board_files_written": False, "startup_path_changed": False},
            "evidence": [record(THRESHOLD_REL, "threshold_comparison"), record(THRESHOLD_REPORT_REL, "human_findings"), record(THRESHOLD_VISUAL_REL, "private_fresh_frame_visual_review"), record(THRESHOLD_CONTACT_REL, "private_contact_sheet"), record(THRESHOLD_PUBLIC_REL, "device_aggregate"), record(THRESHOLD_PRIVATE_REL, "private_device_predictions")],
        },
        {
            "id": "a_full_policy_device_sleep_wake_v1",
            "claim": "One volatile actual-Nicla A-full detector/control diagnostic saw three quality-passing no-face samples, requested CSI sensor sleep, then returned a usable 240x240 frame after a 1500ms periodic interval. Wake-to-frame time was 659.136ms, total gap 2297ms under the 5000ms bound, and no timeout occurred. It used four FOMO calls and no identity model.",
            "status": "one_block_component_control_verified_not_standalone_or_power_qualified",
            "future_final_used": False,
            "metrics": {"policy_id": "A-full", "no_face_cycles_before_sleep": 3, "fomo_calls": 4, "identity_model_calls": 0, "camera_sleep_requests": 1, "camera_wakes": 1, "wake_frame_geometry": [240, 240], "wake_to_frame_ms": 659.136, "max_sample_gap_ms": 2297, "configured_max_sample_gap_ms": 5000, "timeouts": 0, "physical_setups": 1, "independent_physical_blocks": 0, "cold_power_cycles": 0, "board_files_written": False, "startup_path_changed": False, "power_measured": False},
            "evidence": [record(POLICY_REL, "device_result"), record(POLICY_MANIFEST_REL, "volatile_program_manifest"), record(POLICY_REPORT_REL, "human_findings"), record(POLICY_REVIEW_REL, "private_scene_visual_review")],
        },
        {
            "id": "camera_control_host_contract_tests_v1",
            "claim": "The focused geometry, FOMO peak, policy-controller and binary32 decision suites passed 31/31 tests. Full Nicla-study discovery passed 59/62; three I-matrix tests expect 160 bundles keyed to their current matrix but resolve zero eligible entries, while the staging directory has 480 manifests with a different key set.",
            "status": "camera_control_contracts_pass_i_matrix_fixture_mismatch_preserved",
            "future_final_used": False,
            "metrics": {"focused_tests_passed": 31, "focused_tests_failed": 0, "full_tests": 62, "full_tests_passed": 59, "full_suite_i_matrix_failures": 3, "expected_i_matrix_entries": 160, "staged_manifest_count": 480, "staged_manifest_set_modified": False, "device_sleep_wake_separate": True},
            "evidence": [record(TEST_REL, "test_receipt"), record(TEST_REPORT_REL, "human_findings"), record("work/project-research-v5/reports/camera_control_contract_tests_v2/focused_camera_control.txt", "test_log"), record("work/project-research-v5/reports/camera_control_contract_tests_v2/full_nicla_study_suite.txt", "test_log")],
        },
    ]
    claims_doc["claims"].extend(row for row in additions if row["id"] not in existing)
    save(claims_path, claims_doc)

    registry_path = STUDY / "EXPERIMENT_REGISTRY.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    present = {row.get("experiment_id") for row in registry}
    new_rows = [
        {
            "schema": "nicla-v6-experiment-1", "experiment_id": "v6_camera_fomo_threshold04_live_v1",
            "status": "0.4_threshold_did_not_improve_known_acceptance_added_ambiguous_clusters",
            "future_final_used": False, "scope": "Matched one-shot live Nicla FOMO 0.4 versus 0.5 across K01-K05, OTHER and BLANK using the same one-root head; each scene was render-acknowledged, dwelled 2.5 seconds, previewed and visually checked.",
            "rows_per_arm": 7, "known_correct_accepted_0_4": 0, "known_wrong_accepts_0_4": 1,
            "known_single_cluster_0_5": 4, "known_single_cluster_0_4": 3,
            "unknown_decision_0_5": comparison["unknown_and_blank"]["threshold_0_5_unknown"]["decision"],
            "unknown_decision_0_4": comparison["unknown_and_blank"]["threshold_0_4_unknown"]["decision"],
            "latency_ms": comparison["latency_ms"], "physical_setups": 1, "independent_physical_blocks": 0,
            "board_files_written": False, "startup_path_changed": False, "power_measured": False,
            "evidence": [record(THRESHOLD_REL, "comparison"), record(THRESHOLD_REPORT_REL, "human_findings"), record(THRESHOLD_VISUAL_REL, "private_visual_review")],
        },
        {
            "schema": "nicla-v6-experiment-1", "experiment_id": "v6_a_full_policy_device_sleep_wake_v3",
            "status": "one_block_a_full_no_face_sleep_periodic_wake_verified",
            "future_final_used": False, "scope": "One volatile device run of maintained A-full controller source, actual CSI camera and ROM FOMO on one verified K02 development screen frame; no identity model call.",
            "device_blocks": 1, "cycles_before_sleep": 3, "fomo_calls": 4, "identity_model_calls": 0,
            "wake_to_frame_ms": 659.136, "gap_ms": 2297, "timeout_count": 0,
            "board_files_written": False, "startup_path_changed": False, "power_measured": False,
            "evidence": [record(POLICY_REL, "device_result"), record(POLICY_REPORT_REL, "human_findings"), record(POLICY_REVIEW_REL, "private_visual_review")],
        },
    ]
    registry.extend(row for row in new_rows if row["experiment_id"] not in present)
    registry_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in registry), encoding="utf-8")

    append_markdown(STUDY / "MAC_FINDINGS.md", """
## Current threshold and device-control additions (2026-11-02)

The live FOMO 0.4 comparison did not recover the K02 detector miss or improve correct known acceptance. Both threshold arms produced 1/5 known raw top-1, 0/5 correct accepts and one wrong known accept. The lower threshold added ambiguous clusters and reduced known single-cluster detections from 4/5 to 3/5. See `CAMERA_DETECTOR_THRESHOLD_V1.md`; all preview frames and raw outputs remain private.

The maintained `A-full` controller was exercised once on the actual Nicla in a volatile raw-REPL program. Three quality-passing FOMO no-face cycles triggered CSI sleep; a periodic wake returned a 240×240 frame after 659.136ms with a 2297ms total sample gap and no timeout. Four FOMO calls ran; no identity model ran. This is a component control demonstration, not a standalone application, startup, motion, power or independent-block result. See `CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md`.

Focused geometry, peak-merging, controller and binary32 tests pass 31/31. Full discovery is 59/62 because three I-matrix tests expect 160 keyed bundles but resolve zero eligible entries while 480 different manifests exist in that separate staging tree. The I-matrix artifacts were left untouched. See `CAMERA_CONTROL_CONTRACT_TESTS_V2.md`.

These additions do not change recognition or Windows readiness: no useful identity model, exact standalone release, positive deploy/rollback or current Windows transfer exists.
""")
    append_markdown(STUDY / "STATUS.md", """
## Latest Mac continuation — threshold and live A-full control

The existing FOMO 0.4 threshold did not fix detection or identity on the seven verified screen conditions; it added ambiguous clusters and left one wrong known acceptance. One actual Nicla A-full detector-only control block did exercise quality gating, three-frame no-face hysteresis, sensor sleep, periodic wake and bounded-gap recovery. Focused camera/control contracts pass 31/31; full test discovery has a separate I-matrix fixture/key mismatch (59/62). Mac research remains open, recognition and standalone release remain unqualified, and the current Windows transfer remains unavailable.
""")
    append_markdown(STUDY / "DECISIONS.md", """
## Decision — preserve FOMO 0.5; do not promote 0.4 from this diagnostic

The existing 0.4 setting failed to recover K02 and kept the known false acceptance while increasing multi-cluster outputs. The seven one-shot pairs are too small to qualify a threshold, but provide no basis for replacing the current diagnostic setting. Leave production recognition unqualified and preserve the full camera calibration question.

## Decision — A-full sensor sleep stays disabled in a release

One volatile Nicla action verified the controller trigger and GC2145 wake-to-frame path. It did not measure power or qualify repeated startup, scene, motion or battery behavior. Keep release `CAMERA_SLEEP_ENABLED=False` pending those measurements and do not change the active release.
""")
    append_markdown(STUDY / "FINAL_BUILD_GAPS.md", """
## Current Mac additions — 2026-11-02

- FOMO 0.4 versus 0.5 matched-scene trial: no known-accept improvement, K02 remains NO_FACE, one wrong known acceptance remains, and multi-cluster ambiguity increases. No threshold selected.
- One actual volatile A-full control block: 3 quality-passing no-face cycles, sleep request, periodic wake, usable 240×240 frame at 659.136ms, bounded 2297ms sample gap, zero identity-model calls. No power, startup, motion, cold-cycle, soak or independent-block result.
- Focused geometry/policy/dtype contracts: 31/31 pass. Full suite: 59/62; the other three are I-matrix fixture-key mismatches (expected 160, zero eligible under the current test lookup, 480 different staged manifests left untouched).
- No useful model, standalone identity application, positive deploy/readback/recovery/rollback, exact current Windows package or final readiness.
""")
    append_markdown(STUDY / "CAMERA_IDENTITY_SURVEY_V2.md", """
## Follow-up comparison and controller evidence

The pre-existing 0.4 FOMO setting did not improve the corrected one-root live result; details are in `CAMERA_DETECTOR_THRESHOLD_V1.md`. One actual A-full detector/control run exercised the no-face sleep and periodic wake path; details are in `CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md`. Neither promotes an identity candidate or qualifies a release.
""")
    append_markdown(STUDY / "02_WINDOWS_RESEARCH_AND_RETURN.md", """
## Current Mac correction status — 2026-11-02

The historical v5 ZIP remains unchanged and instrument-only. The corrected Mac branch now includes a seven-scene 0.4-versus-0.5 detector diagnostic and one volatile Nicla A-full sensor sleep/wake control receipt, but still has no useful deployable identity model or standalone release. Do not use the historical package for corrected recognition or policy claims. Windows may continue only the independent FNIRSI/Nano commissioning already described above; a new complete current transfer remains blocked on a useful exact-input model, tested standalone control/recovery and package verification. Keep screen imagery, per-image identity predictions and current physical recovery files on the Mac.

Current Mac reports: `CAMERA_DETECTOR_THRESHOLD_V1.md`, `CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md`, `CAMERA_CONTROL_CONTRACT_TESTS_V2.md`. The separate I-matrix bundle set is not qualified by these diagnostics.
""")

    checkpoint = """# Current session checkpoint V7 — threshold comparison and A-full device control (2026-11-02)

## Verified in this continuation

- Physical device and screen target remain present. The Codex side window is restored to K01 at 100 vmin; fresh camera preview and target services are running. Re-identify process owners and the serial port before another board action.
- FOMO threshold 0.4 was compared against the prior 0.5 diagnostic on the same seven reviewed scenes, with new fresh camera previews and 2.5-second dwell. Both arms had 1/5 known raw top-1, 0/5 correct known accepts and one wrong known accept. At 0.4 the K02 miss remained, known single-cluster detections fell 4/5→3/5, and extra ambiguity appeared on K03, K05 and OTHER. Median latency fell from 1881.384ms to 1509.836ms because ambiguous/no-face paths skip identity inference; all observed calls remained above 1000ms. No p95 claim.
- The maintained `PolicyController('A-full')` source ran in a volatile raw-REPL diagnostic using the actual GC2145 and ROM FOMO on the verified K02 face scene. Three quality-passing no-face cycles triggered sensor sleep. After the 1500ms periodic interval it woke and returned a usable 240×240 frame in 659.136ms; total gap 2297ms, no 5000ms timeout. Four detector calls, zero identity-model calls. The v1 host harness failure and v2 counter-instrumentation miss are preserved; v3 is the corrected result.
- Focused geometry, peak-merging, policy and binary32 tests pass 31/31. Full Nicla-study discovery passes 59/62; three I-matrix preparation tests expect 160 matching entries but resolve zero eligible while the separate staging root has 480 different manifests. That set was not modified.

## Current task status

The coordinate and device tensor parity repairs remain valid. The five-person identity candidate remains unusable; there are zero correct accepts and a wrong known acceptance in both threshold arms. No standalone identity application, LED legend, positive deploy/readback/recovery/rollback, startup ACK, cold power, model reuse/soak, battery, power, new Windows transfer or final qualification exists. The active runtime configuration read previously returned EIO; preserve release and backups. `READY_FOR_WINDOWS.json` and `ready_for_final_build` remain false.

## Next work

1. Continue model work only with an exact valid-input train-only learning control and a justified next route; do not reuse v5 crops or deploy the one-root head.
2. Integrate a genuinely useful model with the versioned standalone controller/LED application, then positively qualify doctor, deploy, readback, recovery and rollback before startup/cold-start actions.
3. Keep sensor sleep disabled in a release until repeated application recovery and power evidence exist. Preserve the one-block action as component evidence only.
4. Prepare a new Windows transfer only when it includes the actual current source/model/config, allowed stimuli and tested control/recovery commands. The historic ZIP remains instrument-only.

Artifacts: `CAMERA_DETECTOR_THRESHOLD_V1.md`, `CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md`, `CAMERA_CONTROL_CONTRACT_TESTS_V2.md`; machine state in `STATE.json`, `BRANCH_LEDGER.json`, `CLAIM_EVIDENCE.json`, `EXPERIMENT_REGISTRY.jsonl`, and `READY_FOR_WINDOWS.json`.
"""
    checkpoint_path = STUDY / "CURRENT_SESSION_CHECKPOINT_V7.md"
    checkpoint_path.write_text(checkpoint, encoding="utf-8")

    readiness_path = STUDY / "READY_FOR_WINDOWS.json"
    readiness = load(readiness_path)
    readiness["generated_utc"] = NOW
    readiness["generated_by"] = "FaceRecognition/nicla_study/v6_update_evidence_records_v6.py"
    readiness["status"] = "mac_research_open_no_useful_identity_candidate_a_full_component_tested"
    readiness["ready_for_windows"] = False
    readiness["ready_for_final_build"] = False
    readiness["recognition_qualified"] = False
    readiness["candidate_application_deployable"] = False
    readiness["current_repair_v6_threshold04"] = {
        "revision": "matched_live_fomo_threshold04_diagnostic_v1",
        "comparison": record(THRESHOLD_REL, "threshold_comparison"),
        "human_findings": record(THRESHOLD_REPORT_REL, "human_findings"),
        "known_correct_accepts": "0/5", "wrong_known_accepts": 1,
        "known_single_cluster_detections_05": "4/5", "known_single_cluster_detections_04": "3/5",
        "ready_for_windows": False, "ready_for_final_build": False, "recognition_qualified": False,
    }
    readiness["current_repair_v7_a_full_device_control"] = {
        "revision": "volatile_a_full_camera_sleep_wake_device_v3",
        "result": record(POLICY_REL, "device_control_result"),
        "human_findings": record(POLICY_REPORT_REL, "human_findings"),
        "fomo_calls": 4, "identity_model_calls": 0, "wake_to_frame_ms": 659.136,
        "sample_gap_ms": 2297, "timeouts": 0, "device_blocks": 1,
        "board_files_written": False, "startup_path_changed": False, "power_measured": False,
        "ready_for_windows": False, "ready_for_final_build": False,
    }
    readiness["camera_control_test_status"] = {
        "receipt": record(TEST_REL, "test_receipt"),
        "human_findings": record(TEST_REPORT_REL, "human_findings"),
        "focused_pass": "31/31", "full_suite": "59/62; three unrelated I-matrix fixture-key mismatches",
    }
    readiness["open_gates"] = [
        "No useful corrected identity model exists. The exact 0.4-versus-0.5 live diagnostic left 0/5 correct known accepts and one wrong known acceptance; do not deploy the one-root head.",
        "K02 remained a cleanly verified FOMO miss at both thresholds; 0.4 added multi-cluster ambiguity. Detector calibration and identity quality remain separate.",
        "One A-full device block verified a detector-only sleep/wake action; no identity, motion, startup, cold-start, independent-block or power qualification exists.",
        "No standalone camera/model/controller release, LED legend, positive deploy/readback/recovery/rollback, startup ACK blocks, model reuse/swap soak or full-application memory measurement is qualified.",
        "The active release runtime_config.json read returned EIO. Preserve the active release and recovery backups until safe recovery evidence exists.",
        "One fixed physical setup only; no cold power cycle, battery-only operation, energy or final statistical validation. Thirty development unknowns cannot establish the 1% one-sided 95% risk target.",
        "The historical v5 ZIP remains instrument-only; current corrected source and diagnostics are Mac-local and not packaged. Windows may conduct independent instrument commissioning only.",
        "Full Nicla-study test discovery includes three I-matrix failures caused by expected bundle-key mismatch (160 expected, zero eligible, 480 staged manifests); the staged set remains untouched. Focused camera/control tests pass 31/31.",
        "No final-role data, proposal edit, publication, participant communication, or final statistical claim was made.",
    ]

    state_path = STUDY / "STATE.json"
    state = load(state_path)
    state["updated_utc"] = NOW
    state["execution_status"] = "threshold04_no_identity_improvement_a_full_sleep_wake_demo_completed_useful_candidate_application_open"
    state["ready_for_windows"] = False
    state["ready_for_final_build"] = False
    state["completion_status"]["mac_research"] = "open_no_useful_live_model_standalone_application_and_lifecycle"
    state["completion_status"]["recognition_application"] = "not_qualified_zero_correct_accepts_one_wrong_known_accept_in_both_live_threshold_diagnostics"
    state["completion_status"]["ready_for_final_build"] = False
    state["completion_status"]["windows_transfer"] = "historical_v5_instrument_transfer_verified_current_corrected_source_not_packaged"
    state["scope"] = "Corrected camera-space crop, host/device tensor parity, corrected host model ladder, seven-scene live identity survey, FOMO 0.4-vs-0.5 diagnostic, host policy contracts and one actual A-full detector-only sleep/wake component block are recorded. No useful identity candidate or qualified standalone release exists; final readiness remains false."
    state["current_v6_threshold04_diagnostic"] = {
        "comparison": THRESHOLD_REL,
        "known_correct_accepts_0_5": "0/5", "known_correct_accepts_0_4": "0/5",
        "wrong_known_accepts_0_5": 1, "wrong_known_accepts_0_4": 1,
        "single_cluster_0_5": "4/5", "single_cluster_0_4": "3/5",
        "known_median_latency_ms_arms": {"0.5": 1881.384, "0.4": 1509.836},
        "p95_claimed": False, "ready_for_windows": False, "ready_for_final_build": False,
    }
    state["current_v7_policy_device_diagnostic"] = {
        "result": POLICY_REL, "policy_id": "A-full", "device_blocks": 1,
        "cycles_before_sleep": 3, "fomo_calls": 4, "identity_model_calls": 0,
        "wake_to_usable_frame_ms": 659.136, "sample_gap_ms": 2297,
        "max_gap_ms": 5000, "timeouts": 0, "board_files_written": False,
        "startup_path_changed": False, "power_measured": False,
    }
    state["camera_control_tests_v2"] = {"receipt": TEST_REL, "focused": "31/31 passed", "full_suite": "59/62; three I-matrix fixture-key mismatches"}
    port_pid_path = STUDY / "private/live_camera.pid"
    display_pid_path = STUDY / "runs/optical/display.pid"
    state["live_preview"] = {
        "exclusive_serial_owner": True,
        "serial_owner_pid": int(port_pid_path.read_text().strip()),
        "serial_owner_script": "FaceRecognition/tools/research_v5/live_camera.py",
        "serial_port": "/dev/cu.usbmodem3270395F35322",
        "target_display_pid": int(display_pid_path.read_text().strip()),
        "target_display_script": "FaceRecognition/tools/research_v5/optical_display.py",
        "status": "K01 restored at 100 vmin; threshold comparison and one A-full detector sleep/wake diagnostic completed; live camera preview running",
        "url": "http://127.0.0.1:8767/", "preview_url": "http://127.0.0.1:8766/",
        "physical_onset_verified": True,
        "cursor_policy": "target page requests cursor:none; the operator moved the pointer to the upper-left margin away from all faces",
    }
    state["pending_actions"] = [
        "Continue only with a justified model route on exact repaired inputs; require a useful train-only pilot before ten roots and preserve separate train/stop/calibration/development roles.",
        "Implement and verify a useful five-person/OTHER/UNCERTAIN identity candidate; keep detection misses, multi-cluster ambiguity and identity errors separate.",
        "Integrate a useful model with the versioned standalone controller and LED/output legend; test full application memory and response latency.",
        "Qualify doctor/deploy/readback/configure/demo/recovery/rollback positively only after exact candidate and safe recovery evidence; the active runtime_config read previously returned EIO.",
        "Run actual fixed high/low and policy arms, startup ACK, cold-start, model reuse/swap/soak, and repeated wake recovery. Keep sensor sleep disabled in release until repeated behavior and power are measured.",
        "Prepare a new complete Windows transfer only after current source/models/config, allowed development stimuli, tested deploy/control/recovery, and archive/relocation checks are present. The historic v5 ZIP remains instrument-only.",
        "Keep final-role data sealed; no paper, proposal edit, publication, participant communication or final statistical claim is authorized by these diagnostics.",
    ]
    amendment_rows = state.setdefault("current_repair_amendments", [])
    for row in [
        {"revision": "matched-live-fomo-threshold04-diagnostic-v1", "updated_utc": NOW, "result": THRESHOLD_REL, "known_correct_accepts": "0/5", "wrong_known_accepts": 1, "single_cluster_detections_05": "4/5", "single_cluster_detections_04": "3/5", "ready_for_windows": False, "ready_for_final_build": False},
        {"revision": "volatile-a-full-device-sleep-wake-v3", "updated_utc": NOW, "result": POLICY_REL, "cycles_before_sleep": 3, "fomo_calls": 4, "identity_model_calls": 0, "wake_to_frame_ms": 659.136, "sample_gap_ms": 2297, "timeouts": 0, "ready_for_windows": False, "ready_for_final_build": False},
        {"revision": "camera-control-focused-tests-v2", "updated_utc": NOW, "receipt": TEST_REL, "focused_tests": "31/31 pass", "full_tests": "59/62; I-matrix fixture-key mismatch", "ready_for_windows": False, "ready_for_final_build": False},
    ]:
        if not any(old.get("revision") == row["revision"] for old in amendment_rows):
            amendment_rows.append(row)
    save(state_path, state)

    readiness["current_repair_v6_threshold04"]["evidence"] = [record(THRESHOLD_REL, "comparison"), record(THRESHOLD_REPORT_REL, "findings")]
    readiness["current_repair_v7_a_full_device_control"]["evidence"] = [record(POLICY_REL, "device_result"), record(POLICY_REPORT_REL, "findings")]
    readiness["generated_utc"] = NOW
    readiness["full_suite_result"] = record(TEST_REL, "test_receipt")
    readiness["readiness_summary"] = "The correct crop and one-frame input parity remain supported; identity is not useful, standalone controls are not release-qualified, and Windows current transfer is not ready."

    # Keep every prior readiness binding current, then add the new sources and records.
    binding_rows = {row["path"]: row for row in readiness.get("bindings", [])}
    extra_files = [
        "FaceRecognition/nicla_study/v6_camera_identity_threshold04_v1.py",
        "FaceRecognition/nicla_study/v6_summarize_camera_threshold04_v1.py",
        "FaceRecognition/nicla_study/v6_run_policy_sleep_wake_device_v1.py",
        "FaceRecognition/nicla_study/v6_run_policy_sleep_wake_device_v2.py",
        "FaceRecognition/nicla_study/v6_run_policy_sleep_wake_device_v3.py",
        "FaceRecognition/nicla_study/v6_summarize_policy_sleep_wake_device_v3.py",
        "FaceRecognition/nicla_study/v6_record_camera_control_tests_v1.py",
        "FaceRecognition/nicla_study/v6_record_camera_control_tests_v2.py",
        "FaceRecognition/nicla_study/v6_update_evidence_records_v6.py",
        "FaceRecognition/nicla_study/tests/test_camera_geometry_v6.py",
        "FaceRecognition/nicla_study/tests/test_fomo_peak_merge_v6.py",
        "FaceRecognition/nicla_study/tests/test_policy_controller_v6.py",
        "FaceRecognition/nicla_study/tests/test_runtime_decision_dtype_v6.py",
        THRESHOLD_REL, THRESHOLD_PUBLIC_REL, THRESHOLD_PRIVATE_REL,
        THRESHOLD_REPORT_REL, THRESHOLD_VISUAL_REL, THRESHOLD_CONTACT_REL,
        POLICY_REL, POLICY_MANIFEST_REL, POLICY_REPORT_REL, POLICY_REVIEW_REL,
        TEST_REL, TEST_REPORT_REL,
        "work/project-research-v5/CURRENT_SESSION_CHECKPOINT_V7.md",
        "work/project-research-v5/STATE.json", "work/project-research-v5/BRANCH_LEDGER.json",
        "work/project-research-v5/CLAIM_EVIDENCE.json", "work/project-research-v5/EXPERIMENT_REGISTRY.jsonl",
        "work/project-research-v5/STATUS.md", "work/project-research-v5/MAC_FINDINGS.md",
        "work/project-research-v5/DECISIONS.md", "work/project-research-v5/FINAL_BUILD_GAPS.md",
        "work/project-research-v5/CAMERA_IDENTITY_SURVEY_V2.md", "work/project-research-v5/02_WINDOWS_RESEARCH_AND_RETURN.md",
        "work/project-research-v5/runs/optical/CONTROL.json", "work/project-research-v5/runs/optical/RENDER_EVENTS.jsonl",
    ]
    for rel in extra_files:
        binding_rows.setdefault(rel, {"path": rel, "kind": "source_or_research_record"})
    for subtree in [
        STUDY / "runs/camera_identity_threshold04_v1",
        STUDY / "private/camera_identity_threshold04_v1",
        STUDY / "runs/camera_policy_sleep_wake_device_v3",
        STUDY / "private/camera_policy_sleep_wake_device_v3",
        STUDY / "reports/camera_control_contract_tests_v2",
    ]:
        for path in subtree.rglob("*"):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                binding_rows.setdefault(rel, {"path": rel, "kind": "private_diagnostic_or_test_evidence"})
    for rel, row in binding_rows.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("readiness binding missing: " + rel)
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha(path)
    readiness["bindings"] = list(binding_rows.values())

    archive = STUDY / "handoffs/nicla-camera-research-v5-windows.zip"
    sidecar = STUDY / "handoffs/nicla-camera-research-v5-windows.zip.sha256"
    archive_sha = sha(archive)
    if archive_sha != sidecar.read_text(encoding="utf-8").split()[0]:
        raise ValueError("historical Windows archive hash changed")
    readiness["historical_v5_transfer"]["archive_sha256"] = archive_sha
    readiness["historical_v5_transfer"]["status"] = "unchanged_instrument_only_archive_hash_verified"
    save(readiness_path, readiness)
    print(json.dumps({"updated_utc": NOW, "readiness_bindings": len(binding_rows), "ready_for_windows": readiness["ready_for_windows"], "ready_for_final_build": readiness["ready_for_final_build"], "recognition_qualified": readiness["recognition_qualified"], "historical_zip_sha256": archive_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
