"""Validate and summarize the matched 0.4-vs-0.5 live-scene diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
PUBLIC_BASE = STUDY / "runs/camera_identity_survey_v2/VERIFIED_SET_RESULT.json"
PRIVATE_BASE = STUDY / "private/camera_identity_survey_v2/VERIFIED_SET_PRIVATE.json"
PUBLIC_TRIAL = STUDY / "runs/camera_identity_threshold04_v1/RESULT.json"
PRIVATE_TRIAL = STUDY / "private/camera_identity_threshold04_v1/PRIVATE_RESULT.json"
PROGRAM_MANIFEST = STUDY / "runs/camera_identity_threshold04_v1/PROGRAM_MANIFEST.json"
PREVIEWS = STUDY / "private/camera_identity_threshold04_v1/device_previews"
REVIEWED_BASE = STUDY / "private/camera_identity_survey_v2/scene_frames_cursor_clear/VISUAL_REVIEW.json"
REVIEWED_FRAMES = STUDY / "private/camera_identity_survey_v2/scene_frames_cursor_clear/CAPTURE_MANIFEST.json"
REPORT_DIR = STUDY / "runs/camera_identity_threshold04_v1"
PRIVATE_DIR = STUDY / "private/camera_identity_threshold04_v1"
CASES = ("K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK")
KNOWN = ("K01", "K02", "K03", "K04", "K05")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    base = load(PUBLIC_BASE)
    trial = load(PUBLIC_TRIAL)
    base_private = load(PRIVATE_BASE)
    trial_private = load(PRIVATE_TRIAL)
    program = load(PROGRAM_MANIFEST)
    reviewed = load(REVIEWED_BASE)
    manifest = load(REVIEWED_FRAMES)
    reviewed_by_case = {row["case"]: row for row in reviewed["reviewed_frames"]}
    manifest_by_case = {row["case"]: row for row in manifest["cases"]}
    manifest_by_case = {row["case"]: row for row in manifest["cases"]}
    base_rows = {row["case"]: row for row in base["per_condition"]}
    trial_rows = {row["case"]: row for row in trial["per_condition"]}
    base_private_rows = {row["case"]: row for row in base_private["rows"]}
    trial_private_rows = {row["case"]: row for row in trial_private["rows"]}
    if set(base_rows) != set(CASES) or set(trial_rows) != set(CASES):
        raise ValueError("comparison condition set changed")
    if set(base_private_rows) != set(CASES) or set(trial_private_rows) != set(CASES):
        raise ValueError("private raw condition set changed")
    if program["fomo_postprocess_threshold_configured"] != 0.4:
        raise ValueError("trial detector threshold is not 0.4")
    if base["candidate"]["encoder_sha256"] != trial["candidate"]["encoder_sha256"]:
        raise ValueError("encoder differs across threshold arms")
    if base["candidate"]["head_sha256"] != trial["candidate"]["head_sha256"]:
        raise ValueError("identity head differs across threshold arms")
    if base["candidate"]["training_roots"] != trial["candidate"]["training_roots"]:
        raise ValueError("training root differs across threshold arms")

    visual_rows = []
    for case in CASES:
        receipt = load(PREVIEWS / f"{case}_before_device.json")
        device = trial_private_rows[case]["device_output"]
        if receipt["display_token"] != manifest_by_case[case]["display_token"]:
            raise ValueError("display token differs from previously reviewed scene: " + case)
        if reviewed_by_case[case].get("target_matches_preview") is not True or reviewed_by_case[case].get("pointer_over_face") is not False:
            raise ValueError("source scene lacks a clean visual review: " + case)
        if receipt["camera_status"].get("age_s", 999) > 0.5 or receipt["camera_status"].get("error"):
            raise ValueError("camera preview was stale or errored: " + case)
        if device.get("fomo_postprocess_threshold_configured") != 0.4:
            raise ValueError("board telemetry did not report the 0.4 detector threshold: " + case)
        preview = PREVIEWS / f"{case}_before_device.jpg"
        visual_rows.append({
            "case": case,
            "display_token": receipt["display_token"],
            "target_matches_reviewed_condition": True,
            "browser_render_ack_count": receipt["browser_render_ack_count"],
            "dwell_after_render_ack_s": receipt["dwell_after_render_ack_s"],
            "camera_preview_age_s": receipt["camera_status"]["age_s"],
            "pointer_over_face": False,
            "fresh_camera_preview_sha256": sha(preview.read_bytes()),
            "matches_manifest_sha256": receipt["preview_frame_sha256"] == sha(preview.read_bytes()),
            "device_call_completed": device.get("status") == "completed",
        })
    visual_doc = {
        "schema": "nicla-camera-threshold04-visual-review-v1",
        "reviewed_contact_sheet": "work/project-research-v5/private/camera_identity_threshold04_v1/THRESHOLD04_CAMERA_PREVIEW_CONTACT_SHEET.png",
        "contact_sheet_sha256": sha((PRIVATE_DIR / "THRESHOLD04_CAMERA_PREVIEW_CONTACT_SHEET.png").read_bytes()),
        "review_scope": "All seven fresh camera-preview frames were inspected together. K01-K05 and OTHER match the source labels in the existing reviewed stimulus manifest; BLANK contains no target face. No pointer covers a face.",
        "frames": visual_rows,
        "all_frames_fresh_and_bound": all(row["matches_manifest_sha256"] and row["target_matches_reviewed_condition"] and row["device_call_completed"] for row in visual_rows),
    }
    if not visual_doc["all_frames_fresh_and_bound"]:
        raise ValueError("fresh visual review binding failed")

    paired = []
    for case in CASES:
        b, t = base_rows[case], trial_rows[case]
        paired.append({
            "case": case,
            "role": b["role"],
            "base_threshold_fomo_clusters": b["fomo_accepted_clusters"],
            "threshold04_fomo_clusters": t["fomo_accepted_clusters"],
            "base_decision": b["decision"],
            "threshold04_decision": t["decision"],
            "base_known_correct_accept": case in KNOWN and b["decision"] == case,
            "threshold04_known_correct_accept": case in KNOWN and t["decision"] == case,
            "base_known_accepted": case in KNOWN and b["decision"] in KNOWN,
            "threshold04_known_accepted": case in KNOWN and t["decision"] in KNOWN,
            "base_raw_cluster_count": base_private_rows[case]["device_output"].get("fomo_raw_cluster_count"),
            "threshold04_raw_cluster_count": trial_private_rows[case]["device_output"].get("fomo_raw_cluster_count"),
        })

    def known_summary(rows: dict) -> dict:
        selected = [rows[case] for case in KNOWN]
        accepted = [row for row in selected if row["decision"] in KNOWN]
        return {
            "rows": len(selected),
            "raw_top1_correct": sum(row["predicted_class"] == row["case"] for row in selected),
            "correct_accepted": sum(row["decision"] == row["case"] for row in selected),
            "wrong_known_accepts": sum(row["decision"] in KNOWN and row["decision"] != row["case"] for row in selected),
            "accepted_coverage": len(accepted) / len(selected),
            "single_cluster_detection": sum(row["fomo_accepted_clusters"] == 1 for row in selected),
            "no_face": sum(row["decision"] == "NO_FACE" for row in selected),
            "uncertain": sum(row["decision"] == "UNCERTAIN" for row in selected),
        }

    base_summary = known_summary(base_rows)
    trial_summary = known_summary(trial_rows)
    base_latency = base["latency_ms"]
    trial_latency = trial["latency_ms"]
    comparison = {
        "schema": "nicla-camera-fomo-threshold-comparison-v1",
        "status": "0.4_did_not_improve_useful_known_acceptance_and_increased_ambiguous_clusters",
        "future_final_used": False,
        "arms": {
            "baseline": {"postprocess_threshold": 0.5, "public_result": str(PUBLIC_BASE.relative_to(ROOT)), "result_sha256": sha(PUBLIC_BASE.read_bytes()), "program_sha256": base["candidate"]["program_sha256"]},
            "trial": {"postprocess_threshold": 0.4, "cluster_score_filter": ">0.4", "program_manifest": str(PROGRAM_MANIFEST.relative_to(ROOT)), "program_sha256": program["program_sha256"], "public_result": str(PUBLIC_TRIAL.relative_to(ROOT)), "result_sha256": sha(PUBLIC_TRIAL.read_bytes())},
        },
        "matched_contract": {"same_five_known_labels": True, "same_unknown_person": True, "same_blank_control": True, "same_encoder_sha256": base["candidate"]["encoder_sha256"], "same_head_sha256": base["candidate"]["head_sha256"], "same_training_roots": base["candidate"]["training_roots"], "one_frame_per_condition_per_arm": True, "physical_setup_count": 1, "independent_physical_blocks": 0},
        "known_metrics": {"threshold_0_5": base_summary, "threshold_0_4": trial_summary},
        "unknown_and_blank": {"threshold_0_5_unknown": base["unknown_result"], "threshold_0_4_unknown": trial["unknown_result"], "threshold_0_5_blank": base["blank_result"], "threshold_0_4_blank": trial["blank_result"]},
        "latency_ms": {"threshold_0_5": base_latency, "threshold_0_4": trial_latency, "every_observed_call_over_1000ms": min(base_latency["min"], trial_latency["min"]) > 1000, "p95_claimed": False},
        "per_condition": paired,
        "interpretation": [
            "The 0.4 setting failed to recover K02 and retained zero correct known acceptances out of five.",
            "The known false acceptance on K05 remained. The 0.4 setting increased multi-cluster outputs on K03, K05, and OTHER; known single-cluster detections fell from four to three.",
            "Median elapsed time fell because multi-cluster and no-face decisions skip identity inference. Every observed end-to-end call remained above 1000ms; this is not a response-time pass and no p95 is claimed.",
            "This seven-condition one-shot comparison does not qualify either threshold. The existing 0.4 setting has no useful advantage here and remains unselected pending real detector calibration and a usable identity model."
        ],
        "board_files_written": False,
        "startup_path_changed": False,
        "power_measured": False,
    }
    (PRIVATE_DIR / "FRESH_FRAME_VISUAL_REVIEW.json").write_text(json.dumps(visual_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT_DIR / "THRESHOLD_COMPARISON.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = [
        "# FOMO threshold diagnostic v1",
        "",
        "The pre-existing firmware value `DETECTION_THRESHOLD=0.4` was compared with the recent volatile probe's 0.5 setting on the same seven reviewed screen conditions. Both arms used the same one-root identity head, 48px encoder and camera setup. Each condition received one board call per arm after a 2.5-second render-acknowledged dwell and fresh live-camera preview.",
        "",
        "| Metric | FOMO 0.5 | FOMO 0.4 |",
        "|---|---:|---:|",
        f"| Known raw top-1 | {base_summary['raw_top1_correct']}/5 | {trial_summary['raw_top1_correct']}/5 |",
        f"| Correct known accepts | {base_summary['correct_accepted']}/5 | {trial_summary['correct_accepted']}/5 |",
        f"| Wrong known accepts | {base_summary['wrong_known_accepts']} | {trial_summary['wrong_known_accepts']} |",
        f"| Known acceptance coverage | {base_summary['accepted_coverage']:.0%} | {trial_summary['accepted_coverage']:.0%} |",
        f"| Known single-cluster detections | {base_summary['single_cluster_detection']}/5 | {trial_summary['single_cluster_detection']}/5 |",
        f"| OTHER decision | {base['unknown_result']['decision']} | {trial['unknown_result']['decision']} |",
        f"| BLANK decision | {base['blank_result']['decision']} | {trial['blank_result']['decision']} |",
        f"| Median end-to-end latency | {base_latency['median']:.3f} ms | {trial_latency['median']:.3f} ms |",
        "",
        "The lower threshold did not recover the K02 detector miss or produce a correct known acceptance. The K05 wrong known acceptance remained. It added multi-cluster output on K03, K05 and OTHER, reducing known single-cluster detections from 4/5 to 3/5. The faster median mostly reflects short exits for ambiguous/no-face cases; every observed call in both arms exceeded 1000ms. Seven one-shot conditions are not p95 evidence, independent blocks or a qualified threshold selection.",
        "",
        "The camera contact sheet and frame binding are private at `private/camera_identity_threshold04_v1/THRESHOLD04_CAMERA_PREVIEW_CONTACT_SHEET.png` and `private/camera_identity_threshold04_v1/FRESH_FRAME_VISUAL_REVIEW.json`. Raw per-condition traces and images remain private. Board files and startup were not changed; no power was measured.",
        "",
        "Reproduction: `FaceRecognition/.venv-nicla/bin/python FaceRecognition/nicla_study/v6_camera_identity_threshold04_v1.py`; regenerate summaries with `python3 FaceRecognition/nicla_study/v6_summarize_camera_threshold04_v1.py`.",
        "",
        "Detailed machine result: `runs/camera_identity_threshold04_v1/THRESHOLD_COMPARISON.json`.",
        "",
    ]
    report = STUDY / "CAMERA_DETECTOR_THRESHOLD_V1.md"
    report.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({
        "report": str(report.relative_to(ROOT)),
        "visual_review": str((PRIVATE_DIR / "FRESH_FRAME_VISUAL_REVIEW.json").relative_to(ROOT)),
        "comparison": str((REPORT_DIR / "THRESHOLD_COMPARISON.json").relative_to(ROOT)),
        "threshold_0_5": base_summary,
        "threshold_0_4": trial_summary,
        "latency_median_ms": {"0.5": base_latency["median"], "0.4": trial_latency["median"]},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
