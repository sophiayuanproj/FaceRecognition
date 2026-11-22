"""Validate and summarize the corrected one-block Nicla A-full diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
RUNS = STUDY / "runs/camera_policy_sleep_wake_device_v3"
PRIVATE = STUDY / "private/camera_policy_sleep_wake_device_v3"
RESULT = RUNS / "RESULT.json"
MANIFEST = RUNS / "PROGRAM_MANIFEST.json"
PREVIEW_RECEIPT = PRIVATE / "device_previews/K02_before_device.json"
PREVIEW_FRAME = PRIVATE / "device_previews/K02_before_device.jpg"
REVIEW = PRIVATE / "K02_CAMERA_VISUAL_REVIEW.json"
REPORT = STUDY / "CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    result = load(RESULT)
    manifest = load(MANIFEST)
    device = result["device_result"]
    receipt = load(PREVIEW_RECEIPT)
    source = (ROOT / manifest["embedded_policy_source"]).read_bytes()
    program = (ROOT / manifest["program"]).read_bytes()
    frame = PREVIEW_FRAME.read_bytes()
    if result["status"] != "completed" or device["status"] != "completed":
        raise ValueError("device policy probe did not complete")
    if manifest["policy_source_sha256"] != sha(source) or manifest["program_sha256"] != sha(program):
        raise ValueError("source or volatile program hash mismatch")
    if result["program_sha256"] != sha(program) or result["policy_source_sha256"] != sha(source):
        raise ValueError("device result is not bound to its program and policy source")
    if receipt["display_token"] != "469571835963515a4171d1f4":
        raise ValueError("K02 scene token changed")
    if receipt["camera_status"].get("age_s", 999) > 0.5 or receipt["camera_status"].get("error"):
        raise ValueError("K02 camera preview is stale or errored")
    if receipt["preview_frame_sha256"] != sha(frame):
        raise ValueError("K02 frame hash does not match preview receipt")
    if not device["sleep_requested"] or not device["wake_frame_verified"]:
        raise ValueError("A-full sleep request or wake frame was not verified")
    if len(device["cycles"]) != 3 or not all(row["quality_ok"] and row["fomo_cluster_count"] == 0 for row in device["cycles"]):
        raise ValueError("expected three quality-passing, no-face detector cycles")
    counters = device["controller_counters"]
    if counters["detector_calls"] != 4 or counters["captures"] != 4 or counters["camera_wakes"] != 1 or counters["camera_sleep_requests"] != 1:
        raise ValueError("controller counters do not match actual board work")
    if device["identity_model_calls"] != 0 or device["identity_decisions"] != 0:
        raise ValueError("identity-model scope changed")
    wake = device["sleep_and_wake"]
    if wake["elapsed_since_previous_capture_ms"] > 5000 or wake["bounded_gap_timeout"] is not None:
        raise ValueError("wake recovery exceeded the controller's maximum gap")
    if any(result.get(key) is not False for key in ("board_files_written", "startup_path_changed", "power_measured", "future_final_used")):
        raise ValueError("scope marker changed")

    review = {
        "schema": "nicla-policy-device-preview-review-v1",
        "target_condition": "K02 public-development stimulus",
        "display_token": receipt["display_token"],
        "target_matches_reviewed_source": True,
        "browser_render_ack_count": receipt["browser_render_ack_count"],
        "dwell_after_render_ack_s": receipt["dwell_after_render_ack_s"],
        "camera_preview_age_s": receipt["camera_status"]["age_s"],
        "preview_sha256": sha(frame),
        "pointer_over_face": False,
        "visual_observation": "The live frame shows the Colin Powell development image centered in the camera. No pointer covers the face.",
    }
    REVIEW.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# Nicla A-full sleep/wake control diagnostic v1",
        "",
        "A volatile raw-REPL program exercised the repository's maintained `PolicyController('A-full')` source on the actual Nicla Vision. The camera continuously sampled a render-acknowledged K02 development face scene. The frame-quality gate passed, but FOMO returned zero clusters in three consecutive captures.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| No-face, quality-passing cycles before sleep | {len(device['cycles'])} |",
        f"| FOMO model calls (controller counter) | {counters['detector_calls']} |",
        f"| Identity model calls | {device['identity_model_calls']} |",
        f"| Sensor sleep requests / wakes | {counters['camera_sleep_requests']} / {counters['camera_wakes']} |",
        f"| Periodic stationary interval | {device['controller_interval_ms']} ms |",
        f"| Wake-to-usable 240×240 frame | {wake['wake_to_usable_frame_ms']:.3f} ms |",
        f"| Gap from prior frame through wake | {wake['elapsed_since_previous_capture_ms']} ms (controller limit 5000 ms) |",
        f"| Controller timeouts | {counters['timeouts']} |",
        "",
        "The controller requested sensor sleep only after the third consecutive quality-passing no-face sample. After the 1500ms periodic interval, the camera woke and delivered a usable 240×240 frame; the controller reported no stale-evidence timeout. This verifies one volatile `A-full` detector/sleep/wake action on one setup. It does not test an identity model, motion input, startup ACK, cold start, battery operation, model swap/soak, power, or independent physical blocks.",
        "",
        "The v1 host harness stopped before opening the board because its output directories were not created; the v2 board execution succeeded but failed to increment the controller's detector-call counter. Both traces are preserved privately. The v3 source corrected the counter and is the only run summarized here.",
        "",
        "Reproduce: `python3 FaceRecognition/nicla_study/v6_run_policy_sleep_wake_device_v3.py`; regenerate this summary with `python3 FaceRecognition/nicla_study/v6_summarize_policy_sleep_wake_device_v3.py`.",
        "",
        "Raw board output and preview frames remain private at `private/camera_policy_sleep_wake_device_v3/`. Board files and startup were not changed; power was not measured.",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "status": "verified_one_block_a_full_sensor_sleep_wake",
        "report": str(REPORT.relative_to(ROOT)),
        "visual_review": str(REVIEW.relative_to(ROOT)),
        "cycles": len(device["cycles"]),
        "fomo_calls": counters["detector_calls"],
        "identity_model_calls": device["identity_model_calls"],
        "wake_to_usable_frame_ms": wake["wake_to_usable_frame_ms"],
        "bounded_gap_ms": wake["elapsed_since_previous_capture_ms"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
