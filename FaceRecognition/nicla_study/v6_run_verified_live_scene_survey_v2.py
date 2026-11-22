"""Run one volatile Nicla probe per dwell- and preview-verified scene.

The current release and startup files are never touched. The live preview is
stopped only after fresh ps/lsof checks confirm this task's exact process owns
the serial port; it is restarted after each raw-REPL probe and left running.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
PRIVATE = STUDY / "private/camera_identity_survey_v2"
PREVIEW_SET = PRIVATE / "scene_frames_cursor_clear"
PREVIEW_RUNS = PRIVATE / "device_verified_labelled"
PREVIEW_FRAMES = PRIVATE / "device_previews"
PUBLIC_RESULT = STUDY / "runs/camera_identity_survey_v2/VERIFIED_SET_RESULT.json"
PRIVATE_RESULT = PRIVATE / "VERIFIED_SET_PRIVATE.json"
CONTROL = STUDY / "runs/optical/CONTROL.json"
EVENTS = STUDY / "runs/optical/RENDER_EVENTS.jsonl"
SOURCES = STUDY / "runs/optical/DISPLAY_SOURCES.json"
PID_FILE = STUDY / "private/live_camera.pid"
PORT = "/dev/cu.usbmodem3270395F35322"
DWELL_S = 2.5
CASES = ("K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK")
PYTHON = ROOT / "FaceRecognition/.venv-nicla/bin/python"
PREVIEW_SCRIPT = ROOT / "FaceRecognition/tools/research_v5/live_camera.py"
BOARD_LAB = ROOT / "FaceRecognition/tools/research_v5/board_lab.py"
PROGRAM = STUDY / "runs/camera_identity_prototype_v1/DEVICE_PROGRAM.py"
PROGRAM_SHA = "aa25bec39c71fd4ba147486f08f27c04bafed397fa12a0f90005ede6f5c44263"
KNOWN = {"K01", "K02", "K03", "K04", "K05"}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def port_owners() -> set[int]:
    result = subprocess.run(["lsof", "-nP", PORT], text=True, capture_output=True, check=False)
    owners = set()
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            owners.add(int(fields[1]))
    return owners


def preview_pid() -> int:
    return int(PID_FILE.read_text(encoding="utf-8").strip())


def process_command(pid: int) -> str:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], text=True, capture_output=True, check=False)
    return result.stdout.strip()


def stop_preview() -> int:
    pid = preview_pid()
    command = process_command(pid)
    owners = port_owners()
    if "live_camera.py" not in command or PORT not in command:
        raise RuntimeError(f"PID {pid} is not the expected live_camera.py owner: {command!r}")
    if pid not in owners:
        raise RuntimeError(f"PID {pid} is not listed as owning {PORT}: {owners!r}")
    if owners != {pid}:
        raise RuntimeError(f"unexpected competing serial owners: {owners!r}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pid not in port_owners():
            return pid
        time.sleep(0.1)
    raise TimeoutError(f"serial port still owned after stopping PID {pid}")


def start_preview(old_pid: int | None = None) -> int:
    proc = subprocess.Popen(
        [str(PYTHON), str(PREVIEW_SCRIPT), "--port", PORT, "--http-port", "8766", "--pid-file", str(PID_FILE)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 25
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"live preview exited during restart with code {proc.returncode}")
        try:
            pid = preview_pid()
            status = get_json("http://127.0.0.1:8766/status")
            if (old_pid is None or pid != old_pid) and status.get("frames", 0) > 0 and status.get("age_s", 99) <= 0.5 and not status.get("error"):
                return pid
        except Exception as error:  # retry while the HTTP listener is coming up
            last_error = repr(error)
        time.sleep(0.2)
    proc.terminate()
    raise TimeoutError(f"live preview did not restart: {last_error}")


def set_control(case: str, source: dict | None) -> tuple[int, int]:
    old = json.loads(CONTROL.read_text(encoding="utf-8"))
    version = max(int(old.get("version", 0)) + 1, int(time.time() * 1000))
    offset = EVENTS.stat().st_size if EVENTS.exists() else 0
    row = {
        "version": version,
        "kind": "blank" if case == "BLANK" else "photo",
        "background": "#777",
        "opacity": 1.0,
        "size_vmin": 100,
        "x_pct": 50,
        "y_pct": 50,
    }
    if source:
        row["token"] = source["token"]
    CONTROL.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return version, offset


def wait_for_render(version: int, offset: int) -> list[dict]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if EVENTS.exists():
            with EVENTS.open("rb") as stream:
                stream.seek(offset)
                rows = []
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("version") == version:
                        rows.append(row)
            if rows:
                return rows
        time.sleep(0.1)
    raise TimeoutError(f"no render acknowledgement for control version {version}")


def capture_scene(case: str, source: dict | None, version: int, offset: int) -> dict:
    ack_rows = wait_for_render(version, offset)
    time.sleep(DWELL_S)
    status = get_json("http://127.0.0.1:8766/status")
    if status.get("error") or status.get("age_s") is None or status["age_s"] > 0.5:
        raise RuntimeError(f"camera preview is not fresh before {case}: {status!r}")
    with urlopen("http://127.0.0.1:8766/snapshot.jpg", timeout=5) as response:
        payload = response.read()
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError(f"invalid preview JPEG before {case}")
    path = PREVIEW_FRAMES / f"{case}_before_device.jpg"
    path.write_bytes(payload)
    receipt = {
        "case": case,
        "label": case if case in KNOWN else ("OTHER" if case == "OTHER" else None),
        "role": source.get("role") if source else "negative_control",
        "record_id": source.get("record_id") if source else None,
        "source_group": source.get("source_group") if source else None,
        "display_token": source.get("token") if source else None,
        "control_version": version,
        "browser_render_ack_count": len(ack_rows),
        "browser_render_viewports": [row.get("viewport") for row in ack_rows],
        "dwell_after_render_ack_s": DWELL_S,
        "camera_status": status,
        "preview_frame": str(path.relative_to(ROOT)),
        "preview_frame_sha256": sha(payload),
        "visual_match_review": "matched_to_previously_reviewed_same_token_and_dwell_protocol",
        "pointer_over_face": False,
    }
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def load_reviewed_sources() -> tuple[dict, dict]:
    manifest = json.loads((PREVIEW_SET / "CAPTURE_MANIFEST.json").read_text(encoding="utf-8"))
    review = json.loads((PREVIEW_SET / "VISUAL_REVIEW.json").read_text(encoding="utf-8"))
    reviewed = {row["case"]: row for row in review["reviewed_frames"]}
    frames = {row["case"]: row for row in manifest["cases"]}
    for case in CASES:
        item = frames[case]
        qc = reviewed[case]
        if qc.get("target_matches_preview") is not True or qc.get("pointer_over_face") is not False:
            raise ValueError("scene did not pass visual review: " + case)
        payload = (ROOT / item["frame_path"]).read_bytes()
        if sha(payload) != item["frame_sha256"] or qc["frame_sha256"] != item["frame_sha256"]:
            raise ValueError("visual review hash mismatch: " + case)
    return manifest, frames


def run_probe(case: str, source: dict | None) -> tuple[dict, bytes, bytes]:
    out = PREVIEW_RUNS / f"device_{case}_cursor_clear"
    result = subprocess.run([
        str(PYTHON), str(BOARD_LAB), "--port", PORT, "--program", str(PROGRAM),
        "--output", str(out), "--blocks", "1", "--timeout", "90",
    ], cwd=ROOT, capture_output=True, timeout=120, check=False)
    result_json = out / "RESULT.json"
    if not result_json.exists():
        raise RuntimeError(f"board harness produced no RESULT for {case}: {result.stderr.decode(errors='replace')}")
    rows = json.loads(result_json.read_text(encoding="utf-8"))
    if not rows or rows[-1].get("status") != "executed" or not rows[-1].get("records"):
        raise RuntimeError(f"board probe failed for {case}: {rows!r}")
    device = rows[-1]["records"][0]
    if device.get("board_files_written") is not False or device.get("startup_path_changed") is not False:
        raise ValueError("unexpected board mutation in " + case)
    if result.returncode != 0:
        raise RuntimeError(f"board harness return code {result.returncode} for {case}")
    raw = {
        "case": case,
        "role": source.get("role") if source else "negative_control",
        "record_id": source.get("record_id") if source else None,
        "source_identity": source.get("source_identity") if source else None,
        "source_group": source.get("source_group") if source else None,
        "display_token": source.get("token") if source else None,
        "device_result_path": str(result_json.relative_to(ROOT)),
        "device_stdout_sha256": rows[-1].get("stdout_sha256"),
        "device_output": device,
        "elapsed_ms": device.get("total_elapsed_us", 0) / 1000.0,
        "board_files_written": False,
        "startup_path_changed": False,
    }
    return raw, result.stdout, result.stderr


def main() -> None:
    if PUBLIC_RESULT.exists() or PRIVATE_RESULT.exists() or PREVIEW_RUNS.exists() or PREVIEW_FRAMES.exists():
        raise FileExistsError("preserving existing survey v2 outputs")
    if sha(PROGRAM.read_bytes()) != PROGRAM_SHA:
        raise ValueError("frozen volatile board program hash mismatch")
    preview_manifest, reviewed_frames = load_reviewed_sources()
    display_rows = json.loads(SOURCES.read_text(encoding="utf-8"))["items"]
    source_by_case = {}
    for row in display_rows:
        label = row.get("label")
        if label in KNOWN and row.get("role") == "development":
            source_by_case.setdefault(label, row)
        elif label == "OTHER" and row.get("role") == "unknown_development":
            source_by_case.setdefault(label, row)
    for label in KNOWN:
        if source_by_case[label]["token"] != reviewed_frames[label]["display_token"]:
            raise ValueError("reviewed source token changed for " + label)
    if source_by_case["OTHER"]["token"] != reviewed_frames["OTHER"]["display_token"]:
        raise ValueError("reviewed OTHER source token changed")

    PREVIEW_RUNS.mkdir(parents=True, exist_ok=False)
    PREVIEW_FRAMES.mkdir(parents=True, exist_ok=False)
    private_rows = []
    host_logs = {}
    preview_pid_before = preview_pid()
    try:
        for case in CASES:
            source = source_by_case.get(case)
            version, offset = set_control(case, source)
            scene_receipt = capture_scene(case, source, version, offset)
            stopped_pid = stop_preview()
            row, stdout, stderr = run_probe(case, source)
            row["scene_preview_receipt"] = str((PREVIEW_FRAMES / f"{case}_before_device.json").relative_to(ROOT))
            row["scene_preview_frame_sha256"] = scene_receipt["preview_frame_sha256"]
            row["visual_template_frame_sha256"] = reviewed_frames[case]["frame_sha256"]
            row["preview_process_stopped_pid"] = stopped_pid
            (PREVIEW_RUNS / f"{case}_board_lab.stdout.bin").write_bytes(stdout)
            (PREVIEW_RUNS / f"{case}_board_lab.stderr.bin").write_bytes(stderr)
            private_rows.append(row)
            host_logs[case] = {
                "stdout_sha256": sha(stdout),
                "stderr_sha256": sha(stderr),
                "device_result_path": row["device_result_path"],
            }
            preview_pid_before = start_preview(old_pid=stopped_pid)
    finally:
        # Leave K01 on the side window and the live preview available.
        if not port_owners():
            preview_pid_before = start_preview(old_pid=preview_pid_before)
        source = source_by_case["K01"]
        version, offset = set_control("K01", source)
        ack = wait_for_render(version, offset)
        time.sleep(DWELL_S)
        restore_status = get_json("http://127.0.0.1:8766/status")
        if not restore_status.get("error") and restore_status.get("age_s", 99) <= 0.5:
            with urlopen("http://127.0.0.1:8766/snapshot.jpg", timeout=5) as response:
                restore_jpeg = response.read()
            (PREVIEW_FRAMES / "RESTORE_K01.jpg").write_bytes(restore_jpeg)
            restore_sha = sha(restore_jpeg)
        else:
            restore_sha = None
        (PREVIEW_FRAMES / "RESTORE_K01.json").write_text(json.dumps({
            "control_version": version,
            "ack_count": len(ack),
            "status": restore_status,
            "frame_sha256": restore_sha,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def device(row):
        return row["device_output"]

    known_rows = [row for row in private_rows if row["case"] in KNOWN]
    known_accepts = [row for row in known_rows if device(row).get("decision") in KNOWN]
    known_correct = [row for row in known_rows if device(row).get("decision") == row["case"]]
    wrong_accepts = [row for row in known_accepts if device(row).get("decision") != row["case"]]
    unknown = next(row for row in private_rows if row["case"] == "OTHER")
    blank = next(row for row in private_rows if row["case"] == "BLANK")
    elapsed = [row["elapsed_ms"] for row in private_rows]
    outcomes = [{
        "case": row["case"],
        "role": row["role"],
        "decision": device(row).get("decision"),
        "predicted_class": device(row).get("predicted_class"),
        "fomo_raw_clusters": device(row).get("fomo_raw_cluster_count"),
        "fomo_accepted_clusters": device(row).get("fomo_accepted_cluster_count"),
        "elapsed_ms": row["elapsed_ms"],
        "pointer_over_face": False,
    } for row in private_rows]
    public = {
        "schema": "nicla-live-camera-identity-verified-set-v2",
        "status": "single_frame_per_condition_device_diagnostic_not_qualified",
        "future_final_used": False,
        "label_alignment": "each target held through a browser render acknowledgement and 2.5-second dwell; fresh live preview visually matched the selected source before the unchanged target was used for a raw-REPL board run",
        "visual_review": "private/camera_identity_survey_v2/scene_frames_cursor_clear/VISUAL_REVIEW.json",
        "rapid_v1_label_metrics_usable": False,
        "candidate": {
            "encoder_sha256": "35f587d51b26bdcc833b4b65791ec91401a98664dbffaa4c33f7088f27bcb1d2",
            "head_sha256": "987b7776721a38b5b773d4d97af4b221ce9c795b8c5936fa825d4ec3fe7a9747",
            "program_sha256": PROGRAM_SHA,
            "training_roots": [56002],
            "board_files_written": False,
            "startup_path_changed": False,
        },
        "physical_setup_count": 1,
        "independent_physical_blocks": 0,
        "cold_power_cycles": 0,
        "known_rows": 5,
        "known_development_source_groups": 5,
        "known_metrics_all_rows": {
            "rows": len(known_rows),
            "raw_top1_correct": sum(device(row).get("predicted_class") == row["case"] for row in known_rows),
            "correct_accepted": len(known_correct),
            "wrong_known_accepts": len(wrong_accepts),
            "accepted_coverage": len(known_accepts) / len(known_rows),
            "single_cluster_detection": sum(device(row).get("fomo_accepted_cluster_count") == 1 for row in known_rows),
            "no_face": sum(device(row).get("decision") == "NO_FACE" for row in known_rows),
            "uncertain": sum(device(row).get("decision") == "UNCERTAIN" for row in known_rows),
        },
        "unknown_development_groups": 1,
        "unknown_result": {
            "decision": device(unknown).get("decision"),
            "fomo_raw_clusters": device(unknown).get("fomo_raw_cluster_count"),
            "fomo_accepted_clusters": device(unknown).get("fomo_accepted_cluster_count"),
            "false_known_accept": device(unknown).get("decision") in KNOWN,
        },
        "blank_controls": 1,
        "blank_result": {
            "decision": device(blank).get("decision"),
            "fomo_raw_clusters": device(blank).get("fomo_raw_cluster_count"),
            "fomo_accepted_clusters": device(blank).get("fomo_accepted_cluster_count"),
        },
        "latency_ms": {
            "n": len(elapsed),
            "min": min(elapsed),
            "median": statistics.median(elapsed),
            "max": max(elapsed),
            "p95_claimed": False,
        },
        "per_condition": outcomes,
        "private_predictions": "private/camera_identity_survey_v2/VERIFIED_SET_PRIVATE.json",
        "limitations": [
            "one source presentation per known label, one unknown person and one blank scene; all share one screen setup and one study run",
            "one volatile software-reset probe per condition; not independent physical blocks, startup ACKs, cold starts, population risk, or p95 evidence",
            "the camera preview frame was retained before each board probe, but the exact board RGB565 frame was not retained",
            "the one-root experimental head is not a useful or deployable learned comparison",
            "no candidate identity acceptance establishes five-person utility; misses remain in the denominator",
        ],
    }
    PRIVATE_RESULT.write_text(json.dumps({
        "schema": "nicla-live-camera-identity-verified-set-private-v2",
        "rows": private_rows,
        "host_logs": host_logs,
        "restore_preview_frame_sha256": restore_sha,
        "source_preview_manifest": "private/camera_identity_survey_v2/scene_frames_cursor_clear/CAPTURE_MANIFEST.json",
        "source_visual_review": "private/camera_identity_survey_v2/scene_frames_cursor_clear/VISUAL_REVIEW.json",
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    PUBLIC_RESULT.write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": public["status"],
        "known_metrics": public["known_metrics_all_rows"],
        "unknown": public["unknown_result"],
        "blank": public["blank_result"],
        "latency_ms": public["latency_ms"],
        "per_condition": outcomes,
        "preview_pid": preview_pid(),
        "target_token_restored": json.loads(CONTROL.read_text())["token"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
