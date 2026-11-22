"""Capture dwell-verified development stimulus scenes from the live Nicla preview.

This stages visual evidence only. It does not score recognition or modify the
board. Review every saved camera frame before assigning it an expected label.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
OUT = STUDY / "private/camera_identity_survey_v2/scene_frames_cursor_clear"
CONTROL = STUDY / "runs/optical/CONTROL.json"
SOURCES = STUDY / "runs/optical/DISPLAY_SOURCES.json"
EVENTS = STUDY / "runs/optical/RENDER_EVENTS.jsonl"
DISPLAY_STATE_URL = "http://127.0.0.1:8767/state"
CAMERA_STATUS_URL = "http://127.0.0.1:8766/status"
CAMERA_FRAME_URL = "http://127.0.0.1:8766/snapshot.jpg"
DWELL_S = 2.5
CASES = ("K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def wait_for_ack(version: int, offset: int, timeout_s: float = 10.0) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    rows = []
    while time.monotonic() < deadline:
        if EVENTS.exists():
            with EVENTS.open("rb") as stream:
                stream.seek(offset)
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
    raise TimeoutError(f"no browser render acknowledgement for version {version}")


def set_control(row: dict | None, kind: str) -> int:
    previous = json.loads(CONTROL.read_text(encoding="utf-8"))
    version = max(int(previous.get("version", 0)) + 1, int(time.time() * 1000))
    if kind == "photo":
        control = {
            "version": version,
            "kind": "photo",
            "token": row["token"],
            "background": "#777",
            "opacity": 1.0,
            "size_vmin": 100,
            "x_pct": 50,
            "y_pct": 50,
        }
    else:
        control = {
            "version": version,
            "kind": "blank",
            "background": "#777",
            "opacity": 1.0,
            "size_vmin": 100,
            "x_pct": 50,
            "y_pct": 50,
        }
    CONTROL.write_text(json.dumps(control, sort_keys=True) + "\n", encoding="utf-8")
    return version


def capture(case: str, source: dict | None, version: int, offset: int) -> dict:
    ack_rows = wait_for_ack(version, offset)
    time.sleep(DWELL_S)
    status = get_json(CAMERA_STATUS_URL)
    if status.get("error"):
        raise RuntimeError("camera preview error: " + str(status["error"]))
    if status.get("age_s") is None or status["age_s"] > 0.5:
        raise RuntimeError("camera preview frame is stale: " + repr(status))
    with urlopen(CAMERA_FRAME_URL, timeout=5) as response:
        payload = response.read()
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("camera preview did not return a JPEG")
    stem = f"{case}_preview.jpg"
    path = OUT / stem
    path.write_bytes(payload)
    row = {
        "case": case,
        "expected_label": case if case.startswith("K0") else ("OTHER" if case == "OTHER" else None),
        "role": source.get("role") if source else "negative_control",
        "record_id": source.get("record_id") if source else None,
        "source_group": source.get("source_group") if source else None,
        "display_token": source.get("token") if source else None,
        "control_version": version,
        "browser_render_ack_count": len(ack_rows),
        "browser_render_viewports": [ack.get("viewport") for ack in ack_rows],
        "dwell_after_first_ack_s": DWELL_S,
        "camera_status": status,
        "frame_path": str(path.relative_to(ROOT)),
        "frame_bytes": len(payload),
        "frame_sha256": sha(payload),
        "visual_match_review": "pending",
    }
    (OUT / f"{case}_preview.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))["items"]
    selected = {}
    for row in sources:
        label = row.get("label")
        if label in {"K01", "K02", "K03", "K04", "K05"} and row.get("role") == "development":
            selected.setdefault(label, row)
        elif label == "OTHER" and row.get("role") == "unknown_development":
            selected.setdefault(label, row)
    if set(selected) != {"K01", "K02", "K03", "K04", "K05", "OTHER"}:
        raise ValueError("missing frozen development source for a required label")

    before = get_json(DISPLAY_STATE_URL)
    cursor = EVENTS.stat().st_size if EVENTS.exists() else 0
    output = []
    try:
        for case in CASES:
            source = selected.get(case)
            kind = "blank" if case == "BLANK" else "photo"
            version = set_control(source, kind)
            output.append(capture(case, source, version, cursor))
            cursor = EVENTS.stat().st_size if EVENTS.exists() else cursor
    finally:
        # Leave the original K01 development target in the side window.
        source = selected["K01"]
        restore_cursor = EVENTS.stat().st_size if EVENTS.exists() else 0
        version = set_control(source, "photo")
        restore_acks = wait_for_ack(version, restore_cursor)
        time.sleep(DWELL_S)
        restore = {
            "kind": "photo",
            "label": "K01",
            "token": source["token"],
            "control_version": version,
            "ack_count": len(restore_acks),
            "status": get_json(CAMERA_STATUS_URL),
        }
        (OUT / "RESTORE.json").write_text(json.dumps(restore, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "nicla-live-target-frame-capture-v2",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": output,
        "initial_display_state": before,
        "restored_display_state": restore,
        "future_final_used": False,
        "device_runs_performed": False,
        "label_assignment_policy": "visual review of saved camera JPEGs is required before scoring any board output",
    }
    (OUT / "CAPTURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(output), "restore_label": restore["label"], "frames": [{"case": x["case"], "bytes": x["frame_bytes"], "sha256": x["frame_sha256"]} for x in output]}, sort_keys=True))


if __name__ == "__main__":
    main()
