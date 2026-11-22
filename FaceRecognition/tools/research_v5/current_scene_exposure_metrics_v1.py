#!/usr/bin/env python3
"""Host-side visual QC for the volatile GC2145 exposure probe (not FOMO)."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def measure(path, cascade):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (240, 240):
        raise ValueError("invalid 240x240 JPEG: " + str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(18, 18))
    faces = []
    for x, y, w, h in boxes:
        roi = gray[y:y+h, x:x+w]
        faces.append({
            "xywh_sensor_px": [int(x), int(y), int(w), int(h)],
            "center_xy_px": [float(x + w / 2), float(y + h / 2)],
            "fraction_of_frame": float(w * h / (240 * 240)),
            "mean_gray_u8": float(roi.mean()),
            "std_gray_u8": float(roi.std()),
            "fraction_ge_245": float((roi >= 245).mean()),
            "laplacian_variance_host_proxy": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        })
    return {
        "frame_mean_gray_u8": float(gray.mean()),
        "frame_std_gray_u8": float(gray.std()),
        "frame_p05_gray_u8": float(np.percentile(gray, 5)),
        "frame_p50_gray_u8": float(np.percentile(gray, 50)),
        "frame_p95_gray_u8": float(np.percentile(gray, 95)),
        "frame_fraction_ge_245": float((gray >= 245).mean()),
        "frame_fraction_le_10": float((gray <= 10).mean()),
        "laplacian_variance_host_proxy": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "haar_face_count": int(len(boxes)),
        "haar_faces": faces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    events = []
    for event in result.get("events", []):
        if event.get("kind") != "frame":
            continue
        path = args.research_root / event["jpeg_path"]
        events.append({
            "arm": event["arm"],
            "requested_exposure_us": event.get("requested_exposure_us"),
            "reported_exposure_us": event.get("reported_exposure_us"),
            "reported_gain_db": event.get("reported_gain_db"),
            "frame_index": event["frame_index"],
            "jpeg_sha256": event["jpeg_sha256"],
            "metrics": measure(path, cascade),
        })
    grouped = {}
    for event in events:
        key = (event["arm"], event.get("requested_exposure_us"))
        grouped.setdefault(key, []).append(event)
    summaries = {}
    for key, frame_events in grouped.items():
        metrics_rows = [event["metrics"] for event in frame_events]
        summaries[str(key)] = {
            "arm": key[0],
            "requested_exposure_us": key[1],
            "n_frames": len(metrics_rows),
            "mean_frame_gray": float(np.mean([r["frame_mean_gray_u8"] for r in metrics_rows])),
            "sd_frame_gray": float(np.std([r["frame_mean_gray_u8"] for r in metrics_rows], ddof=1)) if len(metrics_rows) > 1 else None,
            "mean_fraction_ge_245": float(np.mean([r["frame_fraction_ge_245"] for r in metrics_rows])),
            "haar_detection_count": int(sum(r["haar_face_count"] > 0 for r in metrics_rows)),
            "mean_face_fraction_ge_245": float(np.mean([
                np.mean([f["fraction_ge_245"] for f in r["haar_faces"]])
                for r in metrics_rows if r["haar_faces"]
            ])) if any(r["haar_faces"] for r in metrics_rows) else None,
            "note": "within-scene frames are technical repeats, not independent physical blocks",
        }
    qc = {
        "schema": "current-scene-exposure-host-qc-v1",
        "scope": "private JPEG replay visual QC; OpenCV Haar only, not deployed FOMO, not identity",
        "opencv_version": cv2.__version__,
        "frame_count": len(events),
        "events": events,
        "groups": summaries,
        "limitations": [
            "JPEG frames are encoded from RGB565 framebuffer and are not the exact pixel buffer used by the FOMO model.",
            "Haar detections and Laplacian variance are host diagnostics, not validated face-quality metrics.",
            "The screen target is stationary; repeated frames do not provide independent people, scenes, or physical blocks.",
        ],
    }
    output = args.result.with_name("HOST_QC.json")
    output.write_text(json.dumps(qc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "frame_count": len(events), "groups": summaries}, indent=2))


if __name__ == "__main__":
    main()
