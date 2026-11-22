"""Render private visual QA panels for historical and corrected camera ROIs."""

import ast
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import v6_camera_crop_features48 as replay
from .v5_native_rgb import native_rgb, unpack


OUT = replay.STUDY / "private/camera_corrected_input_v1"


def padded_crop(rgb, x, y, side):
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(rgb.shape[1], x + side), min(rgb.shape[0], y + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = rgb[sy0:sy1, sx0:sx1]
    return out, 1.0 - float(max(0, sx1 - sx0) * max(0, sy1 - sy0)) / float(side * side)


def _pick(rows, ref_boxes):
    dev = [r for r in rows if r["capture_set"] == "optical_development"]
    singles = [r for r in dev if len(r["boxes"]) == 1 and float(r["boxes"][0][1]) >= 0.5]
    picks = {}
    for label in ("K01", "K02", "K03", "K04", "K05"):
        matching = [r for r in singles if r["label"] == label]
        if not matching:
            raise ValueError("no FOMO-single development capture for " + label)
        picks[label] = matching[0]
    unknown = [r for r in singles if r["role"] == "unknown_development" or r["label"] == "OTHER"]
    if unknown:
        picks["OTHER"] = unknown[0]
    multis = [r for r in dev if len(r["boxes"]) > 1]
    if multis:
        picks["MULTI_CLUSTER"] = multis[0]
    ref_areas = [(float(box[2]) * float(box[3]), r) for r in singles
                 for box in [ref_boxes.get(r["request"]["capture_id"])] if box]
    if ref_areas:
        picks["SMALLEST_REFERENCE_FACE_DIAGNOSTIC"] = min(ref_areas, key=lambda x: x[0])[1]
        picks["LARGEST_REFERENCE_FACE_DIAGNOSTIC"] = max(ref_areas, key=lambda x: x[0])[1]
        def offcenter(r):
            x, y, w, h = [float(v) for v in r["boxes"][0][0]]
            cx, cy = x + w / 2, y + h / 2
            return abs(cx - 120) + abs(cy - 120)
        picks["OFF_CENTER"] = max(singles, key=offcenter)
    negative = [r for r in rows if r["capture_set"] == "negative_calibration"]
    for label, key in (("CHECKER", "calibration"), ("BLANK", "blank")):
        match = [r for r in negative if r["request"].get("display", {}).get("kind") == key]
        if match:
            picks[label] = match[0]
    return picks


def _tile(row, sample_label, ref_boxes):
    raw = (row["folder"] / "camera_rgb565.bin").read_bytes()
    pixels = replay.rgb565_array(raw)
    rgb = unpack(pixels)
    boxes = row["boxes"]
    scene = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    scene = scene.copy()
    rects = []
    center = None
    corrected_roi = None
    old_fraction = 1.0
    if boxes:
        for detection in boxes:
            x, y, w, h = [int(round(float(v))) for v in detection[0]]
            rects.append([x, y, w, h, float(detection[1])])
            cv2.rectangle(scene, (x * 128 // 240, y * 128 // 240),
                          ((x + w) * 128 // 240, (y + h) * 128 // 240), (0, 255, 255), 1)
        if len(boxes) == 1:
            rect = boxes[0][0]
            center = replay.fomo_center_camera_px(rect, 240, 240)
            roi = replay.centered_roi_clamped(center, replay.ROI_SIDE_CAMERA_PX, 240, 240)
            corrected_roi = roi
            x0, y0, side, _ = roi
            cv2.rectangle(scene, (x0 * 128 // 240, y0 * 128 // 240),
                          ((x0 + side) * 128 // 240, (y0 + side) * 128 // 240), (0, 0, 255), 1)
            old_center = (center[0] * 2.5, center[1] * 2.5)
            old_x = int(math.floor(old_center[0] - 60))
            old_y = int(math.floor(old_center[1] - 60))
            old_crop, old_fraction = padded_crop(rgb, old_x, old_y, 120)
        else:
            old_crop = np.zeros((120, 120, 3), dtype=np.uint8)
    else:
        old_crop = np.zeros((120, 120, 3), dtype=np.uint8)

    if corrected_roi:
        corrected = rgb[corrected_roi[1]:corrected_roi[1] + 120,
                        corrected_roi[0]:corrected_roi[0] + 120]
        tensor48 = native_rgb(pixels, roi=corrected_roi, width=48, height=48)
        roi_bounds = list(corrected_roi)
        new_pad = 0.0
    else:
        corrected = np.zeros((120, 120, 3), dtype=np.uint8)
        tensor48 = np.zeros((48, 48, 3), dtype=np.uint8)
        roi_bounds = None
        new_pad = None

    if row["request"]["capture_id"] in ref_boxes:
        x, y, w, h = ref_boxes[row["request"]["capture_id"]]
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(scene, (x0 * 128 // 240, y0 * 128 // 240),
                      (x1 * 128 // 240, y1 * 128 // 240), (255, 80, 0), 1)

    panel = Image.new("RGB", (540, 220), (18, 18, 22))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), sample_label + " | " + row["request"]["capture_id"], fill=(245, 245, 245))
    draw.text((6, 20), "yellow=FOMO; red=correct 120px ROI; orange=host diagnostic box", fill=(185, 185, 185))
    tiles = [scene, cv2.resize(old_crop, (128, 128), interpolation=cv2.INTER_NEAREST),
             cv2.resize(corrected, (128, 128), interpolation=cv2.INTER_NEAREST),
             cv2.resize(tensor48, (128, 128), interpolation=cv2.INTER_NEAREST)]
    for i, (caption, value) in enumerate(zip(("scene", "old x2.5 crop", "fixed camera ROI", "48x48 RGB tensor"), tiles)):
        x = 8 + i * 132
        draw.text((x, 39), caption, fill=(220, 220, 220))
        panel.paste(Image.fromarray(value), (x, 58))
    return panel, {
        "sample": sample_label, "capture_id": row["request"]["capture_id"],
        "label": row["label"], "role": row["role"], "fomo_rectangles_camera_xywh": rects,
        "center_xy_camera_px": list(center) if center else None,
        "legacy_center_xy_camera_px_after_wrong_rescale": [center[0] * 2.5, center[1] * 2.5] if center else None,
        "corrected_roi_xywh_camera_px": roi_bounds,
        "legacy_black_padding_fraction": old_fraction, "corrected_black_padding_fraction": new_pad,
        "tensor_shape": [48, 48, 3], "host_reference_box_usage": "diagnostic overlay only" if row["request"]["capture_id"] in ref_boxes else None,
    }


def main():
    rows = replay.load_sets()
    audit = json.loads((replay.STUDY / "audits/2026-11-01-handoff-review/CROP_GEOMETRY_AUDIT.json").read_text())
    ref_boxes = {row["capture_id"]: row["host_reference_face_box"] for row in audit["examples"]}
    picks = _pick(rows, ref_boxes)
    labels = list(picks)
    panels, records = [], []
    for label in labels:
        panel, record = _tile(picks[label], label, ref_boxes)
        panels.append(panel)
        records.append(record)
    cols = 2
    sheet = Image.new("RGB", (cols * 540, ((len(panels) + cols - 1) // cols) * 220), (8, 8, 10))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * 540, (i // cols) * 220))
    OUT.mkdir(parents=True, exist_ok=True)
    image_path = OUT / "CROP_GEOMETRY_CONTACT_SHEET.png"
    sheet.save(image_path)
    report = {
        "schema": "v6-crop-geometry-visual-qc-1", "image": str(image_path),
        "pixels_are_private_camera_recaptures": True,
        "labels": labels, "rows": records,
        "legend": {"yellow": "saved FOMO camera-coordinate rectangles", "red": "corrected clamped 120 camera-pixel ROI", "orange": "existing host-reference box, diagnostic overlay only"},
        "new_roi": {"camera_side_px": 120, "encoder_tensor_hwc": [48, 48, 3], "padding_fraction": 0.0},
        "interpretation": "Image review is a QC gate; it is not identity accuracy evidence. Multiple clusters remain ambiguous and have no identity crop.",
    }
    (OUT / "CROP_GEOMETRY_CONTACT_SHEET.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
