"""Shared WIDER/FOMO geometry and inference helpers for Job C."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


FRAME_W, FRAME_H = 320, 240
SCENE_ROI = (40, 0, 240, 240)  # centered square in the 320x240 camera frame
CAMERA_WINDOW_SIZE = 240  # OpenMV cam.window((240, 240)) output consumed by FOMO/crop
MODEL_SIZE = 96
FOMO_STRIDE = 8
IDENTITY_SIDE = 96
INPUT_SCALE = 1.0 / 255.0
INPUT_ZERO_POINT = -128


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_wider(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Parse WIDER FACE's filename/count/10-column-box text format.

    Four local train annotations contain count=0 followed by the all-zero
    placeholder row. These are recorded as zero-box entries and never promoted
    to negative scenes. Unexpected placeholder shapes fail closed.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records: list[dict] = []
    zero_box: list[dict] = []
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        i += 1
        if not name or i >= len(lines):
            raise ValueError(f"Malformed WIDER record near line {i}: {name!r}")
        try:
            count = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Bad face count after {name}: {lines[i]!r}") from exc
        i += 1
        if count < 0:
            raise ValueError(f"Negative face count for {name}")
        boxes = []
        for _ in range(count):
            if i >= len(lines):
                raise ValueError(f"Truncated face rows for {name}")
            values = [int(value) for value in lines[i].split()]
            i += 1
            if len(values) != 10:
                raise ValueError(f"Expected 10 annotation values for {name}, got {values}")
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = values
            boxes.append({
                "xywh": [x, y, w, h], "blur": blur, "expression": expression,
                "illumination": illumination, "invalid": invalid,
                "occlusion": occlusion, "pose": pose,
            })
        row = {"relative_path": name, "event_group": name.split("/", 1)[0],
               "face_count": count, "boxes": boxes}
        if count == 0:
            if i < len(lines) and len(lines[i].split()) == 10 and all(v == "0" for v in lines[i].split()):
                row["placeholder_row"] = lines[i].strip()
                i += 1
            zero_box.append(row)
        else:
            records.append(row)
    return records, zero_box


def fit_geometry(src_w: int, src_h: int, out_w: int = FRAME_W,
                 out_h: int = FRAME_H) -> tuple[float, float, float, float]:
    """Return crop-left, crop-top, crop-width, crop-height used by ImageOps.fit."""
    src_ratio = src_w / src_h
    out_ratio = out_w / out_h
    if src_ratio > out_ratio:
        crop_h = float(src_h)
        crop_w = crop_h * out_ratio
        left = (src_w - crop_w) / 2.0
        top = 0.0
    else:
        crop_w = float(src_w)
        crop_h = crop_w / out_ratio
        left = 0.0
        top = (src_h - crop_h) / 2.0
    return left, top, crop_w, crop_h


def box_to_frame(xywh: Iterable[float], src_w: int, src_h: int) -> tuple[float, float, float, float]:
    """Map a WIDER edge-coordinate box through center-fit to 320x240."""
    x, y, w, h = [float(v) for v in xywh]
    left, top, crop_w, crop_h = fit_geometry(src_w, src_h)
    sx, sy = FRAME_W / crop_w, FRAME_H / crop_h
    x1, y1 = (x - left) * sx, (y - top) * sy
    x2, y2 = (x + w - left) * sx, (y + h - top) * sy
    return x1, y1, x2, y2


def frame_box_to_detector(box_xyxy: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    roi_x, roi_y, roi_w, roi_h = SCENE_ROI
    scale_x, scale_y = MODEL_SIZE / roi_w, MODEL_SIZE / roi_h
    return ((x1 - roi_x) * scale_x, (y1 - roi_y) * scale_y,
            (x2 - roi_x) * scale_x, (y2 - roi_y) * scale_y)


def sensor_box_to_camera_window(box_xyxy: Iterable[float]) -> tuple[float, float, float, float]:
    """Map QVGA sensor-frame edge coordinates into OpenMV's centered 240x240 window."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    roi_x, roi_y, _, _ = SCENE_ROI
    return x1 - roi_x, y1 - roi_y, x2 - roi_x, y2 - roi_y


def camera_window_from_sensor_frame(frame: np.ndarray) -> np.ndarray:
    """Apply the actual cam.window((240, 240)) centered crop to the QVGA frame."""
    if frame.shape[:2] != (FRAME_H, FRAME_W):
        raise ValueError(f"Expected QVGA {FRAME_W}x{FRAME_H} sensor frame, got {frame.shape}")
    x, y, w, h = SCENE_ROI
    window = frame[y:y + h, x:x + w]
    if window.shape[:2] != (CAMERA_WINDOW_SIZE, CAMERA_WINDOW_SIZE):
        raise ValueError("Camera window is outside the sensor frame")
    return np.ascontiguousarray(window)


def frame_to_detector_input(frame: np.ndarray, mode: str = "gray") -> np.ndarray:
    """Exact deployment resize: center 240x240 ROI then bilinear resize to 96."""
    if frame.shape[:2] != (FRAME_H, FRAME_W):
        raise ValueError(f"Expected {FRAME_W}x{FRAME_H} frame, got {frame.shape}")
    x, y, w, h = SCENE_ROI
    roi = frame[y:y + h, x:x + w]
    if roi.shape[:2] != (240, 240):
        raise ValueError("Scene ROI is outside frame")
    if mode == "gray":
        if roi.ndim == 3:
            roi = np.asarray(Image.fromarray(roi.astype(np.uint8), "RGB").convert("L"))
        image = Image.fromarray(roi.astype(np.uint8), "L")
    elif mode == "rgb":
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError("RGB input requested for a non-RGB frame")
        image = Image.fromarray(roi.astype(np.uint8), "RGB")
    else:
        raise ValueError(f"Unsupported input mode {mode}")
    resized = image.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def source_to_frame(image: Image.Image, mode: str = "gray") -> np.ndarray:
    """Decode source image and emulate a 320x240 camera frame with ImageOps.fit."""
    image = ImageOps.exif_transpose(image)
    if mode == "gray":
        image = image.convert("L")
    elif mode == "rgb":
        image = image.convert("RGB")
    else:
        raise ValueError(mode)
    fitted = ImageOps.fit(image, (FRAME_W, FRAME_H), method=Image.Resampling.BILINEAR,
                          centering=(0.5, 0.5))
    return np.asarray(fitted, dtype=np.uint8)


def map_detector_center_to_frame(cx: float, cy: float) -> tuple[float, float]:
    """Map detector-input coordinates to the captured 240x240 camera-window frame."""
    _, _, roi_w, roi_h = SCENE_ROI
    return cx * roi_w / MODEL_SIZE, cy * roi_h / MODEL_SIZE


def map_detector_center_to_sensor_frame(cx: float, cy: float) -> tuple[float, float]:
    """Map detector-input coordinates back to the original 320x240 QVGA sensor frame."""
    roi_x, roi_y, roi_w, roi_h = SCENE_ROI
    return roi_x + cx * roi_w / MODEL_SIZE, roi_y + cy * roi_h / MODEL_SIZE


def identity_crop(frame_gray: np.ndarray, center_xy: Iterable[float], side: int = IDENTITY_SIDE):
    """Return exact gray identity bytes/box; crossing the frame is UNCERTAIN."""
    cx, cy = [float(v) for v in center_xy]
    x = math.floor(cx - side / 2.0 + 0.5)
    y = math.floor(cy - side / 2.0 + 0.5)
    if x < 0 or y < 0 or x + side > frame_gray.shape[1] or y + side > frame_gray.shape[0]:
        return None, (x, y, side, side)
    crop = frame_gray[y:y + side, x:x + side]
    if crop.shape != (side, side):
        return None, (x, y, side, side)
    return np.ascontiguousarray(crop, dtype=np.uint8), (x, y, side, side)


def identity_input_gray96(crop_gray: np.ndarray) -> np.ndarray:
    """Bilinearly resize a valid native grayscale crop to D's raw gray8 input."""
    if crop_gray.ndim != 2 or crop_gray.shape[0] != crop_gray.shape[1]:
        raise ValueError("Identity crop must be a square grayscale array")
    image = Image.fromarray(crop_gray.astype(np.uint8), "L")
    return np.asarray(image.resize((IDENTITY_SIDE, IDENTITY_SIDE), Image.Resampling.BILINEAR), dtype=np.uint8)


def quantize_gray_input(gray96: np.ndarray, scale: float = INPUT_SCALE,
                        zero_point: int = INPUT_ZERO_POINT) -> np.ndarray:
    """Quantize gray bytes using the detector tensor's actual affine input contract."""
    if gray96.shape != (MODEL_SIZE, MODEL_SIZE):
        raise ValueError(f"Expected 96x96 grayscale, got {gray96.shape}")
    if scale <= 0:
        raise ValueError(f"Invalid quantization scale {scale}")
    normalized = gray96.astype(np.float32) / 255.0
    quantized = np.rint(normalized / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def connected_fomo_centroids(probabilities: np.ndarray, threshold: float,
                             stride: int = FOMO_STRIDE) -> list[dict]:
    """8-connected FOMO heatmap regions; output sizes are never interpreted as boxes."""
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim == 4:
        probs = probs[0]
    if probs.ndim != 3 or probs.shape[-1] != 2:
        raise ValueError(f"Expected [grid_h,grid_w,2] FOMO probabilities, got {probs.shape}")
    face = probs[..., 1]
    mask = (face >= threshold) & (face > probs[..., 0])
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    out = []
    for gy in range(h):
        for gx in range(w):
            if not mask[gy, gx] or visited[gy, gx]:
                continue
            stack = [(gy, gx)]
            visited[gy, gx] = True
            cells = []
            while stack:
                yy, xx = stack.pop()
                cells.append((yy, xx))
                for ny in range(max(0, yy - 1), min(h, yy + 2)):
                    for nx in range(max(0, xx - 1), min(w, xx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            weights = np.array([face[yy, xx] for yy, xx in cells], np.float64)
            total = float(weights.sum())
            cx = sum((xx + 0.5) * stride * wt for (yy, xx), wt in zip(cells, weights)) / total
            cy = sum((yy + 0.5) * stride * wt for (yy, xx), wt in zip(cells, weights)) / total
            out.append({"center_input_xy": [float(cx), float(cy)],
                        "center_frame_xy": list(map(float, map_detector_center_to_frame(cx, cy))),
                        "score": float(max(weights)), "cell_count": len(cells)})
    return out


# Copyright (C) 2024 OpenMV, LLC. Adapted from the Fomo/NMS behavior in
# OpenMV v5.0.1 scripts/libraries/ml/ml-postprocessing/edgeimpulse.py and
# ml-core/ml/utils.py. BSD-3-Clause; full notice is in outputs/C/OPENMV_LICENSE.txt.
def openmv_fomo_detections(probabilities: np.ndarray, threshold: float,
                           input_width: int = MODEL_SIZE, input_height: int = MODEL_SIZE,
                           roi: tuple[int, int, int, int] = (0, 0, CAMERA_WINDOW_SIZE, CAMERA_WINDOW_SIZE),
                           w_scale: float = 1.414214, h_scale: float = 1.414214,
                           nms_threshold: float = 0.1, nms_sigma: float = 0.001) -> list[dict]:
    """Reproduce OpenMV 5.0.1's Fomo cell thresholding and NMS box mapping.

    FOMO's tiny NMS rectangles are artificial suppression regions. Their
    centers are useful, but their extents are not treated as face-size boxes.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim == 4:
        probs = probs[0]
    if probs.ndim != 3 or probs.shape[-1] != 2:
        raise ValueError(f"Expected [grid_h,grid_w,2] FOMO probabilities, got {probs.shape}")
    gh, gw, _ = probs.shape
    # OpenMV uses a strict quantized-score > threshold comparison. Dequantizing
    # first is equivalent for positive affine output scales.
    candidates = []
    for flat_index, score in enumerate(probs[..., 1].reshape(-1)):
        if float(score) <= threshold:
            continue
        gy, gx = divmod(flat_index, gw)
        # Fomo uses relative grid centers and widths, then scales by model input.
        x_center_rel, y_center_rel = (gx + 0.5) / gw, (gy + 0.5) / gh
        w_rel, h_rel = (w_scale / gw) * 0.5, (h_scale / gh) * 0.5
        x1 = max(0.0, min((x_center_rel - w_rel) * input_width, input_width))
        y1 = max(0.0, min((y_center_rel - h_rel) * input_height, input_height))
        x2 = max(0.0, min((x_center_rel + w_rel) * input_width, input_width))
        y2 = max(0.0, min((y_center_rel + h_rel) * input_height, input_height))
        ix, iy = int(x1), int(y1)
        iw, ih = int(x2 - x1), int(y2 - y1)
        if iw > 0 and ih > 0 and 0.0 <= float(score) <= 1.0:
            candidates.append({"x": ix, "y": iy, "w": iw, "h": ih,
                               "score": float(score), "class": 1,
                               "grid_xy": [gx, gy]})

    # OpenMV ml.utils.NMS: stable descending-score order followed by Gaussian
    # soft suppression (sigma=.001), filtering scores below .1 after each box.
    candidates.sort(key=lambda box: box["score"], reverse=True)
    sigma_scale = (-1.0 / nms_sigma) if nms_sigma > 0.0 else 0.0
    selected = []
    remaining = candidates
    while remaining:
        chosen = remaining.pop(0)
        selected.append(chosen)
        for box in remaining:
            x1 = max(chosen["x"], box["x"])
            y1 = max(chosen["y"], box["y"])
            x2 = min(chosen["x"] + chosen["w"], box["x"] + box["w"])
            y2 = min(chosen["y"] + chosen["h"], box["y"] + box["h"])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = chosen["w"] * chosen["h"] + box["w"] * box["h"] - inter
            iou = float(inter) / float(union) if union else 0.0
            box["score"] *= math.exp(sigma_scale * iou * iou)
            if box["score"] < nms_threshold:
                box["score"] = 0.0
        remaining = [box for box in remaining if box["score"] > 0.0]
        remaining.sort(key=lambda box: box["score"], reverse=True)

    roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]
    x_scale, y_scale = roi_w / float(input_width), roi_h / float(input_height)
    detections = []
    for box in selected:
        fx = int(box["x"] * x_scale + roi_x)
        fy = int(box["y"] * y_scale + roi_y)
        fw = int(box["w"] * x_scale)
        fh = int(box["h"] * y_scale)
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        detections.append({
            "center_input_xy": [cx * input_width / roi_w, cy * input_height / roi_h],
            "center_frame_xy": [cx, cy],
            "score": float(box["score"]),
            "grid_xy": box["grid_xy"],
            "nms_rect_input_xywh": [box["x"], box["y"], box["w"], box["h"]],
            "nms_rect_window_xywh": [fx, fy, fw, fh],
        })
    return detections


def interpreter_probabilities(interpreter, gray96: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run genuine host TFLite inference with its declared input/output quantization."""
    in_info = interpreter.get_input_details()[0]
    out_info = interpreter.get_output_details()[0]
    if tuple(in_info["shape"].tolist()) != (1, 96, 96, 1):
        raise ValueError(f"Unexpected detector input: {in_info['shape']}")
    if in_info["dtype"] == np.int8:
        scale, zero = in_info["quantization"]
        tensor = quantize_gray_input(gray96, scale, zero)[None, ..., None]
    elif in_info["dtype"] == np.float32:
        tensor = gray96.astype(np.float32)[None, ..., None] / 255.0
    else:
        raise ValueError(f"Unsupported input dtype: {in_info['dtype']}")
    interpreter.set_tensor(in_info["index"], tensor)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_info["index"])
    if out_info["dtype"] == np.int8:
        scale, zero = out_info["quantization"]
        probs = (raw.astype(np.float32) - zero) * scale
    else:
        probs = raw.astype(np.float32)
    return probs, {"input_dtype": str(in_info["dtype"]), "input_quantization": list(in_info["quantization"]),
                   "output_dtype": str(out_info["dtype"]), "output_quantization": list(out_info["quantization"]),
                   "output_shape": list(out_info["shape"])}
