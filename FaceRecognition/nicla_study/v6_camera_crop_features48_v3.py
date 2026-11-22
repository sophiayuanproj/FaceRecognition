"""Corrected host replay with a conservative logged near-duplicate peak rule.

Outputs are isolated from the invalid v5 crop-feature cache.  The camera ROI
side is expressed directly in 240px camera pixels; the final encoder tensor is
48x48 RGB888 quantized to the model's signed-int8 input contract.
"""

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
FIRMWARE = Path(__file__).resolve().parents[1] / "firmware"
sys.path.insert(0, str(FIRMWARE))
from camera_geometry import centered_roi_clamped, fomo_center_camera_px, roi_clip_plan
from identity_contracts import strict_score_pass, merge_near_duplicate_fomo_peaks
from open_set_decision import (accepts, accepts_quantized, decide, quantized_score,
                               threshold_above_max, threshold_above_quantized_max,
                               validate_group_rows, validate_probabilities)
from .v5_native_rgb import native_rgb

ROI_SIDE_CAMERA_PX = 120
TENSOR_SIDE_PX = 48
CAMERA_WIDTH_PX = 240
CAMERA_HEIGHT_PX = 240
OUT = STUDY / "runs/camera_corrected_input_v3"
PRIVATE = STUDY / "private/camera_corrected_input_v3"
DETECTION_THRESHOLD = 0.5
MERGE_SPAN_CAMERA_PX = 48.0


def sha256(payload):
    if isinstance(payload, Path):
        payload = payload.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def latest_receipts(name):
    latest = (STUDY / "runs/optical" / name).read_text().strip()
    path = ROOT / latest / "RECEIPTS.json"
    return path, json.loads(path.read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clusters_for(metadata):
    raw = metadata.get("fomo", "()")
    raw = ast.literal_eval(raw) if isinstance(raw, str) and raw else raw
    return [row for channel in raw for row in channel]


def rgb565_array(payload, width=CAMERA_WIDTH_PX, height=CAMERA_HEIGHT_PX):
    if len(payload) != width * height * 2:
        raise ValueError("camera_rgb565.bin length does not match receipt geometry")
    return np.frombuffer(payload, dtype="<u2").reshape(height, width)


def model_contract(model_path):
    it = tf.lite.Interpreter(
        model_path=str(model_path), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    if inp["shape"].tolist() != [1, TENSOR_SIDE_PX, TENSOR_SIDE_PX, 3] or inp["dtype"] != np.int8:
        raise ValueError("unexpected encoder input contract")
    if out["shape"].tolist() != [1, 128] or out["dtype"] != np.int8:
        raise ValueError("unexpected encoder output contract")
    scale, zero = inp["quantization"]
    if not scale or scale <= 0:
        raise ValueError("invalid input quantization")
    pixels = np.arange(256, dtype=np.float64)
    lut = np.clip(np.rint(((pixels - 127.5) / 128.0) / float(scale)) + int(zero), -128, 127).astype(np.int8)
    return it, inp, out, lut


def geometry_fixtures():
    """Ten seeded coordinate/axis/bounds cases, independent of learned scores."""
    rows = []
    for seed in range(10):
        rng = np.random.RandomState(seed + 7701)
        width = int(rng.randint(180, 641))
        height = int(rng.randint(180, 641))
        side = int(rng.randint(32, min(width, height) + 1))
        x = float(rng.randint(0, width - 1))
        y = float(rng.randint(0, height - 1))
        w = float(rng.randint(1, width - int(x) + 1))
        h = float(rng.randint(1, height - int(y) + 1))
        center = fomo_center_camera_px((x, y, w, h), width, height)
        roi = centered_roi_clamped(center, side, width, height)
        rx, ry, rw, rh = roi
        assert rw == rh == side and 0 <= rx <= width - side and 0 <= ry <= height - side
        assert center == (x + w / 2.0, y + h / 2.0)
        plan = roi_clip_plan(roi, width, height)
        assert plan["black_padding_fraction"] == 0.0
        row, col = (int(center[1]), int(center[0]))
        assert row < height and col < width
        rows.append({"seed": seed, "frame_xy": [width, height], "fomo_rect_camera_xywh": [x, y, w, h],
                     "center_xy_camera_px": list(center), "roi_xywh_camera_px": list(roi),
                     "row_col": [row, col], "padding_fraction": 0.0, "passed": True})
    return rows


def load_sets():
    manifests = [
        ("camera_train_calibration", "LATEST_ADAPTATION.txt", 135),
        ("optical_development", "LATEST_SESSION.txt", 100),
        ("negative_calibration", "LATEST_NEGATIVE_CALIBRATION.txt", 20),
    ]
    all_rows = []
    for set_name, pointer, expected in manifests:
        receipt_path, receipts = latest_receipts(pointer)
        if len(receipts) != expected:
            raise ValueError("unexpected frozen capture count for " + set_name)
        for entry in receipts:
            request, meta = entry["request"], entry["metadata"]
            folder = ROOT / entry["directory"]
            raw_path = folder / "camera_rgb565.bin"
            payload = raw_path.read_bytes()
            frame_file = next(row for row in entry["files"] if row["name"] == "camera_rgb565.bin")
            if len(payload) != frame_file["bytes"] or sha256(payload) != frame_file["sha256"]:
                raise ValueError("camera frame hash mismatch: " + request["capture_id"])
            if [int(meta["width"]), int(meta["height"])] != [CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX]:
                raise ValueError("unexpected saved camera frame size")
            raw_boxes = clusters_for(meta)
            boxes = [item for item in raw_boxes if strict_score_pass(item[1], DETECTION_THRESHOLD)]
            label = request.get("expected_truth")
            source = request.get("source", {})
            if label is None:
                label = source.get("label", "OTHER")
            if set_name == "camera_train_calibration":
                role = source.get("role")
                if role not in ("train", "other_train", "threshold_calibration", "unknown_calibration"):
                    raise ValueError("unexpected adaptation role")
                phase = request.get("phase")
            elif set_name == "optical_development":
                role, phase = source.get("role", label), "development"
            else:
                role, phase = "negative_control", "negative_calibration"
            all_rows.append({"entry": entry, "folder": folder, "request": request, "metadata": meta,
                             "source": source, "capture_set": set_name, "label": label,
                             "role": role, "phase": phase, "boxes": boxes, "raw_boxes": raw_boxes,
                             "camera_frame_sha256": frame_file["sha256"],
                             "capture_receipt_sha256": sha256_file(folder / "RECEIPT.json"),
                             "manifest_sha256": sha256_file(receipt_path)})
    return all_rows


def extract(rows):
    model_path = STUDY / "runs/operator_fused_encoder48/int8.tflite"
    preflight = json.loads((STUDY / "reports/M5/operator_fused_encoder48_PREFLIGHT.json").read_text())
    model_sha = sha256(model_path)
    if preflight.get("model_sha256") != model_sha or preflight.get("successful_camera_invocations", 0) < 10:
        raise ValueError("encoder does not match the existing ten-block resource preflight")
    it, inp, out, lut = model_contract(model_path)
    feature_rows, quantized_rows, rgb_rows, index = [], [], [], []
    for row in rows:
        raw = (row["folder"] / "camera_rgb565.bin").read_bytes()
        pixels565 = rgb565_array(raw)
        clusters = row["boxes"]
        raw_clusters = row["raw_boxes"]
        scores = [float(detection[1]) for detection in clusters]
        raw_scores = [float(detection[1]) for detection in raw_clusters]
        resolved_clusters, merged_duplicate_peaks, merge_span = merge_near_duplicate_fomo_peaks(
            [{"center": fomo_center_camera_px(detection[0], CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX),
              "score": float(detection[1]), "rect": [float(v) for v in detection[0]]}
             for detection in clusters], MERGE_SPAN_CAMERA_PX, enabled=True)
        usable = len(resolved_clusters) == 1
        crop_quality = ("fomo_duplicate_peaks_merged" if merged_duplicate_peaks else
                        "fomo_single_cluster" if usable else
                        "multiple_clusters" if len(clusters) > 1 else "no_face")
        roi = None
        center = None
        if usable:
            rect = resolved_clusters[0]["rect"]
            center = fomo_center_camera_px(rect, CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX)
            roi = centered_roi_clamped(center, ROI_SIDE_CAMERA_PX, CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX)
            rgb = native_rgb(pixels565, roi=roi, width=TENSOR_SIDE_PX, height=TENSOR_SIDE_PX)
            if rgb.shape != (48, 48, 3):
                raise ValueError("wrong resized tensor geometry")
            if not np.any(rgb):
                crop_quality = "all_black_input"
                usable = False
            else:
                values = (rgb.astype(np.float32) - 127.5) / 128.0
                scale, zero = inp["quantization"]
                q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
                it.set_tensor(inp["index"], q[None])
                it.invoke()
                raw_vector = it.get_tensor(out["index"])[0]
                scale, zero = out["quantization"]
                vector = (raw_vector.astype(np.float32) - zero) * scale
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-8:
                    crop_quality = "invalid_embedding"
                    usable = False
                    vector = np.zeros(128, dtype=np.float32)
                else:
                    vector = vector / norm
        else:
            rgb = np.zeros((48, 48, 3), dtype=np.uint8)
            q = np.zeros((48, 48, 3), dtype=np.int8)
            vector = np.zeros(128, dtype=np.float32)
        if not usable:
            q = np.zeros((48, 48, 3), dtype=np.int8)
        rgb_rows.append(rgb)
        quantized_rows.append(q)
        feature_rows.append(vector)
        index.append({
            "capture_id": row["request"]["capture_id"], "capture_set": row["capture_set"],
            "session": row["request"].get("session"), "block": int(row["request"].get("block", -1)),
            "phase": row["phase"], "role": row["role"], "label": row["label"],
            "source_record_id": row["source"].get("record_id"),
            "source_group": row["source"].get("source_group"),
            "source_identity": row["source"].get("source_identity"),
            "camera_frame_sha256": row["camera_frame_sha256"],
            "capture_receipt_sha256": row["capture_receipt_sha256"],
            "manifest_sha256": row["manifest_sha256"], "frame_geometry_xy": [240, 240],
            "pixel_format": "RGB565 little-endian", "fomo_coordinate_space": "camera_xywh_px",
            "fomo_clusters": len(clusters), "fomo_scores": scores,
            "fomo_resolved_clusters": len(resolved_clusters),
            "fomo_duplicate_peaks_merged": bool(merged_duplicate_peaks),
            "fomo_merge_span_camera_px": merge_span,
            "fomo_raw_clusters": len(raw_clusters), "fomo_raw_scores": raw_scores,
            "fomo_detection_threshold": DETECTION_THRESHOLD,
            "fomo_threshold_comparison": "strict_gt_float32",
            "usable_for_identity": bool(usable), "quality_status": crop_quality,
            "center_xy_camera_px": list(center) if center else None,
            "roi_xywh_camera_px": list(roi) if roi else None,
            "roi_side_camera_px": ROI_SIDE_CAMERA_PX if roi else None,
            "tensor_shape_hwc": [48, 48, 3], "tensor_color_order": "RGB",
            "rgb888_tensor_sha256": sha256(rgb.tobytes()),
            "int8_tensor_sha256": sha256(q.tobytes()),
        })
    features = np.asarray(feature_rows, dtype=np.float32)
    rgb_tensor = np.asarray(rgb_rows, dtype=np.uint8)
    int8_tensor = np.asarray(quantized_rows, dtype=np.int8)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    feature_path, rgb_path, int8_path = PRIVATE / "features.npz", PRIVATE / "rgb888_inputs.npy", PRIVATE / "int8_inputs.npy"
    np.savez_compressed(feature_path, features=features)
    np.save(rgb_path, rgb_tensor)
    np.save(int8_path, int8_tensor)
    index_path = OUT / "CAPTURE_INDEX.jsonl"
    index_path.write_text("".join(json.dumps(v, sort_keys=True, allow_nan=False) + "\n" for v in index))
    metadata = {
        "schema": "v6-camera-crop-inputs-3", "status": "score_gated_inputs_with_logged_duplicate_peak_policy",
        "parent_input_experiment": "runs/camera_corrected_input_v2/INPUTS.json",
        "parent_invalid_experiment": "runs/camera_adapt_task48; generated by v5_camera_crop_features48.py",
        "roi_side_camera_px": ROI_SIDE_CAMERA_PX, "encoder_tensor_hwc": [48, 48, 3],
        "coordinate_contract": "OpenMV FOMO postprocessed rectangles are camera_xywh_px; no 96-to-240 center scaling",
        "roi_policy": "fixed 120px camera ROI centered on the highest-scoring peak only after an explicit 48px pair-center merge; shifted inward at frame border; no padding",
        "detector_threshold": DETECTION_THRESHOLD, "detector_comparison": "strict_gt_float32",
        "duplicate_peak_merge_policy": "merge exactly two above-threshold FOMO peaks only when camera-pixel center distance <=48; retain higher-score peak and record the merge; otherwise MULTIPLE_OR_AMBIGUOUS",
        "raw_fomo_clusters_preserved_in_capture_index": True,
        "input_transform": "saved RGB565 little-endian -> OpenMV 5.0.1 native_rgb emulation with 16.16 geometry and RGB565 bilinear rounding -> RGB888 -> (pixel-127.5)/128 -> model affine int8 quantization, ties-to-even",
        "host_backend": "TensorFlow 2.10.1 BUILTIN_REF", "model_sha256": model_sha,
        "input_shape": inp["shape"].tolist(), "input_quantization": [float(inp["quantization"][0]), int(inp["quantization"][1])],
        "output_quantization": [float(out["quantization"][0]), int(out["quantization"][1])],
        "capture_rows": len(index), "usable_single_cluster_rows": sum(v["usable_for_identity"] for v in index),
        "duplicate_peak_merged_rows": sum(v["fomo_duplicate_peaks_merged"] for v in index),
        "quality_counts": {key: sum(v["quality_status"] == key for v in index) for key in sorted(set(v["quality_status"] for v in index))},
        "features_sha256": sha256_file(feature_path), "rgb888_inputs_sha256": sha256_file(rgb_path),
        "int8_inputs_sha256": sha256_file(int8_path), "capture_index_sha256": sha256_file(index_path),
        "future_final_used": False,
        "board_input_parity": "the v6 single-cluster camera-to-encoder tensor path is exact; the duplicate-peak merge branch has not yet been observed on-device",
        "geometry_fixtures": geometry_fixtures(), "geometry_fixture_passed": 10,
        "face_box_fraction": "not inferred from FOMO clusters; reference boxes are kept only as a separately labeled diagnostic",
    }
    dump(OUT / "INPUTS.json", metadata)
    return metadata, index, rgb_tensor, int8_tensor, features


def decisions_for_tests():
    """Seeded score, quantization, serialization and row-fault controls."""
    import struct
    base = 0.19614700973033905
    threshold = threshold_above_max([base])
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", float(x)))[0]
    checks = []
    checks.append(("float64-nextafter-collapses", f32(float(np.nextafter(np.float64(base), np.float64(np.inf)))) == base))
    checks.append(("float32-threshold-above", threshold > base))
    checks.append(("tie-rejected", not accepts(base, threshold)))
    checks.append(("threshold-equality-accepted", accepts(threshold, threshold)))
    next_value = np.nextafter(np.float32(base), np.float32(np.inf))
    checks.append(("adjacent-float32-rejected", not accepts(base, threshold) and accepts(next_value, threshold)))
    step = 1.0 / 127.0
    qi = quantized_score(base, step)
    qthreshold = threshold_above_quantized_max([qi, qi])
    checks.append(("quantized-score-step", accepts_quantized(qi, qthreshold) is False))
    checks.append(("quantized-threshold-next-step", accepts_quantized(qthreshold, qthreshold)))
    checks.append(("scalar-vector-agreement", [accepts(v, threshold) for v in [base, threshold]] == [False, True]))
    checks.append(("serialized-agreement", accepts(float(json.loads(json.dumps({"score": base}))["score"]), threshold) is False))
    checks.append(("missing-detection-is-no-face", decide(False, 0, 1.0, threshold) == "NO_FACE"))
    checks.append(("multiple-clusters-uncertain", decide(True, 0, 1.0, threshold, multiple=True) == "UNCERTAIN"))
    checks.append(("poor-quality-uncertain", decide(True, 0, 1.0, threshold, quality_ok=False) == "UNCERTAIN"))
    checks.append(("valid-known-decision", decide(True, 0, threshold, threshold) == "K01"))
    checks.append(("below-threshold-uncertain", decide(True, 0, base, threshold) == "UNCERTAIN"))
    checks.append(("other-output", decide(True, 5, 0.0, threshold) == "OTHER"))
    checks.append(("probability-vector-valid", sum(validate_probabilities([.1,.1,.1,.1,.1,.5])) > 0))
    for name, probabilities in (("probability-nan", [.2,.2,.2,.2,.1,float("nan")]),
                                ("probability-out-of-range", [.3,.3,.3,.2,-.1,0]),
                                ("probability-bad-count", [.2,.2,.2,.2,.2])):
        try:
            validate_probabilities(probabilities)
            checks.append((name + "-rejected", False))
        except ValueError:
            checks.append((name + "-rejected", True))
    try:
        validate_group_rows([{"record_id":"a","group_id":"p1"}], expected_count=2)
        checks.append(("row-count-mismatch-rejected", False))
    except ValueError:
        checks.append(("row-count-mismatch-rejected", True))
    try:
        validate_group_rows([{"record_id":"a","group_id":"p1"}], expected_groups=2)
        checks.append(("group-count-mismatch-rejected", False))
    except ValueError:
        checks.append(("group-count-mismatch-rejected", True))
    for name, value in (("nan", float("nan")), ("positive-inf", float("inf")), ("negative-inf", float("-inf"))):
        try:
            accepts(value, threshold)
            checks.append((name + "-rejected", False))
        except ValueError:
            checks.append((name + "-rejected", True))
    try:
        threshold_above_max([])
        checks.append(("empty-calibration-rejected", False))
    except ValueError:
        checks.append(("empty-calibration-rejected", True))
    if len(checks) < 10 or not all(value for _, value in checks):
        raise AssertionError("threshold decision contract failure: %r" % (checks,))
    return [{"case": name, "passed": bool(value)} for name, value in checks]


def main():
    inputs = load_sets()
    metadata, index, rgb, quantized, features = extract(inputs)
    decision_cases = decisions_for_tests()
    dump(OUT / "DECISION_CONTRACT.json", {
        "schema": "float32-open-set-decision-contract-2",
        "contract": "scores and thresholds are rounded to binary32; quantized score domains use integer thresholds; missing detections and multi-cluster/poor-quality inputs map to NO_FACE or UNCERTAIN",
        "cases": decision_cases, "passed": len(decision_cases), "failed": sum(not r["passed"] for r in decision_cases),
        "threshold_example": threshold_above_max([0.19614700973033905]),
    })
    print(json.dumps(metadata, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
