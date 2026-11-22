"""Extract private 48px encoder features from the frozen optical captures.

Raw camera frames stay under project-research-v5/private.  This script stores
only embeddings, hashed provenance, FOMO cluster summaries, and role metadata.
"""

import ast
import datetime
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
CROP_FOMO_PX = (48, 64, 80, 96)
FOMO_TO_SENSOR = 2.5  # 96px detector input on a240px camera frame


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def latest_receipts(name):
    latest = (STUDY / "runs/optical" / name).read_text().strip()
    path = ROOT / latest / "RECEIPTS.json"
    return path, json.loads(path.read_text())


def rgb565_to_rgb(payload):
    if len(payload) != 240 * 240 * 2:
        raise ValueError("camera RGB565 frame is not240x240")
    p = np.frombuffer(payload, dtype="<u2").reshape(240, 240).astype(np.uint32)
    r = (p >> 11) & 31
    g = (p >> 5) & 63
    b = p & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], axis=-1).astype(np.uint8)


def clusters_for(metadata):
    text = metadata.get("fomo", "()")
    raw = ast.literal_eval(text) if isinstance(text, str) and text else text
    return [row for channel in raw for row in channel]


def centered_crop(rgb, center_fomo, side_fomo):
    side = int(round(side_fomo * FOMO_TO_SENSOR))
    cx = float(center_fomo[0]) * FOMO_TO_SENSOR
    cy = float(center_fomo[1]) * FOMO_TO_SENSOR
    x0 = int(np.floor(cx - side / 2.0))
    y0 = int(np.floor(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(240, x1), min(240, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    clipped = 1.0 - float(max(0, sx1 - sx0) * max(0, sy1 - sy0)) / float(side * side)
    return out, clipped


def main():
    cv2.setNumThreads(2)
    adaptation_path, adaptation = latest_receipts("LATEST_ADAPTATION.txt")
    development_path, development = latest_receipts("LATEST_SESSION.txt")
    negative_path, negative = latest_receipts("LATEST_NEGATIVE_CALIBRATION.txt")
    if (len(adaptation), len(development), len(negative)) != (135, 100, 20):
        raise ValueError("expected frozen135/100/20 role-preserving capture sets")
    model_path = STUDY / "runs/operator_fused_encoder48/int8.tflite"
    preflight_path = STUDY / "reports/M5/operator_fused_encoder48_PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text())
    model_sha = sha256(model_path)
    if preflight.get("model_sha256") != model_sha or preflight.get("successful_camera_invocations", 0) < 10:
        raise ValueError("48px model does not match ten-block camera resource preflight")

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if inp["shape"].tolist() != [1, 48, 48, 3] or out["shape"].tolist() != [1, 128]:
        raise ValueError("unexpected 48px encoder tensor contract")

    records = []
    feature_rows = []
    all_receipts = [
        ("camera_train_calibration", adaptation),
        ("optical_development", development),
        ("negative_calibration", negative),
    ]
    for set_name, receipts in all_receipts:
        for entry in receipts:
            request = entry["request"]
            meta = entry["metadata"]
            folder = ROOT / entry["directory"]
            raw_path = folder / "camera_rgb565.bin"
            raw = raw_path.read_bytes()
            file_row = next(row for row in entry["files"] if row["name"] == "camera_rgb565.bin")
            if len(raw) != file_row["bytes"] or sha256_bytes(raw) != file_row["sha256"]:
                raise ValueError("capture frame failed its receipt hash: " + request["capture_id"])
            rgb = rgb565_to_rgb(raw)
            boxes = clusters_for(meta)
            scores = [float(item[1]) for item in boxes]
            one_cluster = len(boxes) == 1 and scores[0] >= 0.5
            if set_name == "negative_calibration" and boxes:
                raise ValueError("new blank/checker negative calibration unexpectedly triggered FOMO")

            label = request.get("expected_truth")
            source = request.get("source", {})
            if label is None:
                label = source.get("label", "OTHER")
            if set_name == "camera_train_calibration":
                role = source.get("role")
                phase = request.get("phase")
                if role not in ("train", "other_train", "threshold_calibration", "unknown_calibration"):
                    raise ValueError("unexpected camera adaptation role")
            elif set_name == "optical_development":
                role = source.get("role", label)
                phase = "development"
            else:
                role = "negative_control"
                phase = "negative_calibration"

            center = None
            if one_cluster:
                rect = boxes[0][0]
                center = (float(rect[0]) + float(rect[2]) / 2.0, float(rect[1]) + float(rect[3]) / 2.0)
            by_size = []
            for crop_side in CROP_FOMO_PX:
                if not one_cluster:
                    by_size.append(np.zeros(128, dtype=np.float32))
                    continue
                crop, _ = centered_crop(rgb, center, crop_side)
                pixels = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_LINEAR)
                values = (pixels.astype(np.float32) - 127.5) / 128.0
                scale, zero = inp["quantization"]
                quantized = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
                interpreter.set_tensor(inp["index"], quantized[None])
                interpreter.invoke()
                raw_out = interpreter.get_tensor(out["index"])[0]
                scale, zero = out["quantization"]
                vector = (raw_out.astype(np.float32) - zero) * scale
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-8:
                    raise ValueError("invalid camera embedding: " + request["capture_id"])
                by_size.append(vector / norm)

            row = {
                "capture_id": request["capture_id"],
                "capture_set": set_name,
                "session": request.get("session"),
                "block": int(request.get("block", -1)),
                "phase": phase,
                "role": role,
                "label": label,
                "source_record_id": source.get("record_id"),
                "source_group": source.get("source_group"),
                "source_identity": source.get("source_identity"),
                "camera_frame_sha256": file_row["sha256"],
                "capture_receipt_sha256": sha256(folder / "RECEIPT.json"),
                "fomo_clusters": len(boxes),
                "fomo_scores": scores,
                "fomo_single_crop_usable": bool(one_cluster),
                "fomo_program_sha256": entry.get("program_sha256"),
                "frame_geometry": [int(meta["width"]), int(meta["height"])],
            }
            records.append(row)
            feature_rows.append(by_size)

    features = np.asarray(feature_rows, dtype=np.float32)
    if features.shape != (255, len(CROP_FOMO_PX), 128):
        raise ValueError("unexpected camera feature matrix shape: %r" % (features.shape,))
    private = STUDY / "private/camera-adapt-features"
    private.mkdir(parents=True, exist_ok=True)
    feature_path = private / "fomo48_crop_embeddings_int8.npz"
    np.savez_compressed(feature_path, features=features, crop_sizes_fomo_px=np.asarray(CROP_FOMO_PX, dtype=np.int16))
    records_path = STUDY / "runs/camera_adapt_task48/CAPTURE_FEATURE_INDEX.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records))
    metadata = {
        "schema": "v5-camera-crop-features48-1",
        "role_preserving_captures": {"adaptation": len(adaptation), "development": len(development), "negative_calibration": len(negative)},
        "crop_sizes_fomo_input_px": list(CROP_FOMO_PX),
        "fomo_acceptance": "exactly one cluster and score>=0.5; multiple clusters retained as ambiguous and no crop emitted",
        "negative_control_result": {"n": len(negative), "fomo_false_triggers": 0},
        "model_sha256": model_sha,
        "host_backend": "TF2.10.1 BUILTIN_REF",
        "input_shape": inp["shape"].tolist(),
        "input_quantization": [float(inp["quantization"][0]), int(inp["quantization"][1])],
        "output_quantization": [float(out["quantization"][0]), int(out["quantization"][1])],
        "transform": "RGB565 bit-replicated to RGB888; center crop scaled96-to240; black padding; OpenCV INTER_LINEAR to48; (pixel-127.5)/128; np.rint input quantization; dequantized L2 output",
        "transform_limit": "host replay only; exact native variable ROI crop/48px resize/input-byte parity has not been checked on Nicla",
        "rows": len(records),
        "single_cluster_crop_rows": sum(row["fomo_single_crop_usable"] for row in records),
        "feature_cache_sha256": sha256(feature_path),
        "feature_cache_bytes": feature_path.stat().st_size,
        "feature_cache_private_path": "work/project-research-v5/private/camera-adapt-features/fomo48_crop_embeddings_int8.npz",
        "capture_index_sha256": sha256(records_path),
        "receipt_manifests": {
            "adaptation": sha256(adaptation_path),
            "development": sha256(development_path),
            "negative_calibration": sha256(negative_path),
        },
        "future_final_used": False,
        "board_qualified": False,
    }
    dump(STUDY / "runs/camera_adapt_task48/CAPTURE_FEATURES.json", metadata)
    print(json.dumps(metadata, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
