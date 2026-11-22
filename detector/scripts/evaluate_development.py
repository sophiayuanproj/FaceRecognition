"""Compare stock RGB FOMO and every learned float/full-INT8 root on development scenes."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from common import dump_json, identity_input_gray96, openmv_fomo_detections, quantize_gray_input


THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
CROP_SIDES = (80, 96, 112, 128)
ROOT_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105,
              20261106, 20261107, 20261108, 20261109, 20261110)
MATCH_RADIUS_INPUT_PX = 10.0
FACE_SIZE_BINS = ((0.0, 4.0, "<4"), (4.0, 8.0, "4-7"), (8.0, 12.0, "8-11"),
                  (12.0, 20.0, "12-19"), (20.0, 32.0, "20-31"), (32.0, float("inf"), "32+"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def invoke(interpreter, image: np.ndarray) -> tuple[np.ndarray, float]:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    tensor = image[None]
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if in_detail["dtype"] == np.int8:
        scale, zero = in_detail["quantization"]
        if tensor.dtype == np.uint8:
            if tensor.shape[-1] == 1:
                tensor = quantize_gray_input(tensor[0, ..., 0], scale, zero)[None, ..., None]
            else:
                normalized = tensor.astype(np.float32) / 255.0
                tensor = np.clip(np.rint(normalized / scale + zero), -128, 127).astype(np.int8)
        else:
            raise ValueError(f"Unexpected input for int8 model: {tensor.dtype}")
    elif in_detail["dtype"] == np.float32:
        tensor = tensor.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported input dtype {in_detail['dtype']}")
    interpreter.set_tensor(in_detail["index"], tensor)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_detail["index"])
    if out_detail["dtype"] == np.int8:
        scale, zero = out_detail["quantization"]
        output = (raw.astype(np.float32) - zero) * scale
    else:
        output = raw.astype(np.float32)
    return output, 0.0


def gt_centers(row: dict) -> list[dict]:
    out = []
    for box in row.get("face_boxes", []):
        if not box.get("center_visible_in_detector_roi"):
            continue
        cx, cy = [float(v) for v in box["center_detector_xy"]]
        x1, y1, x2, y2 = [float(v) for v in box["xyxy_detector"]]
        out.append({"center": (cx, cy), "box": (x1, y1, x2, y2),
                    "invalid": bool(box.get("invalid", 0)),
                    "max_side_detector_px": max(x2 - x1, y2 - y1),
                    "blur": box.get("blur"), "occlusion": box.get("occlusion")})
    return out


def match(predictions: list[dict], targets: list[dict]) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    pairs = []
    for pi, pred in enumerate(predictions):
        for ti, target in enumerate(targets):
            dx = pred["center_input_xy"][0] - target["center"][0]
            dy = pred["center_input_xy"][1] - target["center"][1]
            dist = math.hypot(dx, dy)
            if dist <= MATCH_RADIUS_INPUT_PX:
                pairs.append((dist, pi, ti))
    pairs.sort()
    used_p, used_t, selected = set(), set(), []
    for dist, pi, ti in pairs:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi); used_t.add(ti); selected.append((pi, ti, dist))
    return selected, used_p, used_t


def crop_geometry(pred: dict, target: dict | None, side: int) -> dict:
    cx, cy = pred["center_frame_xy"]
    x, y = math.floor(cx - side / 2.0 + 0.5), math.floor(cy - side / 2.0 + 0.5)
    valid = x >= 0 and y >= 0 and x + side <= 240 and y + side <= 240
    covered = None
    face_scale = None
    if target is not None:
        x1, y1, x2, y2 = [coord * 2.5 for coord in target["box"]]
        covered = valid and x <= x1 and y <= y1 and x + side >= x2 and y + side >= y2
        face_scale = max(x2 - x1, y2 - y1) / float(side)
    return {"xywh_window": [x, y, side, side], "crop_inside_camera_window": valid,
            "contains_annotated_face_box": covered, "max_face_side_to_crop_ratio": face_scale}


def legacy_crop_geometry(pred: dict, target: dict | None = None) -> dict:
    """Match F's current integer rectangle-center and fixed 96-pixel crop exactly."""
    x, y, width, height = pred["nms_rect_window_xywh"]
    legacy_center = {"center_frame_xy": [x + width // 2, y + height // 2]}
    return crop_geometry(legacy_center, target, 96)


def identity_crop_quality(window: np.ndarray, geometry: dict) -> dict | None:
    if not geometry["crop_inside_camera_window"]:
        return None
    x, y, side, _ = geometry["xywh_window"]
    native = window[y:y + side, x:x + side]
    if native.shape != (side, side):
        return None
    identity = identity_input_gray96(native)
    native_mean, native_stdev = float(native.mean()), float(native.std())
    mean, stdev = float(identity.mean()), float(identity.std())
    a = identity.astype(np.float32)
    lap = (-4.0 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] +
           a[1:-1, :-2] + a[1:-1, 2:])
    lap_var = float(lap.var()) if lap.size else 0.0
    native_quality_pass = 12.0 <= native_mean <= 243.0 and native_stdev >= 5.0
    input_quality_pass = 12.0 <= mean <= 243.0 and stdev >= 5.0
    return {"native_crop_mean_u8": native_mean, "native_crop_stdev_u8": native_stdev,
            "mean_u8": mean, "stdev_u8": stdev, "laplacian_variance": lap_var,
            "native_crop_quality_gate_pass": native_quality_pass,
            "identity_input_quality_gate_pass": input_quality_pass}


def evaluate_probabilities(probability_rows: list[np.ndarray], metadata: list[dict],
                           threshold: float, window_frames: np.ndarray | None = None,
                           crop_selection: str = "candidate", pipeline_crop_side: int = 96) -> dict:
    if crop_selection not in ("candidate", "legacy_current"):
        raise ValueError(f"Unsupported crop selection status policy: {crop_selection}")
    scene_records = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "predictions": 0, "targets": 0,
              "face_scenes": 0, "detected_face_scenes": 0, "no_face_scenes": 0,
              "false_trigger_scenes": 0}
    centroid_errors, valid_prediction_count = [], 0
    pipeline_status_counts = {"NO_FACE": 0, "UNCERTAIN_MULTIPLE_FACES": 0,
                              "UNCERTAIN_CROP_OUTSIDE": 0, "UNCERTAIN_LOW_QUALITY_CROP": 0,
                              "ONE_FACE_CROP_READY": 0}
    crop_stats = {str(side): {"matched": 0, "valid_crop": 0, "face_covered": 0,
                              "all_predictions": 0, "all_predictions_valid": 0,
                              "all_predictions_quality_pass": 0, "matched_quality_pass": 0,
                              "all_predictions_native_quality_pass": 0, "matched_native_quality_pass": 0,
                              "crop_quality_values": [],
                              "scale_ratios": []} for side in CROP_SIDES}
    face_size_stats = {str(side): {label: {"targets": 0, "matched": 0, "face_covered": 0}
                                   for _, _, label in FACE_SIZE_BINS} for side in CROP_SIDES}
    legacy_stat = {"matched": 0, "valid_crop": 0, "face_covered": 0,
                   "all_predictions": 0, "all_predictions_valid": 0,
                   "all_predictions_quality_pass": 0, "matched_quality_pass": 0,
                   "all_predictions_native_quality_pass": 0, "matched_native_quality_pass": 0,
                   "crop_quality_values": [], "scale_ratios": []}
    legacy_face_size_stats = {label: {"targets": 0, "matched": 0, "face_covered": 0}
                              for _, _, label in FACE_SIZE_BINS}
    for row_index, (probs, row) in enumerate(zip(probability_rows, metadata)):
        predictions = openmv_fomo_detections(probs, threshold)
        targets = gt_centers(row)
        for target in targets:
            size = target["max_side_detector_px"]
            bucket = next(label for lo, hi, label in FACE_SIZE_BINS if lo <= size < hi)
            for side in CROP_SIDES:
                face_size_stats[str(side)][bucket]["targets"] += 1
            legacy_face_size_stats[bucket]["targets"] += 1
        pairs, used_p, used_t = match(predictions, targets)
        if not predictions:
            pipeline_status = "NO_FACE"
        elif len(predictions) != 1:
            pipeline_status = "UNCERTAIN_MULTIPLE_FACES"
        else:
            geometry = (legacy_crop_geometry(predictions[0]) if crop_selection == "legacy_current"
                        else crop_geometry(predictions[0], None, pipeline_crop_side))
            if not geometry["crop_inside_camera_window"]:
                pipeline_status = "UNCERTAIN_CROP_OUTSIDE"
            elif window_frames is not None:
                quality = identity_crop_quality(window_frames[row_index], geometry)
                pipeline_status = ("ONE_FACE_CROP_READY" if quality and
                                   quality["native_crop_quality_gate_pass"] and
                                   quality["identity_input_quality_gate_pass"]
                                   else "UNCERTAIN_LOW_QUALITY_CROP")
            else:
                pipeline_status = "ONE_FACE_CROP_READY"
        pipeline_status_counts[pipeline_status] += 1
        tp, fp, fn = len(pairs), len(predictions) - len(used_p), len(targets) - len(used_t)
        counts["tp"] += tp; counts["fp"] += fp; counts["fn"] += fn
        counts["predictions"] += len(predictions); counts["targets"] += len(targets)
        if targets:
            counts["face_scenes"] += 1
            if tp:
                counts["detected_face_scenes"] += 1
        if row.get("is_real_no_face_negative"):
            counts["no_face_scenes"] += 1
            if predictions:
                counts["false_trigger_scenes"] += 1
        for prediction in predictions:
            legacy_geometry = legacy_crop_geometry(prediction)
            legacy_stat["all_predictions"] += 1
            legacy_stat["all_predictions_valid"] += int(legacy_geometry["crop_inside_camera_window"])
            if window_frames is not None:
                legacy_quality = identity_crop_quality(window_frames[row_index], legacy_geometry)
                if legacy_quality is not None:
                    legacy_stat["crop_quality_values"].append(legacy_quality)
                    legacy_stat["all_predictions_quality_pass"] += int(legacy_quality["identity_input_quality_gate_pass"])
                    legacy_stat["all_predictions_native_quality_pass"] += int(legacy_quality["native_crop_quality_gate_pass"])
            for side in CROP_SIDES:
                geometry = crop_geometry(prediction, None, side)
                crop_stats[str(side)]["all_predictions"] += 1
                crop_stats[str(side)]["all_predictions_valid"] += int(geometry["crop_inside_camera_window"])
                if window_frames is not None:
                    quality = identity_crop_quality(window_frames[row_index], geometry)
                    if quality is not None:
                        crop_stats[str(side)]["crop_quality_values"].append(quality)
                        crop_stats[str(side)]["all_predictions_quality_pass"] += int(quality["identity_input_quality_gate_pass"])
                        crop_stats[str(side)]["all_predictions_native_quality_pass"] += int(quality["native_crop_quality_gate_pass"])
        for pi, ti, dist in pairs:
            centroid_errors.append(dist)
            legacy_geometry = legacy_crop_geometry(predictions[pi], targets[ti])
            legacy_stat["matched"] += 1
            legacy_stat["valid_crop"] += int(legacy_geometry["crop_inside_camera_window"])
            legacy_stat["face_covered"] += int(bool(legacy_geometry["contains_annotated_face_box"]))
            legacy_stat["scale_ratios"].append(legacy_geometry["max_face_side_to_crop_ratio"])
            size_bucket = next(label for lo, hi, label in FACE_SIZE_BINS
                               if lo <= targets[ti]["max_side_detector_px"] < hi)
            legacy_face_size_stats[size_bucket]["matched"] += 1
            legacy_face_size_stats[size_bucket]["face_covered"] += int(bool(legacy_geometry["contains_annotated_face_box"]))
            if window_frames is not None:
                legacy_quality = identity_crop_quality(window_frames[row_index], legacy_geometry)
                if legacy_quality is not None:
                    legacy_stat["matched_quality_pass"] += int(legacy_quality["identity_input_quality_gate_pass"])
                    legacy_stat["matched_native_quality_pass"] += int(legacy_quality["native_crop_quality_gate_pass"])
            for side in CROP_SIDES:
                geometry = crop_geometry(predictions[pi], targets[ti], side)
                stat = crop_stats[str(side)]
                stat["matched"] += 1
                stat["valid_crop"] += int(geometry["crop_inside_camera_window"])
                stat["face_covered"] += int(bool(geometry["contains_annotated_face_box"]))
                stat["scale_ratios"].append(geometry["max_face_side_to_crop_ratio"])
                bucket = next(label for lo, hi, label in FACE_SIZE_BINS
                              if lo <= targets[ti]["max_side_detector_px"] < hi)
                face_size_stats[str(side)][bucket]["matched"] += 1
                face_size_stats[str(side)][bucket]["face_covered"] += int(bool(geometry["contains_annotated_face_box"]))
                if window_frames is not None:
                    quality = identity_crop_quality(window_frames[row_index], geometry)
                    if quality is not None:
                        stat["matched_quality_pass"] += int(quality["identity_input_quality_gate_pass"])
                        stat["matched_native_quality_pass"] += int(quality["native_crop_quality_gate_pass"])
        for pi, pred in enumerate(predictions):
            if all(pi != pair[0] for pair in pairs):
                side96 = crop_geometry(pred, None, 96)
                valid_prediction_count += int(side96["crop_inside_camera_window"])
        scene_records.append({"role": row.get("role"), "event_group": row.get("event_group"),
                              "is_real_no_face_negative": row.get("is_real_no_face_negative", False),
                              "target_count": len(targets), "prediction_count": len(predictions),
                              "tp": tp, "fp": fp, "fn": fn, "pipeline_status": pipeline_status,
                              "predictions": predictions,
                              "matched_centroid_errors_input_px": [float(p[2]) for p in pairs]})
    precision = counts["tp"] / max(1, counts["predictions"])
    recall = counts["tp"] / max(1, counts["targets"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    crop_summary = {}
    face_size_summary = {}
    for side, stat in crop_stats.items():
        crop_summary[side] = {
            "matched": stat["matched"], "valid_crop_fraction": stat["valid_crop"] / max(1, stat["matched"]),
            "annotated_face_box_coverage_fraction": stat["face_covered"] / max(1, stat["matched"]),
            "end_to_end_annotated_face_coverage_fraction": stat["face_covered"] / max(1, counts["targets"]),
            "all_prediction_valid_crop_fraction": stat["all_predictions_valid"] / max(1, stat["all_predictions"]),
            "identity_input_quality_gate_pass_fraction": stat["matched_quality_pass"] / max(1, stat["matched"]),
            "native_crop_quality_gate_pass_fraction": stat["matched_native_quality_pass"] / max(1, stat["matched"]),
            "all_prediction_identity_input_quality_gate_pass_fraction": stat["all_predictions_quality_pass"] / max(1, stat["all_predictions"]),
            "all_prediction_native_crop_quality_gate_pass_fraction": stat["all_predictions_native_quality_pass"] / max(1, stat["all_predictions"]),
            "identity_input_mean_stdev_laplacian_quantiles": ({
                "mean_u8_p50": float(np.median([v["mean_u8"] for v in stat["crop_quality_values"]])),
                "stdev_u8_p50": float(np.median([v["stdev_u8"] for v in stat["crop_quality_values"]])),
                "laplacian_variance_p50": float(np.median([v["laplacian_variance"] for v in stat["crop_quality_values"]]))
            } if stat["crop_quality_values"] else None),
            "max_face_side_to_crop_ratio_quantiles": ({
                "p05": float(np.quantile(stat["scale_ratios"], 0.05)),
                "p50": float(np.quantile(stat["scale_ratios"], 0.50)),
                "p95": float(np.quantile(stat["scale_ratios"], 0.95)),
            } if stat["scale_ratios"] else None),
        }
        face_size_summary[side] = {}
        for label, size_stat in face_size_stats[side].items():
            face_size_summary[side][label] = {
                **size_stat,
                "detection_recall": size_stat["matched"] / max(1, size_stat["targets"]),
                "conditional_face_box_coverage": size_stat["face_covered"] / max(1, size_stat["matched"]),
                "end_to_end_face_box_coverage": size_stat["face_covered"] / max(1, size_stat["targets"]),
            }
    crop_summary["legacy_current_fixed_96"] = {
        "matched": legacy_stat["matched"],
        "valid_crop_fraction": legacy_stat["valid_crop"] / max(1, legacy_stat["matched"]),
        "annotated_face_box_coverage_fraction": legacy_stat["face_covered"] / max(1, legacy_stat["matched"]),
        "end_to_end_annotated_face_coverage_fraction": legacy_stat["face_covered"] / max(1, counts["targets"]),
        "all_prediction_valid_crop_fraction": legacy_stat["all_predictions_valid"] / max(1, legacy_stat["all_predictions"]),
        "identity_input_quality_gate_pass_fraction": legacy_stat["matched_quality_pass"] / max(1, legacy_stat["matched"]),
        "native_crop_quality_gate_pass_fraction": legacy_stat["matched_native_quality_pass"] / max(1, legacy_stat["matched"]),
        "all_prediction_identity_input_quality_gate_pass_fraction": legacy_stat["all_predictions_quality_pass"] / max(1, legacy_stat["all_predictions"]),
        "all_prediction_native_crop_quality_gate_pass_fraction": legacy_stat["all_predictions_native_quality_pass"] / max(1, legacy_stat["all_predictions"]),
        "identity_input_mean_stdev_laplacian_quantiles": ({
            "mean_u8_p50": float(np.median([v["mean_u8"] for v in legacy_stat["crop_quality_values"]])),
            "stdev_u8_p50": float(np.median([v["stdev_u8"] for v in legacy_stat["crop_quality_values"]])),
            "laplacian_variance_p50": float(np.median([v["laplacian_variance"] for v in legacy_stat["crop_quality_values"]]))
        } if legacy_stat["crop_quality_values"] else None),
        "max_face_side_to_crop_ratio_quantiles": ({
            "p05": float(np.quantile(legacy_stat["scale_ratios"], 0.05)),
            "p50": float(np.quantile(legacy_stat["scale_ratios"], 0.50)),
            "p95": float(np.quantile(legacy_stat["scale_ratios"], 0.95)),
        } if legacy_stat["scale_ratios"] else None),
    }
    face_size_summary["legacy_current_fixed_96"] = {}
    for label, size_stat in legacy_face_size_stats.items():
        face_size_summary["legacy_current_fixed_96"][label] = {
            **size_stat,
            "detection_recall": size_stat["matched"] / max(1, size_stat["targets"]),
            "conditional_face_box_coverage": size_stat["face_covered"] / max(1, size_stat["matched"]),
            "end_to_end_face_box_coverage": size_stat["face_covered"] / max(1, size_stat["targets"]),
        }
    return {
        "threshold": float(threshold), "counts": counts, "precision": precision,
        "recall": recall, "f1": f1,
        "positive_scene_recall": counts["detected_face_scenes"] / max(1, counts["face_scenes"]),
        "false_trigger_scene_rate": counts["false_trigger_scenes"] / max(1, counts["no_face_scenes"]),
        "pipeline_status_counts": pipeline_status_counts,
        "end_to_end_usable_scene_coverage": pipeline_status_counts["ONE_FACE_CROP_READY"] / max(1, len(metadata)),
        "centroid_error_input_px": {"n": len(centroid_errors),
             "median": float(np.median(centroid_errors)) if centroid_errors else None,
             "p90": float(np.quantile(centroid_errors, 0.90)) if centroid_errors else None},
        "crop_methods": crop_summary,
        "face_size_stratified": face_size_summary,
        "pipeline_crop_policy": crop_selection,
        "pipeline_crop_side_window_px": 96 if crop_selection == "legacy_current" else pipeline_crop_side,
        "valid_unmatched_crop_fraction_side96": valid_prediction_count / max(1, counts["fp"]),
        "scene_records": scene_records,
    }


def run_model(path: Path, gray_images: np.ndarray, rgb_images: np.ndarray,
              is_rgb_model: bool) -> tuple[list[np.ndarray], dict]:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    details = interpreter.get_input_details()[0]
    probs = []
    # Warm-up is excluded from host wall time. This is a host runtime timing only.
    probe = rgb_images[0] if is_rgb_model else gray_images[0][..., None]
    invoke(interpreter, probe)
    import time
    t0 = time.perf_counter()
    for index in range(len(gray_images)):
        image = rgb_images[index] if is_rgb_model else gray_images[index][..., None]
        output, _ = invoke(interpreter, image)
        probs.append(output)
    elapsed = time.perf_counter() - t0
    out_detail = interpreter.get_output_details()[0]
    return probs, {"path": str(path), "file_bytes": path.stat().st_size,
                   "input_shape": details["shape"].tolist(), "input_dtype": str(details["dtype"]),
                   "input_quantization": list(details["quantization"]),
                   "output_shape": out_detail["shape"].tolist(), "output_dtype": str(out_detail["dtype"]),
                   "output_quantization": list(out_detail["quantization"]),
                   "host_interpreter_ms_per_scene": elapsed * 1000 / len(gray_images)}


def choose_threshold(metrics: list[dict]) -> dict:
    acceptable = [m for m in metrics if m["false_trigger_scene_rate"] <= 0.05]
    candidates = acceptable or metrics
    return max(candidates, key=lambda m: (m["f1"], m["recall"], m["threshold"]))


def compare_float_int8(float_probs: list[np.ndarray], int8_probs: list[np.ndarray],
                       metadata: list[dict], threshold: float = 0.40) -> dict:
    changed_counts = changed_presence = 0
    changed_crops = {str(side): 0 for side in CROP_SIDES}
    changed_crops["legacy_current_fixed_96"] = 0
    matched_center_shifts = []
    total_scenes = len(metadata)
    for f_probs, q_probs, row in zip(float_probs, int8_probs, metadata):
        f_preds = openmv_fomo_detections(f_probs, threshold)
        q_preds = openmv_fomo_detections(q_probs, threshold)
        if len(f_preds) != len(q_preds):
            changed_counts += 1
        if bool(f_preds) != bool(q_preds):
            changed_presence += 1
        pairs = []
        for fi, fp in enumerate(f_preds):
            for qi, qp in enumerate(q_preds):
                distance = math.hypot(fp["center_input_xy"][0] - qp["center_input_xy"][0],
                                      fp["center_input_xy"][1] - qp["center_input_xy"][1])
                pairs.append((distance, fi, qi))
        pairs.sort(); used_f, used_q = set(), set()
        for distance, fi, qi in pairs:
            if fi in used_f or qi in used_q:
                continue
            used_f.add(fi); used_q.add(qi); matched_center_shifts.append(distance)
            if (legacy_crop_geometry(f_preds[fi])["xywh_window"][:2] !=
                    legacy_crop_geometry(q_preds[qi])["xywh_window"][:2]):
                changed_crops["legacy_current_fixed_96"] += 1
            for side in CROP_SIDES:
                f_crop = crop_geometry(f_preds[fi], None, side)["xywh_window"][:2]
                q_crop = crop_geometry(q_preds[qi], None, side)["xywh_window"][:2]
                if f_crop != q_crop:
                    changed_crops[str(side)] += 1
    return {"comparison_threshold": threshold, "scenes": total_scenes,
            "scene_detection_count_changed": changed_counts,
            "scene_face_presence_changed": changed_presence,
            "matched_prediction_center_shift_input_px": {
                "n": len(matched_center_shifts),
                "median": float(np.median(matched_center_shifts)) if matched_center_shifts else None,
                "p90": float(np.quantile(matched_center_shifts, 0.90)) if matched_center_shifts else None},
            "matched_crop_origin_changed_count_by_side": changed_crops,
            "matching_note": "nearest cross-export prediction pairs at fixed threshold; count/presence changes are reported separately"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--stock-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache_dir / "development_cache.npz") as cache:
        gray, rgb, windows = cache["gray96"], cache["rgb96"], cache["gray_window"]
    metadata = read_jsonl(args.cache_dir / "development_scene_metadata.jsonl")
    stock_probabilities, stock_audit = run_model(args.stock_model, gray, rgb, True)
    stock_detailed = [evaluate_probabilities(stock_probabilities, metadata, threshold, windows,
                                              crop_selection="legacy_current") for threshold in THRESHOLDS]
    stock_metrics = [{k: v for k, v in metric.items() if k != "scene_records"} for metric in stock_detailed]
    result = {"schema": "job-c-development-model-comparison-1",
              "dataset_roles": {"WIDER": "development event groups only", "PASS": "development uploaders only",
                                "final_scenes_loaded": False},
              "matching": {"rule": "greedy one-to-one OpenMV Fomo rectangle-center match", "radius_input_px": MATCH_RADIUS_INPUT_PX,
                           "one_grid_stride": 8,
                           "postprocessor": "OpenMV 5.0.1 Fomo: strict score threshold, artificial 1.414214/12 NMS cells, Gaussian soft NMS threshold 0.1 sigma 0.001, exact integer rectangle/ROI mapping"},
              "crop_methods": {"native_square_sides_window_px": list(CROP_SIDES),
                               "resized_to_identity_input": [96, 96], "out_of_window": "reject as UNCERTAIN"},
              "conditional_identity_quality": None,
              "conditional_identity_quality_status": "Not measured: no current D handoff or deployable D candidate is available. The historical 64-D embedding model is not a classifier and its weights trained on four of B's five identities, so it was excluded from crop selection. Reported legacy gray mean/stdev checks are not D-conditioned recognition quality.",
              "stock_reference": {"artifact_sha256": "fc3aec94b4b02dc6819db3e2d63bfd8055c9757fb350008538d4c1604aea4cb4",
                                  "fixed_reference_threshold": 0.40, "audit": stock_audit,
                                  "threshold_metrics": stock_metrics},
              "learned_roots": []}
    for arm_dir in sorted(p for p in args.model_dir.iterdir() if p.is_dir()):
        for root_dir in sorted(arm_dir.glob("seed-*")):
            if not (root_dir / "detector_full_int8.tflite").exists():
                continue
            root_result = {"arm": arm_dir.name, "root_seed": int(root_dir.name.split("-")[-1]), "exports": {}}
            probability_by_export = {}
            for export_kind, filename in (("float", "detector_float.tflite"),
                                          ("full_int8", "detector_full_int8.tflite")):
                path = root_dir / filename
                probabilities, audit = run_model(path, gray, rgb, False)
                probability_by_export[export_kind] = probabilities
                detailed_metrics = [evaluate_probabilities(probabilities, metadata, threshold, windows) for threshold in THRESHOLDS]
                metrics = [{k: v for k, v in metric.items() if k != "scene_records"} for metric in detailed_metrics]
                selected = choose_threshold(detailed_metrics)
                root_result["exports"][export_kind] = {"audit": audit,
                    "threshold_metrics": metrics, "selected_development_threshold": selected["threshold"],
                    "selected_metrics": {k: v for k, v in selected.items() if k != "scene_records"}}
            root_result["float_int8_change_at_0p4"] = compare_float_int8(
                probability_by_export["float"], probability_by_export["full_int8"], metadata, 0.40)
            result["learned_roots"].append(root_result)
            print(json.dumps({"event": "development_root_evaluated", "arm": arm_dir.name,
                              "root_seed": root_result["root_seed"],
                              "int8_f1": root_result["exports"]["full_int8"]["selected_metrics"]["f1"]}), flush=True)
    aggregates = {}
    for arm in sorted({row["arm"] for row in result["learned_roots"]}):
        roots = [row for row in result["learned_roots"] if row["arm"] == arm]
        aggregates[arm] = {}
        for export_kind in ("float", "full_int8"):
            keys = ("precision", "recall", "f1", "positive_scene_recall", "false_trigger_scene_rate")
            aggregates[arm][export_kind] = {
                "root_count": len(roots),
                "metrics_median": {key: float(np.median([
                    row["exports"][export_kind]["selected_metrics"][key] for row in roots])) for key in keys},
                "metrics_min_max": {key: [float(np.min([
                    row["exports"][export_kind]["selected_metrics"][key] for row in roots])),
                    float(np.max([row["exports"][export_kind]["selected_metrics"][key] for row in roots]))] for key in keys},
                "development_thresholds": [row["exports"][export_kind]["selected_development_threshold"] for row in roots],
            }
    result["learned_arm_aggregates"] = aggregates
    result["fixed_stock_at_0p4"] = next(m for m in stock_metrics if abs(m["threshold"] - 0.4) < 1e-6)
    result["selection_rule"] = "use development only; require <=5% PASS no-face false-trigger scene rate when any candidate threshold qualifies; maximize face-centroid F1; aggregate ten roots by median; compare crop face-box coverage and scale; no final-scene scores used"
    dump_json(args.out, result)
    print(json.dumps({"event": "development_comparison_complete", "stock_f1_at_0p4": result["fixed_stock_at_0p4"]["f1"],
                      "learned_arms": aggregates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
