#!/usr/bin/env python3
"""Evaluate synthetic corruption curves on matched development roles only."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
RUN_ROOT = STUDY / "runs/E"
os.environ["KERAS_HOME"] = str(STUDY / "models/keras_cache")
sys.path.insert(0, str(ROOT / "FaceRecognition"))

import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

from nicla_study import d_campaign as d
from nicla_study.state import resource_lease
from e_pilot import digest_file, write_json

SEEDS = [20261101 + i for i in range(10)]
CONDITIONS = [
    {"id": "clean", "family": "clean"},
    {"id": "low_light_050", "family": "low_light", "factor": 0.5},
    {"id": "low_light_070", "family": "low_light", "factor": 0.7},
    {"id": "blur_sigma_08", "family": "blur", "sigma": 0.8,
     "implementation": "Pillow ImageFilter.GaussianBlur radius=0.8 pixels"},
    {"id": "blur_sigma_16", "family": "blur", "sigma": 1.6,
     "implementation": "Pillow ImageFilter.GaussianBlur radius=1.6 pixels"},
    {"id": "scale_crop_070", "family": "scale_center_crop", "retained_side_fraction": 0.7},
    {"id": "scale_crop_085", "family": "scale_center_crop", "retained_side_fraction": 0.85},
]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def corrupt(images, condition):
    values = np.asarray(images, dtype=np.uint8)
    family = condition["family"]
    if family == "clean":
        return values.copy()
    result = []
    if family == "low_light":
        factor = float(condition["factor"])
        for image in values:
            result.append(np.clip(np.rint(image.astype(np.float32) * factor), 0, 255).astype(np.uint8))
    elif family == "blur":
        sigma = float(condition["sigma"])
        for image in values:
            plane = Image.fromarray(image[:, :, 0], mode="L")
            blurred = np.asarray(plane.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.uint8)
            result.append(blurred[:, :, None])
    elif family == "scale_center_crop":
        fraction = float(condition["retained_side_fraction"])
        side = max(1, int(round(96 * fraction)))
        start = (96 - side) // 2
        for image in values:
            crop = image[start:start + side, start:start + side, 0]
            resized = np.asarray(Image.fromarray(crop, mode="L").resize((96, 96), Image.Resampling.BILINEAR))
            result.append(resized[:, :, None])
    else:
        raise ValueError("unknown corruption family " + family)
    return np.stack(result).astype(np.uint8)


def infer_batched(interpreter, images, chunk=32):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    signature = np.asarray(input_detail.get("shape_signature", input_detail["shape"]))
    dynamic = len(signature) == 4 and signature[0] == -1
    target_shape = (chunk, 96, 96, 1)
    current_shape = tuple(int(value) for value in input_detail["shape"])
    if dynamic and current_shape != target_shape:
        interpreter.resize_tensor_input(input_detail["index"], list(target_shape), strict=True)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
    outputs = []
    step = chunk if dynamic else 1
    for start in range(0, len(images), step):
        batch = images[start:start + step]
        x = d.gray_float(batch)
        if input_detail["dtype"] in (np.int8, np.uint8):
            scale, zero = input_detail["quantization"]
            lo, hi = (-128, 127) if input_detail["dtype"] == np.int8 else (0, 255)
            x = np.clip(np.rint(x / scale + zero), lo, hi).astype(input_detail["dtype"])
        else:
            x = x.astype(input_detail["dtype"])
        if dynamic and len(batch) < chunk:
            pad_value = input_detail["quantization"][1] if input_detail["dtype"] in (np.int8, np.uint8) else 0
            padded = np.full((chunk, 96, 96, 1), pad_value, dtype=input_detail["dtype"])
            padded[:len(batch)] = x
            x = padded
        interpreter.set_tensor(input_detail["index"], x)
        interpreter.invoke()
        value = interpreter.get_tensor(output_detail["index"])
        if output_detail["dtype"] in (np.int8, np.uint8):
            scale, zero = output_detail["quantization"]
            value = (value.astype(np.float32) - zero) * scale
        outputs.append(value[:len(batch)].astype(np.float32))
    return np.concatenate(outputs, axis=0)


def model_rows():
    rows = []
    for alpha in (0.10, 0.25, 0.50, 0.75):
        arm = "capacity_a%03d" % int(alpha * 100)
        for seed in SEEDS:
            root = RUN_ROOT / "roots" / arm / ("seed-%d" % seed)
            result_path = root / "EXPORT_RESULT.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            export = result.get("exports", {}).get("full_int8_ptq_train_only_rep", {})
            if export.get("status") == "complete_verified":
                rows.append({"family": "capacity", "arm_id": arm, "alpha": alpha, "seed": seed,
                             "path": export["path"], "sha256": export["sha256"],
                             "threshold": export["calibration"]["threshold"], "root_result": str(result_path)})
    for method_dir in sorted((RUN_ROOT / "roots").glob("optimization_*/seed-*")):
        result_path = method_dir / "ROOT_RESULT.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete_verified":
            continue
        export = result.get("exports", {}).get("full_int8_ptq_train_only_rep", {})
        if export.get("status") == "complete_verified":
            rows.append({"family": "optimization", "arm_id": result["method"],
                         "alpha": result["alpha"], "seed": result["root_seed"],
                         "path": export["path"], "sha256": export["sha256"],
                         "threshold": export["calibration"]["threshold"], "root_result": str(result_path)})
    return rows


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    data = d.load_role_arrays(ROOT, d.study_paths(ROOT), "survface_96")
    images = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    rows = model_rows()
    output = RUN_ROOT / "robustness"
    output.mkdir(parents=True, exist_ok=True)
    changed_by_condition = {condition["id"]: corrupt(images, condition) for condition in CONDITIONS}
    previous_by_key = {}
    prior_progress = output / "PROGRESS.json"
    if prior_progress.is_file():
        for previous in json.loads(prior_progress.read_text(encoding="utf-8")).get("models", []):
            previous_by_key[(previous.get("family"), previous.get("arm_id"), previous.get("seed"))] = previous
    report = {"schema": "job-e-synthetic-robustness-1", "status": "in_progress",
              "conditions": CONDITIONS, "data_role": "development_and_unknown_development_only",
              "root_seed_matched": SEEDS, "models_planned": len(rows), "models": [],
              "inference_batch_size": 32, "dynamic_input_resized_once_per_model": True,
              "camera_claim": "synthetic corruptions only; actual camera evidence not claimed",
              "threshold_rule": "reuse each model's threshold fitted on clean calibration roles; do not recalibrate per corruption",
              "final_role_accessed": False}
    with resource_lease(STUDY, "heavy_training", owner="E_synthetic_robustness_all_full_int8_roots", timeout_s=0):
        for model_row in rows:
            key = (model_row["family"], model_row["arm_id"], model_row["seed"])
            filename = model_row["family"] + "_" + model_row["arm_id"] + "_seed-%d.json" % model_row["seed"]
            previous = previous_by_key.get(key)
            if previous and previous.get("status") == "complete_verified" and previous.get("sha256") == model_row["sha256"]:
                report["models"].append(previous)
                atomic_json(output / "PROGRESS.json", report)
                print(json.dumps({"event": "robustness_model", "family": model_row["family"],
                                  "arm_id": model_row["arm_id"], "seed": model_row["seed"],
                                  "status": "already_recorded"}), flush=True)
                continue
            model_record = {**model_row, "status": "complete_verified", "conditions": []}
            try:
                interpreter = tf.lite.Interpreter(model_path=model_row["path"], num_threads=2)
                interpreter.allocate_tensors()
                for condition in CONDITIONS:
                    changed = changed_by_condition[condition["id"]]
                    logits = infer_batched(interpreter, changed)
                    metrics = d.known_macro_and_open_metrics(logits, data["dev_known"], data["dev_unknown"],
                                                              model_row["threshold"], 6)
                    model_record["conditions"].append({"condition": condition, "metrics": metrics,
                                                       "input_count": len(changed), "final_role_accessed": False})
                del interpreter
            except Exception as exc:
                model_record["status"] = "failed_needs_repair"
                model_record["error_type"] = type(exc).__name__
                model_record["error"] = str(exc)[:1800]
                model_record["traceback_tail"] = __import__("traceback").format_exc()[-2500:]
            report["models"].append(model_record)
            atomic_json(output / filename, model_record)
            atomic_json(output / "PROGRESS.json", report)
            print(json.dumps({"event": "robustness_model", "family": model_row["family"],
                              "arm_id": model_row["arm_id"], "seed": model_row["seed"],
                              "status": model_record["status"]}), flush=True)
        report["status"] = "complete_verified" if len(report["models"]) == len(rows) and all(
            row["status"] == "complete_verified" for row in report["models"]) else "partial"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(output / "RESULT.json", report)
    print(json.dumps({"event": "robustness_complete", "status": report["status"],
                      "models": len(report["models"]), "report": str(output / "RESULT.json")}), flush=True)


if __name__ == "__main__":
    main()
