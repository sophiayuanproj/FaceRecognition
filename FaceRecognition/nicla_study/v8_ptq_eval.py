"""Post-training full-INT8 export and reference-kernel replay for the v8 alpha0.25 scout.

Representative images come only from the frozen LFW train/other_train roles.
Calibration and development are opened only after the host checkpoint is frozen.
This script never opens a device or any future-final image/frame path.
"""
from __future__ import annotations

import hashlib
import h5py
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/nicla_study"))
from nicla_study import d_campaign as d
from nicla_study import v7_accuracy_runner as runner

RUN_ID = "v8_lfw_source_mnv1_a025_ei_scout57101_launch01"
RUN = runner.RUN_ROOT / RUN_ID
CONFIG = runner.RUN_ROOT / "configs" / (RUN_ID + ".json")
PREFLIGHT = runner.RUN_ROOT / "preflight" / (RUN_ID + ".json")
KNOWN = set(runner.KNOWN)
EVAL_ROLES = {"threshold_calibration", "unknown_calibration", "development", "unknown_development"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected_weights(model, path: Path) -> dict:
    """Use Keras topology first, then the same exact named-tensor fallback as D."""
    try:
        model.load_weights(str(path))
        return {"status": "verified", "method": "keras_hdf5_topological"}
    except ValueError as exc:
        loaded_paths = []
        with h5py.File(path, "r") as source:
            datasets = {}
            def collect(name, item):
                if isinstance(item, h5py.Dataset):
                    datasets[name] = np.asarray(item)
            source.visititems(collect)
            for variable in model.weights:
                matches = [(name, value) for name, value in datasets.items()
                           if name.endswith(variable.name)]
                if not matches:
                    raise ValueError("selected host checkpoint is missing " + variable.name) from exc
                expected = tuple(int(value) for value in variable.shape)
                if any(tuple(value.shape) != expected for _, value in matches):
                    raise ValueError("selected host checkpoint shape mismatch for " + variable.name) from exc
                reference = matches[0][1]
                if any(not np.array_equal(reference, value) for _, value in matches[1:]):
                    raise ValueError("selected checkpoint has conflicting copies of " + variable.name) from exc
                variable.assign(reference)
                loaded_paths.append(matches[0][0])
        if len(loaded_paths) != len(model.weights) or len(set(loaded_paths)) != len(model.weights):
            raise ValueError("selected checkpoint tensor assignment was incomplete or duplicated") from exc
        return {"status": "verified", "method": "exact_named_hdf5_tensor_assignment",
                "tensor_count": len(loaded_paths), "native_loader_error": str(exc)}


def as_eval_row(row: dict) -> dict:
    return {
        "record_id": str(row["record_id"]), "label": str(row["label"]),
        "role": str(row["role"]), "source_identity": str(row.get("source_identity", "")),
        "source_group": str(row.get("source_group", "")),
    }


def source_metrics(model, interpreter, source_rows: list[dict], image_map: dict[str, np.ndarray]) -> dict:
    cal_rows = [as_eval_row(row) for row in source_rows if row["role"] in ("threshold_calibration", "unknown_calibration")]
    dev_rows = [as_eval_row(row) for row in source_rows if row["role"] in ("development", "unknown_development")]
    cal_x = np.stack([image_map[row["record_id"]] for row in cal_rows]).astype(np.uint8)
    dev_x = np.stack([image_map[row["record_id"]] for row in dev_rows]).astype(np.uint8)
    float_cal = runner.eval_logits(model, cal_x)
    float_dev = runner.eval_logits(model, dev_x)
    int8_cal = tflite_reference_logits(interpreter, cal_x)
    int8_dev = tflite_reference_logits(interpreter, dev_x)
    float_threshold = runner.threshold_from_calibration(cal_rows, float_cal)
    int8_threshold = runner.threshold_from_calibration(cal_rows, int8_cal)
    float_result = runner.evaluate_open_set(dev_rows, float_dev, float_threshold)
    int8_result = runner.evaluate_open_set(dev_rows, int8_dev, int8_threshold)
    return {
        "calibration_rows": len(cal_rows), "development_rows": len(dev_rows),
        "float_threshold": float_threshold, "int8_threshold": int8_threshold,
        "float_development": float_result, "int8_development": int8_result,
        "logit_delta": delta_report(float_cal, int8_cal, float_dev, int8_dev),
        "records": prediction_rows("LFW", cal_rows, int8_cal, int8_threshold, [True] * len(cal_rows))
                   + prediction_rows("LFW", dev_rows, int8_dev, int8_threshold, [True] * len(dev_rows)),
    }


def camera_metrics(model, interpreter) -> dict:
    camera_roles, image_map, camera_receipt = runner.load_camera_images(
        roi_side=80, allowed_roles=EVAL_ROLES)
    rows = []
    images = []
    valid = []
    for row in camera_roles:
        capture_id = str(row["request"]["capture_id"])
        source = row["source"]
        is_valid = len(row["boxes"]) == 1 and capture_id in image_map
        rows.append({
            "record_id": capture_id,
            "label": str(row["label"]),
            "role": str(row["role"]),
            "source_identity": str(source.get("source_identity") or capture_id),
            "source_group": str(source.get("source_group") or capture_id),
        })
        valid.append(is_valid)
        images.append(image_map[capture_id] if is_valid else np.zeros((96, 96, 1), dtype=np.uint8))
    valid_array = np.asarray(valid, dtype=bool)
    image_array = np.stack(images).astype(np.uint8)
    cal_indices = [i for i, row in enumerate(rows) if row["role"] in ("threshold_calibration", "unknown_calibration") and valid[i]]
    dev_indices = [i for i, row in enumerate(rows) if row["role"] in ("development", "unknown_development")]
    cal_rows = [rows[i] for i in cal_indices]
    dev_rows = [rows[i] for i in dev_indices]
    float_all = runner.eval_logits(model, image_array)
    int8_all = tflite_reference_logits(interpreter, image_array)
    float_cal, int8_cal = float_all[cal_indices], int8_all[cal_indices]
    dev_valid = valid_array[dev_indices]
    float_dev, int8_dev = float_all[dev_indices], int8_all[dev_indices]
    float_threshold = runner.threshold_from_calibration(cal_rows, float_cal)
    int8_threshold = runner.threshold_from_calibration(cal_rows, int8_cal)
    float_result = runner.evaluate_open_set(dev_rows, float_dev, float_threshold, dev_valid)
    int8_result = runner.evaluate_open_set(dev_rows, int8_dev, int8_threshold, dev_valid)
    return {
        "capture_receipt": {k: v for k, v in camera_receipt.items() if k != "private_input_records"},
        "calibration_rows_total": sum(row["role"] in ("threshold_calibration", "unknown_calibration") for row in rows),
        "calibration_valid_rows": len(cal_rows), "development_rows": len(dev_rows),
        "development_valid_rows": int(dev_valid.sum()),
        "float_threshold": float_threshold, "int8_threshold": int8_threshold,
        "float_development": float_result, "int8_development": int8_result,
        "logit_delta": delta_report(float_cal, int8_cal, float_dev[dev_valid], int8_dev[dev_valid]),
        "records": prediction_rows("old_optical_camera", cal_rows, int8_cal, int8_threshold, [True] * len(cal_rows))
                   + prediction_rows("old_optical_camera", dev_rows, int8_dev, int8_threshold, dev_valid.tolist()),
    }


def tflite_reference_logits(interpreter, uint8_images: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8 or not input_scale or not output_scale:
        raise ValueError("expected full-INT8 TFLite input/output with affine quantization")
    outputs = []
    for image in np.asarray(uint8_images, dtype=np.uint8):
        x = image.astype(np.float32)[None] / np.float32(255.0)
        qx = np.clip(np.rint(x / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], qx)
        interpreter.invoke()
        qy = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        outputs.append((qy - np.float32(output_zero)) * np.float32(output_scale))
    result = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 6), dtype=np.float32)
    if result.shape != (len(uint8_images), 6) or not np.isfinite(result).all():
        raise ValueError("reference INT8 replay returned malformed or nonfinite logits")
    return result


def delta_report(float_cal, int8_cal, float_dev, int8_dev) -> dict:
    float_all = np.concatenate([float_cal.reshape(-1), float_dev.reshape(-1)]).astype(np.float32)
    int8_all = np.concatenate([int8_cal.reshape(-1), int8_dev.reshape(-1)]).astype(np.float32)
    delta = np.abs(float_all - int8_all)
    float_classes = np.argmax(np.concatenate([float_cal, float_dev]), axis=1)
    int8_classes = np.argmax(np.concatenate([int8_cal, int8_dev]), axis=1)
    return {
        "compared_logit_values": int(len(delta)),
        "mean_absolute_logit_error": float(delta.mean()) if len(delta) else 0.0,
        "p95_absolute_logit_error": float(np.quantile(delta, 0.95)) if len(delta) else 0.0,
        "max_absolute_logit_error": float(delta.max()) if len(delta) else 0.0,
        "argmax_class_agreement": float(np.mean(float_classes == int8_classes)) if len(float_classes) else 1.0,
    }


def prediction_rows(scope: str, rows: list[dict], logits: np.ndarray, threshold: dict,
                    valid: list[bool]) -> list[dict]:
    probs = runner.softmax32(logits)
    score = np.max(probs[:, :5], axis=1).astype(np.float32)
    pred = np.argmax(probs, axis=1)
    cutoff = np.float32(threshold["threshold_float32"])
    out = []
    for i, row in enumerate(rows):
        out.append({
            "scope": scope, "record_id": row["record_id"], "role": row["role"],
            "true_label": row["label"], "source_identity": row.get("source_identity"),
            "valid_input": bool(valid[i]), "predicted_class": runner.CLASSES[int(pred[i])],
            "score_max_known_float32": float(score[i]), "threshold_float32": float(cutoff),
            "accepted": bool(valid[i] and pred[i] < 5 and score[i] >= cutoff),
        })
    return out


def main() -> None:
    result_path = RUN / "RESULT.json"
    config, config_sha = runner.model_config(CONFIG)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if config_sha != result.get("config_sha256") or config_sha != preflight.get("config_sha256"):
        raise ValueError("launch config is not bound to both exact preflight and completed result")
    if config.get("future_final_access") is not False or preflight["source_rows"]["future_final_paths_resolved"] is not False:
        raise ValueError("future-final boundary failed before quantization")

    output_path = RUN / "model_int8_trainrep_v1.tflite"
    report_path = RUN / "INT8_QUANTIZATION_EVAL.json"
    private_predictions_path = RUN / "private/INT8_PREDICTIONS.jsonl"
    for path in (output_path, report_path, private_predictions_path):
        if path.exists():
            raise FileExistsError("refusing to replace existing PTQ artifact: " + str(path))

    # Recreate the selected Keras graph in the same trainability order used by
    # the scout, then bind the exact selected weights before any conversion.
    model, embedding_model, init_info = d.build_classifier(
        runner.ROOT, float(config["alpha"]), bool(config["pretrained"]), 6,
        int(config["seed"]), "v7_" + RUN_ID)
    model.get_layer("head_dropout").rate = 0.0
    d.unfreeze_top_fraction(model, float(config.get("unfreeze_top_fraction", 0.25)))
    weight_load = load_selected_weights(model, RUN / "best_model.weights.h5")
    selected_weight_sha = runner.model_weight_hash(model)
    del embedding_model, init_info

    previous_predictions = [json.loads(line) for line in (RUN / "private/PREDICTIONS.jsonl").read_text().splitlines() if line.strip()]
    expected_weight_hashes = {row["model_weight_sha256"] for row in previous_predictions}
    if expected_weight_hashes != {selected_weight_sha}:
        raise ValueError("selected checkpoint weight hash differs from the frozen scout predictions")

    # Filter the manifest to fit-only source roles before resolving/opening paths.
    source_rows, source_manifest_receipt = runner.load_role_rows()
    train_rows = [row for row in source_rows if row["role"] in ("train", "other_train")]
    if not train_rows or any(row["role"] not in ("train", "other_train") for row in train_rows):
        raise ValueError("representative dataset is empty or contains a non-fit source role")
    representative = np.stack([runner.load_source_gray(row) for row in train_rows]).astype(np.uint8)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        for gray in representative:
            yield [gray.astype(np.float32)[None] / np.float32(255.0)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    started = time.monotonic()
    payload = converter.convert()
    output_path.write_bytes(payload)

    interpreter = tf.lite.Interpreter(
        model_path=str(output_path), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    tensor_details = interpreter.get_tensor_details()
    ops = [str(op.get("op_name")) for op in interpreter._get_ops_details()]
    tensor_dtype_counts = dict(Counter(str(tensor["dtype"]) for tensor in tensor_details))
    internal_float_count = sum(tensor["dtype"] in (np.float16, np.float32, np.float64) for tensor in tensor_details)
    audit = {
        "input": {"shape": input_detail["shape"].tolist(), "dtype": str(input_detail["dtype"]),
                  "quantization": list(input_detail["quantization"])},
        "output": {"shape": output_detail["shape"].tolist(), "dtype": str(output_detail["dtype"]),
                   "quantization": list(output_detail["quantization"])},
        "ops": ops, "operator_count": len(ops), "tensor_dtype_counts": tensor_dtype_counts,
        "internal_float_tensor_count": int(internal_float_count),
        "builtin_ref_available": True,
    }
    if input_detail["shape"].tolist() != [1, 96, 96, 1] or output_detail["shape"].tolist() != [1, 6]:
        raise ValueError("TFLite tensor shape does not match gray96/six-logit contract")
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise ValueError("full INT8 TFLite I/O conversion failed")

    eval_source_rows = [row for row in source_rows if row["role"] in EVAL_ROLES]
    eval_source_rows, eval_source_images, eval_source_receipt = runner.load_lfw_images(eval_source_rows)
    source_eval = source_metrics(model, interpreter, eval_source_rows, eval_source_images)
    camera_eval = camera_metrics(model, interpreter)

    report = {
        "schema": "v8-alpha025-int8-posttraining-eval-v1",
        "created_utc": runner.utc_now(), "status": "complete_host_reference_only",
        "run_id": RUN_ID,
        "parent_result": {"path": str(result_path.relative_to(runner.ROOT)), "sha256": sha(result_path)},
        "config_sha256": config_sha,
        "preflight_sha256": sha(PREFLIGHT),
        "selected_model_weight_sha256": selected_weight_sha,
        "selected_checkpoint_load": weight_load,
        "representative_source": {
            "roles": ["train", "other_train"], "role_counts": dict(Counter(row["role"] for row in train_rows)),
            "rows": len(train_rows), "source_role_fingerprint": source_manifest_receipt["selected_rows_sha256"],
            "conversion": "LFW source JPEG -> grayscale -> area resize 96x96 uint8 -> float32 / 255 for representative input",
            "calibration_development_future_final_used_for_representative_set": False,
        },
        "export": {"path": str(output_path.relative_to(runner.ROOT)), "sha256": sha(output_path),
                   "bytes": output_path.stat().st_size, "conversion_seconds": time.monotonic() - started,
                   "keras_float_h5_bytes": (RUN / "model_float.h5").stat().st_size,
                   "model_parameter_count_from_scout": result["model"]["parameter_count"],
                   "host_reference_backend": "TensorFlow 2.10.1 TFLite BUILTIN_REF",
                   "audit": audit,
                   "has_internal_float_tensors": internal_float_count > 0,
                   "device_qualified": False},
        "source_lfw": {
            "input_receipt": eval_source_receipt,
            "calibration": source_eval["float_threshold"],
            "development_float": source_eval["float_development"],
            "development_int8": source_eval["int8_development"],
            "logit_delta": source_eval["logit_delta"],
        },
        "old_optical_camera": {
            "camera_receipt": camera_eval["capture_receipt"],
            "calibration_float": camera_eval["float_threshold"],
            "calibration_int8": camera_eval["int8_threshold"],
            "development_float": camera_eval["float_development"],
            "development_int8": camera_eval["int8_development"],
            "logit_delta": camera_eval["logit_delta"],
        },
        "private_predictions_path": str(private_predictions_path.relative_to(runner.ROOT)),
        "limits": ["one source-model training seed", "host BUILTIN_REF only", "no Nicla operator/output/memory/latency test",
                   "old optical setup only; intended camera distance and lighting remain unverified",
                   "30 unknown identity groups do not certify a 1% one-sided upper risk bound"],
    }
    all_prediction_rows = source_eval["records"] + camera_eval["records"]
    with private_predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_prediction_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    report["private_predictions"] = {"path": str(private_predictions_path.relative_to(runner.ROOT)),
                                     "rows": len(all_prediction_rows), "sha256": sha(private_predictions_path)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "model": str(output_path.relative_to(runner.ROOT)),
                      "model_sha256": sha(output_path), "model_bytes": output_path.stat().st_size,
                      "report": str(report_path.relative_to(runner.ROOT)),
                      "report_sha256": sha(report_path), "internal_float_tensor_count": internal_float_count,
                      "source_dev": source_eval["int8_development"],
                      "camera_dev": camera_eval["int8_development"]}, sort_keys=True))


if __name__ == "__main__":
    main()
