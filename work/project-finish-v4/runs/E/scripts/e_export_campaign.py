#!/usr/bin/env python3
"""Export, execute and calibrate every frozen capacity root in four TFLite formats."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
RUN_ROOT = STUDY / "runs/E"
os.environ["KERAS_HOME"] = str(STUDY / "models/keras_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(RUN_ROOT / "scripts"))

import numpy as np
import tensorflow as tf
import h5py

from nicla_study import d_campaign as d
from nicla_study.macs import layer_macs, tflite_operator_macs, compare_mac_accounts
from nicla_study.state import resource_lease
from e_pilot import digest_file, imagenet_gray_mobilenet, write_json

SEEDS = [20261101 + i for i in range(10)]
FORMATS = ("float32", "fp16_weights_float_io", "dynamic_range_int8_weights_float_io", "full_int8_ptq_train_only_rep")


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def all_layer_macs(model):
    counted, excluded, seen_models, seen_layers, total = [], [], set(), set(), 0

    def visit(container):
        nonlocal total
        if id(container) in seen_models:
            return
        seen_models.add(id(container))
        for layer in container.layers:
            if id(layer) in seen_layers:
                continue
            seen_layers.add(id(layer))
            effective = layer
            while hasattr(effective, "layer") and id(effective) not in seen_models:
                effective = effective.layer
            if isinstance(effective, tf.keras.Model):
                visit(effective)
                continue
            kind = type(effective).__name__
            if kind not in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D", "Dense"}:
                excluded.append({"name": layer.name, "type": type(layer).__name__,
                                 "reason": "outside multiply-accumulate convention"})
                continue
            kwargs = {"input_shape": layer.input_shape, "output_shape": layer.output_shape}
            if kind in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"}:
                kwargs["kernel_size"] = effective.kernel_size
            if kind == "Conv2D":
                kwargs["filters"] = int(effective.filters)
                kwargs["groups"] = int(getattr(effective, "groups", 1))
            elif kind == "DepthwiseConv2D":
                kwargs["depth_multiplier"] = int(effective.depth_multiplier)
            elif kind == "SeparableConv2D":
                kwargs["filters"] = int(effective.filters)
                kwargs["depth_multiplier"] = int(effective.depth_multiplier)
            else:
                kwargs["units"] = int(effective.units)
            count = layer_macs(kind, **kwargs)
            counted.append({"name": layer.name, "type": kind, "macs": count,
                            "wrapped_type": type(layer).__name__ if type(layer) is not type(effective) else None})
            total += count

    visit(model)
    return {"method": "recursive_unique_model_layers_v1", "macs_per_inference": total,
            "counted_layers": counted, "excluded_layers": excluded, "complete": True}


def convert(model, kind: str, train_images: np.ndarray, representative_indices: list[int], path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if kind == "fp16_weights_float_io":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif kind == "dynamic_range_int8_weights_float_io":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif kind == "full_int8_ptq_train_only_rep":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: (
            [d.gray_float(train_images[index:index + 1])] for index in representative_indices)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    data = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def _qparams(detail):
    scale, zero = detail.get("quantization", (0.0, 0))
    return float(scale), int(zero)


def tflite_infer(interpreter, images: np.ndarray, measure: bool = False):
    ins = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()[0]
    input_dtype = ins["dtype"]
    input_scale, input_zero = _qparams(ins)
    output_scale, output_zero = _qparams(outs)
    logits, timings = [], []
    for image in images:
        x = d.gray_float(image[None, ...])
        if input_dtype in (np.int8, np.uint8):
            lo, hi = (-128, 127) if input_dtype == np.int8 else (0, 255)
            x = np.clip(np.rint(x / input_scale + input_zero), lo, hi).astype(input_dtype)
        else:
            x = x.astype(input_dtype)
        interpreter.set_tensor(ins["index"], x)
        start = time.perf_counter()
        interpreter.invoke()
        if measure:
            timings.append((time.perf_counter() - start) * 1000.0)
        value = interpreter.get_tensor(outs["index"])
        if outs["dtype"] in (np.int8, np.uint8):
            value = (value.astype(np.float32) - output_zero) * output_scale
        logits.append(np.asarray(value[0], dtype=np.float32))
    return np.stack(logits) if logits else np.zeros((0, int(outs["shape"][-1])), np.float32), timings


def tensor_audit(interpreter):
    rows = []
    for tensor in interpreter.get_tensor_details():
        shape = [int(v) for v in tensor["shape"]]
        dtype = np.dtype(tensor["dtype"])
        quant = tensor.get("quantization_parameters", {})
        rows.append({"index": int(tensor["index"]), "name": str(tensor["name"]), "shape": shape,
                     "dtype": str(dtype), "bytes": int(np.prod(shape)) * dtype.itemsize,
                     "scales": [float(v) for v in quant.get("scales", [])],
                     "zero_points": [int(v) for v in quant.get("zero_points", [])],
                     "quantized_dimension": int(quant.get("quantized_dimension", 0))})
    dtype_counts = {}
    for row in rows:
        dtype_counts[row["dtype"]] = dtype_counts.get(row["dtype"], 0) + 1
    input_id = int(interpreter.get_input_details()[0]["index"])
    output_id = int(interpreter.get_output_details()[0]["index"])
    produced, consumers, constant_ids = {}, {}, set()
    ops = interpreter._get_ops_details()
    for op_index, op in enumerate(ops):
        for tensor_id in op["outputs"]:
            if int(tensor_id) >= 0:
                produced[int(tensor_id)] = op_index
        for tensor_id in op["inputs"]:
            if int(tensor_id) >= 0:
                consumers.setdefault(int(tensor_id), []).append(op_index)
    for op in ops:
        for tensor_id in op["inputs"][1:]:
            if int(tensor_id) >= 0 and int(tensor_id) not in produced:
                constant_ids.add(int(tensor_id))
    tensor_by_id = {row["index"]: row for row in rows}
    intervals = []
    for tensor_id, row in tensor_by_id.items():
        if tensor_id in constant_ids or tensor_id == output_id:
            continue
        start = produced.get(tensor_id, 0)
        ends = consumers.get(tensor_id, [start])
        intervals.append((start, max(ends), row["bytes"], tensor_id))
    peak = 0
    for op_index in range(len(ops) + 1):
        live = sum(nbytes for start, end, nbytes, _ in intervals if start <= op_index <= end)
        peak = max(peak, live)
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    return {"tensors": rows, "tensor_dtype_counts": dtype_counts,
            "input": {"index": input_id, "shape": [int(v) for v in in_detail["shape"]],
                      "dtype": str(in_detail["dtype"]), "quantization": [float(v) for v in in_detail["quantization"]]},
            "output": {"index": output_id, "shape": [int(v) for v in out_detail["shape"]],
                       "dtype": str(out_detail["dtype"]), "quantization": [float(v) for v in out_detail["quantization"]]},
            "operator_names": [str(op["op_name"]) for op in ops],
            "operator_count": len(ops),
            "all_int8_boundary": in_detail["dtype"] == np.int8 and out_detail["dtype"] == np.int8,
            "internal_float_tensor_count": sum(1 for row in rows if row["dtype"] in ("float16", "float32")),
            "activation_peak_bytes_estimate": peak,
            "activation_estimate_method": "static tensor-shape liveness over TFLite graph; constants excluded; not actual TFLM arena"}


def load_capacity_model(alpha: float, seed: int, weights_path: Path):
    if alpha in (0.10, 0.25):
        model, embedding, info = d.load_weights_exact(
            ROOT, alpha, True, 6, seed, "e_export_a%03d_seed-%d" % (alpha * 100, seed), weights_path)
    else:
        old = d.keras_mobilenet
        d.keras_mobilenet = imagenet_gray_mobilenet
        try:
            model, embedding, info = d.build_classifier(
                ROOT, alpha, True, 6, seed, "e_export_a%03d_seed-%d" % (alpha * 100, seed))
            info["weight_loading_audit"] = load_weights_by_exact_variable_name(model, weights_path)
        finally:
            d.keras_mobilenet = old
    return model, embedding, info


def load_weights_by_exact_variable_name(model: tf.keras.Model, weights_path: Path) -> dict:
    """Load E's shared nested-model HDF5 weights by exact variable path and shape.

    Keras 2.10's topological nested-model loader misidentifies shared backbone
    weights in these checkpoints as a legacy convolution layout, then attempts
    an invalid transpose. The saved HDF5 contains duplicate, identical copies
    for the classifier and embedding graphs. Verify every model variable maps
    to at least one exact-name, exact-shape dataset and that duplicates agree.
    """
    datasets = []
    with h5py.File(str(weights_path), "r") as source:
        group = source["model_weights"] if "model_weights" in source else source

        def collect(current, prefix=""):
            for name, value in current.items():
                path = prefix + "/" + name
                if isinstance(value, h5py.Dataset):
                    datasets.append((path, np.asarray(value)))
                elif isinstance(value, h5py.Group):
                    collect(value, path)

        collect(group)

    assigned = []
    missing = []
    shape_mismatches = []
    duplicate_mismatches = []
    for variable in model.weights:
        matches = [array for path, array in datasets if path.endswith("/" + variable.name)]
        if not matches:
            missing.append(variable.name)
            continue
        expected_shape = tuple(int(value) for value in variable.shape)
        if any(tuple(array.shape) != expected_shape for array in matches):
            shape_mismatches.append({"variable": variable.name,
                                     "expected": list(expected_shape),
                                     "found": [list(array.shape) for array in matches]})
            continue
        if any(not np.array_equal(matches[0], array) for array in matches[1:]):
            duplicate_mismatches.append(variable.name)
            continue
        variable.assign(matches[0])
        assigned.append(variable.name)

    if missing or shape_mismatches or duplicate_mismatches or len(assigned) != len(model.weights):
        raise ValueError("exact HDF5 weight assignment failed: " + json.dumps({
            "missing": missing, "shape_mismatches": shape_mismatches,
            "duplicate_mismatches": duplicate_mismatches,
            "assigned_count": len(assigned), "model_weight_count": len(model.weights)}))
    return {"status": "complete_verified", "method": "exact_variable_name_shape_assignment",
            "model_variable_count": len(model.weights), "assigned_variable_count": len(assigned),
            "hdf5_dataset_count": len(datasets), "duplicate_tensor_copies_verified": True,
            "weights_sha256": digest_file(weights_path)}


def root_paths(alpha: float, seed: int):
    arm = "capacity_a%03d" % int(alpha * 100)
    output = RUN_ROOT / "roots" / arm / ("seed-%d" % seed)
    if alpha in (0.10, 0.25):
        d_arm = "sv_mnv1_a%03d_%s_6" % (int(alpha * 100), "transfer" if alpha == 0.10 else "uniform_select")
        d_root = STUDY / "runs/D/survface" / d_arm / ("seed-%d" % seed)
        weights = d_root / "classifier.weights.h5"
    else:
        d_root = STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6" / ("seed-%d" % seed)
        weights = output / "classifier.weights.h5"
    return arm, output, d_root, weights


def representative_indices(root_rows: list[dict], train_images: np.ndarray, limit: int = 128):
    if len(train_images) <= limit:
        return list(range(len(train_images)))
    # Deterministic subset over intensity quantile; the sensitivity analysis below uses
    # ten independently seeded intensity/blur-stratified pools.
    means = np.mean(train_images.astype(np.float32), axis=(1, 2, 3))
    order = np.argsort(means)
    return np.linspace(0, len(order) - 1, limit, dtype=np.int64).tolist()


def export_one(model, alpha, seed, output, train_images, rep_indices, data):
    out_dir = output / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    root_rows = {}
    cal_images = np.concatenate([data["cal_known_gray"], data["cal_unknown_gray"]], axis=0)
    cal_rows = data["cal_known"] + data["cal_unknown"]
    dev_images = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    exports = {}
    keras_macs = all_layer_macs(model)
    for kind in FORMATS:
        filename = kind + ".tflite"
        path = out_dir / filename
        try:
            if not path.is_file():
                convert(model, kind, train_images, rep_indices, path)
            interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
            interpreter.allocate_tensors()
            tensors = tensor_audit(interpreter)
            cal_logits, _ = tflite_infer(interpreter, cal_images)
            calibration = d.fit_open_threshold(cal_logits[:len(data["cal_known"])], data["cal_known"],
                                                cal_logits[len(data["cal_known"]):], data["cal_unknown"], 6)
            dev_logits, timing = tflite_infer(interpreter, dev_images, measure=True)
            development = d.known_macro_and_open_metrics(dev_logits, data["dev_known"], data["dev_unknown"],
                                                           calibration["threshold"], 6)
            if kind == "float32":
                float_logits = dev_logits
                root_rows["float_reference_logits"] = dev_logits
            elif "float_reference_logits" in root_rows:
                float_logits = root_rows["float_reference_logits"]
            else:
                raise RuntimeError("float32 export must be processed before quantized controls")
            labels_a, scores_a, other_a = d.prediction_components(float_logits, 6)
            labels_b, scores_b, other_b = d.prediction_components(dev_logits, 6)
            mac_report = tflite_operator_macs(interpreter)
            receipt = {"status": "complete_verified", "host_inference_status": "pass", "format": kind,
                   "path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path),
                   "parameter_count": int(model.count_params()),
                   "keras_macs": keras_macs, "tflite_macs": mac_report,
                   "mac_accounting": compare_mac_accounts(keras_macs, mac_report),
                   "flatbuffer": tensors,
                   "representative_sample_count": len(rep_indices) if kind.startswith("full_int8") else 0,
                   "representative_role_sources": ["train", "unknown_train"] if kind.startswith("full_int8") else [],
                   "calibration": calibration,
                   "development": development,
                   "float_vs_export_logit_max_abs_drift": float(np.max(np.abs(float_logits - dev_logits))),
                   "known_label_disagreement_count_vs_float": int(np.sum(labels_a[:len(data["dev_known"])] != labels_b[:len(data["dev_known"])])),
                   "unknown_label_disagreement_count_vs_float": int(np.sum(labels_a[len(data["dev_known"]):] != labels_b[len(data["dev_known"]):])),
                   "other_disagreement_count_vs_float": int(np.sum(other_a != other_b)),
                   "known_max_probability_mean_abs_drift_vs_float": float(np.mean(np.abs(scores_a - scores_b))),
                   "host_latency_ms": {"p50": float(np.percentile(timing, 50)) if timing else None,
                                       "p95": float(np.percentile(timing, 95)) if timing else None,
                                       "n": len(timing), "measurements_are_host_only": True},
                   "role_counts": {"cal_known": len(data["cal_known"]), "cal_unknown": len(data["cal_unknown"]),
                                   "dev_known": len(data["dev_known"]), "dev_unknown": len(data["dev_unknown"])},
                   "class_order": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
                   "normalization": "gray8/255 float32; ImageNet-derived graph contains [-1,1] rescaling",
                   "decision_score": "max known-class softmax; reject below per-export calibration threshold or if argmax=OTHER",
                   "final_role_accessed": False}
            del interpreter
        except Exception as exc:
            receipt = {"status": "unsupported_or_failed", "host_inference_status": "not_run",
                       "format": kind, "path": str(path) if path.exists() else None,
                       "error_type": type(exc).__name__, "error": str(exc)[:2000],
                       "traceback_tail": __import__("traceback").format_exc()[-3000:],
                       "final_role_accessed": False}
        exports[kind] = receipt
        atomic_json(out_dir / (kind + ".json"), receipt)
    return exports


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    data = d.load_role_arrays(ROOT, d.study_paths(ROOT), "survface_96")
    export_registry = {"schema": "job-e-capacity-export-registry-1", "status": "in_progress",
                       "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
                       "final_role_accessed": False, "roots": []}
    registry_path = RUN_ROOT / "roots/CAPACITY_EXPORT_REGISTRY.json"
    with resource_lease(STUDY, "heavy_training", owner="E_capacity_export_infer_calibrate_40_roots", timeout_s=0):
        for alpha in (0.10, 0.25, 0.50, 0.75):
            for seed in SEEDS:
                arm, output, d_root, weights = root_paths(alpha, seed)
                if not weights.is_file():
                    row = {"arm_id": arm, "alpha": alpha, "root_seed": seed,
                           "status": "failed_needs_repair", "reason": "source weights missing"}
                    export_registry["roots"].append(row)
                    atomic_json(registry_path, export_registry)
                    continue
                done_path = output / "EXPORT_RESULT.json"
                if done_path.is_file():
                    old = json.loads(done_path.read_text(encoding="utf-8"))
                    if old.get("status") == "complete_verified" and all(
                            old.get("exports", {}).get(fmt, {}).get("status") in ("complete_verified", "unsupported_or_failed")
                            for fmt in FORMATS):
                        export_registry["roots"].append({"arm_id": arm, "alpha": alpha, "root_seed": seed,
                                                         "status": "already_complete", "exports": list(FORMATS)})
                        atomic_json(registry_path, export_registry)
                        continue
                if alpha in (0.10, 0.25):
                    train_rows = d.read_jsonl(d_root / "training_rows.jsonl")
                    train_images = d.cache_gray8(ROOT, train_rows, "train")
                else:
                    train_rows = d.read_jsonl(d_root / "training_rows.jsonl")
                    train_images = d.cache_gray8(ROOT, train_rows, "train")
                rep_indices = representative_indices(train_rows, train_images)
                try:
                    model, _, info = load_capacity_model(alpha, seed, weights)
                    exports = export_one(model, alpha, seed, output, train_images, rep_indices, data)
                    root_result = {"schema": "job-e-capacity-export-result-1", "status": "complete_verified",
                                   "arm_id": arm, "alpha": alpha, "root_seed": seed,
                                   "source_root_result": str(d_root / "ROOT_RESULT.json"),
                                   "source_root_sha256": digest_file(d_root / "ROOT_RESULT.json"),
                                   "weights_path": str(weights), "weights_sha256": digest_file(weights),
                                   "initialization": info.get("initialization"),
                                   "checkpoint_sha256": info.get("checkpoint_sha256"),
                                   "exports": exports, "final_role_accessed": False}
                    atomic_json(done_path, root_result)
                    if alpha in (0.50, 0.75):
                        capacity_root_result = json.loads((output / "ROOT_RESULT.json").read_text(encoding="utf-8"))
                        capacity_root_result["exports"] = {fmt: {"path": value.get("path"), "sha256": value.get("sha256"),
                                                                  "bytes": value.get("bytes"), "status": value["status"]}
                                                            for fmt, value in exports.items()}
                        atomic_json(output / "ROOT_RESULT.json", capacity_root_result)
                    else:
                        atomic_json(output / "ROOT_RESULT.json", root_result)
                    row = {"arm_id": arm, "alpha": alpha, "root_seed": seed,
                           "status": "complete_verified", "formats": list(exports.keys()),
                           "bytes": {fmt: item.get("bytes") for fmt, item in exports.items()},
                           "format_status": {fmt: item["status"] for fmt, item in exports.items()},
                           "utility_int8": (exports[FORMATS[-1]].get("development") or {}).get("development_macro_utility")}
                    del model
                    tf.keras.backend.clear_session()
                except Exception as exc:
                    row = {"arm_id": arm, "alpha": alpha, "root_seed": seed,
                           "status": "failed_needs_repair", "error_type": type(exc).__name__,
                           "error": str(exc)[:1500], "traceback_tail": __import__("traceback").format_exc()[-2500:]}
                    atomic_json(output / "EXPORT_RESULT.json", row)
                export_registry["roots"].append(row)
                atomic_json(registry_path, export_registry)
                print(json.dumps({"event": "capacity_exports", **row}), flush=True)
        export_registry["status"] = "complete_verified" if all(
            row["status"] in ("complete_verified", "already_complete") for row in export_registry["roots"]) else "partial"
        export_registry["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(registry_path, export_registry)
    print(json.dumps({"event": "capacity_exports_complete", "status": export_registry["status"],
                      "registry": str(registry_path)}), flush=True)


if __name__ == "__main__":
    main()
