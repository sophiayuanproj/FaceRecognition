#!/usr/bin/env python3
"""Pre-commit conversion, operator, tensor-shape and graph-surgery pilot for Job E."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
RUN_ROOT = STUDY / "runs/E"
PILOT_DIR = RUN_ROOT / "pilot"
os.environ["KERAS_HOME"] = str(STUDY / "models/keras_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(ROOT / "FaceRecognition"))

import hashlib
import json
import time
import traceback

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from nicla_study import d_campaign as d
from nicla_study.macs import keras_model_macs, tflite_operator_macs, compare_mac_accounts, layer_macs
from nicla_study.state import resource_lease


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def imagenet_gray_mobilenet(workspace: Path, alpha: float, pretrained: bool, seed: int):
    if alpha not in (0.50, 0.75) or not pretrained:
        return d.keras_mobilenet(workspace, alpha, pretrained, seed)
    tf.keras.utils.set_random_seed(int(seed))
    filename = "mobilenet_%s_224_tf_no_top.h5" % ("5_0" if alpha == 0.50 else "7_5")
    checkpoint_path = STUDY / "models/keras_cache/models" / filename
    expected_sha256 = {
        0.50: "df860c59562f4c4af7a2fb2ae90731caef39fd5d03e5fcd6ca46988965f0a4b7",
        0.75: "1da9019b66f05a0256dedfed0f321fcaf8bfa71f9facf3f9117e62820be39aa5",
    }[alpha]
    if not checkpoint_path.is_file() or digest_file(checkpoint_path) != expected_sha256:
        raise ValueError("missing or hash-mismatched official Keras checkpoint: %s" % checkpoint_path)
    reference = tf.keras.applications.MobileNet(
        input_shape=(96, 96, 3), alpha=alpha, include_top=False, weights=None)
    reference.load_weights(str(checkpoint_path))
    gray_body = tf.keras.applications.MobileNet(
        input_shape=(96, 96, 1), alpha=alpha, include_top=False, weights=None)
    copied = []
    for target_layer in gray_body.layers:
        if isinstance(target_layer, tf.keras.layers.InputLayer):
            continue
        target_weights = target_layer.get_weights()
        if not target_weights:
            continue
        source_layer = reference.get_layer(target_layer.name)
        source_weights = source_layer.get_weights()
        if target_layer.name == "conv1":
            if len(source_weights) != 1 or len(target_weights) != 1:
                raise ValueError("unexpected first-convolution bias/weight structure")
            collapsed = np.sum(source_weights[0], axis=2, keepdims=True)
            if collapsed.shape != target_weights[0].shape:
                raise ValueError("RGB-to-gray first-convolution shape mismatch")
            target_layer.set_weights([collapsed])
            copied.append({"layer": target_layer.name, "method": "sum_RGB_kernel_channels",
                           "source_shape": list(source_weights[0].shape),
                           "target_shape": list(collapsed.shape)})
        else:
            source_shapes = [list(value.shape) for value in source_weights]
            target_shapes = [list(value.shape) for value in target_weights]
            if source_shapes != target_shapes:
                raise ValueError("RGB/grayscale reference layer mismatch at %s: %r != %r" %
                                 (target_layer.name, source_shapes, target_shapes))
            target_layer.set_weights(source_weights)
            copied.append({"layer": target_layer.name, "method": "shape_identical_copy",
                           "shapes": target_shapes})
    gray_input = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray01")
    normalized = tf.keras.layers.Rescaling(
        scale=2.0, offset=-1.0, name="imagenet_preprocess_minus1_plus1")(gray_input)
    gray_features = tf.keras.Model(
        gray_input, gray_body(normalized), name="mobilenetv1_imagenet_gray_adapt_%s" % alpha)
    gray_input_examples = np.asarray([0, 31, 64, 127, 191, 255], dtype=np.float32) / 255.0
    gray_input_examples = np.tile(gray_input_examples.reshape(6, 1, 1, 1), (1, 96, 96, 1))
    rgb_reference_input = np.repeat(gray_input_examples * 2.0 - 1.0, 3, axis=-1)
    reference_values = reference(rgb_reference_input, training=False).numpy()
    adapted_values = gray_features(gray_input_examples, training=False).numpy()
    max_error = float(np.max(np.abs(reference_values - adapted_values)))
    equivalence = {
        "status": "pass" if np.allclose(reference_values, adapted_values, atol=5e-4, rtol=2e-5) else "fail",
        "max_abs_error": max_error,
        "mean_abs_error": float(np.mean(np.abs(reference_values - adapted_values))),
        "test_samples": int(len(gray_input_examples)),
        "comparison": "RGB ImageNet reference on replicated 96x96 gray in [-1,1] versus adapted grayscale feature graph on gray8/255 with internal [-1,1] scaling",
        "tolerance": {"atol": 5e-4, "rtol": 2e-5,
                      "rationale": "allow FP32 accumulation-order roundoff after summing three RGB kernels into one grayscale kernel"},
    }
    if equivalence["status"] != "pass":
        raise ValueError("grayscale MobileNet transfer failed numerical equivalence: %r" % equivalence)
    sha = digest_file(checkpoint_path)
    info = {
        "alpha": float(alpha), "input_shape": [1, 96, 96, 1],
        "initialization": "Keras MobileNetV1 ImageNet RGB reference; exact grayscale first-convolution channel-sum adaptation",
        "checkpoint_sha256": sha, "checkpoint_path": str(checkpoint_path),
        "checkpoint_source_url": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/" + filename,
        "full_reference_model": reference, "feature_model": gray_features,
        "input_range": "gray8/255 in [0,1], rescaled to [-1,1] inside graph",
        "backbone_feature_layer": "MobileNetV1 conv_pw_13_relu",
        "backbone_feature_shape": [int(value) if value is not None else None
                                   for value in gray_features.output_shape[1:]],
        "checkpoint_head_removed": ["ImageNet classifier; Keras include_top=False"],
        "rgb_to_gray_weight_transfer": copied,
        "rgb_grayscale_numeric_equivalence": equivalence,
    }
    return gray_body, gray_features, info


def load_model(alpha: float, seed: int, pretrained: bool, weights_path: Path | None = None):
    if alpha not in (0.50, 0.75):
        if weights_path is not None:
            return d.load_weights_exact(
                ROOT, alpha, pretrained, 6, seed, "e_pilot_reload_%s" % alpha, weights_path)
        return d.build_classifier(ROOT, alpha, pretrained, 6, seed, "e_pilot_%s" % alpha)
    original = d.keras_mobilenet
    d.keras_mobilenet = imagenet_gray_mobilenet
    try:
        model, embedding, info = d.build_classifier(
            ROOT, alpha, pretrained, 6, seed, "e_pilot_%s" % alpha)
        if weights_path is not None:
            model.load_weights(str(weights_path))
    finally:
        d.keras_mobilenet = original
    return model, embedding, info


def create_physical_channel_surgery(model: tf.keras.Model):
    base = model._d_base
    conv = base.get_layer("conv_pw_13")
    kernel = conv.get_weights()[0]
    channel_scores = np.sum(np.abs(kernel), axis=(0, 1, 2))
    keep_count = max(1, int(np.ceil(kernel.shape[-1] * 0.50)))
    keep = np.sort(np.argsort(channel_scores)[-keep_count:]).astype(np.int64)
    keep_set = set(int(value) for value in keep)

    def clone_layer(layer):
        config = layer.get_config()
        if layer.name == "conv_pw_13":
            config["filters"] = keep_count
            return layer.__class__.from_config(config)
        return layer.__class__.from_config(config)

    small_base = tf.keras.models.clone_model(base, clone_function=clone_layer)
    for new_layer in small_base.layers:
        old_layer = base.get_layer(new_layer.name)
        old_weights = old_layer.get_weights()
        if not old_weights:
            continue
        if new_layer.name == "conv_pw_13":
            new_layer.set_weights([old_weights[0][:, :, :, keep]])
        elif new_layer.name == "conv_pw_13_bn":
            new_layer.set_weights([value[keep] for value in old_weights])
        elif [list(value.shape) for value in old_weights] == [list(value.shape) for value in new_layer.get_weights()]:
            new_layer.set_weights(old_weights)
        else:
            raise ValueError("physical surgery created an unhandled shape dependency at %s: %r -> %r" %
                             (new_layer.name, [list(value.shape) for value in old_weights],
                              [list(value.shape) for value in new_layer.get_weights()]))

    inputs = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_float01")
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(small_base(inputs, training=False))
    hidden = tf.keras.layers.Dense(64, activation="relu", name="embedding_64")(pooled)
    dropped = tf.keras.layers.Dropout(0.1, name="head_dropout")(hidden)
    logits = tf.keras.layers.Dense(6, activation=None, name="classifier_logits")(dropped)
    small_model = tf.keras.Model(inputs, logits, name="physical_channel_pruned_classifier")
    old_hidden = model.get_layer("embedding_64").get_weights()
    small_model.get_layer("embedding_64").set_weights([old_hidden[0][keep, :], old_hidden[1]])
    small_model.get_layer("classifier_logits").set_weights(model.get_layer("classifier_logits").get_weights())
    small_model._d_base = small_base
    small_model._d_embedding_model = tf.keras.Model(inputs, hidden)

    mask = np.zeros(kernel.shape[-1], dtype=np.float32)
    mask[keep] = 1.0
    ref_input = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_masked_reference")
    ref_features = base(ref_input, training=False)
    masked_features = tf.keras.layers.Multiply()(
        [ref_features, tf.constant(mask.reshape(1, 1, 1, -1), dtype=tf.float32)])
    ref_hidden = model.get_layer("global_average_pooling")(masked_features)
    ref_hidden = model.get_layer("embedding_64")(ref_hidden)
    ref_dropped = model.get_layer("head_dropout")(ref_hidden, training=False)
    ref_logits = model.get_layer("classifier_logits")(ref_dropped)
    masked_model = tf.keras.Model(ref_input, ref_logits)
    return small_model, masked_model, {
        "pruned_layer": "conv_pw_13",
        "parent_shape": list(kernel.shape),
        "new_shape": list(small_base.get_layer("conv_pw_13").get_weights()[0].shape),
        "parent_output_channels": int(kernel.shape[-1]),
        "new_output_channels": int(keep_count),
        "keep_fraction": float(keep_count / kernel.shape[-1]),
        "selected_channel_indices": keep.tolist(),
        "selected_channel_indices_sha256": digest_bytes(keep.astype("<i8").tobytes()),
        "batch_norm_dependency": "conv_pw_13_bn gamma/beta/moving_mean/moving_variance sliced with the same channel indices",
        "downstream_dependency": "global-average-pool output and embedding_64 kernel input axis sliced with the same channel indices",
        "unhandled_dependencies": [],
        "training_checkpoint_l1_filter_importance": True,
    }


def export_int8(model: tf.keras.Model, train_gray: np.ndarray, path: Path, indices: list[int]):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [d.gray_float(train_gray[index:index + 1])] for index in indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model_bytes)
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    tensors = []
    for item in interpreter.get_tensor_details():
        q = item.get("quantization_parameters", {})
        tensors.append({
            "index": int(item["index"]), "name": str(item["name"]),
            "shape": [int(value) for value in item["shape"]],
            "dtype": str(item["dtype"]),
            "num_bytes": int(item["shape"].prod()) * int(np.dtype(item["dtype"]).itemsize),
            "scales": [float(value) for value in q.get("scales", [])],
            "zero_points": [int(value) for value in q.get("zero_points", [])],
            "quantized_dimension": int(q.get("quantized_dimension", 0)),
        })

    def recursive_macs(nested_model):
        rows, excluded, total = [], [], 0
        for layer in nested_model.layers:
            if isinstance(layer, tf.keras.Model):
                sub = recursive_macs(layer)
                rows.extend(sub["counted_layers"])
                excluded.extend(sub["excluded_layers"])
                total += sub["macs_per_inference"]
                continue
            kind = type(layer).__name__
            if kind not in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D", "Dense"}:
                excluded.append({"name": layer.name, "type": kind,
                                 "reason": "outside multiply-accumulate convention"})
                continue
            kwargs = {"input_shape": layer.input_shape, "output_shape": layer.output_shape}
            if kind in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"}:
                kwargs["kernel_size"] = layer.kernel_size
            if kind == "Conv2D":
                kwargs["filters"] = int(layer.filters)
                kwargs["groups"] = int(getattr(layer, "groups", 1))
            elif kind == "DepthwiseConv2D":
                kwargs["depth_multiplier"] = int(layer.depth_multiplier)
            elif kind == "SeparableConv2D":
                kwargs["filters"] = int(layer.filters)
                kwargs["depth_multiplier"] = int(layer.depth_multiplier)
            elif kind == "Dense":
                kwargs["units"] = int(layer.units)
            count = layer_macs(kind, **kwargs)
            rows.append({"name": layer.name, "type": kind, "macs": count})
            total += count
        return {"method": "keras_recursive_layer_formula_v1", "macs_per_inference": total,
                "counted_layers": rows, "excluded_layers": excluded, "complete": True}

    layer_macs_report = recursive_macs(model)
    op_macs = tflite_operator_macs(interpreter)
    sample = d.gray_float(train_gray[:1])
    if input_detail["dtype"] == np.int8:
        input_scale, input_zero = input_detail["quantization"]
        sample = np.clip(np.rint(sample / input_scale + input_zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_detail["index"], sample.astype(input_detail["dtype"]))
    interpreter.invoke()
    pilot_output = interpreter.get_tensor(output_detail["index"])
    audit = {
        "path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path),
        "parameter_count": int(model.count_params()),
        "input": {"shape": [int(value) for value in input_detail["shape"]],
                  "dtype": str(input_detail["dtype"]), "quantization": [float(value) for value in input_detail["quantization"]]},
        "output": {"shape": [int(value) for value in output_detail["shape"]],
                   "dtype": str(output_detail["dtype"]), "quantization": [float(value) for value in output_detail["quantization"]]},
        "operators": [str(item["op_name"]) for item in interpreter._get_ops_details()],
        "tensor_types": sorted({str(item["dtype"]) for item in interpreter.get_tensor_details()}),
        "all_int8_boundary": input_detail["dtype"] == np.int8 and output_detail["dtype"] == np.int8,
        "internal_float_tensor_count": sum(1 for item in interpreter.get_tensor_details()
                                           if item["dtype"] in (np.float16, np.float32)),
        "representative_sample_count": int(len(indices)),
        "tensor_audit": tensors,
        "host_inference_smoke": {"status": "pass", "finite": bool(np.isfinite(pilot_output).all()),
                                 "output_shape": [int(value) for value in pilot_output.shape],
                                 "output_dtype": str(pilot_output.dtype)},
        "mac_accounting": compare_mac_accounts(layer_macs_report, op_macs),
        "keras_mac_report": layer_macs_report,
        "tflite_mac_report": op_macs,
    }
    return audit


def try_conversion(label, model, train_gray, indices):
    path = PILOT_DIR / (label + ".tflite")
    started = time.monotonic()
    try:
        audit = export_int8(model, train_gray, path, indices)
        audit.update({"status": "pass", "elapsed_seconds": time.monotonic() - started})
    except Exception as error:
        audit = {"status": "unsupported_or_failed", "error_type": type(error).__name__,
                 "error": str(error)[:1200], "traceback_tail": traceback.format_exc()[-2500:],
                 "elapsed_seconds": time.monotonic() - started}
    print(json.dumps({"event": "pilot_conversion", "label": label,
                      "status": audit["status"], "elapsed_seconds": audit["elapsed_seconds"]}),
          flush=True)
    return audit


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = PILOT_DIR / "PROGRESS.json"
    report = {
        "schema": "job-e-startup-operator-shape-pilot-1",
        "status": "in_progress", "final_role_accessed": False,
        "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
        "starts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [], "failures": [],
    }
    write_json(progress_path, report)
    with resource_lease(STUDY, "heavy_training", owner="E_preflight_conversion_operator_shape_pilot", timeout_s=0):
        paths = d.study_paths(ROOT)
        data = d.load_role_arrays(ROOT, paths, "survface_96")
        base_dir = STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6/seed-20261101"
        train_rows = d.read_jsonl(base_dir / "training_rows.jsonl")
        train_gray = d.cache_gray8(ROOT, train_rows, "train")
        representative = np.linspace(0, len(train_gray) - 1, min(32, len(train_gray)),
                                     dtype=np.int64).tolist()
        report["data_receipt"] = {
            "training_root_rows": len(train_rows), "training_image_shape": list(train_gray.shape),
            "calibration_known": len(data["cal_known"]), "calibration_unknown": len(data["cal_unknown"]),
            "development_known": len(data["dev_known"]), "development_unknown": len(data["dev_unknown"]),
            "representative_source_roles": ["train", "unknown_train"],
            "final_role_accessed": False,
        }
        report["steps"].append({"id": "role_checked_data_load", "status": "pass"})
        write_json(progress_path, report)

        weight_cache_rows = []
        for alpha in (0.50, 0.75):
            try:
                reference, _, info = imagenet_gray_mobilenet(ROOT, alpha, True, 76101)
                checkpoint = Path(info["checkpoint_path"])
                weight_cache_rows.append({
                    "alpha": alpha, "status": "pass", "checkpoint_path": str(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": info["checkpoint_sha256"],
                    "checkpoint_source_url": info["checkpoint_source_url"],
                    "grayscale_adaptation": info["rgb_to_gray_weight_transfer"],
                    "equivalence": info["rgb_grayscale_numeric_equivalence"],
                    "feature_shape": info["backbone_feature_shape"],
                })
                del reference
                tf.keras.backend.clear_session()
            except Exception as error:
                row = {"alpha": alpha, "status": "failed", "error_type": type(error).__name__,
                       "error": str(error)[:1200], "traceback_tail": traceback.format_exc()[-2500:]}
                weight_cache_rows.append(row)
                print(json.dumps({"event": "imagenet_weight_adaptation", **row}), flush=True)
            write_json(PILOT_DIR / "imagenet_gray_adaptation.json", {"arms": weight_cache_rows})
        report["imagenet_gray_adaptation"] = weight_cache_rows
        report["steps"].append({"id": "wider_pretrained_grayscale_adaptation",
                                "status": "pass" if all(row["status"] == "pass" for row in weight_cache_rows) else "failed"})
        write_json(progress_path, report)

        capacity_models = []
        baseline_specs = [
            (0.10, "sv_mnv1_a010_transfer_6", True),
            (0.25, "sv_mnv1_a025_uniform_select_6", True),
        ]
        for alpha, arm_id, pretrained in baseline_specs:
            model, _, info = load_model(alpha, 20261101, pretrained,
                                        STUDY / "runs/D/survface" / arm_id / "seed-20261101" / "classifier.weights.h5")
            capacity_models.append((alpha, model, info))
        for alpha in (0.50, 0.75):
            if not any(row.get("alpha") == alpha and row.get("status") == "pass" for row in weight_cache_rows):
                continue
            model, _, info = load_model(alpha, 76101, True)
            capacity_models.append((alpha, model, info))

        conversion_rows = []
        for alpha, model, info in capacity_models:
            path = PILOT_DIR / ("capacity_a%03d_ptq_int8.tflite" % int(alpha * 100))
            row = try_conversion("capacity_a%03d_ptq_int8" % int(alpha * 100), model, train_gray, representative)
            row["alpha"] = alpha
            row["model_shape"] = [int(value) if value is not None else None
                                  for value in model.output_shape]
            row["keras_parameter_count"] = int(model.count_params())
            row["initialization"] = info.get("initialization")
            conversion_rows.append(row)
            report["steps"].append({"id": "capacity_%s_int8_conversion" % alpha,
                                    "status": "pass" if row["status"] == "pass" else "failed"})
            write_json(progress_path, report)
            del model
            tf.keras.backend.clear_session()
        report["capacity_int8_conversions"] = conversion_rows

        reference_model, _, info = load_model(
            0.25, 20261101, True,
            STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6/seed-20261101/classifier.weights.h5")
        method_rows = []
        method_rows.append(try_conversion("method_dense_reference", reference_model, train_gray, representative))

        def attempt(name, make_model, strip):
            try:
                wrapped = make_model()
                candidate = strip(wrapped)
                row = try_conversion(name, candidate, train_gray, representative)
                row["status"] = "pass" if row["status"] == "pass" else row["status"]
            except Exception as error:
                row = {"status": "unsupported_or_failed", "error_type": type(error).__name__,
                       "error": str(error)[:1200], "traceback_tail": traceback.format_exc()[-2500:]}
                print(json.dumps({"event": "pilot_method_wrapper", "label": name,
                                  "status": row["status"], "error": row["error"][:500]}), flush=True)
            row["method"] = name
            method_rows.append(row)
            write_json(PILOT_DIR / "method_conversion_pilot.json", {"methods": method_rows})

        attempt("method_qat", lambda: tfmot.quantization.keras.quantize_apply(
            tfmot.quantization.keras.quantize_annotate_model(reference_model)),
            lambda value: value)
        for sparsity in (0.50, 0.75):
            attempt("method_magnitude_%02d" % int(sparsity * 100),
                    lambda s=sparsity: tfmot.sparsity.keras.prune_low_magnitude(
                        reference_model, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                            target_sparsity=s, begin_step=0, frequency=10)),
                    tfmot.sparsity.keras.strip_pruning)
        attempt("method_structured_2of4",
                lambda: tfmot.sparsity.keras.prune_low_magnitude(
                    reference_model, sparsity_m_by_n=(2, 4)),
                tfmot.sparsity.keras.strip_pruning)
        attempt("method_clustering_16",
                lambda: tfmot.clustering.keras.cluster_weights(reference_model, number_of_clusters=16),
                tfmot.clustering.keras.strip_clustering)
        try:
            depthwise = reference_model._d_base.get_layer("conv_dw_13")
            tfmot.clustering.keras.cluster_weights(depthwise, number_of_clusters=16)
            depthwise_result = {"status": "supported_wrapper_created",
                                "layer": depthwise.name, "shape": [list(value.shape) for value in depthwise.get_weights()]}
        except Exception as error:
            depthwise_result = {"status": "unsupported", "error_type": type(error).__name__,
                                "error": str(error)[:1200], "traceback_tail": traceback.format_exc()[-2000:]}
        report["depthwise_cluster_cardinality_16"] = depthwise_result
        write_json(PILOT_DIR / "method_conversion_pilot.json",
                   {"methods": method_rows, "depthwise_cluster_cardinality_16": depthwise_result})

        try:
            physical, masked_reference, surgery = create_physical_channel_surgery(reference_model)
            inputs = d.gray_float(train_gray[:8])
            expected = masked_reference(inputs, training=False).numpy()
            actual = physical(inputs, training=False).numpy()
            max_error = float(np.max(np.abs(expected - actual)))
            surgery["masked_parent_max_abs_error"] = max_error
            surgery["masked_parent_graph_equivalence"] = bool(
                np.allclose(expected, actual, atol=1e-5, rtol=1e-5))
            surgery["logit_shape"] = [int(value) for value in physical.output_shape]
            surgery["parameter_count_before"] = int(reference_model.count_params())
            surgery["parameter_count_after"] = int(physical.count_params())
            surgery["changed_tensor_shapes"] = {
                "conv_pw_13_kernel": [list(value.shape) for value in reference_model._d_base.get_layer("conv_pw_13").get_weights()] +
                                     [list(value.shape) for value in physical._d_base.get_layer("conv_pw_13").get_weights()],
                "conv_pw_13_bn": [list(value.shape) for value in reference_model._d_base.get_layer("conv_pw_13_bn").get_weights()] +
                                  [list(value.shape) for value in physical._d_base.get_layer("conv_pw_13_bn").get_weights()],
                "embedding_64": [list(value.shape) for value in reference_model.get_layer("embedding_64").get_weights()] +
                                [list(value.shape) for value in physical.get_layer("embedding_64").get_weights()],
            }
            surgery["pilot_pass"] = surgery["masked_parent_graph_equivalence"]
            report["physical_channel_surgery_pilot"] = surgery
            report["steps"].append({"id": "physical_channel_graph_surgery",
                                    "status": "pass" if surgery["pilot_pass"] else "failed"})
        except Exception as error:
            report["physical_channel_surgery_pilot"] = {
                "status": "unsupported_or_failed", "error_type": type(error).__name__,
                "error": str(error)[:1200], "traceback_tail": traceback.format_exc()[-2500:]}
            report["steps"].append({"id": "physical_channel_graph_surgery", "status": "failed"})

        report["prerequisite_frozen_root"] = str(base_dir / "ROOT_RESULT.json")
        report["source_weight_sha256"] = digest_file(base_dir / "classifier.weights.h5")
        report["training_budget_note"] = "This pilot used zero optimizer updates and is not counted as a training root."
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        failed = [row for row in report["steps"] if row["status"] != "pass"]
        report["status"] = "complete_with_supported_failures" if not failed else "pilot_blockers_recorded"
        write_json(PILOT_DIR / "PILOT_RESULT.json", report)
        write_json(progress_path, report)
        print(json.dumps({"event": "pilot_complete", "status": report["status"],
                          "failed_steps": failed, "report": str(PILOT_DIR / "PILOT_RESULT.json")}),
              flush=True)
    return 0 if report["status"] == "complete_with_supported_failures" else 2


if __name__ == "__main__":
    raise SystemExit(main())
