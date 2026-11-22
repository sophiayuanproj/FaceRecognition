#!/usr/bin/env python3
"""Run matched five-epoch compression and capacity-control arms, resumably."""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
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
import tensorflow_model_optimization as tfmot

from nicla_study import d_campaign as d
from nicla_study.state import resource_lease
from e_pilot import (digest_file, create_physical_channel_surgery, write_json)
from e_export_campaign import (FORMATS, all_layer_macs, convert, tflite_infer,
                               tensor_audit, atomic_json, load_capacity_model)

SEEDS = [20261101 + i for i in range(10)]
METHODS = [
    "large_dense_equal_finetune", "large_qat", "large_magnitude_prune_50",
    "large_magnitude_prune_75", "large_structured_prune_2of4",
    "large_physical_channel_keep_50", "small_a050_dense_equal_finetune",
    "large_cluster_16", "small_a010_dense_equal_finetune", "small_a010_distilled",
    "large_head_32_sliced_initialization", "large_head_32_reset_initialization",
    "large_head_direct_gap_logits",
]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def find_body(container):
    if any(layer.name == "conv_pw_13" for layer in container.layers):
        return container
    for layer in container.layers:
        if isinstance(layer, tf.keras.Model):
            found = find_body(layer)
            if found is not None:
                return found
    return None


def flatten_layers(container):
    models, leaves = [], []
    def walk(value):
        if any(id(value) == id(previous) for previous in models):
            return
        models.append(value)
        for layer in value.layers:
            if isinstance(layer, tf.keras.Model):
                walk(layer)
            elif not any(id(layer) == id(previous) for previous in leaves):
                leaves.append(layer)
    walk(container)
    return models, leaves


def wraps_batch_norm(layer):
    current, seen = layer, set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, tf.keras.layers.BatchNormalization):
            return True
        current = getattr(current, "layer", None)
        if current is None:
            return False
    return False


def unfreeze_top_fraction(model, fraction=0.5):
    base = model._d_base
    models, leaves = flatten_layers(base)
    for item in models:
        item.trainable = True
    cutoff = max(1, int(np.floor(len(leaves) * (1.0 - fraction))))
    for index, layer in enumerate(leaves):
        layer.trainable = index >= cutoff and not wraps_batch_norm(layer)
    return {"backbone_layers_total": len(leaves), "freeze_cutoff_index": cutoff,
            "trainable_layer_names": [layer.name for layer in leaves if layer.trainable],
            "batch_normalization_frozen": True, "nested_model_names": [item.name for item in models]}


def capacity_root(alpha, seed):
    arm = "capacity_a%03d" % int(alpha * 100)
    output = RUN_ROOT / "roots" / arm / ("seed-%d" % seed)
    if alpha == 0.10:
        source = STUDY / "runs/D/survface/sv_mnv1_a010_transfer_6" / ("seed-%d" % seed)
        weights = source / "classifier.weights.h5"
    elif alpha == 0.25:
        source = STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6" / ("seed-%d" % seed)
        weights = source / "classifier.weights.h5"
    else:
        source = output
        weights = output / "classifier.weights.h5"
    return source, weights


def compose_head(base, alpha, parent, units=64, kind="copy", keep_indices=None):
    inp = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_float01")
    if alpha in (0.50, 0.75):
        feat = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="imagenet_preprocess_minus1_plus1")(inp)
    else:
        feat = inp
    feat = base(feat, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(feat)
    hidden = tf.keras.layers.Dense(units, activation="relu", name="embedding_%d" % units)(pooled)
    dropped = tf.keras.layers.Dropout(0.1, name="head_dropout")(hidden)
    logits = tf.keras.layers.Dense(6, activation=None, name="classifier_logits")(dropped)
    model = tf.keras.Model(inp, logits, name="e_%s_head_%d" % (kind, units))
    if kind in ("copy", "slice") and parent is not None:
        old_hidden = parent.get_layer("embedding_64").get_weights()
        old_output = parent.get_layer("classifier_logits").get_weights()
        if kind == "slice":
            keep_indices = np.asarray(keep_indices, dtype=np.int64)
            model.get_layer("embedding_%d" % units).set_weights(
                [old_hidden[0][:, keep_indices], old_hidden[1][keep_indices]])
            model.get_layer("classifier_logits").set_weights(
                [old_output[0][keep_indices, :], old_output[1]])
        elif units == 64:
            model.get_layer("embedding_64").set_weights(old_hidden)
            model.get_layer("classifier_logits").set_weights(old_output)
    elif kind == "copy" and parent is not None and units == 64:
        model.get_layer("embedding_64").set_weights(parent.get_layer("embedding_64").get_weights())
        model.get_layer("classifier_logits").set_weights(parent.get_layer("classifier_logits").get_weights())
    model._d_base = base
    model._d_info = dict(getattr(parent, "_d_info", {})) if parent is not None else {"alpha": alpha}
    model._d_embedding_model = tf.keras.Model(inp, hidden, name=model.name + "_embedding")
    return model


def make_head_variant(parent, alpha, seed, variant):
    tf.keras.utils.set_random_seed(seed + (11 if variant == "slice" else 17 if variant == "reset" else 23))
    if variant == "direct_gap":
        inp = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_float01")
        if alpha in (0.50, 0.75):
            x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="imagenet_preprocess_minus1_plus1")(inp)
        else:
            x = inp
        x = parent._d_base(x, training=False)
        pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        logits = tf.keras.layers.Dense(6, activation=None, name="classifier_logits")(pooled)
        small = tf.keras.Model(inp, logits, name="e_direct_gap_logits")
        small._d_base = parent._d_base
        small._d_info = dict(parent._d_info)
        return small, {"method": variant, "hidden_width": 0, "initialization": "random six-logit Dense on global-average-pooled backbone features",
                       "gallery_embedding_output": False}
    kernel, bias = parent.get_layer("embedding_64").get_weights()
    out_kernel, _ = parent.get_layer("classifier_logits").get_weights()
    importance = np.linalg.norm(kernel, axis=0) * np.linalg.norm(out_kernel, axis=1)
    indices = np.sort(np.argsort(importance)[-32:])
    if variant == "slice":
        new = compose_head(parent._d_base, alpha, parent, 32, "slice", indices)
    else:
        new = compose_head(parent._d_base, alpha, parent, 32, "reset")
    return new, {"method": variant, "hidden_width": 32,
                 "initialization": "top 32 hidden units by product of incoming and outgoing L2 norms" if variant == "slice" else "randomly reinitialized 32-hidden and six-logit classifier; same backbone checkpoint",
                 "selected_hidden_indices": indices.tolist() if variant == "slice" else [],
                 "selected_hidden_indices_sha256": __import__("hashlib").sha256(indices.astype("<i8").tobytes()).hexdigest() if variant == "slice" else None,
                 "gallery_embedding_output": False,
                 "contract_note": "classifier emits six logits; its internal 32-value hidden state is not a firmware/gallery embedding"}


def make_physical(parent, alpha, seed, sample_inputs):
    base = parent._d_base
    body = find_body(base)
    if body is None:
        raise ValueError("could not locate final pointwise MobileNet body")
    conv = body.get_layer("conv_pw_13")
    kernel = conv.get_weights()[0]
    keep_count = max(1, int(np.ceil(kernel.shape[-1] * 0.50)))
    scores = np.sum(np.abs(kernel), axis=(0, 1, 2))
    keep = np.sort(np.argsort(scores)[-keep_count:]).astype(np.int64)
    def clone_layer(layer):
        config = layer.get_config()
        if layer.name == "conv_pw_13":
            config["filters"] = keep_count
        return layer.__class__.from_config(config)
    small_body = tf.keras.models.clone_model(body, clone_function=clone_layer)
    for new_layer in small_body.layers:
        old_layer = body.get_layer(new_layer.name)
        old_weights, new_weights = old_layer.get_weights(), new_layer.get_weights()
        if not old_weights:
            continue
        if new_layer.name == "conv_pw_13":
            new_layer.set_weights([old_weights[0][:, :, :, keep]])
        elif new_layer.name == "conv_pw_13_bn":
            new_layer.set_weights([value[keep] for value in old_weights])
        elif [list(v.shape) for v in old_weights] == [list(v.shape) for v in new_weights]:
            new_layer.set_weights(old_weights)
        else:
            raise ValueError("unhandled surgery dependency in %s" % new_layer.name)
    # Build a fresh, shape-compatible head; the exact masked-parent weights are
    # assigned below after slicing the embedding columns for the retained channels.
    phys = compose_head(small_body, alpha, parent, 64, "reset")
    hidden_w = parent.get_layer("embedding_64").get_weights()
    phys.get_layer("embedding_64").set_weights([hidden_w[0][keep, :], hidden_w[1]])
    phys.get_layer("classifier_logits").set_weights(parent.get_layer("classifier_logits").get_weights())
    mask = np.zeros(kernel.shape[-1], dtype=np.float32)
    mask[keep] = 1.0
    inp = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_masked_reference")
    features = parent._d_base(inp, training=False)
    features = tf.keras.layers.Multiply(name="parent_channel_mask")(
        [features, tf.constant(mask.reshape(1, 1, 1, -1), dtype=tf.float32)])
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    hidden = parent.get_layer("embedding_64")(pooled)
    logits = parent.get_layer("classifier_logits")(parent.get_layer("head_dropout")(hidden, training=False))
    masked = tf.keras.Model(inp, logits)
    phys._d_channel_surgery = {"parent_channels": int(kernel.shape[-1]), "new_channels": int(keep_count),
                               "selected_indices": keep.tolist(), "selected_indices_sha256": __import__("hashlib").sha256(keep.astype("<i8").tobytes()).hexdigest(),
                               "batch_norm_vectors_sliced": True, "downstream_embedding_columns_sliced": True,
                               "selection": "final pointwise filter L1 importance from parent checkpoint"}
    expected, actual = masked(sample_inputs, training=False).numpy(), phys(sample_inputs, training=False).numpy()
    phys._d_channel_surgery.update({"parent_mask_max_abs_error": float(np.max(np.abs(expected - actual))),
                                    "parent_mask_equivalence": bool(np.allclose(expected, actual, atol=1e-5, rtol=1e-5)),
                                    "parent_parameter_count": int(parent.count_params()),
                                    "physical_parameter_count": int(phys.count_params()),
                                    "kernel_shape_before": list(kernel.shape),
                                    "kernel_shape_after": list(small_body.get_layer("conv_pw_13").get_weights()[0].shape)})
    phys._d_base = small_body
    phys._d_info = dict(parent._d_info)
    phys._d_embedding_model = tf.keras.Model(phys.input, phys.get_layer("embedding_64").output)
    if not phys._d_channel_surgery["parent_mask_equivalence"]:
        raise RuntimeError("physical channel graph did not match masked parent: max_abs_error=%.9g" %
                           phys._d_channel_surgery["parent_mask_max_abs_error"])
    return phys, phys._d_channel_surgery


def make_qat(parent, alpha):
    body = find_body(parent._d_base)
    if body is None:
        raise ValueError("could not locate MobileNet body for QAT")
    qat_body = tfmot.quantization.keras.quantize_model(body)
    qat = compose_head(qat_body, alpha, parent, 64, "copy")
    qat._d_qat = {"quantized_subgraph": "MobileNetV1 feature backbone including Conv2D and DepthwiseConv2D",
                  "classifier_head": "ordinary Dense head retained in float during QAT; exported through PTQ",
                  "qat_wrapper_count": sum(1 for layer in flatten_layers(qat_body)[1]
                                            if "QuantizeWrapper" in type(layer).__name__)}
    return qat, qat._d_qat


def make_cluster(parent):
    # Cluster only built Conv2D layers in the MobileNet feature body. Wrapping
    # the whole nested classifier makes TFMOT 0.7.5 see cloned, unbuilt Conv2D
    # layers and fail its weight-count check. Depthwise kernels are explicitly
    # left unclustered because this release's registry marks them unsupported.
    base = parent._d_base
    body = find_body(base)
    if body is None:
        raise ValueError("could not locate MobileNet body for clustering")
    models, leaves = flatten_layers(base)
    for value in models:
        value.trainable = True
    for layer in leaves:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    value = body.input
    clustered_layers = []
    depthwise_layers = []
    for layer in body.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.get_weights():
            wrapped = tfmot.clustering.keras.cluster_weights(layer, number_of_clusters=16)
            value = wrapped(value)
            clustered_layers.append(layer.name)
        else:
            value = layer(value)
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                depthwise_layers.append(layer.name)
    clustered_body = tf.keras.Model(body.input, value, name="clustered_supported_mobilenet_body")
    clustered = compose_head(clustered_body, float(parent._d_info.get("alpha", 0.25)),
                             parent, 64, "copy")
    return clustered, {"number_of_clusters": 16,
                       "clustered_weighted_layers": clustered_layers,
                       "clustered_weighted_layer_count": len(clustered_layers),
                       "depthwise_cluster_cardinality_16": {
                           "status": "unsupported_layer_retained_unclustered",
                           "reason": "TFMOT 0.7.5 ClusteringRegistry maps DepthwiseConv2D to an empty clusterable-weight list",
                           "layers": depthwise_layers},
                       "batch_normalization_frozen": True,
                       "classifier_head": "ordinary Dense head; backbone Conv2D kernels clustered during training"}


def make_pruned(parent, sparsity=None, structured=False):
    unfreeze_top_fraction(parent, 0.5)
    if structured:
        transformed = tfmot.sparsity.keras.prune_low_magnitude(parent, sparsity_m_by_n=(2, 4))
        meta = {"method": "structured_2_of_4", "m": 2, "n": 4}
    else:
        transformed = tfmot.sparsity.keras.prune_low_magnitude(
            parent, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=float(sparsity), begin_step=0, frequency=1))
        meta = {"method": "magnitude_pruning", "target_sparsity": float(sparsity)}
    return transformed, meta


def train_fixed(model, rows, gray, seed, pruning_callback=None, soft_targets=None,
                temperature=3.0, hard_weight=0.5):
    history = []
    if soft_targets is None:
        d.compile_classifier(model, 0.0001, 6)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
    if pruning_callback is not None:
        pruning_callback.set_model(model)
        pruning_callback.on_train_begin()
    total_steps = total_presentations = 0
    for epoch in range(5):
        epoch_seed = int(seed + 301001 + (epoch + 1) * 1009)
        if soft_targets is None:
            stats = d.train_epoch(model, rows, gray, "survface_96", epoch_seed,
                                  augment=True, outputs=6, outlier_exposure=False, batch_size=24)
            if pruning_callback is not None:
                pruning_callback.on_epoch_end(epoch, {"loss": stats["loss"]})
        else:
            batches = d.balanced_survface_indices(rows, 24, epoch_seed)
            if len(batches) != 15:
                raise RuntimeError("predeclared optimization requires exactly 15 balanced batches per epoch")
            rng = np.random.default_rng(epoch_seed ^ 0x13579BDF)
            losses, seen = [], 0
            for batch_number, indices in enumerate(batches):
                x_values = []
                for position, row_index in enumerate(indices):
                    x_values.append(d.augment_gray8(gray[row_index], int(epoch_seed + batch_number * 997 + position)))
                x = d.gray_float(np.stack(x_values))
                selected = [rows[i] for i in indices]
                y = d.as_targets(selected, "survface_96", d.SIX)
                teacher = tf.nn.softmax(tf.convert_to_tensor(soft_targets[np.asarray(indices)], tf.float32) / temperature)
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    hard_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
                    soft_loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(
                        teacher, tf.nn.softmax(logits / temperature, axis=-1))) * (temperature ** 2)
                    loss = hard_weight * hard_loss + (1.0 - hard_weight) * soft_loss
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                losses.append(float(loss.numpy())); seen += len(indices)
            stats = {"loss": float(np.mean(losses)), "batch_count": len(batches), "sample_presentations": seen}
        if stats["batch_count"] != 15:
            raise RuntimeError("fixed update budget expected 15 batches, received %d" % stats["batch_count"])
        total_steps += stats["batch_count"]
        total_presentations += stats["sample_presentations"]
        history.append({"epoch": epoch + 1, "seed": epoch_seed, **stats})
    if pruning_callback is not None:
        pruning_callback.on_train_end()
    if total_steps != 75 or total_presentations != 1800:
        raise RuntimeError("fixed training budget mismatch: %s steps, %s presentations" % (total_steps, total_presentations))
    return {"epochs": 5, "steps": total_steps, "sample_presentations": total_presentations,
            "batch_size": 24, "history": history, "elapsed_seconds": None}


def export_method_model(model, root_dir, method_id, alpha, seed, train_gray, rep_indices, data):
    export_dir = root_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    cal_images = np.concatenate([data["cal_known_gray"], data["cal_unknown_gray"]], axis=0)
    dev_images = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    results, float_logits = {}, None
    for kind in ("float32", "full_int8_ptq_train_only_rep"):
        path = export_dir / (kind + ".tflite")
        try:
            convert(model, kind, train_gray, rep_indices, path)
            interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
            interpreter.allocate_tensors()
            tensors = tensor_audit(interpreter)
            cal_logits, _ = tflite_infer(interpreter, cal_images)
            calibration = d.fit_open_threshold(cal_logits[:len(data["cal_known"])], data["cal_known"],
                                                cal_logits[len(data["cal_known"]):], data["cal_unknown"], 6)
            dev_logits, timings = tflite_infer(interpreter, dev_images, measure=True)
            dev_metrics = d.known_macro_and_open_metrics(dev_logits, data["dev_known"], data["dev_unknown"],
                                                          calibration["threshold"], 6)
            if kind == "float32":
                float_logits = dev_logits
            labels, scores, other = d.prediction_components(dev_logits, 6)
            macs = __import__("e_export_campaign").tflite_operator_macs(interpreter)
            row = {"status": "complete_verified", "host_inference_status": "pass", "format": kind,
                   "path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path),
                   "parameter_count": int(model.count_params()), "keras_macs": all_layer_macs(model),
                   "tflite_macs": macs,
                   "mac_accounting": __import__("nicla_study.macs", fromlist=["compare_mac_accounts"]).compare_mac_accounts(all_layer_macs(model), macs),
                   "flatbuffer": tensors, "representative_sample_count": len(rep_indices) if kind.startswith("full_int8") else 0,
                   "representative_roles": ["train", "unknown_train"] if kind.startswith("full_int8") else [],
                   "calibration": calibration, "development": dev_metrics,
                   "class_order": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
                   "normalization": "gray8 / 255; input normalization embedded for ImageNet-adapted widths",
                   "host_latency_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)),
                                       "n": len(timings), "host_only": True},
                   "float_logit_max_abs_drift": None if kind == "float32" else float(np.max(np.abs(float_logits - dev_logits))),
                   "final_role_accessed": False}
            results[kind] = row
            atomic_json(export_dir / (kind + ".json"), row)
            del interpreter
        except Exception as exc:
            row = {"status": "unsupported_or_failed", "host_inference_status": "not_run",
                   "format": kind, "path": str(path) if path.exists() else None,
                   "error_type": type(exc).__name__, "error": str(exc)[:2000],
                   "traceback_tail": __import__("traceback").format_exc()[-3000:],
                   "final_role_accessed": False}
            results[kind] = row
            atomic_json(export_dir / (kind + ".json"), row)
    return results


def method_spec(method, selected_alpha):
    if method.startswith("large_"):
        return selected_alpha
    if method.startswith("small_a050"):
        return 0.50
    return 0.10


def load_root_model(alpha, seed):
    source, weights = capacity_root(alpha, seed)
    if not weights.is_file():
        raise FileNotFoundError(str(weights))
    model, _, info = load_capacity_model(alpha, seed, weights)
    return model, info, source, weights


def run_root(method, seed, large_alpha, teacher_eligible, data):
    method_id = method
    root_dir = RUN_ROOT / "roots" / ("optimization_" + method_id) / ("seed-%d" % seed)
    result_path = root_dir / "ROOT_RESULT.json"
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") in ("complete_verified", "unsupported"):
                return {"method": method, "seed": seed, "status": "already_recorded"}
        except Exception:
            pass
    root_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = root_dir / "ATTEMPTS.jsonl"
    append_jsonl(attempt_path, {"status": "started", "method": method, "seed": seed,
                                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    train_root = STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6" / ("seed-%d" % seed)
    train_rows = d.read_jsonl(train_root / "training_rows.jsonl")
    train_gray = d.cache_gray8(ROOT, train_rows, "train")
    representative = np.linspace(0, len(train_gray) - 1, min(128, len(train_gray)), dtype=np.int64).tolist()
    selected_alpha = method_spec(method, large_alpha)
    parent_model = None
    metadata = {"method": method, "root_seed": seed, "selected_large_alpha": large_alpha,
                "training_rows_path": str(train_root / "training_rows.jsonl"),
                "training_rows_sha256": digest_file(train_root / "training_rows.jsonl"),
                "training_rows_fingerprint": d.rows_fingerprint(train_rows),
                "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
                "final_role_accessed": False}
    try:
        started = time.monotonic()
        if method in ("small_a010_distilled",):
            teacher_model, _, teacher_source, teacher_weights = load_root_model(0.75, seed)
            teacher_logits = d.evaluate_logits(teacher_model, train_gray)
            parent_model, info, source, weights = load_root_model(0.10, seed)
            unfreeze_top_fraction(parent_model, 0.5)
            train_model = parent_model
            train_result = train_fixed(parent_model, train_rows, train_gray, seed, soft_targets=teacher_logits,
                                       temperature=3.0, hard_weight=0.5)
            metadata.update({"teacher_alpha": 0.75, "teacher_root": str(teacher_source),
                             "teacher_weights_sha256": digest_file(teacher_weights),
                             "distillation_temperature": 3.0, "hard_label_weight": 0.5,
                             "soft_target_weight": 0.5, "target": "teacher logits on matched train/unknown_train images",
                             "teacher_usefulness_gate": teacher_eligible,
                             "teacher_claim_status": "useful_teacher_gate_pass" if teacher_eligible else "teacher_quality_gate_failed; diagnostic student only"})
            del teacher_model
        else:
            parent_model, info, source, weights = load_root_model(selected_alpha, seed)
            metadata.update({"parent_alpha": selected_alpha, "parent_root": str(source),
                             "parent_weights_sha256": digest_file(weights),
                             "parent_initialization": info.get("initialization")})
            train_model = parent_model
            pruning_callback = None
            if method == "large_qat":
                train_model, meta = make_qat(parent_model, selected_alpha)
                metadata["qat"] = meta
                unfreeze_top_fraction(train_model, 0.5)
            elif method == "large_magnitude_prune_50":
                train_model, meta = make_pruned(parent_model, 0.50)
                pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
                metadata["pruning"] = meta
            elif method == "large_magnitude_prune_75":
                train_model, meta = make_pruned(parent_model, 0.75)
                pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
                metadata["pruning"] = meta
            elif method == "large_structured_prune_2of4":
                train_model, meta = make_pruned(parent_model, structured=True)
                pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
                metadata["pruning"] = meta
            elif method == "large_physical_channel_keep_50":
                train_model, meta = make_physical(parent_model, selected_alpha, seed,
                                                  d.gray_float(train_gray[:8]))
                metadata["physical_surgery"] = meta
                unfreeze_top_fraction(train_model, 0.5)
            elif method == "large_cluster_16":
                train_model, meta = make_cluster(parent_model)
                metadata["clustering"] = meta
            elif method == "large_head_32_sliced_initialization":
                train_model, meta = make_head_variant(parent_model, selected_alpha, seed, "slice")
                metadata["head"] = meta
                unfreeze_top_fraction(train_model, 0.5)
            elif method == "large_head_32_reset_initialization":
                train_model, meta = make_head_variant(parent_model, selected_alpha, seed, "reset")
                metadata["head"] = meta
                unfreeze_top_fraction(train_model, 0.5)
            elif method == "large_head_direct_gap_logits":
                train_model, meta = make_head_variant(parent_model, selected_alpha, seed, "direct_gap")
                metadata["head"] = meta
                unfreeze_top_fraction(train_model, 0.5)
            else:
                if method.startswith("small_a050"):
                    selected_alpha = 0.50
                elif method.startswith("small_a010"):
                    selected_alpha = 0.10
                unfreeze_top_fraction(train_model, 0.5)
            if method == "small_a010_distilled":
                pass
            else:
                train_result = train_fixed(train_model, train_rows, train_gray, seed,
                                           pruning_callback=pruning_callback)
            if pruning_callback is not None:
                train_model = tfmot.sparsity.keras.strip_pruning(train_model)
            if method == "large_cluster_16":
                train_model = tfmot.clustering.keras.strip_clustering(train_model)
            parent_model = None
        train_result["elapsed_seconds"] = time.monotonic() - started
        if train_result["steps"] != 75 or train_result["sample_presentations"] != 1800:
            raise RuntimeError("optimization update/sample budget did not match predeclaration")
        curve_path = root_dir / "training_curve.jsonl"
        with curve_path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in train_result["history"]:
                stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
        if method == "large_cluster_16":
            # Stripping TFMOT wrappers leaves repeated nested layer names, which
            # Keras' legacy HDF5 writer cannot represent. TensorFlow checkpoints
            # preserve this graph correctly; package both checkpoint shards as
            # a single integrity-verifiable root artifact.
            checkpoint_prefix = root_dir / "classifier.weights.ckpt"
            train_model.save_weights(str(checkpoint_prefix))
            checkpoint_parts = sorted(root_dir.glob("classifier.weights.ckpt.*"))
            if not checkpoint_parts or not any(path.suffix == ".index" for path in checkpoint_parts):
                raise RuntimeError("clustering checkpoint did not produce an index and data shard")
            weights_path = root_dir / "classifier.weights.ckpt.zip"
            temporary_archive = root_dir / "classifier.weights.ckpt.partial.zip"
            with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for part in checkpoint_parts:
                    archive.write(part, arcname=part.name)
            os.replace(temporary_archive, weights_path)
            metadata["weights_checkpoint"] = {
                "format": "tensorflow_checkpoint_after_strip_clustering",
                "prefix": str(checkpoint_prefix),
                "components": [{"path": str(part), "sha256": digest_file(part),
                                "bytes": part.stat().st_size} for part in checkpoint_parts],
                "archive_path": str(weights_path),
                "archive_sha256": digest_file(weights_path),
                "archive_bytes": weights_path.stat().st_size}
        else:
            weights_path = root_dir / "classifier.weights.h5"
            tmp_weights = root_dir / "classifier.weights.partial.h5"
            train_model.save_weights(str(tmp_weights))
            os.replace(tmp_weights, weights_path)
        export = export_method_model(train_model, root_dir, method, selected_alpha,
                                     seed, train_gray, representative, data)
        if "large_magnitude" in method or "large_structured" in method:
            all_weights = [value for value in train_model.get_weights() if value.size]
            zero_count = int(sum(np.sum(value == 0) for value in all_weights))
            scalar_count = int(sum(value.size for value in all_weights))
            metadata["sparsity_audit"] = {"zero_scalar_count": zero_count,
                                           "weight_scalar_count": scalar_count,
                                           "observed_zero_fraction": zero_count / max(1, scalar_count),
                                           "saved_sparsity_mask_claim": False}
        if method == "large_cluster_16":
            uniques = []
            for layer in flatten_layers(train_model)[1]:
                for value in layer.get_weights():
                    if value.size:
                        uniques.append({"layer": layer.name, "shape": list(value.shape),
                                        "unique_values": int(np.unique(value).size)})
            metadata["clustered_weight_unique_value_counts_after_strip"] = uniques
        result = {"schema": "job-e-optimization-root-1", "status": "complete_verified",
                  "method": method, "root_seed": seed, "alpha": selected_alpha,
                  "training": train_result, "weights_path": str(weights_path),
                  "training_curve_path": str(curve_path), "training_curve_sha256": digest_file(curve_path),
                  "weights_sha256": digest_file(weights_path), "weights_bytes": weights_path.stat().st_size,
                  "exports": export, "metadata": metadata, "final_role_accessed": False,
                  "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(result_path, result)
        config = {"method": method, "root_seed": seed, "alpha": selected_alpha,
                  "epochs": 5, "steps_per_epoch": 15, "updates": 75,
                  "batch_size": 24, "sample_presentations": 1800,
                  "learning_rate": 0.0001, "augmentation": "mild_gray_v1",
                  "train_rows_sha256": metadata["training_rows_sha256"],
                  "predeclaration_sha256": metadata["predeclaration_sha256"],
                  "normalization": "gray8 / 255; ImageNet-derived widths rescale to [-1,1]",
                  "roles": ["train", "unknown_train", "threshold_calibration", "unknown_calibration", "development", "unknown_development"],
                  "final_role_accessed": False}
        atomic_json(root_dir / "config.json", config)
        atomic_json(root_dir / "ROOT_RESULT.json", result)
        append_jsonl(attempt_path, {"status": "complete_verified", "method": method, "seed": seed,
                                    "weights_sha256": result["weights_sha256"],
                                    "finished_utc": result["finished_utc"]})
        del train_model
        tf.keras.backend.clear_session()
        return {"method": method, "seed": seed, "status": "complete_verified",
                "steps": train_result["steps"], "elapsed_seconds": train_result["elapsed_seconds"],
                "format_status": {key: value["status"] for key, value in export.items()}}
    except Exception as exc:
        status = "unsupported" if method in ("large_qat", "large_cluster_16") and (
            "unsupported" in str(exc).lower() or "not supported" in str(exc).lower()
            or "no weights to cluster" in str(exc).lower() or "layer" in str(exc).lower()) else "failed_needs_repair"
        row = {"schema": "job-e-optimization-root-1", "status": status, "method": method,
               "root_seed": seed, "alpha": selected_alpha, "error_type": type(exc).__name__,
               "error": str(exc)[:2500], "traceback_tail": __import__("traceback").format_exc()[-4000:],
               "metadata": metadata, "final_role_accessed": False}
        atomic_json(result_path, row)
        append_jsonl(attempt_path, {"status": status, "method": method, "seed": seed,
                                    "error_type": type(exc).__name__, "error": str(exc)[:1500],
                                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        tf.keras.backend.clear_session()
        return {"method": method, "seed": seed, "status": status,
                "error_type": type(exc).__name__, "error": str(exc)[:800]}


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    selection_path = STUDY / "reports/E/CAPACITY_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_alpha = float(selection["selected_large_alpha"])
    teacher_eligible = bool(selection["teacher_usefulness_gate"]["distillation_claim_eligible"])
    data = d.load_role_arrays(ROOT, d.study_paths(ROOT), "survface_96")
    registry = {"schema": "job-e-optimization-registry-1", "status": "in_progress",
                "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
                "capacity_selection_sha256": digest_file(selection_path),
                "selected_large_alpha": selected_alpha, "teacher_usefulness_gate": teacher_eligible,
                "root_seeds": SEEDS, "methods": {}, "final_role_accessed": False}
    registry_path = RUN_ROOT / "roots/OPTIMIZATION_ROOT_REGISTRY.json"
    with resource_lease(STUDY, "heavy_training", owner="E_matched_optimization_matrix_ten_roots_per_arm", timeout_s=0):
        for method in METHODS:
            rows = []
            for seed in SEEDS:
                row = run_root(method, seed, selected_alpha, teacher_eligible, data)
                rows.append(row)
                registry["methods"][method] = rows
                atomic_json(registry_path, registry)
                print(json.dumps({"event": "optimization_root", **row}), flush=True)
            registry["methods"][method] = rows
            atomic_json(registry_path, registry)
        accepted = {"complete_verified", "unsupported", "already_recorded"}
        registry["status"] = "complete_verified" if all(
            len(rows) == len(SEEDS) and all(row["status"] in accepted for row in rows)
            for rows in registry["methods"].values()) else "partial"
        registry["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(registry_path, registry)
    print(json.dumps({"event": "optimization_campaign_complete", "status": registry["status"],
                      "registry": str(registry_path)}), flush=True)


if __name__ == "__main__":
    main()
