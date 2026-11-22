"""Executable Job D transfer-learning and five-person recognition campaign.

The code intentionally keeps role access through nicla_study.data_pipeline,
records every model root, and acquires the shared heavy-training lease before
any gradient update.  Run from the workspace root with PYTHONPATH=FaceRecognition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import h5py
import numpy as np
import tensorflow as tf

from nicla_study.data_pipeline import load_pixels, read_jsonl, safe_path
from nicla_study.fs import atomic_write_json, sha256_file
from nicla_study.roles import assert_role_access, validate_manifest
from nicla_study.state import resource_lease


SCHEMA = "job-d-campaign-1"
KNOWN = ["K01", "K02", "K03", "K04", "K05"]
SIX = KNOWN + ["OTHER"]
EI_CHECKPOINTS = {
    0.1: "work/project-finish-v4/models/pretrained/edgeimpulse_mobilenetv1_0.1_96_grayscale.h5",
    0.25: "work/project-finish-v4/models/pretrained/edgeimpulse_mobilenetv1_0.25_96_grayscale.h5",
}
EI_HASHES = {
    0.1: "401cc0929ae5592f53ec25c5f6cd9228e5bf905b0065644f1c33483b5b2589c3",
    0.25: "9b2f73e58d3ea8c86ff04b38cd1ea26235b73e3270c9b87afe6633dd58e94eff",
}
EI_URLS = {
    0.1: "https://cdn.edgeimpulse.com/transfer-learning-weights/edgeimpulse/MobileNetV1.0_1.96x96.grayscale.bsize_96.lr_0_05.epoch_363.val_accuracy_0.14.hdf5",
    0.25: "https://cdn.edgeimpulse.com/transfer-learning-weights/edgeimpulse/MobileNetV1.0_25.96x96.grayscale.bsize_96.lr_0_05.epoch_286.val_loss_3.54.val_accuracy_0.28.hdf5",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes((json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())


def write_json(path: Path, value) -> None:
    atomic_write_json(path, value)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()


def env_receipt(workspace: Path) -> dict:
    lock = workspace / "work/project-finish-v4/config/training-environment.lock"
    try:
        lock_values = lock.read_text(encoding="utf-8").splitlines()
    except OSError:
        lock_values = []
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "tensorflow": tf.__version__, "keras": getattr(tf.keras, "__version__", "bundled-with-tensorflow"),
        "numpy": np.__version__, "h5py": h5py.__version__, "processor": platform.processor(),
        "interpreter": sys.executable, "thread_budget": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "training_environment_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "training_environment_lock_lines": lock_values,
    }


def study_paths(workspace: Path) -> dict:
    study = workspace / "work/project-finish-v4"
    return {
        "study": study,
        "data": study / "data",
        "models": study / "models",
        "runs": study / "runs/D",
        "reports": study / "reports/D",
        "handoff": study / "handoffs/D",
        "primary": study / "data/primary_split_manifest.jsonl",
        "olivetti": study / "data/olivetti_sanity_manifest.jsonl",
        "presentation": study / "data/presentation_manifest.jsonl",
        "source": study / "data/source_manifest.jsonl",
        "quality": study / "data/quality_inventory.jsonl",
        "phashes": study / "data/permitted_survface_phashes.jsonl",
        "cross_phash": study / "data/cross_identity_phash_candidates.jsonl",
        "roster": study / "data/roster.json",
        "seed_registry": study / "config/seed_registry.json",
        "matrix": study / "D_MATRIX_v1.json",
        "pretrained": study / "models/pretrained",
    }


def dataset_role_rows(workspace: Path, manifest: Path, dataset: str, roles: tuple[str, ...], action: str,
                      frozen_receipt: dict | None = None) -> list[dict]:
    rows = read_jsonl(manifest)
    rows = [r for r in rows if r.get("dataset") == dataset and r.get("role") in set(roles)]
    if not rows:
        raise ValueError("no rows for %s/%s in %s" % (dataset, roles, manifest))
    assert_role_access(rows, action, frozen_receipt=frozen_receipt)
    for row in rows:
        safe_path(row, workspace)
    return rows


def cache_gray8(workspace: Path, rows: list[dict], action: str, frozen_receipt: dict | None = None) -> np.ndarray:
    values = [load_pixels(row, workspace, output="gray8", action=action, frozen_receipt=frozen_receipt)
              for row in rows]
    result = np.stack(values).astype(np.uint8)
    if result.shape[1:] != (96, 96, 1):
        raise ValueError("unexpected image shape: %r" % (result.shape,))
    return result


def augment_gray8(image: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.uint8)
    if bool(rng.integers(0, 2)):
        pixels = pixels[:, ::-1].copy()
    gain = float(rng.uniform(0.92, 1.08))
    offset = float(rng.uniform(-8.0, 8.0))
    return np.clip(np.rint(pixels.astype(np.float32) * gain + offset), 0, 255).astype(np.uint8)


def gray_float(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / np.float32(255.0)


def keras_mobilenet(workspace: Path, alpha: float, pretrained: bool, seed: int) -> tuple[tf.keras.Model, tf.keras.Model, dict]:
    tf.keras.utils.set_random_seed(int(seed))
    full = tf.keras.applications.MobileNet(
        input_shape=(96, 96, 1), alpha=float(alpha), weights=None,
        include_top=True, classes=1000, dropout=0.001)
    loaded_hash = None
    if pretrained:
        path = workspace / EI_CHECKPOINTS[float(alpha)]
        actual_hash = sha256_file(path)
        if actual_hash != EI_HASHES[float(alpha)]:
            raise ValueError("pretrained checkpoint hash mismatch for alpha %s" % alpha)
        full.load_weights(str(path))
        loaded_hash = actual_hash
    # Explicitly keep the reference Keras/EI topology through conv_pw_13_relu,
    # then reproduce its global-average feature pooling without the 1000-way head.
    features = tf.keras.Model(full.input, full.get_layer("conv_pw_13_relu").output,
                              name="mobilenetv1_%s_96_gray_backbone" % str(alpha))
    info = {
        "alpha": float(alpha), "input_shape": [1, 96, 96, 1],
        "initialization": "Edge Impulse HDF5" if pretrained else "random Keras MobileNetV1",
        "checkpoint_sha256": loaded_hash, "full_reference_model": full,
        "feature_model": features,
        "input_range": "float32 gray8/255 in [0,1]",
        "backbone_feature_layer": "conv_pw_13_relu",
        "backbone_feature_shape": [int(x) if x is not None else None for x in features.output_shape[1:]],
        "checkpoint_head_removed": ["global_average_pooling2d", "reshape", "dropout", "conv_preds", "reshape", "predictions"],
    }
    return full, features, info


def build_classifier(workspace: Path, alpha: float, pretrained: bool, outputs: int,
                     seed: int, name: str) -> tuple[tf.keras.Model, tf.keras.Model, dict]:
    _, backbone, info = keras_mobilenet(workspace, alpha, pretrained, seed)
    x_in = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_float01")
    base = tf.keras.Model(backbone.input, backbone.output, name="selected_backbone")
    base.trainable = bool(not pretrained)
    x = base(x_in, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    embedding = tf.keras.layers.Dense(64, activation="relu", name="embedding_64")(x)
    head = tf.keras.layers.Dropout(0.1, name="head_dropout")(embedding)
    logits = tf.keras.layers.Dense(int(outputs), activation=None, name="classifier_logits")(head)
    model = tf.keras.Model(x_in, logits, name=name)
    embed_model = tf.keras.Model(x_in, embedding, name=name + "_embedding")
    model._d_base = base
    # Do not attach the complete 1000-class reference model to the task model:
    # its removed conv_preds variables would otherwise be tracked as unused trainables.
    model._d_info = {k: v for k, v in info.items() if k not in ("full_reference_model", "feature_model")}
    model._d_embedding_model = embed_model
    return model, embed_model, info


def compile_classifier(model: tf.keras.Model, lr: float, outputs: int, outlier_exposure: bool = False,
                       oe_weight: float = 0.25) -> None:
    if outlier_exposure:
        def loss(y_true, logits):
            known_targets = y_true[:, :5]
            unknown = y_true[:, 5]
            logp = tf.nn.log_softmax(logits, axis=-1)
            p = tf.nn.softmax(logits, axis=-1)
            ce = -tf.reduce_sum(known_targets * logp, axis=-1)
            kl_uniform = tf.reduce_sum(p * (tf.math.log(tf.maximum(p, 1e-7)) + tf.math.log(5.0)), axis=-1)
            known_sum = tf.reduce_sum(1.0 - unknown) + 1e-7
            unknown_sum = tf.reduce_sum(unknown) + 1e-7
            return 0.5 * tf.reduce_sum(ce * (1.0 - unknown)) / known_sum + \
                   0.5 * oe_weight * tf.reduce_sum(kl_uniform * unknown) / unknown_sum
        model.compile(optimizer=tf.keras.optimizers.Adam(float(lr)), loss=loss)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(float(lr)),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))


def unfreeze_top_fraction(model: tf.keras.Model, fraction: float) -> dict:
    base = model._d_base
    base.trainable = True
    layers = list(base.layers)
    cutoff = max(1, int(math.floor(len(layers) * (1.0 - float(fraction)))))
    for index, layer in enumerate(layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return {
        "backbone_layers_total": len(layers), "freeze_cutoff_index": cutoff,
        "backbone_layers_trainable": [layer.name for layer in layers if layer.trainable],
        "batch_normalization_frozen": True,
    }


def as_targets(rows: list[dict], dataset: str, classes: list[str], for_oe: bool = False) -> np.ndarray:
    if for_oe:
        out = np.zeros((len(rows), 6), dtype=np.float32)
        for i, row in enumerate(rows):
            label = str(row.get("label_id"))
            if label == "OTHER" or row.get("identity_partition") == "unknown":
                out[i, 5] = 1.0
            else:
                out[i, classes.index(label)] = 1.0
        return out
    ids = {label: index for index, label in enumerate(classes)}
    if dataset == "survface_96":
        return np.asarray([ids[str(row["label_id"])] for row in rows], dtype=np.int64)
    labels = sorted({str(row["label_id"]) for row in rows})
    label_ids = {label: index for index, label in enumerate(labels)}
    return np.asarray([label_ids[str(row["label_id"])] for row in rows], dtype=np.int64)


def balanced_survface_indices(rows: list[dict], batch_size: int, seed: int) -> list[list[int]]:
    if batch_size != 24:
        raise ValueError("frozen SurvFace batch size is 24")
    by_label = {label: [] for label in SIX}
    for i, row in enumerate(rows):
        label = "OTHER" if row.get("identity_partition") == "unknown" or row.get("label_id") == "OTHER" else row.get("label_id")
        if label not in by_label:
            raise ValueError("unexpected SurvFace training label %s" % label)
        by_label[label].append(i)
    missing = [label for label, values in by_label.items() if not values]
    if missing:
        raise ValueError("missing training classes: %s" % missing)
    rng = np.random.default_rng(seed)
    unknown_by_id: dict[str, list[int]] = defaultdict(list)
    for index in by_label["OTHER"]:
        unknown_by_id[str(rows[index]["identity_id"])].append(index)
    identities = sorted(unknown_by_id)
    if len(identities) < 10:
        raise ValueError("OTHER training batches require broad identity diversity")
    # Each of the five known classes traverses its 60 rows once per epoch.
    for label in KNOWN:
        rng.shuffle(by_label[label])
    class_cursors = {label: 0 for label in KNOWN}
    batches = []
    n_batches = max(1, math.ceil(max(len(by_label[label]) for label in KNOWN) / 4))
    for b in range(n_batches):
        chosen = []
        for label in KNOWN:
            pool = by_label[label]
            for _ in range(4):
                chosen.append(pool[class_cursors[label] % len(pool)])
                class_cursors[label] += 1
        picked_ids = rng.choice(identities, size=4, replace=False)
        for identity in picked_ids:
            chosen.append(int(rng.choice(unknown_by_id[str(identity)])))
        rng.shuffle(chosen)
        batches.append(chosen)
    return batches


def balanced_olivetti_indices(rows: list[dict], batch_size: int, seed: int) -> list[list[int]]:
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_id[str(row["label_id"])].append(i)
    ids = sorted(by_id)
    if batch_size != len(ids):
        raise ValueError("Olivetti batches are one image per identity")
    rng = np.random.default_rng(seed)
    for values in by_id.values():
        rng.shuffle(values)
    n_batches = max(len(v) for v in by_id.values())
    batches = []
    for b in range(n_batches):
        batch = [by_id[identity][b % len(by_id[identity])] for identity in ids]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def train_epoch(model: tf.keras.Model, rows: list[dict], gray: np.ndarray, dataset: str,
                seed: int, augment: bool, outputs: int, outlier_exposure: bool = False,
                batch_size: int = 24) -> dict:
    batches = (balanced_survface_indices(rows, 24, seed) if dataset == "survface_96"
               else balanced_olivetti_indices(rows, len(sorted({r['label_id'] for r in rows})), seed))
    losses = []
    correct = 0
    total_known = 0
    seen = 0
    rng = np.random.default_rng(seed ^ 0x13579BDF)
    for batch_number, indices in enumerate(batches):
        x_values = []
        for position, row_index in enumerate(indices):
            image = gray[row_index]
            if augment:
                image = augment_gray8(image, int(seed + batch_number * 997 + position))
            x_values.append(image)
        x = gray_float(np.stack(x_values))
        selected = [rows[i] for i in indices]
        if outlier_exposure:
            y = as_targets(selected, dataset, KNOWN, for_oe=True)
        elif dataset == "survface_96":
            y = as_targets(selected, dataset, SIX if outputs == 6 else KNOWN)
        else:
            labels = sorted({str(row["label_id"]) for row in rows})
            y = np.asarray([labels.index(str(row["label_id"])) for row in selected], dtype=np.int64)
        result = model.train_on_batch(x, y, return_dict=True)
        losses.append(float(result["loss"]))
        logits = model(x, training=False).numpy()
        pred = np.argmax(logits, axis=-1)
        if outlier_exposure:
            known = [i for i, row in enumerate(selected) if row.get("identity_partition") != "unknown" and row.get("label_id") != "OTHER"]
            target = np.asarray([KNOWN.index(str(selected[i]["label_id"])) for i in known])
            correct += int(np.sum(pred[known] == target)); total_known += len(known)
        else:
            target = np.asarray(y)
            correct += int(np.sum(pred == target)); total_known += len(target)
        seen += len(indices)
    return {"loss": float(np.mean(losses)), "batch_count": len(batches), "sample_presentations": seen,
            "balanced_training_accuracy": correct / max(1, total_known)}


def evaluate_logits(model: tf.keras.Model, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    output = []
    for start in range(0, len(images), batch_size):
        output.append(model(gray_float(images[start:start + batch_size]), training=False).numpy())
    return np.concatenate(output, axis=0) if output else np.zeros((0, model.output_shape[-1]), np.float32)


def macro_known_accuracy(logits: np.ndarray, rows: list[dict], labels: list[str]) -> float:
    y_pred = np.argmax(logits, axis=1)
    rates = []
    for i, label in enumerate(labels):
        idx = [j for j, r in enumerate(rows) if str(r.get("label_id")) == label]
        if idx:
            rates.append(float(np.mean(y_pred[idx] == i)))
    return float(np.mean(rates)) if rates else float("nan")


def stop_metrics(model: tf.keras.Model, stop_rows: list[dict], stop_gray: np.ndarray) -> dict:
    labels = KNOWN
    logits = evaluate_logits(model, stop_gray)
    return {"known_macro_accuracy": macro_known_accuracy(logits, stop_rows, labels),
            "known_mean_loss": float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(
                as_targets(stop_rows, "survface_96", labels), logits, from_logits=True).numpy())),
            "logits_shape": list(logits.shape)}


def olivetti_stop_metrics(model: tf.keras.Model, rows: list[dict], gray: np.ndarray) -> dict:
    labels = sorted({str(row["label_id"]) for row in rows})
    logits = evaluate_logits(model, gray)
    y_true = np.asarray([labels.index(str(row["label_id"])) for row in rows], dtype=np.int64)
    pred = np.argmax(logits, axis=1)
    per_id = [float(np.mean(pred[[i for i, row in enumerate(rows) if row["label_id"] == label]] == index))
              for index, label in enumerate(labels)]
    return {"known_macro_accuracy": float(np.mean(per_id)),
            "known_mean_loss": float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(
                y_true, logits, from_logits=True).numpy())),
            "logits_shape": list(logits.shape)}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=1, keepdims=True)
    ex = np.exp(values)
    return ex / np.sum(ex, axis=1, keepdims=True)


def prediction_components(logits: np.ndarray, outputs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = softmax_np(logits)
    if outputs == 6:
        known_probs = probs[:, :5]
        labels = np.argmax(known_probs, axis=1)
        scores = np.max(known_probs, axis=1)
        predicted_other = np.argmax(probs, axis=1) == 5
        return labels, scores, predicted_other
    if outputs == 5:
        labels = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)
        return labels, scores, np.zeros(len(labels), dtype=bool)
    raise ValueError("open-set prediction components need five or six outputs")


def row_label_index(row: dict) -> int:
    return KNOWN.index(str(row["label_id"]))


def known_macro_and_open_metrics(logits: np.ndarray, known_rows: list[dict], unknown_rows: list[dict],
                                 threshold: float, outputs: int) -> dict:
    known_logits = np.asarray(logits[:len(known_rows)])
    unknown_logits = np.asarray(logits[len(known_rows):])
    kn_pred, kn_score, kn_other = prediction_components(known_logits, outputs)
    un_pred, un_score, un_other = prediction_components(unknown_logits, outputs)
    kn_true = np.asarray([row_label_index(row) for row in known_rows], dtype=np.int64)
    kn_accept = (kn_score >= float(threshold)) & ~kn_other
    un_accept = (un_score >= float(threshold)) & ~un_other
    per_person = {}
    for index, label in enumerate(KNOWN):
        selected = np.asarray([row_label_index(row) == index for row in known_rows])
        accept = kn_accept[selected]
        pred = kn_pred[selected]
        true = kn_true[selected]
        per_person[label] = {
            "n": int(np.sum(selected)),
            "correct_identification_rate": float(np.mean(pred == true)),
            "correct_known_acceptance_rate": float(np.mean(accept & (pred == true))),
            "wrong_known_identity_rate": float(np.mean(accept & (pred != true))),
            "known_rejection_rate": float(np.mean(~accept)),
            "accepted_coverage": float(np.mean(accept)),
        }
    unknown_by_id = defaultdict(list)
    for row, accepted, predicted_other in zip(unknown_rows, un_accept, un_other):
        unknown_by_id[str(row.get("identity_id"))].append((bool(accepted), bool(predicted_other)))
    unknown_identity_fpir = [np.mean([a for a, _ in items]) for items in unknown_by_id.values()]
    unknown_rejection = float(np.mean(~un_accept)) if len(un_accept) else None
    correct_accept = [row["correct_known_acceptance_rate"] for row in per_person.values()]
    known_ident_accuracy = float(np.mean([row["correct_identification_rate"] for row in per_person.values()]))
    class_macro_utility = float((np.mean(correct_accept) + unknown_rejection) / 2.0) if unknown_rejection is not None else None
    other_accuracy = None
    if outputs == 6:
        other_accuracy = float(np.mean(un_other)) if len(un_other) else None
    return {
        "outputs": outputs, "score": "max known softmax probability; model-specific calibrated threshold",
        "threshold": float(threshold), "known_n": len(known_rows), "unknown_n": len(unknown_rows),
        "known_identity_macro_identification": known_ident_accuracy,
        "known_identity_macro_correct_acceptance": float(np.mean(correct_accept)),
        "known_identity_macro_wrong_known_identity": float(np.mean([r["wrong_known_identity_rate"] for r in per_person.values()])),
        "known_identity_macro_rejection": float(np.mean([r["known_rejection_rate"] for r in per_person.values()])),
        "known_identity_macro_coverage": float(np.mean([r["accepted_coverage"] for r in per_person.values()])),
        "unknown_false_identification_rate": float(np.mean(un_accept)) if len(un_accept) else None,
        "unknown_identity_macro_false_identification_rate": float(np.mean(unknown_identity_fpir)) if unknown_identity_fpir else None,
        "unknown_rejection_rate": unknown_rejection,
        "other_accuracy": other_accuracy,
        "development_macro_utility": class_macro_utility,
        "per_person": per_person,
        "unknown_identity_count": len(unknown_by_id),
        "unknown_identity_coverage_min": float(min(unknown_identity_fpir)) if unknown_identity_fpir else None,
        "unknown_identity_coverage_max": float(max(unknown_identity_fpir)) if unknown_identity_fpir else None,
    }


def fit_open_threshold(known_cal_logits: np.ndarray, known_rows: list[dict],
                       unknown_cal_logits: np.ndarray, unknown_rows: list[dict], outputs: int,
                       max_unknown_fpir: float = 0.05) -> dict:
    kn_pred, kn_score, kn_other = prediction_components(known_cal_logits, outputs)
    un_pred, un_score, un_other = prediction_components(unknown_cal_logits, outputs)
    kn_true = np.asarray([row_label_index(row) for row in known_rows], dtype=np.int64)
    thresholds = sorted(set([0.0] + [float(v) for v in np.concatenate([kn_score, un_score])]))
    thresholds.append(float(np.nextafter(max(thresholds), math.inf)))
    candidates = []
    for threshold in thresholds:
        accepted_known = (kn_score >= threshold) & ~kn_other
        accepted_unknown = (un_score >= threshold) & ~un_other
        correct = accepted_known & (kn_pred == kn_true)
        per_class = [float(np.mean(correct[[i for i, row in enumerate(known_rows)
                                           if row_label_index(row) == ci]]))
                     for ci in range(5)]
        macro_correct = float(np.mean(per_class))
        fpir = float(np.mean(accepted_unknown)) if len(accepted_unknown) else 0.0
        coverage = float(np.mean(accepted_known)) if len(accepted_known) else 0.0
        candidates.append({"threshold": threshold, "calibration_macro_correct_acceptance": macro_correct,
                           "calibration_known_coverage": coverage, "calibration_unknown_fpir": fpir,
                           "constraint_met": fpir <= max_unknown_fpir})
    eligible = [row for row in candidates if row["constraint_met"]]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["calibration_macro_correct_acceptance"],
                                                row["calibration_known_coverage"],
                                                -row["calibration_unknown_fpir"], row["threshold"]))
    else:
        chosen = min(candidates, key=lambda row: (row["calibration_unknown_fpir"],
                                                   -row["calibration_macro_correct_acceptance"],
                                                   -row["threshold"]))
    return {"status": "calibrated", "score_name": "max_known_softmax_probability",
            "outputs": outputs, "threshold": float(chosen["threshold"]),
            "max_unknown_fpir_target": max_unknown_fpir,
            "calibration_constraint_met": bool(chosen["constraint_met"]),
            "calibration_metrics": chosen, "threshold_candidates": len(candidates),
            "calibration_known_n": len(known_rows), "calibration_unknown_n": len(unknown_rows),
            "calibration_unknown_identity_count": len({r.get("identity_id") for r in unknown_rows})}


def stage_plan(config: dict, pretrained: bool, is_scratch: bool = False) -> list[dict]:
    if is_scratch:
        return [{"stage": "scratch_all", "lr": float(config["learning_rate"]),
                 "epochs_max": int(config["epochs_max"]), "unfreeze_top_fraction": 1.0,
                 "augmentation": config["augmentation"]}]
    return [
        {"stage": "frozen_head", "lr": float(config["head_lr"]),
         "epochs_max": int(config["head_epochs_max"]), "unfreeze_top_fraction": 0.0,
         "augmentation": config["augmentation"]},
        {"stage": "partial_finetune", "lr": float(config["fine_tune_lr"]),
         "epochs_max": int(config["fine_tune_epochs_max"]),
         "unfreeze_top_fraction": float(config["unfreeze_top_fraction"]),
         "augmentation": config["augmentation"]},
    ]


def train_model(workspace: Path, alpha: float, pretrained: bool, outputs: int, seed: int,
                name: str, train_rows: list[dict], train_gray: np.ndarray,
                stop_rows: list[dict], stop_gray: np.ndarray, config: dict,
                is_scratch: bool = False, outlier_exposure: bool = False,
                dataset: str = "survface_96", patience: int = 3) -> tuple[tf.keras.Model, dict]:
    started = time.monotonic()
    model, embedding, info = build_classifier(workspace, alpha, pretrained, outputs, seed, name)
    if dataset == "survface_96":
        metric_fn = lambda: stop_metrics(model, stop_rows, stop_gray)
    else:
        metric_fn = lambda: olivetti_stop_metrics(model, stop_rows, stop_gray)
    if is_scratch:
        info["trainability"] = unfreeze_top_fraction(model, 1.0)
    curves = []
    best_weights = None
    best_score = -math.inf
    best_loss = math.inf
    best_stage = None
    best_epoch_global = 0
    epochs_run = 0
    sample_presentations = 0
    steps = 0
    global_epoch = 0
    for stage in stage_plan(config, pretrained=pretrained, is_scratch=is_scratch):
        if stage["stage"] == "frozen_head":
            model._d_base.trainable = False
            stage_layers = {"backbone_layers_total": len(model._d_base.layers),
                            "backbone_layers_trainable": [], "batch_normalization_frozen": True}
        else:
            stage_layers = unfreeze_top_fraction(model, stage["unfreeze_top_fraction"])
        compile_classifier(model, stage["lr"], outputs, outlier_exposure=outlier_exposure)
        stage_best = -math.inf
        stale_epochs = 0
        stage_best_weights = None
        stage_best_loss = math.inf
        for epoch in range(int(stage["epochs_max"])):
            global_epoch += 1
            epoch_seed = int(seed + global_epoch * 1009)
            stats = train_epoch(model, train_rows, train_gray, dataset, epoch_seed,
                                augment=(stage["augmentation"] == "mild_gray_v1"),
                                outputs=outputs, outlier_exposure=outlier_exposure,
                                batch_size=24 if dataset == "survface_96" else len(sorted({r['label_id'] for r in train_rows})))
            metrics = metric_fn()
            record = {"stage": stage["stage"], "stage_epoch": epoch + 1, "global_epoch": global_epoch,
                      "learning_rate": float(stage["lr"]), "augmentation": stage["augmentation"],
                      "trainability": stage_layers,
                      **stats, "stop": metrics}
            curves.append(record)
            epochs_run += 1
            sample_presentations += int(stats["sample_presentations"])
            steps += int(stats["batch_count"])
            score = float(metrics["known_macro_accuracy"])
            loss = float(metrics["known_mean_loss"])
            improved = score > stage_best + 1e-6 or (abs(score - stage_best) <= 1e-6 and loss < stage_best_loss)
            if improved:
                stage_best = score
                stage_best_loss = loss
                stage_best_weights = [np.array(value, copy=True) for value in model.get_weights()]
                stale_epochs = 0
            else:
                stale_epochs += 1
            global_improved = score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and loss < best_loss)
            if global_improved:
                best_score, best_loss = score, loss
                best_weights = [np.array(value, copy=True) for value in model.get_weights()]
                best_epoch_global = global_epoch
                best_stage = stage["stage"]
            if stale_epochs >= patience:
                break
        # Restore each stage's best state before advancing to fine-tuning.
        if stage_best_weights is not None:
            model.set_weights(stage_best_weights)
    if best_weights is None:
        raise RuntimeError("training produced no selectable checkpoint")
    model.set_weights(best_weights)
    elapsed = time.monotonic() - started
    history = {
        "schema": "job-d-training-run-1", "run_name": name, "root_seed": int(seed),
        "dataset": dataset, "outputs": int(outputs), "pretrained": bool(pretrained),
        "initialization": info["initialization"], "checkpoint_sha256": info["checkpoint_sha256"],
        "alpha": float(alpha), "training_config": config, "stage_plan": stage_plan(config, pretrained, is_scratch),
        "outlier_exposure": bool(outlier_exposure), "outlier_exposure_loss_weight": 0.25 if outlier_exposure else None,
        "trainability": getattr(model, "_d_info", {}).get("trainability"),
        "best_stage": best_stage, "selected_global_epoch": best_epoch_global,
        "selected_stopping_macro_accuracy": float(best_score), "selected_stopping_loss": float(best_loss),
        "epochs_run": epochs_run, "steps_run": steps, "sample_presentations": sample_presentations,
        "train_record_count": len(train_rows), "stop_record_count": len(stop_rows),
        "elapsed_seconds": float(elapsed), "curve": curves,
        "input_contract": {"dtype": "float32", "shape": [None, 96, 96, 1],
                           "normalization": "gray8 / 255.0", "channel_order": "single grayscale"},
        "backbone": {k: v for k, v in info.items() if k not in ("full_reference_model", "feature_model")},
    }
    return model, history


def _sample_training_gray(train_rows: list[dict], train_gray: np.ndarray, limit: int = 128) -> np.ndarray:
    if len(train_gray) <= limit:
        return train_gray
    # Deterministic, class-ordered representative data, never from a holdout role.
    indices = np.linspace(0, len(train_gray) - 1, limit, dtype=np.int64)
    return train_gray[indices]


def checkpoint_reference_audit(workspace: Path) -> dict:
    result = {"checkpoints": {}, "runtime": env_receipt(workspace)}
    for alpha in (0.1, 0.25):
        path = workspace / EI_CHECKPOINTS[alpha]
        actual = sha256_file(path)
        if actual != EI_HASHES[alpha]:
            raise ValueError("pretrained checkpoint SHA mismatch: " + str(path))
        full, features, info = keras_mobilenet(workspace, alpha, True, 14101 + int(alpha * 1000))
        x = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(1, 96, 96, 1)
        logits = full(x, training=False).numpy()
        features_out = features(x, training=False).numpy()
        first_conv = full.get_layer("conv1")
        result["checkpoints"][str(alpha)] = {
            "source_url": EI_URLS[alpha], "path": str(path), "sha256": actual,
            "bytes": path.stat().st_size, "loaded_into": "tf.keras.applications.MobileNet",
            "architecture": {"version": "MobileNetV1", "alpha": alpha,
                             "input_shape": [96, 96, 1], "include_top": True,
                             "top_classes": 1000, "feature_layer": "conv_pw_13_relu",
                             "feature_shape": [int(v) for v in features.output_shape[1:]],
                             "first_conv_kernel_shape": list(first_conv.get_weights()[0].shape),
                             "parameter_count": int(full.count_params())},
            "strict_topological_load": True, "head_removed_from_adapted_graph": [
                "global_average_pooling2d", "reshape", "dropout", "conv_preds", "reshape", "predictions"],
            "published_normalization": "gray8 uint8 / 255 to float32 [0,1]",
            "source_head_output_shape": list(logits.shape), "source_head_output_finite": bool(np.isfinite(logits).all()),
            "source_head_output_sum_min": float(np.min(np.sum(softmax_np(logits), axis=-1))),
            "source_head_output_sum_max": float(np.max(np.sum(softmax_np(logits), axis=-1))),
            "feature_output_finite": bool(np.isfinite(features_out).all()),
            "feature_output_shape": list(features_out.shape),
            "published_pretraining_corpus_and_license": "not declared in checkpoint metadata/source page; not inferred from 1000-class head",
        }
    return result


def run_ten_block_pilot(workspace: Path, paths: dict, roots: list[int]) -> dict:
    rows = dataset_role_rows(workspace, paths["primary"], "survface_96", ("train", "unknown_train"), "train")
    train_gray = cache_gray8(workspace, rows, "train")
    pilot_dir = paths["runs"] / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    root_records = []
    conversion_by_width = {}
    for index, seed in enumerate(roots):
        alpha = (0.1, 0.25)[index % 2]
        model, _, info = build_classifier(workspace, alpha, True, 6, seed, "d_pilot_%s_%d" % (alpha, seed))
        batch_idx = balanced_survface_indices(rows, 24, seed)[0]
        x = gray_float(train_gray[batch_idx])
        selected = [rows[i] for i in batch_idx]
        y = as_targets(selected, "survface_96", SIX)
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        gradients = tape.gradient(loss, model.trainable_weights)
        finite_grads = [g for g in gradients if g is not None]
        gradient_norm = float(tf.linalg.global_norm(finite_grads).numpy())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("pilot has missing/nonfinite/zero gradient at seed %d" % seed)
        compile_classifier(model, 0.001, 6)
        model.train_on_batch(x, y)
        model_dir = pilot_dir / ("seed-%d" % seed)
        model_dir.mkdir(parents=True, exist_ok=True)
        representative = np.linspace(0, len(train_gray) - 1, min(64, len(train_gray)), dtype=np.int64).tolist()
        converted_here = False
        if alpha not in conversion_by_width:
            float_path = model_dir / "classifier_float.tflite"
            int8_path = model_dir / "classifier_int8.tflite"
            if float_path.is_file():
                float_export = audit_tflite(float_path, require_int8=False)
                float_export.update({"path": str(float_path), "sha256": sha256_file(float_path), "bytes": float_path.stat().st_size})
            else:
                float_export = convert_float_tflite(model, float_path)
            if int8_path.is_file():
                int8_export = audit_tflite(int8_path, require_int8=True)
                int8_export.update({"path": str(int8_path), "sha256": sha256_file(int8_path), "bytes": int8_path.stat().st_size,
                                    "representative_sample_count": len(representative)})
            else:
                int8_export = convert_int8_tflite(model, train_gray, int8_path, representative)
            conversion_by_width[alpha] = {"float_export": float_export, "int8_export": int8_export,
                                          "float_path": str(float_path), "int8_path": str(int8_path),
                                          "conversion_root_seed": int(seed)}
            converted_here = True
        else:
            cached = conversion_by_width[alpha]
            float_export, int8_export = cached["float_export"], cached["int8_export"]
        cached = conversion_by_width[alpha]
        quant_logits = tflite_outputs(Path(cached["int8_path"]), train_gray[batch_idx[:2]])
        root_records.append({"root_seed": int(seed), "width": alpha,
                             "checkpoint_sha256": info["checkpoint_sha256"],
                             "batch_shape": list(x.shape), "input_min": float(x.min()),
                             "input_max": float(x.max()), "output_shape": list(logits.shape),
                             "loss_before_update": float(loss.numpy()), "gradient_global_norm": gradient_norm,
                             "gradient_count": len(finite_grads), "gradient_finite": True,
                             "float_export": float_export, "int8_export": int8_export,
                             "conversion_performed_for_this_block": converted_here,
                             "conversion_root_seed": cached["conversion_root_seed"],
                             "int8_logits_shape": list(quant_logits.shape),
                             "int8_logits_finite": bool(np.isfinite(quant_logits).all())})
        del model
        tf.keras.backend.clear_session()
    receipt = {"schema": "job-d-ten-block-pilot-1", "status": "complete_verified",
               "purpose": "diagnostic only; ten brief seeded blocks, not full ten-root model trainings",
               "root_count": len(root_records), "roots": root_records,
               "float_input_contract": "[1,96,96,1] float32 gray8/255",
               "unique_width_conversion_count": len(conversion_by_width),
               "unique_full_int8_export_count": sum(1 for row in conversion_by_width.values()
                                                    if row["int8_export"].get("all_int8_boundary")),
               "gradient_and_shape_check_count": len(root_records),
               "source_roles": ["train", "unknown_train"], "created_utc": utc_now()}
    write_json(pilot_dir / "PILOT_RESULT.json", receipt)
    return receipt


def load_role_arrays(workspace: Path, paths: dict, dataset: str) -> dict:
    if dataset == "survface_96":
        train_rows = dataset_role_rows(workspace, paths["primary"], dataset, ("train", "unknown_train"), "train")
        stop_rows = dataset_role_rows(workspace, paths["primary"], dataset, ("stopping",), "train")
        cal_known = dataset_role_rows(workspace, paths["primary"], dataset, ("threshold_calibration",), "threshold_fit")
        cal_unknown = dataset_role_rows(workspace, paths["primary"], dataset, ("unknown_calibration",), "threshold_fit")
        dev_known = dataset_role_rows(workspace, paths["primary"], dataset, ("development",), "select")
        dev_unknown = dataset_role_rows(workspace, paths["primary"], dataset, ("unknown_development",), "select")
        data = {"train_rows": train_rows, "train_gray": cache_gray8(workspace, train_rows, "train"),
                "stop_rows": stop_rows, "stop_gray": cache_gray8(workspace, stop_rows, "train"),
                "cal_known": cal_known, "cal_known_gray": cache_gray8(workspace, cal_known, "threshold_fit"),
                "cal_unknown": cal_unknown, "cal_unknown_gray": cache_gray8(workspace, cal_unknown, "threshold_fit"),
                "dev_known": dev_known, "dev_known_gray": cache_gray8(workspace, dev_known, "select"),
                "dev_unknown": dev_unknown, "dev_unknown_gray": cache_gray8(workspace, dev_unknown, "select")}
        return data
    if dataset == "olivetti_96":
        train_rows = dataset_role_rows(workspace, paths["olivetti"], dataset, ("train",), "train")
        dev_rows = dataset_role_rows(workspace, paths["olivetti"], dataset, ("development",), "select")
        return {"train_rows": train_rows, "train_gray": cache_gray8(workspace, train_rows, "train"),
                "stop_rows": dev_rows, "stop_gray": cache_gray8(workspace, dev_rows, "select"),
                "dev_rows": dev_rows, "dev_gray": cache_gray8(workspace, dev_rows, "select")}
    raise ValueError("unknown dataset: " + dataset)


def rows_fingerprint(rows: list[dict]) -> str:
    return sha256_json([{k: row.get(k) for k in ("record_id", "role", "label_id", "identity_id", "sha256", "relative_path")}
                        for row in sorted(rows, key=lambda value: str(value.get("record_id")))])


def hpo_search(workspace: Path, paths: dict, matrix: dict, data: dict) -> dict:
    search_root = paths["runs"] / "development_search"
    search_root.mkdir(parents=True, exist_ok=True)
    cal_rows = data["cal_known"] + data["cal_unknown"]
    cal_gray = np.concatenate([data["cal_known_gray"], data["cal_unknown_gray"]], axis=0)
    dev_rows = data["dev_known"] + data["dev_unknown"]
    dev_gray = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    specs = []
    search = matrix["epochs_and_stopping"]
    for alpha in (0.1, 0.25):
        for config in search["transfer_initialization_search_candidates"]:
            specs.append({"alpha": alpha, "family": "transfer", "config": config})
        for config in search["scratch_search_candidates"]:
            specs.append({"alpha": alpha, "family": "scratch", "config": config})
    search_results = []
    search_seed = int(search["development_search_seed"])
    for index, item in enumerate(specs):
        alpha, family, config = item["alpha"], item["family"], dict(item["config"])
        run_name = "search_a%s_%s_%s" % (str(alpha).replace(".", ""), family, config["id"])
        directory = search_root / run_name
        result_file = directory / "SEARCH_RESULT.json"
        if result_file.is_file():
            saved = json.loads(result_file.read_text(encoding="utf-8"))
            if saved.get("status") == "complete_verified":
                search_results.append(saved)
                continue
        seed = search_seed + index * 31
        model, history = train_model(
            workspace, alpha, family == "transfer", 6, seed, run_name,
            data["train_rows"], data["train_gray"], data["stop_rows"], data["stop_gray"],
            config, is_scratch=(family == "scratch"), dataset="survface_96")
        weights_path = directory / "model.weights.h5"
        atomic_save_weights(model, weights_path)
        cal_logits = evaluate_logits(model, cal_gray)
        calibration = fit_open_threshold(cal_logits[:len(data["cal_known"])], data["cal_known"],
                                         cal_logits[len(data["cal_known"]):], data["cal_unknown"], 6)
        dev_logits = evaluate_logits(model, dev_gray)
        dev_metrics = known_macro_and_open_metrics(dev_logits, data["dev_known"], data["dev_unknown"],
                                                   calibration["threshold"], 6)
        result = {"schema": "job-d-development-search-result-1", "status": "complete_verified",
                  "id": config["id"], "family": family, "alpha": alpha,
                  "seed": seed, "config": config, "training": history,
                  "training_manifest_fingerprint": rows_fingerprint(data["train_rows"]),
                  "calibration": calibration, "development": dev_metrics,
                  "development_macro_utility": dev_metrics["development_macro_utility"],
                  "weights_path": str(weights_path), "weights_sha256": sha256_file(weights_path),
                  "created_utc": utc_now()}
        directory.mkdir(parents=True, exist_ok=True)
        write_json(result_file, result)
        search_results.append(result)
        del model
        tf.keras.backend.clear_session()
    selected = {}
    for alpha in (0.1, 0.25):
        for family in ("transfer", "scratch"):
            candidates = [row for row in search_results if row["alpha"] == alpha and row["family"] == family]
            if not candidates:
                raise RuntimeError("no HPO candidates for %s/%s" % (alpha, family))
            def lr_of(row):
                return float(row["config"].get("fine_tune_lr") if family == "transfer"
                             else row["config"].get("learning_rate"))
            best = max(candidates, key=lambda row: (float(row["development_macro_utility"]),
                                                     float(row["development"]["known_identity_macro_identification"]),
                                                     -lr_of(row)))
            selected["a%s_%s" % (str(alpha).replace(".", ""), family)] = {
                "selected_candidate_id": best["id"], "selected_config": best["config"],
                "development_macro_utility": best["development_macro_utility"],
                "development_known_macro_identification": best["development"]["known_identity_macro_identification"],
                "calibration_threshold": best["calibration"]["threshold"],
                "tie_break": "higher development utility, then known macro identification, then lower learning rate",
            }
    result = {"schema": "job-d-development-search-1", "status": "complete_verified",
              "search_seed": search_seed, "candidate_count": len(search_results),
              "selection_roles": {"train": ["train", "unknown_train"], "stopping": ["stopping"],
                                  "threshold_fit": ["threshold_calibration", "unknown_calibration"],
                                  "selection": ["development", "unknown_development"]},
              "final_role_opened": False, "candidates": search_results,
              "selected": selected, "created_utc": utc_now()}
    write_json(search_root / "DEVELOPMENT_SEARCH.json", result)
    return result


def selected_config_for(search_result: dict, alpha: float, family: str) -> dict:
    key = "a%s_%s" % (str(alpha).replace(".", ""), family)
    return dict(search_result["selected"][key]["selected_config"])


def _selection_manifest_path(paths: dict, policy: str, seed: int) -> Path:
    return paths["data"] / ("derived/D_selection/%s_seed-%d.jsonl" % (policy, seed))


def _arm_train_rows(workspace: Path, paths: dict, arm_id: str, seed: int, base_rows: list[dict]) -> list[dict]:
    if arm_id == "sv_mnv1_a025_quality_select_6":
        manifest = _selection_manifest_path(paths, "quality", seed)
        rows = read_jsonl(manifest)
        assert_role_access(rows, "train")
        for row in rows:
            safe_path(row, workspace)
        return rows
    if arm_id == "sv_mnv1_a025_uniform_select_6":
        manifest = _selection_manifest_path(paths, "uniform", seed)
        rows = read_jsonl(manifest)
        assert_role_access(rows, "train")
        for row in rows:
            safe_path(row, workspace)
        return rows
    return base_rows


def _arm_config(arm_id: str, alpha: float, search_result: dict, matrix: dict) -> tuple[dict, bool, int, bool]:
    if "scratch" in arm_id:
        cfg = selected_config_for(search_result, alpha, "scratch")
        # Equal maximum epoch and example budgets within each width.
        transfer = selected_config_for(search_result, alpha, "transfer")
        cfg["epochs_max"] = int(transfer["head_epochs_max"] + transfer["fine_tune_epochs_max"])
        return cfg, False, 6, False
    cfg = selected_config_for(search_result, alpha, "transfer")
    if arm_id.endswith("_5_oe"):
        return cfg, True, 5, True
    return cfg, True, 6, False


def _load_matrix(workspace: Path, paths: dict) -> dict:
    matrix = json.loads(paths["matrix"].read_text(encoding="utf-8"))
    if not matrix.get("frozen_before_training") or matrix.get("matrix_id") != "D-five-person-transfer-open-set-gallery-v1":
        raise ValueError("D matrix is missing or not frozen")
    if sha256_file(paths["matrix"]) is None:
        raise ValueError("D matrix SHA unavailable")
    return matrix


def tflite_outputs(path: Path, gray_images: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    idetail = interpreter.get_input_details()[0]
    odetail = interpreter.get_output_details()[0]
    results = []
    for index in range(len(gray_images)):
        x = gray_float(gray_images[index:index + 1])
        if idetail["dtype"] == np.int8:
            scale, zero = idetail["quantization"]
            if not scale:
                raise ValueError("missing TFLite input quantization metadata")
            x = np.clip(np.rint(x / scale + zero), -128, 127).astype(np.int8)
        else:
            x = x.astype(idetail["dtype"])
        interpreter.set_tensor(idetail["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(odetail["index"])
        if odetail["dtype"] == np.int8:
            scale, zero = odetail["quantization"]
            if not scale:
                raise ValueError("missing TFLite output quantization metadata")
            y = (y.astype(np.float32) - float(zero)) * float(scale)
        results.append(np.asarray(y, dtype=np.float32))
    return np.concatenate(results, axis=0) if results else np.zeros((0, int(odetail["shape"][-1])), np.float32)


def _save_rows_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def train_one_classifier(workspace: Path, paths: dict, arm: dict, root_seed: int,
                         search_result: dict, base_data: dict) -> dict:
    arm_id = str(arm["id"])
    alpha = float(arm["width"])
    config, pretrained, outputs, oe = _arm_config(arm_id, alpha, search_result,
                                                   json.loads(paths["matrix"].read_text(encoding="utf-8")))
    root_dir = paths["runs"] / "survface" / arm_id / ("seed-%d" % root_seed)
    result_file = root_dir / "ROOT_RESULT.json"
    if result_file.is_file():
        saved = json.loads(result_file.read_text(encoding="utf-8"))
        if saved.get("status") == "complete_verified" and Path(saved["weights_path"]).is_file():
            return saved
    train_rows = _arm_train_rows(workspace, paths, arm_id, root_seed, base_data["train_rows"])
    train_gray = cache_gray8(workspace, train_rows, "train")
    model, history = train_model(
        workspace, alpha, pretrained, outputs, root_seed, arm_id + "_seed" + str(root_seed),
        train_rows, train_gray, base_data["stop_rows"], base_data["stop_gray"], config,
        is_scratch=(not pretrained), outlier_exposure=oe, dataset="survface_96")
    root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_dir / "training_rows.jsonl"
    manifest_hash = _save_rows_jsonl(manifest_path, train_rows)
    curve_path = root_dir / "training_curve.jsonl"
    with curve_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in history["curve"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    weights_path = root_dir / "classifier.weights.h5"
    atomic_save_weights(model, weights_path)
    float_classifier = convert_float_tflite(model, root_dir / "classifier_float.tflite")
    rep_indices = np.linspace(0, len(train_gray) - 1, min(128, len(train_gray)), dtype=np.int64).tolist()
    int8_classifier = convert_int8_tflite(model, train_gray, root_dir / "classifier_int8.tflite", rep_indices)
    if pretrained and arm_id in ("sv_mnv1_a010_transfer_6", "sv_mnv1_a025_transfer_6"):
        float_embedding = convert_float_tflite(model._d_embedding_model, root_dir / "embedding_float.tflite")
        int8_embedding = convert_int8_tflite(model._d_embedding_model, train_gray,
                                             root_dir / "embedding_int8.tflite", rep_indices)
    else:
        float_embedding = int8_embedding = None
    cal_rows = base_data["cal_known"] + base_data["cal_unknown"]
    cal_gray = np.concatenate([base_data["cal_known_gray"], base_data["cal_unknown_gray"]], axis=0)
    ncal = len(base_data["cal_known"])
    float_cal_logits = tflite_outputs(root_dir / "classifier_float.tflite", cal_gray)
    int8_cal_logits = tflite_outputs(root_dir / "classifier_int8.tflite", cal_gray)
    float_calibration = fit_open_threshold(float_cal_logits[:ncal], base_data["cal_known"],
                                           float_cal_logits[ncal:], base_data["cal_unknown"], outputs)
    int8_calibration = fit_open_threshold(int8_cal_logits[:ncal], base_data["cal_known"],
                                          int8_cal_logits[ncal:], base_data["cal_unknown"], outputs)
    dev_rows = base_data["dev_known"] + base_data["dev_unknown"]
    dev_gray = np.concatenate([base_data["dev_known_gray"], base_data["dev_unknown_gray"]], axis=0)
    host_dev_logits = evaluate_logits(model, dev_gray)
    float_dev_logits = tflite_outputs(root_dir / "classifier_float.tflite", dev_gray)
    int8_dev_logits = tflite_outputs(root_dir / "classifier_int8.tflite", dev_gray)
    float_dev = known_macro_and_open_metrics(float_dev_logits, base_data["dev_known"], base_data["dev_unknown"],
                                             float_calibration["threshold"], outputs)
    int8_dev = known_macro_and_open_metrics(int8_dev_logits, base_data["dev_known"], base_data["dev_unknown"],
                                            int8_calibration["threshold"], outputs)
    host_dev_pred, _, host_dev_other = prediction_components(host_dev_logits[:len(base_data["dev_known"])], outputs)
    float_dev_pred, _, float_dev_other = prediction_components(float_dev_logits[:len(base_data["dev_known"])], outputs)
    int8_dev_pred, _, int8_dev_other = prediction_components(int8_dev_logits[:len(base_data["dev_known"])], outputs)
    root_result = {
        "schema": "job-d-root-result-1", "status": "complete_verified", "arm_id": arm_id,
        "root_seed": int(root_seed), "dataset": "survface_96", "branch": "public K01-K05",
        "width": alpha, "initialization": "Edge Impulse grayscale checkpoint" if pretrained else "random Keras MobileNetV1",
        "pretrained": pretrained, "checkpoint_sha256": EI_HASHES[alpha] if pretrained else None,
        "checkpoint_source_url": EI_URLS[alpha] if pretrained else None,
        "head": "K01-K05+OTHER logits" if outputs == 6 else "K01-K05 logits with uniform outlier exposure",
        "outlier_exposure": oe, "config": config, "config_sha256": sha256_json(config),
        "training": history, "training_rows": len(train_rows), "training_rows_content_sha256": rows_fingerprint(train_rows),
        "training_rows_file": str(manifest_path), "training_rows_file_sha256": manifest_hash,
        "train_manifest_source_sha256": sha256_file(paths["primary"]),
        "curve_file": str(curve_path), "curve_file_sha256": sha256_file(curve_path),
        "weights_path": str(weights_path), "weights_sha256": sha256_file(weights_path),
        "float_classifier_export": float_classifier, "int8_classifier_export": int8_classifier,
        "float_embedding_export": float_embedding, "int8_embedding_export": int8_embedding,
        "label_order": SIX if outputs == 6 else KNOWN,
        "output_contract": "classifier_logits-%d" % outputs,
        "float_calibration": float_calibration, "int8_calibration": int8_calibration,
        "float_development": float_dev, "int8_development": int8_dev,
        "host_model_development": known_macro_and_open_metrics(host_dev_logits, base_data["dev_known"], base_data["dev_unknown"],
                                                               float_calibration["threshold"], outputs),
        "host_vs_float_tflite_known_label_disagreement_count": int(np.sum(host_dev_pred != float_dev_pred)),
        "host_vs_float_tflite_other_disagreement_count": int(np.sum(host_dev_other != float_dev_other)),
        "development_float_int8_known_label_disagreement_count": int(np.sum(float_dev_pred != int8_dev_pred)),
        "development_float_int8_other_disagreement_count": int(np.sum(float_dev_other != int8_dev_other)),
        "development_comparison_role_counts": {"known": len(base_data["dev_known"]),
                                                "unknown": len(base_data["dev_unknown"]),
                                                "unknown_identity_count": len({r["identity_id"] for r in base_data["dev_unknown"]})},
        "representative_sample_indices": rep_indices,
        "representative_rows_content_sha256": sha256_json([train_rows[i]["record_id"] for i in rep_indices]),
        "environment": env_receipt(workspace), "created_utc": utc_now(),
    }
    write_json(root_dir / "ROOT_RESULT.json", root_result)
    write_json(root_dir / "float_calibration.json", float_calibration)
    write_json(root_dir / "int8_calibration.json", int8_calibration)
    write_json(root_dir / "config.json", {"arm": arm, "root_seed": root_seed, "config": config,
                                          "label_order": root_result["label_order"],
                                          "training_rows_content_sha256": root_result["training_rows_content_sha256"],
                                          "normalization": "gray8 / 255", "pretrained": pretrained,
                                          "checkpoint_sha256": root_result["checkpoint_sha256"]})
    del model
    tf.keras.backend.clear_session()
    return root_result


def _olivetti_prediction_metrics(logits: np.ndarray, rows: list[dict]) -> dict:
    classes = sorted({str(row["label_id"]) for row in rows})
    pred = np.argmax(logits, axis=1)
    true = np.asarray([classes.index(str(row["label_id"])) for row in rows], dtype=np.int64)
    per_id = {}
    for index, label in enumerate(classes):
        mask = true == index
        per_id[label] = {"n": int(np.sum(mask)), "accuracy": float(np.mean(pred[mask] == true[mask]))}
    return {"n": len(rows), "identity_count": len(classes),
            "closed_set_accuracy": float(np.mean(pred == true)),
            "macro_identity_accuracy": float(np.mean([r["accuracy"] for r in per_id.values()])),
            "per_person": per_id}


def train_one_olivetti(workspace: Path, paths: dict, arm: dict, root_seed: int,
                       search_result: dict, base_data: dict) -> dict:
    arm_id = str(arm["id"])
    alpha = float(arm["width"])
    transfer = "scratch" not in arm_id
    config = selected_config_for(search_result, alpha, "transfer" if transfer else "scratch")
    if not transfer:
        # Match the primary selected transfer epoch cap within the same width.
        transfer_cfg = selected_config_for(search_result, alpha, "transfer")
        config = dict(config)
        config["epochs_max"] = int(transfer_cfg["head_epochs_max"] + transfer_cfg["fine_tune_epochs_max"])
    root_dir = paths["runs"] / "olivetti" / arm_id / ("seed-%d" % root_seed)
    result_file = root_dir / "ROOT_RESULT.json"
    if result_file.is_file():
        saved = json.loads(result_file.read_text(encoding="utf-8"))
        if saved.get("status") == "complete_verified" and Path(saved["weights_path"]).is_file():
            return saved
    pretrained = transfer
    model, history = train_model(workspace, alpha, pretrained, 40, root_seed,
                                 arm_id + "_seed" + str(root_seed), base_data["train_rows"],
                                 base_data["train_gray"], base_data["stop_rows"], base_data["stop_gray"],
                                 config, is_scratch=(not pretrained), dataset="olivetti_96")
    root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_dir / "training_rows.jsonl"
    manifest_hash = _save_rows_jsonl(manifest_path, base_data["train_rows"])
    curve_path = root_dir / "training_curve.jsonl"
    with curve_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in history["curve"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    weights_path = root_dir / "classifier.weights.h5"
    atomic_save_weights(model, weights_path)
    float_classifier = convert_float_tflite(model, root_dir / "classifier_float.tflite")
    rep_indices = np.linspace(0, len(base_data["train_gray"]) - 1,
                              min(128, len(base_data["train_gray"])), dtype=np.int64).tolist()
    int8_classifier = convert_int8_tflite(model, base_data["train_gray"],
                                          root_dir / "classifier_int8.tflite", rep_indices)
    float_embedding = int8_embedding = None
    host_logits = evaluate_logits(model, base_data["dev_gray"])
    float_logits = tflite_outputs(root_dir / "classifier_float.tflite", base_data["dev_gray"])
    int8_logits = tflite_outputs(root_dir / "classifier_int8.tflite", base_data["dev_gray"])
    float_dev = _olivetti_prediction_metrics(float_logits, base_data["dev_rows"])
    int8_dev = _olivetti_prediction_metrics(int8_logits, base_data["dev_rows"])
    root_result = {
        "schema": "job-d-root-result-1", "status": "complete_verified", "arm_id": arm_id,
        "root_seed": int(root_seed), "dataset": "olivetti_96", "branch": "separate 40-way closed-set sanity",
        "width": alpha, "initialization": "Edge Impulse grayscale checkpoint" if pretrained else "random Keras MobileNetV1",
        "pretrained": pretrained, "checkpoint_sha256": EI_HASHES[alpha] if pretrained else None,
        "checkpoint_source_url": EI_URLS[alpha] if pretrained else None,
        "head": "40-way closed-set logits", "config": config, "config_sha256": sha256_json(config),
        "training": history, "training_rows": len(base_data["train_rows"]),
        "training_rows_content_sha256": rows_fingerprint(base_data["train_rows"]),
        "training_rows_file": str(manifest_path), "training_rows_file_sha256": manifest_hash,
        "train_manifest_source_sha256": sha256_file(paths["olivetti"]),
        "curve_file": str(curve_path), "curve_file_sha256": sha256_file(curve_path),
        "weights_path": str(weights_path), "weights_sha256": sha256_file(weights_path),
        "float_classifier_export": float_classifier, "int8_classifier_export": int8_classifier,
        "float_embedding_export": float_embedding, "int8_embedding_export": int8_embedding,
        "label_order": sorted({str(row["label_id"]) for row in base_data["train_rows"]}),
        "output_contract": "classifier_logits-40", "float_development": float_dev,
        "int8_development": int8_dev,
        "host_model_development": _olivetti_prediction_metrics(host_logits, base_data["dev_rows"]),
        "host_vs_float_tflite_label_disagreement_count": int(np.sum(np.argmax(host_logits, axis=1) != np.argmax(float_logits, axis=1))),
        "environment": env_receipt(workspace), "created_utc": utc_now(),
        "representative_sample_indices": rep_indices,
        "representative_rows_content_sha256": sha256_json([base_data["train_rows"][i]["record_id"] for i in rep_indices]),
        "open_set_claim": False,
    }
    write_json(root_dir / "ROOT_RESULT.json", root_result)
    del model
    tf.keras.backend.clear_session()
    return root_result


def run_training_campaign(workspace: Path, paths: dict) -> dict:
    matrix = _load_matrix(workspace, paths)
    roots = list(matrix["training_roots"])
    if len(roots) != 10 or len(set(roots)) != 10:
        raise ValueError("frozen campaign requires ten distinct roots")
    progress_path = paths["runs"] / "CAMPAIGN_PROGRESS.json"
    paths["runs"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    with resource_lease(paths["study"], "heavy_training", owner="D", timeout_s=0):
        pilot_path = paths["runs"] / "pilot/PILOT_RESULT.json"
        if pilot_path.is_file():
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            if pilot.get("status") != "complete_verified" or pilot.get("root_count") != 10:
                pilot = run_ten_block_pilot(workspace, paths, matrix["pilot_roots"])
        else:
            pilot = run_ten_block_pilot(workspace, paths, matrix["pilot_roots"])
        progress = {"status": "pilot_complete", "pilot_roots": 10, "started_utc": utc_now()}
        write_json(progress_path, progress)
        surv_data = load_role_arrays(workspace, paths, "survface_96")
        search = hpo_search(workspace, paths, matrix, surv_data)
        progress.update({"status": "development_search_complete", "search_candidate_count": search["candidate_count"],
                         "completed_survface_roots": 0, "completed_olivetti_roots": 0,
                         "updated_utc": utc_now()})
        write_json(progress_path, progress)
        for arm in matrix["survface_arms"]:
            arm_results = []
            for seed in roots:
                result = train_one_classifier(workspace, paths, arm, seed, search, surv_data)
                arm_results.append(result)
                progress.update({"status": "survface_training", "last_arm": arm["id"],
                                 "last_root_seed": seed,
                                 "completed_survface_roots": len(list(paths["runs"].glob("survface/*/seed-*/ROOT_RESULT.json"))),
                                 "updated_utc": utc_now()})
                write_json(progress_path, progress)
            arm_receipt = {"schema": "job-d-arm-summary-1", "status": "complete_verified",
                           "dataset": "survface_96", "arm_id": arm["id"], "roots": arm_results,
                           "development_mean_utility": float(np.mean([r["float_development"]["development_macro_utility"] for r in arm_results])),
                           "development_sd_utility": float(np.std([r["float_development"]["development_macro_utility"] for r in arm_results], ddof=1)),
                           "created_utc": utc_now()}
            write_json(paths["reports"] / (str(arm["id"]) + ".json"), arm_receipt)
        ol_data = load_role_arrays(workspace, paths, "olivetti_96")
        for arm in matrix["olivetti_arms"]:
            arm_results = []
            for seed in roots:
                result = train_one_olivetti(workspace, paths, arm, seed, search, ol_data)
                arm_results.append(result)
                progress.update({"status": "olivetti_training", "last_arm": arm["id"],
                                 "last_root_seed": seed,
                                 "completed_olivetti_roots": len(list(paths["runs"].glob("olivetti/*/seed-*/ROOT_RESULT.json"))),
                                 "updated_utc": utc_now()})
                write_json(progress_path, progress)
            write_json(paths["reports"] / (str(arm["id"]) + ".json"), {
                "schema": "job-d-arm-summary-1", "status": "complete_verified", "dataset": "olivetti_96",
                "arm_id": arm["id"], "roots": arm_results,
                "development_mean_macro_identity_accuracy": float(np.mean([r["float_development"]["macro_identity_accuracy"] for r in arm_results])),
                "development_sd_macro_identity_accuracy": float(np.std([r["float_development"]["macro_identity_accuracy"] for r in arm_results], ddof=1)),
                "open_set_claim": False, "created_utc": utc_now()})
        campaign = {"schema": "job-d-training-campaign-1", "status": "complete_verified",
                    "training_roots": roots, "survface_arms": len(matrix["survface_arms"]),
                    "olivetti_arms": len(matrix["olivetti_arms"]),
                    "full_classifier_training_count": 10 * (len(matrix["survface_arms"]) + len(matrix["olivetti_arms"])),
                    "pilot": pilot, "development_search": search,
                    "run_artifacts_root": str(paths["runs"]), "created_utc": utc_now()}
        write_json(paths["runs"] / "TRAINING_CAMPAIGN.json", campaign)
        progress.update({"status": "training_complete", "campaign_path": str(paths["runs"] / "TRAINING_CAMPAIGN.json"),
                         "updated_utc": utc_now()})
        write_json(progress_path, progress)
    return campaign


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _gallery_ranked_rows(rows: list[dict], seed: int) -> list[dict]:
    by_camera = defaultdict(list)
    for row in rows:
        by_camera[str(row.get("camera_source_group") or "unknown")].append(row)
    for values in by_camera.values():
        values.sort(key=lambda row: (float(row.get("quality_score") or 0.0), str(row["record_id"])), reverse=True)
    cameras = sorted(by_camera)
    ranked = []
    # Select one high-quality image per source camera first, then the next-highest group.
    while True:
        progressed = False
        for camera in cameras:
            position = sum(1 for row in ranked if str(row.get("camera_source_group") or "unknown") == camera)
            if position < len(by_camera[camera]):
                ranked.append(by_camera[camera][position])
                progressed = True
        if not progressed:
            break
    return ranked


def choose_gallery_rows(rows: list[dict], count: int, policy: str, seed: int) -> list[dict]:
    ranked = _gallery_ranked_rows(rows, seed)
    if count >= len(ranked):
        return ranked
    quality_selected = ranked[:count]
    if policy == "quality":
        return quality_selected
    if policy != "uniform":
        raise ValueError("gallery policy must be quality or uniform")
    target = Counter(str(row.get("camera_source_group") or "unknown") for row in quality_selected)
    rng = np.random.default_rng(seed ^ 0x71A1)
    result = []
    for camera, camera_count in sorted(target.items()):
        candidates = [row for row in rows if str(row.get("camera_source_group") or "unknown") == camera]
        if len(candidates) < camera_count:
            raise ValueError("gallery camera stratum is too small")
        picked = rng.choice(len(candidates), size=camera_count, replace=False)
        result.extend(candidates[int(index)] for index in picked)
    return result


def build_gallery_prototypes(template_embeddings: dict[str, np.ndarray], rows: list[dict],
                             template_count: int, policy: str, seed: int) -> tuple[np.ndarray, list[str], dict]:
    classes = KNOWN
    prototypes, selection = [], {}
    for index, label in enumerate(classes):
        person_rows = [row for row in rows if str(row.get("label_id")) == label]
        selected = choose_gallery_rows(person_rows, template_count, policy, seed + index * 113)
        selected_ids = [str(row["record_id"]) for row in selected]
        values = np.stack([template_embeddings[rid] for rid in selected_ids])
        prototype = _normalize_vectors(np.mean(values, axis=0, keepdims=True))[0]
        prototypes.append(prototype)
        selection[label] = {"template_count": len(selected_ids), "record_ids": selected_ids,
                            "camera_strata": dict(Counter(str(row.get("camera_source_group") or "unknown")
                                                           for row in selected))}
    return np.stack(prototypes), classes, selection


def gallery_scores(query_embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return _normalize_vectors(query_embeddings) @ prototypes.T


def fit_gallery_threshold(known_score_matrix: np.ndarray, known_rows: list[dict],
                          unknown_score_matrix: np.ndarray, unknown_rows: list[dict],
                          target_fpir: float = 0.05) -> dict:
    kp = np.argmax(known_score_matrix, axis=1)
    ks = np.max(known_score_matrix, axis=1)
    up = np.argmax(unknown_score_matrix, axis=1)
    us = np.max(unknown_score_matrix, axis=1)
    true = np.asarray([row_label_index(row) for row in known_rows], dtype=np.int64)
    thresholds = sorted(set([float(np.min(np.concatenate([ks, us])))] +
                             [float(v) for v in np.concatenate([ks, us])]))
    thresholds.append(float(np.nextafter(max(thresholds), math.inf)))
    candidates = []
    for threshold in thresholds:
        accepted_known = ks >= threshold
        accepted_unknown = us >= threshold
        correct = accepted_known & (kp == true)
        per_class = [float(np.mean(correct[[i for i, row in enumerate(known_rows)
                                           if row_label_index(row) == ci]])) for ci in range(5)]
        fpir = float(np.mean(accepted_unknown)) if len(us) else 0.0
        candidates.append({"threshold": threshold,
                           "calibration_macro_correct_acceptance": float(np.mean(per_class)),
                           "calibration_known_coverage": float(np.mean(accepted_known)),
                           "calibration_unknown_fpir": fpir, "constraint_met": fpir <= target_fpir})
    eligible = [row for row in candidates if row["constraint_met"]]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["calibration_macro_correct_acceptance"],
                                                row["calibration_known_coverage"],
                                                -row["calibration_unknown_fpir"], row["threshold"]))
    else:
        chosen = min(candidates, key=lambda row: (row["calibration_unknown_fpir"],
                                                   -row["calibration_macro_correct_acceptance"],
                                                   -row["threshold"]))
    return {"score": "maximum cosine similarity to identity-mean prototype",
            "threshold": float(chosen["threshold"]), "target_unknown_fpir": target_fpir,
            "constraint_met": bool(chosen["constraint_met"]), "calibration": chosen,
            "known_n": len(known_rows), "unknown_n": len(unknown_rows),
            "unknown_identity_count": len({row.get("identity_id") for row in unknown_rows}),
            "threshold_candidates": len(candidates)}


def gallery_metrics(known_matrix: np.ndarray, known_rows: list[dict], unknown_matrix: np.ndarray,
                    unknown_rows: list[dict], threshold: float) -> dict:
    kp, ks = np.argmax(known_matrix, axis=1), np.max(known_matrix, axis=1)
    up, us = np.argmax(unknown_matrix, axis=1), np.max(unknown_matrix, axis=1)
    true = np.asarray([row_label_index(row) for row in known_rows], dtype=np.int64)
    known_accept, unknown_accept = ks >= threshold, us >= threshold
    per_person = {}
    for index, label in enumerate(KNOWN):
        mask = true == index
        per_person[label] = {"n": int(np.sum(mask)),
                             "identification_rate": float(np.mean(kp[mask] == true[mask])),
                             "correct_known_acceptance_rate": float(np.mean(known_accept[mask] & (kp[mask] == true[mask]))),
                             "wrong_known_identity_rate": float(np.mean(known_accept[mask] & (kp[mask] != true[mask]))),
                             "rejection_rate": float(np.mean(~known_accept[mask])),
                             "coverage": float(np.mean(known_accept[mask]))}
    unknown_by_identity = defaultdict(list)
    for row, accepted in zip(unknown_rows, unknown_accept):
        unknown_by_identity[str(row["identity_id"])].append(bool(accepted))
    per_person_recall = [value["identification_rate"] for value in per_person.values()]
    correct_accept = [value["correct_known_acceptance_rate"] for value in per_person.values()]
    unknown_rejection = float(np.mean(~unknown_accept)) if len(unknown_accept) else None
    return {"threshold": float(threshold), "known_n": len(known_rows), "unknown_n": len(unknown_rows),
            "unknown_identity_count": len(unknown_by_identity),
            "known_macro_identification": float(np.mean(per_person_recall)),
            "known_macro_correct_acceptance": float(np.mean(correct_accept)),
            "known_macro_wrong_known_identity": float(np.mean([v["wrong_known_identity_rate"] for v in per_person.values()])),
            "known_macro_rejection": float(np.mean([v["rejection_rate"] for v in per_person.values()])),
            "known_macro_coverage": float(np.mean([v["coverage"] for v in per_person.values()])),
            "unknown_false_identification_rate": float(np.mean(unknown_accept)) if len(unknown_accept) else None,
            "unknown_identity_macro_fpir": float(np.mean([np.mean(v) for v in unknown_by_identity.values()])) if unknown_by_identity else None,
            "unknown_rejection_rate": unknown_rejection,
            "development_macro_utility": float((np.mean(correct_accept) + unknown_rejection) / 2.0) if unknown_rejection is not None else None,
            "per_person": per_person}


def run_gallery_campaign(workspace: Path, paths: dict, matrix: dict, base_data: dict) -> list[dict]:
    gallery_root = paths["runs"] / "gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    expected_paths = [gallery_root / ("a%s" % str(alpha).replace(".", "")) /
                      ("seed-%d" % seed) / ("%s_%s.json" % (backend, policy))
                      for alpha in (0.1, 0.25) for seed in matrix["training_roots"]
                      for backend in ("float", "int8") for policy in ("quality", "uniform")]
    if all(path.is_file() for path in expected_paths):
        return [json.loads(path.read_text(encoding="utf-8")) | {"result_path": str(path)}
                for path in expected_paths]
    train_known = [row for row in base_data["train_rows"] if row.get("identity_partition") != "unknown"
                   and row.get("label_id") in KNOWN]
    cal_known, cal_unknown = base_data["cal_known"], base_data["cal_unknown"]
    dev_known, dev_unknown = base_data["dev_known"], base_data["dev_unknown"]
    cal_gray = {"known": base_data["cal_known_gray"], "unknown": base_data["cal_unknown_gray"]}
    dev_gray = {"known": base_data["dev_known_gray"], "unknown": base_data["dev_unknown_gray"]}
    by_record_rows = train_known + cal_known + cal_unknown + dev_known + dev_unknown
    results = []
    for alpha in (0.1, 0.25):
        arm_id = "sv_mnv1_a%s_transfer_6" % ("010" if alpha == 0.1 else "025")
        for seed in matrix["training_roots"]:
            model_dir = paths["runs"] / "survface" / arm_id / ("seed-%d" % seed)
            weights_path = model_dir / "classifier.weights.h5"
            model, _, info = load_weights_exact(workspace, alpha, True, 6, int(seed),
                                                 "gallery_reloaded_%s_%d" % (alpha, seed), weights_path)
            # Materialize features once per backend, then reuse them for gallery counts/policies.
            all_rows = by_record_rows
            all_images = np.concatenate([base_data["train_gray"][[i for i, row in enumerate(base_data["train_rows"])
                                                                      if row.get("identity_partition") != "unknown" and row.get("label_id") in KNOWN]],
                                         cal_gray["known"], cal_gray["unknown"], dev_gray["known"], dev_gray["unknown"]], axis=0)
            float_features = _normalize_vectors(tflite_outputs(model_dir / "embedding_float.tflite", all_images))
            int8_features = _normalize_vectors(tflite_outputs(model_dir / "embedding_int8.tflite", all_images))
            feature_maps = []
            for feature_array in (float_features, int8_features):
                feature_maps.append({str(row["record_id"]): feature_array[i] for i, row in enumerate(all_rows)})
            offsets = {"train": 0, "cal_known": len(train_known),
                       "cal_unknown": len(train_known) + len(cal_known),
                       "dev_known": len(train_known) + len(cal_known) + len(cal_unknown),
                       "dev_unknown": len(train_known) + len(cal_known) + len(cal_unknown) + len(dev_known)}
            width_dir = gallery_root / ("a%s" % str(alpha).replace(".", "")) / ("seed-%d" % seed)
            width_dir.mkdir(parents=True, exist_ok=True)
            for backend, embedding_map in zip(("float", "int8"), feature_maps):
                for policy in ("quality", "uniform"):
                    candidate_results = []
                    for count in (1, 3, 5, 10, len(KNOWN) * 12):
                        prototypes, classes, selection = build_gallery_prototypes(
                            embedding_map, train_known, count, policy, int(seed))
                        def matrix_for(rows):
                            return np.stack([embedding_map[str(row["record_id"])] for row in rows]) @ prototypes.T
                        cal_known_matrix = matrix_for(cal_known)
                        cal_unknown_matrix = matrix_for(cal_unknown)
                        dev_known_matrix = matrix_for(dev_known)
                        dev_unknown_matrix = matrix_for(dev_unknown)
                        cal = fit_gallery_threshold(cal_known_matrix, cal_known, cal_unknown_matrix, cal_unknown)
                        dev = gallery_metrics(dev_known_matrix, dev_known, dev_unknown_matrix, dev_unknown,
                                              cal["threshold"])
                        candidate_results.append({"template_count_requested": count,
                                                  "template_count_actual": selection[KNOWN[0]]["template_count"],
                                                  "calibration": cal, "development": dev,
                                                  "selection": selection,
                                                  "prototype_sha256": sha256_json(prototypes.tolist())})
                    chosen = max(candidate_results, key=lambda row: (
                        float(row["development"]["development_macro_utility"]),
                        float(row["development"]["known_macro_identification"]),
                        -int(row["template_count_requested"])))
                    result = {"schema": "job-d-gallery-result-1", "status": "complete_verified",
                              "arm_id": arm_id, "root_seed": int(seed), "width": alpha,
                              "backend": backend, "policy": policy,
                              "representation": "64-D L2-normalized learned penultimate embedding",
                              "pretraining": "Edge Impulse MobileNetV1 gray checkpoint; classifier fine-tuned for K01-K05+OTHER",
                              "head": "mean of selected training-role template embeddings per known identity; cosine rank-1",
                              "candidate_counts": candidate_results, "selected": chosen,
                              "train_gallery_rows_sha256": rows_fingerprint(train_known),
                              "calibration_roles": ["threshold_calibration", "unknown_calibration"],
                              "selection_roles": ["development", "unknown_development"],
                              "final_role_opened": False, "created_utc": utc_now()}
                    result_path = width_dir / ("%s_%s.json" % (backend, policy))
                    write_json(result_path, result)
                    result["result_path"] = str(result_path)
                    results.append(result)
            del model
            tf.keras.backend.clear_session()
    return results


def _sample_group(row: dict) -> str:
    return str(row.get("duplicate_group_id") or row.get("d_group_component") or row.get("record_id"))


def classifier_reliability(logits: np.ndarray, rows: list[dict], outputs: int, bins: int = 10) -> dict:
    known_rows = [row for row in rows if row.get("label_id") in KNOWN]
    known_logits = np.asarray(logits[:len(known_rows)])
    probs = softmax_np(known_logits)
    known_probs = probs[:, :5]
    predicted = np.argmax(known_probs, axis=1)
    confidence = np.max(known_probs, axis=1)
    truth = np.asarray([row_label_index(row) for row in known_rows])
    correct = (predicted == truth).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability_bins = []
    ece = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < bins - 1 else confidence <= edges[i + 1])
        count = int(np.sum(mask))
        if count:
            avg_conf = float(np.mean(confidence[mask]))
            avg_acc = float(np.mean(correct[mask]))
            ece += count / max(1, len(confidence)) * abs(avg_conf - avg_acc)
        else:
            avg_conf = avg_acc = None
        reliability_bins.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                                 "n": count, "mean_confidence": avg_conf, "accuracy": avg_acc})
    top2 = np.partition(known_probs, -2, axis=1)[:, -2:]
    margins = np.max(top2, axis=1) - np.min(top2, axis=1)
    # Correct ordering: largest minus second-largest probability.
    sorted_top = np.sort(known_probs, axis=1)
    margins = sorted_top[:, -1] - sorted_top[:, -2]
    unknown_rows = [row for row in rows if row.get("identity_partition") == "unknown"]
    unknown_logits = np.asarray(logits[len(known_rows):]) if len(logits) > len(known_rows) else np.zeros((0, outputs))
    if len(unknown_logits):
        _, unknown_scores, unknown_other = prediction_components(unknown_logits, outputs)
    else:
        unknown_scores, unknown_other = np.zeros(0), np.zeros(0, dtype=bool)
    return {"known_n": len(known_rows), "ece_top1_known_score": float(ece),
            "known_confidence_mean": float(np.mean(confidence)),
            "known_confidence_median": float(np.median(confidence)),
            "known_top1_top2_margin_mean": float(np.mean(margins)),
            "known_correctness_rate": float(np.mean(correct)),
            "reliability_bins": reliability_bins,
            "unknown_n": len(unknown_rows),
            "unknown_max_known_score_mean": float(np.mean(unknown_scores)) if len(unknown_scores) else None,
            "unknown_max_known_score_median": float(np.median(unknown_scores)) if len(unknown_scores) else None,
            "unknown_other_output_rate": float(np.mean(unknown_other)) if len(unknown_other) else None,
            "calibration_scope": "descriptive only; decision threshold fit separately on threshold_calibration and unknown_calibration"}


def open_set_curve(logits: np.ndarray, known_rows: list[dict], unknown_rows: list[dict], outputs: int,
                   max_points: int = 101) -> list[dict]:
    known_logits = np.asarray(logits[:len(known_rows)])
    unknown_logits = np.asarray(logits[len(known_rows):])
    kp, ks, ko = prediction_components(known_logits, outputs)
    up, us, uo = prediction_components(unknown_logits, outputs)
    kt = np.asarray([row_label_index(row) for row in known_rows])
    all_scores = np.concatenate([ks, us])
    quantiles = np.linspace(0.0, 1.0, min(max_points, len(all_scores)))
    thresholds = sorted(set(float(v) for v in np.quantile(all_scores, quantiles)))
    thresholds.append(float(np.nextafter(max(thresholds), math.inf)))
    curve = []
    for threshold in thresholds:
        accepted_known = (ks >= threshold) & ~ko
        accepted_unknown = (us >= threshold) & ~uo
        curve.append({"threshold": float(threshold),
                      "tpir_correct_identification": float(np.mean(accepted_known & (kp == kt))),
                      "known_acceptance": float(np.mean(accepted_known)),
                      "fpir_unknown_false_identification": float(np.mean(accepted_unknown)) if len(us) else None})
    return curve


def _open_confusion(logits: np.ndarray, known_rows: list[dict], unknown_rows: list[dict],
                    threshold: float, outputs: int) -> dict:
    names = KNOWN + ["OTHER", "REJECT"]
    true_names = KNOWN + ["UNKNOWN"]
    counts = {true_name: {name: 0 for name in names} for true_name in true_names}
    known_logits = logits[:len(known_rows)]
    unknown_logits = logits[len(known_rows):]
    for row, value in zip(known_rows, known_logits):
        pred, score, other = prediction_components(value[None, :], outputs)
        if bool(other[0]):
            name = "OTHER"
        elif float(score[0]) < threshold:
            name = "REJECT"
        else:
            name = KNOWN[int(pred[0])]
        counts[str(row["label_id"])][name] += 1
    for row, value in zip(unknown_rows, unknown_logits):
        pred, score, other = prediction_components(value[None, :], outputs)
        if bool(other[0]):
            name = "OTHER"
        elif float(score[0]) < threshold:
            name = "REJECT"
        else:
            name = KNOWN[int(pred[0])]
        counts["UNKNOWN"][name] += 1
    return {"row_labels": true_names, "column_labels": names, "counts": counts}


def bootstrap_open_metrics(logits: np.ndarray, known_rows: list[dict], unknown_rows: list[dict],
                           threshold: float, outputs: int, seed: int, replicates: int = 1000) -> dict:
    known_logits = np.asarray(logits[:len(known_rows)])
    unknown_logits = np.asarray(logits[len(known_rows):])
    kp, ks, ko = prediction_components(known_logits, outputs)
    up, us, uo = prediction_components(unknown_logits, outputs)
    kt = np.asarray([row_label_index(row) for row in known_rows])
    ka = (ks >= threshold) & ~ko
    ua = (us >= threshold) & ~uo
    by_class_group = {label: defaultdict(list) for label in KNOWN}
    for i, row in enumerate(known_rows):
        by_class_group[str(row["label_id"])][_sample_group(row)].append(i)
    by_unknown_id = defaultdict(list)
    for i, row in enumerate(unknown_rows):
        by_unknown_id[str(row["identity_id"])].append(i)
    rng = np.random.default_rng(int(seed) ^ 0x391D7A)
    metrics = defaultdict(list)
    for _ in range(replicates):
        selected_known = {label: [] for label in KNOWN}
        for label in KNOWN:
            groups = list(by_class_group[label].values())
            picks = rng.integers(0, len(groups), size=len(groups))
            selected_known[label] = [idx for pick in picks for idx in groups[int(pick)]]
        per_ident, per_correct, per_wrong, per_reject = [], [], [], []
        for label in KNOWN:
            idx = np.asarray(selected_known[label], dtype=np.int64)
            per_ident.append(float(np.mean(kp[idx] == kt[idx])))
            per_correct.append(float(np.mean(ka[idx] & (kp[idx] == kt[idx]))))
            per_wrong.append(float(np.mean(ka[idx] & (kp[idx] != kt[idx]))))
            per_reject.append(float(np.mean(~ka[idx])))
        id_groups = list(by_unknown_id.values())
        picks = rng.integers(0, len(id_groups), size=len(id_groups))
        selected_unknown = np.asarray([idx for pick in picks for idx in id_groups[int(pick)]], dtype=np.int64)
        fpir = float(np.mean(ua[selected_unknown])) if len(selected_unknown) else 0.0
        unknown_reject = 1.0 - fpir
        metrics["known_macro_identification"].append(float(np.mean(per_ident)))
        metrics["known_macro_correct_acceptance"].append(float(np.mean(per_correct)))
        metrics["known_macro_wrong_known_identity"].append(float(np.mean(per_wrong)))
        metrics["known_macro_rejection"].append(float(np.mean(per_reject)))
        metrics["unknown_fpir"].append(fpir)
        metrics["unknown_rejection"].append(unknown_reject)
        metrics["macro_utility"].append(float((np.mean(per_correct) + unknown_reject) / 2.0))
    return {"method": "stratified image-group bootstrap within each known identity; unknown identity-cluster bootstrap",
            "replicates": replicates,
            "intervals_95_percent": {name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                                      for name, values in metrics.items()},
            "known_image_group_counts": {label: len(by_class_group[label]) for label in KNOWN},
            "unknown_identity_count": len(by_unknown_id)}


def bootstrap_olivetti_accuracy(logits: np.ndarray, rows: list[dict], seed: int,
                                replicates: int = 1000) -> dict:
    labels = sorted({str(row["label_id"]) for row in rows})
    pred = np.argmax(logits, axis=1)
    true = np.asarray([labels.index(str(row["label_id"])) for row in rows])
    by_identity = defaultdict(list)
    for i, row in enumerate(rows):
        by_identity[str(row["label_id"])].append(i)
    rng = np.random.default_rng(int(seed) ^ 0xA0167)
    values = []
    identities = sorted(by_identity)
    for _ in range(replicates):
        picks = rng.integers(0, len(identities), size=len(identities))
        accuracy = []
        for pick in picks:
            idx = by_identity[identities[int(pick)]]
            sampled = rng.choice(idx, size=len(idx), replace=True)
            accuracy.append(float(np.mean(pred[sampled] == true[sampled])))
        values.append(float(np.mean(accuracy)))
    return {"method": "identity-cluster bootstrap, with image resampling inside identity",
            "replicates": replicates,
            "macro_identity_accuracy_95_percent_interval": [float(np.quantile(values, 0.025)),
                                                             float(np.quantile(values, 0.975))],
            "identity_count": len(identities),
            "final_images_per_identity": {identity: len(indices) for identity, indices in by_identity.items()}}


def development_candidate_selection(paths: dict, matrix: dict, gallery_results: list[dict]) -> dict:
    classifier_results = []
    for result_path in sorted((paths["runs"] / "survface").glob("*/seed-*/ROOT_RESULT.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete_verified":
            classifier_results.append(result)
    expected = len(matrix["survface_arms"]) * len(matrix["training_roots"])
    if len(classifier_results) != expected:
        raise ValueError("need all %d SurvFace root results before candidate selection; found %d" %
                         (expected, len(classifier_results)))
    arm_means = {}
    for arm in matrix["survface_arms"]:
        results = [r for r in classifier_results if r["arm_id"] == arm["id"]]
        if len(results) != 10:
            raise ValueError("missing ten roots for " + str(arm["id"]))
        values = [float(r["float_development"]["development_macro_utility"]) for r in results]
        arm_means[arm["id"]] = {"root_count": len(results), "mean_development_macro_utility": float(np.mean(values)),
                                 "mean_development_known_correct_acceptance": float(np.mean([r["float_development"]["known_identity_macro_correct_acceptance"] for r in results])),
                                 "mean_development_known_identification": float(np.mean([r["float_development"]["known_identity_macro_identification"] for r in results])),
                                 "sd_development_macro_utility": float(np.std(values, ddof=1)),
                                 "per_person_mean": {label: {
                                     metric: float(np.mean([r["float_development"]["per_person"][label][metric] for r in results]))
                                     for metric in ("correct_identification_rate", "correct_known_acceptance_rate",
                                                    "wrong_known_identity_rate", "known_rejection_rate", "accepted_coverage")
                                 } for label in KNOWN},
                                 "per_root": [{"root_seed": r["root_seed"],
                                               "utility": r["float_development"]["development_macro_utility"],
                                               "known_correct_acceptance": r["float_development"]["known_identity_macro_correct_acceptance"],
                                               "known_macro_identification": r["float_development"]["known_identity_macro_identification"],
                                               "unknown_fpir": r["float_development"]["unknown_false_identification_rate"]}
                                              for r in results]}
    small_candidates = [a for a in matrix["survface_arms"] if float(a["width"]) == 0.1]
    large_candidates = [a for a in matrix["survface_arms"] if float(a["width"]) == 0.25]
    best_small = max(small_candidates, key=lambda arm: (arm_means[arm["id"]]["mean_development_macro_utility"],
                                                        arm_means[arm["id"]]["mean_development_known_identification"]))
    best_large = max(large_candidates, key=lambda arm: (arm_means[arm["id"]]["mean_development_macro_utility"],
                                                        arm_means[arm["id"]]["mean_development_known_identification"]))
    best_classifier = max(matrix["survface_arms"], key=lambda arm: (arm_means[arm["id"]]["mean_development_macro_utility"],
                                                                     arm_means[arm["id"]]["mean_development_known_identification"]))
    gallery_scores = defaultdict(list)
    gallery_details = defaultdict(list)
    for result in gallery_results:
        family = "a%s_%s_%s" % ("010" if result["width"] == 0.1 else "025", result["backend"], result["policy"])
        gallery_scores[family].append(float(result["selected"]["development"]["development_macro_utility"]))
        gallery_details[family].append({"root_seed": result["root_seed"],
                                        "utility": result["selected"]["development"]["development_macro_utility"],
                                        "known_correct_acceptance": result["selected"]["development"]["known_macro_correct_acceptance"],
                                        "template_count": result["selected"]["template_count_requested"],
                                        "threshold": result["selected"]["calibration"]["threshold"]})
    gallery_means = {family: {"root_count": len(values), "mean_development_macro_utility": float(np.mean(values)),
                              "mean_development_known_correct_acceptance": float(np.mean([row["known_correct_acceptance"] for row in gallery_details[family]])),
                              "mean_development_known_identification": float(np.mean([r["selected"]["development"]["known_macro_identification"]
                                  for r in gallery_results if ("a%s_%s_%s" % ("010" if r["width"] == 0.1 else "025", r["backend"], r["policy"])) == family])),
                              "per_person_mean": {label: {
                                  metric: float(np.mean([r["selected"]["development"]["per_person"][label][metric]
                                                         for r in gallery_results
                                                         if ("a%s_%s_%s" % ("010" if r["width"] == 0.1 else "025",
                                                                            r["backend"], r["policy"])) == family]))
                                  for metric in ("identification_rate", "correct_known_acceptance_rate",
                                                 "wrong_known_identity_rate", "rejection_rate", "coverage")
                              } for label in KNOWN},
                              "per_root": gallery_details[family]}
                     for family, values in gallery_scores.items()}
    best_gallery = max(gallery_means, key=lambda family: gallery_means[family]["mean_development_macro_utility"])
    deployment_routes = [
        {"family_type": "classifier", "family_id": arm["id"], "width": float(arm["width"]),
         "backend": "float", "mean_development_macro_utility": arm_means[arm["id"]]["mean_development_macro_utility"],
         "mean_development_known_correct_acceptance": arm_means[arm["id"]]["mean_development_known_correct_acceptance"],
         "mean_development_known_identification": arm_means[arm["id"]]["mean_development_known_identification"],
         "root_count": arm_means[arm["id"]]["root_count"], "deployment_eligible": False}
        for arm in matrix["survface_arms"]
    ]
    for family, values in gallery_means.items():
        width = 0.1 if family.startswith("a010_") else 0.25
        deployment_routes.append({"family_type": "embedding_gallery", "family_id": family,
                                  "width": width, "backend": family.split("_")[1],
                                  "mean_development_macro_utility": values["mean_development_macro_utility"],
                                  "mean_development_known_correct_acceptance": values["mean_development_known_correct_acceptance"],
                                  "mean_development_known_identification": values["mean_development_known_identification"],
                                  "root_count": values["root_count"], "deployment_eligible": False})
    student_registry_path = paths["reports"] / "TEACHER_MODEL_REGISTRY_FROZEN.json"
    student_registry = None
    if (paths["study"] / "D_TEACHER_ADDENDUM_v1.json").is_file() and not student_registry_path.is_file():
        raise FileNotFoundError("the frozen SFace/student investigation must finish before candidate selection")
    student_route_means = {}
    if student_registry_path.is_file():
        student_registry = json.loads(student_registry_path.read_text(encoding="utf-8"))
        if student_registry.get("status") != "complete_verified" or student_registry.get("final_role_opened") is not False:
            raise ValueError("teacher/student development models must be frozen before candidate selection")
        for width, width_tag in ((0.1, "010"), (0.25, "025")):
            for backend in ("float", "int8"):
                root_metrics = []
                for entry in student_registry["models"]:
                    if entry.get("family_type") != "sface_distilled_mobilenet_student" or entry.get("model_id") != "sv_sface_distill_mnv1_a" + width_tag:
                        continue
                    root = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))
                    selected_student = root["selected_float" if backend == "float" else "selected_int8"]
                    root_metrics.append({"root_seed": root["root_seed"],
                                         "utility": selected_student["development"]["development_macro_utility"],
                                         "identification": selected_student["development"]["known_macro_identification"],
                                         "correct_acceptance": selected_student["development"]["known_macro_correct_acceptance"],
                                         "unknown_fpir": selected_student["development"]["unknown_false_identification_rate"]})
                if len(root_metrics) != 10:
                    raise ValueError("student route requires ten frozen roots for width %s/%s" % (width, backend))
                key = "a%s_%s" % (width_tag, backend)
                student_route_means[key] = {"width": width, "backend": backend, "root_count": len(root_metrics),
                                            "mean_development_macro_utility": float(np.mean([r["utility"] for r in root_metrics])),
                                            "mean_development_known_correct_acceptance": float(np.mean([r["correct_acceptance"] for r in root_metrics])),
                                            "mean_development_known_identification": float(np.mean([r["identification"] for r in root_metrics])),
                                            "per_root": root_metrics}
                deployment_routes.append({"family_type": "sface_distilled_student", "family_id": key,
                                          "width": width, "backend": backend,
                                          "mean_development_macro_utility": student_route_means[key]["mean_development_macro_utility"],
                                          "mean_development_known_correct_acceptance": student_route_means[key]["mean_development_known_correct_acceptance"],
                                          "mean_development_known_identification": student_route_means[key]["mean_development_known_identification"],
                                          "root_count": len(root_metrics), "deployment_eligible": False,
                                          "teacher_provenance": "OpenCV Zoo SFace; pretraining overlap unresolved"})
    teacher_route = None
    if student_registry is not None:
        teacher_path = paths["reports"] / "SFACE_TEACHER_RESULT.json"
        if not teacher_path.is_file():
            raise FileNotFoundError("frozen teacher registry has no SFace development reference result")
        teacher_result = json.loads(teacher_path.read_text(encoding="utf-8"))
        teacher_dev = teacher_result["crop_mode_results"]["selected"]["development"]
        teacher_route = {"family_type": "fixed_sface_host_teacher", "family_id": "sface_2021dec_host_reference",
                         "mean_development_macro_utility": float(teacher_dev["development_macro_utility"]),
                         "mean_development_known_correct_acceptance": float(teacher_dev["known_macro_correct_acceptance"]),
                         "mean_development_known_identification": float(teacher_dev["known_macro_identification"]),
                         "known_macro_identification": float(teacher_dev["known_macro_identification"]),
                         "known_macro_correct_acceptance": float(teacher_dev["known_macro_correct_acceptance"]),
                         "unknown_fpir": float(teacher_dev["unknown_false_identification_rate"]),
                         "root_count": 1, "deployment_eligible": False,
                         "reason": "host-only reference; exact pretraining corpus and authorization unresolved"}
    best_task_route = max(deployment_routes, key=lambda row: (
        row["mean_development_known_correct_acceptance"] > 0,
        row["mean_development_macro_utility"], row["mean_development_known_identification"], row["family_id"]))
    small_routes = [row for row in deployment_routes if row["width"] == 0.1]
    large_routes = [row for row in deployment_routes if row["width"] == 0.25]
    best_small_route = max(small_routes, key=lambda row: (
        row["mean_development_known_correct_acceptance"] > 0,
        row["mean_development_macro_utility"], row["mean_development_known_identification"], row["family_id"]))
    best_large_route = max(large_routes, key=lambda row: (
        row["mean_development_known_correct_acceptance"] > 0,
        row["mean_development_macro_utility"], row["mean_development_known_identification"], row["family_id"]))
    best_task_route["useful_non_reject_all"] = best_task_route["mean_development_known_correct_acceptance"] > 0
    best_small_route["useful_non_reject_all"] = best_small_route["mean_development_known_correct_acceptance"] > 0
    best_large_route["useful_non_reject_all"] = best_large_route["mean_development_known_correct_acceptance"] > 0
    reference_options = list(deployment_routes)
    if teacher_route:
        reference_options.append(teacher_route)
    useful_options = [row for row in reference_options if row.get("mean_development_known_correct_acceptance", 0) > 0]
    best_any = max(useful_options or reference_options,
                   key=lambda row: (row["mean_development_macro_utility"],
                                    row.get("mean_development_known_correct_acceptance", 0),
                                    row.get("mean_development_known_identification", 0), row["family_id"]))
    historical_summary_path = paths["reports"] / "HISTORICAL_GALLERY_RESULTS.json"
    historical_summary = None
    historical_registry_path = paths["reports"] / "HISTORICAL_MODEL_REGISTRY_FROZEN.json"
    historical_declared = any(row.get("id") == "v3_custom_base_tiny_historical_gallery"
                              for row in matrix.get("representation_comparisons", []))
    if historical_declared and not historical_registry_path.is_file():
        raise FileNotFoundError("matrix-declared historical controls must be evaluated and frozen before selection")
    if historical_declared and not historical_summary_path.is_file():
        raise FileNotFoundError("matrix-declared historical development summary is missing")
    historical_registry = None
    if historical_registry_path.is_file():
        historical_registry = json.loads(historical_registry_path.read_text(encoding="utf-8"))
        if (historical_registry.get("status") != "complete_verified" or
                historical_registry.get("final_role_opened") is not False or
                historical_registry.get("root_count") != 20 or
                historical_registry.get("backend_policy_evaluation_count") != 80):
            raise ValueError("historical controls are not complete and frozen")
    if historical_summary_path.is_file():
        historical_summary = json.loads(historical_summary_path.read_text(encoding="utf-8"))
        if historical_summary.get("status") != "complete_verified" or historical_summary.get("final_role_opened") is not False:
            raise ValueError("historical development controls must be frozen before candidate selection")
    quality_pair = []
    quality_by_seed = {r["root_seed"]: r for r in classifier_results if r["arm_id"] == "sv_mnv1_a025_quality_select_6"}
    uniform_by_seed = {r["root_seed"]: r for r in classifier_results if r["arm_id"] == "sv_mnv1_a025_uniform_select_6"}
    for seed in matrix["training_roots"]:
        q = quality_by_seed[seed]["float_development"]["development_macro_utility"]
        u = uniform_by_seed[seed]["float_development"]["development_macro_utility"]
        quality_pair.append({"root_seed": seed, "quality_utility": q, "uniform_utility": u,
                             "paired_difference_quality_minus_uniform": float(q - u)})
    selection = {
        "schema": "job-d-candidate-selection-1", "status": "frozen_before_final_role",
        "selection_source": "SurvFace threshold_calibration + unknown_calibration for thresholds and development + unknown_development for candidate selection",
        "final_role_opened": False,
        "classifier_arm_development_summary": arm_means,
        "candidate_small_family": {"arm_id": best_small["id"], "width": 0.1,
                                   "mean_development_macro_utility": arm_means[best_small["id"]]["mean_development_macro_utility"],
                                   "mean_development_known_correct_acceptance": arm_means[best_small["id"]]["mean_development_known_correct_acceptance"],
                                   "useful_non_reject_all": arm_means[best_small["id"]]["mean_development_known_correct_acceptance"] > 0},
        "candidate_large_family": {"arm_id": best_large["id"], "width": 0.25,
                                   "mean_development_macro_utility": arm_means[best_large["id"]]["mean_development_macro_utility"],
                                   "mean_development_known_correct_acceptance": arm_means[best_large["id"]]["mean_development_known_correct_acceptance"],
                                   "useful_non_reject_all": arm_means[best_large["id"]]["mean_development_known_correct_acceptance"] > 0},
        "strong_reference": {"family_type": best_any["family_type"], "family_id": best_any["family_id"],
                             "mean_development_macro_utility": float(best_any["mean_development_macro_utility"]),
                             "mean_development_known_correct_acceptance": float(best_any.get("mean_development_known_correct_acceptance", 0)),
                             "mean_development_known_identification": float(best_any.get("mean_development_known_identification", 0)),
                             "deployment_eligible": bool(best_any.get("deployment_eligible", False)),
                             "useful_non_reject_all": bool(best_any.get("mean_development_known_correct_acceptance", 0) > 0),
                             "interpretation": "If this is false, every compared route rejected all known faces correctly at the calibrated operating point; report no useful model.",
                             "selection_basis": "maximize development utility among routes with nonzero known correct acceptance when any exist; historical controls excluded from selection"},
        "best_task_route": best_task_route,
        "candidate_small_best_host_route": best_small_route,
        "candidate_large_best_host_route": best_large_route,
        "student_route_development_summary": student_route_means,
        "fixed_teacher_development_summary": teacher_route,
        "historical_control_development_summary_path": str(historical_summary_path),
        "historical_control_development_summary": historical_summary,
        "historical_control_registry_path": str(historical_registry_path) if historical_registry else None,
        "historical_control_registry_sha256": sha256_file(historical_registry_path) if historical_registry else None,
        "historical_controls_used_for_selection": False,
        "best_classifier_arm_overall": {"arm_id": best_classifier["id"],
                                         "mean_development_macro_utility": arm_means[best_classifier["id"]]["mean_development_macro_utility"]},
        "best_embedding_gallery_family": {"family_id": best_gallery,
                                           "mean_development_macro_utility": gallery_means[best_gallery]["mean_development_macro_utility"]},
        "gallery_family_development_summary": gallery_means,
        "quality_vs_uniform_selection_sensitivity": {
            "arm_pair": ["sv_mnv1_a025_quality_select_6", "sv_mnv1_a025_uniform_select_6"],
            "same_unknown_training_records": True, "same_root_initialization_and_training_config": True,
            "paired_per_root_development_differences": quality_pair,
            "mean_paired_difference": float(np.mean([r["paired_difference_quality_minus_uniform"] for r in quality_pair])),
        },
        "created_utc": utc_now(),
    }
    write_json(paths["reports"] / "CANDIDATE_SELECTION.json", selection)
    return selection


def freeze_model_registry(paths: dict, matrix: dict, selection: dict,
                          gallery_results: list[dict]) -> dict:
    selection_path = paths["reports"] / "CANDIDATE_SELECTION.json"
    protocol_hash = sha256_json({"matrix_sha256": sha256_file(paths["matrix"]),
                                 "candidate_selection_sha256": sha256_file(selection_path)})
    frozen_models = []
    for result_path in sorted((paths["runs"] / "survface").glob("*/seed-*/ROOT_RESULT.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        root_dir = result_path.parent
        config_path = root_dir / "config.json"
        calibration_path = root_dir / "float_calibration.json"
        int8_calibration_path = root_dir / "int8_calibration.json"
        if not all(path.is_file() for path in (config_path, calibration_path, int8_calibration_path,
                                               root_dir / "classifier_float.tflite", root_dir / "classifier_int8.tflite")):
            raise FileNotFoundError("cannot freeze incomplete root: " + str(root_dir))
        config_hash = sha256_file(config_path)
        calibration_hash = sha256_json({"float": sha256_file(calibration_path),
                                        "int8": sha256_file(int8_calibration_path)})
        receipt = {"schema": "job-d-model-freeze-1", "status": "frozen",
                   "model_sha256": sha256_file(root_dir / "classifier.weights.h5"),
                   "model_artifacts": {"host_checkpoint_sha256": sha256_file(root_dir / "classifier.weights.h5"),
                                       "float_tflite_sha256": sha256_file(root_dir / "classifier_float.tflite"),
                                       "int8_tflite_sha256": sha256_file(root_dir / "classifier_int8.tflite")},
                   "config_sha256": config_hash, "calibration_sha256": calibration_hash,
                   "protocol_sha256": protocol_hash, "final_role_opened": False,
                   "model_id": result["arm_id"], "root_seed": result["root_seed"],
                   "label_order": result["label_order"], "output_contract": result["output_contract"],
                   "created_utc": utc_now()}
        write_json(root_dir / "FROZEN.json", receipt)
        frozen_models.append({"dataset": "survface_96", "model_id": result["arm_id"],
                              "root_seed": result["root_seed"], "result_path": str(result_path),
                              "freeze_receipt": str(root_dir / "FROZEN.json"), **receipt})
    for result_path in sorted((paths["runs"] / "olivetti").glob("*/seed-*/ROOT_RESULT.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        root_dir = result_path.parent
        config_path = root_dir / "config.json"
        if not config_path.is_file():
            if sha256_json(result.get("config")) != result.get("config_sha256"):
                raise ValueError("Olivetti root configuration hash is invalid: " + str(result_path))
            write_json(config_path, result["config"])
        if sha256_json(json.loads(config_path.read_text(encoding="utf-8"))) != result.get("config_sha256"):
            raise ValueError("Olivetti root configuration changed after training: " + str(result_path))
        closed_cal = root_dir / "closed_set_calibration.json"
        write_json(closed_cal, {"status": "closed_set_only", "score_calibration": False,
                                "dataset": "olivetti_96", "reason": "40-way closed-set sanity protocol"})
        receipt = {"schema": "job-d-model-freeze-1", "status": "frozen",
                   "model_sha256": sha256_file(root_dir / "classifier.weights.h5"),
                   "model_artifacts": {"host_checkpoint_sha256": sha256_file(root_dir / "classifier.weights.h5"),
                                       "float_tflite_sha256": sha256_file(root_dir / "classifier_float.tflite"),
                                       "int8_tflite_sha256": sha256_file(root_dir / "classifier_int8.tflite")},
                   "config_sha256": sha256_file(config_path), "calibration_sha256": sha256_file(closed_cal),
                   "protocol_sha256": protocol_hash, "final_role_opened": False,
                   "model_id": result["arm_id"], "root_seed": result["root_seed"],
                   "label_order": result["label_order"], "output_contract": result["output_contract"],
                   "created_utc": utc_now()}
        write_json(root_dir / "FROZEN.json", receipt)
        frozen_models.append({"dataset": "olivetti_96", "model_id": result["arm_id"],
                              "root_seed": result["root_seed"], "result_path": str(result_path),
                              "freeze_receipt": str(root_dir / "FROZEN.json"), **receipt})
    frozen_galleries = []
    for result in gallery_results:
        result_path = Path(result["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        root_dir = result_path.parent
        gallery_key = "%s_%s" % (result["backend"], result["policy"])
        base_dir = paths["runs"] / "survface" / result["arm_id"] / ("seed-%d" % result["root_seed"])
        export_key = "float_embedding_export" if result["backend"] == "float" else "int8_embedding_export"
        model_path = base_dir / ("embedding_float.tflite" if result["backend"] == "float" else "embedding_int8.tflite")
        config_path = root_dir / (gallery_key + ".CONFIG.json")
        calibration_path = root_dir / (gallery_key + ".CALIBRATION.json")
        write_json(config_path, {"arm_id": result["arm_id"], "root_seed": result["root_seed"],
                                 "backend": result["backend"], "policy": result["policy"],
                                 "selected": result["selected"], "representation": result["representation"],
                                 "output_contract": "embedding-64/gallery-cosine"})
        write_json(calibration_path, result["selected"]["calibration"])
        receipt = {"schema": "job-d-gallery-freeze-1", "status": "frozen",
                   "model_sha256": sha256_file(model_path), "model_path": str(model_path),
                   "config_sha256": sha256_file(config_path), "calibration_sha256": sha256_file(calibration_path),
                   "protocol_sha256": protocol_hash, "final_role_opened": False,
                   "model_id": result["arm_id"], "root_seed": result["root_seed"],
                   "backend": result["backend"], "policy": result["policy"],
                   "output_contract": "embedding-64/gallery-cosine", "created_utc": utc_now()}
        freeze_path = root_dir / (gallery_key + ".FROZEN.json")
        write_json(freeze_path, receipt)
        frozen_galleries.append({"dataset": "survface_96", "family_type": "embedding_gallery",
                                 "result_path": str(result_path), "freeze_receipt": str(freeze_path), **receipt})
    if len(frozen_models) != len(matrix["survface_arms"]) * 10 + len(matrix["olivetti_arms"]) * 10:
        raise ValueError("frozen registry has a missing trained classifier root")
    teacher_registry_path = paths["reports"] / "TEACHER_MODEL_REGISTRY_FROZEN.json"
    teacher_registry = None
    teacher_declared = (paths["study"] / "D_TEACHER_ADDENDUM_v1.json").is_file()
    if teacher_declared and not teacher_registry_path.is_file():
        raise FileNotFoundError("the frozen teacher/student investigation must finish before final access")
    if teacher_registry_path.is_file():
        teacher_registry = json.loads(teacher_registry_path.read_text(encoding="utf-8"))
        if teacher_registry.get("status") != "complete_verified" or teacher_registry.get("final_role_opened") is not False:
            raise ValueError("teacher/student registry must be complete and frozen before final access")
        if teacher_registry.get("student_root_count") != 20 or teacher_registry.get("model_count") != 21:
            raise ValueError("teacher/student registry must contain one fixed host teacher and twenty student roots")
        for entry in teacher_registry.get("models", []):
            receipt = json.loads(Path(entry["freeze_receipt"]).read_text(encoding="utf-8"))
            if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False:
                raise ValueError("teacher/student model freeze receipt is invalid before final access")
            if entry.get("root_seed") is None:
                model_result = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))
                model_path = Path(model_result["checkpoint"]["sface"]["path"])
                if not model_path.is_file() or sha256_file(model_path) != receipt.get("model_sha256"):
                    raise ValueError("fixed host teacher artifact changed after freeze")
            else:
                model_result = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))
                checked = [(Path(model_result["weights_path"]), receipt.get("model_sha256")),
                           (Path(model_result["float_export"]["path"]), receipt.get("model_artifacts", {}).get("float_tflite_sha256")),
                           (Path(model_result["int8_export"]["path"]), receipt.get("model_artifacts", {}).get("int8_tflite_sha256"))]
                if any(not path.is_file() or sha256_file(path) != digest for path, digest in checked):
                    raise ValueError("student artifact changed after freeze")
    historical_registry_path = paths["reports"] / "HISTORICAL_MODEL_REGISTRY_FROZEN.json"
    historical_registry = None
    historical_declared = any(row.get("id") == "v3_custom_base_tiny_historical_gallery"
                              for row in matrix.get("representation_comparisons", []))
    if historical_declared:
        if not historical_registry_path.is_file():
            raise FileNotFoundError("matrix-declared v3 historical controls must be audited and frozen before final access")
        historical_registry = json.loads(historical_registry_path.read_text(encoding="utf-8"))
        if historical_registry.get("status") != "complete_verified" or historical_registry.get("final_role_opened") is not False:
            raise ValueError("historical control registry must be complete and frozen before final access")
        if historical_registry.get("root_count") != 20 or historical_registry.get("backend_policy_evaluation_count") != 80:
            raise ValueError("historical controls require twenty roots and eighty backend-policy receipts")
        for gallery in historical_registry.get("galleries", []):
            freeze = json.loads(Path(gallery["freeze_receipt"]).read_text(encoding="utf-8"))
            if freeze.get("status") != "frozen" or freeze.get("final_role_opened") is not False:
                raise ValueError("every historical gallery threshold and model must be frozen before final access")
    registry = {"schema": "job-d-model-registry-1", "status": "complete_verified",
                "matrix_id": matrix["matrix_id"], "protocol_sha256": protocol_hash,
                "candidate_selection": str(selection_path), "candidate_selection_sha256": sha256_file(selection_path),
                "models": frozen_models, "galleries": frozen_galleries,
                "teacher_student_registry": str(teacher_registry_path) if teacher_registry else None,
                "teacher_student_registry_sha256": sha256_file(teacher_registry_path) if teacher_registry else None,
                "teacher_student_models": teacher_registry.get("models", []) if teacher_registry else [],
                "historical_control_registry": str(historical_registry_path) if historical_registry else None,
                "historical_control_registry_sha256": sha256_file(historical_registry_path) if historical_registry else None,
                "historical_control_models": historical_registry.get("roots", []) if historical_registry else [],
                "historical_control_galleries": historical_registry.get("galleries", []) if historical_registry else [],
                "families_for_E": {"strong_reference": selection["strong_reference"],
                                   "candidate_small_family": selection["candidate_small_family"],
                                   "candidate_large_family": selection["candidate_large_family"],
                                   "best_embedding_gallery_family": selection["best_embedding_gallery_family"],
                                   "candidate_small_best_host_route": selection.get("candidate_small_best_host_route"),
                                   "candidate_large_best_host_route": selection.get("candidate_large_best_host_route")},
                "final_role_opened": False, "created_utc": utc_now()}
    write_json(paths["reports"] / "MODEL_REGISTRY_FROZEN.json", registry)
    return registry


def _accepted_decisions(logits: np.ndarray, known_rows: list[dict], unknown_rows: list[dict],
                        threshold: float, outputs: int) -> tuple[list[str], list[str]]:
    predictions, scores, other = prediction_components(logits, outputs)
    all_rows = known_rows + unknown_rows
    decisions = []
    truths = []
    for row, pred, score, is_other in zip(all_rows, predictions, scores, other):
        truths.append(str(row.get("label_id")) if row in known_rows else "UNKNOWN")
        if is_other:
            decisions.append("OTHER")
        elif float(score) < float(threshold):
            decisions.append("REJECT")
        else:
            decisions.append(KNOWN[int(pred)])
    return decisions, truths


def _prediction_rows(all_rows: list[dict], truths: list[str], decisions: list[str],
                     labels_raw: np.ndarray, scores: np.ndarray, accepted: np.ndarray,
                     backend: str) -> list[dict]:
    result = []
    for i, row in enumerate(all_rows):
        truth = truths[i]
        decision = decisions[i]
        result.append({"record_id": row["record_id"], "identity_id": row.get("identity_id"),
                       "sample_group_id": _sample_group(row), "role": row.get("role"),
                       "ground_truth": truth, "predicted_decision": decision,
                       "accepted_as_known": bool(accepted[i]),
                       "raw_known_prediction": KNOWN[int(labels_raw[i])] if int(labels_raw[i]) < 5 else "OTHER",
                       "max_known_score": float(scores[i]), "backend": backend,
                       "correct_known_identity": bool(truth in KNOWN and decision == truth)})
    return result


def _write_predictions(path: Path, rows: list[dict]) -> str:
    return _save_rows_jsonl(path, rows)


def _classifier_final_for_root(paths: dict, result: dict, final_data: dict,
                               root_registry_receipt: dict) -> dict:
    root_dir = Path(result["weights_path"]).parent
    result_path = root_dir / "FINAL_RESULT.json"
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete_verified":
            return saved
    known_rows, unknown_rows = final_data["known_rows"], final_data["unknown_rows"]
    gray = np.concatenate([final_data["known_gray"], final_data["unknown_gray"]], axis=0)
    float_logits = tflite_outputs(root_dir / "classifier_float.tflite", gray)
    int8_logits = tflite_outputs(root_dir / "classifier_int8.tflite", gray)
    outputs = int(float_logits.shape[1])
    float_threshold = float(result["float_calibration"]["threshold"])
    int8_threshold = float(result["int8_calibration"]["threshold"])
    float_metrics = known_macro_and_open_metrics(float_logits, known_rows, unknown_rows, float_threshold, outputs)
    int8_metrics = known_macro_and_open_metrics(int8_logits, known_rows, unknown_rows, int8_threshold, outputs)
    float_reliability = classifier_reliability(float_logits, known_rows + unknown_rows, outputs)
    int8_reliability = classifier_reliability(int8_logits, known_rows + unknown_rows, outputs)
    float_pred, float_scores, float_other = prediction_components(float_logits, outputs)
    int8_pred, int8_scores, int8_other = prediction_components(int8_logits, outputs)
    float_accept = (float_scores >= float_threshold) & ~float_other
    int8_accept = (int8_scores >= int8_threshold) & ~int8_other
    float_decisions, truths = _accepted_decisions(float_logits, known_rows, unknown_rows, float_threshold, outputs)
    int8_decisions, _ = _accepted_decisions(int8_logits, known_rows, unknown_rows, int8_threshold, outputs)
    n_known = len(known_rows)
    float_prediction_path = root_dir / "final_predictions_float.jsonl"
    int8_prediction_path = root_dir / "final_predictions_int8.jsonl"
    float_predictions = _prediction_rows(known_rows + unknown_rows, truths, float_decisions,
                                         float_pred, float_scores, float_accept, "float_tflite")
    int8_predictions = _prediction_rows(known_rows + unknown_rows, truths, int8_decisions,
                                        int8_pred, int8_scores, int8_accept, "int8_tflite")
    float_prediction_hash = _write_predictions(float_prediction_path, float_predictions)
    int8_prediction_hash = _write_predictions(int8_prediction_path, int8_predictions)
    per_person_group_counts = {label: len({_sample_group(row) for row in known_rows if row.get("label_id") == label})
                               for label in KNOWN}
    root_freeze_path = root_dir / "FROZEN.json"
    receipt = json.loads(root_freeze_path.read_text(encoding="utf-8"))
    receipt["final_role_opened"] = True
    receipt["final_opened_utc"] = utc_now()
    write_json(root_freeze_path, receipt)
    final_result = {
        "schema": "job-d-final-root-result-1", "status": "complete_verified",
        "dataset": "survface_96", "arm_id": result["arm_id"], "root_seed": result["root_seed"],
        "label_order": result["label_order"], "output_contract": result["output_contract"],
        "pretrained": result["pretrained"], "checkpoint_sha256": result["checkpoint_sha256"],
        "float_threshold_calibrated_on": ["threshold_calibration", "unknown_calibration"],
        "int8_threshold_calibrated_on": ["threshold_calibration", "unknown_calibration"],
        "float_calibration": result["float_calibration"], "int8_calibration": result["int8_calibration"],
        "float_metrics": float_metrics, "int8_metrics": int8_metrics,
        "float_confusion": _open_confusion(float_logits, known_rows, unknown_rows, float_threshold, outputs),
        "int8_confusion": _open_confusion(int8_logits, known_rows, unknown_rows, int8_threshold, outputs),
        "float_open_set_curve": open_set_curve(float_logits, known_rows, unknown_rows, outputs),
        "int8_open_set_curve": open_set_curve(int8_logits, known_rows, unknown_rows, outputs),
        "float_reliability": float_reliability, "int8_reliability": int8_reliability,
        "float_bootstrap": bootstrap_open_metrics(float_logits, known_rows, unknown_rows, float_threshold,
                                                    outputs, int(result["root_seed"]), 1000),
        "int8_bootstrap": bootstrap_open_metrics(int8_logits, known_rows, unknown_rows, int8_threshold,
                                                  outputs, int(result["root_seed"]) + 100, 1000),
        "float_int8_decision_comparison": {
            "total_final_records": len(float_decisions),
            "raw_known_label_disagreement_count": int(np.sum(float_pred != int8_pred)),
            "reject_or_other_disagreement_count": int(np.sum(float_decisions != int8_decisions)),
            "accept_reject_disagreement_count": int(np.sum(float_accept != int8_accept)),
            "mean_absolute_calibrated_threshold_drift": float(abs(float_threshold - int8_threshold)),
            "float_threshold": float_threshold, "int8_threshold": int8_threshold,
            "float_int8_threshold_delta_int8_minus_float": float(int8_threshold - float_threshold),
        },
        "sample_counts": {"known": n_known, "unknown": len(unknown_rows),
                           "known_by_identity": {label: sum(row.get("label_id") == label for row in known_rows) for label in KNOWN},
                           "known_image_groups_by_identity": per_person_group_counts,
                           "unknown_identity_count": len({row["identity_id"] for row in unknown_rows}),
                           "unknown_images_per_identity": dict(Counter(str(row["identity_id"]) for row in unknown_rows)),
                           "conditional_on_supplied_crop": True, "full_camera_pipeline_measured": False},
        "predictions": {"float_path": str(float_prediction_path), "float_sha256": float_prediction_hash,
                        "int8_path": str(int8_prediction_path), "int8_sha256": int8_prediction_hash},
        "frozen_receipt": str(root_freeze_path), "created_utc": utc_now(),
    }
    write_json(result_path, final_result)
    return final_result


def gallery_final_for_result(paths: dict, result: dict, final_data: dict,
                             train_known_rows: list[dict], train_known_gray: np.ndarray) -> dict:
    result_path = Path(result["result_path"])
    root_dir = result_path.parent
    final_path = root_dir / ("%s_%s.FINAL.json" % (result["backend"], result["policy"]))
    if final_path.is_file():
        saved = json.loads(final_path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete_verified":
            return saved
    receipt_path = root_dir / ("%s_%s.FROZEN.json" % (result["backend"], result["policy"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    all_rows = train_known_rows + final_data["known_rows"] + final_data["unknown_rows"]
    all_gray = np.concatenate([train_known_gray, final_data["known_gray"], final_data["unknown_gray"]], axis=0)
    model_dir = paths["runs"] / "survface" / result["arm_id"] / ("seed-%d" % result["root_seed"])
    export = model_dir / ("embedding_float.tflite" if result["backend"] == "float" else "embedding_int8.tflite")
    features = _normalize_vectors(tflite_outputs(export, all_gray))
    feature_by_id = {str(row["record_id"]): features[i] for i, row in enumerate(all_rows)}
    selection = result["selected"]["selection"]
    prototypes = []
    for label in KNOWN:
        record_ids = selection[label]["record_ids"]
        mean = np.mean(np.stack([feature_by_id[rid] for rid in record_ids]), axis=0, keepdims=True)
        prototypes.append(_normalize_vectors(mean)[0])
    prototypes = np.stack(prototypes)
    start = len(train_known_rows)
    known_features = features[start:start + len(final_data["known_rows"])]
    unknown_features = features[start + len(final_data["known_rows"]):]
    known_matrix = known_features @ prototypes.T
    unknown_matrix = unknown_features @ prototypes.T
    threshold = float(result["selected"]["calibration"]["threshold"])
    metrics = gallery_metrics(known_matrix, final_data["known_rows"], unknown_matrix,
                              final_data["unknown_rows"], threshold)
    kp, ks = np.argmax(known_matrix, axis=1), np.max(known_matrix, axis=1)
    up, us = np.argmax(unknown_matrix, axis=1), np.max(unknown_matrix, axis=1)
    known_accept, unknown_accept = ks >= threshold, us >= threshold
    decisions = [KNOWN[int(p)] if a else "REJECT" for p, a in zip(kp, known_accept)] + \
                [KNOWN[int(p)] if a else "REJECT" for p, a in zip(up, unknown_accept)]
    truths = [str(row["label_id"]) for row in final_data["known_rows"]] + ["UNKNOWN"] * len(final_data["unknown_rows"])
    raw_preds = list(kp) + list(up)
    scores = list(ks) + list(us)
    accepts = list(known_accept) + list(unknown_accept)
    prediction_rows = _prediction_rows(final_data["known_rows"] + final_data["unknown_rows"], truths, decisions,
                                       np.asarray(raw_preds), np.asarray(scores), np.asarray(accepts),
                                       result["backend"] + "_gallery")
    pred_path = root_dir / ("%s_%s.final_predictions.jsonl" % (result["backend"], result["policy"]))
    pred_hash = _write_predictions(pred_path, prediction_rows)
    # ROC-style identification/rejection operating curve for cosine scores.
    all_scores = np.concatenate([ks, us])
    thresholds = sorted(set(float(v) for v in np.quantile(all_scores, np.linspace(0, 1, 101))))
    thresholds.append(float(np.nextafter(max(thresholds), math.inf)))
    curve = []
    true_known = np.asarray([row_label_index(row) for row in final_data["known_rows"]])
    for point in thresholds:
        ka = ks >= point
        ua = us >= point
        curve.append({"threshold": point,
                      "tpir_correct_identification": float(np.mean(ka & (kp == true_known))),
                      "fpir_unknown_false_identification": float(np.mean(ua)),
                      "known_acceptance": float(np.mean(ka))})
    receipt["final_role_opened"] = True
    receipt["final_opened_utc"] = utc_now()
    write_json(receipt_path, receipt)
    output = {"schema": "job-d-gallery-final-result-1", "status": "complete_verified",
              "arm_id": result["arm_id"], "root_seed": result["root_seed"],
              "width": float(result["width"]),
              "backend": result["backend"], "policy": result["policy"],
              "selected_template_count": result["selected"]["template_count_requested"],
              "selected_templates": selection, "threshold": threshold,
              "calibration": result["selected"]["calibration"], "metrics": metrics,
              "bootstrap": bootstrap_gallery_metrics(known_matrix, final_data["known_rows"],
                                                       unknown_matrix, final_data["unknown_rows"],
                                                       threshold, int(result["root_seed"]), 1000),
              "reliability": gallery_reliability(known_matrix, final_data["known_rows"],
                                                 unknown_matrix, threshold),
              "tpir_fpir_curve": curve,
              "float_int8_gallery_comparison": None,
              "sample_counts": {"known": len(final_data["known_rows"]),
                                "unknown": len(final_data["unknown_rows"]),
                                "unknown_identity_count": len({row["identity_id"] for row in final_data["unknown_rows"]}),
                                "known_groups_by_identity": {label: len({_sample_group(row) for row in final_data["known_rows"] if row["label_id"] == label}) for label in KNOWN}},
              "predictions": {"path": str(pred_path), "sha256": pred_hash},
              "frozen_receipt": str(receipt_path), "created_utc": utc_now()}
    write_json(final_path, output)
    return output


def bootstrap_gallery_metrics(known_matrix: np.ndarray, known_rows: list[dict],
                             unknown_matrix: np.ndarray, unknown_rows: list[dict],
                             threshold: float, seed: int, replicates: int = 1000) -> dict:
    kp, ks = np.argmax(known_matrix, axis=1), np.max(known_matrix, axis=1)
    up, us = np.argmax(unknown_matrix, axis=1), np.max(unknown_matrix, axis=1)
    truth = np.asarray([row_label_index(row) for row in known_rows])
    ka, ua = ks >= threshold, us >= threshold
    known_groups = {label: defaultdict(list) for label in KNOWN}
    unknown_groups = defaultdict(list)
    for i, row in enumerate(known_rows):
        known_groups[str(row["label_id"])][_sample_group(row)].append(i)
    for i, row in enumerate(unknown_rows):
        unknown_groups[str(row["identity_id"])].append(i)
    rng = np.random.default_rng(int(seed) ^ 0x7A11E2)
    values = defaultdict(list)
    for _ in range(replicates):
        recall, correct, wrong, reject = [], [], [], []
        for class_index, label in enumerate(KNOWN):
            groups = list(known_groups[label].values())
            picks = rng.integers(0, len(groups), size=len(groups))
            idx = np.asarray([i for pick in picks for i in groups[int(pick)]], dtype=np.int64)
            recall.append(float(np.mean(kp[idx] == truth[idx])))
            correct.append(float(np.mean(ka[idx] & (kp[idx] == truth[idx]))))
            wrong.append(float(np.mean(ka[idx] & (kp[idx] != truth[idx]))))
            reject.append(float(np.mean(~ka[idx])))
        identities = list(unknown_groups)
        picks = rng.integers(0, len(identities), size=len(identities))
        idx = np.asarray([j for pick in picks for j in unknown_groups[identities[int(pick)]]], dtype=np.int64)
        fpir = float(np.mean(ua[idx])) if len(idx) else 0.0
        values["known_macro_identification"].append(float(np.mean(recall)))
        values["known_macro_correct_acceptance"].append(float(np.mean(correct)))
        values["known_macro_wrong_identity"].append(float(np.mean(wrong)))
        values["known_macro_rejection"].append(float(np.mean(reject)))
        values["unknown_fpir"].append(fpir)
        values["unknown_rejection"].append(1.0 - fpir)
        values["macro_utility"].append(float((np.mean(correct) + 1.0 - fpir) / 2.0))
    return {"method": "stratified known image-group bootstrap and unknown identity-cluster bootstrap",
            "replicates": replicates,
            "intervals_95_percent": {name: [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
                                      for name, samples in values.items()},
            "known_image_group_counts": {label: len(known_groups[label]) for label in KNOWN},
            "unknown_identity_count": len(unknown_groups)}


def gallery_reliability(known_matrix: np.ndarray, known_rows: list[dict],
                        unknown_matrix: np.ndarray, threshold: float) -> dict:
    kp = np.argmax(known_matrix, axis=1)
    ks = np.max(known_matrix, axis=1)
    truth = np.asarray([row_label_index(row) for row in known_rows])
    sorted_values = np.sort(known_matrix, axis=1)
    margins = sorted_values[:, -1] - sorted_values[:, -2]
    unknown_scores = np.max(unknown_matrix, axis=1)
    return {"known_cosine_score_mean": float(np.mean(ks)),
            "known_cosine_score_median": float(np.median(ks)),
            "known_top1_top2_margin_mean": float(np.mean(margins)),
            "known_correct_at_calibrated_threshold": float(np.mean((ks >= threshold) & (kp == truth))),
            "unknown_cosine_score_mean": float(np.mean(unknown_scores)),
            "unknown_cosine_score_median": float(np.median(unknown_scores)),
            "calibrated_cosine_threshold": float(threshold),
            "interpretation": "cosine scores are model/gallery-specific and are not probabilities"}


def _olivetti_confusion(logits: np.ndarray, rows: list[dict], labels: list[str]) -> dict:
    pred = np.argmax(logits, axis=1)
    truth = np.asarray([labels.index(str(row["label_id"])) for row in rows])
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guessed in zip(truth, pred):
        matrix[int(actual), int(guessed)] += 1
    return {"row_labels": labels, "column_labels": labels, "counts": matrix.tolist()}


def olivetti_final_for_root(result: dict, final_data: dict) -> dict:
    root_dir = Path(result["weights_path"]).parent
    output_path = root_dir / "FINAL_RESULT.json"
    if output_path.is_file():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete_verified":
            return saved
    float_logits = tflite_outputs(root_dir / "classifier_float.tflite", final_data["gray"])
    int8_logits = tflite_outputs(root_dir / "classifier_int8.tflite", final_data["gray"])
    labels = result["label_order"]
    float_metrics = _olivetti_prediction_metrics(float_logits, final_data["rows"])
    int8_metrics = _olivetti_prediction_metrics(int8_logits, final_data["rows"])
    float_predictions = np.argmax(float_logits, axis=1)
    int8_predictions = np.argmax(int8_logits, axis=1)
    prediction_rows = []
    for index, row in enumerate(final_data["rows"]):
        prediction_rows.append({"record_id": row["record_id"], "identity_id": row["identity_id"],
                                "sample_group_id": _sample_group(row), "role": row["role"],
                                "ground_truth": row["label_id"],
                                "float_prediction": labels[int(float_predictions[index])],
                                "int8_prediction": labels[int(int8_predictions[index])],
                                "float_correct": bool(labels[int(float_predictions[index])] == row["label_id"]),
                                "int8_correct": bool(labels[int(int8_predictions[index])] == row["label_id"])})
    pred_path = root_dir / "final_predictions.jsonl"
    pred_hash = _write_predictions(pred_path, prediction_rows)
    receipt_path = root_dir / "FROZEN.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["final_role_opened"] = True
    receipt["final_opened_utc"] = utc_now()
    write_json(receipt_path, receipt)
    output = {"schema": "job-d-olivetti-final-root-1", "status": "complete_verified",
              "dataset": "olivetti_96", "arm_id": result["arm_id"], "root_seed": result["root_seed"],
              "label_order": labels, "float_metrics": float_metrics, "int8_metrics": int8_metrics,
              "float_confusion": _olivetti_confusion(float_logits, final_data["rows"], labels),
              "int8_confusion": _olivetti_confusion(int8_logits, final_data["rows"], labels),
              "float_bootstrap": bootstrap_olivetti_accuracy(float_logits, final_data["rows"], int(result["root_seed"]), 1000),
              "int8_bootstrap": bootstrap_olivetti_accuracy(int8_logits, final_data["rows"], int(result["root_seed"]) + 7, 1000),
              "float_int8_decision_disagreement_count": int(np.sum(float_predictions != int8_predictions)),
              "prediction_path": str(pred_path), "prediction_sha256": pred_hash,
              "sample_counts": {"final_rows": len(final_data["rows"]),
                                "identity_count": len({row["label_id"] for row in final_data["rows"]}),
                                "open_set_claim": False, "dataset_role": "final"},
              "created_utc": utc_now()}
    write_json(output_path, output)
    return output


def _paired_classifier_comparison(paths: dict, selection: dict, matrix: dict) -> dict:
    small_id = selection["candidate_small_family"]["arm_id"]
    large_id = selection["candidate_large_family"]["arm_id"]
    comparisons = []
    for seed in matrix["training_roots"]:
        small_path = paths["runs"] / "survface" / small_id / ("seed-%d" % seed) / "final_predictions_float.jsonl"
        large_path = paths["runs"] / "survface" / large_id / ("seed-%d" % seed) / "final_predictions_float.jsonl"
        small = read_jsonl(small_path)
        large = read_jsonl(large_path)
        if [row["record_id"] for row in small] != [row["record_id"] for row in large]:
            raise ValueError("candidate small and large roots do not use aligned final records")
        known = [i for i, row in enumerate(small) if row["ground_truth"] in KNOWN]
        small_correct = np.asarray([small[i]["predicted_decision"] == small[i]["ground_truth"] for i in known])
        large_correct = np.asarray([large[i]["predicted_decision"] == large[i]["ground_truth"] for i in known])
        unknown = [i for i, row in enumerate(small) if row["ground_truth"] == "UNKNOWN"]
        small_fpir = float(np.mean([small[i]["predicted_decision"] in KNOWN for i in unknown]))
        large_fpir = float(np.mean([large[i]["predicted_decision"] in KNOWN for i in unknown]))
        erroneous = ~small_correct
        comparisons.append({"root_seed": seed,
                            "known_decision_agreement": float(np.mean([small[i]["predicted_decision"] == large[i]["predicted_decision"] for i in known])),
                            "small_correct_acceptance": float(np.mean(small_correct)),
                            "large_correct_acceptance": float(np.mean(large_correct)),
                            "large_corrects_small_errors_count": int(np.sum(erroneous & large_correct)),
                            "large_corrects_small_errors_fraction_of_small_errors": float(np.sum(erroneous & large_correct) / max(1, np.sum(erroneous))),
                            "small_correct_large_wrong_count": int(np.sum(small_correct & ~large_correct)),
                            "unknown_fpir_small": small_fpir, "unknown_fpir_large": large_fpir,
                            "unknown_fpir_large_minus_small": large_fpir - small_fpir})
    return {"small_family": small_id, "large_family": large_id,
            "paired_final_comparison": comparisons,
            "mean_known_decision_agreement": float(np.mean([r["known_decision_agreement"] for r in comparisons])),
            "mean_large_corrects_small_errors_fraction": float(np.mean([r["large_corrects_small_errors_fraction_of_small_errors"] for r in comparisons])),
            "mean_unknown_fpir_large_minus_small": float(np.mean([r["unknown_fpir_large_minus_small"] for r in comparisons])),
            "interpretation": "paired on identical public final crops by registered root; root variability is not person/session variability"}


def _paired_arm_metrics(final_roots: list[dict], matrix: dict) -> dict:
    lookup = {(r["arm_id"], int(r["root_seed"])): r for r in final_roots}
    pair_specs = [
        ("a010_transfer_vs_scratch_6", "sv_mnv1_a010_transfer_6", "sv_mnv1_a010_scratch_6"),
        ("a025_transfer_vs_scratch_6", "sv_mnv1_a025_transfer_6", "sv_mnv1_a025_scratch_6"),
        ("a010_six_vs_five_oe", "sv_mnv1_a010_transfer_6", "sv_mnv1_a010_transfer_5_oe"),
        ("a025_six_vs_five_oe", "sv_mnv1_a025_transfer_6", "sv_mnv1_a025_transfer_5_oe"),
        ("a025_quality_vs_uniform", "sv_mnv1_a025_quality_select_6", "sv_mnv1_a025_uniform_select_6"),
    ]
    output = {}
    metric_keys = ["known_identity_macro_identification", "known_identity_macro_correct_acceptance",
                   "known_identity_macro_wrong_known_identity", "unknown_false_identification_rate",
                   "unknown_rejection_rate"]
    for name, left_id, right_id in pair_specs:
        per_root = []
        for seed in matrix["training_roots"]:
            left = lookup[(left_id, int(seed))]["float_metrics"]
            right = lookup[(right_id, int(seed))]["float_metrics"]
            per_root.append({"root_seed": int(seed), **{key: float(right[key] - left[key]) for key in metric_keys},
                             "left_arm": left_id, "right_arm": right_id})
        output[name] = {"per_root_right_minus_left": per_root,
                        "mean_paired_differences": {key: float(np.mean([r[key] for r in per_root])) for key in metric_keys}}
    return output


def _aggregate_records(rows: list[dict], metric_path: tuple[str, ...], group_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        value = row
        for key in metric_path:
            value = value[key]
        groups[row[group_key]].append(float(value))
    return {name: {"root_count": len(values), "mean": float(np.mean(values)),
                   "sd_across_registered_roots": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                   "min": float(np.min(values)), "max": float(np.max(values)), "per_root": values}
            for name, values in groups.items()}


def _gallery_pair_comparisons(gallery_final: list[dict]) -> dict:
    by_key = {(row["arm_id"], int(row["root_seed"]), row["policy"], row["backend"]): row
              for row in gallery_final}
    comparisons = []
    for (arm, seed, policy, backend), left in sorted(by_key.items()):
        if backend != "float":
            continue
        right = by_key.get((arm, seed, policy, "int8"))
        if not right:
            continue
        left_preds = read_jsonl(Path(left["predictions"]["path"]))
        right_preds = read_jsonl(Path(right["predictions"]["path"]))
        if [r["record_id"] for r in left_preds] != [r["record_id"] for r in right_preds]:
            raise ValueError("gallery float/int8 final rows are misaligned")
        comparisons.append({"arm_id": arm, "root_seed": seed, "policy": policy,
                            "float_template_count": left["selected_template_count"],
                            "int8_template_count": right["selected_template_count"],
                            "decision_disagreement_count": sum(a["predicted_decision"] != b["predicted_decision"]
                                                                for a, b in zip(left_preds, right_preds)),
                            "float_threshold": left["threshold"], "int8_threshold": right["threshold"],
                            "threshold_drift": float(right["threshold"] - left["threshold"]),
                            "float_fpir": left["metrics"]["unknown_false_identification_rate"],
                            "int8_fpir": right["metrics"]["unknown_false_identification_rate"]})
    return {"per_root_policy_comparisons": comparisons,
            "mean_decision_disagreement_count": float(np.mean([r["decision_disagreement_count"] for r in comparisons])) if comparisons else None,
            "mean_threshold_drift": float(np.mean([r["threshold_drift"] for r in comparisons])) if comparisons else None}


def _selection_manifest_summary(paths: dict) -> dict:
    index_path = paths["data"] / "derived/D_selection/INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    summaries = []
    for item in index["selections"]:
        value = json.loads((paths["study"].parent.parent / item["summary"]).read_text(encoding="utf-8"))
        summaries.append({"policy": item["policy"], "seed": item["seed"],
                          "selected_known_record_count": value["selected_known_record_count"],
                          "selected_unknown_record_count": value["selected_unknown_record_count"],
                          "mean_quality_by_person": {label: value["by_identity"][label]["selected_quality_mean"] for label in KNOWN},
                          "camera_strata": {label: value["by_identity"][label]["camera_strata"] for label in KNOWN},
                          "manifest_sha256": item["manifest_file_sha256"]})
    return {"index_path": str(index_path), "index_sha256": sha256_file(index_path), "selections": summaries}


def run_final_evaluations(workspace: Path, paths: dict) -> dict:
    matrix = _load_matrix(workspace, paths)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    existing_summary = paths["handoff"] / "FINAL_SUMMARY.json"
    if existing_summary.is_file():
        saved_summary = json.loads(existing_summary.read_text(encoding="utf-8"))
        if saved_summary.get("status") == "complete_verified":
            return saved_summary
    base_data = load_role_arrays(workspace, paths, "survface_96")
    gallery_results = run_gallery_campaign(workspace, paths, matrix, base_data)
    selection_path = paths["reports"] / "CANDIDATE_SELECTION.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "frozen_before_final_role" or selection.get("final_role_opened") is not False:
            raise ValueError("candidate selection was not frozen before final access")
    else:
        selection = development_candidate_selection(paths, matrix, gallery_results)
    registry_path = paths["reports"] / "MODEL_REGISTRY_FROZEN.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("status") != "complete_verified":
            raise ValueError("frozen model registry is invalid")
        if registry.get("candidate_selection_sha256") != sha256_file(selection_path):
            raise ValueError("candidate selection changed after model freeze")
        teacher_registry_path = paths["reports"] / "TEACHER_MODEL_REGISTRY_FROZEN.json"
        if teacher_registry_path.is_file() and registry.get("teacher_student_registry_sha256") != sha256_file(teacher_registry_path):
            raise ValueError("teacher/student registry changed after model freeze")
        historical_registry_path = paths["reports"] / "HISTORICAL_MODEL_REGISTRY_FROZEN.json"
        if historical_registry_path.is_file() and registry.get("historical_control_registry_sha256") != sha256_file(historical_registry_path):
            raise ValueError("historical control registry changed after model freeze")
    else:
        registry = freeze_model_registry(paths, matrix, selection, gallery_results)
    # Visual QA is restricted to the development and unknown-development roles.
    # It is diagnostic only and runs after all model/calibration selections freeze.
    from nicla_study.d_review import build_development_visual_review
    visual_review = build_development_visual_review(workspace, paths, selection, gallery_results, base_data)
    registry["development_visual_review_path"] = str(paths["reports"] / "DEVELOPMENT_VISUAL_REVIEW/SUMMARY.json")
    registry["development_visual_review_sha256"] = sha256_file(Path(registry["development_visual_review_path"]))
    write_json(registry_path, registry)
    final_gate_path = paths["reports"] / "FINAL_ROLE_OPENED.json"
    completed_classifier = list((paths["runs"] / "survface").glob("*/seed-*/FINAL_RESULT.json"))
    expected_classifier_count = len(matrix["survface_arms"]) * 10
    if len(completed_classifier) < expected_classifier_count:
        candidates = [row for row in registry["models"] if row["dataset"] == "survface_96"]
        unopened = next((row for row in candidates if not row.get("final_role_opened")), None)
        if unopened is None:
            raise PermissionError("no unconsumed frozen SurvFace receipt is available for final-role loading")
        receipt = json.loads(Path(unopened["freeze_receipt"]).read_text(encoding="utf-8"))
        final_known_rows = dataset_role_rows(workspace, paths["primary"], "survface_96", ("final",),
                                             "evaluate_final", receipt)
        final_unknown_rows = dataset_role_rows(workspace, paths["primary"], "survface_96", ("unknown_final",),
                                               "evaluate_final", receipt)
        final_data = {
            "known_rows": final_known_rows,
            "known_gray": cache_gray8(workspace, final_known_rows, "evaluate_final", receipt),
            "unknown_rows": final_unknown_rows,
            "unknown_gray": cache_gray8(workspace, final_unknown_rows, "evaluate_final", receipt),
        }
        open_receipt = {"schema": "job-d-final-role-opened-1", "status": "opened_after_freeze",
                        "opened_utc": utc_now(), "protocol_sha256": registry["protocol_sha256"],
                        "matrix_sha256": sha256_file(paths["matrix"]),
                        "candidate_selection_sha256": sha256_file(selection_path),
                        "model_registry_sha256_before_final": sha256_file(registry_path),
                        "teacher_student_registry_sha256_before_final": registry.get("teacher_student_registry_sha256"),
                        "historical_control_registry_sha256_before_final": registry.get("historical_control_registry_sha256"),
                        "known_role_image_hash_manifest_sha256": rows_fingerprint(final_known_rows),
                        "unknown_role_image_hash_manifest_sha256": rows_fingerprint(final_unknown_rows),
                        "known_rows": len(final_known_rows), "unknown_rows": len(final_unknown_rows),
                        "unknown_identity_count": len({row["identity_id"] for row in final_unknown_rows}),
                        "opened_by": "D final evaluation after classifier, gallery and calibration freeze"}
        write_json(final_gate_path, open_receipt)
        final_roots = []
        for model_record in registry["models"]:
            if model_record["dataset"] != "survface_96":
                continue
            root_result = json.loads(Path(model_record["result_path"]).read_text(encoding="utf-8"))
            final_roots.append(_classifier_final_for_root(paths, root_result, final_data, model_record))
    else:
        final_roots = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((paths["runs"] / "survface").glob("*/seed-*/FINAL_RESULT.json"))]

    completed_gallery_paths = list((paths["runs"] / "gallery").glob("a*/seed-*/*.FINAL.json"))
    expected_gallery_count = 2 * 10 * 2 * 2
    if len(completed_gallery_paths) < expected_gallery_count:
        # All classifier and gallery receipts were frozen together before the first final read.
        gallery_freeze_records = registry["galleries"]
        if "final_data" not in locals():
            opened = json.loads(final_gate_path.read_text(encoding="utf-8"))
            unopened = next((row for row in registry["models"] if row["dataset"] == "survface_96"
                             and not json.loads(Path(row["freeze_receipt"]).read_text(encoding="utf-8")).get("final_role_opened")), None)
            if unopened is None:
                unopened = next((row for row in registry["galleries"]
                                 if not json.loads(Path(row["freeze_receipt"]).read_text(encoding="utf-8")).get("final_role_opened")), None)
            if unopened is None:
                raise PermissionError("no frozen receipt remains for resumed final evaluation")
            receipt = json.loads(Path(unopened["freeze_receipt"]).read_text(encoding="utf-8"))
            final_known_rows = dataset_role_rows(workspace, paths["primary"], "survface_96", ("final",), "evaluate_final", receipt)
            final_unknown_rows = dataset_role_rows(workspace, paths["primary"], "survface_96", ("unknown_final",), "evaluate_final", receipt)
            final_data = {"known_rows": final_known_rows,
                          "known_gray": cache_gray8(workspace, final_known_rows, "evaluate_final", receipt),
                          "unknown_rows": final_unknown_rows,
                          "unknown_gray": cache_gray8(workspace, final_unknown_rows, "evaluate_final", receipt)}
        train_known_rows = [row for row in base_data["train_rows"]
                            if row.get("identity_partition") != "unknown" and row.get("label_id") in KNOWN]
        train_known_indices = [i for i, row in enumerate(base_data["train_rows"])
                               if row.get("identity_partition") != "unknown" and row.get("label_id") in KNOWN]
        train_known_gray = base_data["train_gray"][train_known_indices]
        for gallery_record in gallery_freeze_records:
            source_path = Path(gallery_record["result_path"])
            backend, policy = json.loads(source_path.read_text(encoding="utf-8"))["backend"], json.loads(source_path.read_text(encoding="utf-8"))["policy"]
            final_path = source_path.parent / ("%s_%s.FINAL.json" % (backend, policy))
            if final_path.is_file() and json.loads(final_path.read_text(encoding="utf-8")).get("status") == "complete_verified":
                continue
            result = json.loads(source_path.read_text(encoding="utf-8"))
            result["result_path"] = str(source_path)
            gallery_final_for_result(paths, result, final_data, train_known_rows, train_known_gray)
    gallery_final = []
    for gallery_final_path in sorted((paths["runs"] / "gallery").glob("a*/seed-*/*.FINAL.json")):
        result = json.loads(gallery_final_path.read_text(encoding="utf-8"))
        if "width" not in result:
            result["width"] = 0.1 if "_a010_" in result["arm_id"] else 0.25
            write_json(gallery_final_path, result)
        gallery_final.append(result)

    olivetti_result_paths = sorted((paths["runs"] / "olivetti").glob("*/seed-*/ROOT_RESULT.json"))
    expected_olivetti_roots = len(matrix["olivetti_arms"]) * 10
    completed_olivetti = list((paths["runs"] / "olivetti").glob("*/seed-*/FINAL_RESULT.json"))
    if len(completed_olivetti) < expected_olivetti_roots:
        if "ol_final_data" not in locals():
            ol_unopened = next((row for row in registry["models"] if row["dataset"] == "olivetti_96"
                                and not json.loads(Path(row["freeze_receipt"]).read_text(encoding="utf-8")).get("final_role_opened")), None)
            if ol_unopened is None:
                raise PermissionError("no unconsumed frozen Olivetti receipt is available for final-role loading")
            ol_receipt = json.loads(Path(ol_unopened["freeze_receipt"]).read_text(encoding="utf-8"))
            ol_rows = dataset_role_rows(workspace, paths["olivetti"], "olivetti_96", ("final",), "evaluate_final", ol_receipt)
            ol_final_data = {"rows": ol_rows, "gray": cache_gray8(workspace, ol_rows, "evaluate_final", ol_receipt)}
        olivetti_final = []
        for result_path in olivetti_result_paths:
            root_result = json.loads(result_path.read_text(encoding="utf-8"))
            olivetti_final.append(olivetti_final_for_root(root_result, ol_final_data))
    else:
        olivetti_final = [json.loads(path.read_text(encoding="utf-8"))
                          for path in sorted((paths["runs"] / "olivetti").glob("*/seed-*/FINAL_RESULT.json"))]

    classifier_aggregate = {}
    for arm in matrix["survface_arms"]:
        arm_rows = [row for row in final_roots if row["arm_id"] == arm["id"]]
        classifier_aggregate[arm["id"]] = {}
        for backend in ("float_metrics", "int8_metrics"):
            keys = ["known_identity_macro_identification", "known_identity_macro_correct_acceptance",
                    "known_identity_macro_wrong_known_identity", "known_identity_macro_rejection",
                    "known_identity_macro_coverage", "unknown_false_identification_rate",
                    "unknown_rejection_rate", "other_accuracy"]
            classifier_aggregate[arm["id"]][backend] = {}
            for key in keys:
                values = [r[backend][key] for r in arm_rows if r[backend].get(key) is not None]
                classifier_aggregate[arm["id"]][backend][key] = {
                    "mean_across_roots": float(np.mean(values)) if values else None,
                    "sd_across_roots": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "root_count": len(values),
                    "per_root": [r[backend].get(key) for r in arm_rows]}
    olivetti_aggregate = {}
    for arm in matrix["olivetti_arms"]:
        arm_rows = [row for row in olivetti_final if row["arm_id"] == arm["id"]]
        olivetti_aggregate[arm["id"]] = {
            "float_macro_identity_accuracy_mean": float(np.mean([r["float_metrics"]["macro_identity_accuracy"] for r in arm_rows])),
            "float_macro_identity_accuracy_sd_across_roots": float(np.std([r["float_metrics"]["macro_identity_accuracy"] for r in arm_rows], ddof=1)),
            "int8_macro_identity_accuracy_mean": float(np.mean([r["int8_metrics"]["macro_identity_accuracy"] for r in arm_rows])),
            "per_root_float": [r["float_metrics"]["macro_identity_accuracy"] for r in arm_rows],
            "open_set_claim": False}
    gallery_aggregate = {}
    for result in gallery_final:
        key = "a%s_%s_%s" % ("010" if result["width"] == 0.1 else "025", result["backend"], result["policy"])
        gallery_aggregate.setdefault(key, []).append(result)
    gallery_aggregate = {key: {"root_count": len(rows),
                               "known_macro_correct_acceptance_mean": float(np.mean([r["metrics"]["known_macro_correct_acceptance"] for r in rows])),
                               "known_macro_identification_mean": float(np.mean([r["metrics"]["known_macro_identification"] for r in rows])),
                               "unknown_fpir_mean": float(np.mean([r["metrics"]["unknown_false_identification_rate"] for r in rows])),
                               "unknown_rejection_mean": float(np.mean([r["metrics"]["unknown_rejection_rate"] for r in rows])),
                               "selected_template_counts": [r["selected_template_count"] for r in rows]}
                         for key, rows in gallery_aggregate.items()}
    final_selection_comparison = _paired_classifier_comparison(paths, selection, matrix)
    paired_arms = _paired_arm_metrics(final_roots, matrix)
    gallery_backend_comparison = _gallery_pair_comparisons(gallery_final)
    registry["final_role_opened"] = True
    registry["final_role_opened_utc"] = utc_now()
    registry["final_results"] = {"survface_root_count": len(final_roots),
                                  "olivetti_root_count": len(olivetti_final),
                                  "gallery_root_count": len(gallery_final),
                                  "final_role_receipt": str(final_gate_path)}
    write_json(registry_path, registry)
    summary = {
        "schema": "job-d-final-summary-1", "status": "complete_verified",
        "job_id": "D", "protocol_id": matrix["protocol_id"], "matrix_id": matrix["matrix_id"],
        "public_branch": "frozen five-known K01-K05; no personal roster",
        "primary_dataset": "SurvFace-derived 96x96 gray crops; exploratory because K01-K05 appeared in v3 development",
        "secondary_dataset": "Olivetti 40-way closed-set sanity; separate and never pooled with SurvFace",
        "training_root_count": len(matrix["training_roots"]), "pilot_root_count": 10,
        "full_survface_training_count": expected_classifier_count,
        "full_olivetti_training_count": expected_olivetti_roots,
        "gallery_root_backend_policy_count": len(gallery_final),
        "candidate_selection": selection,
        "development_selected_candidates": {
            "small": selection["candidate_small_family"], "large": selection["candidate_large_family"],
            "small_best_host_route": selection.get("candidate_small_best_host_route"),
            "large_best_host_route": selection.get("candidate_large_best_host_route"),
            "strong_reference": selection["strong_reference"],
            "best_gallery": selection["best_embedding_gallery_family"]},
        "survface_final_by_arm": classifier_aggregate,
        "olivetti_final_by_arm": olivetti_aggregate,
        "gallery_final_by_family": gallery_aggregate,
        "small_large_paired_final_error_correction": final_selection_comparison,
        "paired_initialization_and_head_comparisons": paired_arms,
        "gallery_float_int8_disagreements_and_calibration_drift": gallery_backend_comparison,
        "quality_control_selection_evidence": _selection_manifest_summary(paths),
        "model_registry_path": str(registry_path), "model_registry_sha256": sha256_file(registry_path),
        "teacher_student_registry_path": registry.get("teacher_student_registry"),
        "teacher_student_registry_sha256": registry.get("teacher_student_registry_sha256"),
        "teacher_student_root_count": max(0, len(registry.get("teacher_student_models", [])) - 1),
        "historical_control_registry_path": registry.get("historical_control_registry"),
        "historical_control_registry_sha256": registry.get("historical_control_registry_sha256"),
        "historical_control_root_count": len(registry.get("historical_control_models", [])),
        "historical_control_gallery_count": len(registry.get("historical_control_galleries", [])),
        "development_visual_review_path": registry.get("development_visual_review_path"),
        "final_role_opened_receipt": str(final_gate_path), "final_role_opened": True,
        "deployment_boundary": {"host_reference_available": True, "full_int8_tflite_available": True,
                                "board_qualified": False, "nicla_observed": False,
                                "camera_pipeline_accuracy_measured": False,
                                "target_arena_latency_power_and_operator_support_pending": True},
        "v3_comparison": {
            "historical_encoders": "v3 custom_base and custom_tiny remain non-face-pretrained historical representation controls; they were trained for a broad identity-verification task, not five-person identity classification.",
            "known_exposure": "K01-K05 public identities appeared in v3 development, so v4 final results are exploratory and do not establish untouched external-person generalization.",
            "measured_improvement_claim": "D adds a ten-root, target-supervised five-person classifier and embedding/gallery comparison with verified transfer initialization, role-gated open-set calibration, paired full-int8 inference, and a separate Olivetti sanity arm. A direct accuracy gain over v3 is only claimed where a same-crop, same-role historical prediction is audited and available.",
        },
        "created_utc": utc_now(),
    }
    write_json(existing_summary, summary)
    return summary


def run_d_prep(workspace: Path) -> dict:
    paths = study_paths(workspace)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    paths["handoff"].mkdir(parents=True, exist_ok=True)
    a_path = paths["handoff"].parent / "A/RESULT.json"
    b_path = paths["handoff"].parent / "B/RESULT.json"
    # C is maintained as a separate but linked Mac lane handoff; use its actual crop contract.
    c_path = paths["handoff"].parent / "C/ORIGINAL_RESULT.json"
    if not a_path.is_file() or not b_path.is_file() or not c_path.is_file():
        raise FileNotFoundError("A, B, and C handoff receipts are all required before D preparation")
    a, b, c = (json.loads(path.read_text(encoding="utf-8")) for path in (a_path, b_path, c_path))
    if a.get("status") != "complete_verified" or b.get("status") != "complete_verified":
        raise ValueError("A and B must be complete_verified")
    if c.get("status") not in ("complete_verified", "complete_verified_with_physical_cells_pending"):
        raise ValueError("C crop contract is not verified")
    crop = c.get("transforms_and_crop", {})
    crop_input = crop.get("detector_input", {})
    c_input = c.get("integration_contract", {}).get("C_to_F_boundary", {}).get("D_input_if_current_embedding_is_used", "")
    if crop_input.get("shape") != [1, 96, 96, 1] or crop_input.get("format") != "gray8" or crop_input.get("normalization") != "u8/255":
        raise ValueError("C crop shape/format/normalization contract does not match D")
    if "scale 1/255" not in c_input or "zero point -128" not in c_input:
        raise ValueError("C's int8 input contract must be gray8 scale 1/255, zero point -128")
    matrix = _load_matrix(workspace, paths)
    primary = read_jsonl(paths["primary"])
    olivetti = read_jsonl(paths["olivetti"])
    primary_validation = validate_manifest(primary)
    olivetti_validation = validate_manifest(olivetti)
    if primary_validation["status"] != "pass" or olivetti_validation["status"] != "pass":
        raise ValueError("B primary or Olivetti role manifest did not validate")
    checkpoint_audit = checkpoint_reference_audit(workspace)
    write_json(paths["reports"] / "pretrained_reference_audit.json", checkpoint_audit)
    selection_index = write_d_selection_manifests(workspace, paths)
    # Reload every D-owned manifest and verify role access and source-hash fields before training.
    selection_checks = []
    selection_root = paths["data"] / "derived/D_selection"
    for item in selection_index["selections"]:
        path = workspace / item["manifest"]
        rows = read_jsonl(path)
        assert_role_access(rows, "train")
        if len([row for row in rows if row.get("label_id") in KNOWN]) != 300:
            raise ValueError("each D selection must have 60 known rows/person")
        unknown = [row for row in rows if row.get("identity_partition") == "unknown"]
        if len(unknown) != 300:
            raise ValueError("each D selection must reuse the same 300 unknown_train images")
        if len({row["record_id"] for row in rows}) != len(rows):
            raise ValueError("D selection contains repeated source records")
        selection_checks.append({"policy": item["policy"], "seed": item["seed"],
                                 "manifest_sha256": sha256_file(path), "row_count": len(rows),
                                 "known_rows": 300, "unknown_rows": 300})
    try:
        from nicla_study.inventory import fresh_device_discovery
        discovery = fresh_device_discovery()
    except Exception as exc:
        discovery = {"status": "unavailable", "error": repr(exc)}
    write_json(paths["reports"] / "D-fresh-device-discovery.json", discovery)
    checks = [
        {"id": "A_B_prerequisites", "status": "pass", "detail": "A and B handoffs are complete_verified.",
         "evidence": [str(a_path), str(b_path)]},
        {"id": "C_gray8_crop_contract", "status": "pass", "detail": "C provides gray8 96x96x1, u8/255, with int8 scale 1/255 and zero point -128.",
         "evidence": [str(c_path), str(paths["study"] / "config/contracts/v1/image-crop.schema.json")]},
        {"id": "real_pretrained_checkpoint_architecture", "status": "pass", "detail": "Both downloaded EI checkpoints loaded strictly into reference-compatible Keras MobileNetV1 and produced finite 1000-way and backbone outputs at 96x96 grayscale.",
         "evidence": [str(paths["reports"] / "pretrained_reference_audit.json")]},
        {"id": "frozen_finite_matrix_and_roots", "status": "pass", "detail": "The exact D matrix is frozen and declares ten registered training roots for each required arm.",
         "evidence": [str(paths["matrix"])]},
        {"id": "role_gated_data_and_selection_controls", "status": "pass", "detail": "Primary/Olivetti manifests validate and all twenty D-derived selection manifests have 300 known plus the same 300 unknown training images.",
         "evidence": [str(paths["primary"]), str(paths["olivetti"]), str(selection_root / "INDEX.json")]},
        {"id": "Mac_training_host_and_device_probe", "status": "pass", "detail": "Training remains on the Mac host. Fresh discovery was recorded; if no Nicla is observed, independent host training proceeds while physical qualification remains pending.",
         "evidence": [str(paths["reports"] / "D-fresh-device-discovery.json")]},
    ]
    prep = {"schema": "study-job-result-1", "job_id": "D_prep_mac", "status": "complete_verified",
            "created_utc": utc_now(), "protocol_id": matrix["protocol_id"],
            "platform": "Mac host only; training will not move to Windows",
            "prerequisites": {"A": {"path": str(a_path), "sha256": sha256_file(a_path)},
                              "B": {"path": str(b_path), "sha256": sha256_file(b_path)},
                              "C": {"path": str(c_path), "sha256": sha256_file(c_path),
                                    "status": c["status"], "crop_contract": crop_input,
                                    "identity_boundary": c.get("integration_contract", {}).get("C_to_F_boundary", {})}},
            "matrix": {"path": str(paths["matrix"]), "sha256": sha256_file(paths["matrix"]),
                       "matrix_id": matrix["matrix_id"], "survface_arms": len(matrix["survface_arms"]),
                       "olivetti_arms": len(matrix["olivetti_arms"]), "roots": matrix["training_roots"]},
            "pretrained_reference_audit": {"path": str(paths["reports"] / "pretrained_reference_audit.json"),
                                            "sha256": sha256_file(paths["reports"] / "pretrained_reference_audit.json"),
                                            "checkpoints": checkpoint_audit["checkpoints"]},
            "primary_manifest_validation": primary_validation, "olivetti_manifest_validation": olivetti_validation,
            "selection_index": {"path": str(selection_root / "INDEX.json"), "sha256": sha256_file(selection_root / "INDEX.json"),
                                "selections": selection_checks},
            "device_discovery": discovery,
            "acceptance_checks": checks,
            "evidence_paths": {"matrix": str(paths["matrix"]),
                               "pretrained_reference_audit": str(paths["reports"] / "pretrained_reference_audit.json"),
                               "selection_index": str(selection_root / "INDEX.json"),
                               "device_discovery": str(paths["reports"] / "D-fresh-device-discovery.json")},
            "notes": ["The EI H5 1000-output head is retained only for reference forward verification and removed from task graphs.",
                      "Published EI checkpoint metadata does not declare exact pretraining corpus/licensing; no corpus is inferred.",
                      "C physical arena cells remain pending and are outside the host training dependency."]}
    write_json(paths["reports"] / "PREP_RESULT.json", prep)
    return prep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Mac-only Job D model preparation, training and final evaluation.")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("action", choices=("prep", "train", "finish"))
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    paths = study_paths(workspace)
    if args.action == "prep":
        result = run_d_prep(workspace)
        print(json.dumps({"status": result["status"], "receipt": str(paths["reports"] / "PREP_RESULT.json"),
                          "sha256": sha256_file(paths["reports"] / "PREP_RESULT.json")}, indent=2))
        return 0
    if args.action == "train":
        result = run_training_campaign(workspace, paths)
        print(json.dumps({"status": result["status"], "campaign": str(paths["runs"] / "TRAINING_CAMPAIGN.json"),
                          "full_classifier_training_count": result["full_classifier_training_count"]}, indent=2))
        return 0
    result = run_final_evaluations(workspace, paths)
    print(json.dumps({"status": result["status"], "summary": str(paths["handoff"] / "FINAL_SUMMARY.json"),
                      "survface_roots": result["full_survface_training_count"],
                      "olivetti_roots": result["full_olivetti_training_count"]}, indent=2))
    return 0


def atomic_save_weights(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.h5")
    if tmp.exists():
        tmp.unlink()
    model.save_weights(str(tmp))
    os.replace(tmp, path)


def load_weights_exact(workspace: Path, alpha: float, pretrained: bool, outputs: int,
                       seed: int, name: str, weight_path: Path) -> tuple[tf.keras.Model, tf.keras.Model, dict]:
    model, embedding, info = build_classifier(workspace, alpha, pretrained, outputs, seed, name)
    try:
        model.load_weights(str(weight_path))
        info["checkpoint_reload"] = {"status": "verified", "method": "keras_hdf5_topological"}
    except ValueError as exc:
        # Keras 2.10's legacy HDF5 loader can treat the tracked shared embedding
        # submodel as an old nested model and transpose its 4-D kernels with an
        # incompatible layout. The saved weights are named and shape-identical;
        # load every tensor directly and fail closed on any missing, mismatched,
        # or ambiguous value instead of skipping weights.
        loaded_paths = []
        with h5py.File(weight_path, "r") as source:
            datasets = {}
            def collect(name, item):
                if isinstance(item, h5py.Dataset):
                    datasets[name] = np.asarray(item)
            source.visititems(collect)
            for variable in model.weights:
                matches = [(path, value) for path, value in datasets.items()
                           if path.endswith(variable.name)]
                if not matches:
                    raise ValueError("host checkpoint is missing tensor %s" % variable.name) from exc
                expected_shape = tuple(int(value) for value in variable.shape)
                if any(tuple(value.shape) != expected_shape for _, value in matches):
                    raise ValueError("host checkpoint tensor shape mismatch for %s" % variable.name) from exc
                reference = matches[0][1]
                if any(not np.array_equal(reference, value) for _, value in matches[1:]):
                    raise ValueError("host checkpoint has conflicting named copies of %s" % variable.name) from exc
                variable.assign(reference)
                loaded_paths.append(matches[0][0])
        if len(loaded_paths) != len(model.weights) or len(set(loaded_paths)) != len(model.weights):
            raise ValueError("host checkpoint tensor assignment was incomplete or duplicated") from exc
        info["checkpoint_reload"] = {
            "status": "verified", "method": "exact_named_hdf5_tensor_assignment",
            "tensor_count": len(loaded_paths), "native_loader_error": str(exc),
        }
    return model, embedding, info


def convert_float_tflite(model: tf.keras.Model, path: Path) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_bytes = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model_bytes)
    audit = audit_tflite(path, require_int8=False)
    audit.update({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return audit


def convert_int8_tflite(model: tf.keras.Model, train_gray: np.ndarray, path: Path,
                        representative_indices: list[int]) -> dict:
    if not representative_indices:
        raise ValueError("empty INT8 representative set")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data():
        for index in representative_indices:
            yield [gray_float(train_gray[index:index + 1])]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model_bytes)
    audit = audit_tflite(path, require_int8=True)
    expected_scale = 1.0 / 255.0
    actual_input = audit["input"]
    if actual_input["zero_point"] != -128 or abs(float(actual_input["scale"]) - expected_scale) > 1e-7:
        raise ValueError("full-int8 D input does not match C crop contract gray8-128: %r" % actual_input)
    audit.update({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
                  "representative_source": "SurvFace/Olivetti train role only",
                  "representative_sample_count": len(representative_indices)})
    return audit


def audit_tflite(path: Path, require_int8: bool = True) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    ops = interpreter._get_ops_details()
    op_names = [str(op.get("op_name")) for op in ops]
    all_tensors = interpreter.get_tensor_details()
    tensor_types = {str(item["dtype"]) for item in all_tensors}
    if len(inputs) != 1 or inputs[0]["shape"].tolist() != [1, 96, 96, 1]:
        raise ValueError("TFLite input contract mismatch: %s" % [item["shape"].tolist() for item in inputs])
    if require_int8 and np.int8 not in [item["dtype"] for item in inputs + outputs]:
        raise ValueError("TFLite full-int8 conversion did not yield int8 I/O")
    if not require_int8 and (inputs[0]["dtype"] != np.float32 or outputs[0]["dtype"] != np.float32):
        raise ValueError("float TFLite export did not preserve float32 I/O")
    def qinfo(item):
        scales = item["quantization_parameters"].get("scales", [])
        zeros = item["quantization_parameters"].get("zero_points", [])
        return {"scale": float(scales[0]) if len(scales) else None,
                "zero_point": int(zeros[0]) if len(zeros) else None}
    return {
        "input": {"shape": inputs[0]["shape"].tolist(), "dtype": str(inputs[0]["dtype"]), **qinfo(inputs[0])},
        "output": {"shape": outputs[0]["shape"].tolist(), "dtype": str(outputs[0]["dtype"]), **qinfo(outputs[0])},
        "op_names": op_names, "operator_count": len(op_names), "tensor_types": sorted(tensor_types),
        "tensor_dtype_counts": dict(Counter(str(item["dtype"]) for item in all_tensors)),
        "internal_float_tensor_count": sum(item["dtype"] in (np.float16, np.float32, np.float64) for item in all_tensors),
        "all_int8_boundary": inputs[0]["dtype"] == np.int8 and outputs[0]["dtype"] == np.int8,
        "full_int8_requested": bool(require_int8),
    }


def tflite_logits(path: Path, gray_images: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    idetail = interpreter.get_input_details()[0]
    odetail = interpreter.get_output_details()[0]
    scale_i, zero_i = idetail["quantization"]
    scale_o, zero_o = odetail["quantization"]
    if not scale_i or not scale_o:
        raise ValueError("missing TFLite quantization parameters")
    results = []
    for i in range(len(gray_images)):
        x = gray_float(gray_images[i:i + 1])
        q = np.clip(np.rint(x / scale_i + zero_i), -128, 127).astype(np.int8)
        interpreter.set_tensor(idetail["index"], q)
        interpreter.invoke()
        qout = interpreter.get_tensor(odetail["index"]).astype(np.float32)
        results.append((qout - float(zero_o)) * float(scale_o))
    return np.concatenate(results, axis=0)

# Campaign orchestration is appended below.


def _read_primary_split(paths: dict) -> list[dict]:
    return read_jsonl(paths["primary"])


def _popcount64(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            prior = self.parent[value]
            self.parent[value] = root
            value = prior
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


_D_SELECTION_POOL_CACHE = {}


def _selection_pool(paths: dict) -> dict:
    cache_key = str(paths["study"])
    if cache_key in _D_SELECTION_POOL_CACHE:
        return _D_SELECTION_POOL_CACHE[cache_key]
    primary = _read_primary_split(paths)
    src_to_label = {}
    known_train = [r for r in primary if r.get("dataset") == "survface_96" and r.get("role") == "train"]
    unknown_train = [r for r in primary if r.get("dataset") == "survface_96" and r.get("role") == "unknown_train"]
    for row in known_train:
        src_to_label[str(row["source_identity_id"])] = (str(row["label_id"]), str(row["identity_id"]))
    wanted_source_ids = set(src_to_label)
    target_counts = defaultdict(Counter)
    for row in known_train:
        target_counts[row["label_id"]][str(row.get("camera_source_group") or "unknown")] += 1
    heldout_records = {str(r["record_id"]) for r in primary
                       if r.get("dataset") == "survface_96" and r.get("role") in {
                           "stopping", "threshold_calibration", "development", "final"}}
    presentation = read_jsonl(paths["presentation"])
    heldout_records.update(str(r["record_id"]) for r in presentation
                           if r.get("dataset") == "survface_96")
    permitted = {}
    for row in read_jsonl(paths["source"]):
        if (row.get("dataset") == "survface_96" and row.get("access_state") == "reverified_this_job"
                and str(row.get("source_identity_id")) in wanted_source_ids):
            permitted[str(row["record_id"])] = row
    quality_rows = {}
    for row in read_jsonl(paths["quality"]):
        rid = str(row.get("record_id"))
        if row.get("dataset") == "survface_96" and rid in permitted:
            quality_rows[rid] = row
    phash = {}
    for row in read_jsonl(paths["phashes"]):
        rid = str(row.get("record_id"))
        if rid in quality_rows:
            try:
                phash[rid] = int(str(row["phash64_hex"]), 16)
            except (KeyError, ValueError):
                continue
    if len(quality_rows) < 600 or len(phash) != len(quality_rows):
        raise ValueError("D candidate pool is missing reverified rows or permitted pHashes: %d/%d" %
                         (len(phash), len(quality_rows)))
    heldout_components = set(heldout_records)
    duplicate_to_records = defaultdict(list)
    for row in primary:
        if row.get("dataset") == "survface_96" and str(row.get("source_identity_id")) in wanted_source_ids:
            if row.get("duplicate_group_id"):
                duplicate_to_records[str(row["duplicate_group_id"])].append(str(row["record_id"]))
    for record_ids in duplicate_to_records.values():
        if any(rid in heldout_records for rid in record_ids):
            heldout_components.update(record_ids)
    record_to_source = {rid: str(row.get("source_identity_id")) for rid, row in permitted.items()}
    with Path(paths["cross_phash"]).open(encoding="utf-8") as stream:
        for line in stream:
            edge = json.loads(line)
            left, right = str(edge.get("left_record_id")), str(edge.get("right_record_id"))
            if left in heldout_records and record_to_source.get(right) in wanted_source_ids:
                heldout_components.add(right)
            if right in heldout_records and record_to_source.get(left) in wanted_source_ids:
                heldout_components.add(left)
    ids_by_source = defaultdict(list)
    for rid in quality_rows:
        source_id = str(permitted[rid]["source_identity_id"])
        if rid in phash:
            ids_by_source[source_id].append(rid)
    union = _UnionFind(quality_rows)
    for source_id, record_ids in ids_by_source.items():
        for i, left in enumerate(record_ids):
            for right in record_ids[i + 1:]:
                if _popcount64(phash[left], phash[right]) <= 6:
                    union.union(left, right)
    for group_records in duplicate_to_records.values():
        existing = [rid for rid in group_records if rid in quality_rows]
        for rid in existing[1:]:
            union.union(existing[0], rid)
    groups = defaultdict(list)
    for rid in quality_rows:
        groups[union.find(rid)].append(rid)
    blocked_roots = {union.find(rid) for rid in heldout_components if rid in union.parent}
    pool = {"primary": primary, "src_to_label": src_to_label, "known_train": known_train,
            "unknown_train": unknown_train, "wanted_source_ids": wanted_source_ids,
            "target_counts": target_counts, "heldout_records": heldout_records,
            "permitted": permitted, "quality_rows": quality_rows, "phash": phash,
            "duplicate_to_records": duplicate_to_records, "groups": groups,
            "blocked_roots": blocked_roots,
            "source_inventory_sha256": sha256_file(paths["source"]),
            "quality_inventory_sha256": sha256_file(paths["quality"]),
            "phash_inventory_sha256": sha256_file(paths["phashes"]),
            "cross_identity_edges_sha256": sha256_file(paths["cross_phash"])}
    _D_SELECTION_POOL_CACHE[cache_key] = pool
    return pool


def make_selection_candidates(workspace: Path, paths: dict, seed: int, policy: str) -> tuple[list[dict], dict]:
    """Create a D-only group-safe, matched-budget known training selection.

    Candidate membership comes only from B's reverified labeled quality inventory
    and permitted pHashes. All primary held-out and presentation groups are closed
    before any file pixels are loaded.
    """
    if policy not in {"quality", "uniform"}:
        raise ValueError("policy must be quality or uniform")
    pool = _selection_pool(paths)
    primary, src_to_label = pool["primary"], pool["src_to_label"]
    known_train, unknown_train = pool["known_train"], pool["unknown_train"]
    wanted_source_ids, target_counts = pool["wanted_source_ids"], pool["target_counts"]
    permitted, quality_rows, phash = pool["permitted"], pool["quality_rows"], pool["phash"]
    duplicate_to_records, groups, blocked_roots = pool["duplicate_to_records"], pool["groups"], pool["blocked_roots"]

    rng = np.random.default_rng(int(seed))
    chosen_rows = []
    summary = {"policy": policy, "seed": int(seed), "by_identity": {},
               "candidate_record_count": len(quality_rows), "permitted_phash_count": len(phash),
               "component_count_before_holdout_filter": len(groups),
               "blocked_component_count": len(blocked_roots),
               "source_inventory_sha256": pool["source_inventory_sha256"],
               "quality_inventory_sha256": pool["quality_inventory_sha256"],
               "phash_inventory_sha256": pool["phash_inventory_sha256"],
               "cross_identity_edges_sha256": pool["cross_identity_edges_sha256"]}
    for label in KNOWN:
        source_id = next(source for source, value in src_to_label.items() if value[0] == label)
        candidates_by_camera = defaultdict(list)
        for root, member_ids in groups.items():
            if root in blocked_roots:
                continue
            members = [rid for rid in member_ids if str(permitted[rid].get("source_identity_id")) == source_id]
            if not members:
                continue
            # Quality policy uses the most informative verified member as group representative.
            if policy == "quality":
                best = max(members, key=lambda rid: (float(quality_rows[rid].get("quality_score") or 0), rid))
            else:
                best = str(rng.choice(members))
            camera = str(quality_rows[best].get("camera_source_group") or "unknown")
            candidates_by_camera[camera].append({"root": root, "record_id": best,
                                                 "members": members,
                                                 "quality": float(quality_rows[best].get("quality_score") or 0)})
        selected = []
        strata_summary = {}
        wanted_cameras = target_counts[label]
        for camera, count in sorted(wanted_cameras.items()):
            candidates = candidates_by_camera.get(camera, [])
            if len(candidates) < count:
                raise ValueError("insufficient group-safe candidates for %s/%s: %d < %d" %
                                 (label, camera, len(candidates), count))
            if policy == "quality":
                weights = np.asarray([max(0.001, c["quality"]) for c in candidates], dtype=np.float64)
                weights /= weights.sum()
                indices = rng.choice(len(candidates), size=count, replace=False, p=weights)
            else:
                indices = rng.choice(len(candidates), size=count, replace=False)
            selected.extend(candidates[int(i)] for i in indices)
            strata_summary[camera] = {"requested": count, "available": len(candidates), "selected": count}
        labels_seen = set()
        for item in selected:
            rid = item["record_id"]
            source = permitted[rid]
            q = quality_rows[rid]
            label_id, identity_id = src_to_label[source_id]
            row = dict(source)
            row.update({k: q[k] for k in ("quality_score", "quality_valid", "gray_mean_u8", "gray_std_u8",
                                          "laplacian_variance") if k in q})
            row.update({"dataset": "survface_96", "role": "train", "identity_partition": "known",
                        "label_id": label_id, "class_index": KNOWN.index(label_id),
                        "identity_id": identity_id, "source_identity_id": source_id,
                        "source_sha256": source["sha256"], "sha256": source["sha256"],
                        "camera_source_group": q.get("camera_source_group"),
                        "d_group_component": str(item["root"]), "d_selection_policy": policy,
                        "d_selection_seed": int(seed)})
            chosen_rows.append(row)
            labels_seen.add(rid)
        summary["by_identity"][label] = {
            "source_identity_id": source_id, "selected_groups": len(selected),
            "selected_records": sorted(labels_seen),
            "camera_strata": strata_summary,
            "selected_quality_mean": float(np.mean([i["quality"] for i in selected])),
            "selected_quality_min": float(np.min([i["quality"] for i in selected])),
            "selected_quality_max": float(np.max([i["quality"] for i in selected])),
            "selected_quality_valid_fraction": float(np.mean([bool(quality_rows[i["record_id"]].get("quality_valid"))
                                                               for i in selected])),
        }
    # Unknown information is identical by construction in both paired arms.
    chosen_rows.extend(unknown_train)
    chosen_rows.sort(key=lambda row: (str(row.get("label_id")), str(row.get("identity_id")), str(row.get("record_id"))))
    summary["unknown_train_record_ids_sha256"] = sha256_json(sorted(r["record_id"] for r in unknown_train))
    summary["manifest_sha256"] = sha256_json([{k: row.get(k) for k in (
        "record_id", "role", "dataset", "label_id", "identity_id", "sha256", "relative_path",
        "d_group_component", "d_selection_policy", "d_selection_seed")} for row in chosen_rows])
    summary["selected_known_record_count"] = len(chosen_rows) - len(unknown_train)
    summary["selected_unknown_record_count"] = len(unknown_train)
    return chosen_rows, summary


def write_d_selection_manifests(workspace: Path, paths: dict) -> dict:
    out_root = paths["data"] / "derived/D_selection"
    out_root.mkdir(parents=True, exist_ok=True)
    registry = []
    seeds = [20261101, 20261102, 20261103, 20261104, 20261105,
             20261106, 20261107, 20261108, 20261109, 20261110]
    for seed in seeds:
        for policy in ("quality", "uniform"):
            stem = "%s_seed-%d" % (policy, seed)
            manifest_path = out_root / (stem + ".jsonl")
            summary_path = out_root / (stem + ".summary.json")
            if manifest_path.is_file() and summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                rows, summary = make_selection_candidates(workspace, paths, seed, policy)
                with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                write_json(summary_path, summary)
            registry.append({"policy": policy, "seed": seed,
                             "manifest": manifest_path.relative_to(workspace).as_posix(),
                             "manifest_file_sha256": sha256_file(manifest_path),
                             "summary": summary_path.relative_to(workspace).as_posix(),
                             "summary_file_sha256": sha256_file(summary_path),
                             "manifest_content_sha256": summary["manifest_sha256"]})
            write_json(out_root / "PREP_PROGRESS.json", {"status": "building_D_candidate_selection",
                                                          "completed_selection_count": len(registry),
                                                          "expected_selection_count": len(seeds) * 2,
                                                          "last_policy": policy, "last_seed": seed,
                                                          "updated_utc": utc_now()})
    index = {"schema": "job-d-selection-index-1", "mat": "D-five-person-transfer-open-set-gallery-v1",
             "created_utc": utc_now(), "immutable_B_inputs": True, "selections": registry}
    write_json(out_root / "INDEX.json", index)
    index["index_sha256"] = sha256_file(out_root / "INDEX.json")
    return index


if __name__ == "__main__":
    raise SystemExit(main())

