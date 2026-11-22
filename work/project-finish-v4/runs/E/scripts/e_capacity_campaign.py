#!/usr/bin/env python3
"""Train the predeclared wider capacity arms from matched D roots, resumably."""
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
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(RUN_ROOT / "scripts"))

import numpy as np
import tensorflow as tf

from nicla_study import d_campaign as d
from nicla_study.state import resource_lease
from e_pilot import digest_file, imagenet_gray_mobilenet, write_json

BASE_MOBILENET = d.keras_mobilenet

SEEDS = [20261101 + i for i in range(10)]
T3 = {"id": "T3", "head_epochs_max": 6, "fine_tune_epochs_max": 18,
      "head_lr": 0.0005, "fine_tune_lr": 0.0001,
      "unfreeze_top_fraction": 0.5, "augmentation": "mild_gray_v1"}


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def nested_unfreeze_top_fraction(model: tf.keras.Model, fraction: float) -> dict:
    base = model._d_base
    models, leaves = [], []

    def walk(container):
        if id(container) in {id(value) for value in models}:
            return
        models.append(container)
        for layer in container.layers:
            if isinstance(layer, tf.keras.Model):
                walk(layer)
            elif id(layer) not in {id(value) for value in leaves}:
                leaves.append(layer)

    walk(base)
    # Enable parent containers first, then apply the leaf-level freeze pattern.
    for container in models:
        container.trainable = True
    cutoff = max(1, int(np.floor(len(leaves) * (1.0 - float(fraction)))))
    for index, layer in enumerate(leaves):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    trainable = [layer.name for layer in leaves if layer.trainable]
    return {"backbone_layers_total": len(leaves), "freeze_cutoff_index": cutoff,
            "backbone_layers_trainable": trainable, "batch_normalization_frozen": True,
            "nested_models": [model.name for model in models]}


def wider_mobilenet(workspace: Path, alpha: float, pretrained: bool, seed: int):
    if alpha in (0.50, 0.75) and pretrained:
        return imagenet_gray_mobilenet(workspace, alpha, pretrained, seed)
    return BASE_MOBILENET(workspace, alpha, pretrained, seed)


def train_one(alpha: float, seed: int, shared_data: dict):
    arm_id = "capacity_a%03d" % int(alpha * 100)
    output = RUN_ROOT / "roots" / arm_id / ("seed-%d" % seed)
    result_path = output / "ROOT_RESULT.json"
    if result_path.is_file():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete_verified" and old.get("weights_sha256") == digest_file(output / "classifier.weights.h5"):
            return {"seed": seed, "alpha": alpha, "status": "already_complete",
                    "elapsed_seconds": old.get("training", {}).get("elapsed_seconds")}
    d_root = STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6" / ("seed-%d" % seed)
    train_rows = d.read_jsonl(d_root / "training_rows.jsonl")
    train_gray = d.cache_gray8(ROOT, train_rows, "train")
    output.mkdir(parents=True, exist_ok=True)
    append_jsonl(output / "ATTEMPTS.jsonl", {"status": "started", "seed": seed,
                                              "alpha": alpha, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    original_mobilenet = d.keras_mobilenet
    original_unfreeze = d.unfreeze_top_fraction
    d.keras_mobilenet = wider_mobilenet
    d.unfreeze_top_fraction = nested_unfreeze_top_fraction
    try:
        started = time.monotonic()
        model, history = d.train_model(
            ROOT, alpha, True, 6, seed, arm_id + "_seed-" + str(seed),
            train_rows, train_gray, shared_data["stop_rows"], shared_data["stop_gray"],
            dict(T3), dataset="survface_96", patience=3)
        output.mkdir(parents=True, exist_ok=True)
        weight_path = output / "classifier.weights.h5"
        tmp_weights = output / "classifier.weights.partial.h5"
        model.save_weights(str(tmp_weights))
        os.replace(tmp_weights, weight_path)
        curve_path = output / "training_curve.jsonl"
        with curve_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in history["curve"]:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        config = {"protocol_id": "nicla-five-known-study-v4", "job": "E",
                  "arm_id": arm_id, "root_seed": seed, "alpha": alpha,
                  "config": dict(T3), "pretrained": True,
                  "pretraining": "Keras MobileNetV1 ImageNet RGB no-top; exact grayscale first-convolution kernel sum; embedded [-1,1] scaling",
                  "checkpoint_sha256": history["backbone"].get("checkpoint_sha256"),
                  "training_rows_file": str(d_root / "training_rows.jsonl"),
                  "training_rows_file_sha256": digest_file(d_root / "training_rows.jsonl"),
                  "training_rows_fingerprint": d.rows_fingerprint(train_rows),
                  "label_order": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
                  "normalization": "gray8 / 255; then Rescaling(2*x-1) before adapted ImageNet backbone",
                  "allowed_roles": ["train", "unknown_train", "stopping"],
                  "final_role_accessed": False}
        atomic_json(output / "config.json", config)
        result = {"schema": "job-e-capacity-root-1", "status": "complete_verified",
                  "arm_id": arm_id, "alpha": alpha, "root_seed": seed,
                  "initialization": history["initialization"],
                  "checkpoint_sha256": history["checkpoint_sha256"],
                  "training_rows_fingerprint": d.rows_fingerprint(train_rows),
                  "training_rows_file_sha256": digest_file(d_root / "training_rows.jsonl"),
                  "training": history,
                  "weights_path": str(weight_path), "weights_sha256": digest_file(weight_path),
                  "weights_bytes": weight_path.stat().st_size,
                  "config_path": str(output / "config.json"), "config_sha256": digest_file(output / "config.json"),
                  "curve_path": str(curve_path), "curve_sha256": digest_file(curve_path),
                  "elapsed_wall_seconds": time.monotonic() - started,
                  "final_role_accessed": False}
        atomic_json(result_path, result)
        append_jsonl(output / "ATTEMPTS.jsonl", {"status": "complete_verified", "seed": seed,
                                                   "alpha": alpha, "weights_sha256": result["weights_sha256"],
                                                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        del model
        tf.keras.backend.clear_session()
        return {"seed": seed, "alpha": alpha, "status": "complete_verified",
                "elapsed_seconds": result["elapsed_wall_seconds"],
                "epochs_run": history["epochs_run"], "steps_run": history["steps_run"],
                "weights_sha256": result["weights_sha256"]}
    finally:
        d.keras_mobilenet = original_mobilenet
        d.unfreeze_top_fraction = original_unfreeze


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    paths = d.study_paths(ROOT)
    data = d.load_role_arrays(ROOT, paths, "survface_96")
    predecl = json.loads((STUDY / "reports/E/PREDECLARATION.json").read_text(encoding="utf-8"))
    manifest = {"schema": "job-e-capacity-root-registry-1", "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
                "status": "in_progress", "root_seeds": SEEDS, "arms": {}, "final_role_accessed": False}
    registry_path = RUN_ROOT / "roots/CAPACITY_ROOT_REGISTRY.json"
    with resource_lease(STUDY, "heavy_training", owner="E_capacity_train_alpha_050_075_ten_roots", timeout_s=0):
        for alpha in (0.50, 0.75):
            rows = []
            for seed in SEEDS:
                try:
                    row = train_one(alpha, seed, data)
                except Exception as exc:
                    output = RUN_ROOT / "roots" / ("capacity_a%03d" % int(alpha * 100)) / ("seed-%d" % seed)
                    failure = {"schema": "job-e-capacity-root-1", "status": "failed_needs_repair",
                               "arm_id": "capacity_a%03d" % int(alpha * 100), "alpha": alpha,
                               "root_seed": seed, "error_type": type(exc).__name__, "error": str(exc)[:2000],
                               "traceback_tail": __import__("traceback").format_exc()[-4000:],
                               "final_role_accessed": False}
                    atomic_json(output / "ROOT_RESULT.json", failure)
                    append_jsonl(output / "ATTEMPTS.jsonl", {"status": "failed_needs_repair", "seed": seed,
                                                              "alpha": alpha, "error_type": type(exc).__name__,
                                                              "error": str(exc)[:2000],
                                                              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    row = {"seed": seed, "alpha": alpha, "status": "failed_needs_repair",
                           "error_type": type(exc).__name__, "error": str(exc)[:800]}
                rows.append(row)
                manifest["arms"]["a%03d" % int(alpha * 100)] = rows
                atomic_json(registry_path, manifest)
                print(json.dumps({"event": "capacity_root", **row}), flush=True)
            manifest["arms"]["a%03d" % int(alpha * 100)] = rows
            atomic_json(registry_path, manifest)
        manifest["status"] = "complete_verified"
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["data_role_receipt"] = {"train_pool_rows": len(data["train_rows"]),
                                          "stop_rows": len(data["stop_rows"]),
                                          "final_role_accessed": False}
        atomic_json(registry_path, manifest)
    print(json.dumps({"event": "capacity_campaign_complete", "status": manifest["status"],
                      "registry": str(registry_path)}), flush=True)


if __name__ == "__main__":
    main()
