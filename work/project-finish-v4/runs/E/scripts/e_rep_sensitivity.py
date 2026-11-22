#!/usr/bin/env python3
"""Test ten train-only, intensity/blur-stratified PTQ calibration pools."""
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
sys.path.insert(0, str(RUN_ROOT / "scripts"))

import numpy as np
import tensorflow as tf

from nicla_study import d_campaign as d
from nicla_study.state import resource_lease
from e_pilot import digest_file, write_json
from e_export_campaign import (convert, tflite_infer, tensor_audit, all_layer_macs,
                               root_paths, load_capacity_model)
from nicla_study.macs import tflite_operator_macs, compare_mac_accounts

POOL_SEEDS = [73101 + i for i in range(10)]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stratified_pool(images, rows, seed, count=128):
    values = images.astype(np.float32) / 255.0
    intensity = np.mean(values, axis=(1, 2, 3))
    blur_energy = []
    for image in values:
        plane = image[:, :, 0]
        padded = np.pad(plane, 1, mode="reflect")
        response = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                    + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * plane)
        blur_energy.append(float(np.var(response)))
    blur_energy = np.asarray(blur_energy, dtype=np.float64)
    q_intensity = np.quantile(intensity, [0.25, 0.50, 0.75])
    q_blur = np.quantile(blur_energy, [0.25, 0.50, 0.75])
    i_bins = np.digitize(intensity, q_intensity, right=True)
    b_bins = np.digitize(blur_energy, q_blur, right=True)
    rng = np.random.default_rng(seed)
    chosen = []
    allocation = max(1, count // 16)
    for i in range(4):
        for b in range(4):
            candidates = np.flatnonzero((i_bins == i) & (b_bins == b))
            if not len(candidates):
                continue
            amount = min(allocation, len(candidates))
            chosen.extend(rng.choice(candidates, size=amount, replace=False).tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < count:
        remaining = np.asarray([index for index in range(len(images)) if index not in set(chosen)], dtype=np.int64)
        chosen.extend(rng.choice(remaining, size=min(count - len(chosen), len(remaining)), replace=False).tolist())
    chosen = np.asarray(chosen[:count], dtype=np.int64)
    receipt = {"seed": seed, "count": int(len(chosen)),
               "intensity_mean": float(np.mean(intensity[chosen])),
               "intensity_min": float(np.min(intensity[chosen])), "intensity_max": float(np.max(intensity[chosen])),
               "laplacian_variance_mean": float(np.mean(blur_energy[chosen])),
               "intensity_quantile_edges": q_intensity.tolist(), "blur_quantile_edges": q_blur.tolist(),
               "cell_counts": {"%d_%d" % (i, b): int(np.sum((i_bins[chosen] == i) & (b_bins[chosen] == b)))
                               for i in range(4) for b in range(4)},
               "selected_record_ids": [rows[index].get("record_id") for index in chosen],
               "selected_image_sha256": [rows[index].get("sha256") for index in chosen]}
    return chosen.tolist(), receipt


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    selection_path = STUDY / "reports/E/CAPACITY_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    alpha = float(selection["selected_large_alpha"])
    seed = 20261101
    data = d.load_role_arrays(ROOT, d.study_paths(ROOT), "survface_96")
    source, weights = root_paths(alpha, seed)[2:4]
    rows = d.read_jsonl(source / "training_rows.jsonl")
    images = d.cache_gray8(ROOT, rows, "train")
    model, _, info = load_capacity_model(alpha, seed, weights)
    root_dir = RUN_ROOT / "representative_pools" / ("capacity_a%03d_seed-%d" % (int(alpha * 100), seed))
    result = {"schema": "job-e-representative-pool-sensitivity-1", "status": "in_progress",
              "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
              "capacity_selection_sha256": digest_file(selection_path), "alpha": alpha,
              "model_weights_sha256": digest_file(weights), "model_initialization": info.get("initialization"),
              "representative_role_sources": ["train", "unknown_train"], "pool_size": 128,
              "pool_seeds": POOL_SEEDS, "final_role_accessed": False, "pools": []}
    cal_images = np.concatenate([data["cal_known_gray"], data["cal_unknown_gray"]], axis=0)
    dev_images = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    with resource_lease(STUDY, "heavy_training", owner="E_ten_train_only_representative_pool_sensitivity", timeout_s=0):
        for pool_seed in POOL_SEEDS:
            path = root_dir / ("pool-%d.tflite" % pool_seed)
            pool_indices, pool_receipt = stratified_pool(images, rows, pool_seed, 128)
            try:
                convert(model, "full_int8_ptq_train_only_rep", images, pool_indices, path)
                interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
                interpreter.allocate_tensors()
                audit = tensor_audit(interpreter)
                cal_logits, _ = tflite_infer(interpreter, cal_images)
                calibration = d.fit_open_threshold(cal_logits[:len(data["cal_known"])], data["cal_known"],
                                                    cal_logits[len(data["cal_known"]):], data["cal_unknown"], 6)
                dev_logits, _ = tflite_infer(interpreter, dev_images)
                development = d.known_macro_and_open_metrics(dev_logits, data["dev_known"], data["dev_unknown"],
                                                              calibration["threshold"], 6)
                tflite_macs = tflite_operator_macs(interpreter)
                row = {"status": "complete_verified", "pool": pool_receipt,
                       "path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path),
                       "tensor_audit": audit,
                       "mac_accounting": compare_mac_accounts(all_layer_macs(model), tflite_macs),
                       "tflite_macs": tflite_macs, "calibration": calibration,
                       "development": development, "host_inference_status": "pass",
                       "role_counts": {"cal_known": len(data["cal_known"]), "cal_unknown": len(data["cal_unknown"]),
                                       "dev_known": len(data["dev_known"]), "dev_unknown": len(data["dev_unknown"])},
                       "final_role_accessed": False}
                del interpreter
            except Exception as exc:
                row = {"status": "unsupported_or_failed", "pool": pool_receipt,
                       "path": str(path) if path.exists() else None,
                       "error_type": type(exc).__name__, "error": str(exc)[:1500],
                       "traceback_tail": __import__("traceback").format_exc()[-3000:],
                       "host_inference_status": "not_run", "final_role_accessed": False}
            result["pools"].append(row)
            atomic_json(root_dir / ("pool-%d.json" % pool_seed), row)
            atomic_json(root_dir / "SENSITIVITY_PROGRESS.json", result)
            print(json.dumps({"event": "representative_pool", "seed": pool_seed,
                              "status": row["status"], "bytes": row.get("bytes"),
                              "utility": (row.get("development") or {}).get("development_macro_utility")}), flush=True)
        successful = [row for row in result["pools"] if row["status"] == "complete_verified"]
        result["status"] = "complete_verified" if len(successful) == len(POOL_SEEDS) else "partial"
        result["summary"] = {key: {"mean": float(np.mean([row["development"][key] for row in successful])),
                                  "sd_pool": float(np.std([row["development"][key] for row in successful], ddof=1)) if len(successful) > 1 else 0.0}
                             for key in ("development_macro_utility", "known_identity_macro_correct_acceptance",
                                         "unknown_false_identification_rate")}
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(root_dir / "RESULT.json", result)
    del model
    tf.keras.backend.clear_session()
    print(json.dumps({"event": "representative_sensitivity_complete", "status": result["status"],
                      "report": str(root_dir / "RESULT.json")}), flush=True)


if __name__ == "__main__":
    main()
