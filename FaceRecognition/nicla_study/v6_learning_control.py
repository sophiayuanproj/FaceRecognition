"""Diagnose whether the corrected 48px features support a clean train-only fit."""

import json
from pathlib import Path

import numpy as np

from . import v5_camera_task48 as task48
from . import v6_camera_task48 as experiment


def run(seed=56001):
    source_rows, source_features, source_detected, _ = task48.source_training_data()
    keep = np.asarray([r["role"] in ("train", "other_train", "stopping") for r in source_rows])
    source_rows = [r for r, yes in zip(source_rows, keep) if yes]
    source_features = source_features[keep]
    source_detected = source_detected[keep]
    camera_rows, camera_features, camera_detected = experiment.camera_data()
    rows, features, detected, modes = [], [], [], []
    for row, vector, present in zip(source_rows, source_features, source_detected):
        rows.append(dict(row)); features.append(vector); detected.append(bool(present)); modes.append("source")
    for row, vector, present in zip(camera_rows, camera_features[:, 0], camera_detected):
        role, label, mode = task48.mode_and_label(row)
        if role == "negative_control":
            continue
        rows.append({"record_id": row["capture_id"], "source_group": row.get("source_group") or row["capture_id"],
                     "source_identity": row.get("source_identity"), "label": label, "role": role,
                     "capture_id": row["capture_id"]})
        features.append(vector); detected.append(bool(present)); modes.append(mode)
    x = np.asarray(features, dtype=np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    detected, modes = np.asarray(detected, dtype=bool), np.asarray(modes)
    labels = task48.task.labels_for(rows)
    roles = np.asarray([row["role"] for row in rows])
    train_mask = np.isin(roles, ["train", "other_train"]) & detected
    pools = {label: {mode: np.flatnonzero(train_mask & (labels == label) & (modes == mode))
                     for mode in ("source", "camera")} for label in range(6)}
    shortages = {label: {mode: len(ids) for mode, ids in pool.items()}
                 for label, pool in pools.items() if min(map(len, pool.values())) < 5}
    if shortages:
        raise ValueError("corrected train role cannot support balanced training: " + repr(shortages))
    init_ss, _, control_ss = np.random.SeedSequence(seed).spawn(3)
    init_rng, control_rng = np.random.default_rng(init_ss), np.random.default_rng(control_ss)
    w0 = init_rng.normal(0.0, 0.01, (128, 6)).astype(np.float32)
    b0 = np.zeros(6, dtype=np.float32)
    ctrl_rows = []
    for label in range(6):
        ctrl_rows.extend(control_rng.choice(pools[label]["source"], 3, replace=False))
        ctrl_rows.extend(control_rng.choice(pools[label]["camera"], 2, replace=False))
    ctrl_rows = np.asarray(ctrl_rows, dtype=np.int64)
    y = labels[ctrl_rows]
    w, b = w0.copy(), b0.copy()
    state = task48.task.new_state(w, b)
    checkpoints = []
    step = 0
    for target in (150, 500, 1500, 5000):
        while step < target:
            step += 1
            task48.task.adam_step(x[ctrl_rows], y, w, b, state, step, rate=0.005)
        loss, _, _ = task48.task.loss_and_grads(x[ctrl_rows], y, w, b)
        top1 = float((task48.task.softmax(x[ctrl_rows] @ w + b).argmax(axis=1) == y).mean())
        checkpoints.append({"updates": target, "loss": float(loss), "top1": top1,
                            "class_counts": [int((y == c).sum()) for c in range(6)]})
    report = {
        "schema": "v6-camera-train-only-learning-control-1", "seed": seed,
        "samples": int(len(ctrl_rows)), "source_train_groups_and_camera_recaptures": True,
        "class_balanced": True, "finite_gradients": True,
        "feature_cache": "private/camera_corrected_input_v1/features.npz",
        "checkpoints": checkpoints,
        "diagnosis": "The original 150-step pilot was not a fit pass; this curve distinguishes insufficient updates from nonseparable corrected features.",
        "future_final_used": False,
    }
    out = experiment.STUDY / "runs/camera_corrected_recognition_v1/LEARNING_CONTROL.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    run()
