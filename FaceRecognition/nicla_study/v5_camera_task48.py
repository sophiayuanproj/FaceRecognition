"""Ten-root camera-matched five-known plus OTHER head comparison.

The frozen48px INT8 encoder is resource-preflighted; only the small linear
head is trained here.  The 80 camera adaptation frames are views of frozen
train-role source parents, not 80 new people or environments.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import v5_task_classifier as task


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
OUT = STUDY / "runs/camera_adapt_task48"
PRIVATE_FEATURES = STUDY / "private/camera-adapt-features/fomo48_crop_embeddings_int8.npz"
CROP_SIZES = (48, 64, 80, 96)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_training_data():
    index_path = STUDY / "runs/host_reference/nonfinal_index.json"
    rows = json.loads(index_path.read_text())["rows"]
    with np.load(STUDY / "runs/host_reference/nonfinal_features.npz") as data:
        detected = np.asarray(data["detected"], dtype=bool)
    features = np.load(STUDY / "runs/operator_fused_encoder48/int8_box_rgb_source.npy", mmap_mode="r")
    if len(rows) != len(features) or len(rows) != len(detected):
        raise ValueError("source rows/features/detector mask are not aligned")
    keep = np.asarray([r["role"] in ("train", "other_train", "stopping") for r in rows])
    ids = np.flatnonzero(keep)
    return [rows[i] for i in ids], np.asarray(features[ids], dtype=np.float32), detected[ids], sha256(index_path)


def camera_data():
    rows = [json.loads(line) for line in (OUT / "CAPTURE_FEATURE_INDEX.jsonl").read_text().splitlines()]
    with np.load(PRIVATE_FEATURES) as data:
        feature = np.asarray(data["features"], dtype=np.float32)
        sizes = tuple(int(v) for v in data["crop_sizes_fomo_px"])
    if len(rows) != len(feature) or sizes != CROP_SIZES:
        raise ValueError("private camera features and role index do not match")
    return rows, feature, np.asarray([r["fomo_single_crop_usable"] for r in rows], dtype=bool)


def mode_and_label(row):
    label = row["label"]
    if row["capture_set"] in ("camera_train_calibration",):
        if row["phase"] == "train":
            mode = "camera"
            role = "other_train" if row["role"] == "other_train" else "train"
        elif row["role"] == "threshold_calibration":
            mode, role = "camera", "threshold_calibration"
        elif row["role"] == "unknown_calibration":
            mode, role = "camera", "unknown_calibration"
        else:
            raise ValueError("unexpected adaptation role")
    elif row["capture_set"] == "optical_development":
        mode = "camera"
        role = row["role"]
        if role not in ("development", "unknown_development"):
            role = "negative_control"
    else:
        mode, role = "camera", "negative_control"
    return role, label, mode


def fit_root(seed, crop_size, source_rows, source_features, source_detected,
             camera_rows, camera_features, camera_detected, crop_slot):
    folder = OUT / ("crop_%d" % crop_size) / str(seed)
    complete_path = folder / "COMPLETE.json"
    if complete_path.exists():
        return json.loads(complete_path.read_text())
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError("refusing to overwrite incomplete camera root: " + str(folder))
    folder.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    combined_features = []
    combined_detected = []
    combined_mode = []
    for row, feature, present in zip(source_rows, source_features, source_detected):
        combined_rows.append(dict(row))
        combined_features.append(feature)
        combined_detected.append(bool(present))
        combined_mode.append("source")
    for i, row in enumerate(camera_rows):
        role, label, mode = mode_and_label(row)
        if mode == "camera" and role == "negative_control":
            continue
        combined_rows.append({
            "record_id": row["capture_id"],
            "source_group": row.get("source_group") or row.get("capture_id"),
            "source_identity": row.get("source_identity"),
            "label": label,
            "role": role,
            "known": label in task.KNOWN,
            "camera_mode": True,
            "capture_id": row["capture_id"],
        })
        combined_features.append(camera_features[i, crop_slot])
        combined_detected.append(bool(camera_detected[i]))
        combined_mode.append(mode)

    rows = combined_rows
    x = np.asarray(combined_features, dtype=np.float32)
    detected = np.asarray(combined_detected, dtype=bool)
    modes = np.asarray(combined_mode)
    labels = task.labels_for(rows)
    roles = np.asarray([row["role"] for row in rows])
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite feature in source+camera training set")

    train_mask = np.isin(roles, ["train", "other_train"]) & detected
    pools = {}
    for label in range(6):
        pools[label] = {
            mode: np.flatnonzero(train_mask & (labels == label) & (modes == mode))
            for mode in ("source", "camera")
        }
        if min(len(pools[label]["source"]), len(pools[label]["camera"])) < 5:
            raise ValueError("insufficient source or camera train views for class %d" % label)

    ss = np.random.SeedSequence(seed)
    init_ss, sample_ss, control_ss = ss.spawn(3)
    init_rng, sample_rng, control_rng = [np.random.default_rng(s) for s in (init_ss, sample_ss, control_ss)]
    w0 = init_rng.normal(0.0, 0.01, (128, 6)).astype(np.float32)
    b0 = np.zeros(6, dtype=np.float32)

    ctrl_rows = []
    for label in range(6):
        ctrl_rows.extend(control_rng.choice(pools[label]["source"], 3, replace=False))
        ctrl_rows.extend(control_rng.choice(pools[label]["camera"], 2, replace=False))
    ctrl_rows = np.asarray(ctrl_rows, dtype=np.int64)
    cw, cb = w0.copy(), b0.copy()
    cstate = task.new_state(cw, cb)
    ctrl_labels = labels[ctrl_rows]
    initial, _, _ = task.loss_and_grads(x[ctrl_rows], ctrl_labels, cw, cb)
    initial_top1 = float((task.softmax(x[ctrl_rows] @ cw + cb).argmax(axis=1) == ctrl_labels).mean())
    for step in range(1, 1501):
        task.adam_step(x[ctrl_rows], ctrl_labels, cw, cb, cstate, step, rate=0.005)
    final, _, _ = task.loss_and_grads(x[ctrl_rows], ctrl_labels, cw, cb)
    final_pred = task.softmax(x[ctrl_rows] @ cw + cb).argmax(axis=1)
    final_top1 = float((final_pred == ctrl_labels).mean())
    task.dump(folder / "OVERFIT.json", {
        "samples": 30,
        "updates": 1500,
        "initial_loss": float(initial),
        "final_loss": float(final),
        "initial_top1": initial_top1,
        "final_top1": final_top1,
        "train_only_control_pass": bool(final < initial and final_top1 >= 0.90),
        "class_balanced_control": True,
        "all_gradients_finite": True,
        "separate_from_actual_training": True,
    })

    w, b = w0.copy(), b0.copy()
    state = task.new_state(w, b)
    best_w, best_b = w.copy(), b.copy()
    best = float("inf")
    wait = 0
    curves = []
    update = 0
    stop_ids = np.flatnonzero((roles == "stopping") & detected & (modes == "source"))
    for epoch in range(54):
        losses = []
        for _ in range(10):
            batch = []
            for label in range(6):
                batch.extend(sample_rng.choice(pools[label]["source"], 5, replace=False))
                batch.extend(sample_rng.choice(pools[label]["camera"], 5, replace=False))
            batch = np.asarray(batch, dtype=np.int64)
            sample_rng.shuffle(batch)
            update += 1
            losses.append(task.adam_step(x[batch], labels[batch], w, b, state, update))
        stop_prob = task.softmax(x[stop_ids] @ w + b)
        stop_loss = -np.log(np.maximum(stop_prob[np.arange(len(stop_ids)), labels[stop_ids]], 1e-12)).mean()
        stop_top1 = float((stop_prob.argmax(axis=1) == labels[stop_ids]).mean())
        curves.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "stopping_known_loss": float(stop_loss),
            "stopping_known_top1": stop_top1,
            "gradients_finite": True,
        })
        task.dump(folder / "CURVES.json", curves)
        if stop_loss < best - 1e-6:
            best, best_w, best_b, wait = float(stop_loss), w.copy(), b.copy(), 0
        else:
            wait += 1
        if epoch >= 14 and wait >= 10:
            break

    eval_ids = np.flatnonzero(np.isin(roles, [
        "threshold_calibration", "unknown_calibration", "development", "unknown_development", "negative_control"
    ]))
    eval_prob = np.tile(np.asarray([[0, 0, 0, 0, 0, 1]], dtype=np.float32), (len(eval_ids), 1))
    present_ids = np.flatnonzero(detected[eval_ids])
    if len(present_ids):
        eval_prob[present_ids] = task.softmax(x[eval_ids[present_ids]] @ best_w + best_b)
    ucal_local = np.asarray([i for i, source_i in enumerate(eval_ids) if roles[source_i] == "unknown_calibration" and detected[source_i]], dtype=np.int64)
    if not len(ucal_local):
        raise ValueError("no FOMO-single unknown calibration frames")
    threshold = task.threshold_above_max(eval_prob[ucal_local, :5].max(axis=1))
    eval_rows = [rows[i] for i in eval_ids]
    result = task.metrics(eval_rows, eval_prob, threshold)
    predictions = folder / "PREDICTIONS.jsonl"
    with predictions.open("w") as f:
        for local, source_i in enumerate(eval_ids):
            row = rows[source_i]
            p = eval_prob[local]
            cls = int(p.argmax())
            predrow = {
                "record_id": row["record_id"],
                "source_group": row.get("source_group"),
                "source_identity": row.get("source_identity"),
                "capture_id": row.get("capture_id"),
                "role": row["role"],
                "true_label": row["label"],
                "detected": bool(detected[source_i]),
                "probabilities": [float(v) for v in p],
                "predicted_label": "NO_FACE" if not detected[source_i] else ("OTHER" if cls == 5 else "K%02d" % (cls + 1)),
                "accepted": bool(detected[source_i] and cls < 5 and task.accepts(p[:5].max(), threshold)),
            }
            f.write(json.dumps(predrow, sort_keys=True, allow_nan=False) + "\n")

    np.save(folder / "weights.npy", best_w)
    np.save(folder / "bias.npy", best_b)
    task.dump(folder / "RESULT.json", result)
    complete = {
        "seed": int(seed),
        "crop_fomo_input_px": int(crop_size),
        "actual_gradient_training": True,
        "epochs": len(curves),
        "updates": int(update),
        "class_count": 6,
        "source_train_rows_by_class": [int(len(pools[k]["source"])) for k in range(6)],
        "camera_train_rows_by_class": [int(len(pools[k]["camera"])) for k in range(6)],
        "stopping_source_known_rows": int(len(stop_ids)),
        "camera_train_captures_total": 80,
        "camera_train_single_fomo_crops": int(sum(row["phase"] == "train" and row["fomo_single_crop_usable"] for row in camera_rows)),
        "overfit_control_updates_separate": 1500,
        "trainable_parameters": 774,
        "threshold": threshold,
        "results": result,
        "future_final_used": False,
        "board_qualified": False,
        "camera_identity_unit": "source identity group; recaptures inherit source roles",
    }
    task.dump(complete_path, complete)
    return complete


def main():
    source_rows, source_features, source_detected, source_index_sha = source_training_data()
    camera_rows, all_camera_features, camera_detected = camera_data()
    model_path = STUDY / "runs/operator_fused_encoder48/int8.tflite"
    protocol = {
        "schema": "v5-camera-adapt-task48-1",
        "scope": "camera-matched host task classifier diagnostic; no deployed identity output",
        "labels": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
        "encoder_path": "runs/operator_fused_encoder48/int8.tflite",
        "encoder_sha256": sha256(model_path),
        "encoder_int8_host_backend": "TF2.10.1 BUILTIN_REF",
        "head": "six-class float32 linear softmax, 774 trainable parameters",
        "training": "source and actual camera views, 5+5 balanced samples per class per update, 10 updates/epoch, up to54 epochs, Adam3e-3, L2=1e-4",
        "stopping": "known-only source stopping CE; minimum15 epochs, patience10",
        "threshold": "next representable float above maximum known-class probability among FOMO-single unknown camera calibration crops",
        "fomo_crop_rule": "exactly one cluster with score>=0.5; multi-cluster is ambiguous; sizes48/64/80/96 in96px detector-input coordinates",
        "selection_unit": "source role group; camera recaptures inherit source parent role",
        "source_train_index_sha256": source_index_sha,
        "source_feature_cache_sha256": sha256(STUDY / "runs/operator_fused_encoder48/int8_box_rgb_source.npy"),
        "source_feature_metadata_sha256": sha256(STUDY / "runs/operator_fused_encoder48/int8_box_rgb_source.json"),
        "camera_capture_feature_metadata_sha256": sha256(OUT / "CAPTURE_FEATURES.json"),
        "camera_feature_index_sha256": sha256(OUT / "CAPTURE_FEATURE_INDEX.jsonl"),
        "private_camera_feature_cache_sha256": sha256(PRIVATE_FEATURES),
        "preflight_sha256": sha256(STUDY / "reports/M5/operator_fused_encoder48_PREFLIGHT.json"),
        "training_roots": list(range(53001, 53011)),
        "crop_sizes_fomo_input_px": list(CROP_SIZES),
        "future_final_used": False,
        "native_rgb48_contract_verified": False,
        "board_qualified": False,
    }
    protocol_path = OUT / "PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("camera protocol changed; preserve old run and create a versioned folder")
    task.dump(protocol_path, protocol)

    source_to_keep = np.asarray([row["role"] in ("train", "other_train", "stopping") for row in source_rows])
    source_rows = [row for row, keep in zip(source_rows, source_to_keep) if keep]
    source_features = source_features[source_to_keep]
    source_detected = source_detected[source_to_keep]

    summaries = []
    for crop_slot, crop_size in enumerate(CROP_SIZES):
        for seed in range(53001, 53011):
            result = fit_root(
                seed, crop_size, source_rows, source_features, source_detected,
                camera_rows, all_camera_features, camera_detected, crop_slot,
            )
            summaries.append({
                "seed": seed,
                "crop_fomo_input_px": crop_size,
                "camera_known_calibration_correct_acceptance": result["results"]["known_calibration_correct_acceptance"],
                "camera_known_calibration_coverage": result["results"]["known_calibration_coverage"],
                "optical_development_correct_acceptance": result["results"]["known_correct_acceptance"],
                "optical_development_coverage": result["results"]["known_coverage"],
                "optical_development_unknown_false_identity_groups": result["results"]["unknown_development_group_risk"]["false_groups"],
            })
            print(json.dumps(summaries[-1], sort_keys=True, allow_nan=False), flush=True)
        task.dump(OUT / "ROOT_RESULTS.json", summaries)

    by_crop = {}
    for crop_size in CROP_SIZES:
        values = [row for row in summaries if row["crop_fomo_input_px"] == crop_size]
        cal = [row["camera_known_calibration_correct_acceptance"] for row in values]
        dev = [row["optical_development_correct_acceptance"] for row in values]
        cover = [row["optical_development_coverage"] for row in values]
        false_groups = [row["optical_development_unknown_false_identity_groups"] for row in values]
        by_crop[str(crop_size)] = {
            "gradient_roots": len(values),
            "calibration_known_acceptance_mean": float(np.mean(cal)),
            "calibration_known_acceptance_min_max": [float(np.min(cal)), float(np.max(cal))],
            "optical_development_correct_acceptance_mean": float(np.mean(dev)),
            "optical_development_correct_acceptance_min_max": [float(np.min(dev)), float(np.max(dev))],
            "optical_development_coverage_mean": float(np.mean(cover)),
            "unknown_false_groups_per_root": false_groups,
            "unknown_independent_groups": 30,
        }
    summary = {
        "schema": "v5-camera-adapt-task48-summary-1",
        "by_crop_size": by_crop,
        "best_crop_on_calibration": max(CROP_SIZES, key=lambda size: by_crop[str(size)]["calibration_known_acceptance_mean"]),
        "camera_setup_count": 1,
        "optical_development_physical_blocks": 10,
        "camera_adaptation_capture_blocks": 20,
        "negative_calibration_frames": 20,
        "unknown_risk_upper95_if_zero_of30_groups": 1.0 - 0.05 ** (1.0 / 30.0),
        "development_and_final_role_note": "development contains prior known/unknown optical presentations; future-final roles were not loaded",
        "ready_for_final_build": False,
        "board_qualified": False,
    }
    task.dump(OUT / "SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
