"""Source-only five-known plus OTHER classifier on frozen48px face features.

This is a finite diagnostic of task formulation and rejection.  The backbone
is the resource-preflighted48px INT8 encoder; the 774-parameter head remains
float32 on the host.  Each root has independent sampling and initialization.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from open_set_decision import accepts, threshold_above_max

KNOWN = {"K%02d" % i: i - 1 for i in range(1, 6)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def labels_for(rows):
    labels = np.full(len(rows), 5, dtype=np.int64)
    for i, row in enumerate(rows):
        if row["label"] in KNOWN:
            labels[i] = KNOWN[row["label"]]
    return labels


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grads(x, y, w, b, l2=1e-4):
    p = softmax(x @ w + b)
    n = len(y)
    loss = -np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean() + l2 * float((w * w).sum())
    p[np.arange(n), y] -= 1.0
    p /= n
    return loss, x.T @ p + 2.0 * l2 * w, p.sum(axis=0)


def adam_step(x, y, w, b, state, step, rate=0.003):
    loss, gw, gb = loss_and_grads(x, y, w, b)
    if not (math.isfinite(loss) and np.isfinite(gw).all() and np.isfinite(gb).all()):
        raise ValueError("nonfinite task-classifier loss or gradient")
    mw, vw, mb, vb = state
    mw[:] = 0.9 * mw + 0.1 * gw
    vw[:] = 0.999 * vw + 0.001 * (gw * gw)
    mb[:] = 0.9 * mb + 0.1 * gb
    vb[:] = 0.999 * vb + 0.001 * (gb * gb)
    mw_hat = mw / (1.0 - 0.9 ** step)
    vw_hat = vw / (1.0 - 0.999 ** step)
    mb_hat = mb / (1.0 - 0.9 ** step)
    vb_hat = vb / (1.0 - 0.999 ** step)
    w -= rate * mw_hat / (np.sqrt(vw_hat) + 1e-8)
    b -= rate * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return loss


def new_state(w, b):
    return [np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b)]


def eval_rows(x, y, w, b, train_groups):
    p = softmax(x @ w + b)
    return p


def metrics(rows, probs, threshold):
    roles = np.asarray([r["role"] for r in rows])
    labels = labels_for(rows)
    pred = probs.argmax(axis=1)
    known_score = probs[:, :5].max(axis=1)
    accept = (pred < 5) & np.asarray([accepts(score, threshold) for score in known_score], dtype=bool)
    known = roles == "development"
    known_cal = roles == "threshold_calibration"
    unknown = roles == "unknown_development"
    unknown_cal = roles == "unknown_calibration"

    def group_stats(mask):
        groups = {}
        for i, row in enumerate(rows):
            if mask[i]:
                group = row.get("source_identity") or row.get("source_group") or row["record_id"]
                groups[group] = groups.get(group, False) or bool(accept[i])
        false_groups = sum(groups.values())
        n_groups = len(groups)
        upper = 1.0 - 0.05 ** (1.0 / n_groups) if n_groups and false_groups == 0 else None
        return {"groups": n_groups, "false_groups": int(false_groups), "zero_event_upper95": upper}

    per_person = []
    for k in range(5):
        m = known & (labels == k)
        per_person.append({
            "label": "K%02d" % (k + 1),
            "n": int(m.sum()),
            "raw_top1": float((pred[m] == k).mean()) if m.any() else None,
            "correct_accepted": int((m & accept & (pred == k)).sum()),
            "coverage": float(accept[m].mean()) if m.any() else None,
        })
    return {
        "known_development_n": int(known.sum()),
        "known_raw_top1_full": float((pred[known] == labels[known]).mean()) if known.any() else None,
        "known_correct_acceptance": float((accept[known] & (pred[known] == labels[known])).mean()) if known.any() else None,
        "known_coverage": float(accept[known].mean()) if known.any() else None,
        "known_wrong_accepted": int((accept[known] & (pred[known] != labels[known])).sum()),
        "unknown_development_n": int(unknown.sum()),
        "unknown_false_accepts": int((accept & unknown).sum()),
        "unknown_development_group_risk": group_stats(unknown),
        "known_calibration_n": int(known_cal.sum()),
        "known_calibration_correct_acceptance": float((accept[known_cal] & (pred[known_cal] == labels[known_cal])).mean()) if known_cal.any() else None,
        "known_calibration_coverage": float(accept[known_cal].mean()) if known_cal.any() else None,
        "unknown_calibration_n": int(unknown_cal.sum()),
        "unknown_calibration_false_accepts": int((accept & unknown_cal).sum()),
        "unknown_calibration_group_risk": group_stats(unknown_cal),
        "other_argmax_unknown_development": int(((pred == 5) & unknown).sum()),
        "per_person": per_person,
        "threshold": float(threshold),
        "decision": "argmax over K01..K05+OTHER, then accept known only above the zero-calibration-unknown threshold",
        "future_final_used": False,
        "board_qualified": False,
    }


def train_root(seed, transform, features, rows, detected, out):
    folder = out / str(seed) / transform
    complete = folder / "COMPLETE.json"
    if complete.exists():
        return json.loads(complete.read_text())
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError("Refusing to overwrite incomplete root: " + str(folder))
    folder.mkdir(parents=True, exist_ok=True)

    ss = np.random.SeedSequence(seed)
    init_ss, sample_ss, control_ss = ss.spawn(3)
    init_rng = np.random.default_rng(init_ss)
    sample_rng = np.random.default_rng(sample_ss)
    control_rng = np.random.default_rng(control_ss)

    y_all = labels_for(rows)
    roles = np.asarray([r["role"] for r in rows])
    train_mask = np.isin(roles, ["train", "other_train"]) & detected
    stop_mask = (roles == "stopping") & detected
    pools = [np.flatnonzero(train_mask & (y_all == k)) for k in range(6)]
    if any(len(pool) < 10 for pool in pools):
        raise ValueError("insufficient detected training rows for a class")
    stop_ids = np.flatnonzero(stop_mask)
    train_ids = np.flatnonzero(train_mask)
    for i in train_ids:
        if rows[i]["role"] == "other_train" and rows[i]["label"] != "OTHER":
            raise ValueError("OTHER role label mismatch")
        if rows[i]["role"] == "train" and rows[i]["label"] not in KNOWN:
            raise ValueError("known train role label mismatch")

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 128 or not np.isfinite(x).all():
        raise ValueError("expected finite N x 128 feature cache")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms[detected] <= 1e-8).any():
        raise ValueError("zero feature vector in usable rows")
    x = x / np.maximum(norms, 1e-8)

    w0 = init_rng.normal(0.0, 0.01, (128, 6)).astype(np.float32)
    b0 = np.zeros(6, dtype=np.float32)

    # Clean 30-row memorization control, restored before the actual fit.
    ctrl_ids = np.concatenate([control_rng.choice(pool, 5, replace=False) for pool in pools])
    cw, cb = w0.copy(), b0.copy()
    cs = new_state(cw, cb)
    ctrl_labels = y_all[ctrl_ids]
    initial, _, _ = loss_and_grads(x[ctrl_ids], ctrl_labels, cw, cb)
    for step in range(1, 151):
        adam_step(x[ctrl_ids], ctrl_labels, cw, cb, cs, step, rate=0.005)
    final, _, _ = loss_and_grads(x[ctrl_ids], ctrl_labels, cw, cb)
    dump(folder / "OVERFIT.json", {
        "samples": 30,
        "updates": 150,
        "initial_loss": float(initial),
        "final_loss": float(final),
        "all_gradients_finite": True,
        "separate_from_actual_training": True,
    })

    w, b = w0.copy(), b0.copy()
    state = new_state(w, b)
    best_w, best_b = w.copy(), b.copy()
    best = float("inf")
    patience = 0
    curves = []
    update = 0
    for epoch in range(54):
        losses = []
        for _ in range(10):
            batch = np.concatenate([sample_rng.choice(pool, 10, replace=False) for pool in pools])
            sample_rng.shuffle(batch)
            update += 1
            losses.append(adam_step(x[batch], y_all[batch], w, b, state, update))
        stop_probs = softmax(x[stop_ids] @ w + b)
        stop_loss = -np.log(np.maximum(stop_probs[np.arange(len(stop_ids)), y_all[stop_ids]], 1e-12)).mean()
        stop_top1 = float((stop_probs.argmax(axis=1) == y_all[stop_ids]).mean())
        curves.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "stopping_known_loss": float(stop_loss),
            "stopping_known_top1": stop_top1,
            "gradients_finite": True,
        })
        dump(folder / "CURVES.json", curves)
        if stop_loss < best - 1e-6:
            best, best_w, best_b, patience = float(stop_loss), w.copy(), b.copy(), 0
        else:
            patience += 1
        if epoch >= 14 and patience >= 10:
            break

    # Threshold is fixed from the independent unknown-calibration role; known
    # threshold-calibration rows are reported but do not change the threshold.
    known_cal = np.flatnonzero((roles == "threshold_calibration") & detected)
    unknown_cal = np.flatnonzero((roles == "unknown_calibration") & detected)
    evaluation = np.flatnonzero(np.isin(roles, ["threshold_calibration", "unknown_calibration", "development", "unknown_development"]) & detected)
    if not len(unknown_cal) or not len(evaluation):
        raise ValueError("missing calibration or development rows")
    probs = softmax(x[evaluation] @ best_w + best_b)
    eval_rows = [rows[i] for i in evaluation]
    index = {sample_id: j for j, sample_id in enumerate(evaluation)}
    ucal_probs = softmax(x[unknown_cal] @ best_w + best_b)
    max_unknown = float(ucal_probs[:, :5].max(axis=1).max())
    threshold = threshold_above_max(ucal_probs[:, :5].max(axis=1))
    result = metrics(eval_rows, probs, threshold)

    w_path, b_path = folder / "weights.npy", folder / "bias.npy"
    np.save(w_path, best_w)
    np.save(b_path, best_b)
    with (folder / "PREDICTIONS.jsonl").open("w") as f:
        for i in evaluation:
            p = probs[index[i]]
            row = rows[i]
            outrow = {
                "record_id": row["record_id"],
                "source_group": row.get("source_group"),
                "source_identity": row.get("source_identity"),
                "role": row["role"],
                "true_label": row["label"],
                "probabilities": [float(v) for v in p],
                "predicted_label": "OTHER" if int(p.argmax()) == 5 else "K%02d" % (int(p.argmax()) + 1),
                "accepted": bool((p.argmax() < 5) and accepts(p[:5].max(), threshold)),
            }
            f.write(json.dumps(outrow, allow_nan=False, sort_keys=True) + "\n")
    dump(folder / "RESULT.json", result)
    completed = {
        "seed": int(seed),
        "transform": transform,
        "actual_gradient_training": True,
        "epochs": len(curves),
        "updates": int(update),
        "class_count": 6,
        "training_rows_by_class": [int(len(pool)) for pool in pools],
        "stopping_known_rows": int(len(stop_ids)),
        "development_known_rows": int((roles == "development").sum()),
        "development_unknown_rows": int((roles == "unknown_development").sum()),
        "unknown_identity_groups": 30,
        "overfit_control_updates_separate": 150,
        "trainable_parameters": 774,
        "model_weights_sha256": sha256(w_path),
        "model_bias_sha256": sha256(b_path),
        "future_final_used": False,
        "board_qualified": False,
        "results": result,
    }
    dump(complete, completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=53001)
    args = parser.parse_args()
    if not 1 <= args.roots <= 10:
        raise SystemExit("--roots must be between 1 and 10")

    index_path = STUDY / "runs/host_reference/nonfinal_index.json"
    feature_root = STUDY / "runs/operator_fused_encoder48"
    role_path = STUDY / "data/manifests/lfw_roles.jsonl"
    rows = json.loads(index_path.read_text())["rows"]
    if any("future_final" in row["role"] for row in rows):
        raise ValueError("future-final role entered source diagnostic")
    with np.load(STUDY / "runs/host_reference/nonfinal_features.npz") as data:
        detected = np.asarray(data["detected"], dtype=bool)
    if len(rows) != len(detected):
        raise ValueError("frozen index and detection flags are misaligned")

    out = STUDY / "runs/task_classifier_48"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "v5-source-task-classifier-48-1",
        "scope": "source-only diagnostic using frozen resource-preflighted48px INT8 MobileFaceNet embeddings",
        "head": "six-class K01..K05+OTHER float32 linear softmax; 774 trainable parameters; no full INT8 export",
        "labels": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
        "training_roles": ["train", "other_train"],
        "stopping_role": "stopping",
        "calibration_roles": ["threshold_calibration", "unknown_calibration"],
        "development_roles": ["development", "unknown_development"],
        "future_final_used": False,
        "batching": "10 detected examples per each of six classes per update; 10 updates/epoch; up to 54 epochs",
        "optimizer": "Adam, learning rate 0.003, L2 1e-4",
        "stopping": "known-only stopping cross entropy, minimum 15 epochs, patience 10",
        "threshold": "next representable binary32 above the maximum binary32 known-class probability among unknown_calibration; scores and thresholds are both rounded to binary32 for decisions",
        "independent_unit": "gradient-training root; unknown risk summarized by source identity groups",
        "seeds": list(range(args.seed_start, args.seed_start + args.roots)),
        "base_model": {
            "path": "runs/operator_fused_encoder48/int8.tflite",
            "sha256": sha256(feature_root / "int8.tflite"),
            "input_shape": [1, 48, 48, 3],
            "features_sha256": sha256(feature_root / "int8_box_rgb_source.npy"),
            "features_metadata_sha256": sha256(feature_root / "int8_box_rgb_source.json"),
            "features": "48px source box embeddings, L2-normalized; TF2.10 BUILTIN_REF; host crop/resize approximation",
            "input_transform_limit": "native RGB48 camera crop/resize contract remains unverified",
        },
        "hashes": {
            "source_code_sha256": sha256(Path(__file__)),
            "index_sha256": sha256(index_path),
            "roles_sha256": sha256(role_path),
            "detection_cache_sha256": sha256(STUDY / "runs/host_reference/nonfinal_features.npz"),
        },
    }
    feature_path = feature_root / "int8_box_rgb_source.npy"
    ppath = out / "PROTOCOL.json"
    if ppath.exists() and json.loads(ppath.read_text()) != protocol:
        raise RuntimeError("protocol mismatch; create a new versioned output root")
    dump(ppath, protocol)

    summaries = []
    for seed in protocol["seeds"]:
        features = np.load(feature_path, mmap_mode="r")
        complete = train_root(seed, "int8", features, rows, detected, out)
        summaries.append({
            "seed": seed,
            "transform": "int8_encoder_host_float_head",
            "known_correct_acceptance": complete["results"]["known_correct_acceptance"],
            "known_coverage": complete["results"]["known_coverage"],
            "known_raw_top1_full": complete["results"]["known_raw_top1_full"],
            "unknown_false_accepts": complete["results"]["unknown_false_accepts"],
            "unknown_false_identity_groups": complete["results"]["unknown_development_group_risk"]["false_groups"],
        })
        print(json.dumps(summaries[-1], allow_nan=False), flush=True)
        dump(out / "ROOT_RESULTS.json", summaries)


if __name__ == "__main__":
    main()
