"""Shared local-only data, split, model, and verification helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
MT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "work/dataset-prep"
MANIFEST = DATA_ROOT / "reports/manifest.jsonl"
PAIR_MANIFEST = DATA_ROOT / "reports/survface-pairs.jsonl"


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def load_manifest(include_olivetti=True):
    rows = [row for row in jsonl(MANIFEST)
            if include_olivetti or row["dataset"] != "olivetti_96"]
    by_record = {row["record_id"]: row for row in rows}
    surv = [r for r in rows if r["dataset"] == "survface_96" and r["identity_id"]]
    oliv = [r for r in rows if r["dataset"] == "olivetti_96" and r["identity_id"]]
    return rows, by_record, surv, oliv


def select_training_rows(rows, identity_ids, max_per_identity, seed):
    """Reproduce train_embedder's deterministic per-identity cap selection."""
    allowed = set(identity_ids)
    groups = defaultdict(list)
    for row in rows:
        if row["identity_id"] in allowed:
            groups[row["identity_id"]].append(row)
    rng = random.Random(seed)
    selected = []
    for identity in sorted(groups):
        candidates = sorted(groups[identity], key=lambda row: row["record_id"])
        if max_per_identity and len(candidates) > max_per_identity:
            candidates = rng.sample(candidates, max_per_identity)
        selected.extend(candidates)
    return selected


def image_path(row):
    return DATA_ROOT / row["relative_local_path"]


def load_gray(row):
    with Image.open(image_path(row)) as source:
        image = source.convert("L")
        if image.size != (96, 96):
            raise ValueError("Unexpected image shape in prepared manifest")
        return np.asarray(image, dtype=np.uint8)


def identity_split(rows, seed=20261021, fraction=0.2):
    """Stratify identities by image-count decile, then split entire identities."""
    counts = defaultdict(int)
    for row in rows:
        counts[row["identity_id"]] += 1
    identities = sorted(counts)
    identities.sort(key=lambda identity: (counts[identity], identity))
    bands = [[] for _ in range(10)]
    for index, identity in enumerate(identities):
        band = min(9, (index * 10) // max(1, len(identities)))
        bands[band].append(identity)
    rng = random.Random(seed)
    train_ids, validation_ids = [], []
    for band in bands:
        rng.shuffle(band)
        n_val = max(1, int(round(len(band) * fraction))) if band else 0
        validation_ids.extend(band[:n_val])
        train_ids.extend(band[n_val:])
    return sorted(train_ids), sorted(validation_ids), dict(counts)


def remove_exact_duplicate_conflicts(rows, holdout_rows=()):
    """Drop contradictory labels and train images duplicated in unlabeled holdout; collapse same-ID copies."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    conflicting_hashes = {
        digest for digest, group in by_hash.items()
        if len({row["identity_id"] for row in group}) > 1
    }
    holdout_hashes = {row["sha256"] for row in holdout_rows}
    train_holdout_hashes = set(by_hash) & holdout_hashes
    excluded_conflicts = {
        row["record_id"] for digest in conflicting_hashes for row in by_hash[digest]
    }
    excluded_holdout_duplicates = {
        row["record_id"] for digest in train_holdout_hashes for row in by_hash[digest]
    }
    excluded_all = excluded_conflicts | excluded_holdout_duplicates
    usable = [row for row in rows if row["record_id"] not in excluded_all]
    seen = set()
    deduplicated = []
    collapsed_same_identity = 0
    for row in sorted(usable, key=lambda item: (item["identity_id"], item["sha256"], item["record_id"])):
        key = (row["identity_id"], row["sha256"])
        if key in seen:
            collapsed_same_identity += 1
            continue
        seen.add(key)
        deduplicated.append(row)
    audit = {
        "input_training_records": len(rows),
        "exact_hash_groups": sum(len(group) > 1 for group in by_hash.values()),
        "exact_hash_groups_spanning_identity_labels": len(conflicting_hashes),
        "ambiguous_cross_identity_records_excluded": len(excluded_conflicts),
        "unlabeled_holdout_records_checked_by_hash_only": len(holdout_rows),
        "train_holdout_shared_exact_hash_groups": len(train_holdout_hashes),
        "training_records_overlapping_unlabeled_holdout_excluded": len(excluded_holdout_duplicates),
        "same_identity_exact_duplicate_records_collapsed": collapsed_same_identity,
        "training_records_after_exact_hash_cleanup": len(deduplicated),
        "identity_labels_retained": len({row["identity_id"] for row in deduplicated}),
        "excluded_record_ids": sorted(excluded_all),
    }
    return deduplicated, audit


def exact_duplicate_audit(rows, train_ids, validation_ids):
    train = set(train_ids)
    val = set(validation_ids)
    train_hashes = {r["sha256"] for r in rows if r["identity_id"] in train}
    val_hashes = {r["sha256"] for r in rows if r["identity_id"] in val}
    overlap = train_hashes & val_hashes
    excluded = {r["record_id"] for r in rows if r["identity_id"] in val and r["sha256"] in overlap}
    return {
        "train_validation_shared_sha256_group_count": len(overlap),
        "validation_records_excluded_for_exact_cross_split_duplicate": len(excluded),
        "excluded_validation_record_ids": sorted(excluded),
    }


def perceptual_cross_split_candidates(train_ids, validation_ids):
    report = DATA_ROOT / "reports/perceptual-duplicates.json"
    if not report.is_file():
        return {"cross_split_candidate_pair_count": None, "candidate_file_present": False}
    data = json.loads(report.read_text())["datasets"]["survface_96"]["candidate_pairs"]
    train, val = set(train_ids), set(validation_ids)
    crossing = [
        pair for pair in data
        if (pair["left_identity_id"] in train and pair["right_identity_id"] in val)
        or (pair["left_identity_id"] in val and pair["right_identity_id"] in train)
    ]
    return {
        "cross_split_candidate_pair_count": len(crossing),
        "candidate_file_present": True,
        "matches_were_not_automatically_removed": True,
        "candidate_pair_record_ids": [
            [pair["left_record_id"], pair["right_record_id"]] for pair in crossing
        ],
    }


def make_validation_pairs(rows, validation_ids, excluded_record_ids, seed=20261021):
    by_id = defaultdict(list)
    validation_set = set(validation_ids)
    for row in rows:
        if row["identity_id"] in validation_set and row["record_id"] not in excluded_record_ids:
            by_id[row["identity_id"]].append(row)
    rng = random.Random(seed + 1)
    positive, usable_ids = [], []
    for identity in sorted(by_id):
        choices = sorted(by_id[identity], key=lambda r: r["record_id"])
        rng.shuffle(choices)
        pair = None
        for left_index, left in enumerate(choices):
            for right in choices[left_index + 1:]:
                if left["sha256"] != right["sha256"]:
                    pair = {"pair_type": "positive", "left_record_id": left["record_id"],
                            "right_record_id": right["record_id"]}
                    break
            if pair:
                break
        if pair:
            positive.append(pair)
            usable_ids.append(identity)
    negative = []
    if len(usable_ids) >= 2:
        for i in range(len(positive)):
            left_id = usable_ids[i]
            right_id = usable_ids[(i + 1 + (i % (len(usable_ids) - 1))) % len(usable_ids)]
            if right_id == left_id:
                right_id = usable_ids[(i + 1) % len(usable_ids)]
            left = rng.choice(by_id[left_id])
            right = rng.choice(by_id[right_id])
            negative.append({"pair_type": "negative", "left_record_id": left["record_id"],
                             "right_record_id": right["record_id"]})
    rng.shuffle(negative)
    return positive + negative, len(usable_ids)


def make_olivetti_pairs(rows, seed=20261021):
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["identity_id"]].append(row)
    rng = random.Random(seed + 2)
    positives, negatives = [], []
    identities = sorted(by_id)
    for identity in identities:
        images = sorted(by_id[identity], key=lambda r: r["record_id"])
        rng.shuffle(images)
        n = min(5, len(images) // 2)
        for i in range(n):
            a, b = images[2 * i], images[2 * i + 1]
            if a["sha256"] != b["sha256"]:
                positives.append({"pair_type": "positive", "left_record_id": a["record_id"],
                                  "right_record_id": b["record_id"]})
    for i in range(len(positives)):
        left_id = identities[i % len(identities)]
        right_id = identities[(i + 1 + (i % (len(identities) - 1))) % len(identities)]
        a = rng.choice(by_id[left_id])
        b = rng.choice(by_id[right_id])
        negatives.append({"pair_type": "negative", "left_record_id": a["record_id"],
                          "right_record_id": b["record_id"]})
    return positives + negatives


def pair_scores(pairs, embeddings):
    scores, labels = [], []
    for pair in pairs:
        left = embeddings[pair["left_record_id"]]
        right = embeddings[pair["right_record_id"]]
        scores.append(float(np.dot(left, right)))
        labels.append(1 if pair["pair_type"] == "positive" else 0)
    return np.asarray(scores, np.float64), np.asarray(labels, np.int8)


def auc_eer(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return {"roc_auc": None, "eer": None, "eer_threshold": None}
    all_scores = np.concatenate([positive, negative])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_values = all_scores[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_pos, n_neg = len(positive), len(negative)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    # Sweep all distinct score thresholds in O(n log n), grouping tied scores.
    descending = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[descending]
    sorted_labels = labels[descending]
    cumulative_tp = np.cumsum(sorted_labels == 1, dtype=np.int64)
    cumulative_fp = np.cumsum(sorted_labels == 0, dtype=np.int64)
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    thresholds = np.r_[np.inf, sorted_scores[group_ends]]
    true_accept = np.r_[0.0, cumulative_tp[group_ends] / n_pos]
    false_accept = np.r_[0.0, cumulative_fp[group_ends] / n_neg]
    false_reject = 1.0 - true_accept
    gaps = np.abs(false_accept - false_reject)
    best = int(np.argmin(gaps))
    eer = (false_accept[best] + false_reject[best]) / 2.0
    return {"roc_auc": float(auc), "eer": float(eer),
            "eer_threshold": float(thresholds[best])}


def far_threshold(scores, labels, far_target=0.01):
    negatives = np.sort(scores[labels == 0])[::-1]
    if not len(negatives):
        raise ValueError("Need negative validation pairs to calibrate a threshold")
    allowed = int(math.floor(far_target * len(negatives)))
    if allowed >= len(negatives):
        threshold = float("-inf")
    else:
        threshold = float(np.nextafter(negatives[allowed], np.inf))
    return threshold


def verification_metrics(scores, labels, threshold=None, far_target=0.01, bootstrap=0, seed=20261021):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int8)
    curve = auc_eer(scores, labels)
    threshold = far_threshold(scores, labels, far_target) if threshold is None else float(threshold)
    accepted = scores >= threshold
    positives, negatives = labels == 1, labels == 0
    result = {
        "pair_count": int(len(scores)),
        "positive_pair_count": int(positives.sum()),
        "negative_pair_count": int(negatives.sum()),
        "roc_auc": curve["roc_auc"],
        "eer": curve["eer"],
        "eer_threshold": curve["eer_threshold"],
        "decision_threshold": threshold,
        "tar_at_decision_threshold": float(np.mean(accepted[positives])) if positives.any() else None,
        "far_at_decision_threshold": float(np.mean(accepted[negatives])) if negatives.any() else None,
        "requested_far_target": far_target,
    }
    if bootstrap and positives.sum() and negatives.sum():
        rng = np.random.default_rng(seed)
        values = {"roc_auc": [], "eer": [], "tar_at_decision_threshold": [], "far_at_decision_threshold": []}
        pos_scores, neg_scores = scores[positives], scores[negatives]
        for _ in range(bootstrap):
            ps = rng.choice(pos_scores, size=len(pos_scores), replace=True)
            ns = rng.choice(neg_scores, size=len(neg_scores), replace=True)
            sample_scores = np.concatenate([ps, ns])
            sample_labels = np.r_[np.ones(len(ps), np.int8), np.zeros(len(ns), np.int8)]
            curve_sample = auc_eer(sample_scores, sample_labels)
            values["roc_auc"].append(curve_sample["roc_auc"])
            values["eer"].append(curve_sample["eer"])
            values["tar_at_decision_threshold"].append(float(np.mean(ps >= threshold)))
            values["far_at_decision_threshold"].append(float(np.mean(ns >= threshold)))
        result["bootstrap_95pct_ci"] = {
            key: [float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5))]
            for key, value in values.items()
        }
    return result


def embed_keras(model, record_ids, by_record, batch_size=128):
    vectors = {}
    batch, ids = [], []
    for record_id in record_ids:
        batch.append(load_gray(by_record[record_id]))
        ids.append(record_id)
        if len(batch) >= batch_size:
            _store_vectors(vectors, model(np.asarray(batch, np.float32)[..., None] / 255.0, training=False).numpy(), ids)
            batch, ids = [], []
    if batch:
        _store_vectors(vectors, model(np.asarray(batch, np.float32)[..., None] / 255.0, training=False).numpy(), ids)
    return vectors


def _store_vectors(destination, matrix, record_ids):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    for record_id, vector in zip(record_ids, matrix):
        destination[record_id] = vector.astype(np.float32)


def representative_generator(rows, count=512, seed=20261021):
    selected = list(rows)
    if count is not None:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:count]
    for row in selected:
        image = load_gray(row).astype(np.float32)[None, ..., None] / 255.0
        yield [image]


def pair_record_ids(pairs):
    return sorted({p["left_record_id"] for p in pairs} | {p["right_record_id"] for p in pairs})


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
