"""Versioned LFW + optical camera accuracy experiments (host training).

This runner deliberately loads only the frozen LFW train/stopping/calibration/
development roles. It never resolves either future-final role. Camera crops are
derived from the saved RGB565 frames using the maintained camera-space geometry
and native OpenMV RGB565 interpolation. A camera crop is never silently treated
as valid when FOMO returns zero or multiple accepted clusters.

Run with work/model-training/.venv/bin/python. See
work/project-research-v5/runs/accuracy_search_v1/README.md for commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
RUN_ROOT = STUDY / "runs/accuracy_search_v1"
ROLE_PATH = STUDY / "data/manifests/lfw_roles.jsonl"
KNOWN = ("K01", "K02", "K03", "K04", "K05")
CLASSES = KNOWN + ("OTHER",)
CLASS_ID = {label: index for index, label in enumerate(CLASSES)}
ALLOWED_ROLE_ROWS = {
    "train", "other_train", "stopping", "threshold_calibration",
    "unknown_calibration", "development", "unknown_development",
}

sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/nicla_study"))
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from nicla_study import d_campaign as d
from nicla_study import v6_camera_crop_features48_v2 as camera_replay
from nicla_study.fs import atomic_write_json, sha256_file
from nicla_study.state import resource_lease
from nicla_study.v5_native_rgb import native_rgb
from camera_geometry import centered_roi_clamped, fomo_center_camera_px


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def model_weight_hash(model) -> str:
    h = hashlib.sha256()
    for value in model.get_weights():
        array = np.asarray(value)
        h.update(str(array.shape).encode() + b"\0")
        h.update(array.dtype.str.encode() + b"\0")
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def load_role_rows() -> tuple[list[dict], dict]:
    """Read eligible roles only; never resolve future-final image paths."""
    rows: list[dict] = []
    role_counts: Counter[str] = Counter()
    label_counts: Counter[tuple[str, str]] = Counter()
    manifest = hashlib.sha256()
    with ROLE_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            role = row.get("role")
            if role not in ALLOWED_ROLE_ROWS:
                continue
            if row.get("dataset") != "lfw-v5":
                raise ValueError("unexpected source dataset in the frozen LFW role manifest")
            label = row.get("label")
            if role in ("train", "stopping", "threshold_calibration", "development"):
                if label not in KNOWN:
                    raise ValueError("known role has a label outside the frozen roster")
            elif label != "OTHER":
                raise ValueError("unknown role does not map to OTHER")
            if not row.get("source_group") or not row.get("source_identity"):
                raise ValueError("LFW row lacks group or identity provenance")
            rows.append(row)
            role_counts[role] += 1
            label_counts[(role, label)] += 1
            manifest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    if not rows:
        raise ValueError("no permitted source rows were loaded")
    return rows, {
        "role_counts": dict(role_counts),
        "role_label_counts": {f"{role}:{label}": n for (role, label), n in label_counts.items()},
        "selected_manifest_rows": len(rows),
        "selected_roles_fingerprint_sha256": manifest.hexdigest(),
        "selection": sorted(ALLOWED_ROLE_ROWS),
        "selected_rows_sha256": manifest.hexdigest(),
        "future_final_paths_resolved": False,
    }


def safe_source_path(row: dict) -> Path:
    rel = row.get("path")
    if not isinstance(rel, str) or not rel:
        raise ValueError("source row has no path")
    path = (ROOT / rel).resolve()
    path.relative_to(ROOT)
    return path


def load_source_gray(row: dict) -> np.ndarray:
    path = safe_source_path(row)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = row.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError("LFW source hash mismatch: " + str(row.get("record_id")))
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if list(rgb.shape[:2]) != [int(row.get("height")), int(row.get("width"))]:
        raise ValueError("source image dimensions differ from its manifest")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    if gray.dtype != np.uint8 or gray.shape != (96, 96) or not np.isfinite(gray).all():
        raise ValueError("source image conversion failed the gray96 input contract")
    return gray[:, :, None]


def training_row(row: dict, image: np.ndarray, source_kind: str = "lfw_source") -> dict:
    label = row.get("label")
    role = row.get("role")
    if label not in CLASSES or role not in ("train", "other_train", "stopping"):
        raise ValueError("only LFW train/other_train/stopping rows can be converted")
    return {
        "record_id": str(row["record_id"]), "label": label, "label_id": label,
        "class_index": CLASS_ID[label], "identity_partition": "unknown" if label == "OTHER" else "known",
        "identity_id": str(row["source_identity"]), "source_identity": str(row["source_identity"]),
        "source_group": str(row["source_group"]), "role": role, "source_kind": source_kind,
        "image": image,
    }


def load_lfw_images(rows: list[dict]) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    image_map: dict[str, np.ndarray] = {}
    cache_records = []
    for row in rows:
        image = load_source_gray(row)
        record_id = str(row["record_id"])
        if record_id in image_map:
            raise ValueError("duplicate source record ID")
        image_map[record_id] = image
        cache_records.append({"record_id": record_id, "role": row["role"], "label": row["label"],
                              "source_group": row["source_group"], "source_identity": row["source_identity"],
                              "source_sha256": row["sha256"], "input_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    return rows, image_map, {"rows": len(rows), "input": "RGB source JPEG -> grayscale -> area resize 96x96 uint8",
                             "conversion_records_sha256": canonical_hash(cache_records)}


def load_camera_images(roi_side: int, allowed_roles: set[str] | None = None) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """Replay explicitly eligible old-camera roles, checking roles before opening frames.

    ``allowed_roles`` is used by grouped camera adaptation after the selected
    epoch/model has been frozen, so no camera calibration or development frame
    can be opened during fit or stopping.
    """
    if roi_side not in (64, 80, 96, 120):
        raise ValueError("camera ROI scout must use one of 64, 80, 96, or 120 camera px")
    manifests = [
        ("camera_train_calibration", "LATEST_ADAPTATION.txt", 135),
        ("optical_development", "LATEST_SESSION.txt", 100),
    ]
    rows = []
    image_map: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    skipped_roles: Counter[str] = Counter()
    records = []
    selection_manifest_hashes = {}
    for set_name, pointer, expected_count in manifests:
        receipt_path, receipts = camera_replay.latest_receipts(pointer)
        if len(receipts) != expected_count:
            raise ValueError("unexpected frozen capture count for " + set_name)
        selection_manifest_hashes[set_name] = camera_replay.sha256_file(receipt_path)
        for entry in receipts:
            request, meta = entry["request"], entry["metadata"]
            source = request.get("source", {})
            if set_name == "camera_train_calibration":
                role = str(source.get("role"))
                if role not in ("train", "other_train", "threshold_calibration", "unknown_calibration"):
                    # Ineligible and future-final roles are skipped before any frame path is opened.
                    continue
                label = request.get("expected_truth")
                if label is None:
                    label = source.get("label", "OTHER")
                phase = request.get("phase")
            else:
                role = str(source.get("role", request.get("expected_truth", "OTHER")))
                if role not in ("development", "unknown_development"):
                    skipped_roles[role] += 1
                    continue
                label = request.get("expected_truth")
                if label is None:
                    label = source.get("label", "OTHER")
                phase = "development"
            if allowed_roles is not None and role not in allowed_roles:
                skipped_roles[role] += 1
                continue
            label = str(label)
            if label not in CLASSES:
                raise ValueError("eligible camera role has label outside frozen class roster")
            if ((role in ("train", "stopping", "threshold_calibration", "development") and label not in KNOWN) or
                (role in ("other_train", "unknown_calibration", "unknown_development") and label != "OTHER")):
                raise ValueError("eligible camera role/label contract failed")
            capture_id = str(request["capture_id"])
            folder = (ROOT / entry["directory"]).resolve()
            folder.relative_to(ROOT)
            raw_path = folder / "camera_rgb565.bin"
            frame_file = next((item for item in entry["files"] if item.get("name") == "camera_rgb565.bin"), None)
            if frame_file is None:
                raise ValueError("camera frame absent from capture receipt")
            frame_payload = raw_path.read_bytes()
            frame_sha = hashlib.sha256(frame_payload).hexdigest()
            if len(frame_payload) != frame_file["bytes"] or frame_sha != frame_file["sha256"]:
                raise ValueError("camera frame hash mismatch: " + capture_id)
            if [int(meta["width"]), int(meta["height"])] != [240, 240]:
                raise ValueError("unexpected saved camera frame geometry")
            raw_boxes = camera_replay.clusters_for(meta)
            boxes = [item for item in raw_boxes if camera_replay.strict_score_pass(item[1], 0.5)]
            valid = len(boxes) == 1
            counts[f"{role}:{label}:cluster_count_{len(boxes)}"] += 1
            roi = center = None
            input_digest = None
            if valid:
                pixels565 = camera_replay.rgb565_array(frame_payload)
                rect = boxes[0][0]
                center = fomo_center_camera_px(rect, 240, 240)
                roi = centered_roi_clamped(center, roi_side, 240, 240)
                rgb = native_rgb(pixels565, roi=roi, width=96, height=96)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
                if gray.shape != (96, 96, 1) or gray.dtype != np.uint8:
                    raise ValueError("camera ROI conversion failed gray96 uint8 contract")
                image_map[capture_id] = gray
                input_digest = hashlib.sha256(gray.tobytes()).hexdigest()
            row = {"request": request, "source": source, "folder": folder, "capture_set": set_name,
                   "phase": phase, "role": role, "label": label, "boxes": boxes, "raw_boxes": raw_boxes,
                   "camera_frame_sha256": frame_sha, "capture_receipt_sha256": camera_replay.sha256_file(folder / "RECEIPT.json"),
                   "manifest_sha256": selection_manifest_hashes[set_name]}
            rows.append(row)
            records.append({"capture_id": capture_id, "role": role, "label": label,
                            "source_group": source.get("source_group"),
                            "source_identity": source.get("source_identity"),
                            "capture_set": set_name, "phase": phase,
                            "fomo_clusters_after_strict_0_5": len(boxes),
                            "fomo_raw_clusters": len(raw_boxes), "roi_side_camera_px": roi_side,
                            "center_xy_camera_px": center, "roi_xywh_camera_px": roi,
                            "camera_frame_sha256": frame_sha,
                            "model_input_sha256": input_digest, "valid_input": valid})
    return rows, image_map, {
        "rows": len(records), "input": f"RGB565 sensor240x240 -> camera-space ROI {roi_side}px -> OpenMV native RGB888 -> gray96 uint8",
        "detection_rule": "one FOMO cluster with existing strict float32 threshold >0.5; center coordinates already sensor px",
        "role_cluster_counts": dict(counts), "input_records_sha256": canonical_hash(records),
        "selected_receipt_manifests_sha256": selection_manifest_hashes,
        "future_final_frames_opened": False,
        "skipped_roles_metadata_only": dict(skipped_roles),
        "private_input_records": records,
    }


def load_camera_train_pool(roi_side: int) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """Load only camera train/other_train frames, checking role before frame I/O."""
    if roi_side not in (64, 80, 96, 120):
        raise ValueError("camera ROI scout must use one of 64, 80, 96, or 120 camera px")
    receipt_path, receipts = camera_replay.latest_receipts("LATEST_ADAPTATION.txt")
    if len(receipts) != 135:
        raise ValueError("unexpected frozen camera adaptation capture count")
    receipt_manifest_sha = camera_replay.sha256_file(receipt_path)
    selected, image_map = [], {}
    skipped_roles: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    private_records = []
    for entry in receipts:
        request, meta = entry["request"], entry["metadata"]
        source = request.get("source", {})
        role = str(source.get("role"))
        if role not in ("train", "other_train"):
            # This is the hard boundary: calibration/development/future-final
            # frame paths are not constructed or opened by this training preflight.
            skipped_roles[role] += 1
            continue
        label = request.get("expected_truth")
        if label is None:
            label = source.get("label", "OTHER")
        label = str(label)
        if (role == "train" and label not in KNOWN) or (role == "other_train" and label != "OTHER"):
            raise ValueError("camera training role/label contract failed")
        identity = source.get("source_identity")
        group = source.get("source_group")
        if not identity or not group:
            raise ValueError("camera training row lacks source identity or group")
        folder = (ROOT / entry["directory"]).resolve()
        folder.relative_to(ROOT)
        frame_file = next((item for item in entry["files"] if item.get("name") == "camera_rgb565.bin"), None)
        if frame_file is None:
            raise ValueError("camera frame absent from training capture receipt")
        frame_payload = (folder / "camera_rgb565.bin").read_bytes()
        frame_sha = hashlib.sha256(frame_payload).hexdigest()
        if len(frame_payload) != frame_file["bytes"] or frame_sha != frame_file["sha256"]:
            raise ValueError("camera train frame hash mismatch: " + str(request.get("capture_id")))
        if [int(meta["width"]), int(meta["height"])] != [240, 240]:
            raise ValueError("unexpected saved camera frame geometry")
        raw_boxes = camera_replay.clusters_for(meta)
        boxes = [item for item in raw_boxes if camera_replay.strict_score_pass(item[1], 0.5)]
        valid = len(boxes) == 1
        cluster_counts[f"{role}:{label}:cluster_count_{len(boxes)}"] += 1
        capture_id = str(request["capture_id"])
        input_digest = None
        if valid:
            pixels565 = camera_replay.rgb565_array(frame_payload)
            center = fomo_center_camera_px(boxes[0][0], 240, 240)
            roi = centered_roi_clamped(center, roi_side, 240, 240)
            rgb = native_rgb(pixels565, roi=roi, width=96, height=96)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
            if gray.shape != (96, 96, 1) or gray.dtype != np.uint8:
                raise ValueError("camera train ROI conversion failed gray96 uint8 contract")
            image_map[capture_id] = gray
            input_digest = hashlib.sha256(gray.tobytes()).hexdigest()
        selected.append({"record_id": capture_id, "capture_id": capture_id, "label": label,
                         "role": role, "source_identity": str(identity), "source_group": str(group),
                         "capture_set": "camera_train_calibration", "phase": "train",
                         "camera_frame_sha256": frame_sha, "camera_receipt_manifest_sha256": receipt_manifest_sha,
                         "fomo_clusters_after_strict_0_5": len(boxes), "valid_input": valid,
                         "image_sha256": input_digest, "folder": folder, "request": request,
                         "source": source})
        private_records.append({"capture_id": capture_id, "role": role, "label": label,
                                "source_identity": str(identity), "source_group": str(group),
                                "camera_frame_sha256": frame_sha, "roi_side_camera_px": roi_side,
                                "fomo_clusters_after_strict_0_5": len(boxes),
                                "fomo_raw_clusters": len(raw_boxes), "valid_input": valid,
                                "model_input_sha256": input_digest})
    return selected, image_map, {
        "eligible_rows": len(selected), "valid_rows": len(image_map),
        "selected_roles": sorted({row["role"] for row in selected}),
        "selected_role_label_counts": dict(Counter(f"{row['role']}:{row['label']}" for row in selected)),
        "valid_role_label_counts": dict(Counter(f"{row['role']}:{row['label']}" for row in selected if row["valid_input"])),
        "role_cluster_counts": dict(cluster_counts), "skipped_role_counts_metadata_only": dict(skipped_roles),
        "receipt_manifest_sha256": receipt_manifest_sha,
        "training_inputs_sha256": canonical_hash(private_records),
        "input": f"RGB565 sensor240x240 -> camera-space ROI {roi_side}px -> OpenMV native RGB888 -> gray96 uint8",
        "future_final_frames_opened": False, "calibration_or_development_frame_paths_opened": False,
        "private_input_records": private_records,
    }


def split_camera_training_groups(rows: list[dict], images: dict[str, np.ndarray], seed: int,
                                 stop_fraction: float = 0.2) -> tuple[list[dict], np.ndarray, list[dict], np.ndarray, dict]:
    """Split known photos by source group and OTHER by identity, before balanced sampling."""
    if not 0 < stop_fraction < 0.5:
        raise ValueError("camera stop fraction must be in (0, 0.5)")
    valid_rows = [row for row in rows if row["valid_input"] and row["record_id"] in images]
    if not valid_rows:
        raise ValueError("no valid camera training crops")
    groups_by_label: dict[str, dict[str, list[dict]]] = {label: defaultdict(list) for label in CLASSES}
    for row in valid_rows:
        group = row["source_identity"] if row["label"] == "OTHER" else row["source_group"]
        groups_by_label[row["label"]][str(group)].append(row)
    stop_group_values: dict[str, set[str]] = {}
    fit_raw, stop_raw = [], []
    for label in CLASSES:
        groups = groups_by_label[label]
        if len(groups) < 2:
            raise ValueError("camera grouped split needs at least two usable groups for " + label)
        n_stop = max(1, int(math.floor(len(groups) * stop_fraction)))
        n_stop = min(n_stop, len(groups) - 1)
        ordered = sorted(groups, key=lambda value: hashlib.sha256(
            ("v8-camera-stop:" + str(seed) + ":" + label + ":" + value).encode()).hexdigest())
        held = set(ordered[:n_stop])
        stop_group_values[label] = held
        for group, members in groups.items():
            if group in held:
                stop_raw.extend(members)
            else:
                fit_raw.extend(members)
    fit_rows = [training_row({**row, "role": row["role"]}, images[row["record_id"]],
                             source_kind="camera_replay_train") for row in fit_raw]
    stop_rows = [training_row({**row, "role": "stopping"}, images[row["record_id"]],
                              source_kind="camera_replay_internal_stopping") for row in stop_raw]
    fit_keys = set()
    stop_keys = set()
    for row in fit_rows:
        key = (row["label"], row["identity_id"] if row["label"] == "OTHER" else row["source_group"])
        fit_keys.add(key)
    for row in stop_rows:
        key = (row["label"], row["identity_id"] if row["label"] == "OTHER" else row["source_group"])
        stop_keys.add(key)
    if fit_keys & stop_keys:
        raise AssertionError("camera fit/stopping source groups or OTHER identities overlap")
    fit_x = np.stack([row["image"] for row in fit_rows]).astype(np.uint8)
    stop_x = np.stack([row["image"] for row in stop_rows]).astype(np.uint8)
    fit_groups_by_label = Counter(row["label"] for row in fit_rows)
    stop_groups_by_label = Counter(row["label"] for row in stop_rows)
    fit_other_ids = {row["identity_id"] for row in fit_rows if row["label"] == "OTHER"}
    stop_other_ids = {row["identity_id"] for row in stop_rows if row["label"] == "OTHER"}
    if fit_other_ids & stop_other_ids:
        raise AssertionError("camera OTHER identity crossed fit/stopping boundary")
    split = {
        "split_seed": int(seed), "stop_fraction": float(stop_fraction),
        "group_rule": "known rows split by source_group within class; OTHER split by source_identity; every member of a group stays on one side",
        "fit_rows_by_label": {label: int(fit_groups_by_label[label]) for label in CLASSES},
        "stopping_rows_by_label": {label: int(stop_groups_by_label[label]) for label in CLASSES},
        "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
        "fit_other_identity_groups": len(fit_other_ids), "stopping_other_identity_groups": len(stop_other_ids),
        "fit_stopping_group_overlap": 0,
        "fit_group_counts_by_label": {label: len(groups_by_label[label]) - len(stop_group_values[label]) for label in CLASSES},
        "stopping_group_counts_by_label": {label: len(stop_group_values[label]) for label in CLASSES},
        "group_keys_sha256": canonical_hash({label: {"fit": sorted(group for group in groups_by_label[label] if group not in stop_group_values[label]),
                                                       "stopping": sorted(stop_group_values[label])} for label in CLASSES}),
    }
    return fit_rows, fit_x, stop_rows, stop_x, split


def make_split(source_rows: list[dict], source_images: dict[str, np.ndarray], seed: int) -> tuple[list[dict], np.ndarray, list[dict], np.ndarray, dict, set[str]]:
    train_source = [r for r in source_rows if r["role"] in ("train", "other_train")]
    known_stop = [r for r in source_rows if r["role"] == "stopping"]
    other_ids = sorted({str(r["source_identity"]) for r in train_source if r["role"] == "other_train"})
    if len(other_ids) < 50:
        raise ValueError("insufficient diverse OTHER train identities for a grouped internal holdout")
    # Fixed identity holdout across roots keeps model seed comparisons matched.
    stop_other_ids = set(sorted(other_ids, key=lambda value: hashlib.sha256(("v7stop:" + value).encode()).hexdigest())[:40])
    fit_raw = [r for r in train_source if not (r["role"] == "other_train" and r["source_identity"] in stop_other_ids)]
    stop_raw = known_stop + [r for r in train_source if r["role"] == "other_train" and r["source_identity"] in stop_other_ids]
    fit_rows = [training_row(r, source_images[str(r["record_id"])]) for r in fit_raw]
    stop_rows = [training_row({**r, "role": "stopping"}, source_images[str(r["record_id"])]) for r in stop_raw]
    fit_groups = {(r["label"], r["source_group"]) for r in fit_rows}
    stop_groups = {(r["label"], r["source_group"]) for r in stop_rows}
    if fit_groups & stop_groups:
        raise AssertionError("fit/stopping source groups overlap")
    fit_x = np.stack([r["image"] for r in fit_rows]).astype(np.uint8)
    stop_x = np.stack([r["image"] for r in stop_rows]).astype(np.uint8)
    details = {
        "split_seed": 713001,
        "training_root_seed": int(seed),
        "known_fit_rows_by_label": {label: sum(r["label"] == label for r in fit_rows) for label in KNOWN},
        "known_stopping_rows_by_label": {label: sum(r["label"] == label for r in stop_rows) for label in KNOWN},
        "other_fit_rows": sum(r["label"] == "OTHER" for r in fit_rows),
        "other_fit_distinct_identities": len({r["identity_id"] for r in fit_rows if r["label"] == "OTHER"}),
        "other_stopping_rows": sum(r["label"] == "OTHER" for r in stop_rows),
        "other_stopping_distinct_identities": len({r["identity_id"] for r in stop_rows if r["label"] == "OTHER"}),
        "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
        "fit_stop_source_group_overlap": 0,
        "other_stop_identity_ids_sha256": canonical_hash(sorted(stop_other_ids)),
        "split_rule": "LFW stopping role for known labels; 40/200 OTHER identities held out from other_train by deterministic hash; no calibration/development used for epoch selection",
    }
    return fit_rows, fit_x, stop_rows, stop_x, details, stop_other_ids


def balanced_batch_indices(fit_rows: list[dict], seed: int, batch_size: int = 24,
                           examples_per_class: int = 4) -> list[list[int]]:
    """Make deterministic balanced batches, with a distinct OTHER identity per slot."""
    if examples_per_class <= 0 or batch_size != examples_per_class * len(CLASSES):
        raise ValueError("balanced batch size must equal six classes times examples_per_class")
    class_pools = {label: [i for i, row in enumerate(fit_rows) if row["label"] == label]
                   for label in CLASSES}
    if any(not values for values in class_pools.values()):
        raise ValueError("fit split lacks a class")
    other_by_id: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(fit_rows):
        if row["label"] == "OTHER":
            other_by_id[row["identity_id"]].append(i)
    if len(other_by_id) < 10:
        raise ValueError("balanced OTHER batches require ten or more fit identities")
    n_batches = int(math.ceil(max(len(class_pools[label]) for label in KNOWN) / float(examples_per_class)))
    rng = np.random.default_rng(int(seed))
    batches = []
    identities = np.asarray(sorted(other_by_id), dtype=object)
    for _ in range(n_batches):
        indices = []
        for label in KNOWN:
            indices.extend(int(x) for x in rng.choice(class_pools[label], size=examples_per_class, replace=True))
        picked_identities = rng.choice(identities, size=examples_per_class, replace=False)
        for identity in picked_identities:
            indices.append(int(rng.choice(other_by_id[str(identity)])))
        rng.shuffle(indices)
        if len(indices) != batch_size:
            raise AssertionError("balanced batch size differs from configuration")
        batches.append(indices)
    return batches


def validate_camera_adaptation_batches(fit_rows: list[dict], batches: list[list[int]],
                                        batch_size: int = 24, examples_per_class: int = 4) -> dict:
    """Prove every adaptation gradient batch comes only from camera fit rows."""
    if not fit_rows or not batches:
        raise ValueError("camera adaptation requires non-empty fit rows and batches")
    fit_record_ids = {str(row["record_id"]) for row in fit_rows}
    for batch in batches:
        if len(batch) != batch_size or any(index < 0 or index >= len(fit_rows) for index in batch):
            raise AssertionError("camera adaptation batch references an invalid fit-row index")
        rows = [fit_rows[index] for index in batch]
        if any(row.get("source_kind") != "camera_replay_train" or
               row.get("role") not in ("train", "other_train") for row in rows):
            raise AssertionError("a non-camera-train role entered the adaptation gradient stream")
        if any(str(row["record_id"]) not in fit_record_ids for row in rows):
            raise AssertionError("adaptation batch references a row outside its grouped fit split")
        counts = Counter(row["label"] for row in rows)
        if counts != Counter({label: examples_per_class for label in CLASSES}):
            raise AssertionError("camera adaptation batch is not balanced over the six labels")
        other_ids = {row["identity_id"] for row in rows if row["label"] == "OTHER"}
        if len(other_ids) != examples_per_class:
            raise AssertionError("camera adaptation batch repeats an OTHER identity")
    return {"batches": len(batches), "rows_per_batch": batch_size,
            "fit_record_ids": len(fit_record_ids), "camera_only": True,
            "balanced_six_class": True, "distinct_other_identities_per_batch": examples_per_class}


def require_camera_adaptation_evaluation_freeze(state: dict) -> None:
    """Refuse any calibration/development load before selection is committed."""
    if state.get("status") != "evaluation_pending" or state.get("model_selection_frozen") is not True:
        raise RuntimeError("camera evaluation roles are sealed until model selection is frozen")


def softmax32(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 6 or not np.isfinite(z).all():
        raise ValueError("classifier logits violate finite float32 six-class contract")
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted.astype(np.float32)).astype(np.float32)
    denom = np.sum(e, axis=1, keepdims=True, dtype=np.float32)
    result = (e / denom).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result < 0) or not np.allclose(result.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError("six-way probability contract failed")
    return result


def balanced_metrics(rows: list[dict], logits: np.ndarray) -> dict:
    probs = softmax32(logits)
    target = np.asarray([CLASS_ID[r["label"]] for r in rows], dtype=np.int64)
    pred = np.argmax(logits, axis=1)
    losses = -np.log(np.maximum(probs[np.arange(len(rows)), target], np.float32(1e-8)))
    per = {}
    for label in CLASSES:
        ids = [i for i, row in enumerate(rows) if row["label"] == label]
        per[label] = {"n": len(ids), "accuracy": float(np.mean(pred[ids] == target[ids])) if ids else None,
                      "cross_entropy": float(np.mean(losses[ids])) if ids else None}
    active = [v for v in per.values() if v["n"]]
    return {"rows": len(rows), "accuracy": float(np.mean(pred == target)) if len(rows) else None,
            "macro_accuracy": float(np.mean([v["accuracy"] for v in active])) if active else None,
            "macro_cross_entropy": float(np.mean([v["cross_entropy"] for v in active])) if active else None,
            "per_class": per}


def eval_logits(model, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (96, 96, 1) or values.dtype != np.uint8:
        raise ValueError("evaluation inputs must be uint8 gray96 NHWC")
    out = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size].astype(np.float32) / np.float32(255)
        logits = np.asarray(model(batch, training=False).numpy(), dtype=np.float32)
        if logits.shape != (len(batch), 6) or not np.isfinite(logits).all():
            raise ValueError("nonfinite or malformed model output")
        out.append(logits)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 6), dtype=np.float32)


def threshold_from_calibration(rows: list[dict], logits: np.ndarray) -> dict:
    probs = softmax32(logits)
    scores = np.max(probs[:, :5], axis=1).astype(np.float32)
    pred = np.argmax(probs, axis=1)
    known = [i for i, row in enumerate(rows) if row["label"] in KNOWN]
    unknown = [i for i, row in enumerate(rows) if row["label"] == "OTHER"]
    if not known or not unknown:
        raise ValueError("threshold calibration needs known and unknown groups")
    groups: dict[str, list[int]] = defaultdict(list)
    for i in unknown:
        identity = str(rows[i].get("source_identity") or rows[i].get("identity_id") or rows[i].get("source_group"))
        groups[identity].append(i)
    max_unknown_by_group = {identity: float(max(scores[i] for i in ids)) for identity, ids in groups.items()}
    max_unknown = np.float32(max(max_unknown_by_group.values()))
    threshold = np.nextafter(max_unknown, np.float32(np.inf), dtype=np.float32)
    accepted_known = [i for i in known if pred[i] < 5 and scores[i] >= threshold]
    correct_known = [i for i in accepted_known if pred[i] == CLASS_ID[rows[i]["label"]]]
    return {
        "score": "float32 max known-class softmax; not claimed calibrated probability",
        "acceptance_rule": "argmax must be K01-K05 and max-known score >= nextafter(max calibration OTHER identity score, +inf) in float32",
        "threshold_float32": float(threshold), "calibration_known_rows": len(known),
        "calibration_unknown_rows": len(unknown), "calibration_unknown_identity_groups": len(groups),
        "calibration_max_unknown_score": float(max_unknown),
        "calibration_false_unknown_groups": 0,
        "calibration_known_correct_accepted": len(correct_known),
        "calibration_known_correct_acceptance": len(correct_known) / max(1, len(known)),
        "threshold_candidates_from_calibration_only": True,
        "unknown_group_max_scores_sha256": canonical_hash(max_unknown_by_group),
    }


def evaluate_open_set(rows: list[dict], logits: np.ndarray, calibration: dict,
                      valid: np.ndarray | None = None) -> dict:
    probs = softmax32(logits)
    score = np.max(probs[:, :5], axis=1).astype(np.float32)
    pred = np.argmax(probs, axis=1)
    threshold = np.float32(calibration["threshold_float32"])
    is_valid = np.ones(len(rows), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    accepted = is_valid & (pred < 5) & (score >= threshold)
    known_ids = [i for i, row in enumerate(rows) if row["label"] in KNOWN]
    unknown_ids = [i for i, row in enumerate(rows) if row["label"] == "OTHER"]
    known_correct_top1 = int(sum(bool(is_valid[i]) and int(pred[i]) == CLASS_ID[rows[i]["label"]]
                                 for i in known_ids))
    correct_accept = int(sum(bool(accepted[i]) and int(pred[i]) == CLASS_ID[rows[i]["label"]]
                              for i in known_ids))
    wrong_accept = int(sum(bool(accepted[i]) and int(pred[i]) != CLASS_ID[rows[i]["label"]]
                            for i in known_ids))
    groups: dict[str, list[int]] = defaultdict(list)
    for i in unknown_ids:
        group = str(rows[i].get("source_identity") or rows[i].get("identity_id") or rows[i].get("source_group") or rows[i].get("record_id"))
        groups[group].append(i)
    false_groups = int(sum(any(bool(accepted[i]) for i in ids) for ids in groups.values()))
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_ids if rows[i]["label"] == label]
        valid_ids = [i for i in ids if is_valid[i]]
        per_person[label] = {
            "rows": len(ids), "valid_detection_rows": len(valid_ids),
            "raw_top1_correct": int(sum(int(pred[i]) == CLASS_ID[label] for i in valid_ids)),
            "correct_accepted": int(sum(bool(accepted[i]) and int(pred[i]) == CLASS_ID[label] for i in ids)),
            "wrong_known_accepted": int(sum(bool(accepted[i]) and int(pred[i]) != CLASS_ID[label] for i in ids)),
            "coverage": int(sum(bool(accepted[i]) for i in ids)) / max(1, len(ids)),
        }
    return {
        "threshold_float32": float(threshold), "known_rows": len(known_ids),
        "known_detection_valid_rows": int(sum(is_valid[i] for i in known_ids)),
        "known_conditional_raw_top1": known_correct_top1 / max(1, sum(is_valid[i] for i in known_ids)),
        "known_raw_top1_end_to_end": known_correct_top1 / max(1, len(known_ids)),
        "known_correct_accepted": correct_accept,
        "known_correct_acceptance_end_to_end": correct_accept / max(1, len(known_ids)),
        "known_wrong_accepted": wrong_accept,
        "known_coverage_end_to_end": sum(accepted[i] for i in known_ids) / max(1, len(known_ids)),
        "unknown_rows": len(unknown_ids), "unknown_identity_groups": len(groups),
        "unknown_detection_valid_rows": int(sum(is_valid[i] for i in unknown_ids)),
        "unknown_false_accepted_rows": int(sum(bool(accepted[i]) for i in unknown_ids)),
        "unknown_false_accepted_identity_groups": int(false_groups),
        "per_person": per_person,
        "denominator_note": "detection misses and multiple/ambiguous clusters remain in end-to-end denominators; unknown risk unit is source identity group",
    }


def model_config(config_path: Path) -> tuple[dict, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "accuracy-search-config-v1":
        raise ValueError("unsupported accuracy search config schema")
    required = ("run_id", "seed", "alpha", "pretrained", "roi_side_camera_px", "epochs_head", "epochs_finetune",
                "batch_size", "examples_per_class_per_batch", "lr_head", "lr_finetune", "patience")
    if any(k not in config for k in required):
        raise ValueError("config lacks required field")
    if config["alpha"] not in (0.1, 0.25) or int(config["seed"]) <= 0:
        raise ValueError("unsupported alpha or seed")
    if int(config["batch_size"]) != 6 * int(config["examples_per_class_per_batch"]):
        raise ValueError("batch_size must balance all six classes equally")
    if min(int(config["epochs_head"]), int(config["epochs_finetune"]), int(config["patience"])) <= 0:
        raise ValueError("epoch and patience budgets must be positive")
    if min(float(config["lr_head"]), float(config["lr_finetune"])) <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 < float(config.get("unfreeze_top_fraction", 0.25)) <= 1:
        raise ValueError("unfreeze_top_fraction must be in (0,1]")
    if int(config["roi_side_camera_px"]) not in (64, 80, 96, 120):
        raise ValueError("unsupported camera ROI side")
    if config.get("training_mode") == "grouped_camera_adaptation_from_parent":
        if float(config["alpha"]) != 0.1:
            raise ValueError("grouped camera adaptation must continue the selected alpha0.1 parent")
        if config.get("camera_fit_roles") != ["train", "other_train"]:
            raise ValueError("camera adaptation fit role allowlist must be train/other_train only")
        if config.get("camera_only_adaptation_batches") is not True:
            raise ValueError("camera adaptation gradients must be camera-only")
        if config.get("development_roles_loaded_during_training") is not False or \
                config.get("threshold_calibration_roles_loaded_during_training") is not False:
            raise ValueError("camera calibration/development must remain sealed through model selection")
        if config.get("training_launch_enabled") is True:
            runner_sha = config.get("runner_sha256")
            if not isinstance(runner_sha, str) or len(runner_sha) != 64:
                raise ValueError("launch-bound camera adaptation config must pin the runner hash")
            for field in ("expected_camera_eligible_rows", "expected_camera_valid_rows",
                          "expected_camera_fit_rows", "expected_camera_stopping_rows",
                          "expected_max_gradient_updates"):
                if field not in config:
                    raise ValueError("launch-bound camera adaptation config lacks " + field)
    return config, hashlib.sha256(config_path.read_bytes()).hexdigest()


def contract_checks() -> dict:
    """Fast synthetic checks for labels, threshold precision, and camera misses."""
    rows = [
        {"label": "K01", "source_identity": "known-a", "role": "threshold_calibration"},
        {"label": "K02", "source_identity": "known-b", "role": "threshold_calibration"},
        {"label": "OTHER", "source_identity": "other-a", "role": "unknown_calibration"},
        {"label": "OTHER", "source_identity": "other-a", "role": "unknown_calibration"},
        {"label": "OTHER", "source_identity": "other-b", "role": "unknown_calibration"},
    ]
    logits = np.asarray([
        [4, 0, 0, 0, 0, -1], [0, 4, 0, 0, 0, -1],
        [2, 0, 0, 0, 0, -1], [3, 0, 0, 0, 0, -1], [2.5, 0, 0, 0, 0, -1],
    ], dtype=np.float32)
    calibration = threshold_from_calibration(rows, logits)
    scores = np.max(softmax32(logits[:, :].astype(np.float32))[:, :5], axis=1)
    # The returned threshold must strictly exceed the maximum unknown group score.
    unknown_group_max = max(scores[2], scores[4])
    if not (np.float32(calibration["threshold_float32"]) > unknown_group_max):
        raise AssertionError("float32 threshold did not clear the largest unknown identity score")
    metrics = evaluate_open_set(rows, logits, calibration, np.asarray([True, True, True, False, True]))
    if metrics["unknown_rows"] != 3 or metrics["unknown_identity_groups"] != 2 or metrics["unknown_detection_valid_rows"] != 2:
        raise AssertionError("open-set denominators lost unknown rows/groups/detection misses")
    # RESULT is written with allow_nan=False; counters must be ordinary JSON integers,
    # even when the metric path is fed NumPy boolean arrays.
    json.dumps(metrics, sort_keys=True, allow_nan=False)
    for field in ("known_rows", "known_detection_valid_rows", "known_correct_accepted",
                  "known_wrong_accepted", "unknown_rows", "unknown_false_accepted_rows",
                  "unknown_false_accepted_identity_groups"):
        if type(metrics[field]) is not int:
            raise AssertionError("open-set metric counter is not a JSON-native int: " + field)
    if any(type(person["raw_top1_correct"]) is not int for person in metrics["per_person"].values()):
        raise AssertionError("per-person metric counter is not a JSON-native int")
    rejected = False
    try:
        training_row({"role": "future_final", "label": "K01", "record_id": "blocked",
                      "source_identity": "blocked", "source_group": "blocked"}, np.zeros((96, 96, 1), np.uint8))
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("future-final role was accepted as training data")
    return {"threshold_float32_strictly_above_unknown_identity_max": True,
            "unknown_image_identity_and_detection_denominators": True,
            "future_final_training_row_rejected": True}


def data_preflight(config_path: Path) -> dict:
    config, config_sha = model_config(config_path)
    checkpoint_path = ROOT / d.EI_CHECKPOINTS[float(config["alpha"])]
    expected_checkpoint_sha = d.EI_HASHES[float(config["alpha"])]
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise ValueError("configured MobileNetV1 alpha checkpoint is absent or hash-mismatched")
    source_rows, source_receipt = load_role_rows()
    source_rows, source_images, source_inputs = load_lfw_images(source_rows)
    fit_rows, fit_images, stop_rows, stop_images, split_receipt, stop_other_ids = make_split(
        source_rows, source_images, int(config["seed"]))
    batch_size = int(config["batch_size"])
    examples_per_class = int(config["examples_per_class_per_batch"])
    batches = balanced_batch_indices(fit_rows, int(config["seed"]) + 424242, batch_size, examples_per_class)
    for batch in batches[: min(20, len(batches))]:
        labels = Counter(fit_rows[i]["label"] for i in batch)
        other_ids = [fit_rows[i]["identity_id"] for i in batch if fit_rows[i]["label"] == "OTHER"]
        if labels != Counter({label: examples_per_class for label in CLASSES}) or len(set(other_ids)) != examples_per_class:
            raise AssertionError("balanced sampler violated per-class count or OTHER identity uniqueness")
    camera_rows, camera_images, camera_receipt = load_camera_images(int(config["roi_side_camera_px"]))
    camera_receipt.pop("private_input_records", None)
    role_histogram = Counter(f"{row['role']}:{row['label']}" for row in camera_rows)
    max_known = max(split_receipt["known_fit_rows_by_label"].values())
    epochs_upper = int(config["epochs_head"]) + int(config["epochs_finetune"])
    batch_count = int(math.ceil(max_known / float(examples_per_class)))
    report = {
        "schema": "accuracy-search-preflight-v1", "run_id": config["run_id"],
        "config_sha256": config_sha, "created_utc": utc_now(),
        "initialization_checkpoint": {"alpha": float(config["alpha"]),
                                      "path": str(checkpoint_path.relative_to(ROOT)),
                                      "sha256": sha256_file(checkpoint_path),
                                      "corpus_description": "Edge Impulse generic transfer checkpoint; not verified face-pretrained"},
        "source_rows": source_receipt, "source_inputs": source_inputs,
        "split": split_receipt,
        "camera_replay": {**camera_receipt, "eligible_role_label_rows": dict(role_histogram),
                          "valid_gray96_inputs": len(camera_images)},
        "training_controls": {
            "batch_size": batch_size, "examples_per_class_per_batch": examples_per_class,
            "distinct_other_identities_per_batch": examples_per_class,
            "batches_per_epoch": batch_count,
            "sample_presentations_per_epoch": batch_count * batch_size,
            "maximum_scheduled_epochs": epochs_upper,
            "maximum_scheduled_gradient_updates": batch_count * epochs_upper,
            "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
            "fit_source_group_count": len({row["source_group"] for row in fit_rows}),
            "group_disjoint_fit_stopping": split_receipt["fit_stop_source_group_overlap"] == 0,
            "future_final_files_opened": False,
            "threshold_calibration_or_development_used_for_epoch_selection": False,
            "batch_norm_frozen_during_fine_tuning": True,
        },
        "contract_checks": contract_checks(),
        "resource_limits": {key: os.environ.get(key) for key in
                             ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    }
    out = RUN_ROOT / "preflight" / (str(config["run_id"]) + ".json")
    write_json(out, report)
    report["report_path"] = str(out.relative_to(ROOT))
    return report


def camera_adaptation_preflight(config_path: Path) -> dict:
    """Audit a v8 continuation without touching camera cal/dev or training data roles."""
    config, config_sha = model_config(config_path)
    if config.get("training_mode") != "grouped_camera_adaptation_from_parent":
        raise ValueError("camera adaptation preflight requires grouped_camera_adaptation_from_parent")
    runner_path = Path(__file__).resolve()
    runner_sha = sha256_file(runner_path)
    launch_enabled = config.get("training_launch_enabled") is True
    if launch_enabled and config.get("runner_sha256") != runner_sha:
        raise ValueError("launch config runner hash does not match this accuracy runner")
    parent_result_path = (ROOT / str(config["parent_result_path"])).resolve()
    parent_weights_path = (ROOT / str(config["parent_model_weights_path"])).resolve()
    parent_result_path.relative_to(ROOT)
    parent_weights_path.relative_to(ROOT)
    if not parent_result_path.is_file() or sha256_file(parent_result_path) != config["parent_result_sha256"]:
        raise ValueError("parent source scout result does not match frozen config hash")
    if not parent_weights_path.is_file() or sha256_file(parent_weights_path) != config["parent_model_weights_sha256"]:
        raise ValueError("parent source scout selected weights do not match frozen config hash")
    parent_config_path = None
    if config.get("parent_config_path"):
        parent_config_path = (ROOT / str(config["parent_config_path"])).resolve()
        parent_config_path.relative_to(ROOT)
        if (not parent_config_path.is_file() or
                sha256_file(parent_config_path) != config.get("parent_config_sha256")):
            raise ValueError("parent source scout config does not match frozen config hash")
    parent = json.loads(parent_result_path.read_text(encoding="utf-8"))
    if parent.get("status") != "complete_scout_host_only" or parent.get("run_id") != config["parent_run_id"]:
        raise ValueError("parent must be the completed host-only source scout named in this config")
    if parent.get("model", {}).get("initial_checkpoint_sha256") != d.EI_HASHES[float(config["alpha"])]:
        raise ValueError("parent source scout EI alpha checkpoint differs from v8 config")
    if "alpha=0.1" not in parent.get("model", {}).get("architecture", "") or float(config["alpha"]) != 0.1:
        raise ValueError("v8 scout must continue the same alpha 0.1 parent model")
    if parent.get("training", {}).get("best_stopping_phase") != "partial_finetune":
        raise ValueError("parent selected weights were not selected from the expected partial-finetune phase")

    loaded_parent_weight_hash = None
    parent_prediction_weight_hash = None
    if launch_enabled:
        # The selected v7 weights were saved with the top MobileNet layers
        # trainable. Match that trainability before HDF5 loading; Keras changes
        # HDF5 weight ordering when this flag changes.
        parent_model, parent_embedding, parent_init = d.build_classifier(
            ROOT, float(config["alpha"]), True, 6, int(config["seed"]), "v8_camera_parent_verify")
        parent_model.get_layer("head_dropout").rate = 0.0
        d.unfreeze_top_fraction(parent_model, float(config.get("unfreeze_top_fraction", 0.25)))
        parent_model.load_weights(str(parent_weights_path))
        loaded_parent_weight_hash = model_weight_hash(parent_model)
        parent_prediction_info = parent.get("private_evidence", {})
        if parent_prediction_info.get("per_image_predictions") and parent_prediction_info.get("per_image_predictions_sha256"):
            parent_predictions_path = (ROOT / parent_prediction_info["per_image_predictions"]).resolve()
            parent_predictions_path.relative_to(ROOT)
            if (not parent_predictions_path.is_file() or
                    sha256_file(parent_predictions_path) != parent_prediction_info["per_image_predictions_sha256"]):
                raise ValueError("parent prediction receipt does not match the frozen result")
            with parent_predictions_path.open("r", encoding="utf-8") as stream:
                first_prediction = next((line for line in stream if line.strip()), None)
            if first_prediction:
                parent_prediction_weight_hash = json.loads(first_prediction).get("model_weight_sha256")
                if parent_prediction_weight_hash != loaded_parent_weight_hash:
                    raise ValueError("loaded selected parent weights differ from the parent evaluation weights")
        del parent_model, parent_embedding, parent_init

    camera_rows, camera_images, camera_receipt = load_camera_train_pool(int(config["roi_side_camera_px"]))
    private_records = camera_receipt.pop("private_input_records")
    fit_rows, fit_images, stop_rows, stop_images, split = split_camera_training_groups(
        camera_rows, camera_images, int(config["seed"]), float(config["camera_stop_fraction"]))
    batch_size = int(config["batch_size"])
    examples_per_class = int(config["examples_per_class_per_batch"])
    batches = balanced_batch_indices(fit_rows, int(config["seed"]) + 818181, batch_size, examples_per_class)
    batch_contract = validate_camera_adaptation_batches(fit_rows, batches, batch_size, examples_per_class)
    if config.get("expected_camera_fit_rows") is not None and len(fit_rows) != int(config["expected_camera_fit_rows"]):
        raise ValueError("grouped camera fit row count differs from launch-bound expectation")
    if config.get("expected_camera_stopping_rows") is not None and len(stop_rows) != int(config["expected_camera_stopping_rows"]):
        raise ValueError("grouped camera stopping row count differs from launch-bound expectation")
    if config.get("expected_camera_eligible_rows") is not None and int(camera_receipt["eligible_rows"]) != int(config["expected_camera_eligible_rows"]):
        raise ValueError("camera train/other_train eligible row count differs from launch-bound expectation")
    if config.get("expected_camera_valid_rows") is not None and int(camera_receipt["valid_rows"]) != int(config["expected_camera_valid_rows"]):
        raise ValueError("camera train/other_train valid crop count differs from launch-bound expectation")
    max_class_fit = max(split["fit_rows_by_label"][label] for label in KNOWN)
    batches_per_epoch = int(math.ceil(max_class_fit / float(examples_per_class)))
    epochs_upper = int(config["epochs_head"]) + int(config["epochs_finetune"])
    parent_data_roles = parent.get("source_rows_receipt", {}).get("selection", [])
    expected_source_hash = parent.get("source_rows_receipt", {}).get("selected_rows_sha256")
    if "train" not in parent_data_roles or "other_train" not in parent_data_roles or not expected_source_hash:
        raise ValueError("parent result lacks the full permitted LFW source training receipt")

    run_dir = RUN_ROOT / str(config["run_id"])
    private_dir = run_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_path = private_dir / "CAMERA_TRAIN_POOL.jsonl"
    with private_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in private_records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    report = {
        "schema": "accuracy-search-camera-adaptation-preflight-v1",
        "run_id": config["run_id"], "seed": int(config["seed"]), "config_sha256": config_sha,
        "runner_sha256": runner_sha,
        "created_utc": utc_now(), "preflight_only": True, "gradient_updates_completed": 0,
        "training_launch_enabled": launch_enabled,
        "parent_source_scout": {
            "run_id": parent["run_id"], "result_path": str(parent_result_path.relative_to(ROOT)),
            "result_sha256": sha256_file(parent_result_path),
            "selected_model_weights_path": str(parent_weights_path.relative_to(ROOT)),
            "selected_model_weights_sha256": sha256_file(parent_weights_path),
            "selected_source_role_rows_sha256": expected_source_hash,
            "source_roles": parent_data_roles,
            "status": "complete_scout_host_only",
            "parent_config_path": str(parent_config_path.relative_to(ROOT)) if parent_config_path else None,
            "parent_config_sha256": sha256_file(parent_config_path) if parent_config_path else None,
            "loaded_selected_weight_model_sha256": loaded_parent_weight_hash,
            "parent_prediction_model_weight_sha256": parent_prediction_weight_hash,
        },
        "camera_training_data": camera_receipt,
        "grouped_split": split,
        "training_plan": {
            "mode": "continue selected alpha0.1 LFW source weights; train only on camera train/other_train crops",
            "camera_training_source_rows": split["fit_rows"],
            "camera_stopping_source_rows": split["stopping_rows"],
            "camera_examples_per_class_per_batch": examples_per_class,
            "batch_size": batch_size, "other_identity_groups_per_batch": examples_per_class,
            "batches_per_epoch": batches_per_epoch,
            "sample_presentations_per_epoch": batches_per_epoch * batch_size,
            "head_warmup_epochs_max": int(config["epochs_head"]),
            "partial_finetune_epochs_max": int(config["epochs_finetune"]),
            "gradient_updates_max": batches_per_epoch * epochs_upper,
            "camera_fraction_of_adaptation_gradient_presentations": 1.0,
            "batch_norm_frozen": True,
            "stopping_role": "identity/source-group-disjoint holdout from camera train/other_train only",
            "threshold_calibration_used_for_fit_stopping_or_checkpoint_selection": False,
            "development_used_for_fit_stopping_or_checkpoint_selection": False,
            "threshold_or_development_roles_loaded": False,
            "future_final_roles_opened": False,
            "LFW_training_repeated": False,
        },
        "fit_stopping_contract": {
            "fit_group_count_by_class": split["fit_group_counts_by_label"],
            "stopping_group_count_by_class": split["stopping_group_counts_by_label"],
            "fit_stopping_group_overlap": split["fit_stopping_group_overlap"],
            "other_fit_identities": split["fit_other_identity_groups"],
            "other_stopping_identities": split["stopping_other_identity_groups"],
            "other_identity_overlap": 0,
            "fit_rows_only_role_allowlist": ["train", "other_train"],
            "camera_training_roles_opened": sorted({row["role"] for row in camera_rows}),
            "camera_calibration_development_folders_opened": False,
            "future_final_frame_folders_opened": False,
            "strict_detection_misses_or_multi_cluster_rows_used_as_fit": False,
        },
        "post_freeze_evaluation_contract": {
            "calibration_and_development_read_only_after_model_and_epoch_selection_freeze": True,
            "threshold_source": "known plus unknown calibration rows only; threshold from unknown identity max score",
            "development_denominators": "all camera development rows per known label and unknown identity group; FOMO misses and multi-cluster rows count as end-to-end failures",
            "these_roles_loaded_during_preflight": False,
        },
        "private_input_index": {"path": str(private_path.relative_to(ROOT)),
                                 "sha256": sha256_file(private_path), "rows": len(private_records)},
        "contract_checks": {
            "parent_is_completed_alpha010_source_scout": True,
            "camera_fit_and_stop_groups_disjoint": split["fit_stopping_group_overlap"] == 0,
            "unknown_identity_fit_and_stop_disjoint": True,
            "balanced_camera_batches_checked": len(batches),
            "camera_batch_contract": batch_contract,
            "every_adaptation_batch_is_camera_only": True,
            "calibration_development_and_future_final_not_opened": True,
            "exact_miss_denominator_rule_preserved_for_post_freeze_eval": True,
        },
    }
    output_path = RUN_ROOT / "preflight" / (str(config["run_id"]) + ".json")
    write_json(output_path, report)
    report["report_path"] = str(output_path.relative_to(ROOT))
    return report


def train_run(config_path: Path, resume: bool = False) -> dict:
    run_started = time.monotonic()
    config, config_sha = model_config(config_path)
    run_id = str(config["run_id"])
    run_dir = RUN_ROOT / run_id
    private = run_dir / "private"
    run_dir.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "RESULT.json"
    if result_path.exists() and not resume:
        raise FileExistsError("run result exists; use a new immutable run_id")

    cv2.setNumThreads(2)
    tf.keras.utils.set_random_seed(int(config["seed"]))
    source_rows, source_receipt = load_role_rows()
    source_rows, source_images, source_inputs = load_lfw_images(source_rows)
    fit_rows, fit_images, stop_rows, stop_images, split_receipt, stop_other_ids = make_split(
        source_rows, source_images, int(config["seed"]))
    # v1 runner first establishes the full source pool. Camera adaptation is an
    # explicitly configured follow-up, never silently mixed into this baseline.
    if config.get("include_camera_fit"):
        raise NotImplementedError("camera fitting is staged after the source-only control is inspected")

    if not resume and result_path.exists():
        raise FileExistsError(result_path)
    model, embedding_model, init_info = d.build_classifier(
        ROOT, float(config["alpha"]), bool(config["pretrained"]), 6,
        int(config["seed"]), "v7_" + run_id)
    # No stochastic dropout: epoch/batch sampling and augmentation are derived
    # from fixed root/epoch seeds, so restoring the TensorFlow optimizer checkpoint
    # resumes the same data stream exactly.
    model.get_layer("head_dropout").rate = 0.0
    info = {k: v for k, v in init_info.items() if k not in ("full_reference_model", "feature_model")}
    del init_info
    del embedding_model
    expected_backbone_hash = info.get("checkpoint_sha256")
    model_summary = {
        "architecture": f"MobileNetV1 alpha={config['alpha']} grayscale96 + Dense64-ReLU + six-way logits",
        "initialization": info.get("initialization"),
        "initial_checkpoint_sha256": expected_backbone_hash,
        "head_dropout_rate": 0.0,
        "trainable_backbone_fraction": float(config.get("unfreeze_top_fraction", 0.25)),
        "batch_norm_frozen": True,
        "input": "uint8 gray96 / 255 -> float32 [0,1], NHWC",
        "classes": list(CLASSES), "parameter_count": int(model.count_params()),
    }
    initial_fit = balanced_metrics(fit_rows, eval_logits(model, fit_images))
    initial_stop = balanced_metrics(stop_rows, eval_logits(model, stop_images))
    write_json(run_dir / "INITIAL_TRAINING_DIAGNOSTICS.json", {
        "schema": "accuracy-search-initial-diagnostics-v1", "run_id": run_id,
        "initial_model_weight_sha256": model_weight_hash(model),
        "fit": initial_fit, "stopping": initial_stop,
        "class_balance_per_minibatch": {label: int(config["examples_per_class_per_batch"]) for label in CLASSES},
        "fit_rows": len(fit_rows), "fit_source_groups": len({r["source_group"] for r in fit_rows}),
        "fit_other_identity_count": len({r["identity_id"] for r in fit_rows if r["label"] == "OTHER"}),
        "stopping_rows": len(stop_rows), "stopping_other_identity_count": len(stop_other_ids),
        "role_checks": {"fit_stop_source_groups_disjoint": True,
                        "calibration_not_used_for_stopping": True,
                        "development_not_used_for_stopping": True},
        "input_contract": "uint8 gray96 NHWC; [0,255] at load, float32 [0,1] in model",
    })

    train_state_path = run_dir / "RUN_STATE.json"
    curves_path = run_dir / "CURVES.jsonl"
    latest_prefix = str(run_dir / "checkpoint" / "state")
    run_dir.joinpath("checkpoint").mkdir(exist_ok=True)
    training_started = time.monotonic()

    # Common learning schedule. Each phase has an independent Adam optimizer;
    # checkpoints preserve the active optimizer slots and the epoch counter.
    phases = [
        {"name": "head_warmup", "epochs": int(config["epochs_head"]), "lr": float(config["lr_head"]), "fraction": 0.0},
        {"name": "partial_finetune", "epochs": int(config["epochs_finetune"]), "lr": float(config["lr_finetune"]), "fraction": float(config.get("unfreeze_top_fraction", 0.25))},
    ]
    state = None
    if resume and train_state_path.exists():
        state = json.loads(train_state_path.read_text())
        if state.get("config_sha256") != config_sha or state.get("status") == "complete":
            raise ValueError("resume config mismatch or run is already complete")
    best_global_loss = float(state["best_global_loss"]) if state else float("inf")
    best_global_macro_acc = float(state["best_global_macro_accuracy"]) if state else -1.0
    best_global_phase_index = int(state.get("best_global_phase_index", state.get("phase_index", 0))) if state else -1
    best_weights_path = run_dir / "best_model.weights.h5"
    global_epoch = int(state.get("global_epoch", 0)) if state else 0
    current_phase_index = int(state.get("phase_index", 0)) if state else 0
    resumed_once = False

    with resource_lease(STUDY, "heavy_training", owner="v7_accuracy_runner:" + run_id, timeout_s=0):
        for phase_index in range(current_phase_index, len(phases)):
            phase = phases[phase_index]
            if phase["fraction"] <= 0:
                model._d_base.trainable = False
                trainability = {"backbone_layers_total": len(model._d_base.layers), "backbone_layers_trainable": [],
                                "batch_normalization_frozen": True}
            else:
                trainability = d.unfreeze_top_fraction(model, phase["fraction"])
            d.compile_classifier(model, phase["lr"], 6)
            optimizer = model.optimizer
            if hasattr(optimizer, "_create_all_weights"):
                optimizer._create_all_weights(model.trainable_variables)
            epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32, name="completed_epoch")
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch_var)
            manager = tf.train.CheckpointManager(checkpoint, str(run_dir / "checkpoint" / phase["name"]), max_to_keep=5)
            phase_epoch_start = 0
            phase_best_loss = float("inf")
            phase_best_acc = -1.0
            phase_best_epoch = 0
            stale_epochs = 0
            if resume and not resumed_once and state is not None and phase_index == int(state["phase_index"]):
                if state.get("phase_name") != phase["name"]:
                    raise ValueError("resume phase changed")
                checkpoint.restore(state["checkpoint_path"]).expect_partial()
                phase_epoch_start = int(state["phase_epoch"])
                phase_best_loss = (float(state["phase_best_loss"])
                                   if state.get("phase_best_loss") is not None else float("inf"))
                phase_best_acc = float(state["phase_best_macro_accuracy"])
                phase_best_epoch = int(state["phase_best_epoch"])
                stale_epochs = int(state["stale_epochs"])
                if stale_epochs >= int(config.get("patience", 10)):
                    # A process may resume after the stopping epoch but before
                    # final evaluation; do not add an unplanned gradient update.
                    phase_epoch_start = phase["epochs"]
                resumed_once = True

            for epoch in range(phase_epoch_start, phase["epochs"]):
                epoch_seed = int(config["seed"]) + (phase_index + 1) * 100000 + (epoch + 1) * 1009
                epoch_started = time.monotonic()
                # Fixed balanced six-class batches; OTHER uses distinct identities
                # within each batch and never draws from calibration/development.
                batch_seed = int(epoch_seed ^ 0x13579BDF)
                batch_count = 0
                batch_losses = []
                rng = np.random.default_rng(batch_seed)
                epoch_batches = balanced_batch_indices(
                    fit_rows, batch_seed, int(config["batch_size"]), int(config["examples_per_class_per_batch"]))
                for batch_ix, indices in enumerate(epoch_batches):
                    bx = fit_images[indices].astype(np.float32) / np.float32(255)
                    # Mild, reproducible grayscale photometric + horizontal flip
                    # augmentation. Strong blur/perspective is a later, separately
                    # versioned factor after this clean baseline has learned.
                    aug_rng = np.random.default_rng(epoch_seed + batch_ix * 997)
                    for j in range(len(bx)):
                        if aug_rng.integers(0, 2):
                            bx[j] = bx[j, :, ::-1, :]
                        gain = float(aug_rng.uniform(0.92, 1.08))
                        offset = float(aug_rng.uniform(-8.0, 8.0)) / 255.0
                        bx[j] = np.clip(bx[j] * gain + offset, 0.0, 1.0)
                    by = np.asarray([CLASS_ID[fit_rows[i]["label"]] for i in indices], dtype=np.int64)
                    result = model.train_on_batch(bx, by, return_dict=True)
                    loss = float(result["loss"])
                    if not math.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    batch_losses.append(loss)
                    batch_count += 1

                fit_logits = eval_logits(model, fit_images)
                stop_logits = eval_logits(model, stop_images)
                fit_metrics = balanced_metrics(fit_rows, fit_logits)
                stop_metrics = balanced_metrics(stop_rows, stop_logits)
                global_epoch += 1
                completed = epoch + 1
                epoch_var.assign(completed)
                improved = (stop_metrics["macro_cross_entropy"] < phase_best_loss - 1e-6 or
                            (abs(stop_metrics["macro_cross_entropy"] - phase_best_loss) <= 1e-6 and
                             stop_metrics["macro_accuracy"] > phase_best_acc))
                if improved:
                    phase_best_loss = float(stop_metrics["macro_cross_entropy"])
                    phase_best_acc = float(stop_metrics["macro_accuracy"])
                    phase_best_epoch = completed
                    stale_epochs = 0
                    model.save_weights(str(run_dir / ("best_" + phase["name"] + ".weights.h5")))
                    if phase_best_loss < best_global_loss or (phase_best_loss == best_global_loss and phase_best_acc > best_global_macro_acc):
                        best_global_loss = phase_best_loss
                        best_global_macro_acc = phase_best_acc
                        best_global_phase_index = phase_index
                        best_weights_path = run_dir / "best_model.weights.h5"
                        model.save_weights(str(best_weights_path))
                else:
                    stale_epochs += 1
                ckpt_path = manager.save(checkpoint_number=completed)
                curve = {
                    "phase": phase["name"], "phase_epoch": completed, "global_epoch": global_epoch,
                    "seed": int(config["seed"]), "epoch_seed": epoch_seed, "batch_seed": batch_seed,
                    "learning_rate": phase["lr"], "trainable_backbone_layers": len(trainability["backbone_layers_trainable"]),
                    "batch_norm_frozen": True, "balanced_class_batch": True,
                    "batches": batch_count, "sample_presentations": batch_count * int(config["batch_size"]),
                    "gradient_updates": batch_count,
                    "mean_minibatch_loss": float(np.mean(batch_losses)), "fit": fit_metrics,
                    "stopping": stop_metrics, "elapsed_seconds": time.monotonic() - epoch_started,
                    "checkpoint_path": ckpt_path,
                }
                with curves_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(curve, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                state = {
                    "schema": "accuracy-search-run-state-v1", "run_id": run_id,
                    "config_sha256": config_sha, "phase_index": phase_index, "phase_name": phase["name"],
                    "phase_epoch": completed, "global_epoch": global_epoch,
                    "phase_best_loss": phase_best_loss, "phase_best_macro_accuracy": phase_best_acc,
                    "phase_best_epoch": phase_best_epoch, "stale_epochs": stale_epochs,
                    "best_global_loss": best_global_loss, "best_global_macro_accuracy": best_global_macro_acc,
                    "best_global_phase_index": best_global_phase_index,
                    "checkpoint_path": ckpt_path, "optimizer_slots_preserved": True,
                    "rng_replay": "per-root/per-phase/per-epoch seeds; dropout disabled; augmentation stateless by deterministic epoch seed",
                    "status": "running", "updated_utc": utc_now(),
                }
                write_json(train_state_path, state)
                print(json.dumps({"event": "epoch", "run_id": run_id, "phase": phase["name"],
                                  "epoch": completed, "fit_ce": fit_metrics["macro_cross_entropy"],
                                  "fit_macro_acc": fit_metrics["macro_accuracy"],
                                  "stop_ce": stop_metrics["macro_cross_entropy"],
                                  "stop_macro_acc": stop_metrics["macro_accuracy"],
                                  "elapsed_s": curve["elapsed_seconds"]}), flush=True)
                if stale_epochs >= int(config.get("patience", 10)):
                    break
            if phase_index == 0:
                # Begin the fine-tuning phase from the best warmup checkpoint.
                model.load_weights(str(run_dir / "best_head_warmup.weights.h5"))
                next_phase = phases[1]
                d.unfreeze_top_fraction(model, next_phase["fraction"])
                d.compile_classifier(model, next_phase["lr"], 6)
                next_optimizer = model.optimizer
                if hasattr(next_optimizer, "_create_all_weights"):
                    next_optimizer._create_all_weights(model.trainable_variables)
                next_epoch = tf.Variable(0, trainable=False, dtype=tf.int32, name="completed_epoch")
                next_checkpoint = tf.train.Checkpoint(model=model, optimizer=next_optimizer, epoch=next_epoch)
                next_manager = tf.train.CheckpointManager(next_checkpoint, str(run_dir / "checkpoint" / next_phase["name"]), max_to_keep=5)
                next_path = next_manager.save(checkpoint_number=0)
                state = {
                    "schema": "accuracy-search-run-state-v1", "run_id": run_id,
                    "config_sha256": config_sha, "phase_index": 1, "phase_name": next_phase["name"],
                    "phase_epoch": 0, "global_epoch": global_epoch, "phase_best_loss": None,
                    "phase_best_macro_accuracy": -1.0, "phase_best_epoch": 0, "stale_epochs": 0,
                    "best_global_loss": best_global_loss, "best_global_macro_accuracy": best_global_macro_acc,
                    "best_global_phase_index": best_global_phase_index,
                    "checkpoint_path": next_path, "optimizer_slots_preserved": True,
                    "rng_replay": "per-root/per-phase/per-epoch seeds; dropout disabled; augmentation stateless by deterministic epoch seed",
                    "status": "running", "updated_utc": utc_now(),
                }
                write_json(train_state_path, state)

    training_elapsed = time.monotonic() - training_started
    curve_records = [json.loads(line) for line in curves_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gradient_training_elapsed = float(sum(float(row["elapsed_seconds"]) for row in curve_records))
    gradient_updates_completed = int(sum(int(row["gradient_updates"]) for row in curve_records))
    if state:
        state.update({"status": "evaluation_pending", "updated_utc": utc_now(),
                      "gradient_epochs_completed": len(curve_records),
                      "gradient_updates_completed": gradient_updates_completed,
                      "gradient_epoch_wall_seconds_sum": gradient_training_elapsed})
        write_json(train_state_path, state)

    # The global stopping rule chooses only from known-stopping plus internal
    # OTHER identity holdout; calibration and development roles are untouched.
    if best_weights_path.exists():
        if best_global_phase_index <= 0:
            model._d_base.trainable = False
        else:
            d.unfreeze_top_fraction(model, float(config.get("unfreeze_top_fraction", 0.25)))
        model.load_weights(str(best_weights_path))
    final_fit = balanced_metrics(fit_rows, eval_logits(model, fit_images))
    final_stop = balanced_metrics(stop_rows, eval_logits(model, stop_images))
    # Independent LFW source roles: calibration is only for threshold selection;
    # development remains descriptive and cannot tune the model or rule.
    lfw_cal = [r for r in source_rows if r["role"] in ("threshold_calibration", "unknown_calibration")]
    lfw_cal_images = np.stack([source_images[str(r["record_id"])] for r in lfw_cal]).astype(np.uint8)
    lfw_cal_rows = [{**training_row({**r, "role": "train"}, source_images[str(r["record_id"])]),
                     "role": r["role"]} for r in lfw_cal]
    lfw_cal_logits = eval_logits(model, lfw_cal_images)
    lfw_threshold = threshold_from_calibration(lfw_cal_rows, lfw_cal_logits)
    lfw_dev = [r for r in source_rows if r["role"] in ("development", "unknown_development")]
    lfw_dev_images = np.stack([source_images[str(r["record_id"])] for r in lfw_dev]).astype(np.uint8)
    lfw_dev_rows = [{**training_row({**r, "role": "train"}, source_images[str(r["record_id"])]),
                     "role": r["role"]} for r in lfw_dev]
    lfw_dev_logits = eval_logits(model, lfw_dev_images)
    lfw_dev_metrics = evaluate_open_set(lfw_dev_rows, lfw_dev_logits, lfw_threshold)

    camera_rows, camera_images, camera_receipt = load_camera_images(int(config["roi_side_camera_px"]))
    camera_private_records = camera_receipt.pop("private_input_records")
    camera_private_path = private / "CAMERA_INPUTS.jsonl"
    with camera_private_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in camera_private_records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    cam_cal_rows, cam_cal_images, cam_cal_valid = [], [], []
    cam_dev_rows, cam_dev_images, cam_dev_valid = [], [], []
    for row in camera_rows:
        capture_id = str(row["request"]["capture_id"])
        role, label = str(row["role"]), str(row["label"])
        wrapped = {
            "record_id": capture_id, "label": label, "label_id": label,
            "class_index": CLASS_ID[label], "identity_partition": "unknown" if label == "OTHER" else "known",
            "identity_id": str(row["source"].get("source_identity") or row["source"].get("source_group") or capture_id),
            "source_identity": str(row["source"].get("source_identity") or row["source"].get("source_group") or capture_id),
            "source_group": str(row["source"].get("source_group") or capture_id), "role": role,
            "capture_set": row["capture_set"], "phase": row["phase"],
        }
        if role in ("threshold_calibration", "unknown_calibration") and label in CLASSES:
            cam_cal_rows.append(wrapped)
            valid = capture_id in camera_images
            cam_cal_valid.append(valid)
            cam_cal_images.append(camera_images[capture_id] if valid else np.zeros((96, 96, 1), np.uint8))
        if role in ("development", "unknown_development") and label in CLASSES:
            cam_dev_rows.append(wrapped)
            valid = capture_id in camera_images
            cam_dev_valid.append(valid)
            cam_dev_images.append(camera_images[capture_id] if valid else np.zeros((96, 96, 1), np.uint8))
    cam_cal_array = (np.stack(cam_cal_images).astype(np.uint8) if cam_cal_images else
                     np.zeros((0, 96, 96, 1), dtype=np.uint8))
    cam_cal_logits = eval_logits(model, cam_cal_array)
    cam_cal_valid = np.asarray(cam_cal_valid, dtype=bool)
    valid_cal_rows = [r for r, is_valid in zip(cam_cal_rows, cam_cal_valid) if is_valid]
    valid_cal_logits = cam_cal_logits[cam_cal_valid]
    if valid_cal_rows and any(r["label"] == "OTHER" for r in valid_cal_rows) and any(r["label"] in KNOWN for r in valid_cal_rows):
        cam_threshold = threshold_from_calibration(valid_cal_rows, valid_cal_logits)
        cam_dev_array = (np.stack(cam_dev_images).astype(np.uint8) if cam_dev_images else
                         np.zeros((0, 96, 96, 1), dtype=np.uint8))
        cam_dev_logits = eval_logits(model, cam_dev_array)
        cam_dev_metrics = evaluate_open_set(cam_dev_rows, cam_dev_logits, cam_threshold,
                                            np.asarray(cam_dev_valid, dtype=bool))
    else:
        cam_threshold = {"status": "not_selectable_missing_valid_known_or_unknown_calibration"}
        cam_dev_metrics = {"status": "not_evaluated_missing_valid_known_or_unknown_calibration"}

    # Host float model export and PTQ are included only as a diagnostic. The
    # device interpreter/output parity and memory must be requalified separately.
    model_export = run_dir / "model_float.h5"
    model.save(str(model_export), include_optimizer=False)
    model_digest = sha256_file(model_export)

    predictions_path = private / "PREDICTIONS.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        def emit_predictions(mode, rows, logits, threshold, valid=None):
            probs = softmax32(logits)
            score = np.max(probs[:, :5], axis=1).astype(np.float32)
            pred = np.argmax(probs, axis=1)
            accept_threshold = np.float32(threshold["threshold_float32"])
            mask = np.ones(len(rows), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
            for i, row in enumerate(rows):
                record = {
                    "mode": mode, "record_id": row.get("record_id"), "capture_id": row.get("record_id"),
                    "role": row.get("role"), "true_label": row.get("label"),
                    "source_identity": row.get("source_identity"), "source_group": row.get("source_group"),
                    "valid_input": bool(mask[i]), "predicted_class": CLASSES[int(pred[i])],
                    "score_max_known_float32": float(score[i]), "six_way_logits_float32": [float(v) for v in logits[i]],
                    "threshold_float32": float(accept_threshold),
                    "accepted": bool(mask[i] and pred[i] < 5 and score[i] >= accept_threshold),
                    "model_weight_sha256": model_weight_hash(model),
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        emit_predictions("HOST_REFERENCE_LFW_CALIBRATION", lfw_cal_rows, lfw_cal_logits, lfw_threshold)
        emit_predictions("HOST_REFERENCE_LFW_DEVELOPMENT", lfw_dev_rows, lfw_dev_logits, lfw_threshold)
        if valid_cal_rows:
            emit_predictions("HOST_REFERENCE_OLD_OPTICAL_CALIBRATION", valid_cal_rows, valid_cal_logits, cam_threshold)
        if cam_dev_rows and isinstance(cam_threshold, dict) and cam_threshold.get("threshold_float32") is not None:
            emit_predictions("HOST_REFERENCE_OLD_OPTICAL_DEVELOPMENT", cam_dev_rows, cam_dev_logits,
                             cam_threshold, np.asarray(cam_dev_valid, dtype=bool))
    result = {
        "schema": "accuracy-search-result-v1", "run_id": run_id, "created_utc": utc_now(),
        "status": "complete_scout_host_only", "config_sha256": config_sha,
        "source_manifest_path": str(ROLE_PATH.relative_to(ROOT)),
        "source_selected_role_rows_sha256": source_receipt["selected_rows_sha256"],
        "source_rows_receipt": source_receipt, "source_inputs_receipt": source_inputs,
        "camera_inputs_receipt": camera_receipt,
        "split": split_receipt, "model": model_summary,
        "training": {"best_stopping_macro_cross_entropy": best_global_loss,
                     "best_stopping_macro_accuracy": best_global_macro_acc,
                     "best_stopping_phase": phases[best_global_phase_index]["name"],
                     "completed_gradient_epochs": len(curve_records),
                     "completed_gradient_updates": gradient_updates_completed,
                     "gradient_epoch_wall_seconds_sum": gradient_training_elapsed,
                     "fit_metrics_at_selected_checkpoint": final_fit,
                     "stopping_metrics_at_selected_checkpoint": final_stop,
                     "curves_path": str(curves_path.relative_to(ROOT)),
                     "checkpoint_state_path": str(train_state_path.relative_to(ROOT)),
                     "optimizer_rng_resume": "epoch-boundary TensorFlow model+Adam checkpoint; deterministic stateless batch/augmentation seeds; dropout disabled"},
        "source_lfw": {"threshold": lfw_threshold, "development": lfw_dev_metrics,
                       "development_scope": "source image resized to gray96; no detector; host replay"},
        "old_optical_camera": {"threshold": cam_threshold, "development": cam_dev_metrics,
                               "scope": "historical 2026-11-01 camera setup only; not the user's intended moved/dimmer setup; host replay; FOMO misses included"},
        "exports": {"float_model_path": str(model_export.relative_to(ROOT)), "float_model_sha256": model_digest,
                    "float_model_bytes": model_export.stat().st_size,
                    "int8_export": "not_yet_converted_in_scout_v1"},
        "private_evidence": {"camera_input_rows": str(camera_private_path.relative_to(ROOT)),
                             "camera_input_rows_sha256": sha256_file(camera_private_path),
                             "per_image_predictions": str(predictions_path.relative_to(ROOT)),
                             "per_image_predictions_sha256": sha256_file(predictions_path)},
        "resources": {"host_platform": platform.platform(), "python": platform.python_version(),
                      "tensorflow": tf.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                      "thread_limits": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
                      "training_elapsed_seconds": gradient_training_elapsed,
                      "current_process_training_loop_elapsed_seconds": training_elapsed,
                      "total_run_elapsed_seconds": time.monotonic() - run_started,
                      "max_rss": {"value": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                                  "unit": "bytes on macOS, KiB on Linux"},
                      "device_memory_latency": "not_measured_in_this_host_scout"},
        "limits": ["one exploratory training seed; not a ten-seed comparison", "no current moved-camera/dimmer-room recapture in this run", "host float model only; not deployed or device-qualified", "old optical rows have a single taped-screen setup", "threshold calibration size does not certify the one-sided 1% unknown-risk target"],
    }
    write_json(result_path, result)
    if state:
        state.update({"status": "complete", "updated_utc": utc_now(), "result_sha256": sha256_file(result_path)})
        write_json(train_state_path, state)
    return result


def train_camera_adaptation_run(config_path: Path, resume: bool = False,
                                launch_authorized: bool = False) -> dict:
    """Continue the selected v7 model using grouped camera train rows only.

    Calibration and development frames are not requested from the capture
    loader until the best epoch/checkpoint has been selected and the local run
    state has been durably marked ``evaluation_pending``.
    """
    run_started = time.monotonic()
    config, config_sha = model_config(config_path)
    run_id = str(config["run_id"])
    if config.get("training_mode") != "grouped_camera_adaptation_from_parent":
        raise ValueError("camera adaptation runner requires grouped_camera_adaptation_from_parent")
    if config.get("training_launch_enabled") is not True:
        raise RuntimeError("camera adaptation launch is disabled by this config")
    if launch_authorized is not True:
        raise RuntimeError("camera adaptation requires the explicit --launch-camera-adaptation gate")
    runner_sha = sha256_file(Path(__file__).resolve())
    if config.get("runner_sha256") != runner_sha:
        raise ValueError("launch config runner hash does not match this accuracy runner")

    run_dir = RUN_ROOT / run_id
    private = run_dir / "private"
    run_dir.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "RESULT.json"
    if result_path.exists():
        raise FileExistsError("camera adaptation result exists; use a new immutable run_id")
    preflight_path = RUN_ROOT / "preflight" / (run_id + ".json")
    if not preflight_path.is_file():
        raise FileNotFoundError("launch-bound camera adaptation preflight is required before training")
    preflight_sha = sha256_file(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("run_id") != run_id or preflight.get("config_sha256") != config_sha or
            preflight.get("runner_sha256") != runner_sha or preflight.get("training_launch_enabled") is not True or
            int(preflight.get("gradient_updates_completed", -1)) != 0):
        raise ValueError("launch-bound camera preflight does not match the current immutable run")

    parent_result_path = (ROOT / str(config["parent_result_path"])).resolve()
    parent_weights_path = (ROOT / str(config["parent_model_weights_path"])).resolve()
    parent_result_path.relative_to(ROOT)
    parent_weights_path.relative_to(ROOT)
    if sha256_file(parent_result_path) != config["parent_result_sha256"]:
        raise ValueError("parent result changed after camera adaptation preflight")
    if sha256_file(parent_weights_path) != config["parent_model_weights_sha256"]:
        raise ValueError("parent selected weights changed after camera adaptation preflight")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    if parent_result.get("run_id") != config["parent_run_id"] or parent_result.get("status") != "complete_scout_host_only":
        raise ValueError("camera adaptation parent result is not the pinned completed source scout")

    camera_rows, camera_image_map, camera_train_receipt = load_camera_train_pool(int(config["roi_side_camera_px"]))
    private_train_records = camera_train_receipt.pop("private_input_records")
    fit_rows, fit_images, stop_rows, stop_images, split = split_camera_training_groups(
        camera_rows, camera_image_map, int(config["seed"]), float(config["camera_stop_fraction"]))
    if (fit_rows and any(row.get("source_kind") != "camera_replay_train" or
                         row.get("role") not in ("train", "other_train") for row in fit_rows)):
        raise AssertionError("camera adaptation fit split contains a non-training source")
    if (stop_rows and any(row.get("source_kind") != "camera_replay_internal_stopping" for row in stop_rows)):
        raise AssertionError("camera adaptation stopping split contains a non-camera source")
    if len(fit_rows) != int(config["expected_camera_fit_rows"]) or len(stop_rows) != int(config["expected_camera_stopping_rows"]):
        raise ValueError("camera adaptation grouped row counts differ from the immutable launch contract")
    batches_per_epoch = int(math.ceil(max(split["fit_rows_by_label"][label] for label in KNOWN) /
                                      float(config["examples_per_class_per_batch"])))
    expected_max_updates = batches_per_epoch * (int(config["epochs_head"]) + int(config["epochs_finetune"]))
    if expected_max_updates != int(config["expected_max_gradient_updates"]):
        raise ValueError("camera adaptation update budget differs from the immutable launch contract")

    train_index_path = private / "CAMERA_TRAIN_POOL.jsonl"
    if not train_index_path.is_file() or sha256_file(train_index_path) != preflight["private_input_index"]["sha256"]:
        raise ValueError("camera train-only preflight input index is missing or changed")
    if canonical_hash(private_train_records) != camera_train_receipt.get("training_inputs_sha256"):
        raise AssertionError("camera train pool input receipt changed while constructing fit/stop split")
    if camera_train_receipt.get("calibration_or_development_frame_paths_opened") is not False:
        raise AssertionError("camera training loader opened calibration/development frames")

    state_path = run_dir / "RUN_STATE.json"
    curves_path = run_dir / "CURVES.jsonl"
    state = None
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("cannot resume camera adaptation without a run-local checkpoint state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("config_sha256") != config_sha or state.get("preflight_sha256") != preflight_sha or
                state.get("runner_sha256") != runner_sha or state.get("status") not in
                ("ready", "running", "evaluation_pending")):
            raise ValueError("camera adaptation run state does not match this config/preflight or is not resumable")
    elif state_path.exists() or curves_path.exists():
        raise FileExistsError("camera adaptation artifacts exist; resume the same run or choose a new run_id")
    if resume and curves_path.exists():
        # A crash may occur after appending a curve row but before committing
        # the matching state. Keep only epochs named by the durable state so
        # resume neither double-counts nor skips a gradient update.
        curve_lines = [line for line in curves_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        committed_epoch = int(state.get("global_epoch", 0))
        committed_lines = []
        for line in curve_lines:
            curve = json.loads(line)
            if int(curve["global_epoch"]) <= committed_epoch:
                committed_lines.append(line)
        if len(committed_lines) != len(curve_lines):
            with curves_path.open("w", encoding="utf-8", newline="\n") as stream:
                for line in committed_lines:
                    stream.write(line + "\n")
        committed_epochs = [int(json.loads(line)["global_epoch"]) for line in committed_lines]
        if committed_epochs != list(range(1, committed_epoch + 1)):
            raise ValueError("camera adaptation curve does not match its committed run state")

    cv2.setNumThreads(2)
    tf.keras.utils.set_random_seed(int(config["seed"]))
    model, embedding_model, init_info = d.build_classifier(
        ROOT, float(config["alpha"]), True, 6, int(config["seed"]), "v8_camera_adaptation_" + run_id)
    model.get_layer("head_dropout").rate = 0.0
    # The parent result selected weights from partial fine-tuning. Preserve
    # that HDF5 trainable-weight ordering while loading, then choose phase order.
    parent_trainability = d.unfreeze_top_fraction(model, float(config["unfreeze_top_fraction"]))
    model.load_weights(str(parent_weights_path))
    loaded_parent_weight_hash = model_weight_hash(model)
    if loaded_parent_weight_hash != preflight["parent_source_scout"]["loaded_selected_weight_model_sha256"]:
        raise ValueError("loaded parent model weights differ from the preflight-verified initialization")
    parent_info = {key: value for key, value in init_info.items() if key not in ("full_reference_model", "feature_model")}
    del init_info, embedding_model

    if not resume:
        initial_fit = balanced_metrics(fit_rows, eval_logits(model, fit_images))
        initial_stop = balanced_metrics(stop_rows, eval_logits(model, stop_images))
        write_json(run_dir / "INITIAL_CAMERA_TRAINING_DIAGNOSTICS.json", {
            "schema": "accuracy-search-camera-adaptation-initial-diagnostics-v1", "run_id": run_id,
            "parent_selected_model_weight_sha256": loaded_parent_weight_hash,
            "fit_metrics": initial_fit, "stopping_metrics": initial_stop,
            "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
            "fit_source_kind_counts": dict(Counter(row["source_kind"] for row in fit_rows)),
            "stopping_source_kind_counts": dict(Counter(row["source_kind"] for row in stop_rows)),
            "fit_stopping_group_overlap": split["fit_stopping_group_overlap"],
            "calibration_development_loaded": False, "future_final_files_opened": False,
        })

    phases = [
        {"name": "camera_head_warmup", "epochs": int(config["epochs_head"]),
         "lr": float(config["lr_head"]), "fraction": 0.0},
        {"name": "camera_partial_finetune", "epochs": int(config["epochs_finetune"]),
         "lr": float(config["lr_finetune"]), "fraction": float(config["unfreeze_top_fraction"])},
    ]
    training_started = time.monotonic()
    training_invocation_elapsed = 0.0
    if state is None or state.get("status") != "evaluation_pending":
        current_phase_index = int(state.get("phase_index", 0)) if state else 0
        global_epoch = int(state.get("global_epoch", 0)) if state else 0
        best_global_loss = float(state["best_global_loss"]) if state and state.get("best_global_loss") is not None else float("inf")
        best_global_macro_acc = float(state["best_global_macro_accuracy"]) if state and state.get("best_global_macro_accuracy") is not None else -1.0
        best_global_phase_index = int(state.get("best_global_phase_index", -1)) if state else -1
        best_global_weights_path = state.get("best_global_weights_path") if state else None
        resumed_once = False
        with resource_lease(STUDY, "heavy_training", owner="v8_camera_adaptation:" + run_id, timeout_s=0):
            for phase_index in range(current_phase_index, len(phases)):
                phase = phases[phase_index]
                if phase["fraction"] <= 0:
                    model._d_base.trainable = False
                    trainability = {"backbone_layers_trainable": [], "batch_normalization_frozen": True}
                else:
                    trainability = d.unfreeze_top_fraction(model, phase["fraction"])
                d.compile_classifier(model, phase["lr"], 6)
                optimizer = model.optimizer
                if hasattr(optimizer, "_create_all_weights"):
                    optimizer._create_all_weights(model.trainable_variables)
                epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32, name="camera_completed_epoch")
                checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch_var)
                manager = tf.train.CheckpointManager(
                    checkpoint, str(run_dir / "checkpoint" / phase["name"]), max_to_keep=5)
                phase_epoch_start = 0
                phase_best_loss = float("inf")
                phase_best_acc = -1.0
                phase_best_epoch = 0
                phase_best_weights_path = None
                stale_epochs = 0
                if resume and not resumed_once and state is not None and phase_index == int(state["phase_index"]):
                    if state.get("phase_name") != phase["name"]:
                        raise ValueError("camera adaptation resume phase changed")
                    checkpoint.restore(state["checkpoint_path"]).expect_partial()
                    phase_epoch_start = int(state["phase_epoch"])
                    phase_best_loss = (float(state["phase_best_loss"])
                                       if state.get("phase_best_loss") is not None else float("inf"))
                    phase_best_acc = float(state.get("phase_best_macro_accuracy", -1.0))
                    phase_best_epoch = int(state.get("phase_best_epoch", 0))
                    phase_best_weights_path = state.get("phase_best_weights_path")
                    stale_epochs = int(state.get("stale_epochs", 0))
                    if stale_epochs >= int(config["patience"]):
                        phase_epoch_start = phase["epochs"]
                    resumed_once = True
                elif not resume and phase_index == 0:
                    checkpoint_zero = manager.save(checkpoint_number=0)
                    state = {
                        "schema": "accuracy-search-camera-adaptation-run-state-v1", "run_id": run_id,
                        "config_sha256": config_sha, "runner_sha256": runner_sha,
                        "preflight_sha256": preflight_sha,
                        "parent_result_sha256": config["parent_result_sha256"],
                        "parent_model_weights_sha256": config["parent_model_weights_sha256"],
                        "camera_train_input_sha256": camera_train_receipt["training_inputs_sha256"],
                        "grouped_split_sha256": split["group_keys_sha256"],
                        "phase_index": 0, "phase_name": phase["name"], "phase_epoch": 0,
                        "global_epoch": 0, "phase_best_loss": None, "phase_best_macro_accuracy": -1.0,
                        "phase_best_epoch": 0, "stale_epochs": 0,
                        "best_global_loss": None, "best_global_macro_accuracy": None,
                        "best_global_phase_index": -1, "best_global_weights_path": None,
                        "phase_best_weights_path": None, "checkpoint_path": checkpoint_zero,
                        "optimizer_slots_preserved": True, "status": "ready", "updated_utc": utc_now(),
                    }
                    write_json(state_path, state)

                for epoch in range(phase_epoch_start, phase["epochs"]):
                    epoch_seed = int(config["seed"]) + (phase_index + 1) * 100000 + (epoch + 1) * 1009
                    epoch_started = time.monotonic()
                    batch_seed = int(epoch_seed ^ 0x13579BDF)
                    epoch_batches = balanced_batch_indices(
                        fit_rows, batch_seed, int(config["batch_size"]),
                        int(config["examples_per_class_per_batch"]))
                    batch_contract = validate_camera_adaptation_batches(
                        fit_rows, epoch_batches, int(config["batch_size"]),
                        int(config["examples_per_class_per_batch"]))
                    batch_losses = []
                    for batch_ix, indices in enumerate(epoch_batches):
                        bx = fit_images[indices].astype(np.float32) / np.float32(255)
                        aug_rng = np.random.default_rng(epoch_seed + batch_ix * 997)
                        for j in range(len(bx)):
                            if aug_rng.integers(0, 2):
                                bx[j] = bx[j, :, ::-1, :]
                            gain = float(aug_rng.uniform(0.92, 1.08))
                            offset = float(aug_rng.uniform(-8.0, 8.0)) / 255.0
                            bx[j] = np.clip(bx[j] * gain + offset, 0.0, 1.0)
                        by = np.asarray([CLASS_ID[fit_rows[i]["label"]] for i in indices], dtype=np.int64)
                        output = model.train_on_batch(bx, by, return_dict=True)
                        loss = float(output["loss"])
                        if not math.isfinite(loss):
                            raise ValueError("nonfinite camera adaptation loss")
                        batch_losses.append(loss)
                    fit_metrics = balanced_metrics(fit_rows, eval_logits(model, fit_images))
                    stop_metrics = balanced_metrics(stop_rows, eval_logits(model, stop_images))
                    global_epoch += 1
                    completed = epoch + 1
                    epoch_var.assign(completed)
                    improved = (stop_metrics["macro_cross_entropy"] < phase_best_loss - 1e-6 or
                                (abs(stop_metrics["macro_cross_entropy"] - phase_best_loss) <= 1e-6 and
                                 stop_metrics["macro_accuracy"] > phase_best_acc))
                    # Commit the model/optimizer checkpoint before publishing
                    # any best-weight pointer or curve/state row for this epoch.
                    checkpoint_path = manager.save(checkpoint_number=completed)
                    if improved:
                        phase_best_loss = float(stop_metrics["macro_cross_entropy"])
                        phase_best_acc = float(stop_metrics["macro_accuracy"])
                        phase_best_epoch = completed
                        stale_epochs = 0
                        phase_best_weights_path = run_dir / (
                            f"best_phase_{phase_index}_epoch_{completed}.weights.h5")
                        model.save_weights(str(phase_best_weights_path))
                        if (phase_best_loss < best_global_loss or
                                (phase_best_loss == best_global_loss and phase_best_acc > best_global_macro_acc)):
                            best_global_loss = phase_best_loss
                            best_global_macro_acc = phase_best_acc
                            best_global_phase_index = phase_index
                            best_global_weights_path = run_dir / (
                                f"best_global_phase_{phase_index}_epoch_{completed}.weights.h5")
                            model.save_weights(str(best_global_weights_path))
                    else:
                        stale_epochs += 1
                    curve = {
                        "phase": phase["name"], "phase_epoch": completed, "global_epoch": global_epoch,
                        "seed": int(config["seed"]), "epoch_seed": epoch_seed, "batch_seed": batch_seed,
                        "learning_rate": phase["lr"],
                        "trainable_backbone_layers": len(trainability["backbone_layers_trainable"]),
                        "batch_norm_frozen": True, "balanced_class_batch": True,
                        "camera_only_gradient_source": True, "camera_fit_rows": len(fit_rows),
                        "camera_stop_rows": len(stop_rows), "batch_contract": batch_contract,
                        "batches": len(epoch_batches), "sample_presentations": len(epoch_batches) * int(config["batch_size"]),
                        "gradient_updates": len(epoch_batches), "mean_minibatch_loss": float(np.mean(batch_losses)),
                        "fit": fit_metrics, "stopping": stop_metrics,
                        "elapsed_seconds": time.monotonic() - epoch_started,
                        "checkpoint_path": checkpoint_path,
                    }
                    with curves_path.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(curve, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                    state = {
                        "schema": "accuracy-search-camera-adaptation-run-state-v1", "run_id": run_id,
                        "config_sha256": config_sha, "runner_sha256": runner_sha,
                        "preflight_sha256": preflight_sha,
                        "parent_result_sha256": config["parent_result_sha256"],
                        "parent_model_weights_sha256": config["parent_model_weights_sha256"],
                        "camera_train_input_sha256": camera_train_receipt["training_inputs_sha256"],
                        "grouped_split_sha256": split["group_keys_sha256"],
                        "phase_index": phase_index, "phase_name": phase["name"],
                        "phase_epoch": completed, "global_epoch": global_epoch,
                        "phase_best_loss": phase_best_loss, "phase_best_macro_accuracy": phase_best_acc,
                        "phase_best_epoch": phase_best_epoch,
                        "phase_best_weights_path": str(phase_best_weights_path) if phase_best_weights_path else None,
                        "stale_epochs": stale_epochs,
                        "best_global_loss": best_global_loss, "best_global_macro_accuracy": best_global_macro_acc,
                        "best_global_phase_index": best_global_phase_index,
                        "best_global_weights_path": str(best_global_weights_path) if best_global_weights_path else None,
                        "checkpoint_path": checkpoint_path, "optimizer_slots_preserved": True,
                        "rng_replay": "same root/phase/epoch seeds; deterministic balanced camera batches and stateless photometric augmentation; dropout disabled",
                        "gradient_epochs_completed": global_epoch,
                        "gradient_updates_completed": sum(int(row["gradient_updates"]) for row in
                                                           [json.loads(line) for line in curves_path.read_text(encoding="utf-8").splitlines() if line.strip()]),
                        "status": "running", "updated_utc": utc_now(),
                    }
                    write_json(state_path, state)
                    print(json.dumps({"event": "epoch", "run_id": run_id, "phase": phase["name"],
                                      "epoch": completed, "fit_ce": fit_metrics["macro_cross_entropy"],
                                      "stop_ce": stop_metrics["macro_cross_entropy"],
                                      "elapsed_s": curve["elapsed_seconds"]}), flush=True)
                    if stale_epochs >= int(config["patience"]):
                        break

                if phase_index == 0:
                    # Restore selected head-warmup weights while still frozen,
                    # then switch trainability and create the next phase state.
                    model._d_base.trainable = False
                    if not phase_best_weights_path:
                        raise FileNotFoundError("camera head-warmup phase has no stopping-selected checkpoint")
                    model.load_weights(str(phase_best_weights_path))
                    next_phase = phases[1]
                    d.unfreeze_top_fraction(model, next_phase["fraction"])
                    d.compile_classifier(model, next_phase["lr"], 6)
                    next_optimizer = model.optimizer
                    if hasattr(next_optimizer, "_create_all_weights"):
                        next_optimizer._create_all_weights(model.trainable_variables)
                    next_epoch = tf.Variable(0, trainable=False, dtype=tf.int32, name="camera_completed_epoch")
                    next_checkpoint = tf.train.Checkpoint(model=model, optimizer=next_optimizer, epoch=next_epoch)
                    next_manager = tf.train.CheckpointManager(
                        next_checkpoint, str(run_dir / "checkpoint" / next_phase["name"]), max_to_keep=5)
                    next_path = next_manager.save(checkpoint_number=0)
                    state = {
                        "schema": "accuracy-search-camera-adaptation-run-state-v1", "run_id": run_id,
                        "config_sha256": config_sha, "runner_sha256": runner_sha,
                        "preflight_sha256": preflight_sha,
                        "parent_result_sha256": config["parent_result_sha256"],
                        "parent_model_weights_sha256": config["parent_model_weights_sha256"],
                        "camera_train_input_sha256": camera_train_receipt["training_inputs_sha256"],
                        "grouped_split_sha256": split["group_keys_sha256"],
                        "phase_index": 1, "phase_name": next_phase["name"], "phase_epoch": 0,
                        "global_epoch": global_epoch, "phase_best_loss": None,
                        "phase_best_macro_accuracy": -1.0, "phase_best_epoch": 0, "stale_epochs": 0,
                        "best_global_loss": best_global_loss, "best_global_macro_accuracy": best_global_macro_acc,
                        "best_global_phase_index": best_global_phase_index,
                        "best_global_weights_path": str(best_global_weights_path) if best_global_weights_path else None,
                        "phase_best_weights_path": None,
                        "checkpoint_path": next_path, "optimizer_slots_preserved": True,
                        "status": "running", "updated_utc": utc_now(),
                    }
                    write_json(state_path, state)

        training_invocation_elapsed = time.monotonic() - training_started
        curve_records = [json.loads(line) for line in curves_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        gradient_elapsed = float(sum(float(row["elapsed_seconds"]) for row in curve_records))
        gradient_updates = int(sum(int(row["gradient_updates"]) for row in curve_records))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"status": "evaluation_pending", "model_selection_frozen": True,
                      "postfreeze_evaluation_started": False,
                      "gradient_epochs_completed": len(curve_records),
                      "gradient_updates_completed": gradient_updates,
                      "gradient_epoch_wall_seconds_sum": gradient_elapsed,
                      "updated_utc": utc_now()})
        write_json(state_path, state)
    else:
        curve_records = [json.loads(line) for line in curves_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        gradient_elapsed = float(sum(float(row["elapsed_seconds"]) for row in curve_records))
        gradient_updates = int(sum(int(row["gradient_updates"]) for row in curve_records))
        if not state.get("model_selection_frozen"):
            raise ValueError("evaluation resume lacks a frozen selected camera model")

    best_global_phase_index = int(state["best_global_phase_index"])
    best_weights_path = Path(str(state.get("best_global_weights_path") or ""))
    if not best_weights_path.is_file():
        raise FileNotFoundError("selected camera adaptation checkpoint is absent")
    if best_global_phase_index <= 0:
        model._d_base.trainable = False
    else:
        d.unfreeze_top_fraction(model, float(config["unfreeze_top_fraction"]))
    model.load_weights(str(best_weights_path))
    selected_model_weight_hash = model_weight_hash(model)
    selected_checkpoint_copy = run_dir / "selected_camera_model.weights.h5"
    model.save_weights(str(selected_checkpoint_copy))
    state.update({"status": "evaluation_pending", "model_selection_frozen": True,
                  "postfreeze_evaluation_started": False,
                  "selected_model_weight_sha256": selected_model_weight_hash,
                  "selected_weights_path": str(selected_checkpoint_copy),
                  "selected_weights_file_sha256": sha256_file(selected_checkpoint_copy),
                  "updated_utc": utc_now()})
    write_json(state_path, state)
    final_fit = balanced_metrics(fit_rows, eval_logits(model, fit_images))
    final_stop = balanced_metrics(stop_rows, eval_logits(model, stop_images))

    # This is the first point where calibration/development frame roles are
    # requested. The selection state is persisted as evaluation_pending above.
    state = json.loads(state_path.read_text(encoding="utf-8"))
    require_camera_adaptation_evaluation_freeze(state)
    eval_roles = {"threshold_calibration", "unknown_calibration", "development", "unknown_development"}
    camera_eval_rows, camera_eval_images, camera_eval_receipt = load_camera_images(
        int(config["roi_side_camera_px"]), allowed_roles=eval_roles)
    camera_private_records = camera_eval_receipt.pop("private_input_records")
    camera_eval_private_path = private / "CAMERA_EVALUATION_INPUTS.jsonl"
    with camera_eval_private_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in camera_private_records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    cal_rows, cal_images, cal_valid = [], [], []
    dev_rows, dev_images, dev_valid = [], [], []
    for row in camera_eval_rows:
        capture_id = str(row["request"]["capture_id"])
        role, label = str(row["role"]), str(row["label"])
        source = row["source"]
        identity = str(source.get("source_identity") or source.get("source_group") or capture_id)
        wrapped = {
            "record_id": capture_id, "label": label, "label_id": label,
            "class_index": CLASS_ID[label], "identity_partition": "unknown" if label == "OTHER" else "known",
            "identity_id": identity, "source_identity": identity,
            "source_group": str(source.get("source_group") or capture_id),
            "role": role, "capture_set": row["capture_set"], "phase": row["phase"],
        }
        valid = capture_id in camera_eval_images
        image = camera_eval_images[capture_id] if valid else np.zeros((96, 96, 1), dtype=np.uint8)
        if role in ("threshold_calibration", "unknown_calibration"):
            cal_rows.append(wrapped); cal_images.append(image); cal_valid.append(valid)
        elif role in ("development", "unknown_development"):
            dev_rows.append(wrapped); dev_images.append(image); dev_valid.append(valid)
        else:
            raise AssertionError("post-freeze camera loader returned an unapproved role")
    cal_array = np.stack(cal_images).astype(np.uint8) if cal_images else np.zeros((0, 96, 96, 1), np.uint8)
    dev_array = np.stack(dev_images).astype(np.uint8) if dev_images else np.zeros((0, 96, 96, 1), np.uint8)
    cal_valid_array = np.asarray(cal_valid, dtype=bool)
    dev_valid_array = np.asarray(dev_valid, dtype=bool)
    cal_logits = eval_logits(model, cal_array)
    valid_cal_rows = [row for row, valid in zip(cal_rows, cal_valid_array) if valid]
    valid_cal_logits = cal_logits[cal_valid_array]
    if not any(row["label"] in KNOWN for row in valid_cal_rows) or not any(row["label"] == "OTHER" for row in valid_cal_rows):
        camera_threshold = {"status": "not_selectable_missing_valid_known_or_unknown_calibration"}
        camera_dev_metrics = {"status": "not_evaluated_missing_valid_known_or_unknown_calibration"}
        dev_logits = np.zeros((len(dev_rows), 6), dtype=np.float32)
    else:
        camera_threshold = threshold_from_calibration(valid_cal_rows, valid_cal_logits)
        dev_logits = eval_logits(model, dev_array)
        camera_dev_metrics = evaluate_open_set(dev_rows, dev_logits, camera_threshold, dev_valid_array)

    model_export = run_dir / "model_float.h5"
    model.save(str(model_export), include_optimizer=False)
    model_digest = sha256_file(model_export)
    prediction_path = private / "PREDICTIONS.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        if camera_threshold.get("threshold_float32") is not None:
            def emit_predictions(mode: str, rows: list[dict], logits: np.ndarray, valid: np.ndarray) -> None:
                probs = softmax32(logits)
                scores = np.max(probs[:, :5], axis=1).astype(np.float32)
                predicted = np.argmax(probs, axis=1)
                threshold = np.float32(camera_threshold["threshold_float32"])
                for i, row in enumerate(rows):
                    record = {
                        "mode": mode, "record_id": row["record_id"], "capture_id": row["record_id"],
                        "role": row["role"], "true_label": row["label"],
                        "source_identity": row["source_identity"], "source_group": row["source_group"],
                        "valid_input": bool(valid[i]), "predicted_class": CLASSES[int(predicted[i])],
                        "score_max_known_float32": float(scores[i]),
                        "six_way_logits_float32": [float(value) for value in logits[i]],
                        "threshold_float32": float(threshold),
                        "accepted": bool(valid[i] and predicted[i] < 5 and scores[i] >= threshold),
                        "model_weight_sha256": selected_model_weight_hash,
                    }
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            emit_predictions("HOST_REFERENCE_OLD_OPTICAL_CALIBRATION", cal_rows, cal_logits, cal_valid_array)
            if dev_rows:
                emit_predictions("HOST_REFERENCE_OLD_OPTICAL_DEVELOPMENT", dev_rows, dev_logits, dev_valid_array)

    curve_records = [json.loads(line) for line in curves_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {
        "schema": "accuracy-search-camera-adaptation-result-v1", "run_id": run_id,
        "created_utc": utc_now(), "status": "complete_camera_adaptation_host_only",
        "config_sha256": config_sha, "runner_sha256": runner_sha,
        "preflight_sha256": preflight_sha,
        "parent_source_scout": {
            "run_id": parent_result["run_id"], "result_path": str(parent_result_path.relative_to(ROOT)),
            "result_sha256": sha256_file(parent_result_path),
            "selected_model_weights_path": str(parent_weights_path.relative_to(ROOT)),
            "selected_model_weights_sha256": sha256_file(parent_weights_path),
            "selected_model_weight_sha256_after_load": loaded_parent_weight_hash,
            "selected_source_role_rows_sha256": parent_result["source_selected_role_rows_sha256"],
            "training_repeated": False,
        },
        "camera_training_data": camera_train_receipt,
        "grouped_split": split,
        "model": {
            "architecture": f"MobileNetV1 alpha={config['alpha']} grayscale96 + Dense64-ReLU + six-way logits",
            "initialization": "selected v7 alpha0.1 source checkpoint",
            "initial_checkpoint_sha256": parent_result["model"]["initial_checkpoint_sha256"],
            "parameter_count": int(model.count_params()), "classes": list(CLASSES),
            "batch_norm_frozen": True, "head_dropout_rate": 0.0,
            "selected_model_weight_sha256": selected_model_weight_hash,
        },
        "training": {
            "mode": "grouped_camera_domain_adaptation_from_v7_source_weights",
            "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
            "fit_source_kind_counts": dict(Counter(row["source_kind"] for row in fit_rows)),
            "stopping_source_kind_counts": dict(Counter(row["source_kind"] for row in stop_rows)),
            "fit_rows_only_train_other_train": True, "fit_stopping_groups_disjoint": split["fit_stopping_group_overlap"] == 0,
            "calibration_development_used_for_stopping": False,
            "postfreeze_calibration_development_loaded": True,
            "best_stopping_macro_cross_entropy": float(state["best_global_loss"]),
            "best_stopping_macro_accuracy": float(state["best_global_macro_accuracy"]),
            "best_stopping_phase": phases[best_global_phase_index]["name"],
            "best_phase_index": best_global_phase_index,
            "selected_weights_path": str(selected_checkpoint_copy.relative_to(ROOT)),
            "selected_weights_sha256": sha256_file(selected_checkpoint_copy),
            "selected_model_weight_sha256": selected_model_weight_hash,
            "completed_gradient_epochs": len(curve_records),
            "completed_gradient_updates": int(sum(int(row["gradient_updates"]) for row in curve_records)),
            "gradient_epoch_wall_seconds_sum": float(sum(float(row["elapsed_seconds"]) for row in curve_records)),
            "fit_metrics_at_selected_checkpoint": final_fit,
            "stopping_metrics_at_selected_checkpoint": final_stop,
            "curves_path": str(curves_path.relative_to(ROOT)),
            "checkpoint_state_path": str(state_path.relative_to(ROOT)),
            "camera_fit_rows_per_epoch": len(fit_rows),
            "camera_stopping_rows_per_epoch": len(stop_rows),
            "sample_presentations_per_epoch": batches_per_epoch * int(config["batch_size"]),
            "maximum_gradient_updates": expected_max_updates,
            "optimizer_rng_resume": "epoch-boundary TensorFlow model+Adam checkpoint; deterministic camera batch/augmentation seeds; dropout disabled",
        },
        "old_optical_camera": {
            "threshold": camera_threshold, "development": camera_dev_metrics,
            "evaluation_inputs_receipt": camera_eval_receipt,
            "scope": "historical 2026-11-01 capture setup only; full rows retained in end-to-end denominators; not current-scene or device qualification",
        },
        "exports": {"float_model_path": str(model_export.relative_to(ROOT)),
                    "float_model_sha256": model_digest, "float_model_bytes": model_export.stat().st_size,
                    "int8_export": "not converted; no device qualification"},
        "private_evidence": {
            "camera_train_pool_index_path": str(train_index_path.relative_to(ROOT)),
            "camera_train_pool_index_sha256": sha256_file(train_index_path),
            "camera_evaluation_inputs_path": str(camera_eval_private_path.relative_to(ROOT)),
            "camera_evaluation_inputs_sha256": sha256_file(camera_eval_private_path),
            "per_image_predictions": str(prediction_path.relative_to(ROOT)),
            "per_image_predictions_sha256": sha256_file(prediction_path),
        },
        "resources": {
            "host_platform": platform.platform(), "python": platform.python_version(),
            "tensorflow": tf.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "thread_limits": {key: os.environ.get(key) for key in
                              ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
            "max_rss": {"unit": "bytes on macOS, KiB on Linux", "value": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
            "gradient_epoch_wall_seconds_sum": float(sum(float(row["elapsed_seconds"]) for row in curve_records)),
            "training_invocation_elapsed_seconds": training_invocation_elapsed,
            "total_run_elapsed_seconds": time.monotonic() - run_started,
        },
        "limits": ["one exploratory camera adaptation seed; not a ten-seed comparison",
                   "camera captures are from the historical taped-screen setup",
                   "host float model only; no current-scene recapture, int8 conversion, or Nicla qualification",
                   "calibration size does not establish a one-sided 1% unknown-risk bound"],
    }
    write_json(result_path, result)
    state.update({"status": "complete", "updated_utc": utc_now(), "result_sha256": sha256_file(result_path),
                  "postfreeze_evaluation_started": True, "selected_model_weight_sha256": selected_model_weight_hash})
    write_json(state_path, state)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-camera-adaptation", action="store_true")
    parser.add_argument("--launch-camera-adaptation", action="store_true",
                        help="explicit second gate for a coordinator-authorized grouped camera adaptation run")
    parser.add_argument("--check-contract", action="store_true")
    args = parser.parse_args()
    if args.check_contract:
        print(json.dumps(contract_checks(), sort_keys=True))
        return
    if args.preflight:
        report = data_preflight(args.config.resolve())
        print(json.dumps({"run_id": report["run_id"], "report_path": report["report_path"],
                          "source_counts": report["source_rows"]["role_label_counts"],
                          "split": report["split"], "training_controls": report["training_controls"],
                          "camera_replay": report["camera_replay"]}, sort_keys=True))
        return
    if args.preflight_camera_adaptation:
        report = camera_adaptation_preflight(args.config.resolve())
        print(json.dumps({"run_id": report["run_id"], "report_path": report["report_path"],
                          "parent_source_scout": report["parent_source_scout"],
                          "grouped_split": report["grouped_split"],
                          "training_plan": report["training_plan"],
                          "fit_stopping_contract": report["fit_stopping_contract"],
                          "contract_checks": report["contract_checks"]}, sort_keys=True))
        return
    config, _ = model_config(args.config.resolve())
    if config.get("training_launch_enabled") is False:
        raise RuntimeError("this accuracy-search config is preflight-only; training launch is disabled")
    if config.get("training_mode") == "grouped_camera_adaptation_from_parent":
        if not args.launch_camera_adaptation:
            raise RuntimeError("grouped camera adaptation requires --launch-camera-adaptation after coordinator review")
        result = train_camera_adaptation_run(args.config.resolve(), resume=args.resume,
                                             launch_authorized=args.launch_camera_adaptation)
        print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                          "fit": result["training"]["fit_metrics_at_selected_checkpoint"],
                          "stop": result["training"]["stopping_metrics_at_selected_checkpoint"],
                          "camera_dev": result["old_optical_camera"]["development"]}, sort_keys=True))
    else:
        result = train_run(args.config.resolve(), resume=args.resume)
        print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                          "fit": result["training"]["fit_metrics_at_selected_checkpoint"],
                          "stop": result["training"]["stopping_metrics_at_selected_checkpoint"],
                          "source_dev": result["source_lfw"]["development"],
                          "camera_dev": result["old_optical_camera"]["development"]}, sort_keys=True))


if __name__ == "__main__":
    main()
