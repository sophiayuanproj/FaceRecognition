"""OpenCV Zoo SFace host reference and fixed-teacher MobileNet distillation.

Run from the study workspace with PYTHONPATH=FaceRecognition and the Mac
training environment. Final roles are deliberately handled only by ``finish``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf

from nicla_study import d_campaign as d
from nicla_study.fs import sha256_file
from nicla_study.state import resource_lease


SFACE_REL = Path("work/project-finish-v4/models/pretrained/sface/face_recognition_sface_2021dec.onnx")
YUNET_REL = Path("work/project-finish-v4/models/pretrained/sface/face_detection_yunet_2023mar.onnx")
SFACE_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
ADDENDUM_REL = Path("work/project-finish-v4/D_TEACHER_ADDENDUM_v1.json")
WIDTHS = (0.1, 0.25)
GALLERY_COUNTS = (1, 3, 5, 10, 60)
POLICIES = ("quality", "uniform")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_hash(value) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    d.write_json(path, value)


def paths_for(workspace: Path) -> dict:
    paths = d.study_paths(workspace)
    runs = paths["runs"] / "teacher"
    return paths | {
        "teacher_runs": runs,
        "teacher_cache": runs / "sface_nonfinal_features.npz",
        "teacher_index": runs / "sface_nonfinal_feature_index.json",
        "teacher_result": paths["reports"] / "SFACE_TEACHER_RESULT.json",
        "student_search": runs / "student_search",
        "student_roots": runs / "students",
        "teacher_registry": paths["reports"] / "TEACHER_MODEL_REGISTRY_FROZEN.json",
        "teacher_final": paths["handoff"] / "TEACHER_FINAL_SUMMARY.json",
        "progress": runs / "TEACHER_PROGRESS.json",
    }


def _model_hashes(workspace: Path) -> dict:
    sface, yunet = workspace / SFACE_REL, workspace / YUNET_REL
    if not sface.is_file() or not yunet.is_file():
        raise FileNotFoundError("downloaded SFace and YuNet checkpoints are required")
    actual_sface, actual_yunet = sha256_file(sface), sha256_file(yunet)
    if actual_sface != SFACE_SHA or actual_yunet != YUNET_SHA:
        raise ValueError("SFace or YuNet checkpoint checksum mismatch")
    return {"sface": {"path": str(sface), "sha256": actual_sface, "bytes": sface.stat().st_size},
            "yunet": {"path": str(yunet), "sha256": actual_yunet, "bytes": yunet.stat().st_size}}


def _environment(workspace: Path) -> dict:
    return d.env_receipt(workspace) | {"opencv": cv2.__version__, "onnxruntime": ort.__version__,
        "teacher_inference_backend": "OpenCV DNN CPU", "student_training_backend": "TensorFlow CPU"}


def _role_data(data: dict) -> list[tuple[str, list[dict], np.ndarray]]:
    return [("train", data["train_rows"], data["train_gray"]),
            ("stopping", data["stop_rows"], data["stop_gray"]),
            ("threshold_calibration", data["cal_known"], data["cal_known_gray"]),
            ("unknown_calibration", data["cal_unknown"], data["cal_unknown_gray"]),
            ("development", data["dev_known"], data["dev_known_gray"]),
            ("unknown_development", data["dev_unknown"], data["dev_unknown_gray"])]


def _make_models(workspace: Path):
    model_hashes = _model_hashes(workspace)
    sface = cv2.FaceRecognizerSF.create(model_hashes["sface"]["path"], "")
    detector = cv2.FaceDetectorYN.create(model_hashes["yunet"]["path"], "", (96, 96), 0.9, 0.3, 5000)
    return sface, detector, model_hashes


def _one_crop_features(sface, detector, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, dict]:
    image = np.asarray(gray, dtype=np.uint8)
    if image.shape != (96, 96):
        raise ValueError("SFace teacher expected the verified 96x96 gray8 C crop")
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raw = np.asarray(sface.feature(bgr), dtype=np.float32).reshape(-1)
    if raw.shape != (128,) or not np.isfinite(raw).all():
        raise ValueError("SFace feature output failed the observed 128-D finite contract")
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) != 1:
        return raw, None, {"detection_count": 0 if faces is None else int(len(faces)),
                           "detector_confidence": None}
    face = faces[0]
    aligned = sface.alignCrop(bgr, face)
    if aligned.shape != (112, 112, 3):
        raise ValueError("OpenCV SFace alignCrop did not return the reference 112x112 BGR crop")
    aligned_feature = np.asarray(sface.feature(aligned), dtype=np.float32).reshape(-1)
    if aligned_feature.shape != (128,) or not np.isfinite(aligned_feature).all():
        raise ValueError("aligned SFace feature output failed the observed 128-D finite contract")
    return raw, aligned_feature, {"detection_count": 1, "detector_confidence": float(face[-1])}


def _extract_teacher_features(workspace: Path, paths: dict, data: dict) -> dict:
    if paths["teacher_cache"].is_file() and paths["teacher_index"].is_file():
        index = json.loads(paths["teacher_index"].read_text())
        if index.get("status") == "complete_verified" and index.get("cache_sha256") == sha256_file(paths["teacher_cache"]):
            loaded = np.load(paths["teacher_cache"])
            raw = loaded["raw"]
            aligned = loaded["aligned"]
            ids = index["record_ids"]
            if len(ids) == len(raw) == len(aligned):
                return {"raw": {rid: raw[i] for i, rid in enumerate(ids)},
                        "aligned": {rid: aligned[i] if np.isfinite(aligned[i]).all() else None
                                    for i, rid in enumerate(ids)},
                        "index": index}
    sface, detector, model_hashes = _make_models(workspace)
    rows_all, gray_all, role_all = [], [], []
    for role, rows, images in _role_data(data):
        rows_all.extend(rows)
        gray_all.extend(images)
        role_all.extend([role] * len(rows))
    record_ids = [str(row["record_id"]) for row in rows_all]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("non-final teacher feature source contains duplicate record IDs")
    raw_values, aligned_values, detection_rows = [], [], []
    for index, (row, gray, role) in enumerate(zip(rows_all, gray_all, role_all)):
        raw, aligned, detection = _one_crop_features(sface, detector, np.asarray(gray).squeeze())
        raw_values.append(raw)
        aligned_values.append(aligned if aligned is not None else np.full((128,), np.nan, dtype=np.float32))
        detection_rows.append({"record_id": str(row["record_id"]), "role": role,
                               "detection_count": detection["detection_count"],
                               "detector_confidence": detection["detector_confidence"]})
        if (index + 1) % 100 == 0:
            progress = {"status": "teacher_feature_extraction", "features_completed": index + 1,
                        "features_total": len(rows_all), "updated_utc": now()}
            write_json(paths["progress"], progress)
    raw_array = np.asarray(raw_values, dtype=np.float32)
    aligned_array = np.asarray(aligned_values, dtype=np.float32)
    paths["teacher_runs"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["teacher_cache"], raw=raw_array, aligned=aligned_array)
    role_counts = Counter(role_all)
    detected = Counter(row["role"] for row in detection_rows if row["detection_count"] == 1)
    index = {"schema": "job-d-sface-feature-cache-1", "status": "complete_verified",
             "feature_mode": "OpenCV Zoo SFace FaceRecognizerSF.feature on unnormalized BGR8 triplicated grayscale crop, bbox=None",
             "aligned_mode": "YuNet 5-landmark detection at confidence 0.9; FaceRecognizerSF.alignCrop to 112x112 BGR; missing/multiple detections retained as unavailable",
             "record_ids": record_ids, "roles": role_all,
             "rows_sha256": d.rows_fingerprint(rows_all), "role_counts": dict(role_counts),
             "alignment_detection_counts": {role: {"rows": count, "exactly_one_detected": int(detected[role]),
                "coverage": float(detected[role] / count) if count else None} for role, count in role_counts.items()},
             "detection_rows": detection_rows, "checkpoint_hashes": model_hashes,
             "opencv_version": cv2.__version__, "feature_shape": [128],
             "raw_feature_norm_min": float(np.min(np.linalg.norm(raw_array, axis=1))),
             "raw_feature_norm_max": float(np.max(np.linalg.norm(raw_array, axis=1))),
             "created_utc": now()}
    index["cache_sha256"] = sha256_file(paths["teacher_cache"])
    write_json(paths["teacher_index"], index)
    return {"raw": {rid: raw_array[i] for i, rid in enumerate(record_ids)},
            "aligned": {rid: aligned_array[i] if np.isfinite(aligned_array[i]).all() else None
                        for i, rid in enumerate(record_ids)},
            "index": index}


def _gallery_curve(known_scores: np.ndarray, known_rows: list[dict],
                   unknown_scores: np.ndarray, unknown_rows: list[dict], points: int = 101) -> list[dict]:
    ks = np.max(known_scores, axis=1)
    us = np.max(unknown_scores, axis=1)
    pred = np.argmax(known_scores, axis=1)
    true = np.asarray([d.row_label_index(row) for row in known_rows])
    thresholds = np.unique(np.quantile(np.concatenate([ks, us]), np.linspace(0.0, 1.0, points)))
    curve = []
    for threshold in thresholds:
        accept = ks >= threshold
        per_person_tpir = [float(np.mean(accept[(true == i)] & (pred[(true == i)] == i))) for i in range(5)]
        curve.append({"threshold": float(threshold), "tpir_correct_known_acceptance": float(np.mean(per_person_tpir)),
                      "fpir": float(np.mean(us >= threshold)),
                      "known_coverage": float(np.mean(accept)), "unknown_n": len(unknown_rows)})
    return curve


def _candidate_metrics(embeddings: dict[str, np.ndarray], train_known: list[dict],
                       cal_known: list[dict], cal_unknown: list[dict],
                       dev_known: list[dict], dev_unknown: list[dict], seed: int) -> dict:
    candidates = []
    for policy in POLICIES:
        for count in GALLERY_COUNTS:
            prototypes, _, selection = d.build_gallery_prototypes(embeddings, train_known, count, policy, seed)
            score = lambda rows: d.gallery_scores(np.stack([embeddings[str(row["record_id"])] for row in rows]), prototypes)
            cal_kn, cal_un = score(cal_known), score(cal_unknown)
            dev_kn, dev_un = score(dev_known), score(dev_unknown)
            calibration = d.fit_gallery_threshold(cal_kn, cal_known, cal_un, cal_unknown)
            development = d.gallery_metrics(dev_kn, dev_known, dev_un, dev_unknown, calibration["threshold"])
            candidates.append({"template_count_requested": count, "policy": policy,
                               "template_count_actual": selection[d.KNOWN[0]]["template_count"],
                               "selection": selection, "calibration": calibration,
                               "development": development,
                               "development_reliability": d.gallery_reliability(
                                   dev_kn, dev_known, dev_un, calibration["threshold"]),
                               "development_open_set_curve": _gallery_curve(dev_kn, dev_known, dev_un, dev_unknown),
                               "prototype_sha256": json_hash(prototypes.tolist())})
    selected = max(candidates, key=lambda x: (float(x["development"]["development_macro_utility"]),
                                               float(x["development"]["known_macro_identification"]),
                                               -int(x["template_count_requested"]),
                                               x["policy"] == "quality"))
    prototypes, _, _ = d.build_gallery_prototypes(embeddings, train_known,
        int(selected["template_count_requested"]), selected["policy"], seed)
    selected_dev_known = d.gallery_scores(_features_for_rows(embeddings, dev_known), prototypes)
    selected_dev_unknown = d.gallery_scores(_features_for_rows(embeddings, dev_unknown), prototypes)
    selected["development_bootstrap"] = d.bootstrap_gallery_metrics(
        selected_dev_known, dev_known, selected_dev_unknown, dev_unknown,
        float(selected["calibration"]["threshold"]), seed, replicates=500)
    return {"candidates": candidates, "selected": selected}


def _features_for_rows(embeddings: dict[str, np.ndarray], rows: list[dict]) -> np.ndarray:
    return np.stack([embeddings[str(row["record_id"])] for row in rows]).astype(np.float32)


def _evaluate_teacher_reference(workspace: Path, paths: dict, data: dict,
                                feature_store: dict) -> dict:
    train_known = [row for row in data["train_rows"] if row.get("label_id") in d.KNOWN and
                   row.get("identity_partition") != "unknown"]
    result = _candidate_metrics(feature_store["raw"], train_known,
                                data["cal_known"], data["cal_unknown"],
                                data["dev_known"], data["dev_unknown"], 20261300)
    aligned_ids = {rid for rid, value in feature_store["aligned"].items() if value is not None}
    alignment_rows = {str(row["record_id"]): row for row in (data["train_rows"] + data["stop_rows"] +
                      data["cal_known"] + data["cal_unknown"] + data["dev_known"] + data["dev_unknown"])}
    aligned_train = [row for row in train_known if str(row["record_id"]) in aligned_ids]
    aligned_cal_known = [row for row in data["cal_known"] if str(row["record_id"]) in aligned_ids]
    aligned_cal_unknown = [row for row in data["cal_unknown"] if str(row["record_id"]) in aligned_ids]
    aligned_dev_known = [row for row in data["dev_known"] if str(row["record_id"]) in aligned_ids]
    aligned_dev_unknown = [row for row in data["dev_unknown"] if str(row["record_id"]) in aligned_ids]
    aligned_result = None
    if all((len(aligned_train), len(aligned_cal_known), len(aligned_cal_unknown),
            len(aligned_dev_known), len(aligned_dev_unknown))):
        aligned_map = {rid: value for rid, value in feature_store["aligned"].items() if value is not None}
        aligned_result = _candidate_metrics(aligned_map, aligned_train, aligned_cal_known, aligned_cal_unknown,
                                            aligned_dev_known, aligned_dev_unknown, 20261301)
    selected = result["selected"]
    model_hashes = _model_hashes(workspace)
    config = {"teacher": "SFace 2021dec", "feature_mode": "bbox=None raw 96x96 gray crop replicated to BGR8",
              "output_contract": "embedding-128/cosine", "gallery_count": selected["template_count_requested"],
              "gallery_policy": selected["policy"]}
    calibration = selected["calibration"]
    freeze = {"schema": "job-d-model-freeze-1", "status": "frozen",
              "model_sha256": model_hashes["sface"]["sha256"], "config_sha256": json_hash(config),
              "calibration_sha256": json_hash(calibration),
              "protocol_sha256": json_hash({"matrix_sha256": sha256_file(paths["matrix"]),
                                             "addendum_sha256": sha256_file(workspace / ADDENDUM_REL)}),
              "final_role_opened": False, "model_id": "sface_2021dec_host_reference",
              "output_contract": "embedding-128/gallery-cosine", "created_utc": now()}
    write_json(paths["teacher_runs"] / "SFACE_HOST.FROZEN.json", freeze)
    output = {"schema": "job-d-sface-reference-result-1", "status": "complete_verified",
              "dataset": "survface_96", "branch": "public K01-K05; no personal roster",
              "checkpoint": model_hashes, "preprocessing_and_alignment_audit": feature_store["index"],
              "crop_mode": "raw_cropped input, no landmarks, full-crop coverage",
              "input_contract": {"crop_source": "C raw gray8 [1,96,96,1]", "opencv_input": "uint8 BGR [96,96,3] formed by gray-channel triplication", "landmarks": "none in primary crop mode", "canonical_aligned_input": "uint8 BGR [112,112,3] from OpenCV alignCrop"},
              "output_contract": {"onnx_tensor": "float32 [1,128]", "score": "L2-normalized cosine against gallery prototype", "host_only": True},
              "environment": _environment(workspace),
              "crop_mode_results": result,
              "canonical_aligned_diagnostic": {"detector_input_threshold": 0.9,
                  "eligible_counts_by_role": {role: sum(1 for row in feature_store["index"]["detection_rows"]
                      if row["role"] == role and row["detection_count"] == 1)
                      for role in ("train", "stopping", "threshold_calibration", "unknown_calibration",
                                   "development", "unknown_development")},
                  "conditional_results": aligned_result,
                  "note": "Conditional aligned-only metrics are reported only if every needed role has at least one eligible image; no missing crops are imputed."},
              "selected": selected, "freeze_receipt": str(paths["teacher_runs"] / "SFACE_HOST.FROZEN.json"),
              "license_and_exposure": {"directory_license": "Apache-2.0 stated by OpenCV Zoo SFace directory",
                  "exact_checkpoint_training_data": "unresolved; no exact corpus identifier in model metadata or README",
                  "maintainer_status": "OpenCV Zoo issue 313 was open without an answer when checked 2026-10-29",
                  "pretraining_overlap": "unknown; cannot certify no overlap with public K01-K05 or SurvFace",
                  "deployment_eligible": False,
                  "reason": "research reference only until checkpoint training-data and upstream authorization provenance is clarified"},
              "created_utc": now()}
    write_json(paths["teacher_result"], output)
    return output


def _balanced_augment(gray: np.ndarray, seed: int, augment: bool) -> np.ndarray:
    return d.augment_gray8(gray, seed) if augment else np.asarray(gray, dtype=np.uint8)


def _train_distilled(workspace: Path, rows: list[dict], images: np.ndarray,
                     teacher: dict[str, np.ndarray], stop_rows: list[dict], stop_images: np.ndarray,
                     stop_teacher: dict[str, np.ndarray], alpha: float, seed: int,
                     config: dict, run_name: str) -> tuple[tf.keras.Model, dict]:
    model, embedding_model, info = d.build_classifier(workspace, alpha, True, 6, seed, run_name)
    d.unfreeze_top_fraction(model, float(config["unfreeze_top_fraction"]))
    optimizer = tf.keras.optimizers.Adam(float(config["learning_rate"]))
    teacher_train = d._normalize_vectors(np.stack([teacher[str(row["record_id"])] for row in rows]))
    teacher_stop = d._normalize_vectors(np.stack([stop_teacher[str(row["record_id"])] for row in stop_rows]))
    rng = np.random.default_rng(seed ^ 0xA871)
    best_weights, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    curve, steps, presentations, max_gradient = [], 0, 0, 0.0
    started = time.monotonic()
    for epoch in range(int(config["epochs_max"])):
        batch_list = d.balanced_survface_indices(rows, 24, seed + epoch * 1009)
        epoch_losses = []
        for batch_no, batch in enumerate(batch_list):
            x_u8 = np.stack([_balanced_augment(images[index].squeeze(), seed + epoch * 100000 + batch_no * 100 + j,
                                                config["augmentation"] == "mild_gray_v1") for j, index in enumerate(batch)])
            x = d.gray_float(x_u8[..., None])
            target = teacher_train[np.asarray(batch, dtype=np.int64)]
            with tf.GradientTape() as tape:
                prediction = tf.math.l2_normalize(embedding_model(x, training=True), axis=-1)
                target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
                student_sim = tf.matmul(prediction, prediction, transpose_b=True)
                teacher_sim = tf.matmul(target_tensor, target_tensor, transpose_b=True)
                mask = 1.0 - tf.eye(tf.shape(student_sim)[0], dtype=tf.float32)
                loss = tf.reduce_sum(tf.square(student_sim - teacher_sim) * mask) / tf.reduce_sum(mask)
            variables = embedding_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            pairs = [(grad, var) for grad, var in zip(gradients, variables) if grad is not None]
            if not pairs or not all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g, _ in pairs):
                raise ValueError("teacher student training produced a missing or nonfinite gradient")
            grad_norm = float(tf.linalg.global_norm([g for g, _ in pairs]).numpy())
            if not np.isfinite(grad_norm) or grad_norm <= 0:
                raise ValueError("teacher student gradient norm is zero or nonfinite")
            max_gradient = max(max_gradient, grad_norm)
            optimizer.apply_gradients(pairs)
            epoch_losses.append(float(loss.numpy()))
            steps += 1
            presentations += len(batch)
        stop_pred = d._normalize_vectors(embedding_model(d.gray_float(stop_images), training=False).numpy())
        stop_sim = stop_pred @ stop_pred.T
        stop_target_sim = teacher_stop @ teacher_stop.T
        val_loss = float(np.mean(np.square(stop_sim - stop_target_sim)))
        record = {"epoch": epoch + 1, "mean_training_pairwise_cosine_mse": float(np.mean(epoch_losses)),
                  "stopping_pairwise_cosine_mse": val_loss, "batch_count": len(batch_list),
                  "sample_presentations": len(batch_list) * 24, "maximum_gradient_global_norm": max_gradient}
        curve.append(record)
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_weights = [np.array(v, copy=True) for v in embedding_model.get_weights()]
        else:
            stale += 1
        if stale >= 3:
            break
    if best_weights is None:
        raise ValueError("student distillation never produced a selectable stopping checkpoint")
    embedding_model.set_weights(best_weights)
    history = {"root_seed": seed, "width": alpha, "initialization": info["initialization"],
               "initial_checkpoint_sha256": info["checkpoint_sha256"], "config": config,
               "epochs_run": len(curve), "selected_epoch": best_epoch,
               "selected_stopping_pairwise_cosine_mse": best_loss,
               "steps_run": steps, "sample_presentations": presentations,
               "max_gradient_global_norm": max_gradient, "curve": curve,
               "elapsed_seconds": time.monotonic() - started,
               "training_rows": len(rows), "stopping_rows": len(stop_rows),
               "input_contract": {"shape": [None, 96, 96, 1], "dtype": "float32", "normalization": "gray8/255"},
               "teacher_target": "fixed 128-D SFace feature; pairwise cosine similarity matrix; targets generated only from train/unknown_train rows"}
    return model, history


def _embedding_map(embedding_model, images: np.ndarray, rows: list[dict]) -> dict[str, np.ndarray]:
    features = embedding_model(d.gray_float(images), training=False).numpy()
    if features.shape != (len(rows), 64) or not np.isfinite(features).all():
        raise ValueError("student host embedding output did not match [N,64]")
    return {str(row["record_id"]): features[i] for i, row in enumerate(rows)}


def _backend_gallery(embedding_map: dict[str, np.ndarray], data: dict, seed: int) -> dict:
    train_known = [row for row in data["train_rows"] if row.get("label_id") in d.KNOWN and
                   row.get("identity_partition") != "unknown"]
    return _candidate_metrics(embedding_map, train_known, data["cal_known"], data["cal_unknown"],
                              data["dev_known"], data["dev_unknown"], seed)


def _train_candidate_search(workspace: Path, paths: dict, data: dict, feature_store: dict) -> dict:
    candidate_specs = [
        {"id": "D1", "learning_rate": 0.0003, "unfreeze_top_fraction": 0.25,
         "augmentation": "mild_gray_v1", "epochs_max": 8},
        {"id": "D2", "learning_rate": 0.0001, "unfreeze_top_fraction": 0.5,
         "augmentation": "none", "epochs_max": 8},
    ]
    paths["student_search"].mkdir(parents=True, exist_ok=True)
    selected_by_width, candidate_results = {}, []
    for wi, alpha in enumerate(WIDTHS):
        width_candidates = []
        base_seed = 20261120 + wi * 101
        for ci, config in enumerate(candidate_specs):
            result_path = paths["student_search"] / ("a%s_%s.json" % ("010" if alpha == 0.1 else "025", config["id"]))
            if result_path.is_file():
                saved = json.loads(result_path.read_text())
                if saved.get("status") == "complete_verified":
                    width_candidates.append(saved)
                    candidate_results.append(saved)
                    continue
            seed = base_seed
            run_name = "sface_distill_search_a%s_%s" % ("010" if alpha == 0.1 else "025", config["id"])
            model, history = _train_distilled(workspace, data["train_rows"], data["train_gray"],
                feature_store["raw"], data["stop_rows"], data["stop_gray"], feature_store["raw"],
                alpha, seed, config, run_name)
            embedding = model._d_embedding_model
            eval_rows = (data["train_rows"] + data["cal_known"] + data["cal_unknown"] +
                         data["dev_known"] + data["dev_unknown"])
            eval_images = np.concatenate([data["train_gray"], data["cal_known_gray"], data["cal_unknown_gray"],
                                          data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
            emb_map = _embedding_map(embedding, eval_images, eval_rows)
            metrics = _backend_gallery(emb_map, data, seed)
            result = {"schema": "job-d-sface-student-search-1", "status": "complete_verified",
                      "width": alpha, "root_seed": seed, "config": config, "training": history,
                      "selected_gallery": metrics["selected"], "development": metrics["selected"]["development"],
                      "development_macro_utility": metrics["selected"]["development"]["development_macro_utility"],
                      "development_correct_acceptance": metrics["selected"]["development"]["known_macro_correct_acceptance"],
                      "development_known_identification": metrics["selected"]["development"]["known_macro_identification"],
                      "gallery_candidates": metrics["candidates"], "created_utc": now()}
            write_json(result_path, result)
            width_candidates.append(result)
            candidate_results.append(result)
            del model
            tf.keras.backend.clear_session()
        selected_by_width[str(alpha)] = max(width_candidates, key=lambda r: (
            float(r["development_correct_acceptance"]), float(r["development_known_identification"]),
            -float(r["config"]["learning_rate"]), r["config"]["id"] == "D1"))
    result = {"schema": "job-d-sface-student-search-1", "status": "complete_verified",
              "candidates": candidate_results, "selected_by_width": selected_by_width,
              "selected_by_width_sha256": json_hash({w: v["config"] for w, v in selected_by_width.items()}),
              "selection_roles": ["threshold_calibration", "unknown_calibration", "development", "unknown_development"],
              "final_role_opened": False, "created_utc": now()}
    write_json(paths["student_search"] / "SEARCH_RESULT.json", result)
    return result


def _tflite_embeddings(path: Path, images: np.ndarray) -> np.ndarray:
    return d.tflite_outputs(path, images)


def _train_one_student_root(workspace: Path, paths: dict, data: dict, feature_store: dict,
                             alpha: float, seed: int, config: dict) -> dict:
    root = paths["student_roots"] / ("a%s" % ("010" if alpha == 0.1 else "025")) / ("seed-%d" % seed)
    result_path = root / "ROOT_RESULT.json"
    if result_path.is_file():
        saved = json.loads(result_path.read_text())
        if saved.get("status") == "complete_verified":
            receipt_path = Path(saved.get("freeze_receipt", ""))
            if not receipt_path.is_file():
                raise ValueError("completed student root lacks its freeze receipt: " + str(result_path))
            receipt = json.loads(receipt_path.read_text())
            expected = {"weights": (Path(saved["weights_path"]), saved["weights_sha256"]),
                        "float": (Path(saved["float_export"]["path"]), saved["float_export"]["sha256"]),
                        "int8": (Path(saved["int8_export"]["path"]), saved["int8_export"]["sha256"])}
            if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False or any(
                    not path.is_file() or sha256_file(path) != digest for path, digest in expected.values()):
                raise ValueError("completed student root has a missing or changed frozen artifact: " + str(result_path))
            return saved
    model, history = _train_distilled(workspace, data["train_rows"], data["train_gray"], feature_store["raw"],
        data["stop_rows"], data["stop_gray"], feature_store["raw"], alpha, seed, config,
        "sface_distill_a%s_seed%d" % ("010" if alpha == 0.1 else "025", seed))
    embedding = model._d_embedding_model
    root.mkdir(parents=True, exist_ok=True)
    weights_path = root / "student_embedding.weights.h5"
    d.atomic_save_weights(embedding, weights_path)
    float_export = d.convert_float_tflite(embedding, root / "embedding_float.tflite")
    rep_indices = np.linspace(0, len(data["train_gray"]) - 1, min(128, len(data["train_gray"])), dtype=np.int64).tolist()
    int8_export = d.convert_int8_tflite(embedding, data["train_gray"], root / "embedding_int8.tflite", rep_indices)
    eval_rows = (data["train_rows"] + data["cal_known"] + data["cal_unknown"] +
                 data["dev_known"] + data["dev_unknown"])
    eval_images = np.concatenate([data["train_gray"], data["cal_known_gray"], data["cal_unknown_gray"],
                                  data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    float_features = _embedding_map(embedding, eval_images, eval_rows)
    int8_values = _tflite_embeddings(root / "embedding_int8.tflite", eval_images)
    int8_features = {str(row["record_id"]): int8_values[i] for i, row in enumerate(eval_rows)}
    float_metrics = _backend_gallery(float_features, data, seed)
    int8_metrics = _backend_gallery(int8_features, data, seed)
    training_rows_path = root / "training_rows.jsonl"
    d._save_rows_jsonl(training_rows_path, data["train_rows"])
    curve_path = root / "training_curve.jsonl"
    with curve_path.open("w", encoding="utf-8") as stream:
        for row in history["curve"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    config_path = root / "config.json"
    write_json(config_path, config)
    calibration_path = root / "calibrations.json"
    write_json(calibration_path, {"float": float_metrics["selected"]["calibration"],
                                  "int8": int8_metrics["selected"]["calibration"]})
    float_sel, int8_sel = float_metrics["selected"], int8_metrics["selected"]
    root_result = {"schema": "job-d-sface-distilled-student-root-1", "status": "complete_verified",
        "dataset": "survface_96", "branch": "public K01-K05; no personal roster", "root_seed": seed,
        "arm_id": "sv_sface_distill_mnv1_a%s" % ("010" if alpha == 0.1 else "025"),
        "width": alpha, "initialization": "verified Edge Impulse grayscale MobileNetV1 checkpoint",
        "pretrained": True, "teacher": "fixed SFace 2021dec; not independently initialized",
        "teacher_sha256": SFACE_SHA, "teacher_pretraining_exposure": "unknown; no overlap exclusion claim",
        "checkpoint_sha256": d.EI_HASHES[alpha], "checkpoint_source_url": d.EI_URLS[alpha],
        "training": history, "config": config, "config_sha256": sha256_file(config_path),
        "weights_path": str(weights_path), "weights_sha256": sha256_file(weights_path),
        "train_manifest_source_path": str(paths["primary"]),
        "train_manifest_source_sha256": sha256_file(paths["primary"]),
        "training_rows_file": str(training_rows_path), "training_rows_file_sha256": sha256_file(training_rows_path),
        "curve_file": str(curve_path), "curve_file_sha256": sha256_file(curve_path),
        "float_export": float_export, "int8_export": int8_export,
        "output_contract": "embedding-64/gallery-cosine", "label_order": d.KNOWN,
        "input_contract": {"float_tflite": {"shape": [1, 96, 96, 1], "dtype": "float32", "normalization": "gray8/255"},
                           "full_int8_tflite": {"shape": [1, 96, 96, 1], "dtype": "int8", "scale": 1.0 / 255.0, "zero_point": -128}},
        "output_contract_typed": {"float_tflite": {"shape": [1, 64], "dtype": "float32"},
                                   "full_int8_tflite": {"shape": [1, 64], "dtype": "int8", "dequantization": "use output tensor scale and zero point in export audit"}},
        "environment": _environment(workspace),
        "float_gallery_candidates": float_metrics["candidates"], "int8_gallery_candidates": int8_metrics["candidates"],
        "selected_float": float_sel, "selected_int8": int8_sel,
        "calibration_sha256": sha256_file(calibration_path),
        "representative_sample_indices": rep_indices,
        "representative_rows_content_sha256": json_hash([data["train_rows"][i]["record_id"] for i in rep_indices]),
        "host_float_int8_embedding_cosine_drift_mean": float(np.mean([1.0 - float(np.dot(
            d._normalize_vectors(float_features[rid][None, :])[0],
            d._normalize_vectors(int8_features[rid][None, :])[0])) for rid in float_features])),
        "development_decision_disagreement_count": int(_development_disagreements(float_metrics, int8_metrics,
            float_features, int8_features, data, seed)), "created_utc": now()}
    write_json(result_path, root_result)
    protocol = json_hash({"matrix_sha256": sha256_file(paths["matrix"]),
                          "addendum_sha256": sha256_file(workspace / ADDENDUM_REL),
                          "student_search_sha256": sha256_file(paths["student_search"] / "SEARCH_RESULT.json")})
    receipt = {"schema": "job-d-model-freeze-1", "status": "frozen",
        "model_sha256": root_result["weights_sha256"],
        "model_artifacts": {"host_checkpoint_sha256": root_result["weights_sha256"],
            "float_tflite_sha256": float_export["sha256"], "int8_tflite_sha256": int8_export["sha256"]},
        "config_sha256": root_result["config_sha256"], "calibration_sha256": root_result["calibration_sha256"],
        "protocol_sha256": protocol, "final_role_opened": False,
        "model_id": root_result["arm_id"], "root_seed": seed,
        "output_contract": root_result["output_contract"], "created_utc": now()}
    write_json(root / "FROZEN.json", receipt)
    root_result["freeze_receipt"] = str(root / "FROZEN.json")
    write_json(result_path, root_result)
    del model
    tf.keras.backend.clear_session()
    return root_result


def _development_disagreements(float_metrics: dict, int8_metrics: dict, float_map: dict[str, np.ndarray],
                               int8_map: dict[str, np.ndarray], data: dict, seed: int) -> int:
    # Compare the same gallery count/policy with each backend's own calibrated threshold.
    float_sel = float_metrics["selected"]
    int8_match = next(row for row in int8_metrics["candidates"]
        if row["template_count_requested"] == float_sel["template_count_requested"] and
           row["policy"] == float_sel["policy"])
    results = []
    for selection, embedding_map in ((float_sel, float_map), (int8_match, int8_map)):
        prototypes, _, _ = d.build_gallery_prototypes(embedding_map,
            [r for r in data["train_rows"] if r.get("label_id") in d.KNOWN and r.get("identity_partition") != "unknown"],
            int(selection["template_count_requested"]), selection["policy"], seed)
        scores = d.gallery_scores(_features_for_rows(embedding_map, data["dev_known"] + data["dev_unknown"]), prototypes)
        threshold = float(selection["calibration"]["threshold"])
        results.append((np.argmax(scores, axis=1), np.max(scores, axis=1) >= threshold))
    return int(np.sum((results[0][0] != results[1][0]) | (results[0][1] != results[1][1])))


def _seed_map(addendum: dict) -> dict[float, list[int]]:
    return {0.1: addendum["student"]["roots"]["alpha_0.1"],
            0.25: addendum["student"]["roots"]["alpha_0.25"]}


def _freeze_teacher_registry(workspace: Path, paths: dict, teacher_result: dict,
                             search: dict, roots: list[dict]) -> dict:
    if len(roots) != 20:
        raise ValueError("teacher/student registry requires twenty completed student roots")
    teacher_receipt = json.loads(Path(teacher_result["freeze_receipt"]).read_text())
    if teacher_receipt.get("status") != "frozen" or teacher_receipt.get("final_role_opened") is not False:
        raise ValueError("fixed SFace teacher must be frozen before student registry creation")
    teacher_file = Path(teacher_result["checkpoint"]["sface"]["path"])
    if not teacher_file.is_file() or sha256_file(teacher_file) != teacher_receipt.get("model_sha256"):
        raise ValueError("fixed SFace teacher checkpoint changed before registry freeze")
    model_entries = [{"dataset": "survface_96", "model_id": "sface_2021dec_host_reference",
        "family_type": "fixed_face_pretrained_host_teacher", "root_seed": None,
        "result_path": str(paths["teacher_result"]),
        "freeze_receipt": teacher_result["freeze_receipt"], "output_contract": "embedding-128/gallery-cosine",
        "pretrained": True, "deployment_eligible": False, "status": "research_reference"}]
    for result in roots:
        receipt = json.loads(Path(result["freeze_receipt"]).read_text())
        for artifact in (result["weights_path"], result["float_export"]["path"], result["int8_export"]["path"]):
            if not Path(artifact).is_file():
                raise FileNotFoundError("student root freeze references a missing artifact: " + str(artifact))
        if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False:
            raise ValueError("every student root must be frozen before registry creation")
        if (sha256_file(Path(result["weights_path"])) != receipt.get("model_sha256") or
                sha256_file(Path(result["float_export"]["path"])) != receipt.get("model_artifacts", {}).get("float_tflite_sha256") or
                sha256_file(Path(result["int8_export"]["path"])) != receipt.get("model_artifacts", {}).get("int8_tflite_sha256")):
            raise ValueError("student artifact hash mismatch before registry creation")
        model_entries.append({"dataset": "survface_96", "model_id": result["arm_id"],
            "family_type": "sface_distilled_mobilenet_student", "root_seed": result["root_seed"],
            "result_path": result["result_path"], "freeze_receipt": result["freeze_receipt"],
            "output_contract": result["output_contract"], "pretrained": True,
            "initialization_source": result["checkpoint_source_url"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "teacher_sha256": result["teacher_sha256"], "deployment_eligible": False,
            "float_tflite": result["float_export"], "int8_tflite": result["int8_export"]})
    registry = {"schema": "job-d-teacher-student-registry-1", "status": "complete_verified",
        "addendum_id": "D-sface-host-reference-and-mobilenet-distillation-v1",
        "matrix_id": "D-five-person-transfer-open-set-gallery-v1",
        "matrix_sha256": sha256_file(paths["matrix"]), "addendum_sha256": sha256_file(workspace / ADDENDUM_REL),
        "search_path": str(paths["student_search"] / "SEARCH_RESULT.json"),
        "search_sha256": sha256_file(paths["student_search"] / "SEARCH_RESULT.json"),
        "teacher_result_path": str(paths["teacher_result"]),
        "teacher_result_sha256": sha256_file(paths["teacher_result"]),
        "model_count": len(model_entries), "student_root_count": len(roots), "models": model_entries,
        "final_role_opened": False, "created_utc": now()}
    write_json(paths["teacher_registry"], registry)
    return registry


def run_teacher_campaign(workspace: Path) -> dict:
    paths = paths_for(workspace)
    paths["teacher_runs"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    addendum = json.loads((workspace / ADDENDUM_REL).read_text())
    if not addendum.get("frozen_before_student_training"):
        raise ValueError("teacher addendum must be frozen before student training")
    _model_hashes(workspace)
    data = d.load_role_arrays(workspace, paths, "survface_96")
    with resource_lease(paths["study"], "heavy_training", owner="D", timeout_s=0):
        feature_store = _extract_teacher_features(workspace, paths, data)
        if not paths["teacher_result"].is_file():
            teacher_result = _evaluate_teacher_reference(workspace, paths, data, feature_store)
        else:
            teacher_result = json.loads(paths["teacher_result"].read_text())
        search = _train_candidate_search(workspace, paths, data, feature_store)
        root_records = []
        seed_map = _seed_map(addendum)
        total = 20
        done = 0
        for alpha in WIDTHS:
            config = search["selected_by_width"][str(alpha)]["config"]
            for seed in seed_map[alpha]:
                result = _train_one_student_root(workspace, paths, data, feature_store, alpha, int(seed), config)
                result_path = paths["student_roots"] / ("a%s" % ("010" if alpha == 0.1 else "025")) / ("seed-%d" % seed) / "ROOT_RESULT.json"
                result["result_path"] = str(result_path)
                root_records.append(result)
                done += 1
                write_json(paths["progress"], {"status": "student_training", "completed_roots": done,
                    "expected_roots": total, "last_width": alpha, "last_root_seed": int(seed), "updated_utc": now()})
        registry = _freeze_teacher_registry(workspace, paths, teacher_result, search, root_records)
    campaign = {"schema": "job-d-teacher-student-campaign-1", "status": "complete_verified",
        "teacher_result": str(paths["teacher_result"]), "feature_cache": str(paths["teacher_cache"]),
        "student_search": str(paths["student_search"] / "SEARCH_RESULT.json"),
        "student_roots": len(root_records), "registry": str(paths["teacher_registry"]),
        "registry_sha256": sha256_file(paths["teacher_registry"]), "final_role_opened": False,
        "created_utc": now()}
    write_json(paths["teacher_runs"] / "TEACHER_STUDENT_CAMPAIGN.json", campaign)
    write_json(paths["progress"], {"status": "teacher_student_complete", "completed_roots": len(root_records),
                                    "expected_roots": 20, "updated_utc": now()})
    return campaign


def _metrics_for_final(embedding_map: dict[str, np.ndarray], train_known: list[dict],
                       final_known: list[dict], final_unknown: list[dict], selected: dict,
                       seed: int) -> dict:
    prototypes, _, selection = d.build_gallery_prototypes(embedding_map, train_known,
        int(selected["template_count_requested"]), selected["policy"], seed)
    known_matrix = d.gallery_scores(_features_for_rows(embedding_map, final_known), prototypes)
    unknown_matrix = d.gallery_scores(_features_for_rows(embedding_map, final_unknown), prototypes)
    threshold = float(selected["calibration"]["threshold"])
    return {"metrics": d.gallery_metrics(known_matrix, final_known, unknown_matrix, final_unknown, threshold),
            "bootstrap": d.bootstrap_gallery_metrics(known_matrix, final_known, unknown_matrix, final_unknown,
                                                       threshold, seed, replicates=1000),
            "reliability": d.gallery_reliability(known_matrix, final_known, unknown_matrix, threshold),
            "open_set_curve": _gallery_curve(known_matrix, final_known, unknown_matrix, final_unknown),
            "known_score_matrix": known_matrix, "unknown_score_matrix": unknown_matrix,
            "prototype_selection": selection}


def finish_teacher_campaign(workspace: Path) -> dict:
    paths = paths_for(workspace)
    registry_path = paths["teacher_registry"]
    if not registry_path.is_file():
        raise FileNotFoundError("teacher/student registry must be frozen before final evaluation")
    registry = json.loads(registry_path.read_text())
    if registry.get("status") != "complete_verified" or registry.get("final_role_opened") is True:
        raise ValueError("teacher/student registry is incomplete or already consumed")
    frozen_entries = [entry for entry in registry["models"] if entry.get("freeze_receipt")]
    for entry in frozen_entries:
        receipt = json.loads(Path(entry["freeze_receipt"]).read_text())
        if receipt.get("status") != "frozen" or receipt.get("final_role_opened") is not False:
            raise PermissionError("teacher/student freeze receipt is invalid before final access")
        if entry.get("root_seed") is None:
            teacher_result_path = Path(entry["result_path"])
            teacher_result = json.loads(teacher_result_path.read_text())
            model_path = Path(teacher_result["checkpoint"]["sface"]["path"])
            if sha256_file(model_path) != receipt.get("model_sha256"):
                raise ValueError("SFace teacher checkpoint hash changed after freeze")
        else:
            root_result = json.loads(Path(entry["result_path"]).read_text())
            checked = [(Path(root_result["weights_path"]), receipt.get("model_sha256")),
                       (Path(root_result["float_export"]["path"]), receipt.get("model_artifacts", {}).get("float_tflite_sha256")),
                       (Path(root_result["int8_export"]["path"]), receipt.get("model_artifacts", {}).get("int8_tflite_sha256"))]
            if any(not path.is_file() or sha256_file(path) != digest for path, digest in checked):
                raise ValueError("student model artifact hash changed after freeze")
    first_receipt = json.loads(Path(frozen_entries[0]["freeze_receipt"]).read_text())
    known_rows = d.dataset_role_rows(workspace, paths["primary"], "survface_96", ("final",),
                                     "evaluate_final", first_receipt)
    unknown_rows = d.dataset_role_rows(workspace, paths["primary"], "survface_96", ("unknown_final",),
                                       "evaluate_final", first_receipt)
    known_gray = d.cache_gray8(workspace, known_rows, "evaluate_final", first_receipt)
    unknown_gray = d.cache_gray8(workspace, unknown_rows, "evaluate_final", first_receipt)
    write_json(paths["reports"] / "TEACHER_FINAL_ROLE_OPENED.json", {
        "schema": "job-d-teacher-final-role-opened-1", "status": "opened_after_freeze",
        "opened_utc": now(), "registry_sha256": sha256_file(registry_path),
        "matrix_sha256": sha256_file(paths["matrix"]),
        "addendum_sha256": sha256_file(workspace / ADDENDUM_REL),
        "known_role_image_hash_manifest_sha256": d.rows_fingerprint(known_rows),
        "unknown_role_image_hash_manifest_sha256": d.rows_fingerprint(unknown_rows),
        "known_rows": len(known_rows), "unknown_rows": len(unknown_rows),
        "unknown_identity_count": len({r["identity_id"] for r in unknown_rows}),
        "opened_by": "Job D teacher/student final evaluation after registry freeze"})
    data = d.load_role_arrays(workspace, paths, "survface_96")
    train_known = [row for row in data["train_rows"] if row.get("label_id") in d.KNOWN and
                   row.get("identity_partition") != "unknown"]
    teacher = json.loads(paths["teacher_result"].read_text())
    sface, detector, _ = _make_models(workspace)
    teacher_final = {}
    for row, gray in zip(known_rows + unknown_rows, np.concatenate([known_gray, unknown_gray], axis=0)):
        raw, _, _ = _one_crop_features(sface, detector, gray.squeeze())
        teacher_final[str(row["record_id"])] = raw
    teacher_nonfinal = _extract_teacher_features(workspace, paths, data)["raw"]
    teacher_map = dict(teacher_nonfinal) | teacher_final
    teacher_selected = teacher["selected"]
    teacher_eval = _metrics_for_final(teacher_map, train_known, known_rows, unknown_rows, teacher_selected, 20261300)
    teacher_result = {k: v for k, v in teacher_eval.items() if not k.endswith("score_matrix")}
    teacher_result["model_id"] = "sface_2021dec_host_reference"
    teacher_result["dataset"] = "survface_96"
    teacher_result["known_rows"] = len(known_rows)
    teacher_result["unknown_rows"] = len(unknown_rows)
    teacher_result["unknown_identity_count"] = len({r["identity_id"] for r in unknown_rows})
    teacher_result["selected_development_gallery"] = {k: teacher_selected[k] for k in
        ("template_count_requested", "policy", "calibration", "development")}
    teacher_result["per_person_confusion"] = _confusion(teacher_eval["known_score_matrix"], known_rows,
                                                       teacher_eval["unknown_score_matrix"], unknown_rows,
                                                       teacher_selected["calibration"]["threshold"])
    write_json(paths["teacher_runs"] / "SFACE_HOST_FINAL.json", teacher_result)
    student_final = []
    development_final_agreement = []
    for entry in registry["models"]:
        if entry.get("family_type") != "sface_distilled_mobilenet_student":
            continue
        root_result = json.loads(Path(entry["result_path"]).read_text())
        root_dir = Path(root_result["weights_path"]).parent
        # Recreate gallery templates from the permitted training rows only.
        train_indices = [i for i, row in enumerate(data["train_rows"])
                         if row.get("label_id") in d.KNOWN and row.get("identity_partition") != "unknown"]
        train_gray_all = data["train_gray"][train_indices]
        train_rows_this = [data["train_rows"][i] for i in train_indices]
        images_all = np.concatenate([train_gray_all, known_gray, unknown_gray], axis=0)
        rows_all = train_rows_this + known_rows + unknown_rows
        per_root = {"arm_id": root_result["arm_id"], "root_seed": root_result["root_seed"],
                    "float": None, "int8": None}
        backend_arrays = {}
        for backend in ("float", "int8"):
            export_path = root_dir / ("embedding_float.tflite" if backend == "float" else "embedding_int8.tflite")
            features = _tflite_embeddings(export_path, images_all)
            emb_map = {str(row["record_id"]): features[i] for i, row in enumerate(rows_all)}
            selected = root_result["selected_float" if backend == "float" else "selected_int8"]
            eval_result = _metrics_for_final(emb_map, train_rows_this, known_rows, unknown_rows,
                                             selected, int(root_result["root_seed"]))
            backend_arrays[backend] = (eval_result["known_score_matrix"], eval_result["unknown_score_matrix"],
                                       float(selected["calibration"]["threshold"]))
            result_clean = {k: v for k, v in eval_result.items() if not k.endswith("score_matrix")}
            result_clean["selected_development_gallery"] = {k: selected[k] for k in
                ("template_count_requested", "policy", "calibration", "development")}
            result_clean["per_person_confusion"] = _confusion(eval_result["known_score_matrix"], known_rows,
                eval_result["unknown_score_matrix"], unknown_rows, selected["calibration"]["threshold"])
            per_root[backend] = result_clean
        float_known, float_unknown, float_threshold = backend_arrays["float"]
        int8_known, int8_unknown, int8_threshold = backend_arrays["int8"]
        f_pred = np.concatenate([np.argmax(float_known, axis=1), np.argmax(float_unknown, axis=1)])
        q_pred = np.concatenate([np.argmax(int8_known, axis=1), np.argmax(int8_unknown, axis=1)])
        f_score = np.concatenate([np.max(float_known, axis=1), np.max(float_unknown, axis=1)])
        q_score = np.concatenate([np.max(int8_known, axis=1), np.max(int8_unknown, axis=1)])
        f_accept, q_accept = f_score >= float_threshold, q_score >= int8_threshold
        per_root["float_int8_final_decision_disagreement_count"] = int(np.sum((f_pred != q_pred) | (f_accept != q_accept)))
        per_root["float_int8_calibrated_threshold_drift"] = float(int8_threshold - float_threshold)
        per_root["float_int8_final_max_score_mean_absolute_difference"] = float(np.mean(np.abs(f_score - q_score)))
        teacher_known_matrix = teacher_eval["known_score_matrix"]
        teacher_threshold = float(teacher_selected["calibration"]["threshold"])
        teacher_known_pred = np.argmax(teacher_known_matrix, axis=1)
        teacher_known_accept = np.max(teacher_known_matrix, axis=1) >= teacher_threshold
        true_known = np.asarray([d.row_label_index(row) for row in known_rows])
        teacher_correct = teacher_known_accept & (teacher_known_pred == true_known)
        for backend, (student_known_matrix, student_unknown_matrix, student_threshold) in backend_arrays.items():
            student_pred = np.argmax(student_known_matrix, axis=1)
            student_accept = np.max(student_known_matrix, axis=1) >= student_threshold
            student_correct = student_accept & (student_pred == true_known)
            teacher_wrong = ~teacher_correct
            teacher_unknown_fp = np.max(teacher_eval["unknown_score_matrix"], axis=1) >= teacher_threshold
            student_unknown_fp = np.max(student_unknown_matrix, axis=1) >= student_threshold
            teacher_comparison = {"known_decision_agreement": float(np.mean(
                    (teacher_known_pred == student_pred) & (teacher_known_accept == student_accept))),
                "student_correct_when_teacher_wrong_count": int(np.sum(student_correct & teacher_wrong)),
                "student_correction_rate_given_teacher_error": float(np.mean(student_correct[teacher_wrong])) if np.any(teacher_wrong) else None,
                "student_error_when_teacher_correct_count": int(np.sum(~student_correct & teacher_correct)),
                "student_harm_rate_given_teacher_correct": float(np.mean(~student_correct[teacher_correct])) if np.any(teacher_correct) else None,
                "student_rejects_teacher_unknown_false_identification_count": int(np.sum(teacher_unknown_fp & ~student_unknown_fp)),
                "student_new_unknown_false_identification_count": int(np.sum(~teacher_unknown_fp & student_unknown_fp)),
                "unknown_identity_count": len({row["identity_id"] for row in unknown_rows}),
                "known_rows": len(known_rows)}
            per_root.setdefault("teacher_conditional_comparison", {})[backend] = teacher_comparison
        per_root_path = root_dir / "FINAL_RESULT.json"
        per_root["status"] = "complete_verified"
        per_root["dataset"] = "survface_96"
        per_root["known_rows"] = len(known_rows)
        per_root["unknown_rows"] = len(unknown_rows)
        per_root["unknown_identity_count"] = len({r["identity_id"] for r in unknown_rows})
        write_json(per_root_path, per_root)
        entry["final_result_path"] = str(per_root_path)
        receipt_path = Path(root_result["freeze_receipt"])
        receipt = json.loads(receipt_path.read_text())
        receipt["final_role_opened"] = True
        receipt["final_opened_utc"] = now()
        write_json(receipt_path, receipt)
        student_final.append(per_root)
    teacher_receipt_path = Path(teacher["freeze_receipt"])
    teacher_receipt = json.loads(teacher_receipt_path.read_text())
    teacher_receipt["final_role_opened"] = True
    teacher_receipt["final_opened_utc"] = now()
    write_json(teacher_receipt_path, teacher_receipt)
    registry["final_role_opened"] = True
    registry["final_role_opened_utc"] = now()
    registry["final_results"] = {"teacher": str(paths["teacher_runs"] / "SFACE_HOST_FINAL.json"),
                                  "student_root_count": len(student_final)}
    write_json(registry_path, registry)
    parent_registry_path = paths["reports"] / "MODEL_REGISTRY_FROZEN.json"
    parent_registry = None
    if parent_registry_path.is_file():
        parent_registry = json.loads(parent_registry_path.read_text())
        parent_registry["teacher_student_registry_sha256"] = sha256_file(registry_path)
        parent_registry["teacher_student_models"] = registry["models"]
        parent_registry["teacher_student_final_role_opened"] = True
        parent_registry["teacher_student_final_summary"] = str(paths["teacher_final"])
        write_json(parent_registry_path, parent_registry)
    summary = {"schema": "job-d-teacher-student-final-summary-1", "status": "complete_verified",
        "teacher": teacher_result, "student_aggregate_by_width_and_backend": _aggregate_student_metrics(student_final),
        "students": student_final,
        "primary_dataset": "SurvFace 96x96 public grayscale crops; exploratory because K01-K05 appeared in v3 development",
        "role_counts": {"known_final": len(known_rows), "unknown_final": len(unknown_rows),
                        "unknown_identity_count": len({r["identity_id"] for r in unknown_rows})},
        "registry": str(registry_path), "registry_sha256": sha256_file(registry_path),
        "combined_model_registry": str(parent_registry_path) if parent_registry else None,
        "combined_model_registry_sha256": sha256_file(parent_registry_path) if parent_registry else None,
        "final_role_opened": True, "deployment_boundary": "SFace is host-only and provenance-unresolved; no student is board-qualified until physical arena/operator/power tests.",
        "created_utc": now()}
    write_json(paths["teacher_final"], summary)
    final_summary_path = paths["handoff"] / "FINAL_SUMMARY.json"
    if final_summary_path.is_file():
        parent_summary = json.loads(final_summary_path.read_text())
        parent_summary["teacher_student_final_summary_path"] = str(paths["teacher_final"])
        parent_summary["teacher_student_final_summary_sha256"] = sha256_file(paths["teacher_final"])
        parent_summary["teacher_student_registry_sha256"] = sha256_file(registry_path)
        if parent_registry_path.is_file():
            parent_summary["model_registry_sha256"] = sha256_file(parent_registry_path)
        parent_summary["teacher_student_conditional_improvement"] = _aggregate_teacher_student(student_final)
        write_json(final_summary_path, parent_summary)
    return summary


def _aggregate_teacher_student(final_roots: list[dict]) -> dict:
    by_family = {}
    for width in ("010", "025"):
        rows = [root for root in final_roots if ("a" + width) in root["arm_id"]]
        by_family[width] = {}
        for backend in ("float", "int8"):
            values = [root.get("teacher_conditional_comparison", {}).get(backend, {}) for root in rows]
            for metric in ("known_decision_agreement", "student_correction_rate_given_teacher_error",
                           "student_harm_rate_given_teacher_correct"):
                entries = [float(value[metric]) for value in values if value.get(metric) is not None]
                by_family[width].setdefault(backend, {})[metric] = {
                    "mean_across_roots": float(np.mean(entries)) if entries else None,
                    "root_count": len(entries)}
            for metric in ("student_correct_when_teacher_wrong_count", "student_error_when_teacher_correct_count",
                           "student_rejects_teacher_unknown_false_identification_count",
                           "student_new_unknown_false_identification_count"):
                entries = [int(value[metric]) for value in values if metric in value]
                by_family[width][backend][metric] = {"mean_across_roots": float(np.mean(entries)) if entries else None,
                                                      "root_count": len(entries)}
    return by_family


def _aggregate_student_metrics(final_roots: list[dict]) -> dict:
    metric_names = ("known_macro_identification", "known_macro_correct_acceptance",
                    "known_macro_wrong_known_identity", "known_macro_rejection",
                    "known_macro_coverage", "unknown_false_identification_rate",
                    "unknown_identity_macro_fpir", "unknown_rejection_rate", "development_macro_utility")
    result = {}
    for width in ("010", "025"):
        family = [root for root in final_roots if ("a" + width) in root["arm_id"]]
        result[width] = {}
        for backend in ("float", "int8"):
            rows = [root[backend]["metrics"] for root in family if root.get(backend)]
            result[width][backend] = {name: {"mean": float(np.mean([row[name] for row in rows if row.get(name) is not None]))
                if any(row.get(name) is not None for row in rows) else None,
                "sd_across_roots": float(np.std([row[name] for row in rows if row.get(name) is not None], ddof=1))
                if sum(row.get(name) is not None for row in rows) > 1 else None,
                "root_count": sum(row.get(name) is not None for row in rows)} for name in metric_names}
    return result


def _confusion(known_scores: np.ndarray, known_rows: list[dict], unknown_scores: np.ndarray,
               unknown_rows: list[dict], threshold: float) -> dict:
    truth = [str(row["label_id"]) for row in known_rows]
    pred = [d.KNOWN[int(v)] if float(s) >= threshold else "REJECT"
            for v, s in zip(np.argmax(known_scores, axis=1), np.max(known_scores, axis=1))]
    unknown_pred = [d.KNOWN[int(v)] if float(s) >= threshold else "REJECT"
                    for v, s in zip(np.argmax(unknown_scores, axis=1), np.max(unknown_scores, axis=1))]
    labels = d.KNOWN + ["REJECT", "UNKNOWN"]
    mapping = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(truth, pred):
        matrix[mapping[t], mapping[p]] += 1
    for p in unknown_pred:
        matrix[mapping["UNKNOWN"], mapping[p]] += 1
    return {"labels": labels, "counts": matrix.tolist(),
            "unknown_false_identification_count": int(sum(p in d.KNOWN for p in unknown_pred)),
            "unknown_rejection_count": int(unknown_pred.count("REJECT"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SFace teacher/student investigation for Job D")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("action", choices=("train", "finish"))
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    result = run_teacher_campaign(workspace) if args.action == "train" else finish_teacher_campaign(workspace)
    print(json.dumps({"status": result.get("status"), "result": result}, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
