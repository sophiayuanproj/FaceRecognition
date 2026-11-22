#!/usr/bin/env python3
"""Build and calibrate the fixed D gallery using its exact exported INT8 model."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from nicla_study.data_pipeline import load_pixels, read_jsonl
from nicla_study.d_campaign import (_normalize_vectors, fit_gallery_threshold,
    gallery_scores, sha256_file, sha256_json, tflite_outputs)

KNOWN = ["K01", "K02", "K03", "K04", "K05"]
MODEL_ID = "d-a010-int8-seed-20261110"
MODEL_SHA = "0bf2bc5fb05b0c02d5d09217fe01834182578ebd92f9c001cd088c8ce5a9b0cd"


def main():
    study = PROJECT / "work/project-finish-v4"
    stage = ROOT / "models/deployment-v1"
    selection_path = study / "runs/D/gallery/a01/seed-20261110/int8_uniform.CONFIG.json"
    selection_doc = json.loads(selection_path.read_text())
    selected = selection_doc["selected"]
    ids_by_label = {label: [str(x) for x in value["record_ids"]]
                    for label, value in selected["selection"].items()}
    if list(ids_by_label) != KNOWN or any(len(ids) != 3 for ids in ids_by_label.values()):
        raise ValueError("D frozen roster must contain three selected train records per known ID")

    source_manifest = study / "data/primary_split_manifest.jsonl"
    rows = read_jsonl(source_manifest)
    train_rows = [row for row in rows if row.get("dataset") == "survface_96" and row.get("role") == "train"]
    by_id = {str(row["record_id"]): row for row in train_rows}
    selected_rows = []
    for label in KNOWN:
        for record_id in ids_by_label[label]:
            row = by_id.get(record_id)
            if row is None or row.get("label_id") != label:
                raise ValueError("D selected record is absent from its authorized train role: " + record_id)
            selected_rows.append(row)
    cal_known = [row for row in rows if row.get("dataset") == "survface_96" and row.get("role") == "threshold_calibration"]
    cal_unknown = [row for row in rows if row.get("dataset") == "survface_96" and row.get("role") == "unknown_calibration"]
    if len(cal_known) != 50 or len(cal_unknown) != 300:
        raise ValueError("D calibration requires its frozen 50 known / 300 unknown calibration rows")

    selected_images = np.stack([load_pixels(row, PROJECT, output="gray8", action="train")
                                for row in selected_rows]).astype(np.uint8)
    known_images = np.stack([load_pixels(row, PROJECT, output="gray8", action="threshold_fit")
                             for row in cal_known]).astype(np.uint8)
    unknown_images = np.stack([load_pixels(row, PROJECT, output="gray8", action="threshold_fit")
                               for row in cal_unknown]).astype(np.uint8)
    model = stage / "identity_embedding_a010_int8.tflite"
    if sha256_file(model) != MODEL_SHA:
        raise ValueError("staged D embedder hash mismatch")

    # D's campaign L2-normalizes every INT8 embedding before averaging each
    # three-image identity template. Refit only the existing threshold-fit
    # roles if platform inference no longer reproduces the historical vector
    # digest byte-for-byte; never read the held-out final role here.
    all_images = np.concatenate((selected_images, known_images, unknown_images), axis=0)
    outputs = tflite_outputs(model, all_images).astype(np.float32)
    if outputs.shape != (len(all_images), 64) or not np.isfinite(outputs).all():
        raise ValueError("D embedder output must be finite [N,64]")
    normalized = _normalize_vectors(outputs)
    prototype_rows = []
    offset = 0
    for label in KNOWN:
        group = normalized[offset:offset + 3]
        prototype_rows.append(_normalize_vectors(np.mean(group, axis=0, keepdims=True))[0])
        offset += 3
    prototypes = np.stack(prototype_rows).astype(np.float32)
    prototype_sha = sha256_json(prototypes.tolist())

    q_start = len(selected_rows)
    q_mid = q_start + len(cal_known)
    known_scores = gallery_scores(normalized[q_start:q_mid], prototypes)
    unknown_scores = gallery_scores(normalized[q_mid:], prototypes)
    calibration = fit_gallery_threshold(known_scores, cal_known, unknown_scores, cal_unknown,
                                        target_fpir=0.05)
    if not calibration.get("constraint_met"):
        raise ValueError("recalibrated D gallery failed its 5% unknown FPIR constraint")
    threshold = float(calibration["threshold"])
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("D gallery threshold outside cosine range")

    config_path = ROOT / "firmware/runtime_config.json"
    config = json.loads(config_path.read_text())
    model_cfg = config["identity"]["small"]
    model_cfg["threshold"] = threshold
    model_cfg["prototype_sha256"] = prototype_sha
    calibration_source = stage / "embedding_gallery_calibration.json"
    calibration_doc = {"schema": "F-D-gallery-calibration-1", "status": "recalibrated_on_frozen_roles",
        "model_id": MODEL_ID, "model_sha256": MODEL_SHA,
        "selected_template_ids": ids_by_label,
        "prototype_sha256": prototype_sha,
        "historical_D_prototype_sha256": selected["prototype_sha256"],
        "historical_D_threshold": float(selected["calibration"]["threshold"]),
        "threshold": threshold, "calibration": calibration,
        "calibration_known_records": [str(row["record_id"]) for row in cal_known],
        "calibration_unknown_records": [str(row["record_id"]) for row in cal_unknown],
        "roles_used": ["train", "threshold_calibration", "unknown_calibration"],
        "final_role_accessed": False,
        "note": "Recomputed from selected D train IDs and the same frozen calibration roles because the historical prototype digest was not reproducible byte-for-byte in the current host environment."}
    calibration_source.write_text(json.dumps(calibration_doc, sort_keys=True, indent=2) + "\n")
    calibration_sha = sha256_file(calibration_source)
    model_cfg["threshold_source_sha256"] = calibration_sha
    model_cfg["calibration_role"] = "F refit on D threshold_calibration+unknown_calibration"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")

    gallery = {"schema_version": "wearable-roster-1", "status": "ready",
        "roster_id": "K01-K05-public-study-roster", "identity_order": KNOWN,
        "model_id": MODEL_ID, "model_sha256": MODEL_SHA,
        "model_config_sha256": model_cfg["model_config_sha256"],
        "prototype_sha256": prototype_sha,
        "selection_sha256": sha256_file(selection_path),
        "manifest_sha256": sha256_file(source_manifest),
        "calibration": {"threshold": threshold, "threshold_source_sha256": calibration_sha,
            "uncertainty_band": float(config["identity"]["uncertainty_band"]),
            "score": "maximum cosine to L2-normalized identity-mean prototype",
            "role": "F refit on D threshold_calibration+unknown_calibration"},
        "source_record_ids": ids_by_label,
        "historical_D_prototype_sha256": selected["prototype_sha256"],
        "entries": [{"identity_id": label,
                     "vectors": {MODEL_ID: [float(value) for value in prototypes[index]]}}
                    for index, label in enumerate(KNOWN)]}
    gallery_path = stage / "fixed_roster_gallery.json"
    gallery_path.write_text(json.dumps(gallery, sort_keys=True, indent=2) + "\n")
    integration_path = stage / "integration_manifest.json"
    integration = json.loads(integration_path.read_text())
    integration["runtime_config"] = {"path": str(config_path), "sha256": sha256_file(config_path)}
    integration["gallery"] = {"path": str(gallery_path), "sha256": sha256_file(gallery_path),
        "prototype_sha256": prototype_sha, "status": "ready"}
    integration.setdefault("files", {})["embedding_gallery_calibration.json"] = {
        "path": str(calibration_source), "sha256": calibration_sha,
        "bytes": calibration_source.stat().st_size}
    integration["D_gallery_calibration"] = {"path": str(calibration_source),
        "sha256": calibration_sha, "threshold": threshold,
        "historical_D_prototype_reproduced": prototype_sha == selected["prototype_sha256"],
        "final_role_accessed": False}
    integration_path.write_text(json.dumps(integration, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "ready", "prototype_sha256": prototype_sha,
        "historical_D_prototype_sha256": selected["prototype_sha256"],
        "historical_D_prototype_reproduced": prototype_sha == selected["prototype_sha256"],
        "threshold": threshold, "threshold_source_sha256": calibration_sha,
        "calibration_known_n": len(cal_known), "calibration_unknown_n": len(cal_unknown),
        "selected_records": ids_by_label, "gallery_sha256": sha256_file(gallery_path),
        "model_sha256": MODEL_SHA, "final_role_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
