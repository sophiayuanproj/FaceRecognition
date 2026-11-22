"""Fresh corrected-input five-known + OTHER head experiment (ROI120->48).

This wrapper reuses the preserved v5 optimizer implementation but binds it to
the corrected versioned camera features and a separate result namespace.
Training roots 56001-56010 are new initialization/sampling streams.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import v5_camera_task48 as trainer

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
INPUTS = STUDY / "runs/camera_corrected_input_v1"
PRIVATE = STUDY / "private/camera_corrected_input_v1"
OUT = STUDY / "runs/camera_corrected_recognition_v1"
ROI_SIDE_CAMERA_PX = 120
TENSOR_SIDE_PX = 48
ROOTS = tuple(range(56001, 56011))


def camera_data():
    rows = [json.loads(line) for line in (INPUTS / "CAPTURE_INDEX.jsonl").read_text().splitlines()]
    with np.load(PRIVATE / "features.npz") as data:
        features = np.asarray(data["features"], dtype=np.float32)
    if len(rows) != len(features):
        raise ValueError("corrected features and capture role rows do not align")
    for row in rows:
        row["fomo_single_crop_usable"] = row["usable_for_identity"]
    return rows, features[:, None, :], np.asarray([row["usable_for_identity"] for row in rows], dtype=bool)


def run(roots):
    trainer.OUT = OUT / ("roi_%d_camera_px" % ROI_SIDE_CAMERA_PX)
    trainer.CROP_SIZES = (ROI_SIDE_CAMERA_PX,)
    trainer.camera_data = camera_data
    trainer.PRIVATE_FEATURES = PRIVATE / "features.npz"
    trainer.OUT.mkdir(parents=True, exist_ok=True)
    source_rows, source_features, source_detected, source_index_sha = trainer.source_training_data()
    camera_rows, camera_features, camera_detected = camera_data()
    model_path = STUDY / "runs/operator_fused_encoder48/int8.tflite"
    protocol = {
        "schema": "v6-camera-corrected-recognition-1",
        "parent_input_experiment": "runs/camera_corrected_input_v1/INPUTS.json",
        "scope": "exploratory camera-matched six-class linear head on the existing frozen 48px encoder",
        "labels": ["K01", "K02", "K03", "K04", "K05", "OTHER"],
        "roi_side_camera_px": ROI_SIDE_CAMERA_PX,
        "encoder_tensor_hwc": [TENSOR_SIDE_PX, TENSOR_SIDE_PX, 3],
        "encoder_path": "runs/operator_fused_encoder48/int8.tflite",
        "encoder_sha256": trainer.sha256(model_path),
        "backend": "TensorFlow 2.10.1 BUILTIN_REF",
        "head": "float32 six-class linear softmax; 774 trainable parameters; host only",
        "training": "class-balanced source+camera train groups, 5+5 rows/class/update, Adam, source-known stopping role",
        "threshold": "next representable binary32 above maximum binary32 known-class score among detected unknown_calibration rows",
        "coordinate_contract": "FOMO returned xywh rectangles already in camera pixels; fixed 120px ROI shifted inward at borders; final tensor 48x48 RGB",
        "source_train_index_sha256": source_index_sha,
        "corrected_camera_capture_index_sha256": trainer.sha256(INPUTS / "CAPTURE_INDEX.jsonl"),
        "corrected_camera_feature_cache_sha256": trainer.sha256(PRIVATE / "features.npz"),
        "camera_fit_and_threshold_groups_inherit_frozen_roles": True,
        "roots": list(ROOTS), "future_final_used": False, "device_qualified": False,
    }
    protocol_path = trainer.OUT / "PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("protocol changed; create a new result namespace")
    trainer.task.dump(protocol_path, protocol)
    keep = np.asarray([row["role"] in ("train", "other_train", "stopping") for row in source_rows])
    source_rows = [row for row, yes in zip(source_rows, keep) if yes]
    source_features = source_features[keep]
    source_detected = source_detected[keep]
    summaries = []
    for seed in roots:
        result = trainer.fit_root(seed, ROI_SIDE_CAMERA_PX, source_rows, source_features,
                                  source_detected, camera_rows, camera_features,
                                  camera_detected, 0)
        complete = trainer.OUT / ("crop_%d" % ROI_SIDE_CAMERA_PX) / str(seed) / "COMPLETE.json"
        row = json.loads(complete.read_text())
        row.pop("crop_fomo_input_px", None)
        row["roi_side_camera_px"] = ROI_SIDE_CAMERA_PX
        row["encoder_tensor_hwc"] = [48, 48, 3]
        trainer.task.dump(complete, row)
        overfit = json.loads((complete.parent / "OVERFIT.json").read_text())
        summary = {
            "seed": seed,
            "train_only_control_pass": overfit["train_only_control_pass"],
            "control_initial_loss": overfit["initial_loss"],
            "control_final_loss": overfit["final_loss"],
            "control_final_top1": overfit["final_top1"],
            "camera_known_calibration_correct_acceptance": result["results"]["known_calibration_correct_acceptance"],
            "camera_known_calibration_coverage": result["results"]["known_calibration_coverage"],
            "optical_development_correct_acceptance": result["results"]["known_correct_acceptance"],
            "optical_development_coverage": result["results"]["known_coverage"],
            "unknown_false_identity_groups": result["results"]["unknown_development_group_risk"]["false_groups"],
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        trainer.task.dump(trainer.OUT / "ROOT_RESULTS.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    args = parser.parse_args()
    unknown = sorted(set(args.roots) - set(ROOTS))
    if unknown:
        raise ValueError("root id is not in the prespecified new seed set: %r" % unknown)
    run(tuple(args.roots))


if __name__ == "__main__":
    main()
