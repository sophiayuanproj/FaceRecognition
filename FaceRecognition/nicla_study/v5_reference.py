"""Correct aligned host face reference and explicit MCU-style box-crop cache."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from .paths import resolve_study_root

WORKSPACE = Path(__file__).resolve().parents[2]
MODEL_DIR = WORKSPACE / "work/project-finish-v4/models/pretrained/sface"


def main():
    cv2.setNumThreads(2)
    study = resolve_study_root(WORKSPACE, "work/project-research-v5")
    output = study / "runs/host_reference"
    output.mkdir(parents=True, exist_ok=True)
    manifest = study / "data/manifests/lfw_roles.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()
            if "future_final" not in json.loads(line)["role"]]
    cache = output / "nonfinal_features.npz"
    if cache.exists():
        saved = np.load(cache)
        features, detected = saved["features"], saved["detected"]
    else:
        detector = cv2.FaceDetectorYN.create(str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
                                             "", (250, 250), .85, .3, 5000)
        encoder = cv2.FaceRecognizerSF.create(str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"), "")
        features = np.zeros((len(rows), 128), np.float32)
        detected = np.zeros(len(rows), bool)
        boxes = np.zeros((len(rows), 4), np.float32)
        gray96 = np.zeros((len(rows), 96, 96, 1), np.uint8)
        rgb96 = np.zeros((len(rows), 96, 96, 3), np.uint8)
        alignment_rows = []
        started = time.monotonic()
        for index, row in enumerate(rows):
            path = WORKSPACE / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Source hash changed")
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(image)
            count = 0 if faces is None else len(faces)
            alignment_rows.append({"record_id": row["record_id"], "role": row["role"],
                                   "detection_count": count})
            # Strict single-face reference: no identity-driven choice between people.
            if count != 1:
                continue
            face = faces[0]
            x, y, width, height = face[:4]
            side = max(width, height) * 1.15
            cx, cy = x + width / 2, y + height / 2
            left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            right, bottom = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            aligned = encoder.alignCrop(image, face)
            vector = np.asarray(encoder.feature(aligned), np.float32).reshape(-1)
            if vector.shape != (128,) or not np.isfinite(vector).all():
                raise ValueError("Nonfinite/malformed host face reference")
            features[index] = vector / max(float(np.linalg.norm(vector)), 1e-8)
            rgb96[index] = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2RGB)
            gray96[index, :, :, 0] = cv2.cvtColor(rgb96[index], cv2.COLOR_RGB2GRAY)
            boxes[index] = face[:4]
            detected[index] = True
            if (index + 1) % 200 == 0:
                print(json.dumps({"processed": index + 1, "total": len(rows),
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        np.savez_compressed(cache, features=features, detected=detected, boxes=boxes,
                            gray96=gray96, rgb96=rgb96)
        (output / "nonfinal_index.json").write_text(json.dumps({"rows": rows,
            "preprocessing": "YuNet .85 strict single-face; SFace five-landmark alignCrop to112 BGR for reference; separate 1.15 max-side box crop to96 for MCU candidates",
            "alignment_rows": alignment_rows, "final_role_used": False,
            "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}, indent=2) + "\n")
    roles = np.asarray([row["role"] for row in rows])
    labels = np.asarray([int(row["label"][1:]) - 1 if row["known"] else -1 for row in rows])
    results = []
    for size in [1, 5, None]:
        for seed in range(53001, 53011):
            rng = np.random.default_rng(seed)
            gallery = []
            for label in range(5):
                indices = np.flatnonzero((roles == "train") & (labels == label) & detected)
                if not len(indices):
                    raise ValueError("Known class has no valid training reference faces")
                chosen = rng.choice(indices, size=min(size or len(indices), len(indices)), replace=False)
                center = features[chosen].mean(axis=0)
                gallery.append(center / max(float(np.linalg.norm(center)), 1e-8))
            scores = features @ np.stack(gallery).T
            confidence = scores.max(axis=1)
            predicted = scores.argmax(axis=1)
            # Zero calibration false matches; denominator/upper bound reported separately.
            unknown_cal = confidence[(roles == "unknown_calibration") & detected]
            threshold = float(np.nextafter(unknown_cal.max(), np.float32(np.inf)))
            dev_known = roles == "development"
            dev_unknown = roles == "unknown_development"
            accepted = detected & (confidence >= threshold)
            per_person = []
            for label in range(5):
                mask = dev_known & (labels == label)
                per_person.append({"label": "K%02d" % (label + 1), "n": int(mask.sum()),
                    "detection_coverage": float(detected[mask].mean()),
                    "known_correct_acceptance": float(((predicted == label) & accepted)[mask].mean()),
                    "known_coverage": float(accepted[mask].mean()),
                    "known_top1_full_pipeline": float(((predicted == label) & detected)[mask].mean())})
            results.append({"gallery_templates_per_class": size or "all", "seed": seed,
                "root_unit": "gallery sampling; fixed pretrained network, not new gradient training",
                "threshold": threshold, "per_person": per_person,
                "known_correct_acceptance": float(np.mean([v["known_correct_acceptance"] for v in per_person])),
                "known_coverage": float(np.mean([v["known_coverage"] for v in per_person])),
                "known_top1_full_pipeline": float(np.mean([v["known_top1_full_pipeline"] for v in per_person])),
                "unknown_n": int(dev_unknown.sum()),
                "unknown_false_identifications": int((accepted & dev_unknown).sum()),
                "unknown_fpir": float(accepted[dev_unknown].mean()),
                "evidence_tier": "host-only aligned-reference development; not an MCU recognizer"})
    summary = {"status": "executed_reference", "role_counts": dict(Counter(roles)),
        "detection_coverage_by_role": {role: float(detected[roles == role].mean()) for role in set(roles)},
        "template_arms": [{"templates": size or "all", "roots": 10,
            "known_correct_acceptance_mean": float(np.mean([r["known_correct_acceptance"] for r in results
                if r["gallery_templates_per_class"] == (size or "all")])),
            "unknown_fpir_mean": float(np.mean([r["unknown_fpir"] for r in results
                if r["gallery_templates_per_class"] == (size or "all")]))} for size in [1, 5, None]],
        "physical_camera_trials": 0, "future_final_used": False,
        "limits": "Host teacher and box-crop cache only; no board, sensor, identity-exposure or final generalization claim"}
    (output / "ROOT_RESULTS.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (output / "RESULT.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
