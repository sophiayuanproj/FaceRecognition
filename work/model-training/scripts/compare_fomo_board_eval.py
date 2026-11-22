#!/usr/bin/env python3
"""Compare host embeddings with the exact crops used by on-board FOMO pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import MT_ROOT, ROOT, write_json
from convert_eval import Runner


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-log", type=Path, required=True)
    parser.add_argument("--diagnostic-log", type=Path, action="append", default=[],
                        help="Additional threshold-check logs from the same scene manifest")
    parser.add_argument("--manifest", type=Path,
                        default=MT_ROOT / "fomo-board-eval-local/fomo-board-eval-manifest.json")
    parser.add_argument("--scenes", type=Path, default=MT_ROOT / "fomo-board-eval-local")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "FaceRecognition/models/identity_embedding_int8.tflite")
    parser.add_argument("--output", type=Path,
                        default=MT_ROOT / "reports/fomo-board-pipeline-comparison.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = {row["record_id"]: row for row in manifest["images"]}
    board_rows = read_rows(args.board_log)
    detections = {row["record_id"]: row for row in board_rows
                  if row.get("event") == "fomo_board_eval_detection"}
    board_vectors = {row["record_id"]: np.asarray(row["vector"], dtype=np.float64)
                     for row in board_rows if row.get("event") == "fomo_board_eval_embedding"}
    decision_rows = [row for row in board_rows
                     if row.get("event") == "fomo_board_eval_pair_decision"]
    metrics = []
    runner = Runner(args.model, tflite=True)
    for record_id, board_vector in sorted(board_vectors.items()):
        entry = entries[record_id]
        detection = detections.get(record_id)
        if not detection or detection.get("status") != "embedded":
            continue
        x, y, width, height = detection["identity_roi"]
        with Image.open(args.scenes / entry["file"]) as source:
            gray = np.asarray(source.convert("L"), dtype=np.uint8)
        host_vector = runner.run(gray[y:y + height, x:x + width])
        similarity = float(np.dot(host_vector, board_vector) /
                           (np.linalg.norm(host_vector) * np.linalg.norm(board_vector)))
        error = np.abs(host_vector - board_vector)
        metrics.append({"record_id": record_id, "identity_id": entry["identity_id"],
                        "host_board_cosine": similarity,
                        "mean_abs_component_error": float(np.mean(error)),
                        "max_abs_component_error": float(np.max(error))})

    results = [row for row in board_rows if row.get("event") == "fomo_board_eval_complete"]
    diagnostic_runs = []
    for log_path in [args.board_log] + args.diagnostic_log:
        rows = read_rows(log_path)
        started = next((row for row in rows if row.get("event") == "fomo_board_eval_started"), {})
        checks = [row for row in rows if row.get("event") == "fomo_board_eval_detection"]
        completed = next((row for row in rows if row.get("event") == "fomo_board_eval_complete"), {})
        diagnostic_runs.append({
            "log": log_path.name,
            "scene_size": manifest.get("scene_size"),
            "detection_threshold": started.get("detection_threshold"),
            "production_threshold": started.get("detection_threshold") == 0.4,
            "input_scenes": len(checks),
            "detected_scenes": sum(row.get("face_count") == 1 for row in checks),
            "embedded_scenes": completed.get("embedded_images", 0),
        })
    output = {
        "input_scene_count": len(manifest["images"]),
        "detected_scene_count": sum(row.get("face_count") == 1 for row in detections.values()),
        "embedded_scene_count": len(board_vectors),
        "validation_identity_count": len(set(row["identity_id"] for row in entries.values())),
        "board_host_parity": {
            "sample_count": len(metrics),
            "mean_cosine_similarity": float(np.mean([row["host_board_cosine"] for row in metrics])) if metrics else None,
            "minimum_cosine_similarity": float(np.min([row["host_board_cosine"] for row in metrics])) if metrics else None,
            "maximum_mean_absolute_component_error": float(np.max([row["mean_abs_component_error"] for row in metrics])) if metrics else None,
            "maximum_absolute_component_error": float(np.max([row["max_abs_component_error"] for row in metrics])) if metrics else None,
        },
        "latency_us": {
            "detector_inference_mean": float(np.mean([row["detector_inference_us"] for row in detections.values()])) if detections else None,
            "embedding_inference_mean": float(np.mean([row["embedding_inference_us"] for row in detections.values() if row.get("status") == "embedded"])) if board_vectors else None,
            "embedding_inference_p95": float(np.percentile([row["embedding_inference_us"] for row in detections.values() if row.get("status") == "embedded"], 95)) if board_vectors else None,
        },
        "diagnostic_threshold_runs": diagnostic_runs,
        "on_board_template_decisions": {
            "count": len(decision_rows),
            "positive_count": sum(row["verification_pair_type"] == "positive" for row in decision_rows),
            "negative_count": sum(row["verification_pair_type"] == "negative" for row in decision_rows),
            "correct_count": sum(bool(row["decision_correct"]) for row in decision_rows),
            "known_count": sum(row["decision"] == "known" for row in decision_rows),
            "unknown_count": sum(row["decision"] == "unknown" for row in decision_rows),
            "uncertain_count": sum(row["decision"] == "uncertain" for row in decision_rows),
            "scope": "tiny selected pseudonymous validation subset; integration proof only, not an accuracy estimate",
        },
        "result_event": results[-1] if results else None,
        "sample_metrics": metrics,
        "identity_labels_are_pseudonymous": True,
        "temporary_vectors_deleted_after_comparison": True,
    }
    write_json(args.output, output)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
