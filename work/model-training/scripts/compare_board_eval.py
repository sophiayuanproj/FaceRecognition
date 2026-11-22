#!/usr/bin/env python3
"""Compare Nicla embeddings with the host full-INT8 references."""

import argparse
import json
from pathlib import Path

import numpy as np

from common import MT_ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-log", type=Path, required=True,
                        help="JSONL captured from firmware/board_eval_embeddings.py")
    parser.add_argument("--reference", type=Path,
                        default=MT_ROOT / "reports/board-host-reference-embeddings.json")
    parser.add_argument("--calibration", type=Path,
                        default=MT_ROOT / "artifacts/embedding_calibration.json")
    parser.add_argument("--output", type=Path,
                        default=MT_ROOT / "reports/board-host-comparison.json")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    expected = {row["record_id"]: row for row in reference["embeddings"]}
    actual = {}
    load_us = None
    inference_us = []
    pair_decisions = []
    for line in args.board_log.read_text().splitlines():
        try:
            row = json.loads(line.strip())
        except Exception:
            continue
        if row.get("event") == "board_eval_model_loaded":
            load_us = row.get("load_us")
        elif row.get("event") == "board_eval_embedding":
            actual[row["record_id"]] = np.asarray(row["vector"], dtype=np.float64)
            inference_us.append(float(row["inference_us"]))
        elif row.get("event") == "board_eval_pair_decision":
            pair_decisions.append(row)
    common_ids = sorted(set(actual) & set(expected))
    if not common_ids:
        raise ValueError("No matching board outputs found in the supplied log")
    cosine, max_error, mean_error = [], [], []
    for record_id in common_ids:
        exp = np.asarray(expected[record_id]["vector"], dtype=np.float64)
        got = actual[record_id]
        if got.shape != exp.shape:
            raise ValueError("Board/host embedding dimensions differ")
        cosine.append(float(np.dot(exp, got) / (np.linalg.norm(exp) * np.linalg.norm(got))))
        abs_error = np.abs(exp - got)
        max_error.append(float(np.max(abs_error)))
        mean_error.append(float(np.mean(abs_error)))
    groups = {}
    for record_id in common_ids:
        groups.setdefault(expected[record_id]["identity_id"], []).append(record_id)
    positives, negatives = [], []
    identity_ids = sorted(groups)
    for i, left_id in enumerate(identity_ids):
        for a_index, left_record in enumerate(groups[left_id]):
            for right_record in groups[left_id][a_index + 1:]:
                positives.append(float(np.dot(actual[left_record], actual[right_record])))
        for right_id in identity_ids[i + 1:]:
            for left_record in groups[left_id]:
                for right_record in groups[right_id]:
                    negatives.append(float(np.dot(actual[left_record], actual[right_record])))
    threshold = float(json.loads(args.calibration.read_text())["threshold"])
    result = {
        "board_sample_count": len(common_ids),
        "validation_identity_count": len(identity_ids),
        "host_vs_board_cosine_similarity": {
            "mean": float(np.mean(cosine)), "minimum": float(np.min(cosine)),
            "maximum": float(np.max(cosine)),
        },
        "host_vs_board_absolute_embedding_error": {
            "mean": float(np.mean(mean_error)), "maximum": float(np.max(max_error)),
        },
        "board_inference_us": {
            "mean": float(np.mean(inference_us)),
            "p50": float(np.percentile(inference_us, 50)),
            "p95": float(np.percentile(inference_us, 95)),
        },
        "board_model_load_us": load_us,
        "validation_calibrated_threshold": threshold,
        "small_set_same_identity_pair_count": len(positives),
        "small_set_different_identity_pair_count": len(negatives),
        "small_set_same_identity_similarity_mean": float(np.mean(positives)) if positives else None,
        "small_set_different_identity_similarity_mean": float(np.mean(negatives)) if negatives else None,
        "small_set_positive_pairs_above_threshold": sum(x >= threshold for x in positives),
        "small_set_negative_pairs_above_threshold": sum(x >= threshold for x in negatives),
        "identity_labels_are_pseudonymous": True,
        "claim_scope": "board-versus-host numerical parity on a tiny crop subset; not an accuracy estimate",
    }
    if pair_decisions:
        result["on_board_template_decisions"] = {
            "pair_count": len(pair_decisions),
            "positive_pair_count": sum(row["verification_pair_type"] == "positive"
                                        for row in pair_decisions),
            "negative_pair_count": sum(row["verification_pair_type"] == "negative"
                                        for row in pair_decisions),
            "known_count": sum(row["decision"] == "known" for row in pair_decisions),
            "unknown_count": sum(row["decision"] == "unknown" for row in pair_decisions),
            "uncertain_count": sum(row["decision"] == "uncertain" for row in pair_decisions),
            "correct_count": sum(bool(row["decision_correct"]) for row in pair_decisions),
            "decision_rows": pair_decisions,
            "claim_scope": "small selected validation subset using temporary RAM-only templates; not an accuracy estimate",
        }
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
