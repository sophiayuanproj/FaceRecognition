#!/usr/bin/env python3
"""Flag near-duplicate images with a local grayscale perceptual hash.

This is a lightweight image-similarity screen, not face recognition. It reads
only images named in the local manifest and writes pseudonymous record IDs.
Requires Pillow and NumPy; the project's existing .venv-nicla provides both.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

PREP = Path(__file__).resolve().parents[1]
MANIFEST = PREP / "reports" / "manifest.jsonl"
OUTPUT = PREP / "reports" / "perceptual-duplicates.json"
SUMMARY = PREP / "reports" / "dataset-summary.json"
PAIR_MANIFEST = PREP / "reports" / "survface-pairs.jsonl"
HASH_SIZE = 8
RESIZE = 32
NEAR_DUPLICATE_MAX_HAMMING = 6
MAX_EXHAUSTIVE_IMAGES = 5_000


def dct_matrix(n: int) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    u = np.arange(n, dtype=np.float64)[:, None]
    matrix = np.cos(np.pi * (2 * x + 1) * u / (2 * n))
    matrix[0, :] *= np.sqrt(1 / n)
    matrix[1:, :] *= np.sqrt(2 / n)
    return matrix


DCT = dct_matrix(RESIZE)


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("L").resize((RESIZE, RESIZE), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (DCT @ pixels @ DCT.T)[:HASH_SIZE, :HASH_SIZE]
    values = low.flatten()
    threshold = float(np.median(values[1:]))
    result = 0
    for bit in values > threshold:
        result = (result << 1) | int(bit)
    return result


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def main() -> int:
    records: dict[str, list[dict]] = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            path = PREP / row["relative_local_path"]
            if not path.is_file():
                raise FileNotFoundError(f"manifest image missing: {row['record_id']}")
            records.setdefault(row["dataset"], []).append({
                "record_id": row["record_id"],
                "identity_id": row.get("identity_id"),
                "path": path,
            })

    datasets = {}
    for dataset, items in sorted(records.items()):
        pairs = []
        distances = Counter()
        minimum = 64
        comparisons_performed = 0
        same_identity = cross_identity = unlabeled_pairs = 0
        if len(items) <= MAX_EXHAUSTIVE_IMAGES:
            hashed_items = [{**item, "hash": perceptual_hash(item["path"])} for item in items]
            for left, right in combinations(hashed_items, 2):
                distance = hamming(left["hash"], right["hash"])
                comparisons_performed += 1
                minimum = min(minimum, distance)
                if distance <= NEAR_DUPLICATE_MAX_HAMMING:
                    pairs.append({
                        "left_record_id": left["record_id"],
                        "right_record_id": right["record_id"],
                        "left_identity_id": left["identity_id"],
                        "right_identity_id": right["identity_id"],
                        "hamming_distance": distance,
                        "comparison_scope": "exhaustive_within_dataset",
                    })
        elif dataset == "survface_96":
            hashes = {}
            for index, item in enumerate(items, 1):
                hashes[item["record_id"]] = perceptual_hash(item["path"])
                if index % 50_000 == 0:
                    print(f"SurvFace pHash: hashed {index:,}/{len(items):,} images", file=sys.stderr, flush=True)
            identity_groups: dict[str, list[dict]] = {}
            for item in items:
                if item["identity_id"] is not None:
                    identity_groups.setdefault(item["identity_id"], []).append(item)
            for group in identity_groups.values():
                for left, right in combinations(group, 2):
                    distance = hamming(hashes[left["record_id"]], hashes[right["record_id"]])
                    comparisons_performed += 1
                    minimum = min(minimum, distance)
                    if distance <= NEAR_DUPLICATE_MAX_HAMMING:
                        pairs.append({
                            "left_record_id": left["record_id"],
                            "right_record_id": right["record_id"],
                            "left_identity_id": left["identity_id"],
                            "right_identity_id": right["identity_id"],
                            "hamming_distance": distance,
                            "comparison_scope": "within_folder_labeled_training_identity",
                        })
                        same_identity += 1
            if PAIR_MANIFEST.exists():
                for line in PAIR_MANIFEST.read_text(encoding="utf-8").splitlines():
                    pair = json.loads(line)
                    left_id, right_id = pair["left_record_id"], pair["right_record_id"]
                    if left_id not in hashes or right_id not in hashes:
                        continue
                    distance = hamming(hashes[left_id], hashes[right_id])
                    comparisons_performed += 1
                    minimum = min(minimum, distance)
                    if distance <= NEAR_DUPLICATE_MAX_HAMMING:
                        pairs.append({
                            "left_record_id": left_id,
                            "right_record_id": right_id,
                            "left_identity_id": None,
                            "right_identity_id": None,
                            "hamming_distance": distance,
                            "comparison_scope": "bundled_verification_pair",
                            "verification_pair_type": pair["pair_type"],
                        })
                        unlabeled_pairs += 1
            total_possible = len(items) * (len(items) - 1) // 2
            ordered = sorted(pairs, key=lambda p: (p["hamming_distance"], p["left_record_id"], p["right_record_id"]))
            datasets[dataset] = {
                "images_hashed": len(hashes),
                "hash_method": "8x8 low-frequency 2D DCT perceptual hash over 32x32 grayscale Lanczos-resized images",
                "threshold_hamming_bits_inclusive": NEAR_DUPLICATE_MAX_HAMMING,
                "candidate_pair_count": len(pairs),
                "candidate_pair_distance_counts": dict(sorted(Counter(p["hamming_distance"] for p in pairs).items())),
                "same_identity_candidate_pair_count": same_identity,
                "cross_identity_candidate_pair_count": None,
                "unlabeled_identity_candidate_pair_count": unlabeled_pairs,
                "minimum_compared_pair_hamming_distance": minimum if comparisons_performed else None,
                "pairwise_comparisons_performed": comparisons_performed,
                "all_other_pairwise_comparisons_skipped": total_possible - comparisons_performed,
                "candidate_pairs": ordered,
                "status": "completed_scoped_screen",
                "interpretation": "all images were hashed; training images were compared within folder-labeled identity only, and test images were compared along the bundled positive/negative pair annotations. Cross-identity all-pairs comparison was skipped; exact checksum duplicates were checked separately. Candidates are heuristic and must be reviewed locally.",
            }
            continue
        else:
            possible_pairs = len(items) * (len(items) - 1) // 2
            datasets[dataset] = {
                "images_hashed": 0,
                "hash_method": "not run",
                "threshold_hamming_bits_inclusive": NEAR_DUPLICATE_MAX_HAMMING,
                "candidate_pair_count": None,
                "pairwise_comparisons_avoided": possible_pairs,
                "same_identity_candidate_pair_count": None,
                "cross_identity_candidate_pair_count": None,
                "unlabeled_identity_candidate_pair_count": None,
                "minimum_pairwise_hamming_distance": None,
                "candidate_pairs": [],
                "status": "skipped_for_scale",
                "interpretation": f"all-pairs perceptual-hash screening skipped because it would require {possible_pairs:,} comparisons; exact checksum duplicates were still checked; no embeddings were used",
            }
            continue

        ordered = sorted(pairs, key=lambda p: (p["hamming_distance"], p["left_record_id"], p["right_record_id"]))
        same_identity = sum(
            p["left_identity_id"] is not None and p["left_identity_id"] == p["right_identity_id"]
            for p in pairs
        )
        cross_identity = sum(
            p["left_identity_id"] is not None
            and p["right_identity_id"] is not None
            and p["left_identity_id"] != p["right_identity_id"]
            for p in pairs
        )
        datasets[dataset] = {
            "images_hashed": len(items),
            "hash_method": "8x8 low-frequency 2D DCT perceptual hash over 32x32 grayscale Lanczos-resized images",
            "threshold_hamming_bits_inclusive": NEAR_DUPLICATE_MAX_HAMMING,
            "candidate_pair_count": len(pairs),
            "candidate_pair_distance_counts": dict(sorted(Counter(p["hamming_distance"] for p in pairs).items())),
            "same_identity_candidate_pair_count": same_identity,
            "cross_identity_candidate_pair_count": cross_identity,
            "unlabeled_identity_candidate_pair_count": len(pairs) - same_identity - cross_identity,
            "minimum_pairwise_hamming_distance": minimum if comparisons_performed else None,
            "pairwise_comparisons_performed": comparisons_performed,
            "candidate_pairs": ordered,
            "status": "completed",
            "interpretation": "heuristic candidates only; review locally before deduplicating; no embeddings were generated",
        }

    result = {"datasets": datasets, "comparison_scope": "within each dataset only"}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUTPUT.chmod(0o600)

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for dataset, info in datasets.items():
        if info["status"] in {"completed", "completed_scoped_screen"}:
            summary["datasets"][dataset]["perceptual_duplicate_check"] = (
                f"{info['interpretation']} Candidate count: {info['candidate_pair_count']} at Hamming <= "
                f"{NEAR_DUPLICATE_MAX_HAMMING}; see reports/perceptual-duplicates.json"
            )
        else:
            summary["datasets"][dataset]["perceptual_duplicate_check"] = info["interpretation"]
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY.chmod(0o600)
    print(json.dumps({name: {"images_hashed": info["images_hashed"], "candidate_pair_count": info["candidate_pair_count"], "pairwise_comparisons_performed": info.get("pairwise_comparisons_performed", 0), "pairwise_comparisons_avoided": info.get("pairwise_comparisons_avoided", info.get("all_other_pairwise_comparisons_skipped", 0)), "status": info["status"], "minimum_pairwise_hamming_distance": info.get("minimum_pairwise_hamming_distance", info.get("minimum_compared_pair_hamming_distance"))} for name, info in datasets.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
