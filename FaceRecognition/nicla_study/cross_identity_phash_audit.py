"""Exhaustive threshold search for cross-identity SurvFace pHash candidates.

Only source records whose image bytes were explicitly reverified by Job B are
read. Original v3-final images and the unlabeled verification pool remain
excluded. Seven disjoint bit bands provide a lossless candidate index for a
64-bit pHash at Hamming distance <= 6: any such pair must match in at least
one band. Candidate pairs are reported for local review and are never removed
automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .fs import atomic_write_json, sha256_file

HASH_SIZE = 8
RESIZE = 32
MAX_HAMMING = 6
BLOCK_WIDTHS = (9, 9, 9, 9, 9, 9, 10)


def dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    u = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos(np.pi * (2 * x + 1) * u / (2 * size))
    matrix[0, :] *= np.sqrt(1 / size)
    matrix[1:, :] *= np.sqrt(2 / size)
    return matrix


DCT = dct_matrix(RESIZE)


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as source:
        pixels = np.asarray(
            source.convert("L").resize((RESIZE, RESIZE), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    low = (DCT @ pixels @ DCT.T)[:HASH_SIZE, :HASH_SIZE]
    values = low.flatten()
    threshold = float(np.median(values[1:]))
    result = 0
    for bit in values > threshold:
        result = (result << 1) | int(bit)
    return result


def read_allowed_rows(workspace: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("dataset") != "survface_96":
                continue
            if row.get("access_state") != "reverified_this_job":
                continue
            identity = row.get("source_identity_id") or row.get("identity_id")
            if not identity or not row.get("verified_sha256"):
                raise ValueError("reverified labeled source row lacks identity or verified SHA-256")
            image_path = workspace / row["relative_path"]
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))
            rows.append({
                "record_id": row["record_id"],
                "identity_id": str(identity),
                "relative_path": row["relative_path"],
                "verified_sha256": row["verified_sha256"],
                "path": image_path,
            })
    if not rows:
        raise ValueError("no permitted labeled SurvFace records found")
    return rows


def band_groups(hashes: np.ndarray) -> list[dict[int, list[int]]]:
    groups = []
    offset = 0
    for width in BLOCK_WIDTHS:
        mask = (1 << width) - 1
        current: dict[int, list[int]] = defaultdict(list)
        for index, value in enumerate(hashes):
            current[(int(value) >> offset) & mask].append(index)
        groups.append(dict(current))
        offset += width
    if offset != 64:
        raise AssertionError("band widths must partition all 64 hash bits")
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    study = workspace / "work/project-finish-v4"
    data = study / "data"
    manifest_path = data / "source_manifest.jsonl"
    rows = read_allowed_rows(workspace, manifest_path)
    manifest_sha = sha256_file(manifest_path)

    hash_path = data / "permitted_survface_phashes.jsonl"
    temp_hash_path = hash_path.with_suffix(hash_path.suffix + ".partial")
    hashes = np.empty(len(rows), dtype=np.uint64)
    sha_to_indices: dict[str, list[int]] = defaultdict(list)
    with temp_hash_path.open("w", encoding="utf-8", newline="\n") as out:
        for index, row in enumerate(rows):
            actual_sha = sha256_file(row["path"])
            if actual_sha != row["verified_sha256"]:
                raise ValueError("source SHA-256 changed for " + row["record_id"])
            value = perceptual_hash(row["path"])
            hashes[index] = value
            sha_to_indices[actual_sha].append(index)
            out.write(json.dumps({
                "record_id": row["record_id"],
                "source_identity_id": row["identity_id"],
                "relative_path": row["relative_path"],
                "sha256": actual_sha,
                "phash64_hex": "%016x" % int(value),
            }, sort_keys=True) + "\n")
            if (index + 1) % 20000 == 0:
                print("hashed %d/%d permitted images" % (index + 1, len(rows)), flush=True)
    temp_hash_path.replace(hash_path)

    ids = np.asarray([row["identity_id"] for row in rows], dtype=object)
    band_indexes = band_groups(hashes)
    candidate_distances: dict[tuple[int, int], int] = {}
    bucket_comparisons = 0
    cross_identity_bucket_comparisons = 0
    largest_bucket = 0
    for band_index, groups in enumerate(band_indexes):
        print("searching pHash band %d/%d (%d buckets)" % (band_index + 1, len(BLOCK_WIDTHS), len(groups)), flush=True)
        for members in groups.values():
            size = len(members)
            largest_bucket = max(largest_bucket, size)
            if size < 2:
                continue
            index_array = np.asarray(members, dtype=np.int64)
            if size <= 4096:
                left_pos, right_pos = np.triu_indices(size, k=1)
                left = index_array[left_pos]
                right = index_array[right_pos]
                del left_pos, right_pos
                cross_mask = ids[left] != ids[right]
                left = left[cross_mask]
                right = right[cross_mask]
                del cross_mask
                cross_identity_bucket_comparisons += len(left)
                bucket_comparisons += size * (size - 1) // 2
                if len(left):
                    distances = np.bitwise_count(np.bitwise_xor(hashes[left], hashes[right]))
                    hits = np.flatnonzero(distances <= MAX_HAMMING)
                    for hit in hits:
                        pair = (int(left[hit]), int(right[hit]))
                        candidate_distances[pair] = int(distances[hit])
            else:
                for left_pos in range(size - 1):
                    left_id = int(index_array[left_pos])
                    right = index_array[left_pos + 1:]
                    bucket_comparisons += len(right)
                    cross_mask = ids[right] != ids[left_id]
                    right = right[cross_mask]
                    cross_identity_bucket_comparisons += len(right)
                    if len(right):
                        distances = np.bitwise_count(np.bitwise_xor(hashes[right], hashes[left_id]))
                        hits = np.flatnonzero(distances <= MAX_HAMMING)
                        for hit in hits:
                            pair = (left_id, int(right[hit]))
                            candidate_distances[pair] = int(distances[hit])

    identity_counts = Counter(row["identity_id"] for row in rows)
    all_pairs = len(rows) * (len(rows) - 1) // 2
    same_identity_pairs = sum(count * (count - 1) // 2 for count in identity_counts.values())
    exact_sha_pairs = []
    for indices in sha_to_indices.values():
        if len(indices) > 1:
            for left_pos in range(len(indices) - 1):
                for right_pos in range(left_pos + 1, len(indices)):
                    left, right = indices[left_pos], indices[right_pos]
                    if rows[left]["identity_id"] != rows[right]["identity_id"]:
                        exact_sha_pairs.append((left, right))

    known_pairs = {}
    prior_path = workspace / "work/dataset-prep/reports/perceptual-duplicates.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    row_index = {row["record_id"]: index for index, row in enumerate(rows)}
    for pair in prior["datasets"]["survface_96"]["candidate_pairs"]:
        left_id, right_id = pair["left_record_id"], pair["right_record_id"]
        if left_id in row_index and right_id in row_index and pair.get("left_identity_id") == pair.get("right_identity_id"):
            known_pairs[(left_id, right_id)] = int(pair["hamming_distance"])
    checked_prior = 0
    prior_mismatches = []
    for (left_id, right_id), expected_distance in known_pairs.items():
        left, right = row_index[left_id], row_index[right_id]
        actual_distance = int(np.bitwise_count(np.uint64(int(hashes[left]) ^ int(hashes[right]))))
        checked_prior += 1
        if actual_distance != expected_distance:
            prior_mismatches.append({"left_record_id": left_id, "right_record_id": right_id,
                                     "expected": expected_distance, "actual": actual_distance})

    role_by_record = {}
    for filename, role_key in [("primary_split_manifest.jsonl", "role"), ("presentation_manifest.jsonl", "presentation_role")]:
        with (data / filename).open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                role_by_record[row["record_id"]] = row.get(role_key)

    candidate_rows = []
    cross_role_count = 0
    distance_counts = Counter()
    for (left, right), distance in sorted(candidate_distances.items(), key=lambda item: (item[1], item[0])):
        left_row, right_row = rows[left], rows[right]
        left_role = role_by_record.get(left_row["record_id"])
        right_role = role_by_record.get(right_row["record_id"])
        if left_role and right_role and left_role != right_role:
            cross_role_count += 1
        distance_counts[str(distance)] += 1
        candidate_rows.append({
            "left_record_id": left_row["record_id"], "right_record_id": right_row["record_id"],
            "left_identity_id": left_row["identity_id"], "right_identity_id": right_row["identity_id"],
            "left_role": left_role, "right_role": right_role, "hamming_distance": distance,
            "left_path": left_row["relative_path"], "right_path": right_row["relative_path"],
            "exact_sha256_match": left_row["verified_sha256"] == right_row["verified_sha256"],
        })
    pair_path = data / "cross_identity_phash_candidates.jsonl"
    with pair_path.open("w", encoding="utf-8", newline="\n") as out:
        for row in candidate_rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema": "cross-identity-phash-audit-1",
        "status": "complete_exhaustive_threshold_search" if not prior_mismatches else "failed_prior_hash_crosscheck",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "source_manifest_sha256": manifest_sha,
        "images_hashed_and_source_sha_verified": len(rows),
        "identity_count": len(identity_counts),
        "sealed_original_v3_final_images_opened": 0,
        "unlabeled_verification_images_opened": 0,
        "hash_method": "same 64-bit 8x8 low-frequency 2D DCT pHash as prior dataset audit, over 32x32 grayscale Lanczos-resized source images",
        "threshold_hamming_bits_inclusive": MAX_HAMMING,
        "search_method": "Seven disjoint bit bands of widths 9,9,9,9,9,9,10. By the pigeonhole principle, every 64-bit pair with Hamming distance <=6 must match in at least one band; all cross-identity pairs sharing a band are checked exactly.",
        "possible_cross_identity_pairs": all_pairs - same_identity_pairs,
        "cross_identity_bucket_comparisons_including_repeated_band_candidates": cross_identity_bucket_comparisons,
        "all_bucket_pair_comparisons_including_same_identity": bucket_comparisons,
        "largest_hash_band_bucket": largest_bucket,
        "cross_identity_candidate_pair_count": len(candidate_rows),
        "candidate_pair_distance_counts": dict(sorted(distance_counts.items())),
        "cross_role_candidate_pair_count": cross_role_count,
        "exact_cross_identity_sha256_pairs": len(exact_sha_pairs),
        "prior_within_identity_candidates_rechecked": checked_prior,
        "prior_candidate_distance_mismatches": prior_mismatches,
        "candidate_pairs_file": "work/project-finish-v4/data/cross_identity_phash_candidates.jsonl",
        "hashes_file": "work/project-finish-v4/data/permitted_survface_phashes.jsonl",
        "automatic_exclusions": 0,
        "interpretation": "Perceptual-hash matches are heuristic candidates for review; this audit does not change or remove any study records.",
    }
    atomic_write_json(data / "cross_identity_phash_audit.json", report)
    print(json.dumps({k: report[k] for k in ["status", "images_hashed_and_source_sha_verified", "possible_cross_identity_pairs", "cross_identity_bucket_comparisons_including_repeated_band_candidates", "cross_identity_candidate_pair_count", "cross_role_candidate_pair_count", "exact_cross_identity_sha256_pairs", "prior_within_identity_candidates_rechecked", "prior_candidate_distance_mismatches"]}, indent=2))
    return 0 if report["status"] == "complete_exhaustive_threshold_search" else 2


if __name__ == "__main__":
    raise SystemExit(main())
