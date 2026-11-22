"""Read-only recheck of permitted public source images and archive bytes.

This audit intentionally skips original v3-final image files and the unlabeled
SurvFace verification pool. It checks the already-frozen source manifest in
place; it does not regenerate or modify any roster or role manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


SEALED = "sealed_original_v3_final_prior_audit_only"
UNLABELED = "unlabeled_survface_verification_prior_audit_only"
PERMITTED = "reverified_this_job"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    manifest_path = root / "work/project-finish-v4/data/source_manifest.jsonl"
    archive_receipt_path = root / "work/project-finish-v4/reports/source_archive_rehash.json"
    output_path = root / "work/project-finish-v4/data/source_recheck_report.json"

    counts = Counter()
    dimensions = Counter()
    modes = Counter()
    formats = Counter()
    orientations = Counter()
    errors: list[dict[str, str]] = []
    exact_hashes: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    used_record_ids: set[str] = set()
    for rel in (
        "work/project-finish-v4/data/primary_split_manifest.jsonl",
        "work/project-finish-v4/data/presentation_manifest.jsonl",
        "work/project-finish-v4/data/pretraining_manifest.jsonl",
        "work/project-finish-v4/data/olivetti_sanity_manifest.jsonl",
    ):
        with (root / rel).open(encoding="utf-8") as split_stream:
            for split_line in split_stream:
                used_record_ids.add(str(json.loads(split_line).get("record_id")))

    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row_count += 1
            row = json.loads(line)
            state = row.get("access_state")
            counts["manifest_state:" + str(state)] += 1
            if state in (SEALED, UNLABELED):
                continue
            if state != PERMITTED:
                errors.append({"line": str(line_number), "error": "unexpected_access_state"})
                continue

            dataset = str(row.get("dataset"))
            rel_text = str(row.get("relative_path", ""))
            rel = PurePosixPath(rel_text)
            if rel.is_absolute() or ".." in rel.parts:
                errors.append({"record_id": str(row.get("record_id")), "error": "unsafe_relative_path"})
                continue
            path = root.joinpath(*rel.parts)
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise ValueError("path escapes workspace")
                if not resolved.is_file():
                    raise FileNotFoundError("not a regular file")
                actual_size = resolved.stat().st_size
                expected_size = row.get("byte_size")
                if actual_size != expected_size:
                    errors.append({"record_id": str(row.get("record_id")), "error": "byte_size_mismatch"})
                    continue
                digest = sha256_file(resolved)
                expected_hashes = {row.get("sha256"), row.get("source_sha256")}
                expected_hashes.discard(None)
                if expected_hashes != {digest}:
                    errors.append({"record_id": str(row.get("record_id")), "error": "sha256_mismatch"})
                    continue

                with Image.open(resolved) as probe:
                    actual_format = probe.format
                    actual_mode = probe.mode
                    actual_dimensions = probe.size
                    actual_orientation = int(probe.getexif().get(274, 1))
                    probe.load()
                    if probe.size != actual_dimensions or probe.mode != actual_mode:
                        raise ValueError("metadata changed after decode")

                metadata_ok = (
                    actual_dimensions == (row.get("width"), row.get("height"))
                    and actual_mode == row.get("mode")
                    and actual_format == row.get("format")
                    and actual_orientation == row.get("orientation_tag")
                )
                if not metadata_ok:
                    errors.append({"record_id": str(row.get("record_id")), "error": "image_metadata_mismatch"})
                    continue

                counts["decoded:" + dataset] += 1
                dimensions[f"{actual_dimensions[0]}x{actual_dimensions[1]}"] += 1
                modes[actual_mode] += 1
                formats[str(actual_format)] += 1
                orientations[str(actual_orientation)] += 1
                key = (dataset, digest)
                item = exact_hashes.setdefault(key, {"count": 0, "source_ids": set(), "labels": set(), "record_ids": set()})
                item["count"] += 1
                item["source_ids"].add(str(row.get("source_identity_id")))
                item["labels"].add(str(row.get("identity_id")))
                item["record_ids"].add(str(row.get("record_id")))
            except Exception as exc:  # audit every row and report all failures
                errors.append({"record_id": str(row.get("record_id")), "error": type(exc).__name__ + ": " + str(exc)[:240]})

    duplicate_groups = 0
    duplicate_records = 0
    cross_identity_groups = []
    for (dataset, digest), item in exact_hashes.items():
        if item["count"] > 1:
            duplicate_groups += 1
            duplicate_records += item["count"]
        if len(item["source_ids"]) > 1 or len(item["labels"]) > 1:
            cross_identity_groups.append({
                "dataset": dataset,
                "sha256_prefix": digest[:16],
                "record_count": item["count"],
                "source_identity_count": len(item["source_ids"]),
                "label_count": len(item["labels"]),
                "present_in_study_manifests": bool(item["record_ids"] & used_record_ids),
            })

    archive_receipt = load_json(archive_receipt_path)
    archive_results = []
    for prior in archive_receipt.get("archives", []):
        archive_path = root / "work/dataset-prep/raw" / prior["filename"]
        actual_hash = sha256_file(archive_path)
        actual_bytes = archive_path.stat().st_size
        archive_results.append({
            "filename": prior["filename"],
            "byte_size": actual_bytes,
            "sha256": actual_hash,
            "matches_recorded_receipt": actual_hash == prior["sha256"] and actual_bytes == prior["byte_size"],
            "method": "raw archive bytes only; no members opened by this check",
        })

    status = "pass" if (
        not errors
        and all(row["matches_recorded_receipt"] for row in archive_results)
        and all(not item["present_in_study_manifests"] for item in cross_identity_groups)
    ) else "failed"
    report = {
        "schema": "frozen-public-source-recheck-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_manifest": "work/project-finish-v4/data/source_manifest.jsonl",
        "source_manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": row_count,
        "source_images_reopened_and_decoded": sum(v for k, v in counts.items() if k.startswith("decoded:")),
        "access_state_counts": dict(sorted(counts.items())),
        "skipped_without_opening": {
            "sealed_original_v3_final": counts.get("manifest_state:" + SEALED, 0),
            "unlabeled_verification_pool": counts.get("manifest_state:" + UNLABELED, 0),
        },
        "decoded_dimensions": dict(sorted(dimensions.items())),
        "decoded_modes": dict(sorted(modes.items())),
        "decoded_formats": dict(sorted(formats.items())),
        "orientation_tags": dict(sorted(orientations.items())),
        "exact_duplicate_hash_groups": duplicate_groups,
        "records_in_exact_duplicate_groups": duplicate_records,
        "cross_identity_exact_hash_groups": cross_identity_groups,
        "cross_identity_exact_hash_groups_excluded_from_study_manifests": all(
            not item["present_in_study_manifests"] for item in cross_identity_groups
        ),
        "archive_rechecks": archive_results,
        "errors": errors,
        "limitations": [
            "Only source rows already marked permitted/reverified in the frozen source manifest were opened.",
            "Native resolution of the supplied 96x96 derivatives remains unknown.",
            "Source crops have no face boxes, landmarks, or face-presence annotations, so crop coverage is not measurable here.",
            "Near-duplicate pHash audit evidence remains in cross_identity_phash_audit.json; this source recheck does not recompute pHash candidates.",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "source_images_reopened_and_decoded", "skipped_without_opening",
        "decoded_dimensions", "decoded_modes", "decoded_formats", "orientation_tags",
        "exact_duplicate_hash_groups", "cross_identity_exact_hash_groups",
        "cross_identity_exact_hash_groups_excluded_from_study_manifests", "archive_rechecks", "errors",
    )}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
