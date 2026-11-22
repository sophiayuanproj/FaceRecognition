"""Compare detector scene source hashes with B's canonical identity material."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import dump_json, sha256_file


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider-manifest", type=Path, required=True)
    ap.add_argument("--pass-manifest", type=Path, required=True)
    ap.add_argument("--identity-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    detector = jsonl(args.wider_manifest)
    detector_hashes = defaultdict(list)
    for row in detector:
        detector_hashes[row.get("image_sha256", "")].append({"dataset": "WIDER FACE",
            "role": row.get("role"), "event_group": row.get("event_group"),
            "path": row.get("image_relative_path")})
    pass_rows = jsonl(args.pass_manifest)
    pass_hashes = defaultdict(list)
    pass_downloads_by_role = Counter()
    pass_planned_by_role = Counter()
    for row in pass_rows:
        pass_planned_by_role[row.get("role", "unknown")] += 1
        if row.get("download_status") == "pass":
            digest = row.get("local_image_sha256") or sha256_file(row["local_image_path"])
            pass_hashes[digest].append({"dataset": "PASS v3", "role": row.get("role"),
                                        "author": row.get("author_attribution"),
                                        "asset_id": row.get("pass_asset_id")})
            pass_downloads_by_role[row.get("role", "unknown")] += 1
    identity_manifest_names = ("source_manifest.jsonl", "primary_split_manifest.jsonl",
                               "presentation_manifest.jsonl", "pretraining_manifest.jsonl")
    identity = defaultdict(list)
    identity_records = Counter()
    identity_datasets = Counter()
    for name in identity_manifest_names:
        for row in jsonl(args.identity_data / name):
            digest = row.get("sha256") or row.get("source_sha256") or row.get("verified_sha256")
            if digest:
                identity[digest].append({"manifest": name, "dataset": row.get("dataset"),
                                         "identity_id": row.get("identity_id"),
                                         "relative_path": row.get("relative_path"),
                                         "role": row.get("presentation_role") or row.get("split") or row.get("identity_partition")})
            identity_records[name] += 1
            identity_datasets[row.get("dataset", "unknown")] += 1
    exact_wider = []
    for digest, rows in detector_hashes.items():
        if digest in identity:
            exact_wider.append({"sha256": digest, "detector": rows, "identity": identity[digest]})
    exact_pass = []
    for digest, rows in pass_hashes.items():
        if digest in identity:
            exact_pass.append({"sha256": digest, "detector": rows, "identity": identity[digest]})
    report = {
        "schema": "job-c-dataset-overlap-audit-1",
        "detector_source_roles": {"WIDER": "train/development/final event groups from B source; final retained unopened for selection",
                                  "PASS": {"planned_rows_by_role": dict(pass_planned_by_role),
                                           "downloaded_rows_by_role": dict(pass_downloads_by_role),
                                           "final_rows_are_uploader_held_out": True}},
        "identity_source": str(args.identity_data),
        "identity_dataset_names": dict(identity_datasets),
        "identity_manifest_record_counts": dict(identity_records),
        "wider_scene_hash_count": len(detector_hashes), "pass_download_hash_count": len(pass_hashes),
        "exact_wider_identity_hash_overlaps": exact_wider,
        "exact_pass_identity_hash_overlaps": exact_pass,
        "exact_wider_identity_overlap_count": len(exact_wider),
        "exact_pass_identity_overlap_count": len(exact_pass),
        "audit_limit": "Exact decoded-source byte SHA-256 only; identity crops can be transformed crops of a larger scene, so dataset source families and role manifests were separately reviewed. No pHash scene-to-crop comparison was performed.",
    }
    dump_json(args.out, report)
    print(json.dumps({"wider_identity_exact": len(exact_wider), "pass_identity_exact": len(exact_pass),
                      "identity_datasets": dict(identity_datasets), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
