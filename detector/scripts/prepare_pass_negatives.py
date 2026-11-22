"""Select source-licensed PASS no-human scenes and download only a small subset."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi
from PIL import Image, ImageOps

from common import sha256_file


SEED = 20261100
TARGETS = {"train": 100, "development": 100, "final": 50}
SOURCE_ROOT = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def source_url(image_hash: str) -> str:
    image_hash = image_hash.lower().strip()
    # Keep the key exactly as serialized in PASS metadata. Although the field
    # is named "hash", it is an asset key, and left-padding changes its S3 URL
    # (padded keys return 404 while the original unpadded key resolves).
    if not 23 <= len(image_hash) <= 32 or any(ch not in "0123456789abcdef" for ch in image_hash):
        raise ValueError(f"Bad PASS image hash {image_hash!r}")
    return f"{SOURCE_ROOT}/{image_hash[:3]}/{image_hash[3:6]}/{image_hash}.jpg"


def read_rows(metadata: Path) -> list[dict]:
    rows = []
    with metadata.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_hash = (row.get("hash", "") or "").strip().lower()
            image_hash = raw_hash
            if row.get("licensename") != "Attribution License" or not row.get("unickname"):
                continue
            rows.append({
                "source_dataset": "PASS v3 (Zenodo 6615455)",
                "pass_metadata_index": row.get("", ""),
                "author_attribution": row["unickname"],
                "date_taken": row.get("datetaken") or None,
                "source_item_license": "CC BY 2.0 (Flickr Attribution License)",
                "pass_asset_id": image_hash,
                "pass_metadata_hash_raw": raw_hash,
                "pass_metadata_hash_width": len(raw_hash),
                "pass_metadata_hash_left_zero_padded": False,
                "source_url": source_url(image_hash),
                "latitude": row.get("latitude") or None,
                "longitude": row.get("longitude") or None,
            })
    return rows


def make_plan(rows: list[dict]) -> list[dict]:
    by_author: dict[str, list[dict]] = {}
    for row in rows:
        by_author.setdefault(row["author_attribution"], []).append(row)
    authors = sorted(by_author)
    random.Random(SEED).shuffle(authors)
    # Freeze role membership by source uploader, not image, so the same account
    # cannot leak nearly repeated photos across train/development/final.
    n_train = int(len(authors) * 0.70)
    n_dev = int(len(authors) * 0.15)
    roles = {
        "train": authors[:n_train],
        "development": authors[n_train:n_train + n_dev],
        "final": authors[n_train + n_dev:],
    }
    selected = []
    for role, owner_ids in roles.items():
        candidates = []
        for author in owner_ids:
            candidates.extend(by_author[author])
        rng = random.Random(SEED + {"train": 1, "development": 2, "final": 3}[role])
        rng.shuffle(candidates)
        seen_hashes, seen_authors = set(), set()
        for row in candidates:
            if row["pass_asset_id"] in seen_hashes or row["author_attribution"] in seen_authors:
                continue
            chosen = {**row, "role": role, "group_id": row["author_attribution"],
                      "download_status": "held_out_not_downloaded" if role == "final" else "pending"}
            selected.append(chosen)
            seen_hashes.add(row["pass_asset_id"])
            seen_authors.add(row["author_attribution"])
            if len(seen_hashes) == TARGETS[role]:
                break
        if sum(r["role"] == role for r in selected) != TARGETS[role]:
            raise RuntimeError(f"Could not select {TARGETS[role]} distinct {role} authors")
    return selected


def download_one(row: dict, image_dir: Path, attempts: int = 4) -> dict:
    image_hash = row["pass_asset_id"]
    target = image_dir / f"{image_hash}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(attempts):
        try:
            if not target.exists():
                request = urllib.request.Request(row["source_url"], headers={"User-Agent": "JobC-local-research/1.0"})
                with urllib.request.urlopen(request, timeout=45, context=TLS_CONTEXT) as response:
                    body = response.read()
                tmp = target.with_suffix(".download")
                tmp.write_bytes(body)
                tmp.replace(target)
            with Image.open(target) as im:
                orientation = im.getexif().get(274, 1)
                dimensions = im.size
                im.verify()
            return {**row, "download_status": "pass", "local_image_path": str(target),
                    "local_image_sha256": sha256_file(target), "decoded_dimensions": list(dimensions),
                    "source_url_asset_id_matches": target.stem == image_hash,
                    "exif_orientation": orientation, "decode_error": None}
        except Exception as exc:  # retries are bounded and errors stay in the manifest
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404, 410):
                break
            time.sleep(min(2 ** attempt, 8))
    return {**row, "download_status": "error", "local_image_path": None,
            "local_image_sha256": None, "decoded_dimensions": None,
            "exif_orientation": None, "source_url_asset_id_matches": False,
            "decode_error": last_error}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--only-roles", nargs="+", default=["train", "development"],
                    choices=["train", "development", "final"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows = read_rows(args.metadata)
    plan = make_plan(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plan_path = args.outdir / "pass_negative_plan.jsonl"
    with plan_path.open("w") as stream:
        for row in plan:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    selected = [r for r in plan if r["role"] in args.only_roles]
    downloaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, args.image_dir) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            downloaded.append(future.result())
    result_by_hash = {r["pass_asset_id"]: r for r in downloaded}
    out_path = args.outdir / "pass_negative_manifest.jsonl"
    existing_by_hash = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            existing_by_hash[old["pass_asset_id"]] = old
    merged = []
    with out_path.open("w") as stream:
        for row in plan:
            resolved = result_by_hash.get(row["pass_asset_id"])
            if resolved is None:
                old = existing_by_hash.get(row["pass_asset_id"])
                resolved = old if old and old.get("download_status") == "pass" else row
            merged.append(resolved)
            stream.write(json.dumps(resolved, sort_keys=True) + "\n")
    failures = [r for r in downloaded if r["download_status"] != "pass"]
    merged_downloaded = [row for row in merged if row.get("download_status") == "pass"]
    merged_roles = sorted({row["role"] for row in merged_downloaded})
    report = {"schema": "job-c-pass-negative-source-1", "dataset_record": "https://zenodo.org/records/6615455",
              "dataset_record_license": "CC BY 4.0", "image_source_url_template": SOURCE_ROOT + "/{asset_id[0:3]}/{asset_id[3:6]}/{asset_id}.jpg",
              "asset_id_semantics": "PASS metadata hash column preserved exactly as the URL basename; not treated as an image-byte checksum; shorter-than-32 values are not left-padded",
              "download_integrity": "per-image SHA-256 recorded after download; metadata archive MD5 verified against Zenodo record",
              "item_license_filter": "Attribution License only (CC BY 2.0); source author retained for attribution",
              "selection_seed": SEED, "targets_per_role": TARGETS,
              "role_grouping": "Flickr uploader; one image per uploader; uploaders disjoint across roles",
              "downloaded_roles": merged_roles, "downloaded_count": len(merged_downloaded),
              "download_failures": len(failures), "plan_path": str(plan_path), "manifest_path": str(out_path)}
    (args.outdir / "pass_negative_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failures:
        (args.outdir / "pass_negative_download_errors.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(json.dumps({"downloaded_this_run": len(downloaded), "downloaded_total": len(merged_downloaded),
                      "failures": len(failures),
                      "manifest": str(out_path), "roles": args.only_roles}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
