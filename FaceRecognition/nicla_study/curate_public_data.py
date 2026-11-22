"""Build the frozen local public-image benchmark without touching v3 final files.

Run with the existing image-audit environment (Pillow and NumPy) from the
workspace root. The script verifies source images assigned to v3 train,
development and calibration identities, but never opens images assigned to the
sealed v3 final identities or the unlabeled SurvFace pair-test folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
from typing import Any

from .fs import atomic_write_json, sha256_file

QUALITY_VERSION = "gray-crop-quality-v1"
QUALITY_RULE = {
    "brightness_mean_u8": [5.0, 250.0],
    "minimum_gray_std_u8": 2.0,
    "minimum_laplacian_variance": 0.10,
    "score": "0.30*brightness_midrange + 0.30*contrast_clipped_at_64 + 0.40*log1p(laplacian_variance)/log1p(128), each component clipped to [0,1]",
    "use": "descriptive and train-only curation; never used to exclude development, calibration, evaluation, or presentation images",
}
KNOWN_BUDGET = {"train": 60, "stopping": 10, "threshold_calibration": 10,
                "development": 10, "final": 20, "presentation": 20}
KNOWN_MODEL_GROUP_BUDGET = KNOWN_BUDGET["train"] + KNOWN_BUDGET["stopping"] + KNOWN_BUDGET["threshold_calibration"] + KNOWN_BUDGET["development"] + KNOWN_BUDGET["final"]
KNOWN_TOTAL_GROUP_BUDGET = KNOWN_MODEL_GROUP_BUDGET + KNOWN_BUDGET["presentation"]
UNKNOWN_IDENTITIES_PER_ROLE = 50
UNKNOWN_IMAGES_PER_IDENTITY = 6
UNKNOWN_ROLES = ["unknown_train", "unknown_calibration", "unknown_development",
                 "unknown_final", "unknown_presentation"]
CLASS_ORDER = ["K01", "K02", "K03", "K04", "K05", "OTHER"]
CAMERA_RE = re.compile(r"_cam([0-9]+)_", re.IGNORECASE)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSONL at %s:%d" % (path, line_number)) from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _source_id(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if "training_set" in parts:
        index = parts.index("training_set")
        return parts[index + 1] if index + 1 < len(parts) else None
    if "verification_images" in parts:
        return None
    if "olivetti_96" in parts:
        person = next((part for part in parts if part.startswith("person_")), None)
        return person
    return None


def _camera_group(path: str) -> str | None:
    match = CAMERA_RE.search(PurePosixPath(path).name)
    return "cam" + match.group(1) if match else None


def _v3_identity_sources(workspace: Path) -> dict[str, set[str]]:
    base = workspace / "work/research-complete-v3/project/work/research-next/model-system/data/v3"
    result: dict[str, set[str]] = {}
    # Deliberately do not enumerate, stat, hash, or open final_records.jsonl.
    for role, filename in (("train", "train_records.jsonl"),
                           ("development", "development_records.jsonl"),
                           ("calibration", "calibration_records.jsonl")):
        ids = set()
        for row in _read_jsonl(base / filename):
            source = _source_id(row["relative_local_path"])
            if source:
                ids.add(source)
        result[role] = ids
    return result


def _quality(gray: Any) -> dict:
    import numpy as np
    pixels = np.asarray(gray, dtype=np.float32)
    mean = float(pixels.mean())
    std = float(pixels.std())
    core = pixels[1:-1, 1:-1]
    lap = (-4.0 * core + pixels[:-2, 1:-1] + pixels[2:, 1:-1]
           + pixels[1:-1, :-2] + pixels[1:-1, 2:])
    lap_var = float(lap.var())
    brightness = max(0.0, 1.0 - abs(mean - 127.5) / 127.5)
    contrast = min(1.0, std / 64.0)
    sharpness = min(1.0, math.log1p(max(0.0, lap_var)) / math.log1p(128.0))
    score = 0.30 * brightness + 0.30 * contrast + 0.40 * sharpness
    valid = 5.0 <= mean <= 250.0 and std >= 2.0 and lap_var >= 0.10
    return {"gray_mean_u8": mean, "gray_std_u8": std,
            "laplacian_variance": lap_var, "quality_score": score,
            "quality_rule_version": QUALITY_VERSION, "quality_valid": bool(valid)}


def _pixel_hash(path: Path) -> str:
    return sha256_file(path)


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _quality_score_summary(values: list[float]) -> dict:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    def q(p: float) -> float:
        position = (len(ordered) - 1) * p
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)
    return {"min": ordered[0], "p10": q(.1), "median": statistics.median(ordered),
            "p90": q(.9), "max": ordered[-1]}


def _role_rows(base_rows: list[dict], workspace: Path, selected_known: list[dict],
               unknown_assignment: dict[str, str], near_edges: list[tuple[str, str]],
               root_seed: int) -> tuple[list[dict], list[dict], dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in base_rows:
        if row.get("dataset") == "survface_96" and row.get("source_identity_id"):
            if row.get("access_state") == "reverified_this_job":
                by_source[row["source_identity_id"]].append(row)

    # Exact cross-identity checksum collisions are ambiguous labels and are excluded.
    digest_ids: dict[str, set[str]] = defaultdict(set)
    for row in base_rows:
        if row.get("dataset") == "survface_96" and row.get("source_identity_id") and row.get("sha256"):
            digest_ids[row["sha256"]].add(row["source_identity_id"])
    contradictory_hashes = {digest for digest, ids in digest_ids.items() if len(ids) > 1}
    for source in list(by_source):
        by_source[source] = [row for row in by_source[source] if row["sha256"] not in contradictory_hashes]

    # Build within-identity duplicate groups from the existing exact and scoped pHash audit.
    record_to_source = {row["record_id"]: row.get("source_identity_id") for row in base_rows if row.get("record_id")}
    components: dict[str, _UnionFind] = defaultdict(_UnionFind)
    records_by_source: dict[str, set[str]] = {}
    for source, rows in by_source.items():
        digest_records: dict[str, list[str]] = defaultdict(list)
        records_by_source[source] = {row["record_id"] for row in rows}
        for row in rows:
            digest_records[row["sha256"]].append(row["record_id"])
        for records in digest_records.values():
            for record_id in records[1:]:
                components[source].union(records[0], record_id)
    for left, right in near_edges:
        source_left, source_right = record_to_source.get(left), record_to_source.get(right)
        if source_left and source_left == source_right and left in records_by_source.get(source_left, set()) and right in records_by_source.get(source_left, set()):
            components[source_left].union(left, right)

    def groups_for(source: str) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in by_source.get(source, []):
            group = components[source].find(row["record_id"])
            groups[group].append(row)
        return groups

    split_rows: list[dict] = []
    presentation_rows: list[dict] = []
    group_summaries = {}
    def stable_key(text: str, stream: str) -> str:
        return hashlib.sha256(("nicla-v4:%d:%s:%s" % (root_seed, stream, text)).encode()).hexdigest()

    for known in selected_known:
        source = known["source_identity_id"]
        groups = groups_for(source)
        all_representatives = []
        quality_representatives = []
        for group_id, members in groups.items():
            rep = min(members, key=lambda row: stable_key(row["record_id"], "known-representative"))
            all_representatives.append({**rep, "duplicate_group_id": group_id})
            valid_members = [row for row in members if row.get("quality_valid")]
            if valid_members:
                quality_rep = max(valid_members, key=lambda row: (float(row.get("quality_score", 0.0)),
                                      stable_key(row["record_id"], "known-quality-member")))
                quality_representatives.append({**quality_rep, "duplicate_group_id": group_id})
        all_by_camera: dict[str, list[dict]] = defaultdict(list)
        train_by_camera: dict[str, list[dict]] = defaultdict(list)
        for row in all_representatives:
            all_by_camera[row.get("camera_source_group") or "camera_unknown"].append(row)
        for row in quality_representatives:
            train_by_camera[row.get("camera_source_group") or "camera_unknown"].append(row)
        cameras = sorted(set(all_by_camera) | set(train_by_camera))
        for camera in cameras:
            train_by_camera[camera].sort(key=lambda row: (-float(row.get("quality_score", 0.0)),
                                      stable_key(row["duplicate_group_id"], "known-quality-train")))
            all_by_camera[camera].sort(key=lambda row: stable_key(row["duplicate_group_id"], "known-pool"))
        # Training is quality-aware; evaluation, stopping and presentation use
        # every other verified image group, including low-quality examples.
        train_chosen: list[dict] = []
        while len(train_chosen) < KNOWN_BUDGET["train"]:
            progressed = False
            for camera in cameras:
                if train_by_camera[camera] and len(train_chosen) < KNOWN_BUDGET["train"]:
                    train_chosen.append(train_by_camera[camera].pop(0)); progressed = True
            if not progressed:
                break
        if len(train_chosen) < KNOWN_BUDGET["train"]:
            raise ValueError("known identity %s has only %d quality-valid groups for training; need %d" %
                             (source, len(train_chosen), KNOWN_BUDGET["train"]))
        training_groups = {row["duplicate_group_id"] for row in train_chosen}
        remaining_by_camera: dict[str, list[dict]] = defaultdict(list)
        for camera in cameras:
            remaining_by_camera[camera] = [row for row in all_by_camera[camera]
                                           if row["duplicate_group_id"] not in training_groups]
        nontrain_count = KNOWN_MODEL_GROUP_BUDGET - KNOWN_BUDGET["train"]
        presentation_count = KNOWN_BUDGET["presentation"]
        other_chosen: list[dict] = []
        while len(other_chosen) < nontrain_count + presentation_count:
            progressed = False
            for camera in cameras:
                if remaining_by_camera[camera] and len(other_chosen) < nontrain_count + presentation_count:
                    other_chosen.append(remaining_by_camera[camera].pop(0)); progressed = True
            if not progressed:
                break
        if len(other_chosen) < nontrain_count + presentation_count:
            raise ValueError("known identity %s has only %d additional unique groups; need %d" %
                             (source, len(other_chosen), nontrain_count + presentation_count))
        nontrain = other_chosen[:nontrain_count]
        pres = other_chosen[nontrain_count:]
        selected_by_camera: dict[str, list[dict]] = defaultdict(list)
        for row in nontrain:
            selected_by_camera[row.get("camera_source_group") or "camera_unknown"].append(row)
        nontrain_roles = {role: KNOWN_BUDGET[role] for role in KNOWN_BUDGET if role not in {"train", "presentation"}}
        role_camera_counts: dict[str, dict[str, int]] = {role: {} for role in nontrain_roles}
        remaining_counts = {cam: len(rows) for cam, rows in selected_by_camera.items()}
        for role, budget in nontrain_roles.items():
            total_remaining = sum(remaining_counts.values())
            raw = {cam: budget * count / total_remaining for cam, count in remaining_counts.items()}
            base = {cam: math.floor(value) for cam, value in raw.items()}
            left = budget - sum(base.values())
            order = sorted(raw, key=lambda item: (-(raw[item] - base[item]), item))
            while left:
                progressed = False
                for cam in order:
                    if base[cam] < remaining_counts[cam] and left:
                        base[cam] += 1
                        left -= 1
                        progressed = True
                if not progressed:
                    raise AssertionError("role quotas exceed remaining camera strata")
            role_camera_counts[role] = base
            for cam, count in base.items():
                remaining_counts[cam] -= count
        if any(count != 0 for count in remaining_counts.values()):
            raise AssertionError("nontraining role quotas did not conserve selected camera groups")
        assigned: dict[str, list[dict]] = {role: [] for role in nontrain_roles}
        for camera, rows in selected_by_camera.items():
            rows = sorted(rows, key=lambda row: stable_key(row["duplicate_group_id"], "known-role-order"))
            offset = 0
            for role in nontrain_roles:
                count = role_camera_counts[role][camera]
                assigned[role].extend(rows[offset:offset + count]); offset += count
            if offset != len(rows):
                raise AssertionError("per-camera nontraining role allocation did not conserve image groups")
        assigned["train"] = train_chosen
        for role, rows in assigned.items():
            for row in rows:
                split_rows.append({**row, "role": role, "identity_partition": "known",
                                   "label_id": known["pseudonym"], "class_index": CLASS_ORDER.index(known["pseudonym"]),
                                   "evidence_tier": "public_exploratory_historical_v3_development_exposure"})
        for row in pres:
            presentation_rows.append({**row, "presentation_role": "known_presentation",
                                      "identity_partition": "known", "label_id": known["pseudonym"],
                                      "class_index": CLASS_ORDER.index(known["pseudonym"]),
                                      "presentation_evidence_tier": "optical_public_image_prototype_only"})
        group_summaries[source] = {"available_unique_near_duplicate_groups": len(groups),
                                   "available_quality_valid_near_duplicate_groups": len(quality_representatives),
                                   "selected_role_group_counts": {role: len(rows) for role, rows in assigned.items()},
                                   "selected_camera_group_counts": dict(Counter(row.get("camera_source_group") or "camera_unknown" for row in train_chosen + nontrain)),
                                   "presentation_group_count": len(pres)}

    # Unknown IDs are split by identity; six image-level examples per identity are balanced to known training count.
    unknown_counts = Counter()
    for source, role in sorted(unknown_assignment.items(), key=lambda item: item[0]):
        groups = groups_for(source)
        representatives = []
        for group_id, members in groups.items():
            if role == "unknown_train":
                valid_members = [row for row in members if row.get("quality_valid")]
                if not valid_members:
                    continue
                row = max(valid_members, key=lambda item: (float(item.get("quality_score", 0.0)),
                             stable_key(item["record_id"], "unknown-quality-train")))
            else:
                row = min(members, key=lambda item: stable_key(item["record_id"], "unknown-representative"))
            representatives.append({**row, "duplicate_group_id": group_id})
        if len(representatives) < UNKNOWN_IMAGES_PER_IDENTITY:
            raise ValueError("unknown identity %s lacks six unique images" % source)
        by_cam: dict[str, list[dict]] = defaultdict(list)
        for row in representatives:
            by_cam[row.get("camera_source_group") or "camera_unknown"].append(row)
        for camera in by_cam:
            by_cam[camera].sort(key=lambda row: stable_key(row["duplicate_group_id"], role))
        chosen = []
        cameras = sorted(by_cam)
        while len(chosen) < UNKNOWN_IMAGES_PER_IDENTITY:
            progress = False
            for camera in cameras:
                if by_cam[camera] and len(chosen) < UNKNOWN_IMAGES_PER_IDENTITY:
                    chosen.append(by_cam[camera].pop(0)); progress = True
            if not progress:
                break
        for row in chosen:
            target = presentation_rows if role == "unknown_presentation" else split_rows
            record = {**row, "role": role, "identity_partition": "unknown",
                      "label_id": "OTHER", "class_index": CLASS_ORDER.index("OTHER"),
                      "evidence_tier": "public_exploratory_historical_v3_development_or_calibration_exposure"}
            if role == "unknown_presentation":
                record.pop("role", None)
                record["presentation_role"] = role
                record["presentation_evidence_tier"] = "optical_public_image_prototype_only"
            target.append(record)
        unknown_counts[role] += len(chosen)
    return split_rows, presentation_rows, {"known": group_summaries, "unknown_image_counts": dict(unknown_counts),
                                           "cross_identity_exact_duplicate_hash_count": len(contradictory_hashes)}


def run(workspace: Path) -> dict:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    data_root = workspace / "work/project-finish-v4/data"
    extracted_root = workspace / "work/dataset-prep"
    extracted_root_real = extracted_root.resolve()
    source_manifest = extracted_root / "reports/manifest.jsonl"
    prior_duplicates = _read_json(extracted_root / "reports/perceptual-duplicates.json")
    prior_summary = _read_json(extracted_root / "reports/dataset-summary.json")
    prior_archive = _read_json(extracted_root / "reports/archive-inventory.json")
    v3_roles = _v3_identity_sources(workspace)
    if v3_roles["train"] & v3_roles["development"] or v3_roles["train"] & v3_roles["calibration"] or v3_roles["development"] & v3_roles["calibration"]:
        raise ValueError("v3 train/development/calibration identity partitions overlap")

    # Derive the sealed original-v3-final identity set from non-final roles and the
    # already-audited source manifest. No original final-role manifest is opened.
    original_rows = _read_jsonl(source_manifest)
    source_ids = {_source_id(row["relative_local_path"]) for row in original_rows
                  if row.get("dataset") == "survface_96" and "training_set/" in row.get("relative_local_path", "")}
    source_ids.discard(None)
    original_v3_final_ids = source_ids - v3_roles["train"] - v3_roles["development"] - v3_roles["calibration"]
    open_scan_ids = v3_roles["train"] | v3_roles["development"] | v3_roles["calibration"]

    # Reverify and decode permitted labeled data plus Olivetti. Sealed original
    # v3-final identities and unlabeled verification images use the preexisting
    # October audit only and are excluded from all v4 roles.
    import numpy as np
    source_records: list[dict] = []
    quality_rows: list[dict] = []
    decode_errors = []
    dimension_counts = Counter(); mode_counts = Counter(); format_counts = Counter(); bit_depth_counts = Counter()
    orientations = Counter(); old_hash_mismatches = []
    exact_hash_to_sources: dict[str, set[str]] = defaultdict(set)
    identity_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    scannable = 0; sealed_skipped = 0; unlabeled_skipped = 0; missing_files = 0
    for record_index, old in enumerate(original_rows, 1):
        dataset = old["dataset"]
        rel_local = old["relative_local_path"]
        rel_parts = PurePosixPath(rel_local)
        if rel_parts.is_absolute() or ".." in rel_parts.parts:
            raise ValueError("source manifest path is absolute or traverses upward: " + rel_local)
        path = extracted_root_real.joinpath(*rel_parts.parts)
        try:
            path.relative_to(extracted_root_real)
        except ValueError as exc:
            raise ValueError("source manifest path escapes dataset-prep: " + rel_local) from exc
        source = _source_id(rel_local)
        if dataset == "survface_96" and "training_set/" in rel_local:
            if source in original_v3_final_ids:
                access_state = "sealed_original_v3_final_prior_audit_only"
                sealed_skipped += 1
                quality = None
            else:
                access_state = "reverified_this_job"
                quality = "pending"
        elif dataset == "olivetti_96":
            access_state = "reverified_this_job"
            quality = "pending"
        else:
            access_state = "unlabeled_survface_verification_prior_audit_only"
            unlabeled_skipped += 1
            quality = None
        row = {
            "record_id": old["record_id"], "dataset": dataset,
            "identity_id": old.get("identity_id"), "source_identity_id": source,
            "relative_path": "work/dataset-prep/" + rel_local,
            "sha256": old.get("sha256"), "source_sha256": old.get("sha256"),
            "source_archive_sha256": (prior_summary["datasets"][dataset].get("archive_sha256")
                                      if dataset in prior_summary["datasets"] else None),
            "camera_source_group": _camera_group(rel_local), "capture_session": None,
            "original_width": old.get("width"), "original_height": old.get("height"),
            "original_format": old.get("file_format"), "original_mode": old.get("mode"),
            "grayscale_method": "source_is_already_gray8_L; no color-to-gray conversion applied",
            "access_state": access_state,
        }
        if quality == "pending":
            if not path.is_file():
                missing_files += 1
                decode_errors.append({"record_id": row["record_id"], "reason": "missing file"})
                row["access_state"] = "missing"
                source_records.append(row)
                continue
            current_hash = _pixel_hash(path)
            if current_hash != old.get("sha256"):
                old_hash_mismatches.append({"record_id": row["record_id"], "expected": old.get("sha256"), "actual": current_hash})
                row["access_state"] = "hash_mismatch"
                source_records.append(row)
                continue
            try:
                with Image.open(path) as probe:
                    probe.verify()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    image_format, mode = image.format, image.mode
                    orientation = image.getexif().get(274, 1)
                    gray = image.copy() if image.mode == "L" else image.convert("L")
                    q = _quality(gray)
            except Exception as exc:
                decode_errors.append({"record_id": row["record_id"], "reason": type(exc).__name__ + ": " + str(exc)})
                row["access_state"] = "decode_error"
                source_records.append(row)
                continue
            row.update({"verified_sha256": current_hash, "byte_size": path.stat().st_size,
                        "width": width, "height": height, "format": image_format, "mode": mode,
                        "orientation_tag": orientation, "quality_valid": q["quality_valid"],
                        "quality_score": q["quality_score"], "gray_mean_u8": q["gray_mean_u8"],
                        "gray_std_u8": q["gray_std_u8"], "laplacian_variance": q["laplacian_variance"],
                        "quality_rule_version": QUALITY_VERSION})
            if width != 96 or height != 96 or mode != "L":
                decode_errors.append({"record_id": row["record_id"], "reason": "unexpected dimensions or non-L source"})
                row["access_state"] = "metadata_mismatch"
            dimension_counts["%dx%d" % (width, height)] += 1
            mode_counts[mode] += 1; format_counts[str(image_format)] += 1
            bit_depth_counts[old.get("bit_depth")] += 1; orientations[str(orientation)] += 1
            if dataset == "survface_96" and "training_set/" in rel_local:
                exact_hash_to_sources[current_hash].add(str(source))
                identity_rows[(dataset, str(source))].append(row)
            elif dataset == "olivetti_96":
                identity_rows[(dataset, str(source))].append(row)
            quality_rows.append({"record_id": row["record_id"], "dataset": dataset,
                                 "identity_id": row.get("identity_id"), "source_identity_id": source,
                                 "relative_path": row["relative_path"], "sha256": current_hash,
                                 "camera_source_group": row["camera_source_group"], "capture_session": None,
                                 **q})
            scannable += 1
        source_records.append(row)
        if record_index % 25000 == 0:
            print("curation image audit: %d/%d manifest rows; %d images verified" %
                  (record_index, len(original_rows), scannable), file=sys.stderr, flush=True)

    # Distinct exact hashes must not carry contradictory identity labels.
    contradictory_hashes = {digest for digest, ids in exact_hash_to_sources.items() if len(ids) > 1}
    if missing_files or old_hash_mismatches or decode_errors:
        status = "failed_needs_repair"
    else:
        status = "complete_verified"

    # Use prior pHash candidate edges only within their recorded scope; they are
    # duplicate-group safeguards, not a claim of exhaustive cross-identity search.
    surf = prior_duplicates.get("datasets", {}).get("survface_96", {})
    near_edges = []
    for pair in surf.get("candidate_pairs", []):
        left, right = pair.get("left_record_id"), pair.get("right_record_id")
        if left and right:
            near_edges.append((left, right))
    olivetti_near_edges = []
    for pair in prior_duplicates.get("datasets", {}).get("olivetti_96", {}).get("candidate_pairs", []):
        left, right = pair.get("left_record_id"), pair.get("right_record_id")
        if left and right:
            olivetti_near_edges.append((left, right))

    # Per-ID quality inventory: all accessible, historically non-final IDs plus
    # Olivetti. The 530 sealed v3-final identities are summarized as excluded.
    duplicate_components: dict[str, _UnionFind] = defaultdict(_UnionFind)
    record_sources = {row["record_id"]: row.get("source_identity_id") for row in quality_rows}
    quality_by_record = {row["record_id"]: row for row in quality_rows}
    hashes_by_record = {row["record_id"]: row["sha256"] for row in quality_rows}
    source_by_record = {row["record_id"]: row.get("source_identity_id") for row in quality_rows}
    exact_hash_groups: dict[str, list[str]] = defaultdict(list)
    for record_id, digest in hashes_by_record.items():
        exact_hash_groups[digest].append(record_id)
    for rows in exact_hash_groups.values():
        for record_id in rows[1:]:
            a, b = rows[0], record_id
            source = source_by_record.get(a)
            if source and source == source_by_record.get(b):
                duplicate_components[source].union(a, b)
    for left, right in near_edges:
        a, b = record_sources.get(left), record_sources.get(right)
        if a and a == b and left in quality_by_record and right in quality_by_record:
            duplicate_components[a].union(left, right)
    group_counts: Counter = Counter()
    valid_group_counts: Counter = Counter()
    camera_group_counts: dict[str, Counter] = defaultdict(Counter)
    valid_camera_group_counts: dict[str, Counter] = defaultdict(Counter)
    per_id_quality: dict[str, list[dict]] = defaultdict(list)
    group_seen: dict[str, set[str]] = defaultdict(set)
    valid_group_seen: dict[str, set[str]] = defaultdict(set)
    valid_camera_group_seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in quality_rows:
        source = str(row.get("source_identity_id"))
        group = duplicate_components[source].find(row["record_id"])
        if group not in group_seen[source]:
            group_seen[source].add(group); group_counts[source] += 1
            camera_group_counts[source][row.get("camera_source_group") or "camera_unknown"] += 1
        if row["quality_valid"] and group not in valid_group_seen[source]:
            valid_group_seen[source].add(group); valid_group_counts[source] += 1
        camera = row.get("camera_source_group") or "camera_unknown"
        if row["quality_valid"] and (group, camera) not in valid_camera_group_seen[source]:
            valid_camera_group_seen[source].add((group, camera)); valid_camera_group_counts[source][camera] += 1
        per_id_quality[source].append(row)
    source_manifest_by_id = defaultdict(list)
    for row in source_records:
        if row.get("dataset") == "survface_96" and row.get("source_identity_id"):
            source_manifest_by_id[row["source_identity_id"]].append(row)
    quality_inventory = []
    for source in sorted(v3_roles["train"] | v3_roles["development"] | v3_roles["calibration"]):
        rows = per_id_quality.get(source, [])
        source_rows = source_manifest_by_id.get(source, [])
        if not source_rows:
            continue
        qvals = {key: [float(row[key]) for row in rows] for key in ("gray_mean_u8", "gray_std_u8", "laplacian_variance", "quality_score")}
        quality_inventory.append({"dataset": "survface_96", "source_identity_id": source,
          "identity_id": source_rows[0].get("identity_id"), "v3_exposure_role": "train" if source in v3_roles["train"] else "development" if source in v3_roles["development"] else "calibration",
          "image_count_from_prior_source_manifest": len(source_rows), "images_reverified_this_job": len(rows),
          "unique_sha256_count": len({row["sha256"] for row in rows}),
          "non_near_duplicate_image_groups": group_counts[source], "quality_valid_groups": valid_group_counts[source],
          "camera_source_group_counts": dict(camera_group_counts[source]),
          "camera_quality_valid_group_counts": dict(valid_camera_group_counts[source]), "capture_session_metadata": "absent; no session IDs inferred",
          "apparent_resolution": "96x96 in supplied derivative; native source resolution unknown",
          "face_crop_coverage": "not measurable from this crop-only archive: no boxes, landmarks, or face-presence annotations",
          "gray_mean_u8": _quality_score_summary(qvals["gray_mean_u8"]),
          "gray_std_u8": _quality_score_summary(qvals["gray_std_u8"]),
          "laplacian_variance": _quality_score_summary(qvals["laplacian_variance"]),
          "quality_score": _quality_score_summary(qvals["quality_score"])})
    olivetti_counts = Counter(row["source_identity_id"] for row in source_records if row.get("dataset") == "olivetti_96")
    for source in sorted(olivetti_counts):
        rows = per_id_quality.get(source, [])
        quality_inventory.append({"dataset": "olivetti_96", "source_identity_id": source,
          "identity_id": next((row.get("identity_id") for row in source_records if row.get("dataset")=="olivetti_96" and row.get("source_identity_id")==source),None),
          "image_count_from_prior_source_manifest": olivetti_counts[source], "images_reverified_this_job": len(rows),
          "unique_sha256_count": len({row["sha256"] for row in rows}),
          "non_near_duplicate_image_groups": group_counts[source], "quality_valid_groups": valid_group_counts[source],
          "camera_source_group_counts": {}, "capture_session_metadata": "absent",
          "apparent_resolution": "96x96 in transformed PNG derivative; native source resolution unknown",
          "face_crop_coverage": "not measured; legacy controlled face crops without boxes or landmarks",
          "gray_mean_u8": _quality_score_summary([row["gray_mean_u8"] for row in rows]),
          "gray_std_u8": _quality_score_summary([row["gray_std_u8"] for row in rows]),
          "laplacian_variance": _quality_score_summary([row["laplacian_variance"] for row in rows]),
          "quality_score": _quality_score_summary([row["quality_score"] for row in rows])})

    # Rank known-roster candidates only within identities historically used for
    # v3 development, using source coverage and image metadata—not scores.
    summary_by_id = {row["source_identity_id"]: row for row in quality_inventory if row["dataset"] == "survface_96"}
    candidates = []
    for source in sorted(v3_roles["development"]):
        row = summary_by_id.get(source)
        if not row:
            continue
        cams = {cam: count for cam, count in row["camera_quality_valid_group_counts"].items()
                if cam != "camera_unknown" and count >= 10}
        valid = row["quality_valid_groups"]
        unique_groups = row["non_near_duplicate_image_groups"]
        reasons = []
        if unique_groups < KNOWN_TOTAL_GROUP_BUDGET: reasons.append("fewer_than_130_unique_non_near_duplicate_groups_for_all_roles")
        if valid < KNOWN_BUDGET["train"]: reasons.append("fewer_than_60_quality_valid_groups_for_training")
        if len(cams) < 2: reasons.append("fewer_than_two_filename_camera_groups_with_10_quality_valid_groups")
        candidates.append({"source_identity_id": source, "pseudonymous_identity_id": row["identity_id"],
          "raw_images": row["image_count_from_prior_source_manifest"], "unique_groups": unique_groups,
          "quality_valid_groups": valid,
          "camera_groups_with_at_least_10": len(cams), "camera_group_counts": cams,
          "minimum_covered_camera_group_count": min(cams.values()) if cams else 0,
          "eligible": not reasons, "rejection_reasons": reasons,
          "selection_rule":"metadata-only; no v3 scores or held-out performance accessed"})
    candidates.sort(key=lambda row:(-int(row["eligible"]),-row["camera_groups_with_at_least_10"],
                                    -row["minimum_covered_camera_group_count"],-row["quality_valid_groups"],
                                    row["source_identity_id"]))
    eligible = [row for row in candidates if row["eligible"]]
    if len(eligible) < 5:
        raise ValueError("only %d eligible v3-development identities; cannot freeze five-person roster" % len(eligible))
    chosen_rows = eligible[:5]
    selected_known = []
    for index, row in enumerate(chosen_rows, 1):
        selected_known.append({"label": "K%02d" % index,"pseudonym": "K%02d" % index,
          "source_identity_id": row["source_identity_id"], "pseudonymous_identity_id": row["pseudonymous_identity_id"],
          "metadata_only_rank": index, "eligible_images": row["raw_images"],
          "quality_valid_near_duplicate_groups": row["quality_valid_groups"],
          "unique_near_duplicate_groups": row["unique_groups"],
          "source_camera_groups": row["camera_group_counts"],
          "historical_exposure": "identity and all source images in the v3 development role; v3 train identity overlap is zero; v3 development scores are historical exposure, so all new public-image results are exploratory"})

    eligible_unknown = []
    open_ids = v3_roles["development"] | v3_roles["calibration"]
    known_ids = {row["source_identity_id"] for row in selected_known}
    for source in sorted(open_ids - known_ids):
        summary = summary_by_id.get(source)
        if summary is None:
            continue
        if summary["quality_valid_groups"] >= UNKNOWN_IMAGES_PER_IDENTITY:
            eligible_unknown.append(source)
    # Calibration IDs were not in the development candidate summary; derive
    # their quality groups from the same per-record summary inventory.
    by_source_quality = defaultdict(list)
    for row in quality_rows:
        if row.get("dataset") == "survface_96":
            by_source_quality[str(row["source_identity_id"])].append(row)
    all_open_source_records = defaultdict(list)
    for row in source_records:
        if row.get("dataset")=="survface_96" and row.get("source_identity_id"):
            all_open_source_records[row["source_identity_id"]].append(row)
    for source in sorted(v3_roles["calibration"] - known_ids):
        if source in eligible_unknown: continue
        rows=by_source_quality.get(source,[])
        unique_groups={duplicate_components[source].find(row["record_id"]) for row in rows if row["quality_valid"]}
        if len(unique_groups)>=UNKNOWN_IMAGES_PER_IDENTITY: eligible_unknown.append(source)
    eligible_unknown=sorted(set(eligible_unknown))
    required_unknown=len(UNKNOWN_ROLES)*UNKNOWN_IDENTITIES_PER_ROLE
    if len(eligible_unknown)<required_unknown:
        raise ValueError("only %d unknown identities with six quality-valid groups; need %d"%(len(eligible_unknown),required_unknown))
    sensitivity_roots = _read_json(workspace / "work/project-finish-v4/config/seed_registry.json")["data_selection_sensitivity_roots"]
    if len(sensitivity_roots) != 10 or len(set(sensitivity_roots)) != 10:
        raise ValueError("seed registry must supply ten distinct data selection sensitivity roots")
    from .roles import validate_manifest

    # Existing per-identity pHash candidate edges are converted into group constraints.
    near_edges_for_split=[]
    all_record_ids={row["record_id"] for row in source_records if row.get("access_state")=="reverified_this_job"}
    for left,right in near_edges:
        if left in all_record_ids and right in all_record_ids:
            near_edges_for_split.append((left,right))
    sensitivity_outputs=[]
    split_rows=presentation_rows=group_summary=None
    for seed in sensitivity_roots:
        seed_order=sorted(eligible_unknown,key=lambda source:hashlib.sha256(("nicla-v4:%d:unknown-roster:%s"%(seed,source)).encode()).hexdigest())
        assignment={source:UNKNOWN_ROLES[index//UNKNOWN_IDENTITIES_PER_ROLE]
                    for index,source in enumerate(seed_order[:required_unknown])}
        seed_split,seed_presentation,seed_summary=_role_rows(source_records,workspace,selected_known,assignment,near_edges_for_split,int(seed))
        seed_validation=validate_manifest([{"record_id":r["record_id"],"identity_id":r["source_identity_id"],"role":r["role"],"sha256":r["sha256"],"relative_path":r["relative_path"]} for r in seed_split])
        if seed_validation["status"] != "pass":
            raise ValueError("seed %s split invalid: %s"%(seed,"; ".join(seed_validation["errors"][:10])))
        sensitivity_outputs.append({"root_seed":int(seed),"unknown_assignment":assignment,"split_rows":seed_split,
                                    "presentation_rows":seed_presentation,"group_summary":seed_summary,
                                    "selection_fingerprint":hashlib.sha256("\n".join(sorted(r["record_id"]+":"+r["role"] for r in seed_split)).encode()).hexdigest()})
        if int(seed)==int(sensitivity_roots[0]):
            split_rows,presentation_rows,group_summary=seed_split,seed_presentation,seed_summary

    # Olivetti is a separate low-data sanity protocol with group-safe 60/20/20 splits.
    olivetti_rows=[row for row in source_records if row.get("dataset")=="olivetti_96" and row.get("access_state")=="reverified_this_job"]
    olivetti_by_id=defaultdict(list)
    for row in olivetti_rows: olivetti_by_id[row["source_identity_id"]].append(row)
    olivetti_manifest=[]
    for source,rows in sorted(olivetti_by_id.items()):
        ids=[row["record_id"] for row in rows]
        uf=_UnionFind()
        hash_map=defaultdict(list)
        for row in rows: hash_map[row["sha256"]].append(row["record_id"])
        for items in hash_map.values():
            for record_id in items[1:]: uf.union(items[0],record_id)
        for left,right in olivetti_near_edges:
            if left in ids and right in ids: uf.union(left,right)
        groups=defaultdict(list)
        for row in rows: groups[uf.find(row["record_id"])].append(row)
        ordered=sorted(groups.values(),key=lambda group:hashlib.sha256(("nicla-v4:20261201:olivetti:%s:%s"%(source,group[0]["record_id"])).encode()).hexdigest())
        target={"train":6,"development":2,"final":2}
        assigned=Counter(); cursor=0
        for group in ordered:
            role=max(target,key=lambda key:(target[key]-assigned[key],-list(target).index(key)))
            for row in group:
                olivetti_manifest.append({**row,"role":role,"identity_partition":"known","label_id":row["identity_id"],
                  "evidence_tier":"small_data_sanity_only","source_group_id":uf.find(row["record_id"])})
            assigned[role]+=len(group);cursor+=len(group)
    
    # Publish manifests, provenance, and metadata-only ranking.
    data_root.mkdir(parents=True,exist_ok=True)
    sensitivity_receipts=[]
    for block in sensitivity_outputs:
        block_seed=block["root_seed"]
        if block_seed==int(sensitivity_roots[0]):
            continue
        block_dir=data_root/"sensitivity"/str(block_seed)
        block_dir.mkdir(parents=True,exist_ok=True)
        _write_jsonl(block_dir/"primary_split_manifest.jsonl",block["split_rows"])
        _write_jsonl(block_dir/"presentation_manifest.jsonl",block["presentation_rows"])
        role_identity_counts={role:len({row["source_identity_id"] for row in block["split_rows"] if row["role"]==role})
                              for role in ["train","stopping","threshold_calibration","development","final",*UNKNOWN_ROLES]}
        role_identity_counts["unknown_presentation"]=len({row["source_identity_id"] for row in block["presentation_rows"]
                                                           if row.get("presentation_role")=="unknown_presentation"})
        receipt={"schema":"data-selection-sensitivity-block-1","root_seed":block_seed,
                 "primary_roster_unchanged":True,"selected_known_source_ids":[row["source_identity_id"] for row in selected_known],
                 "unknown_identity_counts_by_role":{role:role_identity_counts[role] for role in UNKNOWN_ROLES},
                 "row_counts":{"primary_split":len(block["split_rows"]),"presentation":len(block["presentation_rows"])},
                 "selection_fingerprint":block["selection_fingerprint"],
                 "selection_method":"deterministic identity-disjoint unknown roster assignment and group-safe image split; no model fitting or outcome-based selection"}
        atomic_write_json(block_dir/"BLOCK_RESULT.json",receipt)
        sensitivity_receipts.append(receipt)
    _write_jsonl(data_root/"source_manifest.jsonl",source_records)
    _write_jsonl(data_root/"quality_inventory.jsonl",quality_rows)
    _write_jsonl(data_root/"primary_split_manifest.jsonl",split_rows)
    _write_jsonl(data_root/"presentation_manifest.jsonl",presentation_rows)
    _write_jsonl(data_root/"olivetti_sanity_manifest.jsonl",olivetti_manifest)
    _write_jsonl(data_root/"per_identity_quality.jsonl",quality_inventory)
    rank_path=data_root/"candidate_ranking.csv"
    import csv
    with rank_path.open("w",encoding="utf-8",newline="") as stream:
        fields=["source_identity_id","pseudonymous_identity_id","raw_images","unique_groups","quality_valid_groups","camera_groups_with_at_least_10","minimum_covered_camera_group_count","eligible","rejection_reasons","camera_group_counts","selection_rule"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in candidates: writer.writerow({**row,"rejection_reasons":";".join(row["rejection_reasons"]),"camera_group_counts":json.dumps(row["camera_group_counts"],sort_keys=True)})
    roster={"schema":"public-five-roster-1","protocol_id":"nicla-five-known-study-v4","created_utc":_utc(),
      "dataset":"SurvFace-derived 96x96 grayscale crops","archive_sha256":prior_summary["datasets"]["survface_96"]["archive_sha256"],
      "class_order":CLASS_ORDER,"identity_map":selected_known,"selection_rule":{"candidate_pool":"v3 development-role SurvFace identities only; v3 train, v3 calibration and sealed v3 final identities excluded from known-roster selection",
        "minimum_unique_near_duplicate_groups":KNOWN_TOTAL_GROUP_BUDGET,"minimum_quality_valid_training_groups":KNOWN_BUDGET["train"],
        "minimum_source_camera_groups":2,"minimum_quality_valid_groups_per_covered_camera_group":10,
        "ranking_order":["number of source camera groups with at least ten quality-valid unique image groups, descending","minimum eligible group count among covered camera groups, descending","quality-valid near-duplicate-group count, descending","numeric source identity as lexical tie-break"],
        "prohibited_inputs":["v3 development scores","v3 calibration scores","held-out/final recognition performance","demographics or inferred real names"],
        "historical_exposure":"all five selected identities and images were in v3 development. This v4 public benchmark is exploratory; it is not a fresh confirmatory set."},
      "image_budget_per_known_identity":KNOWN_BUDGET,"training_gallery_parity":{"classifier_train_images_per_identity":KNOWN_BUDGET["train"],"embedding_gallery_images_per_identity":KNOWN_BUDGET["train"],"same_source_record_ids":True,"gallery_role":"train records; no extra permitted information","train_selection":"quality-valid groups only, ranked by quality score within filename-camera groups with hash-based tie break; applied only to the training role. All verified images remain eligible for other roles."},
      "unknown_roles":{"identities_per_role":UNKNOWN_IDENTITIES_PER_ROLE,"images_per_identity":UNKNOWN_IMAGES_PER_IDENTITY,"roles":UNKNOWN_ROLES,"class_balance":"300 OTHER train images total matches the 300 known training images; downstream batches must sample five known classes and OTHER with equal class weights."},
      "original_v3_final":"sealed; excluded from all v4 splits; its source image files were not opened or rehashed by this job",
      "unlabeled_survface_verification_pool":"excluded; never treated as OTHER",
      "quality_rule":QUALITY_RULE,"split_seed":20261201,"split_grouping":"exact checksum and existing within-identity pHash<=6 candidate groups are kept within one role; filename camera groups are strata, not capture sessions; sessions are unknown."}
    atomic_write_json(data_root/"roster.json",roster)
    primary_unknown_identity_counts={role:len({row["source_identity_id"] for row in split_rows if row["role"]==role})
                                     for role in UNKNOWN_ROLES if role!="unknown_presentation"}
    primary_unknown_identity_counts["unknown_presentation"]=len({row["source_identity_id"] for row in presentation_rows
                                                                   if row.get("presentation_role")=="unknown_presentation"})
    primary_sensitivity={"schema":"data-selection-sensitivity-block-1","root_seed":int(sensitivity_roots[0]),
      "primary_roster_unchanged":True,"selected_known_source_ids":[row["source_identity_id"] for row in selected_known],
      "unknown_identity_counts_by_role":primary_unknown_identity_counts,
      "row_counts":{"primary_split":len(split_rows),"presentation":len(presentation_rows)},
      "selection_fingerprint":next(block["selection_fingerprint"] for block in sensitivity_outputs if block["root_seed"]==int(sensitivity_roots[0])),
      "selection_method":"deterministic identity-disjoint unknown roster assignment and group-safe image split; no model fitting or outcome-based selection"}
    atomic_write_json(data_root/"sensitivity"/str(sensitivity_roots[0])/"BLOCK_RESULT.json",primary_sensitivity)
    atomic_write_json(data_root/"sensitivity_blocks.json",{"schema":"data-selection-sensitivity-index-1","protocol_id":"nicla-five-known-study-v4",
      "roots":[int(seed) for seed in sensitivity_roots],"blocks":[primary_sensitivity,*sensitivity_receipts],
      "purpose":"Secondary data-split/unknown-roster robustness. These blocks cannot replace the frozen primary roster after outcomes are viewed."})
    # Role manifest is validated for known/unknown identity separation.
    from .roles import validate_manifest
    validated=validate_manifest([{ "record_id":r["record_id"],"identity_id":r["source_identity_id"],"role":r["role"],"sha256":r["sha256"],"relative_path":r["relative_path"]} for r in split_rows])
    if validated["status"]!="pass":
        raise ValueError("primary split manifest invalid: "+"; ".join(validated["errors"][:20]))
    archive_by_name={row["archive"]:row for row in prior_archive["archives"]}
    verifier={"schema":"public-data-verification-1","created_utc":_utc(),"status":status,
      "source_archive_receipts":archive_by_name,"current_archive_rehash":{"survface_96.zip":False,"olivetti_96.zip":False},
      "archive_hash_policy":"The existing October 21 CRC/hash audit is the source archive receipt. Reopening the entire SurvFace archive would read original sealed v3-final image bytes, so this job reuses its receipted archive hash and directly rehashes/decode-checks only permitted non-final image files plus Olivetti.",
      "source_images_reverified_and_decoded":scannable,"original_v3_final_image_files_skipped":sealed_skipped,
      "unlabeled_verification_images_not_reopened":unlabeled_skipped,"missing_files":missing_files,
      "prior_manifest_hash_mismatches":len(old_hash_mismatches),"decode_or_metadata_errors":decode_errors,
      "current_sha256_mismatches":old_hash_mismatches,"contradictory_exact_duplicate_hashes_across_identity_labels":len(contradictory_hashes),
      "dimensions":dict(dimension_counts),"modes":dict(mode_counts),"formats":dict(format_counts),"bit_depths":dict(bit_depth_counts),"exif_orientation_tags":dict(orientations),
      "source_grouping":"SurvFace filename camN tags are recorded as camera-source groups only; no capture sessions are supplied.",
      "face_crop_coverage":"No face boxes, landmarks, or face-presence annotations accompany the 96x96 crop archive; coverage is reported as not computable and the selected contact sheets receive human visual review.",
      "near_duplicate_scope":"Existing pHash candidates are within-identity for labeled training images and bundled positive verification pairs; cross-identity near-duplicate comparison was not exhaustive. Exact cross-identity SHA-256 collisions are screened and excluded.",
      "datasets_are_separate":True,"prepared_image_copies_created":False}
    atomic_write_json(data_root/"verification_report.json",verifier)
    access={"schema":"public-data-access-exposure-ledger-1","created_utc":_utc(),
      "v3_role_manifests_read":["train_records.jsonl","development_records.jsonl","calibration_records.jsonl"],
      "v3_final_records_manifest_opened":False,"original_v3_final_ids_inferred_by_complement_only":len(original_v3_final_ids),
      "original_v3_final_image_files_opened_or_rehashed":0,"v3_train_identity_count":len(v3_roles["train"]),
      "v3_development_identity_count":len(v3_roles["development"]),"v3_calibration_identity_count":len(v3_roles["calibration"]),
      "v4_known_roster_source_role":"v3 development","v4_known_roster_v3_train_identity_overlap":0,
      "v4_known_roster_v3_development_image_exposure":"The five chosen identities were evaluated in v3 development; source images may have influenced historical v3 model selection. New results are exploratory.",
      "v4_unknown_roles_are_identity_disjoint":True,"v4_unknown_roles_use_only_v3_development_or_calibration_identity_pools":True,
      "v3_train_identities_used_in_v4_unknown_roles":0,"original_v3_final_identities_used_in_any_v4_role":0,
      "unlabeled_survface_images_used_as_other":False,"participant_data_used":False,
      "label_map":"K01-K05 map to public source folder identifiers in roster.json; no real names or demographics are inferred.",
      "rights_and_provenance":"Exact transformed SurvFace archive provenance remains unverified; official source is research-use-only and upstream copyrights remain. All images stay local and are excluded from shareable release.",
      "source_manifest_sha256":sha256_file(data_root/"source_manifest.jsonl"),"v3_data_ledger_sha256":sha256_file(workspace/"work/research-complete-v3/DATA_AND_ACCESS_LEDGER.json")}
    atomic_write_json(data_root/"source_access_exposure_ledger.json",access)
    summary={"schema":"public-five-curation-summary-1","created_utc":_utc(),"roster":roster["identity_map"],
      "known_budget":KNOWN_BUDGET,"unknown_role_counts":group_summary["unknown_image_counts"],
      "known_group_summary":group_summary["known"],"primary_split_rows":len(split_rows),
      "presentation_rows":len(presentation_rows),"olivetti_rows":len(olivetti_manifest),
      "candidate_count":len(candidates),"eligible_candidate_count":len(eligible),
      "status":"pass" if status=="complete_verified" and len(split_rows)>0 else status}
    atomic_write_json(data_root/"curation_summary.json",summary)
    return {"summary":summary,"verification":verifier,"access":access,"roster":roster}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    result = run(args.workspace.resolve())
    print(json.dumps(result["summary"],indent=2,ensure_ascii=False))
    return 0 if result["summary"]["status"]=="pass" else 2


if __name__=="__main__":
    raise SystemExit(main())
