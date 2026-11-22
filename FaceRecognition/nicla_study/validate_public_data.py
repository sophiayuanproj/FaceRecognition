"""Run ten deterministic data-role, loader, and exposure validation blocks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .data_pipeline import (CLASS_ORDER, balanced_class_batches, balanced_identity_batches, load_pixels,
                            read_jsonl, safe_path, select_rows, validate_primary_manifest)
from .roles import assert_role_access
from .curate_public_data import _read_json, _read_jsonl, _source_id, UNKNOWN_ROLES
from .fs import atomic_write_json, sha256_file


def _expect_raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError("expected %s" % exception.__name__)


def _fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(row["record_id"] + ":" + row["role"] for row in rows)).encode()).hexdigest()


def _read_assignment(manifest_rows: list[dict], presentation_rows: list[dict]) -> dict[str, str]:
    assignment = {row["source_identity_id"]: row["role"] for row in manifest_rows
                  if row.get("identity_partition") == "unknown"}
    assignment.update({row["source_identity_id"]: row["presentation_role"] for row in presentation_rows
                       if row.get("identity_partition") == "unknown"})
    return assignment


def run(workspace: Path) -> dict:
    data_root = workspace / "work/project-finish-v4/data"
    split_path = data_root / "primary_split_manifest.jsonl"
    presentation_path = data_root / "presentation_manifest.jsonl"
    primary_rows = read_jsonl(split_path)
    primary_presentation = read_jsonl(presentation_path)
    roster = _read_json(data_root / "roster.json")
    exposure = _read_json(data_root / "source_access_exposure_ledger.json")
    verification = _read_json(data_root / "verification_report.json")
    index = _read_json(data_root / "sensitivity_blocks.json")
    seed_registry = _read_json(workspace / "work/project-finish-v4/config/seed_registry.json")
    roots = seed_registry["data_selection_sensitivity_roots"]
    if roots != index["roots"] or len(roots) != 10:
        raise AssertionError("sensitivity blocks do not match the ten frozen seed roots")
    if roster["class_order"] != CLASS_ORDER:
        raise AssertionError("class order differs from roster contract")

    # Independently reproduce metadata-only known and unknown identity choices.
    ranking = []
    with (data_root / "candidate_ranking.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["eligible"] == "True":
                ranking.append(row)
    ranking.sort(key=lambda row: (-int(row["camera_groups_with_at_least_10"]),
                    -int(row["minimum_covered_camera_group_count"]),
                    -int(row["quality_valid_groups"]), row["source_identity_id"]))
    expected_known = [row["source_identity_id"] for row in ranking[:5]]
    selected_known = [row["source_identity_id"] for row in roster["identity_map"]]
    if selected_known != expected_known:
        raise AssertionError("frozen roster cannot be reproduced from metadata ranking")
    known_set = set(selected_known)
    quality_rows = _read_jsonl(data_root / "per_identity_quality.jsonl")
    eligible_unknown = sorted({row["source_identity_id"] for row in quality_rows
        if row.get("dataset") == "survface_96"
        and row.get("v3_exposure_role") in {"development", "calibration"}
        and row.get("quality_valid_groups", 0) >= 6
        and row["source_identity_id"] not in known_set})
    if len(eligible_unknown) < 250:
        raise AssertionError("metadata inventory cannot reproduce 250 unknown identities")

    source_rows = _read_jsonl(data_root / "source_manifest.jsonl")
    all_source_ids = {_source_id(row["relative_path"].removeprefix("work/dataset-prep/"))
        for row in source_rows if row.get("dataset") == "survface_96"
        and "training_set/" in row.get("relative_path", "")}
    all_source_ids.discard(None)
    v3_base = workspace / "work/research-complete-v3/project/work/research-next/model-system/data/v3"
    from .curate_public_data import _v3_identity_sources
    v3_roles = _v3_identity_sources(workspace)
    sealed_ids = all_source_ids - v3_roles["train"] - v3_roles["development"] - v3_roles["calibration"]
    if exposure.get("v3_final_records_manifest_opened") is not False:
        raise AssertionError("v3 final records access ledger is not fail-closed")
    if exposure.get("original_v3_final_image_files_opened_or_rehashed") != 0:
        raise AssertionError("sealed v3 final image access count is nonzero")
    if verification.get("original_v3_final_image_files_skipped", 0) <= 0:
        raise AssertionError("verification report does not document skipped sealed images")
    for row in source_rows:
        if row.get("access_state") == "sealed_original_v3_final_prior_audit_only":
            if any(key in row for key in ("verified_sha256", "quality_valid", "quality_score")):
                raise AssertionError("sealed row contains fresh image-derived metadata")

    duplicate_report = _read_json(workspace / "work/dataset-prep/reports/perceptual-duplicates.json")
    pair_edges = []
    for pair in duplicate_report["datasets"]["survface_96"]["candidate_pairs"]:
        pair_edges.append((pair["left_record_id"], pair["right_record_id"]))

    blocks = []
    for seed, receipt in zip(roots, index["blocks"]):
        block_dir = data_root / "sensitivity" / str(seed)
        block_rows = read_jsonl(split_path if seed == roots[0] else block_dir / "primary_split_manifest.jsonl")
        block_presentation = read_jsonl(presentation_path if seed == roots[0] else block_dir / "presentation_manifest.jsonl")
        validate_primary_manifest(block_rows)
        actual_fingerprint = _fingerprint(block_rows)
        if actual_fingerprint != receipt["selection_fingerprint"]:
            raise AssertionError("seed %s selection fingerprint mismatch" % seed)
        expected_order = sorted(eligible_unknown,
            key=lambda source: hashlib.sha256(("nicla-v4:%d:unknown-roster:%s" % (seed, source)).encode()).hexdigest())
        expected_assignment = {source: UNKNOWN_ROLES[position // 50]
                               for position, source in enumerate(expected_order[:250])}
        if _read_assignment(block_rows, block_presentation) != expected_assignment:
            raise AssertionError("seed %s unknown roster/split assignment is not reproducible" % seed)
        if any(row["source_identity_id"] in sealed_ids for row in block_rows + block_presentation):
            raise AssertionError("sealed v3 final identity appears in seed %s" % seed)
        if any(row.get("dataset") != "survface_96" for row in block_rows + block_presentation):
            raise AssertionError("Olivetti was mixed into the SurvFace primary task")
        counts = {}
        for known in roster["identity_map"]:
            source = known["source_identity_id"]
            identity_rows = [row for row in block_rows if row["source_identity_id"] == source]
            counts[source] = {role: sum(row["role"] == role for row in identity_rows)
                              for role in ("train", "stopping", "threshold_calibration", "development", "final")}
            if counts[source] != {"train": 60, "stopping": 10, "threshold_calibration": 10,
                                  "development": 10, "final": 20}:
                raise AssertionError("known image budgets differ in seed %s for %s" % (seed, source))
        known_presentation_counts = {source: sum(row.get("identity_partition") == "known"
                                   and row["source_identity_id"] == source
                                   and row.get("presentation_role") == "known_presentation"
                                   for row in block_presentation)
                                   for source in selected_known}
        if set(known_presentation_counts.values()) != {20}:
            raise AssertionError("known presentation budget differs in seed %s" % seed)
        if any(not row.get("quality_valid") for row in block_rows
               if row.get("role") in {"train", "unknown_train"}):
            raise AssertionError("quality-invalid images entered a training role in seed %s" % seed)
        for role in UNKNOWN_ROLES:
            role_rows = (block_presentation if role == "unknown_presentation" else block_rows)
            role_field = "presentation_role" if role == "unknown_presentation" else "role"
            identity_ids = {row["source_identity_id"] for row in role_rows
                            if row.get("identity_partition") == "unknown" and row[role_field] == role}
            if len(identity_ids) != 50:
                raise AssertionError("seed %s role %s does not have 50 unknown identities" % (seed, role))
            if any(sum(row[role_field] == role and row["source_identity_id"] == identity for row in role_rows) != 6
                   for identity in identity_ids):
                raise AssertionError("unknown role %s does not have six groups per identity" % role)
        if len({row["source_identity_id"] for row in block_rows if row.get("identity_partition") == "known"} &
               {row["source_identity_id"] for row in block_rows if row.get("identity_partition") == "unknown"}):
            raise AssertionError("known/unknown identity overlap in seed %s" % seed)
        role_by_record = {row["record_id"]: row["role"] for row in block_rows}
        role_by_record.update({row["record_id"]: row["presentation_role"] for row in block_presentation})
        crossed = [(left, right) for left, right in pair_edges
                   if left in role_by_record and right in role_by_record and role_by_record[left] != role_by_record[right]]
        if crossed:
            raise AssertionError("near duplicate pHash candidate crosses roles in seed %s: %r" % (seed, crossed[:3]))
        hash_roles = {}
        for row in block_rows + block_presentation:
            role = row.get("role", row.get("presentation_role"))
            prior = hash_roles.setdefault(row["sha256"], role)
            if prior != role:
                raise AssertionError("exact duplicate checksum crosses roles in seed %s" % seed)
        for row in block_rows + block_presentation:
            path = safe_path(row, workspace)
            if not path.is_file():
                raise AssertionError("manifest path is missing: " + row["relative_path"])
            if row.get("source_sha256") != row.get("sha256"):
                raise AssertionError("source-to-prepared provenance hash changed")
        blocks.append({"root_seed": seed, "status": "pass", "primary_rows": len(block_rows),
                       "presentation_rows": len(block_presentation), "selection_fingerprint": actual_fingerprint,
                       "reproduced_unknown_assignment": True, "near_duplicate_cross_role_candidates": 0})

    # Loader behavior, conversion arithmetic, augmentation boundary, corrupt
    # bytes, and traversal protection are exercised once for each seed.
    train_row = next(row for row in primary_rows if row["role"] == "train")
    eval_row = next(row for row in primary_rows if row["role"] == "development")
    training_rows = [row for row in primary_rows if row["role"] in {"train", "unknown_train"}]
    balanced_images, balanced_labels, balanced_metadata = next(balanced_class_batches(
        training_rows, workspace, batch_size=24, seed=20261101, augment=False))
    class_counts = {index: int((balanced_labels == index).sum()) for index in range(len(CLASS_ORDER))}
    if balanced_images.shape != (24, 96, 96, 1) or class_counts != {index: 4 for index in range(6)}:
        raise AssertionError("balanced training sampler did not return four records per frozen class")
    if any(row.get("role") not in {"train", "unknown_train"} for row in balanced_metadata):
        raise AssertionError("balanced sampler accessed a nontraining role")
    pretraining_rows = read_jsonl(data_root / "pretraining_manifest.jsonl")
    pretraining_map = _read_json(data_root / "pretraining_class_map.json")
    pretraining_sources = {row["source_identity_id"] for row in pretraining_rows}
    unknown_sources = {row["source_identity_id"] for row in primary_rows + primary_presentation
                       if row.get("identity_partition") == "unknown"}
    if (not pretraining_rows or not pretraining_sources <= v3_roles["train"]
            or pretraining_sources & (known_set | unknown_sources)
            or pretraining_map["class_count"] != len(pretraining_sources)):
        raise AssertionError("optional pretraining pool overlaps a v4 evaluation identity or has an invalid class map")
    prebatch_a = next(balanced_identity_batches(pretraining_rows, workspace, identities_per_batch=32,
                                                seed=20261101, augment=False))
    prebatch_b = next(balanced_identity_batches(pretraining_rows, workspace, identities_per_batch=32,
                                                seed=20261101, augment=False))
    if (prebatch_a[0].shape != (32, 96, 96, 1)
            or len({row["source_identity_id"] for row in prebatch_a[2]}) != 32
            or [row["record_id"] for row in prebatch_a[2]] != [row["record_id"] for row in prebatch_b[2]]):
        raise AssertionError("identity-balanced pretraining loader is unbalanced or not reproducible")
    loader_blocks = []
    for seed in roots:
        raw = load_pixels(train_row, workspace, output="gray8", verify_hash=True, action="train")
        unit = load_pixels(train_row, workspace, output="float32_01", verify_hash=True, action="train")
        quantized = load_pixels(train_row, workspace, output="quantized", quantization=(1.0 / 255.0, -128), verify_hash=True, action="train")
        if raw.shape != (96, 96, 1) or unit.shape != raw.shape or quantized.dtype.name != "int8":
            raise AssertionError("loader output shape/dtype mismatch")
        if float(unit.min()) < 0 or float(unit.max()) > 1 or abs(float(unit[0, 0, 0]) - float(raw[0, 0, 0]) / 255.0) > 1e-6:
            raise AssertionError("grayscale intensity transform mismatch")
        if not (quantized == (raw.astype("int16") - 128).astype("int8")).all():
            raise AssertionError("quantization transform mismatch")
        augmented_a = load_pixels(train_row, workspace, output="gray8", augment=True, seed=seed, verify_hash=True, action="train")
        augmented_b = load_pixels(train_row, workspace, output="gray8", augment=True, seed=seed, verify_hash=True, action="train")
        if not (augmented_a == augmented_b).all():
            raise AssertionError("train augmentation is not deterministic for a root seed")
        _expect_raises(PermissionError, load_pixels, eval_row, workspace, augment=True, seed=seed, action="select")
        _expect_raises(ValueError, safe_path, {"relative_path": "../../etc/passwd"}, workspace)
        with tempfile.TemporaryDirectory(prefix="nicla-corrupt-", dir=str(data_root)) as temp_dir:
            source_path = safe_path(train_row, workspace)
            corrupt_path = Path(temp_dir) / source_path.name
            shutil.copyfile(source_path, corrupt_path)
            with corrupt_path.open("ab") as stream:
                stream.write(b"deliberate-corruption")
            corrupt_row = dict(train_row, relative_path=corrupt_path.relative_to(workspace).as_posix())
            _expect_raises(ValueError, load_pixels, corrupt_row, workspace, verify_hash=True, action="train")
        final_rows = [row for row in primary_rows if row["role"] in {"final", "unknown_final"}]
        _expect_raises(PermissionError, assert_role_access, final_rows, "evaluate_final")
        _expect_raises(PermissionError, select_rows, split_path, "evaluate_final",
                       dataset="survface_96", roles=("final", "unknown_final"))
        loader_blocks.append({"root_seed": seed, "status": "pass", "gray8_float_and_quantized": True,
                              "augmentation_train_only": True, "corrupt_hash_rejected": True,
                              "traversal_rejected": True, "final_gate_fail_closed": True})

    olivetti_rows = read_jsonl(data_root / "olivetti_sanity_manifest.jsonl")
    if not olivetti_rows or {row["dataset"] for row in olivetti_rows} != {"olivetti_96"}:
        raise AssertionError("Olivetti sanity split missing or mixed with SurvFace")
    olivetti_group_roles = {}
    for row in olivetti_rows:
        key = (row["source_identity_id"], row["source_group_id"])
        prior_role = olivetti_group_roles.setdefault(key, row["role"])
        if prior_role != row["role"]:
            raise AssertionError("Olivetti duplicate group crosses train/development/final")
    olivetti_source = {row["record_id"] for row in olivetti_rows}
    olivetti_duplicate_report = _read_json(workspace / "work/dataset-prep/reports/perceptual-duplicates.json")["datasets"]["olivetti_96"]
    cross_identity_exclusions = {record_id for pair in olivetti_duplicate_report["candidate_pairs"]
                                 if pair.get("left_identity_id") != pair.get("right_identity_id")
                                 for record_id in (pair["left_record_id"], pair["right_record_id"])}
    if olivetti_source & cross_identity_exclusions:
        raise AssertionError("Olivetti cross-label pHash candidates were not excluded")
    report = {"schema": "public-data-validation-1", "status": "pass", "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "seed_roots": roots, "sensitivity_blocks": blocks, "loader_blocks": loader_blocks,
              "primary_known_identity_map": roster["identity_map"], "primary_known_budget": counts,
              "source_to_prepared_copies_created": verification.get("prepared_image_copies_created"),
              "source_manifest_sha256": sha256_file(data_root / "source_manifest.jsonl"),
              "primary_split_sha256": sha256_file(split_path), "presentation_sha256": sha256_file(presentation_path),
              "olivetti_sanity_rows": len(olivetti_rows), "olivetti_unique_groups": len(olivetti_group_roles),
              "pretraining_rows": len(pretraining_rows), "pretraining_identities": len(pretraining_sources),
              "pretraining_loader_batch_size": len(prebatch_a[2]),
              "checks": {"ten_predeclared_seed_blocks": "pass", "metadata_only_roster_reproduction": "pass",
                         "unknown_identity_disjointness": "pass", "v3_final_access_control": "pass",
                         "near_duplicate_group_roles": "pass", "corrupt_and_escape_path_rejection": "pass",
                         "class_order_and_intensity_transforms": "pass", "training_only_augmentation": "pass",
                         "source_hash_provenance": "pass", "olivetti_dataset_separation": "pass",
                         "olivetti_group_safe_splits": "pass",
                         "quality_filter_restricted_to_training": "pass",
                         "balanced_six_class_training_batch": "pass",
                         "identity_balanced_pretraining_loader": "pass"},
              "balanced_batch_class_counts": {CLASS_ORDER[index]: count for index, count in class_counts.items()}}
    atomic_write_json(data_root / "validation_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    report = run(args.workspace.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
