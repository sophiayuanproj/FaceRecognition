"""Create an optional fresh-embedding pretraining manifest from v3 train IDs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json

from .curate_public_data import _UnionFind, _read_json, _read_jsonl, _v3_identity_sources, _write_jsonl
from .fs import atomic_write_json


def run(workspace: Path) -> dict:
    data_root = workspace / "work/project-finish-v4/data"
    rows = _read_jsonl(data_root / "source_manifest.jsonl")
    v3_roles = _v3_identity_sources(workspace)
    v3_train_ids = v3_roles["train"]
    selected = [row for row in rows if row.get("dataset") == "survface_96"
                and row.get("source_identity_id") in v3_train_ids
                and row.get("access_state") == "reverified_this_job"
                and row.get("quality_valid")]
    digest_sources = defaultdict(set)
    for row in rows:
        if row.get("dataset") == "survface_96" and row.get("source_identity_id") and row.get("sha256"):
            digest_sources[row["sha256"]].add(row["source_identity_id"])
    contradictory_hashes = {digest for digest, identity_ids in digest_sources.items() if len(identity_ids) > 1}
    selected = [row for row in selected if row["sha256"] not in contradictory_hashes]
    ids = {row["source_identity_id"] for row in selected}
    if not ids or ids & (v3_roles["development"] | v3_roles["calibration"]):
        raise ValueError("pretraining pool overlaps a v3 development/calibration identity or is empty")

    record_ids = {row["record_id"] for row in selected}
    source_by_record = {row["record_id"]: row["source_identity_id"] for row in selected}
    source_by_pseudonym = {row["identity_id"]: row["source_identity_id"] for row in selected}
    uf_by_id = defaultdict(_UnionFind)
    digest_rows = defaultdict(list)
    for row in selected:
        digest_rows[(row["source_identity_id"], row["sha256"])].append(row["record_id"])
    for (source, _digest), members in digest_rows.items():
        for record_id in members[1:]:
            uf_by_id[source].union(members[0], record_id)
    duplicate_report = _read_json(workspace / "work/dataset-prep/reports/perceptual-duplicates.json")
    for pair in duplicate_report["datasets"]["survface_96"]["candidate_pairs"]:
        left, right = pair.get("left_record_id"), pair.get("right_record_id")
        if left not in record_ids or right not in record_ids:
            continue
        source = source_by_record[left]
        right_source = source_by_record[right]
        if source == right_source:
            uf_by_id[source].union(left, right)
    groups = defaultdict(list)
    for row in selected:
        groups[(row["source_identity_id"], uf_by_id[row["source_identity_id"]].find(row["record_id"]))].append(row)
    class_map = {identity_id: index for index, identity_id in enumerate(sorted({row["identity_id"] for row in selected}))}
    best_by_class = source_by_pseudonym
    output = []
    for (source, _group), members in groups.items():
        pseudonym = members[0]["identity_id"]
        representative = max(members, key=lambda row: (float(row.get("quality_score", 0.0)),
                                 hashlib.sha256(("v4-pretrain:%s:%s" % (source, row["record_id"])).encode()).hexdigest()))
        output.append({**representative,"role":"train","identity_partition":"pretraining",
                       "label_id":pseudonym,"class_index":class_map[pseudonym],
                       "pretraining_source_identity_id":source,
                       "pretraining_selection":"one quality-valid, exact/pHash group representative; v3 train identities only"})
    output.sort(key=lambda row: (row["class_index"], row["record_id"]))
    _write_jsonl(data_root / "pretraining_manifest.jsonl", output)
    class_rows = [{"class_index": index, "pseudonymous_identity_id": identity_id,
                   "source_identity_id": best_by_class[identity_id]} for identity_id, index in class_map.items()]
    class_rows.sort(key=lambda row: row["class_index"])
    atomic_write_json(data_root / "pretraining_class_map.json", {
        "schema":"pretraining-class-map-1","pool":"SurvFace v3 train-role identities",
        "class_count":len(class_rows),"identity_map":class_rows,
        "historical_exposure":"These identities were used in v3 training. This manifest is for a fresh v4 training initialization; reuse of a v3 checkpoint is a diagnostic arm, not a fresh independent comparison."})
    summary = {"schema":"pretraining-manifest-summary-1","status":"pass",
               "identity_count":len(ids),"record_count":len(output),
               "all_rows_role_train":all(row["role"]=="train" for row in output),
               "v3_development_calibration_overlap":0,
               "ambiguous_cross_identity_exact_hashes_excluded":len(contradictory_hashes),
               "source_image_copies_created":False,
               "source_hashes_verified_during_curation":all(row.get("verified_sha256")==row.get("source_sha256") for row in output),
               "initialization_policy":"train new weights; starting from a historical v3 checkpoint changes this to a diagnostic arm"}
    atomic_write_json(data_root / "pretraining_manifest_summary.json", summary)
    access_path = data_root / "source_access_exposure_ledger.json"
    access = _read_json(access_path)
    access["v4_fresh_pretraining_pool"] = "v3 train-role identities only; use a new weight initialization"
    access["v3_train_identities_in_fresh_pretraining_manifest"] = len(ids)
    access["v3_train_identity_overlap_with_v4_evaluation_and_unknown_roles"] = 0
    atomic_write_json(access_path, access)
    curation_summary_path = data_root / "curation_summary.json"
    curation_summary = _read_json(curation_summary_path)
    curation_summary["pretraining_manifest_rows"] = len(output)
    curation_summary["pretraining_manifest_identity_count"] = len(ids)
    atomic_write_json(curation_summary_path, curation_summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    result = run(args.workspace.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
