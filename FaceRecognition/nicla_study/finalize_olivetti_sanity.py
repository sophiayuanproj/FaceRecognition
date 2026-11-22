"""Publish an exact/pHash-safe, separate Olivetti 60/20/20 sanity split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .curate_public_data import _UnionFind, _read_json, _read_jsonl, _write_jsonl
from .fs import atomic_write_json


def run(workspace: Path) -> dict:
    data_root = workspace / "work/project-finish-v4/data"
    source_rows = [row for row in _read_jsonl(data_root / "source_manifest.jsonl")
                   if row.get("dataset") == "olivetti_96" and row.get("access_state") == "reverified_this_job"]
    prior = _read_json(workspace / "work/dataset-prep/reports/perceptual-duplicates.json")["datasets"]["olivetti_96"]
    identities = {row["record_id"]: row["source_identity_id"] for row in source_rows}
    digest_ids = defaultdict(set)
    for row in source_rows:
        digest_ids[row["sha256"]].add(row["source_identity_id"])
    contradictory_hashes = {digest for digest, ids in digest_ids.items() if len(ids) > 1}
    excluded_records = {row["record_id"] for row in source_rows if row["sha256"] in contradictory_hashes}
    edges_by_record = defaultdict(list)
    cross_identity_pairs = []
    for pair in prior["candidate_pairs"]:
        left, right = pair["left_record_id"], pair["right_record_id"]
        left_id, right_id = identities.get(left), identities.get(right)
        if not left_id or not right_id:
            continue
        if left_id != right_id:
            cross_identity_pairs.append(pair)
            excluded_records.update((left, right))
        else:
            edges_by_record[left].append(right)
            edges_by_record[right].append(left)
    usable = [row for row in source_rows if row["record_id"] not in excluded_records]
    usable_ids = {row["record_id"] for row in usable}
    by_identity = defaultdict(list)
    for row in usable:
        by_identity[row["source_identity_id"]].append(row)
    output = []
    identity_counts = {}
    for identity, rows in sorted(by_identity.items()):
        uf = _UnionFind()
        digest_rows = defaultdict(list)
        for row in rows:
            digest_rows[row["sha256"]].append(row["record_id"])
        for items in digest_rows.values():
            for record_id in items[1:]:
                uf.union(items[0], record_id)
        for left in usable_ids:
            if identities.get(left) != identity:
                continue
            for right in edges_by_record.get(left, []):
                if right in usable_ids and identities.get(right) == identity:
                    uf.union(left, right)
        groups = defaultdict(list)
        for row in rows:
            groups[uf.find(row["record_id"])].append(row)
        ordered = sorted(groups.items(), key=lambda item: hashlib.sha256(
                         ("nicla-v4:20261201:olivetti:%s:%s" % (identity, item[0])).encode()).hexdigest())
        target_train = round(len(ordered) * 0.6)
        target_development = round(len(ordered) * 0.2)
        selected_roles = (["train"] * target_train + ["development"] * target_development
                          + ["final"] * (len(ordered) - target_train - target_development))
        if len(selected_roles) != len(ordered):
            raise AssertionError("Olivetti 60/20/20 group assignment did not conserve groups")
        # Put the best available groups in train only; difficult groups remain
        # eligible for development and final without quality-based exclusion.
        # Reorder group entries so train gets the highest-score groups; keep the
        # remaining group order deterministic and preserve the split counts.
        train_groups = sorted(ordered,
                              key=lambda item: (-max(float(row.get("quality_score", 0.0)) for row in item[1]), item[0]))[:target_train]
        train_ids = {group_id for group_id, _members in train_groups}
        other_groups = [item for item in ordered if item[0] not in train_ids]
        selected_groups = train_groups + other_groups
        for (group_id, members), role in zip(selected_groups, selected_roles):
            for row in members:
                output.append({**row, "role": role, "identity_partition": "known",
                               "label_id": row["identity_id"], "source_group_id": group_id,
                               "evidence_tier": "small_data_sanity_only"})
        identity_counts[identity] = {"unique_exact_phash_groups": len(ordered),
                                     "train_groups": target_train, "development_groups": target_development,
                                     "final_groups": len(ordered) - target_train - target_development,
                                     "usable_source_rows": len(rows)}
    _write_jsonl(data_root / "olivetti_sanity_manifest.jsonl", output)
    summary = {"schema":"olivetti-sanity-summary-1","status":"pass",
               "dataset":"olivetti_96","identity_count":len(identity_counts),"record_count":len(output),
               "identity_group_counts":identity_counts,"exact_cross_identity_hash_groups_excluded":len(contradictory_hashes),
               "cross_identity_phash_candidate_pairs_excluded":len(cross_identity_pairs),
               "excluded_source_record_count":len(excluded_records),
               "split_policy":"exact duplicates and same-identity pHash<=6 groups stay together; cross-identity exact/pHash candidate records are excluded; target group split is 60/20/20; quality affects training order only",
               "source_archive_hash":_read_json(workspace / "work/dataset-prep/reports/dataset-summary.json")["datasets"]["olivetti_96"]["archive_sha256"]}
    atomic_write_json(data_root / "olivetti_sanity_summary.json", summary)
    curation = _read_json(data_root / "curation_summary.json")
    curation["olivetti_rows"] = len(output)
    atomic_write_json(data_root / "curation_summary.json", curation)
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
