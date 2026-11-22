#!/usr/bin/env python3
"""Verify exact training-only representative pools used by compression exports."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
MT_ROOT = HERE.parents[1]
sys.path.insert(0, str(MT_ROOT / "scripts"))

from common import (  # noqa: E402
    image_path, load_manifest, remove_exact_duplicate_conflicts, select_training_rows, sha256,
)
import train_embedder  # noqa: E402


RUN_DIR = MT_ROOT / "runs/survface_supcon_v2_holdout-clean"
OUT_DIR = MT_ROOT / "compression/strategy_sweep_20261022"


def main():
    split = json.loads((RUN_DIR / "split.json").read_text())
    config = json.loads((RUN_DIR / "train_config.json").read_text())
    summary = json.loads((RUN_DIR / "run_summary.json").read_text())
    all_rows, _, raw_rows, _ = load_manifest(include_olivetti=False)
    unlabeled = [r for r in all_rows if r["dataset"] == "survface_96" and not r["identity_id"]]
    clean, audit = remove_exact_duplicate_conflicts(raw_rows, unlabeled)
    selected = select_training_rows(
        clean, split["train_identity_ids"],
        int(config["max_images_per_train_identity"]), int(split["seed"]),
    )
    if len(selected) != int(summary["training_image_count"]):
        raise RuntimeError("selected training record count differs from original run")

    random_order = list(selected)
    random.Random(int(split["seed"])).shuffle(random_order)
    post_training_rows = random_order[:min(512, len(random_order))]

    groups = train_embedder.grouped_rows(
        clean, split["train_identity_ids"], set(split["excluded_exact_duplicate_record_ids"]),
    )
    _, _, tfmot_rows = train_embedder.load_training_images(
        groups, int(config["max_images_per_train_identity"]), int(split["seed"]),
    )
    np_order = list(tfmot_rows)
    np.random.default_rng(int(split["seed"])).shuffle(np_order)
    tfmot_rows = np_order[:min(512, len(np_order))]

    validation_ids = set(split["validation_identity_ids"])
    result = {
        "protocol": "exact selected training records after exact duplicate exclusion and per-identity cap",
        "run_dir": str(RUN_DIR.relative_to(ROOT)),
        "training_record_count": len(selected),
        "training_identity_count": len(split["train_identity_ids"]),
        "validation_identity_count": len(validation_ids),
        "unlabeled_holdout_records_used_for_exact-hash_exclusion_only": len(unlabeled),
        "final_pair_annotations_loaded": False,
        "olivetti_loaded": False,
        "duplicate_audit": {k: v for k, v in audit.items() if k != "excluded_record_ids"},
        "pools": {},
    }
    for name, rows in (
        ("post_training_quantization", post_training_rows),
        ("pruning_clustering_qat", tfmot_rows),
    ):
        records = []
        mismatches = []
        for row in rows:
            digest = sha256(image_path(row))
            if digest != row["sha256"]:
                mismatches.append(row["record_id"])
            records.append({
                "record_id": row["record_id"],
                "identity_id": row["identity_id"],
                "image_sha256": digest,
            })
        selected_identity_ids = {row["identity_id"] for row in rows}
        result["pools"][name] = {
            "representative_count": len(rows),
            "record_identity_sha256_manifest": hashlib.sha256(json.dumps(
                records, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
            "training_selection_sha256": hashlib.sha256("".join(
                row["record_id"] + "\t" + row["sha256"] + "\n" for row in selected
            ).encode("utf-8")).hexdigest(),
            "manifest": records,
            "image_hash_mismatches": mismatches,
            "validation_identity_overlap_count": len(selected_identity_ids & validation_ids),
            "source": "training identities only; no verification-pair labels",
        }
    if any(pool["image_hash_mismatches"] or pool["validation_identity_overlap_count"]
           for pool in result["pools"].values()):
        raise RuntimeError("representative-pool audit failed")
    out = OUT_DIR / "representative_pool_audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "audit": "passed",
        "training_records": len(selected),
        "validation_identity_overlap": 0,
        "post_training_q_sha256": result["pools"]["post_training_quantization"]["record_identity_sha256_manifest"],
        "tfmot_q_sha256": result["pools"]["pruning_clustering_qat"]["record_identity_sha256_manifest"],
        "report": str(out.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

