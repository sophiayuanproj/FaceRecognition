#!/usr/bin/env python3
"""Prepare a few held-out pseudonymous validation crops for board comparison."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from common import MT_ROOT, ROOT, image_path, jsonl, load_manifest, sha256, write_json
from convert_eval import Runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=MT_ROOT / "runs/survface_mobile_v1")
    p.add_argument("--identities", type=int, default=2)
    p.add_argument("--images-per-identity", type=int, default=3)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    split = json.loads((run_dir / "split.json").read_text())
    used_pair_records = {
        record for pair in jsonl(run_dir / "validation_pairs.jsonl")
        for record in (pair["left_record_id"], pair["right_record_id"])
    }
    excluded = set(split.get("excluded_exact_duplicate_record_ids", []))
    _, by_record, rows, _ = load_manifest()
    by_identity = {}
    val = set(split["validation_identity_ids"])
    for row in rows:
        if (row["dataset"] == "survface_96" and row["identity_id"] in val
                and row["record_id"] not in used_pair_records | excluded):
            by_identity.setdefault(row["identity_id"], []).append(row)
    candidates = [identity for identity in sorted(by_identity)
                  if len(by_identity[identity]) >= args.images_per_identity]
    random.Random(split["seed"] + 91).shuffle(candidates)
    chosen = candidates[:args.identities]
    if len(chosen) < args.identities:
        raise ValueError("Not enough held-out identities with unused images for board comparison")
    output_dir = MT_ROOT / "board-eval-local"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for identity in chosen:
        selected = sorted(by_identity[identity], key=lambda row: row["record_id"])
        random.Random(split["seed"] + 92 + len(entries)).shuffle(selected)
        for row in selected[:args.images_per_identity]:
            name = row["record_id"] + ".pgm"
            with Image.open(image_path(row)) as source:
                source.convert("L").save(output_dir / name, format="PPM")
            entries.append({
                "file": name,
                "record_id": row["record_id"],
                "identity_id": identity,
            })
    manifest = {
        "data_note": "Temporary grayscale 96x96 board test crops; delete from /flash after comparison.",
        "split": "SurvFace training_set identity-disjoint validation identities",
        "pair_calibration_records_excluded": True,
        "images": entries,
    }
    write_json(output_dir / "board-eval-manifest.json", manifest)

    # Use the actual exported full-integer model for the host reference.
    model_path = ROOT / "FaceRecognition/models/identity_embedding_int8.tflite"
    runner = Runner(model_path, tflite=True)
    expected = []
    for entry in entries:
        with Image.open(output_dir / entry["file"]) as source:
            gray = np.asarray(source.convert("L"), dtype=np.uint8)
        vector = runner.run(gray)
        expected.append({"record_id": entry["record_id"], "identity_id": entry["identity_id"],
                         "vector": [float(value) for value in vector]})
    write_json(MT_ROOT / "reports/board-host-reference-embeddings.json", {
        "source": "exported full INT8 TFLite model on host CPU",
        "input_samples": len(expected),
        "identity_labels_are_pseudonymous": True,
        "embeddings": expected,
    })
    print("board_eval_images=%d validation_identities=%d" % (len(entries), len(chosen)))
    print("board_eval_manifest=" + str(output_dir / "board-eval-manifest.json"))


if __name__ == "__main__":
    main()
