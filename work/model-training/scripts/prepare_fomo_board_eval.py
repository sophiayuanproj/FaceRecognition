#!/usr/bin/env python3
"""Prepare private, pseudonymous held-out face scenes for the on-board FOMO path."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from common import MT_ROOT, image_path, jsonl, load_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=MT_ROOT / "runs/survface_supcon_v2_holdout-clean")
    parser.add_argument("--identities", type=int, default=2)
    parser.add_argument("--images-per-identity", type=int, default=3)
    parser.add_argument("--background-u8", type=int, default=127)
    parser.add_argument("--scene-side", type=int, default=160)
    args = parser.parse_args()
    if not 0 <= args.background_u8 <= 255:
        raise ValueError("background-u8 must be in [0,255]")

    run_dir = args.run_dir.resolve()
    split = json.loads((run_dir / "split.json").read_text())
    used_pair_records = {
        record for pair in jsonl(run_dir / "validation_pairs.jsonl")
        for record in (pair["left_record_id"], pair["right_record_id"])
    }
    excluded = set(split.get("excluded_exact_duplicate_record_ids", []))
    _, _, rows, _ = load_manifest()
    val_ids = set(split["validation_identity_ids"])
    by_identity = {}
    for row in rows:
        if (row["dataset"] == "survface_96" and row["identity_id"] in val_ids
                and row["record_id"] not in used_pair_records | excluded):
            by_identity.setdefault(row["identity_id"], []).append(row)
    candidates = [identity for identity in sorted(by_identity)
                  if len(by_identity[identity]) >= args.images_per_identity]
    random.Random(split["seed"] + 411).shuffle(candidates)
    chosen = candidates[:args.identities]
    if len(chosen) < args.identities:
        raise ValueError("Not enough unused validation identities for FOMO board evaluation")

    output_dir = MT_ROOT / "fomo-board-eval-local"
    output_dir.mkdir(parents=True, exist_ok=True)
    face_side = 96
    scene_side = args.scene_side
    if scene_side < face_side or scene_side > 200:
        raise ValueError("scene-side must be between 96 and 200")
    offset = (scene_side - face_side) // 2
    entries = []
    for identity in chosen:
        selected = sorted(by_identity[identity], key=lambda row: row["record_id"])
        random.Random(split["seed"] + 412 + len(entries)).shuffle(selected)
        for row in selected[:args.images_per_identity]:
            with Image.open(image_path(row)) as source:
                face = source.convert("L")
                if face.size != (face_side, face_side):
                    raise ValueError("Expected supplied 96x96 grayscale face crops")
                scene = Image.new("RGB", (scene_side, scene_side),
                                  (args.background_u8,) * 3)
                scene.paste(face.convert("RGB"), (offset, offset))
            # These scenes contain a neutral background plus an RGB-encoded
            # face crop. Keep the .ppm extension so OpenMV loads RGB565 rather
            # than treating the PPM payload as a grayscale PGM.
            filename = row["record_id"] + ".ppm"
            scene.save(output_dir / filename, format="PPM")
            entries.append({
                "file": filename,
                "record_id": row["record_id"],
                "identity_id": identity,
                "source_face_roi": [offset, offset, face_side, face_side],
            })

    manifest = {
        "data_note": "Temporary local-only grayscale scenes made by centering held-out SurvFace face crops on a neutral background; no source image is modified or redistributed.",
        "split": "identity-disjoint SurvFace training_set validation identities",
        "pair_calibration_records_excluded": True,
        "scene_size": [scene_side, scene_side],
        "background_u8": args.background_u8,
        "images": entries,
    }
    write_json(output_dir / "fomo-board-eval-manifest.json", manifest)
    print("fomo_board_eval_images=%d identities=%d" % (len(entries), len(chosen)))
    print("fomo_board_eval_manifest=" + str(output_dir / "fomo-board-eval-manifest.json"))


if __name__ == "__main__":
    main()
