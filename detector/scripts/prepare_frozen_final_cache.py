"""Materialize C's held-out WIDER/PASS scenes after the development freeze only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import dump_json, sha256_file
from prepare_training_cache import combine, scene_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider-root", type=Path, required=True)
    ap.add_argument("--wider-manifest", type=Path, required=True)
    ap.add_argument("--pass-manifest", type=Path, required=True)
    ap.add_argument("--freeze-receipt", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    freeze = json.loads(args.freeze_receipt.read_text(encoding="utf-8"))
    if not freeze.get("final_access_authorized_after_development_freeze"):
        raise ValueError("A hashed development freeze receipt is required before loading final scenes")
    wider = scene_rows(args.wider_manifest, args.wider_root, "final")
    passed = scene_rows(args.pass_manifest, Path("."), "final", negative=True)
    gray, rgb, window, targets, metadata = combine([wider, passed])
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.outdir / "final_cache.npz"
    np.savez_compressed(cache_path, gray96=gray, rgb96=rgb, gray_window=window, targets=targets)
    metadata_path = args.outdir / "final_scene_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as stream:
        for row in metadata:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"schema": "job-c-frozen-final-cache-1", "development_freeze_sha256": sha256_file(args.freeze_receipt),
              "development_freeze_path": str(args.freeze_receipt), "final_scenes_loaded_after_freeze": True,
              "wider_positive_scenes": len(wider[0]), "pass_real_no_face_scenes": len(passed[0]),
              "total_scenes": int(gray.shape[0]), "final_cache_sha256": sha256_file(cache_path),
              "final_metadata_sha256": sha256_file(metadata_path), "model_selection_or_threshold_tuning": False}
    dump_json(args.outdir / "final_cache_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
