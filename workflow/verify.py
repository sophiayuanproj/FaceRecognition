"""Verify the frozen training inputs, with optional restored-image checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "work/project-finish-v4"
WEIGHTS = {
    "edgeimpulse_mobilenetv1_0.1_96_grayscale.h5": "401cc0929ae5592f53ec25c5f6cd9228e5bf905b0065644f1c33483b5b2589c3",
    "edgeimpulse_mobilenetv1_0.25_96_grayscale.h5": "9b2f73e58d3ea8c86ff04b38cd1ea26235b73e3270c9b87afe6633dd58e94eff",
}
ROLES = {"train", "unknown_train", "stopping", "threshold_calibration",
         "unknown_calibration", "development", "unknown_development"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true", help="Also hash the restored primary dataset images")
    args = parser.parse_args()
    for filename, expected in WEIGHTS.items():
        path = STUDY / "models/pretrained" / filename
        if not path.is_file() or digest(path) != expected:
            raise SystemExit("Missing or changed pretrained weights: " + str(path.relative_to(ROOT)))
    matrix = json.loads((STUDY / "D_MATRIX_v1.json").read_text())
    seeds = matrix["training_roots"]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise SystemExit("Expected ten distinct registered training seeds")
    search = json.loads((STUDY / "runs/D/development_search/DEVELOPMENT_SEARCH.json").read_text())
    if not search.get("selected"):
        raise SystemExit("Missing recorded development selection")
    rows = [json.loads(line) for line in (STUDY / "data/primary_split_manifest.jsonl").read_text().splitlines()]
    checked = 0
    for row in rows:
        if row.get("role") not in ROLES:
            raise SystemExit("Unexpected role in the distributed development manifest")
        if args.data:
            path = (ROOT / row["relative_path"]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != row["sha256"]:
                raise SystemExit("Missing or changed dataset image. Run workflow/prepare_data.py with the matching ZIP.")
            checked += 1
    print(json.dumps({"status": "verified", "training_seeds": seeds,
                      "primary_manifest_rows": len(rows), "images_checked": checked,
                      "pretrained_checkpoints_checked": len(WEIGHTS)}, indent=2))


if __name__ == "__main__":
    main()
