"""Render local, pseudonym-only review sheets for the frozen public-image roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .data_pipeline import read_jsonl, safe_path
from .fs import sha256_file, atomic_write_json


def _tile(row: dict, workspace: Path, title: str, width: int = 190, height: int = 160) -> Image.Image:
    path = safe_path(row, workspace)
    if sha256_file(path) != row.get("source_sha256", row.get("sha256")):
        raise ValueError("source hash changed while rendering contact sheet")
    with Image.open(path) as image:
        image.load()
        face = image.convert("L").resize((128, 128), Image.Resampling.LANCZOS).convert("RGB")
    tile = Image.new("RGB", (width, height), "white")
    tile.paste(face, ((width - 128) // 2, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((5, 130), title[:29], fill="black")
    return tile


def run(workspace: Path) -> dict:
    data_root = workspace / "work/project-finish-v4/data"
    split_rows = read_jsonl(data_root / "primary_split_manifest.jsonl")
    presentation_rows = read_jsonl(data_root / "presentation_manifest.jsonl")
    roster = json.loads((data_root / "roster.json").read_text(encoding="utf-8"))
    out_dir = data_root / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []

    def choose(rows: list[dict], key: str, count: int) -> list[dict]:
        return sorted(rows, key=lambda row: hashlib.sha256((key + ":" + row["record_id"]).encode()).hexdigest())[:count]

    # One per selected known identity, with all five model roles and separate
    # presentation examples. Selection is hash-stable and not a quality filter.
    for known in roster["identity_map"]:
        label = known["pseudonym"]
        source = known["source_identity_id"]
        samples = []
        for role in ("train", "stopping", "threshold_calibration", "development", "final"):
            role_rows = [row for row in split_rows if row["source_identity_id"] == source and row["role"] == role]
            samples.extend((row, role) for row in choose(role_rows, label + ":" + role, min(4, len(role_rows))))
        role_rows = [row for row in presentation_rows if row["source_identity_id"] == source]
        samples.extend((row, "presentation") for row in choose(role_rows, label + ":presentation", min(4, len(role_rows))))
        tiles = []
        for row, role in samples:
            cam = row.get("camera_source_group") or "cam?"
            score = row.get("quality_score")
            title = "%s | %s | %s | q=%s" % (label, role, cam, "na" if score is None else "%.2f" % score)
            tiles.append(_tile(row, workspace, title))
        sheet = Image.new("RGB", (190 * 4, 160 * ((len(tiles) + 3) // 4)), "#e7e7e7")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 4) * 190, (index // 4) * 160))
        path = out_dir / (label + ".png")
        sheet.save(path, format="PNG", optimize=True)
        manifests.append({"file": path.name, "selected_identity": label, "tile_count": len(tiles),
                          "roles": sorted({role for _, role in samples})})

    # A separate sheet samples OTHER identities with stable role-local aliases;
    # public source folder IDs and image filenames are omitted from the preview.
    unknown_samples = []
    for role in ("unknown_train", "unknown_calibration", "unknown_development", "unknown_final"):
        identities = sorted({row["source_identity_id"] for row in split_rows
                             if row.get("identity_partition") == "unknown" and row["role"] == role},
                            key=lambda value: hashlib.sha256((role + value).encode()).hexdigest())[:8]
        for ordinal, source in enumerate(identities, 1):
            rows = [row for row in split_rows if row["source_identity_id"] == source and row["role"] == role]
            for row in choose(rows, role + source, min(2, len(rows))):
                unknown_samples.append((row, "%s U%02d | OTHER" % (role.replace("unknown_", ""), ordinal)))
    identities = sorted({row["source_identity_id"] for row in presentation_rows
                         if row.get("identity_partition") == "unknown" and row["presentation_role"] == "unknown_presentation"},
                        key=lambda value: hashlib.sha256(("unknown_presentation" + value).encode()).hexdigest())[:8]
    for ordinal, source in enumerate(identities, 1):
        rows = [row for row in presentation_rows if row["source_identity_id"] == source]
        for row in choose(rows, "unknown_presentation" + source, min(2, len(rows))):
            unknown_samples.append((row, "presentation U%02d | OTHER" % ordinal))
    unknown_tiles = [_tile(row, workspace, title) for row, title in unknown_samples]
    unknown_sheet = Image.new("RGB", (190 * 4, 160 * ((len(unknown_tiles) + 3) // 4)), "#e7e7e7")
    for index, tile in enumerate(unknown_tiles):
        unknown_sheet.paste(tile, ((index % 4) * 190, (index // 4) * 160))
    unknown_path = out_dir / "OTHER_review.png"
    unknown_sheet.save(unknown_path, format="PNG", optimize=True)
    manifests.append({"file": unknown_path.name, "selected_identity": "OTHER", "tile_count": len(unknown_tiles),
                      "roles": ["unknown_train", "unknown_calibration", "unknown_development", "unknown_final", "unknown_presentation"]})
    result = {"schema": "contact-sheet-review-1", "status": "rendered_for_local_visual_review",
              "scope": "pseudonym-only previews of already frozen source records; no source image selection was changed",
              "sheets": manifests}
    atomic_write_json(out_dir / "contact_sheets.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    print(json.dumps(run(args.workspace.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
