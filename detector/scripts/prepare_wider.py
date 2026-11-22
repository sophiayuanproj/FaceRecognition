"""Finish B's staged WIDER FACE parsing and publish frozen scene manifests."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from common import box_to_frame, frame_box_to_detector, parse_wider, sha256_file


SEED = 20261100
TRAIN_GROUPS = [
    "37--Soccer", "51--Dresses", "15--Stock_Market", "42--Car_Racing",
    "1--Handshaking", "32--Worker_Laborer", "2--Demonstration", "12--Group",
    "61--Street_Battle", "52--Photographers", "23--Shoppers", "25--Soldier_Patrol",
    "56--Voter", "19--Couple", "10--People_Marching", "38--Tennis", "14--Traffic",
    "0--Parade", "36--Football", "18--Concerts", "29--Students_Schoolkids",
    "54--Rescue", "49--Greeting", "28--Sports_Fan", "48--Parachutist_Paratrooper",
    "21--Festival", "34--Baseball", "16--Award_Ceremony", "3--Riot",
    "31--Waiter_Waitress", "41--Swimming", "46--Jockey", "40--Gymnastics",
    "58--Hockey", "5--Car_Accident", "39--Ice_Skating", "33--Running",
    "9--Press_Conference", "57--Angler", "26--Soldier_Drilling", "43--Row_Boat",
    "11--Meeting", "30--Surgeons", "4--Dancing", "17--Ceremony", "13--Interview",
    "35--Basketball", "45--Balloonist", "8--Election_Campain",
]
DEV_GROUPS = ["6--Funeral", "7--Cheering", "20--Family_Group",
              "55--Sports_Coach_Trainer", "50--Celebration_Or_Party", "44--Aerobics"]
FINAL_GROUPS = ["24--Soldier_Firing", "47--Matador_Bullfighter", "22--Picnic",
                "59--people--driving--car", "53--Raid", "27--Spa"]


def one_record(root: Path, split: str, ann: dict, role: str) -> dict:
    path = root / f"WIDER_{split}" / "images" / ann["relative_path"]
    row = {**ann, "official_split": split, "role": role,
           "image_relative_path": str(path.relative_to(root)), "image_sha256": None,
           "decode_status": "pending", "width": None, "height": None,
           "exif_orientation": None, "face_boxes": []}
    try:
        row["image_sha256"] = sha256_file(path)
        with Image.open(path) as source:
            row["width"], row["height"] = source.size
            row["exif_orientation"] = source.getexif().get(274, 1)
            source.verify()
        row["decode_status"] = "pass"
        if ann["face_count"] == 0:
            row["role"] = "zero_box_excluded"
            row["exclusion_reason"] = "zero_box_placeholder_not_a_verified_negative"
            return row
        for box in ann["boxes"]:
            frame = box_to_frame(box["xywh"], row["width"], row["height"])
            detector = frame_box_to_detector(frame)
            x1, y1, x2, y2 = detector
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            visible = 0.0 <= center_x < 96.0 and 0.0 <= center_y < 96.0
            box.update({
                "xyxy_frame": [float(v) for v in frame],
                "xyxy_detector": [float(v) for v in detector],
                "center_detector_xy": [float(center_x), float(center_y)],
                "center_visible_in_detector_roi": bool(visible),
                "max_side_detector_px": float(max(x2 - x1, y2 - y1)),
                "max_side_frame_px": float(max(x2 - x1, y2 - y1) * 2.5),
            })
            row["face_boxes"].append(box)
        if len(row["face_boxes"]) != row["face_count"]:
            raise ValueError("face annotation count changed during transform")
    except Exception as exc:  # keep a bounded decode receipt; caller fails closed
        row["decode_status"] = "error"
        row["decode_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider-root", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    root = args.wider_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    split = {
        "schema": "job-c-wider-scene-split-1",
        "grouping": "WIDER event folder; no event folder crosses train/development/final",
        "selection_seed": SEED,
        "selection_basis": "event-folder names and scene counts only; no model scores or final predictions",
        "train_event_groups": TRAIN_GROUPS,
        "development_event_groups": DEV_GROUPS,
        "final_event_groups": FINAL_GROUPS,
        "roles": {
            "train": "WIDER official train partition in train_event_groups only",
            "development": "WIDER official validation partition in development_event_groups only",
            "final": "WIDER official validation partition in final_event_groups only; unopened by model selection",
        },
        "zero_box_policy": "All-zero placeholder rows are excluded, never used as negatives.",
    }
    if len(set(TRAIN_GROUPS + DEV_GROUPS + FINAL_GROUPS)) != len(TRAIN_GROUPS + DEV_GROUPS + FINAL_GROUPS):
        raise ValueError("Event groups overlap across frozen roles")
    (outdir / "wider_event_split.json").write_text(json.dumps(split, indent=2) + "\n")

    rows = []
    zero_rows = []
    annotation_count = Counter()
    for short, split_name in (("train", "train"), ("val", "val")):
        annotations, zero = parse_wider(root / "wider_face_split" / f"wider_face_{short}_bbx_gt.txt")
        zero_rows.extend({**x, "official_split": split_name} for x in zero)
        for ann in annotations:
            group = ann["event_group"]
            if split_name == "train" and group in TRAIN_GROUPS:
                role = "train"
            elif split_name == "val" and group in DEV_GROUPS:
                role = "development"
            elif split_name == "val" and group in FINAL_GROUPS:
                role = "final"
            else:
                role = "excluded_event_group_or_partition"
            rows.append(one_record(root, split_name, ann, role))
        annotation_count[split_name] = len(annotations) + len(zero)

    errors = [r for r in rows if r["decode_status"] != "pass"]
    if errors:
        (outdir / "wider_decode_errors.json").write_text(json.dumps(errors[:100], indent=2) + "\n")
        raise RuntimeError(f"WIDER decode errors: {len(errors)}; see wider_decode_errors.json")

    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["image_sha256"]].append(row)
    cross_role_dupes = []
    within_role_dupes = []
    role_priority = {"final": 0, "development": 1, "train": 2,
                     "excluded_event_group_or_partition": 3}
    for image_hash, dupes in by_hash.items():
        active = [r for r in dupes if r["role"] in {"train", "development", "final"}]
        roles = {r["role"] for r in active}
        if len(roles) > 1:
            cross_role_dupes.append({"sha256": image_hash, "roles": sorted(roles),
                                     "paths": [r["relative_path"] for r in active]})
            for row in active:
                row["role"] = "exact_duplicate_cross_role_excluded"
                row["exclusion_reason"] = "exact_sha256_appears_in_multiple_roles"
        elif len(active) > 1:
            within_role_dupes.append({"sha256": image_hash, "role": active[0]["role"],
                                      "paths": [r["relative_path"] for r in active]})
            for row in active[1:]:
                row["role"] = "exact_duplicate_within_role_excluded"
                row["exclusion_reason"] = "exact_sha256_duplicate"

    stats = {
        "schema": "job-c-wider-index-1",
        "archive_input_root": str(root),
        "annotation_record_counts_including_zero_boxes": dict(annotation_count),
        "zero_box_placeholder_count": len(zero_rows),
        "zero_box_placeholders": zero_rows,
        "decoded_nonzero_scene_records": len(rows),
        "decode_errors": len(errors),
        "exact_cross_role_duplicate_group_count": len(cross_role_dupes),
        "exact_within_role_duplicate_group_count": len(within_role_dupes),
        "source_split": split,
        "roles": {},
        "face_size_detector_input_px_by_role": {},
        "face_box_attributes": {},
    }
    for role in ("train", "development", "final", "excluded_event_group_or_partition",
                 "exact_duplicate_cross_role_excluded", "exact_duplicate_within_role_excluded"):
        selected = [r for r in rows if r["role"] == role]
        sizes = Counter()
        attr = Counter()
        faces_total = faces_visible = faces_invalid = 0
        for r in selected:
            faces_total += r["face_count"]
            for b in r["face_boxes"]:
                attr[f"blur_{b['blur']}"] += 1
                attr[f"occlusion_{b['occlusion']}"] += 1
                attr[f"invalid_{b['invalid']}"] += 1
                if b["invalid"]:
                    faces_invalid += 1
                if b["center_visible_in_detector_roi"]:
                    faces_visible += 1
                    side = b["max_side_detector_px"]
                    bucket = "<4" if side < 4 else "4-7" if side < 8 else "8-11" if side < 12 else "12-19" if side < 20 else "20-31" if side < 32 else "32+"
                    sizes[bucket] += 1
        stats["roles"][role] = {"scenes": len(selected), "annotated_faces": faces_total,
                                "face_centers_in_deployment_roi": faces_visible,
                                "invalid_attribute_faces": faces_invalid}
        stats["face_size_detector_input_px_by_role"][role] = dict(sizes)
        stats["face_box_attributes"][role] = dict(attr)

    role_files = {"train": "wider_train_manifest.jsonl",
                  "development": "wider_development_manifest.jsonl",
                  "final": "wider_final_manifest.jsonl"}
    full_path = outdir / "wider_scene_manifest.jsonl"
    with full_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    for role, filename in role_files.items():
        with (outdir / filename).open("w") as f:
            for row in rows:
                if row["role"] == role:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
    stats["exact_cross_role_duplicate_group_details"] = cross_role_dupes
    stats["exact_within_role_duplicate_group_details"] = within_role_dupes
    (outdir / "wider_index_report.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scene_manifest": str(full_path), "roles": stats["roles"],
                      "zero_box_placeholder_count": len(zero_rows),
                      "cross_role_sha256_duplicates": len(cross_role_dupes)}, indent=2))


if __name__ == "__main__":
    main()
