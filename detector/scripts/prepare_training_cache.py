"""Build exact 96x96 grayscale detector tensors from B's scene manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import (camera_window_from_sensor_frame, dump_json, frame_to_detector_input,
                    source_to_frame, sha256_file)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def target_grid(row: dict) -> np.ndarray:
    """FOMO centroid target: background channel 0, face channel 1, stride 8."""
    target = np.zeros((12, 12, 2), dtype=np.uint8)
    target[..., 0] = 1
    for box in row.get("face_boxes", []):
        if not box.get("center_visible_in_detector_roi"):
            continue
        cx, cy = box["center_detector_xy"]
        gx, gy = int(float(cx) // 8), int(float(cy) // 8)
        if 0 <= gx < 12 and 0 <= gy < 12:
            target[gy, gx, 0] = 0
            target[gy, gx, 1] = 1
    return target


def materialize(row: dict, image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with Image.open(image_path) as raw:
        orientation = raw.getexif().get(274, 1)
        if orientation != 1 and row.get("face_boxes"):
            raise ValueError(f"EXIF orientation requires annotation remapping: {image_path} ({orientation})")
        raw.verify()
    with Image.open(image_path) as raw:
        rgb_frame = source_to_frame(raw.copy(), mode="rgb")
    with Image.open(image_path) as raw:
        gray_frame = source_to_frame(raw.copy(), mode="gray")
    gray_window = camera_window_from_sensor_frame(gray_frame)
    gray96 = frame_to_detector_input(gray_frame, mode="gray")
    rgb96 = frame_to_detector_input(rgb_frame, mode="rgb")
    return gray96, rgb96, gray_window, target_grid(row)


def scene_rows(manifest: Path, image_root: Path, role: str, negative: bool = False) -> tuple[list, list]:
    xs, rgbs, frames, ys, metadata = [], [], [], [], []
    for row in read_jsonl(manifest):
        if row.get("role") != role:
            continue
        if negative:
            if row.get("download_status") != "pass":
                continue
            image_path = Path(row["local_image_path"])
            target = np.zeros((12, 12, 2), dtype=np.uint8)
            target[..., 0] = 1
        else:
            image_path = image_root / row["image_relative_path"]
            target = target_grid(row)
        gray96, rgb96, window, _ = materialize(row, image_path)
        xs.append(gray96)
        rgbs.append(rgb96)
        frames.append(window)
        ys.append(target)
        meta = dict(row)
        meta["local_image_path"] = str(image_path)
        meta["local_image_sha256"] = row.get("local_image_sha256") or row.get("image_sha256") or sha256_file(image_path)
        meta["is_real_no_face_negative"] = bool(negative)
        metadata.append(meta)
    return (xs, rgbs, frames, ys, metadata)


def combine(groups: list[tuple[list, list, list, list, list]]) -> tuple:
    values = [[], [], [], [], []]
    for group in groups:
        for dst, src in zip(values, group):
            dst.extend(src)
    return (np.stack(values[0]).astype(np.uint8), np.stack(values[1]).astype(np.uint8),
            np.stack(values[2]).astype(np.uint8), np.stack(values[3]).astype(np.uint8), values[4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider-root", type=Path, required=True)
    ap.add_argument("--wider-manifest-dir", type=Path, required=True)
    ap.add_argument("--pass-manifest", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--wider-only", action="store_true")
    ap.add_argument("--reuse-wider-cache", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    wider_cache_path = args.outdir / "wider_only_cache.npz"
    if args.reuse_wider_cache:
        with np.load(wider_cache_path) as cache:
            wider_train_x, wider_train_y = cache["train_gray96"], cache["train_targets"]
            wider_dev_x, wider_dev_rgb, wider_dev_window, wider_dev_y = (
                cache["development_gray96"], cache["development_rgb96"],
                cache["development_gray_window"], cache["development_targets"])
        train_pos_meta = read_jsonl(args.outdir / "wider_train_scene_metadata.jsonl")
        dev_pos_meta = read_jsonl(args.outdir / "wider_development_scene_metadata.jsonl")
        train_pos_count, dev_pos_count = len(train_pos_meta), len(dev_pos_meta)
    else:
        train_pos = scene_rows(args.wider_manifest_dir / "wider_train_manifest.jsonl", args.wider_root, "train")
        dev_pos = scene_rows(args.wider_manifest_dir / "wider_development_manifest.jsonl", args.wider_root, "development")
        wider_train_x, wider_train_y = np.stack(train_pos[0]), np.stack(train_pos[3])
        wider_dev_x, wider_dev_rgb, wider_dev_window, wider_dev_y = (
            np.stack(dev_pos[0]), np.stack(dev_pos[1]), np.stack(dev_pos[2]), np.stack(dev_pos[3]))
        train_pos_meta, dev_pos_meta = train_pos[4], dev_pos[4]
        train_pos_count, dev_pos_count = len(train_pos[0]), len(dev_pos[0])
        np.savez_compressed(wider_cache_path, train_gray96=wider_train_x, train_targets=wider_train_y,
                            development_gray96=wider_dev_x, development_rgb96=wider_dev_rgb,
                            development_gray_window=wider_dev_window, development_targets=wider_dev_y)
        for name, rows in (("wider_train", train_pos_meta), ("wider_development", dev_pos_meta)):
            with (args.outdir / f"{name}_scene_metadata.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
    if args.wider_only:
        report = {"schema": "job-c-wider-training-cache-1", "train_scenes": train_pos_count,
                  "development_scenes": dev_pos_count, "cache": str(wider_cache_path),
                  "sha256": sha256_file(wider_cache_path), "final_roles_loaded": False,
                  "label_semantics": ["background", "face"]}
        dump_json(args.outdir / "wider_cache_report.json", report)
        print(json.dumps(report, indent=2))
        return

    train_neg = scene_rows(args.pass_manifest, Path("."), "train", negative=True)
    dev_neg = scene_rows(args.pass_manifest, Path("."), "development", negative=True)
    train_x = np.concatenate([wider_train_x, np.stack(train_neg[0])]).astype(np.uint8)
    train_y = np.concatenate([wider_train_y, np.stack(train_neg[3])]).astype(np.uint8)
    dev_x = np.concatenate([wider_dev_x, np.stack(dev_neg[0])]).astype(np.uint8)
    dev_rgb = np.concatenate([wider_dev_rgb, np.stack(dev_neg[1])]).astype(np.uint8)
    dev_window = np.concatenate([wider_dev_window, np.stack(dev_neg[2])]).astype(np.uint8)
    dev_y = np.concatenate([wider_dev_y, np.stack(dev_neg[3])]).astype(np.uint8)
    train_meta, dev_meta = train_pos_meta + train_neg[4], dev_pos_meta + dev_neg[4]

    np.savez_compressed(args.outdir / "train_cache.npz", gray96=train_x, targets=train_y)
    np.savez_compressed(args.outdir / "development_cache.npz", gray96=dev_x, rgb96=dev_rgb,
                        gray_window=dev_window, targets=dev_y)
    for name, rows in (("train", train_meta), ("development", dev_meta)):
        with (args.outdir / f"{name}_scene_metadata.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "job-c-training-cache-1",
        "camera": {"sensor_frame": [320, 240], "window": [240, 240], "window_origin": [40, 0],
                   "detector_input": [96, 96], "detector_stride": 8, "color": "grayscale"},
        "train": {"scenes": int(train_x.shape[0]), "wider_positive_or_partial": train_pos_count,
                  "pass_real_no_face": len(train_neg[0]), "npz_sha256": sha256_file(args.outdir / "train_cache.npz")},
        "development": {"scenes": int(dev_x.shape[0]), "wider_positive_or_partial": dev_pos_count,
                        "pass_real_no_face": len(dev_neg[0]),
                        "npz_sha256": sha256_file(args.outdir / "development_cache.npz")},
        "zero_box_wider_rows_used_as_negative": 0,
        "final_roles_loaded": False,
        "label_semantics": ["background", "face"],
        "target_rule": "face centroid in 12x12 grid; multiple faces in one cell share one foreground cell; stride 8",
        "transform": "EXIF transpose verified orientation=1; grayscale source; center-fit to QVGA 320x240; centered camera window x=40..279; bilinear resize 240x240 to 96x96",
    }
    dump_json(args.outdir / "cache_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
