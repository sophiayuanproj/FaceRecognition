"""Freeze exact raw grayscale host/board fixtures from development scenes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import tensorflow as tf

from common import identity_input_gray96, quantize_gray_input, sha256_bytes, dump_json
from common import openmv_fomo_detections
from evaluate_development import crop_geometry, invoke, read_jsonl


SEEDS = [20261501, 20261502, 20261503, 20261504, 20261505,
         20261506, 20261507, 20261508, 20261509, 20261510]


def boxes_for(row: dict) -> list[dict]:
    boxes = []
    for box in row.get("face_boxes", []):
        if box.get("center_visible_in_detector_roi"):
            boxes.append({"center": [float(v) for v in box["center_detector_xy"]],
                          "box": [float(v) for v in box["xyxy_detector"]],
                          "max_side_detector_px": float(box["max_side_detector_px"])})
    return boxes


def translate(frame: np.ndarray, dx: int, dy: int, fill: int = 0) -> np.ndarray:
    out = np.full_like(frame, fill)
    src_x1, src_x2 = max(0, -dx), min(240, 240 - dx)
    src_y1, src_y2 = max(0, -dy), min(240, 240 - dy)
    dst_x1, dst_x2 = max(0, dx), min(240, 240 + dx)
    dst_y1, dst_y2 = max(0, dy), min(240, 240 + dy)
    if src_x2 > src_x1 and src_y2 > src_y1:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
    return out


def transform(frame: np.ndarray, boxes: list[dict], case: str, seed: int) -> tuple[np.ndarray, list[dict], dict]:
    rng = np.random.default_rng(seed)
    out_boxes = [json.loads(json.dumps(box)) for box in boxes]
    details = {"case": case, "seed": seed, "rotation_degrees_clockwise": 0,
               "horizontal_mirror": False, "vertical_flip": False,
               "translation_window_px": [0, 0], "brightness_multiplier": 1.0,
               "gaussian_blur_radius": 0, "sensor_noise_sigma": 0}
    if case == "mirror_x":
        frame = np.flip(frame, axis=1).copy(); details["horizontal_mirror"] = True
        for box in out_boxes:
            x1, y1, x2, y2 = box["box"]
            box["box"] = [96 - x2, y1, 96 - x1, y2]
            box["center"][0] = 96 - box["center"][0]
    elif case == "flip_y":
        frame = np.flip(frame, axis=0).copy(); details["vertical_flip"] = True
        for box in out_boxes:
            x1, y1, x2, y2 = box["box"]
            box["box"] = [x1, 96 - y2, x2, 96 - y1]
            box["center"][1] = 96 - box["center"][1]
    elif case == "rotate_90_ccw":
        frame = np.rot90(frame, 1).copy(); details["rotation_degrees_clockwise"] = 270
        for box in out_boxes:
            x1, y1, x2, y2 = box["box"]
            box["box"] = [y1, 96 - x2, y2, 96 - x1]
            box["center"] = [box["center"][1], 96 - box["center"][0]]
    elif case == "center_jitter":
        dx, dy = [int(v) for v in rng.choice([-12, -8, 8, 12], size=2)]
        frame = translate(frame, dx, dy, fill=0)
        details["translation_window_px"] = [dx, dy]
        for box in out_boxes:
            box["box"] = [box["box"][0] + dx / 2.5, box["box"][1] + dy / 2.5,
                          box["box"][2] + dx / 2.5, box["box"][3] + dy / 2.5]
            box["center"] = [box["center"][0] + dx / 2.5, box["center"][1] + dy / 2.5]
    elif case == "low_light":
        factor = float(rng.uniform(0.22, 0.42))
        frame = np.clip(np.rint(frame.astype(np.float32) * factor), 0, 255).astype(np.uint8)
        details["brightness_multiplier"] = factor
    elif case == "blur":
        radius = float(rng.choice([1.0, 1.5, 2.0]))
        frame = np.asarray(Image.fromarray(frame, "L").filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)
        details["gaussian_blur_radius"] = radius
    elif case == "sensor_noise":
        sigma = float(rng.choice([2, 4, 7]))
        noise = rng.normal(0, sigma, size=frame.shape)
        frame = np.clip(np.rint(frame.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        details["sensor_noise_sigma"] = sigma
    elif case == "edge_clip":
        if not out_boxes:
            raise ValueError("edge_clip needs a positive scene")
        center_x = out_boxes[0]["center"][0]
        dx = int(round((10.0 - center_x) * 2.5))
        frame = translate(frame, dx, 0, fill=0)
        details["translation_window_px"] = [dx, 0]
        for box in out_boxes:
            box["box"] = [box["box"][0] + dx / 2.5, box["box"][1],
                          box["box"][2] + dx / 2.5, box["box"][3]]
            box["center"][0] += dx / 2.5
    elif case in ("baseline", "small_face", "multiple_faces", "no_face"):
        pass
    else:
        raise ValueError(case)
    return np.ascontiguousarray(frame, dtype=np.uint8), out_boxes, details


def resize_detector(frame: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(frame, "L").resize((96, 96), Image.Resampling.BILINEAR), dtype=np.uint8)


def int8_tensor(image: np.ndarray, details: dict) -> np.ndarray:
    scale, zero = details["quantization"]
    return quantize_gray_input(image, scale, zero)


def make_detector(interpreter, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    raw_in = int8_tensor(image, {"quantization": inp["quantization"]})[None, ..., None]
    interpreter.set_tensor(inp["index"], raw_in)
    interpreter.invoke()
    raw_out = interpreter.get_tensor(out["index"])
    scale, zero = out["quantization"]
    probs = (raw_out.astype(np.float32) - zero) * scale
    return raw_out, probs


def write_pgm(path: Path, pixels: np.ndarray) -> None:
    path.write_bytes(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii") + pixels.tobytes())


def make_contact_sheets(fixtures: list[dict], frames: list[np.ndarray], detector_inputs: list[np.ndarray],
                        identity_inputs: list[np.ndarray | None], outdir: Path) -> list[str]:
    outputs = []
    for group_index, start in enumerate(range(0, len(fixtures), 5)):
        end = min(start + 5, len(fixtures))
        rows = end - start
        sheet = Image.new("RGB", (720, rows * 280), "#15181d")
        draw = ImageDraw.Draw(sheet)
        for local, index in enumerate(range(start, end)):
            y0 = local * 280
            frame = Image.fromarray(frames[index], "L").convert("RGB")
            input96 = Image.fromarray(detector_inputs[index], "L").resize((240, 240), Image.Resampling.NEAREST).convert("RGB")
            identity = identity_inputs[index]
            crop = Image.new("RGB", (240, 240), "#35393f") if identity is None else Image.fromarray(identity, "L").resize((240, 240), Image.Resampling.NEAREST).convert("RGB")
            sheet.paste(frame, (0, y0 + 28)); sheet.paste(input96, (240, y0 + 28)); sheet.paste(crop, (480, y0 + 28))
            summary = fixtures[index]
            frame_draw = ImageDraw.Draw(sheet)
            for box in summary.get("transformed_annotations_detector_xyxy", []):
                x1, y1, x2, y2 = [float(v) * 2.5 for v in box]
                frame_draw.rectangle((x1, y0 + 28 + y1, x2, y0 + 28 + y2), outline="#48e07a", width=2)
                ix1, iy1, ix2, iy2 = [float(v) * 2.5 + 240 for v in box]
                frame_draw.rectangle((ix1, y0 + 28 + iy1, ix2, y0 + 28 + iy2), outline="#48e07a", width=2)
            crop_xywh = summary.get("expected_crop_geometry", {}).get("crop_xywh_window")
            if crop_xywh:
                x, y, width, height = crop_xywh
                frame_draw.rectangle((x, y0 + 28 + y, x + width, y0 + 28 + y + height), outline="#ff5c6c", width=2)
            status = summary.get("expected_crop_geometry", {}).get("status", "UNSET")
            draw.text((6, y0 + 5), f"{summary['fixture_id']} · {summary['source_role']} · {status}", fill="white")
            draw.text((5, y0 + 268), "240x240 camera window", fill="#cbd3dd")
            draw.text((245, y0 + 268), "96x96 detector input", fill="#cbd3dd")
            draw.text((485, y0 + 268), "raw gray identity input", fill="#cbd3dd")
        path = outdir / f"fixture_contact_sheet_{group_index + 1}.png"
        sheet.save(path)
        outputs.append(str(path))
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--crop-side", type=int, default=96)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache_dir / "development_cache.npz") as cache:
        frames_all = cache["gray_window"]
    metadata = read_jsonl(args.cache_dir / "development_scene_metadata.jsonl")
    pos = [i for i, r in enumerate(metadata) if not r.get("is_real_no_face_negative") and boxes_for(r)]
    single = [i for i in pos if len(boxes_for(metadata[i])) == 1]
    small = [i for i in pos if min(b["max_side_detector_px"] for b in boxes_for(metadata[i])) < 8]
    multi = [i for i in pos if len(boxes_for(metadata[i])) >= 2]
    noface = [i for i, r in enumerate(metadata) if r.get("is_real_no_face_negative")]
    if min(len(single), len(small), len(multi), len(noface)) < 1:
        raise RuntimeError("Development data does not contain required single/small/multiple/no-face fixtures")
    cases = [
        (SEEDS[0], "baseline", single[0]), (SEEDS[1], "mirror_x", single[1 % len(single)]),
        (SEEDS[2], "flip_y", single[2 % len(single)]), (SEEDS[3], "rotate_90_ccw", single[3 % len(single)]),
        (SEEDS[4], "center_jitter", single[4 % len(single)]), (SEEDS[5], "low_light", single[5 % len(single)]),
        (SEEDS[6], "blur", single[6 % len(single)]), (SEEDS[7], "small_face", small[0]),
        (SEEDS[8], "edge_clip", single[8 % len(single)]), (SEEDS[9], "multiple_faces", multi[0]),
    ]
    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=2)
    interpreter.allocate_tensors()
    in_info = interpreter.get_input_details()[0]
    if in_info["dtype"] != np.int8 or tuple(in_info["shape"].tolist()) != (1, 96, 96, 1):
        raise ValueError("Selected fixture model must be a full-int8 96x96x1 detector")

    fixtures, out_frames, detector_inputs, identity_inputs = [], [], [], []
    for seed, case, source_i in cases:
        source = metadata[source_i]
        frame, boxes, t_params = transform(frames_all[source_i], boxes_for(source), case, seed)
        detector_input = resize_detector(frame)
        raw_output, probabilities = make_detector(interpreter, detector_input)
        detections = openmv_fomo_detections(probabilities, args.threshold)
        if len(detections) == 0:
            status = "NO_FACE"
        elif len(detections) != 1:
            status = "UNCERTAIN_MULTIPLE_FACES"
        else:
            geometry = crop_geometry(detections[0], None, args.crop_side)
            status = "UNCERTAIN_CROP_OUTSIDE" if not geometry["crop_inside_camera_window"] else "ONE_FACE_CROP_READY"
        identity_raw = None
        crop = None
        quality = None
        if status == "ONE_FACE_CROP_READY":
            cx, cy = detections[0]["center_frame_xy"]
            x, y = math.floor(cx - args.crop_side / 2.0 + 0.5), math.floor(cy - args.crop_side / 2.0 + 0.5)
            native = frame[y:y + args.crop_side, x:x + args.crop_side]
            crop = identity_input_gray96(native)
            identity_raw = crop.tobytes()
            native_mean, native_stdev = float(native.mean()), float(native.std())
            mean, stdev = float(crop.mean()), float(crop.std())
            a = crop.astype(np.float32)
            lap = (-4.0 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] +
                   a[1:-1, :-2] + a[1:-1, 2:])
            quality = {"native_crop_mean_u8": native_mean, "native_crop_stdev_u8": native_stdev,
                       "identity_input_mean_u8": mean, "identity_input_stdev_u8": stdev,
                       "laplacian_variance": float(lap.var()),
                       "native_crop_quality_gate_pass": 12.0 <= native_mean <= 243.0 and native_stdev >= 5.0,
                       "identity_input_quality_gate_pass": 12.0 <= mean <= 243.0 and stdev >= 5.0}
            if not quality["native_crop_quality_gate_pass"] or not quality["identity_input_quality_gate_pass"]:
                status = "UNCERTAIN_LOW_QUALITY_CROP"
        fixture_id = f"seed-{seed}-{case}"
        fixture_dir = args.outdir / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        frame_bytes = frame.tobytes()
        detector_bytes = detector_input.tobytes()
        quantized_input = int8_tensor(detector_input, {"quantization": in_info["quantization"]})
        (fixture_dir / "camera_window_gray240.u8").write_bytes(frame_bytes)
        (fixture_dir / "detector_input_gray96.u8").write_bytes(detector_bytes)
        (fixture_dir / "detector_input_int8.i8").write_bytes(quantized_input.tobytes())
        (fixture_dir / "detector_output_int8.i8").write_bytes(raw_output.tobytes())
        write_pgm(fixture_dir / "camera_window_gray240.pgm", frame)
        write_pgm(fixture_dir / "detector_input_gray96.pgm", detector_input)
        if identity_raw is not None:
            (fixture_dir / "identity_input_gray96.u8").write_bytes(identity_raw)
            (fixture_dir / "identity_input_int8.i8").write_bytes((crop.astype(np.int16) - 128).astype(np.int8).tobytes())
            write_pgm(fixture_dir / "identity_input_gray96.pgm", crop)
        summary = {
            "fixture_id": fixture_id, "root_seed": seed, "source_role": source.get("role"),
            "source_dataset": source.get("source_dataset", "WIDER FACE"),
            "source_image_path": source.get("local_image_path"),
            "source_image_sha256": source.get("local_image_sha256") or source.get("image_sha256"),
            "transform_parameters": {**t_params, "sensor": {"input_frame": [320, 240], "window": [240, 240],
                                     "window_origin": [40, 0], "rotation": "none", "hmirror": False, "vflip": False},
                                     "detector_resize": "PIL bilinear 240x240 to 96x96",
                                     "quantization": {"input_scale": float(in_info["quantization"][0]),
                                                      "input_zero_point": int(in_info["quantization"][1])}},
            "transformed_annotations_detector_xyxy": [box["box"] for box in boxes],
            "expected_detector_output": {"threshold": args.threshold, "detections": detections,
                                         "raw_output_file": "detector_output_int8.i8",
                                         "output_shape": list(raw_output.shape),
                                         "output_scale_zero_point": list(interpreter.get_output_details()[0]["quantization"])},
            "expected_crop_geometry": {"status": status,
                    "crop_side_window_px": args.crop_side,
                    "predicted_center_window_xy": detections[0]["center_frame_xy"] if len(detections) == 1 else None,
                    "crop_xywh_window": crop_geometry(detections[0], None, args.crop_side)["xywh_window"] if len(detections) == 1 else None,
                    "identity_input_shape": [96, 96] if identity_raw is not None else None,
                    "identity_input_quality": quality},
            "exact_raw_files": {"camera_window_gray240.u8": {"bytes": len(frame_bytes), "sha256": sha256_bytes(frame_bytes)},
                "detector_input_gray96.u8": {"bytes": len(detector_bytes), "sha256": sha256_bytes(detector_bytes)},
                "detector_input_int8.i8": {"bytes": quantized_input.nbytes, "sha256": sha256_bytes(quantized_input.tobytes())},
                "detector_output_int8.i8": {"bytes": raw_output.nbytes, "sha256": sha256_bytes(raw_output.tobytes())},
                "identity_input_gray96.u8": ({"bytes": len(identity_raw), "sha256": sha256_bytes(identity_raw)} if identity_raw is not None else None)},
            "fixture_directory": str(fixture_dir),
        }
        (fixture_dir / "fixture.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        fixtures.append(summary); out_frames.append(frame); detector_inputs.append(detector_input); identity_inputs.append(crop)

    # A genuine no-human PASS image is always included, with its upstream role and source hash.
    for offset, source_i in enumerate(noface[:3], start=1):
        source = metadata[source_i]
        frame = frames_all[source_i].copy(); detector_input = resize_detector(frame)
        raw_output, probabilities = make_detector(interpreter, detector_input)
        detections = openmv_fomo_detections(probabilities, args.threshold)
        seed = 20261600 + offset
        fixture_id = f"seed-{seed}-real_pass_no_face"
        fixture_dir = args.outdir / fixture_id; fixture_dir.mkdir(parents=True, exist_ok=True)
        frame_bytes, detector_bytes = frame.tobytes(), detector_input.tobytes()
        quantized_input = int8_tensor(detector_input, {"quantization": in_info["quantization"]})
        (fixture_dir / "camera_window_gray240.u8").write_bytes(frame_bytes)
        (fixture_dir / "detector_input_gray96.u8").write_bytes(detector_bytes)
        (fixture_dir / "detector_input_int8.i8").write_bytes(quantized_input.tobytes())
        (fixture_dir / "detector_output_int8.i8").write_bytes(raw_output.tobytes())
        write_pgm(fixture_dir / "camera_window_gray240.pgm", frame)
        write_pgm(fixture_dir / "detector_input_gray96.pgm", detector_input)
        summary = {"fixture_id": fixture_id, "root_seed": seed, "source_role": source["role"],
                   "source_dataset": source["source_dataset"], "source_image_sha256": source["local_image_sha256"],
                   "source_item_license": source.get("source_item_license"),
                   "author_attribution": source.get("author_attribution"),
                   "transform_parameters": {"case": "real_pass_no_face", "camera_window": [240, 240],
                                             "detector_resize": "PIL bilinear 240x240 to 96x96"},
                   "expected_detector_output": {"threshold": args.threshold, "detections": detections,
                       "raw_output_file": "detector_output_int8.i8", "output_shape": list(raw_output.shape),
                       "output_scale_zero_point": list(interpreter.get_output_details()[0]["quantization"])},
                   "expected_crop_geometry": {"status": "NO_FACE" if not detections else "FALSE_TRIGGER_UNCERTAIN",
                       "crop_side_window_px": args.crop_side, "identity_input_shape": None},
                   "exact_raw_files": {"camera_window_gray240.u8": {"bytes": len(frame_bytes), "sha256": sha256_bytes(frame_bytes)},
                       "detector_input_gray96.u8": {"bytes": len(detector_bytes), "sha256": sha256_bytes(detector_bytes)},
                       "detector_input_int8.i8": {"bytes": quantized_input.nbytes, "sha256": sha256_bytes(quantized_input.tobytes())},
                       "detector_output_int8.i8": {"bytes": raw_output.nbytes, "sha256": sha256_bytes(raw_output.tobytes())},
                       "identity_input_gray96.u8": None}, "fixture_directory": str(fixture_dir)}
        (fixture_dir / "fixture.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        fixtures.append(summary); out_frames.append(frame); detector_inputs.append(detector_input); identity_inputs.append(None)

    sheets = make_contact_sheets(fixtures, out_frames, detector_inputs, identity_inputs, args.outdir)
    manifest = {"schema": "job-c-raw-parity-fixtures-1", "model_path": str(args.model),
                "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "threshold": args.threshold, "crop_side_window_px": args.crop_side,
                "development_only_sources": True, "final_scenes_loaded": False,
                "transform_fault_block_count": 10, "real_no_face_fixture_count": min(3, len(noface)),
                "fixture_count": len(fixtures), "fixtures": fixtures, "contact_sheets": sheets,
                "raw_byte_policy": "raw gray bytes are stored in .u8/.i8 files; PGM/PNG are previews only; no JPEG inference inputs"}
    dump_json(args.outdir / "fixture_manifest.json", manifest)
    print(json.dumps({"fixture_count": len(fixtures), "transform_fault_block_count": 10,
                      "contact_sheets": sheets, "outdir": str(args.outdir)}, indent=2))


if __name__ == "__main__":
    main()
