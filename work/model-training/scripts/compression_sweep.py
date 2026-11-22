#!/usr/bin/env python3
"""Compare deployable TFLite compression strategies on identity-disjoint validation only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
MT_ROOT = HERE.parents[1]
SCRIPTS = MT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    image_path, jsonl, load_gray, load_manifest, pair_record_ids, pair_scores,
    remove_exact_duplicate_conflicts, select_training_rows, verification_metrics,
)


RUNS = {
    "base": MT_ROOT / "runs/survface_supcon_v2_holdout-clean",
    "tiny": MT_ROOT / "runs/survface_supcon_tiny_v1",
}
ARTIFACTS = MT_ROOT / "compression/strategy_sweep_20261022"
THREADS = 4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def selected_calibration_rows(run_dir):
    split = json.loads((run_dir / "split.json").read_text())
    config = json.loads((run_dir / "train_config.json").read_text())
    all_rows, _, raw_rows, _ = load_manifest(include_olivetti=False)
    holdout = [r for r in all_rows if r["dataset"] == "survface_96" and not r["identity_id"]]
    clean_rows, audit = remove_exact_duplicate_conflicts(raw_rows, holdout)
    rows = select_training_rows(
        clean_rows, split["train_identity_ids"],
        int(config["max_images_per_train_identity"]), int(split["seed"]),
    )
    expected = int(json.loads((run_dir / "run_summary.json").read_text())["training_image_count"])
    if len(rows) != expected:
        raise ValueError("%s selected training count mismatch: %d != %d" % (run_dir, len(rows), expected))
    random.Random(int(split["seed"])).shuffle(rows)
    return rows[:min(512, len(rows))], audit


def rep_dataset(rows):
    for row in rows:
        gray = load_gray(row).astype(np.float32) / 255.0
        yield [gray[None, ..., None]]


def convert_variant(model, representative, out_dir, family):
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {}

    path = out_dir / (family + "_float32.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    path.write_bytes(converter.convert())
    variants["float32"] = path

    path = out_dir / (family + "_float16_weights.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    path.write_bytes(converter.convert())
    variants["float16_weights"] = path

    path = out_dir / (family + "_dynamic_int8_weights.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    path.write_bytes(converter.convert())
    variants["dynamic_int8_weights_float_io"] = path

    path = out_dir / (family + "_full_int8_io.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_dataset(representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())
    variants["full_int8_io"] = path
    return variants


class TFLiteRunner:
    def __init__(self, path):
        self.path = path
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=THREADS)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def run(self, image):
        value = np.asarray(image, dtype=np.float32)[None, ..., None] / 255.0
        if self.input["dtype"] == np.int8:
            scale, zero = self.input["quantization"]
            if not scale or not np.isfinite(scale):
                raise ValueError("Invalid input quantization metadata")
            value = np.clip(np.rint(value / scale + zero), -128, 127).astype(np.int8)
        else:
            value = value.astype(self.input["dtype"])
        self.interpreter.set_tensor(self.input["index"], value)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output["index"])[0]
        if self.output["dtype"] == np.int8:
            scale, zero = self.output["quantization"]
            if not scale or not np.isfinite(scale):
                raise ValueError("Invalid output quantization metadata")
            result = (result.astype(np.float32) - zero) * scale
        else:
            result = result.astype(np.float32)
        norm = float(np.linalg.norm(result))
        if not np.isfinite(norm) or norm <= 1e-12 or not np.all(np.isfinite(result)):
            raise ValueError("Non-finite or zero embedding output")
        return result / norm


def meta(path):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=THREADS)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    ops = sorted({item["op_name"] for item in interpreter._get_ops_details()})
    blob = Path(path).read_bytes()
    return {
        "file_bytes": len(blob),
        "gzip_bytes": len(gzip.compress(blob, compresslevel=9)),
        "gzip_is_not_directly_loadable_by_openmv": True,
        "sha256": sha256(path),
        "input": {
            "shape": inp["shape"].tolist(), "dtype": str(inp["dtype"]),
            "scale": float(inp["quantization"][0]), "zero_point": int(inp["quantization"][1]),
        },
        "output": {
            "shape": out["shape"].tolist(), "dtype": str(out["dtype"]),
            "scale": float(out["quantization"][0]), "zero_point": int(out["quantization"][1]),
        },
        "operators": ops,
        "sparsity": {
            "zero_weight_fraction": None,
            "note": "FlatBuffer sparsity metadata is not assumed from zero-valued tensor weights.",
        },
    }


def make_stress_images():
    yy, xx = np.indices((96, 96))
    rng = np.random.default_rng(20261022)
    ramp = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
    checker = (((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)
    random_noise = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
    center = np.zeros((96, 96), np.uint8)
    center[24:72, 24:72] = 255
    return {
        "all_black_0": np.zeros((96, 96), np.uint8),
        "all_near_black_1": np.ones((96, 96), np.uint8),
        "all_midgray_127": np.full((96, 96), 127, np.uint8),
        "all_midgray_128": np.full((96, 96), 128, np.uint8),
        "all_white_255": np.full((96, 96), 255, np.uint8),
        "horizontal_full_range_ramp": ramp,
        "checkerboard_4px": checker,
        "seeded_uniform_noise": random_noise,
        "center_square_on_black": center,
        "center_square_on_white": 255 - center,
    }


def perturb(image, name):
    source = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    if name == "exposure_gamma_0_6":
        lut = [int(round(255.0 * ((value / 255.0) ** 0.6))) for value in range(256)]
        return np.asarray(source.point(lut), np.uint8)
    if name == "exposure_gamma_1_4":
        lut = [int(round(255.0 * ((value / 255.0) ** 1.4))) for value in range(256)]
        return np.asarray(source.point(lut), np.uint8)
    if name == "contrast_0_5":
        return np.asarray(ImageEnhance.Contrast(source).enhance(0.5), np.uint8)
    if name == "contrast_1_5":
        return np.asarray(ImageEnhance.Contrast(source).enhance(1.5), np.uint8)
    if name == "gaussian_blur_radius_1":
        return np.asarray(source.filter(ImageFilter.GaussianBlur(radius=1.0)), np.uint8)
    if name == "downsample_48_then_resize":
        return np.asarray(source.resize((48, 48), Image.Resampling.BILINEAR).resize((96, 96), Image.Resampling.BILINEAR), np.uint8)
    if name == "downsample_32_then_resize":
        return np.asarray(source.resize((32, 32), Image.Resampling.BILINEAR).resize((96, 96), Image.Resampling.BILINEAR), np.uint8)
    if name == "translate_right_6":
        shifted = Image.new("L", (96, 96), 0)
        shifted.paste(source, (6, 0))
        return np.asarray(shifted, np.uint8)
    if name == "occlude_center_band":
        arr = np.asarray(source).copy()
        arr[43:53, :] = 127
        return arr
    raise KeyError(name)


def main():
    global THREADS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    THREADS = args.threads
    out_root = args.output_dir.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError("Refusing to overwrite a non-empty output folder: " + str(out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    all_rows, by_record, raw_surv_rows, _ = load_manifest(include_olivetti=False)
    unlabeled = [r for r in all_rows if r["dataset"] == "survface_96" and not r["identity_id"]]
    clean_rows, duplicate_audit = remove_exact_duplicate_conflicts(raw_surv_rows, unlabeled)
    report = {
        "protocol": "validation-only compression and input robustness sweep",
        "test_annotations_loaded": False,
        "olivetti_loaded": False,
        "final_pair_labels_reopened": False,
        "model_selection_cohort": "same identity-disjoint validation pairs used by the original development workflow",
        "host": {"tensorflow": tf.__version__, "python": sys.version.split()[0],
                 "threads": THREADS, "platform": sys.platform},
        "duplicate_audit": {k: v for k, v in duplicate_audit.items() if k != "excluded_record_ids"},
        "strategies": {},
        "input_stress": {},
        "robustness": {},
        "limitations": [],
    }

    candidates = {}
    run_details = {}
    for family, run_dir in RUNS.items():
        checkpoint = run_dir / "best_embedding.h5"
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        calibration, audit = selected_calibration_rows(run_dir)
        model = tf.keras.models.load_model(checkpoint, compile=False)
        variant_paths = convert_variant(model, calibration, out_root / family, family)
        candidates.update({family + "/" + name: path for name, path in variant_paths.items()})
        run_details[family] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "calibration_count": len(calibration),
            "calibration_validation_identities": 0,
            "duplicate_audit": {k: v for k, v in audit.items() if k != "excluded_record_ids"},
        }
        del model

    # Ensure architectures use the same identity-disjoint validation pair rows.
    pairs = list(jsonl(RUNS["base"] / "validation_pairs.jsonl"))
    tiny_pairs = list(jsonl(RUNS["tiny"] / "validation_pairs.jsonl"))
    if pairs != tiny_pairs:
        raise ValueError("Base and tiny validation pair definitions differ")
    record_ids = pair_record_ids(pairs)
    report["validation_pairs"] = {
        "pair_count": len(pairs),
        "positive_pairs": sum(p["pair_type"] == "positive" for p in pairs),
        "negative_pairs": sum(p["pair_type"] == "negative" for p in pairs),
        "unique_records": len(record_ids),
        "identities": len({by_record[r]["identity_id"] for r in record_ids}),
        "final_pair_annotations_used": False,
    }

    runners = {}
    base_embeddings = {}
    for key, path in sorted(candidates.items()):
        runner = TFLiteRunner(path)
        runners[key] = runner
        vectors = {}
        timings = []
        warmup = record_ids[:min(10, len(record_ids))]
        for record in warmup:
            runner.run(load_gray(by_record[record]))
        for record in record_ids:
            image = load_gray(by_record[record])
            start = time.perf_counter()
            vectors[record] = runner.run(image)
            timings.append((time.perf_counter() - start) * 1000.0)
        scores, labels = pair_scores(pairs, vectors)
        metrics = verification_metrics(scores, labels, bootstrap=1000, seed=20261022)
        metadata = meta(path)
        metadata["verification"] = metrics
        metadata["host_timing_ms"] = {
            "samples": len(timings),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "mean": float(np.mean(timings)),
            "timing_scope": "TFLite invoke plus quantization/dequantization and L2 normalization; image decode excluded",
        }
        report["strategies"][key] = metadata
        if key == "base/full_int8_io":
            base_embeddings = vectors
        del vectors
        print("evaluated", key, "AUC", metrics["roc_auc"], "bytes", metadata["file_bytes"], flush=True)

    # Deterministic synthetic input extremes test execution only, not recognition.
    stress = make_stress_images()
    for key, runner in runners.items():
        rows = []
        for label, image in stress.items():
            try:
                vector = runner.run(image)
                rows.append({"case": label, "pass": True, "norm": float(np.linalg.norm(vector)),
                             "all_finite": bool(np.all(np.isfinite(vector))),
                             "quantized_input_minmax": [
                                 int(runner.interpreter.get_tensor(runner.input["index"]).min()),
                                 int(runner.interpreter.get_tensor(runner.input["index"]).max()),
                             ] if runner.input["dtype"] == np.int8 else None})
            except Exception as error:
                rows.append({"case": label, "pass": False, "error": repr(error)[:240]})
        report["input_stress"][key] = rows

    # Face-crop perturbation range on the preselected base validation cohort only.
    transformations = [
        "exposure_gamma_0_6", "exposure_gamma_1_4",
        "contrast_0_5", "contrast_1_5",
        "gaussian_blur_radius_1", "downsample_48_then_resize",
        "downsample_32_then_resize", "translate_right_6",
        "occlude_center_band",
    ]
    robust_runner_keys = [
        "base/float32", "base/float16_weights",
        "base/dynamic_int8_weights_float_io", "base/full_int8_io",
        "tiny/full_int8_io",
    ]
    for key in robust_runner_keys:
        runner = runners[key]
        source_vectors = base_embeddings if key == "base/full_int8_io" else {}
        if not source_vectors:
            for record in record_ids:
                source_vectors[record] = runner.run(load_gray(by_record[record]))
        condition_results = {}
        for transform in transformations:
            vectors = {}
            for record in record_ids:
                image = perturb(load_gray(by_record[record]), transform)
                vectors[record] = runner.run(image)
            scores, labels = pair_scores(pairs, vectors)
            metric = verification_metrics(scores, labels, bootstrap=0)
            cosine = [float(np.dot(source_vectors[r], vectors[r])) for r in record_ids]
            condition_results[transform] = {
                "pair_metrics": metric,
                "embedding_cosine_vs_same_variant_clean": {
                    "mean": float(np.mean(cosine)),
                    "p05": float(np.percentile(cosine, 5)),
                    "minimum": float(np.min(cosine)),
                },
                "validation_records_transformed": len(record_ids),
            }
        report["robustness"][key] = condition_results
        print("robustness complete", key, flush=True)

    tfmot_available = False
    tfmot_version = None
    try:
        import tensorflow_model_optimization as tfmot  # noqa: F401
        tfmot_available = True
        tfmot_version = getattr(tfmot, "__version__", "unknown")
    except Exception:
        pass
    report["optimization_tooling"] = {
        "tensorflow_model_optimization_available": tfmot_available,
        "tensorflow_model_optimization_version": tfmot_version,
        "this_sweep_scope": "standard float, float16-weight, dynamic-range INT8-weight, and full-INT8 export strategies for base and tiny architectures",
        "pruning_and_clustering_tested_in_this_sweep": False,
        "separate_advanced_strategy_report": "work/model-training/compression/advanced_strategy_sweep_corrected_20261022/advanced_compression_results.json",
        "separate_qat_report": "work/model-training/compression/qat_strategy_sweep_20261022/advanced_compression_results.json",
        "advanced_strategies_completed": ["50% magnitude pruning", "2:4 structured pruning", "16-cluster weight clustering", "quantization-aware training"],
        "reason": "Pruning, clustering, and QAT are evaluated in the separate TFMOT strategy reports. The standard export sweep also includes the trained narrower-channel architecture.",
        "channel_width_strategy": {
            "base_int8_bytes": report["strategies"]["base/full_int8_io"]["file_bytes"],
            "tiny_int8_bytes": report["strategies"]["tiny/full_int8_io"]["file_bytes"],
            "tiny_relative_size_reduction": 1.0 - report["strategies"]["tiny/full_int8_io"]["file_bytes"] / report["strategies"]["base/full_int8_io"]["file_bytes"],
        },
    }
    report["limitations"].extend([
        "Compression candidate metrics use the development validation cohort; they are not independent test estimates.",
        "The previously evaluated final-pair annotations were deliberately not reopened for compression strategy selection.",
        "Synthetic range inputs test numerical stability, not identity accuracy.",
        "Image perturbations are controlled stress probes and do not model a validated Nicla camera distribution.",
        "Every candidate from the standard and advanced export sweeps was separately included in the recorded 14-candidate Nicla compression matrix; see nicla_compression_matrix_20261022.jsonl.",
    ])
    write_json(out_root / "compression_robustness_results.json", report)
    print("report", out_root / "compression_robustness_results.json", flush=True)


if __name__ == "__main__":
    main()

