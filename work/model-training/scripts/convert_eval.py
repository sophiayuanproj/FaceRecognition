#!/usr/bin/env python3
"""Export float and full-integer TFLite models and evaluate verification pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from common import (
    MT_ROOT, ROOT, embed_keras, image_path, jsonl, load_gray, load_manifest,
    auc_eer, make_olivetti_pairs, pair_record_ids, pair_scores, representative_generator,
    remove_exact_duplicate_conflicts, select_training_rows, sha256, verification_metrics, write_json,
)


INT8_OPS = {
    "AVERAGE_POOL_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
    "MAX_POOL_2D", "MEAN", "RESHAPE", "RELU", "RELU6", "PAD", "ADD",
    "MUL", "QUANTIZE", "DEQUANTIZE", "STRIDED_SLICE", "SQUEEZE",
}


class Runner:
    def __init__(self, model, tflite=False, threads=4):
        self.model = model
        self.tflite = tflite
        self.threads = threads
        self.interpreter = None
        if tflite:
            self.interpreter = tf.lite.Interpreter(model_path=str(model), num_threads=threads)
            self.interpreter.allocate_tensors()
            self.input = self.interpreter.get_input_details()[0]
            self.output = self.interpreter.get_output_details()[0]

    def run(self, image_u8):
        value = image_u8.astype(np.float32)[None, ..., None] / 255.0
        if not self.tflite:
            vector = self.model(value, training=False).numpy()[0]
        else:
            detail = self.input
            if detail["dtype"] == np.int8:
                scale, zero = detail["quantization"]
                if scale <= 0:
                    raise ValueError("TFLite input has no quantization scale")
                value = np.clip(np.rint(value / scale + zero), -128, 127).astype(np.int8)
            else:
                value = value.astype(detail["dtype"])
            self.interpreter.set_tensor(detail["index"], value)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output["index"])[0]
            if self.output["dtype"] == np.int8:
                scale, zero = self.output["quantization"]
                if scale <= 0:
                    raise ValueError("TFLite output has no quantization scale")
                vector = (output.astype(np.float32) - zero) * scale
            else:
                vector = output.astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(float(norm), 1e-12)

    def embed(self, record_ids, by_record):
        if not self.tflite:
            return embed_keras(self.model, record_ids, by_record)
        vectors = {}
        for index, record_id in enumerate(record_ids, start=1):
            vectors[record_id] = self.run(load_gray(by_record[record_id]))
            if index % 2000 == 0:
                print("embedded_records=%d/%d" % (index, len(record_ids)), flush=True)
        return vectors

    def timing(self, record_ids, by_record, limit=200, warmup=10):
        sample = record_ids[:min(len(record_ids), limit + warmup)]
        for record_id in sample[:warmup]:
            self.run(load_gray(by_record[record_id]))
        values = []
        for record_id in sample[warmup:]:
            image = load_gray(by_record[record_id])
            start = time.perf_counter()
            self.run(image)
            values.append((time.perf_counter() - start) * 1000.0)
        return {
            "images": len(values),
            "warmup_images": min(warmup, len(sample)),
            "mean_ms_per_image": float(np.mean(values)) if values else None,
            "p50_ms_per_image": float(np.percentile(values, 50)) if values else None,
            "p95_ms_per_image": float(np.percentile(values, 95)) if values else None,
            "host_cpu_threads": self.threads,
        }


def collect_pairs(path):
    return [row for row in jsonl(path) if row.get("match_status", "matched") == "matched"]


def identity_cluster_bootstrap(pairs, scores, labels, by_record, threshold, replicates=1000, seed=20261021):
    """Vertex-cluster bootstrap: resample validation identities and weight negative dyads by both endpoints."""
    identities = set()
    endpoints = []
    for pair in pairs:
        left_id = by_record[pair["left_record_id"]]["identity_id"]
        right_id = by_record[pair["right_record_id"]]["identity_id"]
        if pair["pair_type"] == "positive" and left_id != right_id:
            raise ValueError("Validation positive pair crosses identity labels")
        if pair["pair_type"] == "negative" and left_id == right_id:
            raise ValueError("Validation negative pair shares an identity label")
        identities.add(left_id)
        identities.add(right_id)
        endpoints.append((left_id, right_id))
    identities = sorted(identities)
    if len(identities) < 2:
        return {"replicates_requested": replicates, "replicates_used": 0,
                "bootstrap_unit": "identity cluster", "bootstrap_95pct_ci": {}}
    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "eer": [], "tar_at_decision_threshold": [], "far_at_decision_threshold": []}
    for _ in range(replicates):
        counts = Counter(rng.choice(identities, size=len(identities), replace=True).tolist())
        sample_scores = []
        sample_labels = []
        for index, (left_id, right_id) in enumerate(endpoints):
            weight = counts[left_id] if labels[index] == 1 else counts[left_id] * counts[right_id]
            if weight:
                sample_scores.extend([float(scores[index])] * weight)
                sample_labels.extend([int(labels[index])] * weight)
        sample_scores = np.asarray(sample_scores, dtype=np.float64)
        sample_labels = np.asarray(sample_labels, dtype=np.int8)
        positive = sample_labels == 1
        negative = sample_labels == 0
        if not positive.any() or not negative.any():
            continue
        curve = auc_eer(sample_scores, sample_labels)
        values["roc_auc"].append(curve["roc_auc"])
        values["eer"].append(curve["eer"])
        values["tar_at_decision_threshold"].append(float(np.mean(sample_scores[positive] >= threshold)))
        values["far_at_decision_threshold"].append(float(np.mean(sample_scores[negative] >= threshold)))
    return {
        "replicates_requested": int(replicates),
        "replicates_used": len(values["roc_auc"]),
        "bootstrap_unit": "validation identity clusters; negative-pair weights are products of endpoint multiplicities",
        "bootstrap_95pct_ci": {
            key: [float(np.percentile(items, 2.5)), float(np.percentile(items, 97.5))]
            for key, items in values.items() if items
        },
    }


def convert(model, representative_rows, float_path, int8_path):
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_blob = float_converter.convert()
    float_path.parent.mkdir(parents=True, exist_ok=True)
    float_path.write_bytes(float_blob)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_generator(
        representative_rows, count=None,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_blob = converter.convert()
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    int8_path.write_bytes(int8_blob)


def model_metadata(model_path, threads=4):
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    ops = sorted({row["op_name"] for row in interpreter._get_ops_details()})
    model_size = Path(model_path).stat().st_size
    return {
        "path": str(Path(model_path).resolve().relative_to(ROOT)),
        "file_bytes": model_size,
        "sha256": sha256(model_path),
        "input": {
            "shape": input_detail["shape"].tolist(),
            "dtype": str(input_detail["dtype"]),
            "scale": float(input_detail["quantization"][0]),
            "zero_point": int(input_detail["quantization"][1]),
        },
        "output": {
            "shape": output_detail["shape"].tolist(),
            "dtype": str(output_detail["dtype"]),
            "scale": float(output_detail["quantization"][0]),
            "zero_point": int(output_detail["quantization"][1]),
        },
        "operators": ops,
        "unsupported_openmv_candidate_ops": sorted(set(ops) - INT8_OPS),
        "input_output_full_int8": input_detail["dtype"] == np.int8 and output_detail["dtype"] == np.int8,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=MT_ROOT / "runs/survface_mobile_v1")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--host-threads", type=int, default=4,
                        help="TensorFlow/TFLite CPU threads for this evaluation process")
    parser.add_argument("--validation-only", action="store_true",
                        help="Do not load the final pair annotations; use for model selection")
    parser.add_argument("--float-output", type=Path,
                        default=MT_ROOT / "artifacts/identity_embedding_float.tflite")
    parser.add_argument("--int8-output", type=Path,
                        default=ROOT / "FaceRecognition/models/identity_embedding_int8.tflite")
    args = parser.parse_args()
    if args.host_threads < 1:
        parser.error("--host-threads must be at least 1")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(args.host_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else run_dir / "best_embedding.h5"
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing trained feature extractor checkpoint")
    split = json.loads((run_dir / "split.json").read_text())
    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    train_config = json.loads((run_dir / "train_config.json").read_text())
    all_rows, by_record, raw_surv_rows, oliv_rows = load_manifest(include_olivetti=not args.validation_only)
    unlabeled_survface_rows = [row for row in all_rows
                               if row["dataset"] == "survface_96" and not row["identity_id"]]
    surv_rows, duplicate_audit = remove_exact_duplicate_conflicts(
        raw_surv_rows, unlabeled_survface_rows,
    )
    if sorted(duplicate_audit["excluded_record_ids"]) != sorted(split["excluded_exact_duplicate_record_ids"]):
        raise ValueError("Reconstructed exact-duplicate exclusions differ from the training split")

    training_ids = list(split["train_identity_ids"])
    identity_limit = int(train_config.get("train_identity_limit") or 0)
    if identity_limit:
        training_ids = sorted(random.Random(int(run_summary["training_seed"]) + 31).sample(
            training_ids, min(identity_limit, len(training_ids)),
        ))
    selected_training_rows = select_training_rows(
        surv_rows, training_ids,
        int(train_config["max_images_per_train_identity"]),
        int(split["seed"]),
    )
    if len(selected_training_rows) != int(run_summary["training_image_count"]):
        raise ValueError("Reconstructed selected training records do not match run_summary training_image_count")

    calibration_rows = list(selected_training_rows)
    random.Random(int(split["seed"])).shuffle(calibration_rows)
    calibration_rows = calibration_rows[:min(512, len(calibration_rows))]
    calibration_records = []
    for row in calibration_rows:
        actual_hash = sha256(image_path(row))
        if actual_hash != row["sha256"]:
            raise ValueError("Calibration image content differs from prepared manifest: " + row["record_id"])
        calibration_records.append({
            "record_id": row["record_id"],
            "identity_id": row["identity_id"],
            "image_sha256": actual_hash,
        })
    manifest_payload = {
        "protocol": "exact selected training records after duplicate exclusion and per-identity cap",
        "split_seed": int(split["seed"]),
        "training_seed": int(run_summary["training_seed"]),
        "training_record_count": len(selected_training_rows),
        "training_selection_sha256": hashlib.sha256("".join(
            row["record_id"] + "\t" + row["sha256"] + "\n" for row in selected_training_rows
        ).encode("utf-8")).hexdigest(),
        "representative_seed": int(split["seed"]),
        "representative_records": calibration_records,
    }
    manifest_payload["representative_manifest_sha256"] = hashlib.sha256(json.dumps(
        calibration_records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    calibration_manifest_path = run_dir / "calibration_manifest.json"
    write_json(calibration_manifest_path, manifest_payload)

    float_path = args.float_output.resolve()
    int8_path = args.int8_output.resolve()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    convert(model, calibration_rows, float_path, int8_path)

    validation_pairs = list(jsonl(run_dir / "validation_pairs.jsonl"))
    datasets = {
        "survface_identity_disjoint_validation": validation_pairs,
    }
    if not args.validation_only:
        final_pairs = collect_pairs(ROOT / "work/dataset-prep/reports/survface-pairs.jsonl")
        olivetti_pairs = make_olivetti_pairs(oliv_rows, seed=split["seed"])
        datasets["survface_pair_annotations_final_only"] = final_pairs
        datasets["olivetti_external_sanity"] = olivetti_pairs
    runners = {
        "keras_float": Runner(model, threads=args.host_threads),
        "tflite_float": Runner(float_path, tflite=True, threads=args.host_threads),
        "tflite_full_int8": Runner(int8_path, tflite=True, threads=args.host_threads),
    }
    results = {}
    threshold_by_runner = {}
    val_pair_key = "survface_identity_disjoint_validation"
    for name, runner in runners.items():
        results[name] = {}
        for dataset_name, pairs in datasets.items():
            ids = pair_record_ids(pairs)
            vectors = runner.embed(ids, by_record)
            scores, labels = pair_scores(pairs, vectors)
            if dataset_name == val_pair_key:
                metrics = verification_metrics(scores, labels, bootstrap=args.bootstrap,
                                               seed=split["seed"])
                threshold_by_runner[name] = metrics["decision_threshold"]
                if args.bootstrap:
                    seed_offset = {"keras_float": 11000, "tflite_float": 22000,
                                   "tflite_full_int8": 33000}[name]
                    metrics["identity_cluster_bootstrap"] = identity_cluster_bootstrap(
                        pairs, scores, labels, by_record, metrics["decision_threshold"],
                        replicates=args.bootstrap, seed=split["seed"] + seed_offset,
                    )
            else:
                metrics = verification_metrics(
                    scores, labels, threshold=threshold_by_runner[name],
                    bootstrap=args.bootstrap, seed=split["seed"],
                )
            results[name][dataset_name] = metrics
            del vectors
            print("evaluated=%s dataset=%s pair_count=%d auc=%s eer=%s" % (
                name, dataset_name, len(pairs), metrics["roc_auc"], metrics["eer"],
            ), flush=True)
    results["model_files"] = {
        "float_tflite": model_metadata(float_path, args.host_threads),
        "full_integer_int8_tflite": model_metadata(int8_path, args.host_threads),
        "keras_hdf5_checkpoint_bytes": checkpoint.stat().st_size,
    }
    int8_meta = results["model_files"]["full_integer_int8_tflite"]
    input_meta = int8_meta["input"]
    output_meta = int8_meta["output"]
    if input_meta["dtype"] != "<class 'numpy.int8'>" or output_meta["dtype"] != "<class 'numpy.int8'>":
        raise ValueError("INT8 export did not preserve signed INT8 input and output")
    if input_meta["shape"] != [1, 96, 96, 1] or output_meta["shape"] != [1, 64]:
        raise ValueError("INT8 export I/O dimensions differ from the board firmware contract")
    if abs(input_meta["scale"] - 1.0 / 255.0) > 0.00001 or input_meta["zero_point"] != -128:
        raise ValueError("INT8 input quantization does not match OpenMV pixel-128 preprocessing")
    if int8_meta["unsupported_openmv_candidate_ops"]:
        print("int8_operator_review_needed=" + json.dumps(int8_meta["unsupported_openmv_candidate_ops"]), flush=True)
    timing_ids = pair_record_ids(validation_pairs)[:300]
    results["host_inference_timing"] = {
        name: runner.timing(timing_ids, by_record)
        for name, runner in runners.items()
    }
    results["calibration"] = {
        "thresholds_by_model": threshold_by_runner,
        "rule": "validation threshold permits at most floor(1% of validation negative pairs) false accepts",
        "validation_pair_ids_used_for_calibration": True,
        "final_pair_annotations_used_for_tuning": False,
        "representative_source": "exact selected training records after duplicate exclusion and per-identity cap",
        "representative_count": len(calibration_rows),
        "representative_seed": int(split["seed"]),
        "representative_validation_identity_count": 0,
        "representative_manifest": str(calibration_manifest_path.resolve().relative_to(ROOT)),
        "representative_manifest_sha256": manifest_payload["representative_manifest_sha256"],
        "training_selection_sha256": manifest_payload["training_selection_sha256"],
        "reconstructed_training_record_count": len(selected_training_rows),
    }
    results["environment"] = {
        "python": __import__("sys").version.split()[0],
        "tensorflow": tf.__version__,
        "platform": "macOS x86_64 Intel CPU",
        "representative_samples": len(calibration_rows),
        "representative_set_source": "exact selected capped SurvFace training records only",
    }
    results["notes"] = {
        "survface_test_images_have_no_inferred_identity_labels": True,
        "final_test_pairs_use_pair_labels_only": True,
        "olivetti_kept_separate": True,
        "quantization_metadata_not_manually_modified": True,
        "verification_threshold_not_tuned_on_final_pair_labels": True,
    }
    write_json(run_dir / "float_int8_evaluation.json", results)
    if not args.validation_only:
        write_json(MT_ROOT / "artifacts/embedding_calibration.json", {
            "threshold": threshold_by_runner["tflite_full_int8"],
            "threshold_policy": results["calibration"]["rule"],
            "validation_auc": results["tflite_full_int8"][val_pair_key]["roc_auc"],
            "validation_far": results["tflite_full_int8"][val_pair_key]["far_at_decision_threshold"],
            "validation_tar": results["tflite_full_int8"][val_pair_key]["tar_at_decision_threshold"],
            "source": "full INT8 model, SurvFace identity-disjoint validation pairs only",
        })
    print("evaluation_complete=" + json.dumps({
        "float_tflite_bytes": results["model_files"]["float_tflite"]["file_bytes"],
        "int8_bytes": results["model_files"]["full_integer_int8_tflite"]["file_bytes"],
        "int8_ops": results["model_files"]["full_integer_int8_tflite"]["operators"],
        "int8_calibrated_threshold": threshold_by_runner["tflite_full_int8"],
        "report": str((run_dir / "float_int8_evaluation.json").relative_to(ROOT)),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
