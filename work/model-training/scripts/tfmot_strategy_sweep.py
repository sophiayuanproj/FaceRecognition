#!/usr/bin/env python3
"""Fine-tune pruning/clustering candidates locally and compare validation quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
MT_ROOT = HERE.parents[1]
SCRIPTS = MT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_embedder as training  # noqa: E402
from common import (  # noqa: E402
    image_path, jsonl, load_gray, load_manifest, pair_record_ids, pair_scores,
    remove_exact_duplicate_conflicts, verification_metrics,
)
from compression_sweep import TFLiteRunner, meta, rep_dataset, sha256  # noqa: E402


RUN_DIR = MT_ROOT / "runs/survface_supcon_v2_holdout-clean"
OUTPUT = MT_ROOT / "compression/advanced_strategy_sweep_20261022"
BASE_INT8 = MT_ROOT / "compression/strategy_sweep_20261022/base/base_full_int8_io.tflite"
THREADS = 4


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def training_data():
    split = json.loads((RUN_DIR / "split.json").read_text())
    config = json.loads((RUN_DIR / "train_config.json").read_text())
    all_rows, _, raw_rows, _ = load_manifest(include_olivetti=False)
    holdout = [r for r in all_rows if r["dataset"] == "survface_96" and not r["identity_id"]]
    clean_rows, audit = remove_exact_duplicate_conflicts(raw_rows, holdout)
    excluded = set(split["excluded_exact_duplicate_record_ids"])
    groups = training.grouped_rows(clean_rows, split["train_identity_ids"], excluded)
    images, identity_to_rows, rows = training.load_training_images(
        groups, int(config["max_images_per_train_identity"]), int(split["seed"]),
    )
    identity_names = sorted(identity_to_rows)
    class_labels = {index: identity_to_rows[identity]
                    for index, identity in enumerate(identity_names)}
    expected = int(json.loads((RUN_DIR / "run_summary.json").read_text())["training_image_count"])
    if len(rows) != expected:
        raise ValueError("Training records differ from the selected baseline: %d != %d" % (len(rows), expected))
    return split, config, images, class_labels, rows, audit


def representative_rows(rows, split):
    # The export calibration pool is selected only from the exact selected training records.
    order = list(rows)
    np.random.default_rng(int(split["seed"])).shuffle(order)
    return order[:min(512, len(order))]


def weight_statistics(model):
    arrays = []
    for weight in model.get_weights():
        if getattr(weight, "ndim", 0) >= 2:
            arrays.append(np.asarray(weight))
    total = sum(array.size for array in arrays)
    zeros = sum(int(np.count_nonzero(array == 0)) for array in arrays)
    unique = [int(np.unique(array).size) for array in arrays]
    return {
        "rank2plus_weight_values": int(total),
        "rank2plus_zero_count": int(zeros),
        "rank2plus_zero_fraction": float(zeros / total) if total else None,
        "unique_values_per_weight_array": unique,
        "maximum_unique_values_in_one_weight_array": max(unique) if unique else 0,
        "one_in_four_groups_with_exactly_two_zeros_fraction": None,
    }


def quantize_candidate(model, out_path, representative, use_sparse):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if use_sparse:
        converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter.representative_dataset = lambda: rep_dataset(representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(converter.convert())
    return meta(out_path)


def evaluate(path, pairs, by_record, key):
    runner = TFLiteRunner(path)
    ids = pair_record_ids(pairs)
    vectors = {record_id: runner.run(load_gray(by_record[record_id])) for record_id in ids}
    scores, labels = pair_scores(pairs, vectors)
    metrics = verification_metrics(scores, labels, bootstrap=1000, seed=20261022)
    return {
        "model": meta(path),
        "validation": metrics,
        "validation_pairs": len(pairs),
        "unique_validation_records": len(ids),
        "test_annotations_loaded": False,
        "final_pair_labels_used": False,
    }


def train_candidate(name, base_model, sequence, epochs, output_dir):
    started = time.time()
    if name == "magnitude_prune_50":
        wrapped = tfmot.sparsity.keras.prune_low_magnitude(
            base_model,
            pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=0.50, begin_step=0, frequency=100,
            ),
        )
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    elif name == "structured_prune_2of4":
        wrapped = tfmot.sparsity.keras.prune_low_magnitude(
            base_model,
            sparsity_m_by_n=(2, 4),
        )
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    elif name == "cluster_16":
        wrapped = tfmot.clustering.keras.cluster_weights(
            base_model, number_of_clusters=16,
        )
        callbacks = []
    elif name == "qat_full_int8":
        annotated = tfmot.quantization.keras.quantize_annotate_model(base_model)
        wrapped = tfmot.quantization.keras.quantize_apply(annotated)
        callbacks = []
    else:
        raise ValueError(name)

    model = training.build_supcon_training_model(wrapped, augmentation_seed=20261024)
    history = model.fit(
        sequence, epochs=epochs, verbose=2, callbacks=callbacks,
        workers=1, use_multiprocessing=False,
    )
    if name.startswith("cluster_"):
        stripped = tfmot.clustering.keras.strip_clustering(wrapped)
    elif name.startswith("qat_"):
        stripped = wrapped
    else:
        stripped = tfmot.sparsity.keras.strip_pruning(wrapped)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "fine_tuned.h5"
    stripped.save(checkpoint_path, include_optimizer=False)
    hist = []
    for index in range(len(history.history.get("loss", []))):
        hist.append({
            "epoch": index + 1,
            "loss": float(history.history["loss"][index]),
        })
    write_json(output_dir / "training_history.json", hist)
    return stripped, {
        "strategy": name,
        "epochs_requested": epochs,
        "epochs_completed": len(hist),
        "steps_per_epoch": 200,
        "training_time_seconds": time.time() - started,
        "checkpoint_path": str(checkpoint_path.resolve().relative_to(ROOT.resolve())),
        "checkpoint_sha256": sha256(checkpoint_path),
        "loss_history": hist,
        "weights_after_finetune_and_strip": weight_statistics(stripped),
        "data_scope": "selected exact-deduplicated capped training records and training identities only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--only-strategy", choices=("magnitude_prune_50", "structured_prune_2of4", "cluster_16", "qat_full_int8"), default=None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("refusing to overwrite non-empty output directory: " + str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    tf.random.set_seed(20261022)
    np.random.seed(20261022)

    if not BASE_INT8.is_file():
        raise FileNotFoundError(str(BASE_INT8))
    split, config, images, class_labels, rows, audit = training_data()
    representative = representative_rows(rows, split)
    all_rows, by_record, raw_rows, _ = load_manifest(include_olivetti=False)
    holdout = [r for r in all_rows if r["dataset"] == "survface_96" and not r["identity_id"]]
    clean_rows, _ = remove_exact_duplicate_conflicts(raw_rows, holdout)
    selected_ids = set(split["validation_identity_ids"])
    val_pairs = list(jsonl(RUN_DIR / "validation_pairs.jsonl"))
    if any(by_record[p["left_record_id"]]["identity_id"] not in selected_ids or
           by_record[p["right_record_id"]]["identity_id"] not in selected_ids for p in val_pairs):
        raise ValueError("Validation pairs are not restricted to held-out validation identities")

    result = {
        "protocol": "training-only structured/unstructured pruning and clustering with 5-epoch supervised-contrastive fine-tune",
        "tensorflow": tf.__version__,
        "tensorflow_model_optimization": tfmot.__version__,
        "training_seed": 20261022,
        "split_seed": int(split["seed"]),
        "fine_tune_epochs": args.epochs,
        "steps_per_epoch": 200,
        "training_identity_count": len(split["train_identity_ids"]),
        "training_record_count": len(rows),
        "representative_calibration_records": len(representative),
        "validation_pair_count": len(val_pairs),
        "final_pair_annotations_loaded": False,
        "olivetti_loaded": False,
        "training_audit": {k: v for k, v in audit.items() if k != "excluded_record_ids"},
        "strategies": {},
        "limitations": [
            "Fine-tuning is a five-epoch compression experiment, not another full training campaign.",
            "The final SurvFace pair annotations remain closed.",
            "A lower sparse flatbuffer size is not assumed to accelerate the OpenMV CPU runtime.",
        ],
    }

    baseline = TFLiteRunner(BASE_INT8)
    val_ids = pair_record_ids(val_pairs)
    baseline_vectors = {record_id: baseline.run(load_gray(by_record[record_id])) for record_id in val_ids}
    scores, labels = pair_scores(val_pairs, baseline_vectors)
    result["baseline_full_int8"] = {
        "file": str(BASE_INT8.relative_to(ROOT)),
        "file_bytes": BASE_INT8.stat().st_size,
        "sha256": sha256(BASE_INT8),
        "validation": verification_metrics(scores, labels, bootstrap=1000, seed=20261022),
        "conversion_strategy": "full integer INT8, no pruning or clustering",
    }

    strategy_names = ("magnitude_prune_50", "structured_prune_2of4", "cluster_16", "qat_full_int8")
    if args.only_strategy:
        strategy_names = (args.only_strategy,)
    for name in strategy_names:
        print("starting strategy", name, flush=True)
        base_copy = tf.keras.models.load_model(RUN_DIR / "best_embedding.h5", compile=False)
        sequence = training.BalancedIdentitySequence(
            images, class_labels, list(range(len(class_labels))), 8, 4, seed=20261022, steps=200,
        )
        strategy_dir = args.output_dir / name
        try:
            stripped, training_result = train_candidate(name, base_copy, sequence, args.epochs, strategy_dir)
        except Exception as error:
            result["strategies"][name] = {
                "status": "failed_before_export",
                "error": type(error).__name__ + ": " + str(error)[:700],
            }
            print("strategy failed", name, repr(error), flush=True)
            del base_copy
            continue
        del base_copy
        strategy_result = {"status": "trained", "training": training_result,
                           "exports": {}, "test_annotations_loaded": False}

        for sparse in ([False, True] if name in ("magnitude_prune_50", "structured_prune_2of4") else [False]):
            export_name = "full_int8_sparse_encoding" if sparse else "full_int8_dense_encoding"
            try:
                model_path = strategy_dir / (export_name + ".tflite")
                model_meta = quantize_candidate(stripped, model_path, representative, sparse)
                evaluation = evaluate(model_path, val_pairs, by_record, name + "/" + export_name)
                strategy_result["exports"][export_name] = {
                    **model_meta,
                    "validation": evaluation["validation"],
                    "validation_pairs": len(val_pairs),
                    "host_tflite_size_bytes": model_path.stat().st_size,
                    "flatbuffer_sha256": sha256(model_path),
                    "sparse_encoding_requested": sparse,
                    "openmv_loader_test_pending": True,
                }
                print("exported", name, export_name,
                      "bytes", model_path.stat().st_size,
                      "AUC", evaluation["validation"]["roc_auc"], flush=True)
            except Exception as error:
                strategy_result["exports"][export_name] = {
                    "status": "conversion_or_eval_failed",
                    "error": type(error).__name__ + ": " + str(error)[:800],
                }
                print("export failed", name, export_name, repr(error), flush=True)
        result["strategies"][name] = strategy_result
        write_json(args.output_dir / "advanced_compression_results.partial.json", result)
        del stripped

    qat_result = result["strategies"].get("qat_full_int8")
    if qat_result and qat_result.get("status") == "trained":
        result["qat"] = {
            "status": "fine_tuned_and_exported",
            "fine_tuning": qat_result["training"],
            "exports": qat_result["exports"],
        }
    else:
        try:
            source = tf.keras.models.load_model(RUN_DIR / "best_embedding.h5", compile=False)
            annotated = tfmot.quantization.keras.quantize_annotate_model(source)
            quantized = tfmot.quantization.keras.quantize_apply(annotated)
            result["qat"] = {"status": "supported_for_fine_tuning",
                             "fine_tuning_run": False,
                             "quantized_layer_types": sorted({type(layer).__name__ for layer in quantized.layers})}
        except Exception as error:
            result["qat"] = {"status": "unsupported_by_pinned_tfmot_graph",
                             "error": type(error).__name__ + ": " + str(error)[:1000],
                             "fine_tuning_run": False}
    write_json(args.output_dir / "advanced_compression_results.json", result)
    print("report", args.output_dir / "advanced_compression_results.json", flush=True)


if __name__ == "__main__":
    main()

