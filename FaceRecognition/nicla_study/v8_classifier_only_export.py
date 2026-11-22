"""Build and verify a classifier-only H5 export from the selected v7 weights.

This versioned repair path intentionally never attaches the auxiliary embedding
submodel to the classifier. It reads historical model/checkpoint artifacts but
writes only beneath the new v8 export-repair directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import h5py
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "work/project-research-v5/runs/accuracy_search_v1"
PARENT_RUN_ID = "v7_lfw_source_mnv1_a010_ei_scout57101"
PARENT_RESULT = RUN_ROOT / PARENT_RUN_ID / "RESULT.json"
PARENT_WEIGHTS = RUN_ROOT / PARENT_RUN_ID / "best_model.weights.h5"
ORIGINAL_EXPORT = RUN_ROOT / PARENT_RUN_ID / "model_float.h5"
OUTPUT_ROOT = RUN_ROOT / "v8_classifier_only_export_repair_v1"
EXPECTED_ARRAYS = 139
EXPECTED_PARAMETERS = 46809
OUTPUT_CLASSES = ("K01", "K02", "K03", "K04", "K05", "OTHER")

sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/nicla_study"))
from nicla_study import d_campaign as d
from nicla_study.fs import sha256_file


def weight_array_hash(weights: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in weights:
        array = np.asarray(value)
        digest.update(str(array.shape).encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _build_classifier_graph(alpha: float = 0.1, seed: int = 57101,
                            name: str = "v8_classifier_only_mnv1_a010") -> tf.keras.Model:
    """Build the graph-only classifier; no auxiliary embedding model is attached."""
    if float(alpha) != 0.1:
        raise ValueError("the selected v7 parent checkpoint is MobileNetV1 alpha0.1")
    full_reference, backbone, info = d.keras_mobilenet(ROOT, alpha, True, seed)
    input_tensor = tf.keras.Input(shape=(96, 96, 1), dtype=tf.float32, name="gray96_float01")
    base = tf.keras.Model(backbone.input, backbone.output, name="selected_backbone")
    base.trainable = False
    features = base(input_tensor, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    embedding = tf.keras.layers.Dense(64, activation="relu", name="embedding_64")(pooled)
    dropout = tf.keras.layers.Dropout(0.1, name="head_dropout")(embedding)
    logits = tf.keras.layers.Dense(len(OUTPUT_CLASSES), activation=None, name="classifier_logits")(dropout)
    model = tf.keras.Model(input_tensor, logits, name=name)

    # Mirror the v7 checkpoint's partial-finetune trainability before HDF5 load.
    # Keras weights are ordered differently when the backbone is frozen.
    base.trainable = True
    layers = list(base.layers)
    cutoff = max(1, int(np.floor(len(layers) * 0.75)))
    for index, layer in enumerate(layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.get_layer("head_dropout").rate = 0.0

    # Keep only graph-connected layers. Do not attach the source feature model,
    # task embedding submodel, reference 1000-way model, or runner helpers.
    del full_reference, backbone, info
    return model


def build_classifier_only(alpha: float = 0.1, seed: int = 57101,
                          name: str = "v8_classifier_only_mnv1_a010") -> tf.keras.Model:
    """Load selected weights, then transfer arrays into a clean graph-only model.

    The legacy checkpoint's HDF5 layer list includes its auxiliary embedding
    model. Load it into the exact legacy topology first, then copy the selected
    classifier graph's 139 arrays into a graph-only clone by array order.
    """
    source_model, source_embedding_model, source_info = d.build_classifier(
        ROOT, alpha, True, len(OUTPUT_CLASSES), seed,
        "v7_" + PARENT_RUN_ID)
    source_model.get_layer("head_dropout").rate = 0.0
    d.unfreeze_top_fraction(source_model, 0.25)
    source_model.load_weights(str(PARENT_WEIGHTS))
    source_weights = [np.array(value, copy=True) for value in source_model.get_weights()]
    source_hash = weight_array_hash(source_weights)
    graph_only = _build_classifier_graph(alpha, seed, name)
    if len(graph_only.get_weights()) != len(source_weights):
        raise AssertionError("graph-only classifier weight array count differs from selected checkpoint graph")
    graph_only.set_weights(source_weights)
    if weight_array_hash(graph_only.get_weights()) != source_hash:
        raise AssertionError("graph-only classifier did not preserve selected checkpoint arrays")
    del source_model, source_embedding_model, source_info
    return graph_only


def count_h5_weight_datasets(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        datasets = []
        paths = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(obj)
                paths.append(name)

        handle.visititems(visit)
        auxiliary_groups = []
        if "model_weights" in handle:
            for name in handle["model_weights"].keys():
                if name.endswith("_embedding"):
                    auxiliary_groups.append(name)
        return {
            "dataset_arrays": len(datasets),
            "parameter_elements": int(sum(int(dataset.size) for dataset in datasets)),
            "auxiliary_embedding_groups": auxiliary_groups,
            "dataset_paths": paths,
        }


def assert_classifier_contract(model: tf.keras.Model, expected_weight_hash: str | None = None) -> dict:
    weights = model.get_weights()
    array_count = len(weights)
    parameter_count = int(sum(int(np.asarray(weight).size) for weight in weights))
    model_parameter_count = int(model.count_params())
    if array_count != EXPECTED_ARRAYS:
        raise AssertionError(f"classifier weight array count {array_count} != {EXPECTED_ARRAYS}")
    if parameter_count != EXPECTED_PARAMETERS or model_parameter_count != EXPECTED_PARAMETERS:
        raise AssertionError(
            f"classifier parameter count mismatch: weights={parameter_count}, model={model_parameter_count}")
    if hasattr(model, "_d_embedding_model"):
        raise AssertionError("classifier contains the auxiliary embedding submodel")
    actual_hash = weight_array_hash(weights)
    if expected_weight_hash and actual_hash != expected_weight_hash:
        raise AssertionError("classifier weights differ from the selected checkpoint's evaluation weights")
    return {"weight_arrays": array_count, "weight_parameter_elements": parameter_count,
            "model_parameter_count": model_parameter_count, "selected_weight_array_sha256": actual_hash,
            "auxiliary_embedding_model_attached": False}


def inspect_original_export(path: Path) -> dict:
    report = count_h5_weight_datasets(path)
    with h5py.File(path, "r") as handle:
        groups = []
        if "model_weights" in handle:
            for name in handle["model_weights"].keys():
                group = handle["model_weights"][name]
                dataset_count = 0
                parameter_count = 0

                def visit(_path, obj):
                    nonlocal dataset_count, parameter_count
                    if isinstance(obj, h5py.Dataset):
                        dataset_count += 1
                        parameter_count += int(obj.size)

                group.visititems(visit)
                if dataset_count:
                    groups.append({"name": name, "arrays": dataset_count,
                                   "parameter_elements": parameter_count})
    report["top_level_weight_groups"] = groups
    report["sha256"] = sha256_file(path)
    report["bytes"] = path.stat().st_size
    return report


def export_classifier_only(output_dir: Path = OUTPUT_ROOT) -> dict:
    """Create a new immutable export directory; refuse to overwrite any run."""
    output_dir = output_dir.resolve()
    output_dir.relative_to(RUN_ROOT.resolve())
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned export directory: {output_dir}")
    if not PARENT_RESULT.is_file() or not PARENT_WEIGHTS.is_file() or not ORIGINAL_EXPORT.is_file():
        raise FileNotFoundError("selected v7 result, weights, or original export is absent")
    parent_result_sha = sha256_file(PARENT_RESULT)
    parent_weights_sha = sha256_file(PARENT_WEIGHTS)
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    if parent.get("run_id") != PARENT_RUN_ID or parent.get("status") != "complete_scout_host_only":
        raise ValueError("selected v7 parent is not the pinned completed source scout")
    if parent.get("model", {}).get("parameter_count") != EXPECTED_PARAMETERS:
        raise ValueError("parent model parameter count differs from classifier-only export contract")
    prediction_info = parent.get("private_evidence", {})
    prediction_path = (ROOT / prediction_info["per_image_predictions"]).resolve()
    prediction_path.relative_to(ROOT)
    if (not prediction_path.is_file() or
            sha256_file(prediction_path) != prediction_info.get("per_image_predictions_sha256")):
        raise ValueError("parent selected-model prediction evidence is absent or hash-mismatched")
    first_prediction = next((json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()), None)
    expected_weight_hash = first_prediction.get("model_weight_sha256") if first_prediction else None
    if not expected_weight_hash:
        raise ValueError("parent prediction evidence has no selected model weight hash")

    model = build_classifier_only()
    selected = assert_classifier_contract(model, expected_weight_hash)
    saved_weights = [np.array(weight, copy=True) for weight in model.get_weights()]

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "classifier_only_model.h5"
    model.save(str(model_path), include_optimizer=False)
    reloaded = tf.keras.models.load_model(str(model_path), compile=False)
    reloaded_evidence = assert_classifier_contract(reloaded, expected_weight_hash)
    reloaded_weights = reloaded.get_weights()
    if len(saved_weights) != len(reloaded_weights) or any(
            not np.array_equal(before, after) for before, after in zip(saved_weights, reloaded_weights)):
        raise AssertionError("classifier-only H5 reload did not preserve every selected weight array exactly")
    h5_counts = count_h5_weight_datasets(model_path)
    if h5_counts["dataset_arrays"] != EXPECTED_ARRAYS or h5_counts["parameter_elements"] != EXPECTED_PARAMETERS:
        raise AssertionError("classifier-only H5 contains a non-classifier or duplicated weight array")
    if h5_counts["auxiliary_embedding_groups"]:
        raise AssertionError("classifier-only H5 contains an auxiliary embedding-model weight group")

    report = {
        "schema": "classifier-only-export-repair-v1", "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "parent_run_id": PARENT_RUN_ID,
        "parent_result_path": str(PARENT_RESULT.relative_to(ROOT)), "parent_result_sha256": parent_result_sha,
        "selected_weights_path": str(PARENT_WEIGHTS.relative_to(ROOT)), "selected_weights_file_sha256": parent_weights_sha,
        "selected_weights_internal_sha256": expected_weight_hash,
        "original_export_path": str(ORIGINAL_EXPORT.relative_to(ROOT)),
        "original_export_sha256": sha256_file(ORIGINAL_EXPORT),
        "original_export_weight_inventory": inspect_original_export(ORIGINAL_EXPORT),
        "classifier_only_model_path": str(model_path.relative_to(ROOT)),
        "classifier_only_model_sha256": sha256_file(model_path),
        "classifier_only_model_bytes": model_path.stat().st_size,
        "classifier_only_h5_weight_inventory": h5_counts,
        "selected_model_contract": selected,
        "reloaded_model_contract": reloaded_evidence,
        "exact_array_parity_after_reload": True,
        "auxiliary_embedding_model_attached": False,
        "auxiliary_embedding_weight_group_present": False,
    }
    manifest_path = output_dir / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["manifest_path"] = str(manifest_path.relative_to(ROOT))
    report["manifest_sha256"] = sha256_file(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = export_classifier_only(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "classifier_only_model_path", "classifier_only_model_sha256", "classifier_only_model_bytes",
        "classifier_only_h5_weight_inventory", "selected_model_contract", "reloaded_model_contract",
        "exact_array_parity_after_reload", "manifest_path", "manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
