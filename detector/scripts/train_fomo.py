"""Train reproducible local FOMO detectors and export float/full-INT8 TFLite."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


REPOSITORY = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPOSITORY / "work/project-finish-v4"
FACE_RECOGNITION = REPOSITORY / "FaceRecognition"
SEED_REGISTRY = STUDY_ROOT / "config/seed_registry.json"
TRAIN_ROOTS = [20261101, 20261102, 20261103, 20261104, 20261105,
               20261106, 20261107, 20261108, 20261109, 20261110]
ARMS = ("mobilenetv2_alpha_0p1", "cheap_sepconv_16")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_model(arm: str) -> tf.keras.Model:
    # The model tensor is grayscale u8 / 255. The full-INT8 converter sees this
    # same real range and can keep the exact deployment preprocessing in sync.
    input_image = tf.keras.Input((96, 96, 1), name="gray_u8_normalized_0_1", dtype=tf.float32)
    if arm == "mobilenetv2_alpha_0p1":
        # Keras MobileNetV2 implementation, alpha=0.1, random initialization.
        # Keras ImageNet checkpoints do not include alpha=0.1; weights=None is
        # deliberate and recorded. block_3 projects to the FOMO 12x12 grid.
        trunk = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 1), alpha=0.1,
                                                   include_top=False, weights=None)
        x = trunk.get_layer("block_3_project_BN").output
        trunk_model = tf.keras.Model(trunk.input, x, name="mobilenetv2_alpha_0p1_stride8")
        x = trunk_model(tf.keras.layers.Rescaling(2.0, offset=-1.0,
                                                  name="mobilenet_minus1_to1")(input_image))
    elif arm == "cheap_sepconv_16":
        # An explicitly smaller learned trigger, not a person/identity model.
        # Three stride-2 stages yield the same 12x12 FOMO centroid grid.
        x = input_image
        for block, width in enumerate((8, 12, 16), start=1):
            x = tf.keras.layers.SeparableConv2D(width, 3, strides=2, padding="same",
                                                 use_bias=False, name=f"cheap_sepconv_{block}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"cheap_bn_{block}")(x)
            x = tf.keras.layers.ReLU(max_value=6, name=f"cheap_relu6_{block}")(x)
        x = tf.keras.layers.SeparableConv2D(16, 3, padding="same", use_bias=False,
                                             name="cheap_sepconv_refine")(x)
        x = tf.keras.layers.BatchNormalization(name="cheap_bn_refine")(x)
        x = tf.keras.layers.ReLU(max_value=6, name="cheap_relu6_refine")(x)
    else:
        raise ValueError(arm)
    probabilities = tf.keras.layers.Conv2D(2, 1, padding="same", activation="softmax",
                                            name="fomo_background_face_softmax")(x)
    model = tf.keras.Model(input_image, probabilities, name=arm)
    if model.output_shape != (None, 12, 12, 2):
        raise ValueError(f"FOMO grid contract mismatch: {model.output_shape}")
    return model


def weighted_categorical_crossentropy(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    face_weight = tf.where(y_true[..., 1] > 0.5, 40.0, 1.0)
    return cross_entropy * face_weight


def make_dataset(x: np.ndarray, y: np.ndarray, seed: int, training: bool):
    x = x.astype(np.float32)[..., None] / 255.0
    y = y.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)

        def augment(image, target):
            flip = tf.random.uniform(()) < 0.5
            image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
            target = tf.cond(flip, lambda: tf.image.flip_left_right(target), lambda: target)
            image = tf.image.random_brightness(image, max_delta=0.07)
            return tf.clip_by_value(image, 0.0, 1.0), target

        dataset = dataset.map(augment, num_parallel_calls=2, deterministic=True)
    dataset = dataset.batch(64).prefetch(1)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options)


def representative_dataset(train_x: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_x), size=min(256, len(train_x)), replace=False)
    for index in indices:
        yield [train_x[index:index + 1].astype(np.float32)[..., None] / 255.0]


def inspect_tflite(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    ops = sorted({row["op_name"] for row in interpreter._get_ops_details()})
    return {
        "sha256": sha256(path), "file_bytes": path.stat().st_size,
        "input": {"shape": input_info["shape"].tolist(), "dtype": str(input_info["dtype"]),
                  "quantization": list(input_info["quantization"])},
        "output": {"shape": output_info["shape"].tolist(), "dtype": str(output_info["dtype"]),
                   "quantization": list(output_info["quantization"])},
        "operators": ops,
    }


def export_root(model: tf.keras.Model, train_x: np.ndarray, outdir: Path,
                root_seed: int, params: int, history: dict) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    weights_path = outdir / "best_training_weights.weights.h5"
    model.save_weights(weights_path)
    with (outdir / "history.json").open("w") as stream:
        json.dump(history, stream, indent=2, sort_keys=True)

    float_path = outdir / "detector_float.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    float_path.write_bytes(converter.convert())

    int8_path = outdir / "detector_full_int8.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(train_x, root_seed + 7000)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_path.write_bytes(converter.convert())
    (outdir / "labels.txt").write_text("background\nface\n", encoding="utf-8")
    result = {
        "root_seed": root_seed, "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "epochs_completed": len(history["loss"]), "parameter_count": int(params),
        "weights_checkpoint": {"path": str(weights_path), "sha256": sha256(weights_path),
                               "deployable": False, "note": "training weights only; TFLite exports are deployment artifacts"},
        "float_tflite": inspect_tflite(float_path),
        "full_int8_tflite": inspect_tflite(int8_path),
        "labels_sha256": sha256(outdir / "labels.txt"),
        "class_semantics": ["background", "face"],
    }
    (outdir / "root_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def train_arm(arm: str, x: np.ndarray, y: np.ndarray, dev_x: np.ndarray,
              dev_y: np.ndarray, roots: list[int], output: Path,
              epochs: int, force: bool) -> list[dict]:
    summaries = []
    for root_seed in roots:
        root_dir = output / arm / f"seed-{root_seed}"
        summary_file = root_dir / "root_summary.json"
        if not force and summary_file.exists():
            try:
                old = json.loads(summary_file.read_text())
                for rel in ("detector_float.tflite", "detector_full_int8.tflite", "labels.txt"):
                    if not (root_dir / rel).is_file():
                        raise FileNotFoundError(rel)
                summaries.append(old)
                print(json.dumps({"event": "root_already_complete", "arm": arm, "root_seed": root_seed}), flush=True)
                continue
            except Exception:
                pass
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(root_seed)
        np.random.seed(root_seed)
        random.seed(root_seed)
        model = build_model(arm)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=weighted_categorical_crossentropy)
        train_ds = make_dataset(x, y, root_seed, training=True)
        dev_ds = make_dataset(dev_x, dev_y, root_seed, training=False)
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                      restore_best_weights=True, min_delta=1e-4)
        history_obj = model.fit(train_ds, validation_data=dev_ds, epochs=epochs,
                                callbacks=[callback], verbose=2)
        history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
        root_summary = export_root(model, x, root_dir, root_seed, model.count_params(), history)
        root_summary["arm"] = arm
        root_summary["training_scene_count"] = int(len(x))
        root_summary["development_scene_count"] = int(len(dev_x))
        root_summary["training_seed_registry"] = str(SEED_REGISTRY)
        summary_file.write_text(json.dumps(root_summary, indent=2, sort_keys=True) + "\n")
        summaries.append(root_summary)
        print(json.dumps({"event": "root_complete", "arm": arm, "root_seed": root_seed,
                          "best_epoch": root_summary["best_epoch"],
                          "int8_bytes": root_summary["full_int8_tflite"]["file_bytes"]}), flush=True)
        del model, history_obj, train_ds, dev_ds
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(FACE_RECOGNITION))
    from nicla_study.state import resource_lease

    seed_registry = json.loads(SEED_REGISTRY.read_text())
    registry_roots = [row["root_seed"] for row in seed_registry["training_roots"]]
    if registry_roots != TRAIN_ROOTS:
        raise ValueError(f"Shared seed registry changed: {registry_roots}")
    with np.load(args.cache_dir / "train_cache.npz") as train:
        x, y = train["gray96"], train["targets"]
    with np.load(args.cache_dir / "development_cache.npz") as development:
        dev_x, dev_y = development["gray96"], development["targets"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    global_report = {
        "schema": "job-c-fomo-training-1", "method": "local TensorFlow/Keras 2.10.1 FOMO-style centroid grid training",
        "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "2.10.x",
        "external_route_assessment": {
            "edge_impulse_open_fomo": "Inspected the official FOMO block's centroid-grid labels and background/face semantics. Its published example path is PyTorch/ExecuTorch .pte, not a TFLite/OpenMV exporter for MobileNetV2 alpha=0.1; used the local Keras/TFLite route to preserve the requested Nicla runtime contract.",
            "reference_url": "https://github.com/edgeimpulse/executorch-pytorch-object-detection-fomo-block",
        },
        "input": {"shape": [1, 96, 96, 1], "pixel_format": "gray8", "normalization": "u8/255", "output_grid": [1, 12, 12, 2]},
        "labels": ["background", "face"], "training_scenes": int(len(x)),
        "development_scenes": int(len(dev_x)), "epochs_max": args.epochs,
        "root_seeds": TRAIN_ROOTS, "arms": args.arms,
        "backbone_provenance": {
            "implementation": "tf.keras.applications.MobileNetV2 from installed TensorFlow/Keras 2.10.1",
            "alpha": 0.1, "weights": None,
            "pretrained_checkpoint": "none; Keras ImageNet weights do not provide alpha=0.1, so this arm trains from random initialization",
            "implementation_source": inspect.getsourcefile(tf.keras.applications.MobileNetV2),
            "implementation_source_sha256": sha256(Path(inspect.getsourcefile(tf.keras.applications.MobileNetV2))),
        },
        "optimization": {"optimizer": "Adam", "learning_rate": 0.001,
                         "face_cell_loss_weight": 40, "batch_size": 64,
                         "augmentations": ["horizontal flip", "brightness ±0.07"]},
        "export": {"float_tflite": True, "full_integer_quantization": True,
                   "representative_images": "random subset from WIDER train + PASS train only"},
    }
    (args.outdir / "training_method.json").write_text(json.dumps(global_report, indent=2, sort_keys=True) + "\n")
    all_summaries = []
    with resource_lease(STUDY_ROOT, "heavy_training", owner="job_C_train_mac", timeout_s=0.0) as lease:
        (args.outdir / "training_lease.json").write_text(json.dumps(lease, indent=2, sort_keys=True) + "\n")
        for arm in args.arms:
            all_summaries.extend(train_arm(arm, x, y, dev_x, dev_y, TRAIN_ROOTS,
                                           args.outdir, args.epochs, args.force))
    summary = {"schema": "job-c-fomo-ten-root-summary-1", "root_count_by_arm": {
                    arm: sum(row.get("arm") == arm for row in all_summaries) for arm in args.arms},
               "all_roots": all_summaries}
    (args.outdir / "ten_root_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "training_complete", "root_count_by_arm": summary["root_count_by_arm"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
