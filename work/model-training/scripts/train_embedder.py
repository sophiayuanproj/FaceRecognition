#!/usr/bin/env python3
"""Train a compact identity-supervised face embedding model locally."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

from common import (
    DATA_ROOT, MT_ROOT, ROOT, embed_keras, exact_duplicate_audit,
    identity_split, image_path, jsonl, load_manifest, make_validation_pairs,
    pair_scores, perceptual_cross_split_candidates, remove_exact_duplicate_conflicts,
    verification_metrics,
    write_json, write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--split-seed", type=int, default=None,
                   help="Fix the identity split and validation pairs independently of the training seed")
    p.add_argument("--augmentation-seed", type=int, default=None,
                   help="Seed training-only augmentation independently of initialization and sampling")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3,
                   help="Stop after this many epochs without validation-AUC improvement")
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--max-images-per-train-identity", type=int, default=24)
    p.add_argument("--identities-per-batch", type=int, default=8)
    p.add_argument("--images-per-identity", type=int, default=4)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--model-variant", choices=("base", "tiny"), default="base")
    p.add_argument("--objective", choices=("softmax", "supcon"), default="softmax")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="Override balanced batches per epoch; 0 covers each training identity once on average")
    p.add_argument("--train-identity-limit", type=int, default=0,
                   help="Debug only; uniformly sampled identity subset, never use for final run")
    p.add_argument("--output", type=Path, default=MT_ROOT / "runs/survface_mobile_v1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--intra-threads", type=int, default=12)
    p.add_argument("--inter-threads", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="One short pipeline smoke run; not a model result")
    return p.parse_args()


def set_seeds(seed, threads, inter_threads):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except RuntimeError:
        pass
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def build_embedder(embedding_dim, variant="base"):
    stem_filters, block_filters = ((8, (12, 16, 24, 32)) if variant == "tiny"
                                   else (16, (24, 32, 48, 64)))
    inputs = tf.keras.Input((96, 96, 1), dtype=tf.float32, name="gray_face_0_1")
    x = tf.keras.layers.Conv2D(stem_filters, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="stem_relu")(x)
    for block, (filters, stride) in enumerate(zip(block_filters, (2, 2, 2, 1)), start=1):
        x = tf.keras.layers.SeparableConv2D(
            filters, 3, strides=stride, padding="same", use_bias=False,
            name="sepconv_%d" % block,
        )(x)
        x = tf.keras.layers.BatchNormalization(name="sepconv_bn_%d" % block)(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name="sepconv_relu_%d" % block)(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    outputs = tf.keras.layers.Dense(embedding_dim, use_bias=False, name="embedding_raw")(x)
    return tf.keras.Model(inputs, outputs, name="survface_mobile_embedding")


def build_training_model(embedder, num_classes, augmentation_seed=101):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=augmentation_seed),
        tf.keras.layers.RandomTranslation(0.015, 0.015, fill_mode="nearest", seed=augmentation_seed + 1),
        tf.keras.layers.RandomContrast(0.12, seed=augmentation_seed + 2),
    ], name="training_only_augmentation")
    inputs = tf.keras.Input((96, 96, 1), dtype=tf.float32, name="gray_face_0_1")
    embeddings = embedder(augmentation(inputs))
    logits = tf.keras.layers.Dense(num_classes, name="identity_logits")(embeddings)
    model = tf.keras.Model(inputs, logits, name="identity_supervised_training_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="closed_set_top1")],
    )
    return model


@tf.keras.utils.register_keras_serializable(package="FaceRecognition")
def supervised_contrastive_loss(labels, embeddings):
    """Supervised contrastive loss with multiple positives per balanced batch."""
    labels = tf.reshape(tf.cast(labels, tf.int32), (-1, 1))
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    temperature = tf.constant(0.1, dtype=embeddings.dtype)
    logits = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
    batch_size = tf.shape(logits)[0]
    self_mask = tf.eye(batch_size, dtype=embeddings.dtype)
    positive_mask = tf.cast(tf.equal(labels, tf.transpose(labels)), embeddings.dtype) * (1.0 - self_mask)
    logits_mask = 1.0 - self_mask
    logits = logits - tf.stop_gradient(tf.reduce_max(
        tf.where(self_mask > 0, tf.fill(tf.shape(logits), tf.cast(-1e9, logits.dtype)), logits),
        axis=1, keepdims=True,
    ))
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)
    positive_count = tf.reduce_sum(positive_mask, axis=1)
    mean_positive_log_prob = tf.reduce_sum(positive_mask * log_prob, axis=1) / tf.maximum(positive_count, 1.0)
    valid_rows = tf.cast(positive_count > 0, embeddings.dtype)
    return -tf.reduce_sum(mean_positive_log_prob * valid_rows) / tf.maximum(tf.reduce_sum(valid_rows), 1.0)


def build_supcon_training_model(embedder, augmentation_seed=101):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=augmentation_seed),
        tf.keras.layers.RandomTranslation(0.015, 0.015, fill_mode="nearest", seed=augmentation_seed + 1),
        tf.keras.layers.RandomContrast(0.12, seed=augmentation_seed + 2),
    ], name="training_only_augmentation")
    inputs = tf.keras.Input((96, 96, 1), dtype=tf.float32, name="gray_face_0_1")
    embeddings = embedder(augmentation(inputs))
    model = tf.keras.Model(inputs, embeddings, name="supcon_training_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=supervised_contrastive_loss)
    return model


class BalancedIdentitySequence(tf.keras.utils.Sequence):
    """Samples identity classes uniformly, then K images within each class."""

    def __init__(self, images, class_labels, class_ids, ids_per_batch, images_per_identity,
                 seed, steps=0):
        self.images = images
        self.class_labels = class_labels
        self.class_ids = list(class_ids)
        self.ids_per_batch = ids_per_batch
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.epoch = 0
        self.steps = steps or max(1, int(np.ceil(len(self.class_ids) / ids_per_batch)))
        self.order = None
        self.on_epoch_end()

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.order = rng.permutation(self.class_ids)
        self.epoch += 1

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch + index)
        start = (index * self.ids_per_batch) % max(1, len(self.order))
        selected = list(self.order[start:start + self.ids_per_batch])
        if len(selected) < self.ids_per_batch:
            selected += list(rng.choice(self.class_ids, self.ids_per_batch - len(selected), replace=False))
        picks = []
        for class_id in selected:
            indices = self.class_labels[class_id]
            picks.extend(rng.choice(indices, self.images_per_identity,
                                    replace=len(indices) < self.images_per_identity).tolist())
        batch = self.images[picks].astype(np.float32) / 255.0
        labels = np.repeat(np.asarray(selected, dtype=np.int32), self.images_per_identity)
        return batch, labels


def grouped_rows(rows, identity_ids, excluded=None):
    allowed = set(identity_ids)
    excluded = excluded or set()
    result = defaultdict(list)
    for row in rows:
        if row["identity_id"] in allowed and row["record_id"] not in excluded:
            result[row["identity_id"]].append(row)
    return result


def load_training_images(groups, max_per_identity, seed):
    rng = random.Random(seed)
    image_rows = []
    row_groups = {}
    for identity in sorted(groups):
        candidates = sorted(groups[identity], key=lambda r: r["record_id"])
        if len(candidates) > max_per_identity:
            candidates = rng.sample(candidates, max_per_identity)
        row_groups[identity] = candidates
        image_rows.extend(candidates)
    images = np.empty((len(image_rows), 96, 96, 1), dtype=np.uint8)
    labels_by_identity = defaultdict(list)
    for index, row in enumerate(image_rows):
        with Image.open(image_path(row)) as source:
            gray = source.convert("L")
            if gray.size != (96, 96):
                raise ValueError("Unexpected image shape in prepared manifest")
            images[index, :, :, 0] = np.asarray(gray, dtype=np.uint8)
        labels_by_identity[row["identity_id"]].append(index)
        if (index + 1) % 10000 == 0:
            print("loaded_training_images=%d/%d" % (index + 1, len(image_rows)), flush=True)
    return images, labels_by_identity, image_rows


class VerificationCheckpointCallback(tf.keras.callbacks.Callback):
    """Evaluate identity-disjoint pairs, save the best feature model, and stop early."""

    def __init__(self, embedder, validation_pairs, record_ids, by_record,
                 history_path, last_model_path, best_embedder_path, patience=3):
        super().__init__()
        self.embedder = embedder
        self.validation_pairs = validation_pairs
        self.record_ids = record_ids
        self.by_record = by_record
        self.history_path = history_path
        self.last_model_path = last_model_path
        self.best_embedder_path = best_embedder_path
        self.patience = patience
        self.best_auc = -1.0
        self.history = []
        if history_path.is_file():
            for line in history_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                    self.history.append(item)
                    self.best_auc = max(self.best_auc, item["validation_verification"]["roc_auc"] or -1.0)
                except Exception:
                    pass
        last_best = max((index for index, item in enumerate(self.history)
                         if item.get("best_checkpoint_saved", False)), default=-1)
        self.epochs_without_improvement = len(self.history) - last_best - 1
        self.epoch_started = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        embeddings = embed_keras(self.embedder, self.record_ids, self.by_record)
        scores, labels = pair_scores(self.validation_pairs, embeddings)
        metrics = verification_metrics(scores, labels, bootstrap=0)
        record = {
            "epoch": epoch + 1,
            "seconds": round(time.time() - self.epoch_started, 3) if self.epoch_started else None,
            "train_loss": float(logs.get("loss", 0.0)),
            "train_closed_set_top1": (float(logs["closed_set_top1"])
                                      if "closed_set_top1" in logs else None),
            "validation_verification": metrics,
        }
        del embeddings
        current_auc = metrics["roc_auc"] or -1.0
        if current_auc > self.best_auc + 1e-5:
            self.best_auc = current_auc
            self.epochs_without_improvement = 0
            self.embedder.save(self.best_embedder_path, include_optimizer=False)
            record["best_checkpoint_saved"] = True
        else:
            self.epochs_without_improvement += 1
            record["best_checkpoint_saved"] = False
        self.model.save(self.last_model_path, include_optimizer=True)
        with self.history_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.history.append(record)
        print("epoch=%d val_auc=%.5f val_eer=%.5f val_tar_at_val_far=%.5f best=%s" % (
            epoch + 1, metrics["roc_auc"], metrics["eer"],
            metrics["tar_at_decision_threshold"], record["best_checkpoint_saved"],
        ), flush=True)
        if self.epochs_without_improvement >= self.patience:
            print("early_stopping=true patience=%d" % self.patience, flush=True)
            self.model.stop_training = True


def main():
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    augmentation_seed = args.seed if args.augmentation_seed is None else args.augmentation_seed
    if args.smoke:
        args.epochs = 1
        args.steps_per_epoch = 2
        args.train_identity_limit = min(args.train_identity_limit or 64, 64)
        args.output = Path(args.output) / "smoke"
    if not 0.05 <= args.validation_fraction <= 0.4:
        raise ValueError("validation-fraction must be in [0.05,0.4]")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    full_run_started = time.time()
    set_seeds(args.seed, args.intra_threads, args.inter_threads)
    previous_summary = {}
    if args.resume and (args.output / "run_summary.json").is_file():
        previous_summary = json.loads((args.output / "run_summary.json").read_text())

    all_rows, by_record, raw_surv_rows, _ = load_manifest()
    unlabeled_survface_rows = [row for row in all_rows
                               if row["dataset"] == "survface_96" and not row["identity_id"]]
    surv_rows, global_duplicate_audit = remove_exact_duplicate_conflicts(
        raw_surv_rows, unlabeled_survface_rows,
    )
    train_ids, val_ids, all_counts = identity_split(
        surv_rows, seed=split_seed, fraction=args.validation_fraction,
    )
    duplicate_audit = exact_duplicate_audit(surv_rows, train_ids, val_ids)
    excluded = set(duplicate_audit["excluded_validation_record_ids"])
    phash_audit = perceptual_cross_split_candidates(train_ids, val_ids)
    train_groups = grouped_rows(surv_rows, train_ids)
    val_groups = grouped_rows(surv_rows, val_ids, excluded)
    val_pairs, val_identity_count = make_validation_pairs(
        surv_rows, val_ids, excluded, seed=split_seed,
    )
    if not val_pairs:
        raise ValueError("No identity-disjoint validation pairs could be constructed")
    split = {
        "seed": split_seed,
        "split_seed": split_seed,
        "training_seed": args.seed,
        "method": "identity-disjoint 80/20 split, stratified by image-count decile",
        "dataset": "survface_96 training_set only",
        "training_identity_count": len(train_ids),
        "validation_identity_count": len(val_ids),
        "validation_identity_with_positive_pairs": val_identity_count,
        "training_images_per_identity_before_cap": {
            "min": min(all_counts[i] for i in train_ids),
            "median": float(np.median([all_counts[i] for i in train_ids])),
            "max": max(all_counts[i] for i in train_ids),
        },
        "train_identity_ids": train_ids,
        "validation_identity_ids": val_ids,
        "exact_duplicate_audit": {k: v for k, v in duplicate_audit.items() if k != "excluded_validation_record_ids"},
        "perceptual_hash_audit": {k: v for k, v in phash_audit.items() if k != "candidate_pair_record_ids"},
        "perceptual_cross_split_candidate_pairs": phash_audit.get("candidate_pair_record_ids", []),
        "validation_pair_count": len(val_pairs),
        "validation_positive_pairs": sum(p["pair_type"] == "positive" for p in val_pairs),
        "validation_negative_pairs": sum(p["pair_type"] == "negative" for p in val_pairs),
        "validation_pairs_use_only_held_out_training_set_identities": True,
        "global_exact_duplicate_audit": {k: v for k, v in global_duplicate_audit.items() if k != "excluded_record_ids"},
        "excluded_exact_duplicate_record_ids": global_duplicate_audit["excluded_record_ids"],
        "training_records_after_exact_hash_cleanup": len(surv_rows),
    }
    write_json(args.output / "split.json", split)
    write_jsonl(args.output / "validation_pairs.jsonl", val_pairs)
    train_config = {
        "seed": args.seed, "training_seed": args.seed,
        "split_seed": split_seed, "augmentation_seed": augmentation_seed,
        "epochs_requested": args.epochs, "early_stopping_patience": args.patience,
        "validation_fraction": args.validation_fraction,
        "max_images_per_train_identity": args.max_images_per_train_identity,
        "identities_per_batch": args.identities_per_batch,
        "images_per_identity": args.images_per_identity,
        "embedding_dim": args.embedding_dim,
        "model_variant": args.model_variant,
        "train_identity_limit": args.train_identity_limit,
        "intra_threads": args.intra_threads,
        "inter_threads": args.inter_threads,
        "objective": ("supervised contrastive loss" if args.objective == "supcon" else "identity-supervised sparse softmax cross-entropy") + " with a %d-D feature head" % args.embedding_dim,
        "augmentation": ["horizontal flip", "translation <= 1.5%", "contrast +/- 12%"],
        "training_only_augmentation": True,
        "dataset_paths_local_only": True,
    }
    config_path = args.output / "train_config.json"
    if args.resume and config_path.is_file():
        previous_config = json.loads(config_path.read_text())
        for key in ("seed", "training_seed", "split_seed", "augmentation_seed",
                    "validation_fraction", "max_images_per_train_identity",
                    "identities_per_batch", "images_per_identity", "embedding_dim",
                    "model_variant", "train_identity_limit", "objective",
                    "early_stopping_patience", "intra_threads", "inter_threads"):
            if key not in previous_config and key in ("intra_threads", "inter_threads"):
                continue
            if previous_config.get(key) != train_config.get(key):
                raise ValueError("Resume config differs for " + key)
    write_json(config_path, train_config)

    train_ids = list(train_ids)
    if args.train_identity_limit:
        sampler = random.Random(args.seed + 31)
        train_ids = sorted(sampler.sample(train_ids, min(args.train_identity_limit, len(train_ids))))
    train_groups = {identity: train_groups[identity] for identity in train_ids}
    print("split_train_ids=%d split_validation_ids=%d validation_pairs=%d exact_cross_split_hashes=%d pHash_cross_split_candidates=%s" % (
        len(split["train_identity_ids"]), len(split["validation_identity_ids"]), len(val_pairs),
        duplicate_audit["train_validation_shared_sha256_group_count"],
        phash_audit["cross_split_candidate_pair_count"],
    ), flush=True)

    load_start = time.time()
    images, class_rows, selected_train_rows = load_training_images(
        train_groups, args.max_images_per_train_identity, split_seed,
    )
    image_load_seconds = time.time() - load_start
    print("training_image_load_seconds=%.1f selected_train_images=%d" % (image_load_seconds, len(selected_train_rows)), flush=True)

    class_ids = sorted(class_rows)
    class_to_index = {identity: index for index, identity in enumerate(class_ids)}
    labels_by_index = {class_to_index[identity]: class_rows[identity] for identity in class_ids}
    embedder = build_embedder(args.embedding_dim, args.model_variant)
    last_model_path = args.output / "last_training_model.h5"
    best_embedder_path = args.output / "best_embedding.h5"
    if args.resume and last_model_path.is_file():
        training_model = tf.keras.models.load_model(
            last_model_path, custom_objects={"supervised_contrastive_loss": supervised_contrastive_loss},
        )
        embedder = training_model.get_layer("survface_mobile_embedding")
        expected_output_dim = args.embedding_dim if args.objective == "supcon" else len(class_ids)
        if training_model.output_shape[-1] != expected_output_dim:
            raise ValueError("Resume checkpoint class count differs from split/config")
        print("resumed_checkpoint=true", flush=True)
    else:
        training_model = (build_supcon_training_model(embedder, augmentation_seed)
                          if args.objective == "supcon" else
                          build_training_model(embedder, len(class_ids), augmentation_seed))
    sequence = BalancedIdentitySequence(
        images, labels_by_index, list(range(len(class_ids))), args.identities_per_batch,
        args.images_per_identity, args.seed,
        steps=args.steps_per_epoch,
    )
    pair_record_ids = sorted({p["left_record_id"] for p in val_pairs} | {p["right_record_id"] for p in val_pairs})
    val_metrics_path = args.output / "validation_history.jsonl"
    callback = VerificationCheckpointCallback(
        embedder, val_pairs, pair_record_ids, by_record, val_metrics_path,
        last_model_path, best_embedder_path, patience=args.patience,
    )
    initial_epoch = len(callback.history)
    sequence.epoch = initial_epoch
    sequence.on_epoch_end()
    training_model.fit(
        sequence, initial_epoch=initial_epoch, epochs=args.epochs, verbose=2, shuffle=False,
        workers=0, use_multiprocessing=False, max_queue_size=1,
        callbacks=[callback],
    )
    if not best_embedder_path.is_file():
        embedder.save(best_embedder_path, include_optimizer=False)
    resumed_seconds = float(previous_summary.get("runtime_seconds", 0.0))
    total_seconds = resumed_seconds + (time.time() - full_run_started)
    final_config = {
        "status": "smoke_only" if args.smoke else "trained",
        "runtime_seconds": round(total_seconds, 3),
        "resume_previous_runtime_seconds": round(resumed_seconds, 3),
        "host": "macOS Darwin 25.6.0 x86_64 Intel Core i9; no discrete GPU used",
        "tensorflow_version": tf.__version__,
        "python_version": sys.version.split()[0],
        "training_seed": args.seed,
        "split_seed": split_seed,
        "augmentation_seed": augmentation_seed,
        "training_identity_count": len(class_ids),
        "training_image_count": len(selected_train_rows),
        "training_image_load_seconds": round(
            float(previous_summary.get("training_image_load_seconds", 0.0)) + image_load_seconds, 3),
        "validation_identity_count": val_identity_count,
        "embedding_dimension": args.embedding_dim,
        "model_variant": args.model_variant,
        "objective": args.objective,
        "best_validation_auc": callback.best_auc,
        "epochs_completed": len(callback.history),
        "early_stopping_patience": args.patience,
        "intra_threads": args.intra_threads,
        "inter_threads": args.inter_threads,
        "checkpoint": str(best_embedder_path.relative_to(ROOT)),
        "final_checkpoint": str(last_model_path.relative_to(ROOT)),
    }
    write_json(args.output / "run_summary.json", final_config)
    print("training_complete=" + json.dumps(final_config, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
