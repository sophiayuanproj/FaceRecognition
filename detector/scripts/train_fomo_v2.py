"""Job C detector trainer v2 (drop next to train_fomo.py in the Job C scripts folder).

Same contract as v1: input [1,96,96,1] gray, output [1,12,12,2] softmax (background, face),
full-INT8 TFLite, same <outdir>/<arm>/seed-N/ layout, so evaluate_development.py,
audit_root_exports.py-style tooling and the deploy path keep working.

What changes versus train_fomo.py (see the notes in the chat for why):
  * receptive-field-matched model  (--arm rf_sepconv_24: RF 95 px vs 31 px)
  * scale augmentation (zoom 0.6x..2.8x) so faces up to ~full frame are seen in training
  * contrast / gamma / blur / noise augmentation (the Nicla image is low-contrast gray)
  * tiny faces (< --min-face-px) are ignored instead of taught
  * PASS negatives and your own Nicla frames are oversampled
  * optional --own-npz from label_own_frames.py (your real camera, your real face)
  * reports, on the INT8 export, how many held-out frames give exactly ONE detection
    at each threshold -- the thing the identity stage needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import fomo_v2_data as D
import train_fomo as base                       # reuse export_root / TRAIN_ROOTS
from common import openmv_fomo_detections
from evaluate_development import invoke

ARMS = ("rf_sepconv_24", "cheap_sepconv_16")
FACE_WEIGHT = 40.0


def build_model(arm: str) -> tf.keras.Model:
    if arm == "cheap_sepconv_16":
        return base.build_model(arm)
    L = tf.keras.layers
    inp = tf.keras.Input((96, 96, 1), name="gray_u8_normalized_0_1", dtype=tf.float32)
    x = inp
    for k, w in enumerate((8, 12, 16), 1):
        x = L.SeparableConv2D(w, 3, strides=2, padding="same", use_bias=False, name=f"rf_sepconv_{k}")(x)
        x = L.BatchNormalization(name=f"rf_bn_{k}")(x)
        x = L.ReLU(max_value=6, name=f"rf_relu6_{k}")(x)
    for k in range(1, 6):                       # five 3x3 context blocks at the 12x12 grid
        x = L.SeparableConv2D(24, 3, padding="same", use_bias=False, name=f"rf_ctx_{k}")(x)
        x = L.BatchNormalization(name=f"rf_ctx_bn_{k}")(x)
        x = L.ReLU(max_value=6, name=f"rf_ctx_relu6_{k}")(x)
    out = L.Conv2D(2, 1, padding="same", activation="softmax", name="fomo_background_face_softmax")(x)
    model = tf.keras.Model(inp, out, name=arm)
    if model.output_shape != (None, 12, 12, 2):
        raise ValueError(model.output_shape)
    return model


RF = {"rf_sepconv_24": D.receptive_field([(3, 2)] * 3 + [(3, 1)] * 5),
      "cheap_sepconv_16": D.receptive_field([(3, 2)] * 3 + [(3, 1)])}


def loss_fn(y_true, y_pred):
    """y_true is (B,12,12,3): background, face, per-cell weight (0 = ignore)."""
    t, w = y_true[..., :2], y_true[..., 2]
    ce = -tf.reduce_sum(t * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1)
    return ce * tf.where(t[..., 1] > 0.5, FACE_WEIGHT, 1.0) * w


class Feed(tf.keras.utils.Sequence):
    def __init__(self, pools, seed, cfg, min_px, batch=64):
        super().__init__()
        self.pools, self.seed, self.cfg, self.min_px, self.batch = pools, seed, cfg, min_px, batch
        self.epoch = 0
        self.items = D.epoch_items(pools, seed, 0)

    def __len__(self):
        return len(self.items) // self.batch

    def __getitem__(self, i):
        part = self.items[i * self.batch:(i + 1) * self.batch]
        return D.make_batch(self.pools, part, self.seed, self.epoch, i, self.cfg, self.min_px)

    def on_epoch_end(self):
        self.epoch += 1
        self.items = D.epoch_items(self.pools, self.seed, self.epoch)


def own_report(int8_path: Path, pools: list[D.Pool]) -> dict:
    """One-detection accuracy on held-out frames, from the exported INT8 model."""
    it = tf.lite.Interpreter(model_path=str(int8_path), num_threads=2)
    it.allocate_tensors()
    rows = {}
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        r = {"pos": 0, "one_correct": 0, "none": 0, "multi": 0, "wrong_place": 0,
             "neg": 0, "neg_false_trigger": 0}
        for pool in pools:
            for img, boxes in zip(pool.images, pool.boxes):
                dets = openmv_fomo_detections(invoke(it, img[..., None])[0], thr)
                if boxes:
                    x1, y1, x2, y2 = boxes[0]
                    c = ((x1 + x2) / 2, (y1 + y2) / 2)
                    r["pos"] += 1
                    if not dets: r["none"] += 1
                    elif len(dets) > 1: r["multi"] += 1
                    else:
                        d = dets[0]["center_input_xy"]
                        near = ((d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2) ** .5 <= max(10.0, 0.35 * max(x2 - x1, y2 - y1))
                        r["one_correct" if near else "wrong_place"] += 1
                else:
                    r["neg"] += 1
                    r["neg_false_trigger"] += int(bool(dets))
        rows[str(thr)] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--arm", choices=ARMS, default="rf_sepconv_24")
    ap.add_argument("--seeds", type=int, nargs="+", default=base.TRAIN_ROOTS[:3])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--own-npz", type=Path, help="output of label_own_frames.py")
    ap.add_argument("--own-repeat", type=int, default=8)
    ap.add_argument("--neg-repeat", type=int, default=4, help="oversampling of PASS no-face scenes")
    ap.add_argument("--min-face-px", type=float, default=6.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = D.AugConfig()
    print(json.dumps({"arm": args.arm, "receptive_field_px_of_96": RF[args.arm]}), flush=True)

    with np.load(args.cache_dir / "train_cache.npz") as z:
        tr_x = z["gray96"]
    with np.load(args.cache_dir / "development_cache.npz") as z:
        dv_x = z["gray96"]
    tr_meta = D.read_jsonl(args.cache_dir / "train_scene_metadata.jsonl")
    dv_meta = D.read_jsonl(args.cache_dir / "development_scene_metadata.jsonl")
    if not (len(tr_x) == len(tr_meta) and len(dv_x) == len(dv_meta)):
        raise ValueError("cache and scene metadata are out of step")
    neg_tr = np.array([bool(m.get("is_real_no_face_negative")) for m in tr_meta])
    neg_dv = np.array([bool(m.get("is_real_no_face_negative")) for m in dv_meta])
    pools = [D.Pool("wider_train", tr_x[~neg_tr], [D.boxes_from_metadata(m) for m, n in zip(tr_meta, neg_tr) if not n], 1),
             D.Pool("pass_neg_train", tr_x[neg_tr], [[] for _ in range(int(neg_tr.sum()))], args.neg_repeat)]
    val_pools = [D.Pool("dev", dv_x, [[] if n else D.boxes_from_metadata(m) for m, n in zip(dv_meta, neg_dv)], 1)]
    own_val_pools = []
    if args.own_npz:
        (ti, tb), (vi, vb) = D.load_own_npz(args.own_npz)
        pools.append(D.Pool("own_train", ti, tb, args.own_repeat))
        own_val = D.Pool("own_val", vi, vb, 1)
        own_val_pools = [own_val]
        val_pools.append(D.Pool("own_val_x3", vi, vb, 3))
        print(json.dumps({"own_train": len(ti), "own_val": len(vi)}), flush=True)
    val_x, val_y = D.static_xy(val_pools, args.min_face_px)
    rep_x = np.concatenate([p.images for p in pools])           # for INT8 calibration

    args.outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        root = args.outdir / args.arm / f"seed-{seed}"
        if not args.force and (root / "root_summary.json").exists():
            print(json.dumps({"event": "root_already_complete", "seed": seed}), flush=True)
            continue
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(args.arm)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
        feed = Feed(pools, seed, cfg, args.min_face_px)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, min_delta=1e-4),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
        hist = model.fit(feed, validation_data=(val_x, val_y), epochs=args.epochs, callbacks=callbacks, verbose=2)
        history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        summary = base.export_root(model, rep_x, root, seed, model.count_params(), history)
        summary.update(arm=args.arm, receptive_field_px=RF[args.arm], v2=True, min_face_px=args.min_face_px,
                       training_items_per_epoch=len(feed.items))
        if own_val_pools:
            summary["own_val_report_full_int8"] = own_report(root / "detector_full_int8.tflite", own_val_pools)
            print(json.dumps(summary["own_val_report_full_int8"], indent=1), flush=True)
        (root / "root_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "root_complete", "seed": seed, "int8_bytes": summary["full_int8_tflite"]["file_bytes"]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
