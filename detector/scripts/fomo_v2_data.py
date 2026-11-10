"""Augmentation + target building for the v2 FOMO detector trainer.

numpy + Pillow only (no TensorFlow), so it can be tested anywhere.
Everything works in the 96x96 detector-input space the board actually feeds the model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SIZE, GRID, STRIDE = 96, 12, 8


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def boxes_from_metadata(row: dict) -> list[tuple[float, float, float, float]]:
    out = []
    for box in row.get("face_boxes", []):
        if box.get("center_visible_in_detector_roi"):
            x1, y1, x2, y2 = [float(v) for v in box["xyxy_detector"]]
            out.append((x1, y1, x2, y2))
    return out


def make_target(boxes, min_px: float) -> np.ndarray:
    """(12,12,3): [background, face, loss_weight]. One foreground cell per face centre.

    The single-cell rule is kept on purpose: the board's NMS treats neighbouring
    foreground cells as separate faces, which would turn every close face into
    UNCERTAIN_MULTIPLE_FACES. Faces smaller than min_px are *ignored* (weight 0 on
    their cell) instead of being taught as faces or as background.
    """
    t = np.zeros((GRID, GRID, 3), np.float32)
    t[..., 0] = 1.0
    t[..., 2] = 1.0
    for x1, y1, x2, y2 in boxes:
        gx, gy = int(((x1 + x2) / 2) // STRIDE), int(((y1 + y2) / 2) // STRIDE)
        if not (0 <= gx < GRID and 0 <= gy < GRID):
            continue
        if max(x2 - x1, y2 - y1) < min_px:
            if t[gy, gx, 1] == 0:
                t[gy, gx, 2] = 0.0
            continue
        t[gy, gx] = (0.0, 1.0, 1.0)
    return t


class AugConfig:
    def __init__(self, zoom_p=0.7, zoom_min=0.6, zoom_max=2.8, flip=True,
                 contrast=(0.45, 1.25), bias=30.0, gamma=(0.7, 1.5),
                 blur_p=0.3, noise_sigmas=(0, 0, 2, 4, 6)):
        self.__dict__.update(locals())
        del self.__dict__["self"]


def _photometric(img: np.ndarray, rng, cfg: AugConfig) -> np.ndarray:
    a = img.astype(np.float32)
    mean = a.mean()
    a = (a - mean) * math.exp(rng.uniform(math.log(cfg.contrast[0]), math.log(cfg.contrast[1]))) + mean
    a = 255.0 * (np.clip(a, 0, 255) / 255.0) ** math.exp(rng.uniform(math.log(cfg.gamma[0]), math.log(cfg.gamma[1])))
    a += rng.uniform(-cfg.bias, cfg.bias)
    a = np.clip(a, 0, 255)
    if rng.random() < cfg.blur_p:
        radius = rng.uniform(0.5, 1.5)
        a = np.asarray(Image.fromarray(a.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius)), np.float32)
    sigma = float(rng.choice(cfg.noise_sigmas))
    if sigma:
        a = a + rng.normal(0.0, sigma, a.shape)
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def augment(img: np.ndarray, boxes, rng, cfg: AugConfig):
    """img: uint8 (96,96). boxes: list of (x1,y1,x2,y2) in 96-px coords. Returns (img, boxes)."""
    boxes = [tuple(b) for b in boxes]
    if rng.random() < cfg.zoom_p:
        zoom = math.exp(rng.uniform(math.log(cfg.zoom_min), math.log(cfg.zoom_max)))
        side = SIZE / zoom                       # source window side, in source px
        if side >= SIZE:                         # zoom out: whole image inside, padded
            x0, y0 = rng.uniform(SIZE - side, 0.0), rng.uniform(SIZE - side, 0.0)
        else:                                    # zoom in: centre on a face if there is one
            if boxes and rng.random() < 0.85:
                x1, y1, x2, y2 = boxes[int(rng.integers(len(boxes)))]
                cx, cy = (x1 + x2) / 2 + rng.uniform(-.3, .3) * side, (y1 + y2) / 2 + rng.uniform(-.3, .3) * side
            else:
                cx, cy = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
            x0 = float(np.clip(cx - side / 2, 0, SIZE - side))
            y0 = float(np.clip(cy - side / 2, 0, SIZE - side))
        s = side / SIZE
        fill = int(img.mean())
        img = np.asarray(Image.fromarray(img, "L").transform(
            (SIZE, SIZE), Image.AFFINE, (s, 0, x0, 0, s, y0), resample=Image.BILINEAR, fillcolor=fill), np.uint8)
        moved = [((x1 - x0) / s, (y1 - y0) / s, (x2 - x0) / s, (y2 - y0) / s) for x1, y1, x2, y2 in boxes]
        boxes = [b for b in moved if 0 <= (b[0] + b[2]) / 2 < SIZE and 0 <= (b[1] + b[3]) / 2 < SIZE]
    if cfg.flip and rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        boxes = [(SIZE - x2, y1, SIZE - x1, y2) for x1, y1, x2, y2 in boxes]
    return _photometric(img, rng, cfg), boxes


class Pool:
    """A set of 96x96 gray images with boxes, repeated `repeat` times per epoch."""

    def __init__(self, name, images, boxes, repeat=1, aug_scale=1.0):
        self.name, self.images, self.boxes, self.repeat = name, np.asarray(images, np.uint8), boxes, int(repeat)
        self.aug_scale = aug_scale  # reserved: lets a pool use gentler zoom if wanted
        assert len(self.images) == len(self.boxes)

    def __len__(self):
        return len(self.images)


def epoch_items(pools: list[Pool], seed: int, epoch: int) -> list[tuple[int, int]]:
    items = [(p, j) for p, pool in enumerate(pools) for _ in range(pool.repeat) for j in range(len(pool))]
    np.random.default_rng([seed, epoch]).shuffle(items)
    return items


def make_batch(pools, items, seed, epoch, batch_index, cfg: AugConfig, min_px: float):
    rng = np.random.default_rng([seed, epoch, batch_index, 7])
    xs, ys = [], []
    for p, j in items:
        img, boxes = augment(pools[p].images[j], pools[p].boxes[j], rng, cfg)
        xs.append(img)
        ys.append(make_target(boxes, min_px))
    return np.stack(xs).astype(np.float32)[..., None] / 255.0, np.stack(ys)


def static_xy(pool_list: list[Pool], min_px: float):
    xs, ys = [], []
    for pool in pool_list:
        for _ in range(pool.repeat):
            for img, boxes in zip(pool.images, pool.boxes):
                xs.append(img)
                ys.append(make_target(boxes, min_px))
    return np.stack(xs).astype(np.float32)[..., None] / 255.0, np.stack(ys)


def receptive_field(layers):
    """layers: [(kernel, stride)] -> receptive field in input px."""
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf


def load_own_npz(path: Path, val_fraction: float = 0.25):
    """Frames labelled by label_own_frames.py -> (train_pool_args, val_pool_args), 96-space boxes.

    Split is by contiguous blocks (last val_fraction of each of positives/negatives)
    because consecutive frames from one session are near-duplicates.
    """
    data = np.load(path, allow_pickle=False)
    frames, boxes240 = data["frames"], data["boxes"]
    imgs = np.stack([np.asarray(Image.fromarray(f, "L").resize((SIZE, SIZE), Image.BILINEAR), np.uint8) for f in frames])
    boxes = [[] if b[0] < 0 else [tuple(float(v) / 2.5 for v in b)] for b in boxes240]
    train_i, val_i = [], []
    for want_pos in (True, False):
        idx = [i for i, b in enumerate(boxes) if bool(b) == want_pos]
        cut = len(idx) - int(round(len(idx) * val_fraction))
        train_i += idx[:cut]
        val_i += idx[cut:]
    pick = lambda ids: (imgs[ids], [boxes[i] for i in ids])
    return pick(train_i), pick(val_i)
