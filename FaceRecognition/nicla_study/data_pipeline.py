"""Role-checked loaders for the frozen SurvFace/Olivetti manifests.

Pixels are read from immutable source files and transformed in memory. No
prepared image copies are made, so the source SHA-256 remains the provenance
anchor for every training and evaluation example.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import random
from typing import Iterable, Iterator

from .fs import sha256_file
from .roles import assert_role_access, validate_manifest

SUPPORTED_DATASETS = {"survface_96", "olivetti_96"}
CLASS_ORDER = ["K01", "K02", "K03", "K04", "K05", "OTHER"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSONL at %s:%d" % (path, line_number)) from exc
    return rows


def safe_path(row: dict, workspace: Path) -> Path:
    rel = row.get("relative_path")
    if not isinstance(rel, str) or not rel:
        raise ValueError("manifest row has no relative_path")
    pure = PurePosixPath(rel.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("manifest path is absolute or traverses upward: " + rel)
    root = Path(workspace).resolve()
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("resolved manifest path escapes workspace: " + rel) from exc
    return path


def validate_primary_manifest(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    result = validate_manifest(rows)
    if result["status"] != "pass":
        raise ValueError("invalid role manifest: " + "; ".join(result["errors"][:20]))
    bad_dataset = sorted({row.get("dataset") for row in rows} - SUPPORTED_DATASETS)
    if bad_dataset:
        raise ValueError("unknown dataset names: " + ", ".join(map(str, bad_dataset)))
    if any(row.get("class_index") != CLASS_ORDER.index(row.get("label_id"))
           for row in rows if row.get("label_id") in CLASS_ORDER):
        raise ValueError("class_index does not match frozen class order")
    return result


def select_rows(manifest_path: Path, action: str, *, dataset: str | None = None,
                roles: Iterable[str] | None = None,
                frozen_receipt: dict | None = None) -> list[dict]:
    if dataset is not None and dataset not in SUPPORTED_DATASETS:
        raise ValueError("unsupported dataset: " + dataset)
    if dataset is None:
        raise ValueError("select a dataset explicitly; SurvFace and Olivetti are separate protocols")
    rows = read_jsonl(Path(manifest_path))
    if dataset is not None:
        rows = [row for row in rows if row.get("dataset") == dataset]
    if roles is not None:
        role_set = set(roles)
        rows = [row for row in rows if row.get("role") in role_set]
    assert_role_access(rows, action, frozen_receipt=frozen_receipt)
    return rows


def load_pixels(row: dict, workspace: Path, *, output: str = "gray8",
                quantization: tuple[float, int] | None = None,
                augment: bool = False, seed: int | None = None,
                verify_hash: bool = True, action: str | None = None,
                frozen_receipt: dict | None = None):
    """Load one gray crop as uint8, float32 [0,1], or quantized model input."""
    import numpy as np
    from PIL import Image, ImageFile

    if row.get("presentation_role"):
        raise PermissionError("presentation rows must use the separate capture player")
    if row.get("role"):
        if action is None:
            raise PermissionError("loading a role row requires an explicit data action")
        assert_role_access([row], action, frozen_receipt=frozen_receipt)
    if augment and row.get("role") not in {"train", "unknown_train"}:
        raise PermissionError("augmentation is restricted to train and unknown_train rows")
    path = safe_path(row, workspace)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = row.get("source_sha256", row.get("sha256"))
    if verify_hash:
        actual = sha256_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError("source hash mismatch for record " + str(row.get("record_id")))
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        if image.size != (96, 96) or image.mode != "L":
            raise ValueError("expected source gray8 96x96 for record " + str(row.get("record_id")))
        pixels = np.asarray(image, dtype=np.uint8).copy()
    if augment:
        rng = np.random.default_rng(seed)
        if bool(rng.integers(0, 2)):
            pixels = pixels[:, ::-1].copy()
        gain = float(rng.uniform(0.92, 1.08))
        offset = float(rng.uniform(-8.0, 8.0))
        pixels = np.clip(np.rint(pixels.astype(np.float32) * gain + offset), 0, 255).astype(np.uint8)
    pixels = pixels[:, :, None]
    if output == "gray8":
        return pixels
    normalized = pixels.astype(np.float32) / np.float32(255.0)
    if output == "float32_01":
        return normalized
    if output == "quantized":
        if not quantization:
            raise ValueError("quantized output requires (scale, zero_point)")
        scale, zero_point = float(quantization[0]), int(quantization[1])
        if not scale > 0:
            raise ValueError("quantization scale must be positive")
        return np.clip(np.rint(normalized / scale + zero_point), -128, 127).astype(np.int8)
    raise ValueError("output must be gray8, float32_01, or quantized")


def balanced_class_batches(rows: Iterable[dict], workspace: Path, *, batch_size: int,
                           seed: int, output: str = "float32_01",
                           augment: bool = True) -> Iterator[tuple[object, object, list[dict]]]:
    """Yield deterministic class-balanced batches from training roles only."""
    import numpy as np
    rows = list(rows)
    assert_role_access(rows, "train")
    if {row.get("dataset") for row in rows} != {"survface_96"}:
        raise ValueError("balanced identity batches are defined only for the SurvFace primary task")
    if batch_size <= 0 or batch_size % len(CLASS_ORDER):
        raise ValueError("batch_size must be a positive multiple of six classes")
    by_class = {index: [row for row in rows if row.get("class_index") == index]
                for index in range(len(CLASS_ORDER))}
    missing = [CLASS_ORDER[index] for index, values in by_class.items() if not values]
    if missing:
        raise ValueError("training manifest has no examples for: " + ", ".join(missing))
    rng = random.Random(seed)
    per_class = batch_size // len(CLASS_ORDER)
    for values in by_class.values():
        rng.shuffle(values)
    batches = max((len(values) + per_class - 1) // per_class for values in by_class.values())
    for batch_index in range(batches):
        selected = []
        for class_index in range(len(CLASS_ORDER)):
            pool = by_class[class_index]
            start = batch_index * per_class
            selected.extend(pool[(start + offset) % len(pool)] for offset in range(per_class))
        # Interleave classes so batch order itself is not an identity block.
        chunks = [selected[index * per_class:(index + 1) * per_class]
                  for index in range(len(CLASS_ORDER))]
        interleaved = [chunks[class_index][offset]
                       for offset in range(per_class)
                       for class_index in range(len(CLASS_ORDER))]
        rng.shuffle(interleaved)
        images = np.stack([load_pixels(row, workspace, output=output,
                          augment=augment, seed=seed + batch_index * batch_size + offset,
                          action="train")
                          for offset, row in enumerate(interleaved)])
        labels = np.asarray([row["class_index"] for row in interleaved], dtype=np.int64)
        yield images, labels, interleaved


def balanced_identity_batches(rows: Iterable[dict], workspace: Path, *,
                              identities_per_batch: int = 32, seed: int = 20261101,
                              output: str = "float32_01", augment: bool = True):
    """Yield one image per identity per batch for optional embedding pretraining."""
    import numpy as np
    rows = list(rows)
    assert_role_access(rows, "train")
    if not rows or {row.get("dataset") for row in rows} != {"survface_96"}:
        raise ValueError("identity-balanced pretraining needs SurvFace training rows only")
    by_class = {}
    for row in rows:
        class_index = row.get("class_index")
        if not isinstance(class_index, int) or class_index < 0:
            raise ValueError("pretraining row needs a nonnegative pseudonymous class_index")
        by_class.setdefault(class_index, []).append(row)
    if identities_per_batch < 2 or identities_per_batch > len(by_class):
        raise ValueError("identities_per_batch must be between 2 and available identities")
    rng = random.Random(seed)
    class_indices = sorted(by_class)
    batches = len(class_indices) // identities_per_batch
    for values in by_class.values():
        rng.shuffle(values)
    for batch_index in range(batches):
        start = batch_index * identities_per_batch
        selected_classes = class_indices[start:start + identities_per_batch]
        selected = [rng.choice(by_class[class_index]) for class_index in selected_classes]
        rng.shuffle(selected)
        images = np.stack([load_pixels(row, workspace, output=output, augment=augment,
                         seed=seed + batch_index * identities_per_batch + offset, action="train")
                         for offset, row in enumerate(selected)])
        labels = np.asarray([row["class_index"] for row in selected], dtype=np.int64)
        yield images, labels, selected


def reproducible_manifest_fingerprint(rows: Iterable[dict]) -> str:
    normalized = [{key: row.get(key) for key in
                   ("record_id", "dataset", "role", "identity_partition", "label_id", "class_index", "sha256", "relative_path")}
                  for row in rows]
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"))
                        for row in sorted(normalized, key=lambda value: str(value["record_id"])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
