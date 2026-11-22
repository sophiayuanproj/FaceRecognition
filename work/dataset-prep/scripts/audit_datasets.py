#!/usr/bin/env python3
"""Inventory, safely extract, and build a local JSONL image manifest for ZIP datasets.

Uses the Python standard library plus Pillow when available (falling back to
macOS `sips`) for image metadata. It never resizes, converts, embeds, or
uploads image data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import median

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional, macOS sips remains the fallback
    Image = None

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw"
EXTRACTED = BASE / "extracted"
PREPARED = BASE / "prepared"
REPORTS = BASE / "reports"
MANIFEST = REPORTS / "manifest.jsonl"
SUMMARY = REPORTS / "dataset-summary.json"
INVENTORY = REPORTS / "archive-inventory.json"
TREE = REPORTS / "archive-tree.jsonl"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pnm"}
NESTED_ARCHIVE_SUFFIXES = {".7z", ".gz", ".rar", ".tar", ".tgz", ".zip"}
DOC_PREFIXES = ("readme", "license", "licence", "copying", "notice", "citation", "metadata", "dataset_info")
NON_LABEL_DIRS = {"image", "images", "train", "training", "training_set", "test", "testing", "test_set", "face_verification_test_set", "val", "valid", "validation", "data"}
SOURCE_FILE_IDS = {
    "olivetti_96": "1fUs7W4lk6dnqf-erJhvaPO0ty6OP2dGN",
    "survface_96": "1lUl0xDyijYE4isG6lJ5DVb2VNwU58TUq",
}
SIPS = Path("/usr/bin/sips")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name: str) -> tuple[PurePosixPath, str | None]:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("empty or non-portable archive path")
    path = PurePosixPath(name)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", name) or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute or traversing archive path")
    normalized = "/".join(path.parts)
    return path, normalized


def inspect_archive(path: Path) -> dict:
    result: dict = {
        "archive": path.name,
        "archive_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "crc_bad_entry": None,
        "crc_checked": False,
        "entry_count": 0,
        "file_count": 0,
        "directory_count": 0,
        "compressed_bytes": 0,
        "uncompressed_bytes": 0,
        "top_level": [],
        "file_extensions": {},
        "documentation_files": [],
        "nested_archives": [],
        "unsafe_entries": 0,
        "symlink_entries": 0,
        "special_entries": 0,
        "encrypted_entries": 0,
        "inventory_error": None,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            result["entry_count"] = len(infos)
            tops: set[str] = set()
            extensions: Counter[str] = Counter()
            seen: set[str] = set()
            file_names: set[str] = set()
            safe_infos: list[tuple[zipfile.ZipInfo, PurePosixPath, str]] = []
            for info in infos:
                try:
                    member, normalized = safe_member_name(info.filename)
                except ValueError:
                    result["unsafe_entries"] += 1
                    continue
                case_key = normalized.casefold()
                if case_key in seen:
                    result["unsafe_entries"] += 1
                    continue
                seen.add(case_key)
                tops.add(member.parts[0])
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode):
                    result["symlink_entries"] += 1
                elif file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    result["special_entries"] += 1
                if info.flag_bits & 0x1:
                    result["encrypted_entries"] += 1
                if info.is_dir():
                    result["directory_count"] += 1
                else:
                    result["file_count"] += 1
                    result["compressed_bytes"] += info.compress_size
                    result["uncompressed_bytes"] += info.file_size
                    extensions[member.suffix.lower() or "[no extension]"] += 1
                    if member.name.lower().startswith(DOC_PREFIXES) or member.suffix.lower() in {".json", ".csv", ".xml", ".yaml", ".yml"}:
                        result["documentation_files"].append(normalized)
                    if member.suffix.lower() in NESTED_ARCHIVE_SUFFIXES or member.name.lower().endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
                        result["nested_archives"].append(normalized)
                    file_names.add(case_key)
                safe_infos.append((info, member, normalized))
            # Detect file-as-parent conflicts before writing any member.
            for _, member, _ in safe_infos:
                for parent in member.parents:
                    if parent == PurePosixPath("."):
                        continue
                    if "/".join(parent.parts).casefold() in file_names:
                        result["unsafe_entries"] += 1
                        break
            result["top_level"] = sorted(tops)
            result["file_extensions"] = dict(sorted(extensions.items()))
            result["crc_bad_entry"] = archive.testzip()
            result["crc_checked"] = True
            result["_safe_infos"] = safe_infos
    except Exception as exc:  # Recorded as a bounded diagnostic, not a traceback containing filenames.
        result["inventory_error"] = type(exc).__name__
    return result


def safe_extract(path: Path, inventory: dict, destination: Path, max_expanded_bytes: int) -> tuple[bool, str | None]:
    if inventory.get("inventory_error"):
        return False, "archive could not be inventoried"
    if inventory.get("crc_bad_entry") is not None:
        return False, "CRC verification failed"
    if inventory.get("unsafe_entries") or inventory.get("symlink_entries") or inventory.get("special_entries"):
        return False, "archive contains unsafe or conflicting paths"
    if inventory.get("encrypted_entries"):
        return False, "archive contains encrypted entries"
    if inventory.get("nested_archives"):
        return False, "nested archives require separate review"
    expanded = inventory["uncompressed_bytes"]
    free = shutil.disk_usage(destination.parent).free
    margin = max(2 * 1024**3, path.stat().st_size * 2)
    if expanded > max_expanded_bytes or expanded + margin > free:
        return False, "expanded size exceeds the configured cap or free-space margin"
    if destination.exists():
        return True, "destination already exists; not overwritten"

    destination.mkdir(parents=True, mode=0o700)
    try:
        with zipfile.ZipFile(path) as archive:
            for info, member, _ in inventory["_safe_infos"]:
                target = destination.joinpath(*member.parts)
                resolved = target.resolve(strict=False)
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError("member escapes extraction root")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("non-regular archive entry")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with archive.open(info, "r") as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                target.chmod(0o600)
        return True, None
    except Exception as exc:
        shutil.rmtree(destination, ignore_errors=True)
        return False, f"extraction failed ({type(exc).__name__})"


def sips_metadata(paths: list[Path]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    if Image is not None:
        for path in paths:
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                    info = image.info
                    image_format = image.format or path.suffix.lstrip(".").upper()
                    image.load()  # force decoding so truncated/corrupt images are counted unreadable
                    bands = image.getbands()
                    bits = info.get("bits")
                    if bits is None:
                        bits = 1 if mode == "1" else (16 if mode.startswith("I;16") else (32 if mode == "I" else 8))
                    records[str(path)] = {
                        "pixelWidth": str(width),
                        "pixelHeight": str(height),
                        "samplesPerPixel": str(len(bands)),
                        "bitsPerSample": str(bits),
                        "format": str(image_format),
                        "space": mode,
                        "hasAlpha": "yes" if "A" in bands or "transparency" in info else "no",
                    }
            except Exception:
                continue
        return records
    if not SIPS.exists():
        return records
    for start in range(0, len(paths), 128):
        chunk = paths[start : start + 128]
        try:
            proc = subprocess.run([str(SIPS), "-g", "all", *map(str, chunk)], capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            continue
        current: dict[str, str] | None = None
        current_path = ""
        for line in proc.stdout.splitlines():
            if line and not line[0].isspace():
                current_path = line.strip().removesuffix(":")
                current = records.setdefault(current_path, {})
            elif current is not None and ":" in line:
                key, value = line.strip().split(":", 1)
                current[key.strip()] = value.strip()
        if proc.returncode != 0:
            # Missing metadata is handled as unreadable/unknown in the manifest.
            pass
    return records


def identity_label(image_path: Path, extraction_root: Path, root_folders: set[str]) -> str | None:
    relative = image_path.relative_to(extraction_root)
    # This branch is a verification-pair benchmark, not an identity-folder tree.
    # Keep its images unlabeled until its positive/negative pair metadata is parsed.
    if relative.parts and relative.parts[0].lower() == "face_verification_test_set":
        return None
    parts = list(relative.parts[:-1])
    # Ignore the ZIP's wrapper directory and conventional split/image folders.
    parts = [part for part in parts if part not in root_folders and part.lower() not in NON_LABEL_DIRS]
    return parts[-1] if parts else None


def run_audit(do_extract: bool, max_expanded_bytes: int) -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    for folder in (RAW, EXTRACTED, PREPARED, REPORTS, BASE / "scripts"):
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.chmod(0o700)

    inventories: list[dict] = []
    summaries: dict[str, dict] = {}
    manifest_rows: list[dict] = []
    tree_rows: list[dict] = []
    missing: list[str] = []

    for archive_path in sorted(RAW.glob("*.zip")):
        dataset = archive_path.stem
        inv = inspect_archive(archive_path)
        dataset_top = inv["top_level"][0] if len(inv["top_level"]) == 1 else ""
        raw_label_names = sorted({
            next((part for part in reversed(member.parts[1:-1]) if part.lower() not in NON_LABEL_DIRS), None)
            for info, member, _ in inv.get("_safe_infos", [])
            if not info.is_dir() and member.parts and member.parts[0].lower() != "face_verification_test_set"
        } - {None})
        tree_identity_ids = {name: f"identity_{n:04d}" for n, name in enumerate(raw_label_names, 1)}
        directory_names: set[str] = set()
        for info, member, _ in inv.get("_safe_infos", []):
            components = member.parts if info.is_dir() else member.parts[:-1]
            for i, part in enumerate(components, 1):
                if i == 1 or part.lower() in NON_LABEL_DIRS or part in tree_identity_ids:
                    continue
                directory_names.add("/".join(components[:i]))
        directory_ids = {name: f"folder_{n:04d}" for n, name in enumerate(sorted(directory_names), 1)}
        for info, member, normalized in inv.get("_safe_infos", []):
            pseudo_parts: list[str] = []
            for index, part in enumerate(member.parts):
                if index == 0:
                    pseudo_parts.append(part)
                elif part.lower() in NON_LABEL_DIRS:
                    pseudo_parts.append(part.lower())
                elif part in tree_identity_ids:
                    pseudo_parts.append(tree_identity_ids[part])
                elif info.is_dir() or index < len(member.parts) - 1:
                    prefix = "/".join(member.parts[: index + 1])
                    pseudo_parts.append(directory_ids.get(prefix, f"folder_{len(directory_ids)+1:04d}"))
                else:
                    pseudo_parts.append("record_" + hashlib.sha256(f"{dataset}\0{normalized}".encode("utf-8")).hexdigest()[:16] + (member.suffix.lower() if member.suffix else ""))
            tree_rows.append({
                "dataset": dataset,
                "path_pseudonymous": "/".join(pseudo_parts),
                "kind": "directory" if info.is_dir() else ("image" if member.suffix.lower() in IMAGE_SUFFIXES else "file"),
                "extension": member.suffix.lower() or None,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
            })
        if do_extract:
            extract_path = EXTRACTED / dataset
            extracted, reason = safe_extract(archive_path, inv, extract_path, max_expanded_bytes)
            inv["extracted"] = extracted
            inv["extraction_reason"] = reason
        else:
            extract_path = EXTRACTED / dataset
            inv["extracted"] = extract_path.exists()
            inv["extraction_reason"] = None if extract_path.exists() else "not requested"
        inventories.append({k: v for k, v in inv.items() if not k.startswith("_")})

        summary = {
            "dataset": dataset,
            "archive": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": inv["sha256"],
            "archive_crc_bad_entry": inv["crc_bad_entry"],
            "archive_integrity_verified": inv["crc_checked"] and inv["crc_bad_entry"] is None,
            "top_level": inv["top_level"],
            "file_count": inv["file_count"],
            "directory_count": inv["directory_count"],
            "compressed_bytes": inv["compressed_bytes"],
            "uncompressed_bytes": inv["uncompressed_bytes"],
            "file_extensions": inv["file_extensions"],
            "documentation_files": inv["documentation_files"],
            "nested_archives": inv["nested_archives"],
            "unsafe_entries": inv["unsafe_entries"],
            "symlink_entries": inv["symlink_entries"],
            "special_entries": inv["special_entries"],
            "encrypted_entries": inv["encrypted_entries"],
            "extracted": inv["extracted"],
            "extraction_reason": inv["extraction_reason"],
        }

        dataset_root = EXTRACTED / dataset
        if dataset_root.exists():
            image_paths = sorted(p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.is_symlink())
            metadata = sips_metadata(image_paths)
            root_folders = set(inv["top_level"])
            identity_raw = {p: identity_label(p, dataset_root, root_folders) for p in image_paths}
            identity_names = sorted({v for v in identity_raw.values() if v is not None})
            identity_ids = {name: f"identity_{n:04d}" for n, name in enumerate(identity_names, 1)}
            counts: Counter[str] = Counter()
            checksums: dict[str, list[str]] = defaultdict(list)
            dimensions: Counter[str] = Counter()
            formats: Counter[str] = Counter()
            modes: Counter[str] = Counter()
            alpha: Counter[str] = Counter()
            image_bit_depth_values: Counter[str] = Counter()
            unreadable = 0
            for image in image_paths:
                rel = image.relative_to(BASE).as_posix()
                raw_label = identity_raw[image]
                identity_id = identity_ids.get(raw_label) if raw_label is not None else None
                counts[identity_id or "[unlabeled]"] += 1
                digest = sha256_file(image)
                checksums[digest].append(rel)
                meta = metadata.get(str(image), {})
                width = int(meta["pixelWidth"]) if meta.get("pixelWidth", "").isdigit() else None
                height = int(meta["pixelHeight"]) if meta.get("pixelHeight", "").isdigit() else None
                if width is None or height is None:
                    unreadable += 1
                dimensions[f"{width}x{height}" if width is not None and height is not None else "unknown"] += 1
                fmt = meta.get("format") or image.suffix.lstrip(".").upper() or "unknown"
                mode = meta.get("space") or "unknown"
                alpha_value = meta.get("hasAlpha", "unknown").lower()
                bit_depth = int(meta["bitsPerSample"]) if meta.get("bitsPerSample", "").isdigit() else None
                formats[fmt] += 1
                modes[mode] += 1
                alpha[alpha_value] += 1
                image_bit_depth_values[str(bit_depth) if bit_depth is not None else "unknown"] += 1
                record_id = hashlib.sha256(f"{dataset}\0{rel}".encode("utf-8")).hexdigest()[:24]
                manifest_rows.append({
                    "record_id": record_id,
                    "dataset": dataset,
                    "source_identifier": dataset,
                    "source_archive_id": SOURCE_FILE_IDS.get(dataset),
                    "relative_local_path": rel,
                    "identity_id": identity_id,
                    "label_status": "folder_label" if identity_id else "missing_or_ambiguous",
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "channels": int(meta["samplesPerPixel"]) if meta.get("samplesPerPixel", "").isdigit() else None,
                    "file_format": fmt,
                    "bit_depth": bit_depth,
                    "has_alpha": alpha_value if alpha_value in {"yes", "no"} else None,
                    "byte_size": image.stat().st_size,
                    "sha256": digest,
                    "capture_condition_metadata": None,
                    "session_metadata": None,
                })
            per_identity = sorted(value for label, value in counts.items() if label != "[unlabeled]")
            duplicate_groups = [paths for paths in checksums.values() if len(paths) > 1]
            summary.update({
                "image_count": len(image_paths),
                "identity_count": len(identity_names),
                "unlabeled_image_count": counts.get("[unlabeled]", 0),
                "images_per_identity_min": min(per_identity) if per_identity else None,
                "images_per_identity_median": median(per_identity) if per_identity else None,
                "images_per_identity_max": max(per_identity) if per_identity else None,
                "image_dimensions": dict(dimensions),
                "image_formats": dict(formats),
                "image_modes": dict(modes),
                "alpha_values": dict(alpha),
                "bit_depth_values": dict(sorted(image_bit_depth_values.items())),
                "unreadable_image_count": unreadable,
                "exact_duplicate_group_count": len(duplicate_groups),
                "exact_duplicate_image_count": sum(len(group) for group in duplicate_groups),
                "identity_counts_pseudonymous": dict(sorted(counts.items())),
                "perceptual_duplicate_check": "not computed during archive inventory; run scripts/perceptual_duplicates.py for a local heuristic screen",
            })
        else:
            summary.update({"image_count": None, "identity_count": None, "unreadable_image_count": None})
        summaries[dataset] = summary

    for expected in ("olivetti_96", "survface_96"):
        if not (RAW / f"{expected}.zip").exists():
            missing.append(expected)
    MANIFEST.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
    TREE.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in tree_rows), encoding="utf-8")
    SUMMARY.write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), "datasets": summaries, "missing_archives": missing}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    INVENTORY.write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), "archives": inventories}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MANIFEST.chmod(0o600)
    TREE.chmod(0o600)
    SUMMARY.chmod(0o600)
    INVENTORY.chmod(0o600)
    print(json.dumps({"archives_seen": len(inventories), "missing_archives": missing, "manifest_rows": len(manifest_rows), "summary": str(SUMMARY.relative_to(BASE)), "manifest": str(MANIFEST.relative_to(BASE)), "archive_tree_rows": len(tree_rows)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="extract archives only after path, link, CRC, nested-archive, and disk checks")
    parser.add_argument("--max-expanded-gib", type=float, default=20.0, help="hard cap on declared expanded bytes per archive (default: 20 GiB)")
    args = parser.parse_args()
    if args.max_expanded_gib <= 0:
        parser.error("--max-expanded-gib must be positive")
    return run_audit(args.extract, int(args.max_expanded_gib * 1024**3))


if __name__ == "__main__":
    raise SystemExit(main())
