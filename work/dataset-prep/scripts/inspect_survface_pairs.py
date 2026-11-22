#!/usr/bin/env python3
"""Read the bundled SurvFace MATLAB positive/negative pair lists locally.

Supports the simple compressed MATLAB v5 cell-of-strings files in this archive
using only the Python standard library. Output contains pseudonymous manifest
record IDs, pair labels, and aggregate match counts; it never copies filenames.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path, PurePosixPath

PREP = Path(__file__).resolve().parents[1]
EXTRACTED = PREP / "extracted" / "survface_96"
MANIFEST = PREP / "reports" / "manifest.jsonl"
OUTPUT = PREP / "reports" / "survface-pairs.jsonl"
SUMMARY_PATH = PREP / "reports" / "survface-pair-summary.json"
DATASET_SUMMARY = PREP / "reports" / "dataset-summary.json"
PAIR_FILES = {
    "positive": "Face_Verification_Test_Set/positive_pairs_names.mat",
    "negative": "Face_Verification_Test_Set/negative_pairs_names.mat",
}


def element(buf: bytes, offset: int, endian: str) -> tuple[int, bytes, int]:
    if offset + 4 > len(buf):
        raise ValueError("truncated MATLAB element tag")
    word = struct.unpack_from(endian + "I", buf, offset)[0]
    upper = word >> 16
    if upper:
        if upper > 4:
            raise ValueError("invalid small MATLAB element")
        kind = word & 0xFFFF
        size = upper
        start = offset + 4
        return kind, buf[start : start + size], offset + 8
    if offset + 8 > len(buf):
        raise ValueError("truncated MATLAB regular element tag")
    kind, size = struct.unpack_from(endian + "II", buf, offset)
    start = offset + 8
    stop = start + size
    if stop > len(buf):
        raise ValueError("truncated MATLAB element data")
    return kind, buf[start:stop], stop + ((-size) % 8)


def parse_matrix(buf: bytes, endian: str) -> tuple[str, tuple[int, ...], list[str]]:
    offset = 0
    kind, flags, offset = element(buf, offset, endian)
    if kind != 6 or len(flags) < 4:
        raise ValueError("unsupported MATLAB array flags")
    matlab_class = struct.unpack_from(endian + "I", flags)[0] & 0xFF
    kind, raw_dims, offset = element(buf, offset, endian)
    if kind != 5 or len(raw_dims) % 4:
        raise ValueError("unsupported MATLAB dimensions")
    dims = struct.unpack(endian + "i" * (len(raw_dims) // 4), raw_dims)
    kind, raw_name, offset = element(buf, offset, endian)
    if kind not in {1, 2, 16}:
        raise ValueError("unsupported MATLAB variable name encoding")
    name = raw_name.decode("utf-8", errors="replace")

    if matlab_class == 4:  # mxCHAR_CLASS
        kind, raw_text, _ = element(buf, offset, endian)
        if kind in {1, 2, 16}:
            text = raw_text.decode("utf-8", errors="replace")
        elif kind in {3, 4, 17}:
            text = raw_text.decode("utf-16-le" if endian == "<" else "utf-16-be", errors="replace")
        elif kind == 18:
            text = raw_text.decode("utf-32-le" if endian == "<" else "utf-32-be", errors="replace")
        else:
            raise ValueError("unsupported MATLAB character encoding")
        return name, dims, [text.rstrip("\x00 ")]

    if matlab_class != 1:  # mxCELL_CLASS
        raise ValueError("pair list is not a MATLAB cell array")
    strings: list[str] = []
    for _ in range(math.prod(dims)):
        kind, child, offset = element(buf, offset, endian)
        if kind != 14:
            raise ValueError("pair list contains a non-matrix cell")
        _, _, values = parse_matrix(child, endian)
        if len(values) != 1:
            raise ValueError("pair list cell is not a scalar string")
        strings.append(values[0])
    return name, dims, strings


def read_mat_cell_strings(path: Path) -> tuple[str, tuple[int, ...], list[str]]:
    buf = path.read_bytes()
    if len(buf) < 128 or buf[126:128] not in {b"IM", b"MI"}:
        raise ValueError("unsupported MATLAB file header")
    endian = "<" if buf[126:128] == b"IM" else ">"
    kind, payload, _ = element(buf, 128, endian)
    if kind != 15:  # miCOMPRESSED
        raise ValueError("expected compressed MATLAB v5 element")
    decoded = zlib.decompress(payload)
    kind, matrix, _ = element(decoded, 0, endian)
    if kind != 14:
        raise ValueError("expected MATLAB matrix")
    return parse_matrix(matrix, endian)


def normalize_reference(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized)).lstrip("/").casefold()


def main() -> int:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    by_path: dict[str, str] = {}
    by_basename: dict[str, list[str]] = {}
    image_rows = [
        r for r in rows
        if r.get("dataset") == "survface_96"
        and r.get("file_format") == "JPEG"
        and "face_verification_test_set/" in r.get("relative_local_path", "").casefold()
    ]
    for row in image_rows:
        local = PurePosixPath(row["relative_local_path"])
        dataset_relative = str(PurePosixPath(*local.parts[local.parts.index("survface_96") + 1 :])).casefold()
        by_path[dataset_relative] = row["record_id"]
        by_basename.setdefault(local.name.casefold(), []).append(row["record_id"])

    def resolve(value: str) -> str | None:
        candidate = normalize_reference(value)
        if candidate in by_path:
            return by_path[candidate]
        marker = "face_verification_test_set/"
        if marker in candidate:
            candidate = candidate[candidate.index(marker) :]
            if candidate in by_path:
                return by_path[candidate]
        base = PurePosixPath(candidate).name.casefold()
        options = by_basename.get(base, [])
        return options[0] if len(options) == 1 else None

    out_rows: list[dict] = []
    details: dict[str, dict] = {}
    for pair_type, relative in PAIR_FILES.items():
        var_name, dims, values = read_mat_cell_strings(EXTRACTED / relative)
        if len(dims) != 2 or dims[1] != 2 or len(values) != math.prod(dims):
            raise ValueError(f"unsupported pair-list shape in {Path(relative).name}")
        pair_count = dims[0]
        matched_pairs = unmatched_pairs = 0
        referenced: set[str] = set()
        for index in range(pair_count):
            left = resolve(values[index])
            right = resolve(values[index + pair_count])
            if left is not None:
                referenced.add(left)
            if right is not None:
                referenced.add(right)
            if left is not None and right is not None:
                matched_pairs += 1
            else:
                unmatched_pairs += 1
            pair_id = hashlib.sha256(f"survface_96\0{pair_type}\0{index}".encode()).hexdigest()[:24]
            out_rows.append({
                "pair_id": pair_id,
                "dataset": "survface_96",
                "pair_type": pair_type,
                "left_record_id": left,
                "right_record_id": right,
                "match_status": "matched" if left is not None and right is not None else "unmatched_reference",
            })
        details[pair_type] = {
            "matlab_variable": var_name,
            "matrix_shape": list(dims),
            "pair_count": pair_count,
            "fully_matched_pair_count": matched_pairs,
            "unmatched_pair_count": unmatched_pairs,
            "unique_image_records_referenced": len(referenced),
        }

    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in out_rows), encoding="utf-8")
    OUTPUT.chmod(0o600)
    report = {
        "dataset": "survface_96",
        "source_files": list(PAIR_FILES.values()),
        "interpretation": "Pair-level verification labels; not image-level identity labels. Filenames are omitted; rows reference pseudonymous manifest record IDs.",
        "pairs": details,
    }
    SUMMARY_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY_PATH.chmod(0o600)

    dataset_summary = json.loads(DATASET_SUMMARY.read_text(encoding="utf-8"))
    target = dataset_summary["datasets"]["survface_96"]
    target["pair_metadata_status"] = "parsed locally from bundled MATLAB v5 files"
    target["positive_pair_count"] = details["positive"]["pair_count"]
    target["negative_pair_count"] = details["negative"]["pair_count"]
    target["fully_matched_pair_count"] = sum(x["fully_matched_pair_count"] for x in details.values())
    target["pair_manifest"] = "reports/survface-pairs.jsonl"
    DATASET_SUMMARY.write_text(json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    DATASET_SUMMARY.chmod(0o600)
    print(json.dumps({k: {"pair_count": v["pair_count"], "fully_matched_pair_count": v["fully_matched_pair_count"], "unmatched_pair_count": v["unmatched_pair_count"], "unique_image_records_referenced": v["unique_image_records_referenced"]} for k, v in details.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
