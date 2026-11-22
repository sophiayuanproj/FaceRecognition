"""Stable, named seed derivation; seeds never stand in for sessions or people."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable


def derive_seed(protocol_id: str, root_seed: int, stream_name: str, *, bits: int = 32) -> int:
    if not protocol_id or not stream_name:
        raise ValueError("protocol_id and stream_name are required")
    if not 1 <= bits <= 64:
        raise ValueError("bits must be in [1, 64]")
    payload = json.dumps([protocol_id, int(root_seed), stream_name], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def validate_roots(roots: Iterable[int], minimum: int = 10) -> list[int]:
    values = [int(root) for root in roots]
    if len(values) < minimum:
        raise ValueError("at least %d distinct training roots are required" % minimum)
    if len(set(values)) != len(values):
        raise ValueError("root seeds must be distinct")
    if any(root < 0 or root > 0xFFFFFFFF for root in values):
        raise ValueError("root seeds must be unsigned 32-bit integers")
    return values


def matched_streams(protocol_id: str, roots: Iterable[int], names: Iterable[str]) -> list[dict]:
    checked = validate_roots(roots)
    stream_names = list(names)
    if len(set(stream_names)) != len(stream_names) or any(not n for n in stream_names):
        raise ValueError("child stream names must be unique and nonempty")
    return [{"root_seed": root, "streams": {name: derive_seed(protocol_id, root, name) for name in stream_names}}
            for root in checked]
