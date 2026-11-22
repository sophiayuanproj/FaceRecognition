"""Versioned shared-contract validation used by host and instrument workers."""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS = ("source_sha256", "model_sha256", "config_sha256", "data_manifest_sha256", "protocol_sha256")


def validate_model_output(output: dict, expected_type: str, expected_size: int,
                          class_labels: list[str] | None = None) -> None:
    if expected_type not in {"classifier_logits", "embedding", "detector_centroids"}:
        raise ValueError("unknown expected model type")
    if output.get("model_type") != expected_type:
        raise TypeError("model output type mismatch: expected %s, got %s" %
                        (expected_type, output.get("model_type")))
    values = output.get("values")
    if not isinstance(values, list) or len(values) != expected_size:
        raise ValueError("model output value count does not match its typed contract")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise ValueError("model output values must be finite numbers")
    if expected_type == "classifier_logits":
        if not class_labels or len(class_labels) != expected_size:
            raise ValueError("classifier logits require a matching ordered class-label list")
        if output.get("class_labels") != class_labels:
            raise ValueError("classifier class order differs from the frozen label contract")
    elif output.get("class_labels") is not None:
        raise ValueError("embeddings/detector centroids cannot carry classifier class labels")


def validate_hardware_event(event: dict) -> None:
    missing = [key for key in HASH_FIELDS if not SHA256_RE.fullmatch(str(event.get(key, "")))]
    if missing:
        raise ValueError("event is missing valid provenance hashes: " + ", ".join(missing))
    if not event.get("reset_id"):
        raise ValueError("event requires reset_id")
    if not isinstance(event.get("sequence"), int) or event["sequence"] < 0:
        raise ValueError("event requires a nonnegative sequence")
    for key in ("host_utc", "device_utc"):
        if event.get(key) is not None:
            try:
                parsed = datetime.fromisoformat(str(event[key]).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(key + " must be ISO-8601") from exc
            if parsed.tzinfo is None:
                raise ValueError(key + " must include a timezone offset")
    if event.get("device_tick_ms") is not None and (not isinstance(event["device_tick_ms"], int) or event["device_tick_ms"] < 0):
        raise ValueError("device_tick_ms must be a nonnegative integer or null")


def power_claim_eligible(evidence: dict) -> tuple[bool, list[str]]:
    """Return whether energy attribution is supportable under the versioned contract."""
    required = ("instrument_qualified", "power_boundary_documented", "target_board_verified",
                "clock_alignment_qualified", "valid_sample_timing", "raw_packets_preserved")
    reasons = ["missing_or_false:" + key for key in required if evidence.get(key) is not True]
    if evidence.get("target_board") != "Nicla Vision":
        reasons.append("target_board_is_not_Nicla_Vision")
    if evidence.get("energy_claim_valid") is True:
        # Eligibility is determined by evidence fields, not by a caller's success flag.
        pass
    return not reasons, reasons
