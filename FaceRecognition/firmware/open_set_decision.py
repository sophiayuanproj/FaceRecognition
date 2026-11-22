"""Portable float32 score/threshold contract for host and MicroPython.

Decision scores and thresholds are rounded to IEEE-754 binary32 before the
comparison. A calibration maximum is advanced by one representable binary32
step, so exact ties are consistently rejected after serialization.
"""

import math
import struct


def as_f32(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return struct.unpack("<f", struct.pack("<f", value))[0]


def next_f32_up(value):
    value = as_f32(value)
    if value == 0.0:
        return struct.unpack("<f", struct.pack("<I", 1))[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = bits - 1 if value < 0.0 else bits + 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def threshold_above_max(scores):
    values = [as_f32(value) for value in scores]
    if not values:
        raise ValueError("cannot fit threshold without calibration scores")
    return next_f32_up(max(values))


def accepts(score, threshold):
    return as_f32(score) >= as_f32(threshold)


def quantized_score(score, step, zero=0):
    """Quantize a finite score to an integer score grid."""
    score, step = as_f32(score), float(step)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("quantization step must be finite and positive")
    return int(round(score / step + int(zero)))


def threshold_above_quantized_max(scores):
    values = [int(value) for value in scores]
    if not values:
        raise ValueError("cannot fit quantized threshold without calibration scores")
    return max(values) + 1


def accepts_quantized(score, threshold):
    if isinstance(score, bool) or isinstance(threshold, bool):
        raise ValueError("quantized scores must be integers")
    if int(score) != score or int(threshold) != threshold:
        raise ValueError("quantized scores must be integers")
    return int(score) >= int(threshold)


def validate_probabilities(probabilities, expected_classes=6):
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != expected_classes:
        raise ValueError("probability vector has the wrong class count")
    values = []
    for value in probabilities:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("probabilities must be numeric")
        score = as_f32(value)
        if score < 0.0 or score > 1.0:
            raise ValueError("probability is outside [0,1]")
        values.append(score)
    if abs(sum(values) - 1.0) > 1e-5:
        raise ValueError("probabilities must sum to one")
    return values


def decide(detected, class_index=None, score=None, threshold=None,
           quality_ok=True, multiple=False):
    """Return a conservative output code without inventing a person ID."""
    if not detected:
        return "NO_FACE"
    if multiple or not quality_ok:
        return "UNCERTAIN"
    if isinstance(class_index, bool) or not isinstance(class_index, int) or not 0 <= class_index <= 5:
        return "UNCERTAIN"
    if class_index == 5:
        return "OTHER"
    try:
        passed = accepts(score, threshold)
    except (TypeError, ValueError, OverflowError):
        return "UNCERTAIN"
    if not passed:
        return "UNCERTAIN"
    return "K%02d" % (class_index + 1)


def validate_group_rows(rows, expected_count=None, expected_groups=None):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple")
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError("row count mismatch")
    record_ids, groups = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        record_id, group = row.get("record_id"), row.get("group_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("missing or duplicate record_id")
        if not isinstance(group, str) or not group:
            raise ValueError("missing group_id")
        record_ids.add(record_id)
        groups.add(group)
    if expected_groups is not None and len(groups) != int(expected_groups):
        raise ValueError("independent group count mismatch")
    return {"rows": len(record_ids), "groups": len(groups)}
