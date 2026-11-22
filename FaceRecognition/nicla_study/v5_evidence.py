"""Strict portable evidence binding and row-derived candidate qualification.

Qualification never trusts a summary metrics JSON. It derives decisions from
per-example probabilities, calibration rows, detected flags, identity groups,
and optional raw response observations. A manually authored status label is
not evidence.
"""

import hashlib
import json
import math
import numbers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "firmware"))
from open_set_decision import accepts, threshold_above_max


KNOWN_LABELS = {"K%02d" % i: i - 1 for i in range(1, 6)}
ALL_LABELS = set(KNOWN_LABELS) | {"OTHER"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite evidence")
    if isinstance(value, dict):
        for x in value.values():
            finite_tree(x)
    elif isinstance(value, list):
        for x in value:
            finite_tree(x)


def strict_load(path):
    def bad(x):
        raise ValueError("Nonfinite JSON " + x)

    obj = json.loads(Path(path).read_text(), parse_constant=bad)
    finite_tree(obj)
    return obj


def strict_jsonl(path):
    def bad(x):
        raise ValueError("Nonfinite JSON " + x)

    rows = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=bad)
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError("Invalid JSONL row %d: %s" % (line_number, error))
            finite_tree(row)
            if not isinstance(row, dict):
                raise ValueError("Prediction row %d is not an object" % line_number)
            rows.append(row)
    if not rows:
        raise ValueError("Empty evidence JSONL")
    return rows


def bound_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Workspace-relative evidence paths required")
    dest = (Path(root) / relative).resolve()
    try:
        dest.relative_to(Path(root).resolve())
    except ValueError:
        raise ValueError("Evidence path escape")
    if not dest.is_file():
        raise ValueError("Missing evidence file: " + relative)
    return dest


def verify_bindings(root, bindings):
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("Empty bindings")
    seen = set()
    for row in bindings:
        path = row["path"]
        if path in seen:
            raise ValueError("Duplicate binding")
        seen.add(path)
        dest = bound_path(root, path)
        if sha(dest) != row["sha256"] or dest.stat().st_size != row["bytes"]:
            raise ValueError("Stale/tampered binding: " + path)
    return len(seen)


def verify_raw_blocks(root, relative, minimum=10):
    path = bound_path(root, relative)
    rows = strict_load(path)
    if not isinstance(rows, list) or len(rows) < minimum:
        raise ValueError("Insufficient actual blocks")
    ids = set()
    for row in rows:
        block = row.get("block")
        if isinstance(block, bool) or not isinstance(block, int) or block in ids:
            raise ValueError("Invalid/duplicate block")
        ids.add(block)
        if row.get("status") != "executed" or row.get("stderr"):
            raise ValueError("Failed raw block")
        raw = path.parent / ("block_%02d.stdout.bin" % block)
        if not raw.is_file() or sha(raw) != row.get("stdout_sha256"):
            raise ValueError("Missing/stale raw block transcript")
    return rows


def _upper_binomial_95(events, n):
    """One-sided exact 95% Clopper-Pearson upper bound."""
    if n <= 0 or events < 0 or events > n:
        raise ValueError("Invalid binomial counts")
    if events == n:
        return 1.0
    if events == 0:
        return 1.0 - 0.05 ** (1.0 / n)

    def cdf(p):
        return sum(math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j)) for j in range(events + 1))

    low, high = 0.0, 1.0
    for _ in range(90):
        mid = (low + high) / 2.0
        if cdf(mid) > 0.05:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def derive_metrics_from_predictions(root, relative):
    """Recompute calibration, acceptance, group risk and latency from rows.

    The rejection threshold is independently reconstructed from the maximum
    known-class score of detected unknown-calibration rows, matching the
    documented zero-calibration-false-accept policy. No RESULT/metrics file is
    read. If response_ms is absent, response_p95_ms is None and qualification
    must remain false.
    """
    path = bound_path(root, relative)
    rows = strict_jsonl(path)
    seen = set()
    for i, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("Missing/duplicate record_id at prediction row %d" % i)
        seen.add(record_id)
        role = row.get("role")
        if role not in {"threshold_calibration", "unknown_calibration", "development", "unknown_development", "negative_control"}:
            raise ValueError("Unsupported evaluation role: %r" % role)
        label = row.get("true_label")
        if label not in ALL_LABELS | {"NO_FACE", "CALIBRATION"}:
            raise ValueError("Unsupported label: %r" % label)
        if role in {"threshold_calibration", "development"} and label not in KNOWN_LABELS:
            raise ValueError("Known role has non-known label")
        if role in {"unknown_calibration", "unknown_development"} and label not in {"OTHER", "NO_FACE"}:
            raise ValueError("Unknown role has known label")
        detected = row.get("detected")
        if not isinstance(detected, bool):
            raise ValueError("Missing boolean detected flag")
        p = row.get("probabilities")
        if not isinstance(p, list) or len(p) != 6:
            raise ValueError("Expected six class probabilities")
        if any(isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or x < 0 or x > 1 for x in p):
            raise ValueError("Invalid class probability")
        if abs(sum(p) - 1.0) > 1e-5:
            raise ValueError("Class probabilities do not sum to one")
        row["_class_index"] = _argmax(p)
        row["_known_score"] = max(p[:5])
        row["_detected"] = detected

    unknown_cal_scores = [r["_known_score"] for r in rows if r["role"] == "unknown_calibration" and r["_detected"]]
    if not unknown_cal_scores:
        raise ValueError("No detected unknown calibration rows")
    threshold = threshold_above_max(unknown_cal_scores)
    for row in rows:
        row["_accepted"] = bool(row["_detected"] and row["_class_index"] < 5 and accepts(row["_known_score"], threshold))
        if "accepted" in row and row["accepted"] is not row["_accepted"]:
            raise ValueError("Stored accepted flag disagrees with row-derived decision")
        if "predicted_label" in row:
            index = row["_class_index"]
            expected = "NO_FACE" if not row["_detected"] else ("OTHER" if index == 5 else "K%02d" % (index + 1))
            if row["predicted_label"] != expected:
                raise ValueError("Stored predicted label disagrees with probabilities/detection")

    known = [r for r in rows if r["role"] == "development"]
    unknown = [r for r in rows if r["role"] == "unknown_development"]
    known_cal = [r for r in rows if r["role"] == "threshold_calibration"]
    unknown_cal = [r for r in rows if r["role"] == "unknown_calibration"]
    if not known or not unknown:
        raise ValueError("Missing known/unknown development rows")

    def group_risk(subset):
        grouped = {}
        for row in subset:
            identity = row.get("source_identity") or row.get("source_group")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Development row lacks independent identity/group key")
            grouped[identity] = grouped.get(identity, False) or row["_accepted"]
        false_groups = sum(bool(value) for value in grouped.values())
        return {
            "groups": len(grouped),
            "false_groups": int(false_groups),
            "upper95": _upper_binomial_95(false_groups, len(grouped)),
        }

    correct_known = sum(
        row["_accepted"] and row["_class_index"] == KNOWN_LABELS[row["true_label"]]
        for row in known
    )
    accepted_known = sum(row["_accepted"] for row in known)
    wrong_known = accepted_known - correct_known
    false_unknown = sum(row["_accepted"] for row in unknown)
    per_person = []
    for label, index in KNOWN_LABELS.items():
        subset = [r for r in known if r["true_label"] == label]
        correct = sum(r["_accepted"] and r["_class_index"] == index for r in subset)
        accepted = sum(r["_accepted"] for r in subset)
        top1 = sum(r["_class_index"] == index for r in subset)
        per_person.append({
            "label": label,
            "n": len(subset),
            "raw_top1": top1 / len(subset) if subset else None,
            "correct_accepted": int(correct),
            "coverage": accepted / len(subset) if subset else None,
        })

    response = [r.get("response_ms") for r in rows if r.get("response_ms") is not None]
    if response:
        if any(isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or x < 0 for x in response):
            raise ValueError("Invalid response_ms observation")
        response.sort()
        response_p95 = float(response[max(0, math.ceil(0.95 * len(response)) - 1)])
    else:
        response_p95 = None

    unknown_risk = group_risk(unknown)
    unknown_cal_risk = group_risk(unknown_cal)
    return {
        "derived_from": relative,
        "threshold": threshold,
        "known_n": len(known),
        "known_raw_top1": sum(r["_class_index"] == KNOWN_LABELS[r["true_label"]] for r in known) / len(known),
        "known_correct_acceptance": correct_known / len(known),
        "known_coverage": accepted_known / len(known),
        "known_wrong_accepted": int(wrong_known),
        "unknown_n": len(unknown),
        "unknown_false_accepts": int(false_unknown),
        "unknown_groups": unknown_risk["groups"],
        "unknown_false_groups": unknown_risk["false_groups"],
        "unknown_group_upper95": unknown_risk["upper95"],
        "known_calibration_n": len(known_cal),
        "known_calibration_correct_acceptance": (
            sum(r["_accepted"] and r["_class_index"] == KNOWN_LABELS[r["true_label"]] for r in known_cal) / len(known_cal)
            if known_cal else None
        ),
        "unknown_calibration_n": len(unknown_cal),
        "unknown_calibration_false_groups": unknown_cal_risk["false_groups"],
        "per_person": per_person,
        "response_observations_n": len(response),
        "response_p95_ms": response_p95,
        "future_final_used": False,
    }


def qualify_candidate(root, record):
    """Derive candidate qualification from bound evidence, never summary labels."""
    reasons = []
    derived = None
    try:
        verify_bindings(root, record["bindings"])
        required = {"model", "source", "configuration", "data_roles", "roster", "runtime"}
        if not required.issubset({r.get("kind") for r in record["bindings"]}):
            raise ValueError("Missing binding kinds")
        for name in ["control_blocks", "parity_blocks", "optical_blocks", "camera_memory_blocks"]:
            rows = verify_raw_blocks(root, record[name], 10)
            if name == "optical_blocks":
                for row in rows:
                    for field in ["source_id", "session_id", "condition", "ground_truth_source"]:
                        if not row.get(field):
                            raise ValueError("Optical block lacks independent ground truth")
                    if row.get("evidence_tier") != "actual_optical_camera":
                        raise ValueError("Replay is not optical")
        derived = derive_metrics_from_predictions(root, record["predictions"])
        if derived["known_correct_acceptance"] < 0.90:
            raise ValueError("Row-derived known correct acceptance target missed")
        if derived["known_coverage"] < 0.90:
            raise ValueError("Row-derived known coverage target missed")
        if derived["unknown_group_upper95"] > 0.01:
            raise ValueError("Row-derived unknown group uncertainty target missed")
        if derived["response_p95_ms"] is None:
            raise ValueError("No row-level response_ms evidence")
        if derived["response_p95_ms"] > 1000:
            raise ValueError("Row-derived response p95 target missed")
        if not isinstance(derived.get("known_n"), int) or derived["known_n"] <= 0:
            raise ValueError("No known trials")
    except (ValueError, KeyError, TypeError, OSError) as error:
        reasons.append(str(error))
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "derived_metrics": derived,
        "manual_metrics_json_ignored": True,
        "manual_complete_label_ignored": True,
    }


def binding(root, path, kind):
    p = Path(path)
    p = p if p.is_absolute() else Path(root) / p
    return {
        "path": p.resolve().relative_to(Path(root).resolve()).as_posix(),
        "sha256": sha(p),
        "bytes": p.stat().st_size,
        "kind": kind,
    }
