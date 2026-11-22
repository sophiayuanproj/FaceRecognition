"""Hash-bound development-only selector for the J/I measured-cost pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
from pathlib import Path
import random
import re

HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
BOOTSTRAP_REPLICATES = 10000


def _log_binomial_probability(k: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    logs = [_log_binomial_probability(i, n, p) for i in range(k + 1)]
    peak = max(logs)
    return math.exp(peak) * sum(math.exp(value - peak) for value in logs)


def clopper_pearson_upper_one_sided(errors: int, encounters: int, alpha: float = 0.05) -> float:
    """Exact one-sided binomial upper bound, used as a conservative floor."""
    if encounters <= 0 or errors < 0 or errors > encounters:
        raise ValueError("invalid false-identification counts")
    if errors == encounters:
        return 1.0
    if errors == 0:
        return 1.0 - alpha ** (1.0 / encounters)
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if _binomial_cdf(errors, encounters, mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def cluster_bootstrap_upper(groups: list[dict], seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> float:
    if not groups:
        raise ValueError("unknown identity clusters are required")
    rng = random.Random(seed)
    count = len(groups)
    rates = []
    for _ in range(replicates):
        errors = encounters = 0
        for _index in range(count):
            row = groups[rng.randrange(count)]
            errors += int(row["false_identifications"])
            encounters += int(row["encounters"])
        rates.append(errors / encounters if encounters else 0.0)
    rates.sort()
    return rates[min(len(rates) - 1, math.ceil(0.95 * len(rates)) - 1)]


def unknown_upper_bound(groups: list[dict], model_id: str) -> dict:
    encounters = sum(int(row.get("encounters", 0)) for row in groups)
    errors = sum(int(row.get("false_identifications", 0)) for row in groups)
    seed = int.from_bytes(hashlib.sha256(model_id.encode("utf-8")).digest()[:4], "big")
    cluster_upper = cluster_bootstrap_upper(groups, seed)
    binomial_upper = clopper_pearson_upper_one_sided(errors, encounters)
    return {"unknown_identity_count": len(groups), "unknown_encounters": encounters,
            "unknown_false_identifications": errors,
            "identity_cluster_bootstrap_upper_95": cluster_upper,
            "exact_encounter_binomial_upper_95": binomial_upper,
            "one_sided_upper_95": max(cluster_upper, binomial_upper),
            "method": "maximum of identity-cluster bootstrap and exact encounter-level Clopper-Pearson upper bound"}


def _finite_real(value) -> float | None:
    """JSON metrics are numbers, not truth values or numeric-looking strings."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _validate_candidate(row: dict, requirements: dict) -> tuple[bool, list[str], dict | None]:
    reasons = []
    if not isinstance(row, dict):
        return False, ["candidate_not_object"], None
    if row.get("data_role") != "development":
        reasons.append("not_development_role")
    if row.get("used_final_data") is not False:
        reasons.append("final_data_use_not_explicitly_false")
    for field in ("model_sha256", "runtime_config_sha256", "controller_sha256", "roster_sha256", "calibration_sha256"):
        if not isinstance(row.get(field), str) or not HASH_RE.fullmatch(row[field]):
            reasons.append("missing_or_invalid_" + field)
    if row.get("measurement_status") != "complete_verified":
        reasons.append("measurement_not_complete_verified")
    blocks = row.get("independent_paired_blocks")
    if type(blocks) is not int or blocks < 0:
        reasons.append("invalid_independent_paired_blocks")
    elif blocks < requirements["minimum_paired_blocks"]:
        reasons.append("insufficient_independent_paired_blocks")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    values = {}
    for key in ("known_macro_top1", "known_macro_accepted_coverage", "response_p95_ms",
                "energy_per_accepted_correct_known_vs_fixed_high"):
        value = _finite_real(metrics.get(key))
        if value is None:
            reasons.append("missing_or_nonfinite_or_invalid_" + key)
        elif value < 0 or (key.startswith("known_") and value > 1):
            reasons.append("out_of_range_" + key)
            value = None
        values[key] = value
    top1 = values["known_macro_top1"]
    coverage = values["known_macro_accepted_coverage"]
    latency = values["response_p95_ms"]
    energy = values["energy_per_accepted_correct_known_vs_fixed_high"]
    if top1 is not None and top1 < requirements["known_macro_top1_min"]:
        reasons.append("known_accuracy_below_target")
    if coverage is not None and coverage < requirements["known_macro_coverage_min"]:
        reasons.append("known_coverage_below_target")
    if latency is not None and latency > requirements["response_p95_ms_max"]:
        reasons.append("response_p95_above_target")
    if energy is not None and energy > requirements["energy_ratio_max"]:
        reasons.append("energy_improvement_below_target")
    groups = row.get("unknown_by_identity")
    upper = None
    if not isinstance(groups, list) or not groups:
        reasons.append("unknown_identity_cluster_counts_missing")
    else:
        valid = all(isinstance(group, dict) and
                    isinstance(group.get("identity_id"), str) and group["identity_id"].strip() and
                    type(group.get("encounters")) is int and group["encounters"] > 0 and
                    type(group.get("false_identifications")) is int and
                    0 <= group["false_identifications"] <= group["encounters"] for group in groups)
        if not valid:
            reasons.append("invalid_unknown_identity_cluster_counts")
        elif len({group["identity_id"].strip() for group in groups}) != len(groups):
            reasons.append("duplicate_unknown_identity_clusters")
        else:
            if len(groups) < requirements["unknown_identity_min"]:
                reasons.append("too_few_unknown_identities")
            if sum(group["encounters"] for group in groups) < requirements["unknown_encounter_min"]:
                reasons.append("too_few_unknown_encounters")
            if not reasons:
                upper = unknown_upper_bound(groups, str(row.get("model_id", "candidate")))
                if upper["one_sided_upper_95"] > requirements["unknown_upper_95_max"]:
                    reasons.append("unknown_false_identification_upper_bound_above_target")
    score = {"known_macro_top1": top1, "known_macro_accepted_coverage": coverage,
             "response_p95_ms": latency, "energy_ratio_vs_fixed_high": energy, "unknown": upper}
    return not reasons, reasons, score


def select_development_candidates(rows: list[dict], requirements: dict | None = None) -> dict:
    requirements = requirements or {
        "minimum_paired_blocks": 10,
        "known_macro_top1_min": 0.90,
        "known_macro_coverage_min": 0.90,
        "unknown_identity_min": 30,
        "unknown_encounter_min": 300,
        "unknown_upper_95_max": 0.01,
        "response_p95_ms_max": 1000.0,
        "energy_ratio_max": 0.90,
    }
    attempts = []
    accepted = []
    for index, row in enumerate(rows):
        model_id = str(row.get("model_id", "candidate-%03d" % (index + 1))) if isinstance(row, dict) else "invalid-%03d" % index
        passed, reasons, score = _validate_candidate(row, requirements)
        attempt = {"model_id": model_id, "accepted": passed, "reasons": reasons, "metrics": score,
                   "attempted_utc": row.get("attempted_utc") if isinstance(row, dict) else None,
                   "attempted_setting": row.get("controller_setting") if isinstance(row, dict) else None}
        attempts.append(attempt)
        if passed:
            accepted.append((float(score["energy_ratio_vs_fixed_high"]),
                             -float(score["known_macro_top1"]), model_id, row))
    winner = min(accepted, key=lambda item: item[:3])[3] if accepted else None
    return {
        "schema": "J-development-only-selection-1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "selected_development_candidate" if winner else "no_qualified_candidate",
        "data_role": "development_only",
        "used_final_data": False,
        "requirements": requirements,
        "selection_rule": "among candidates passing every quality, identity-cluster unknown, response, measured-cost and hash gate, choose lowest paired energy per accepted correct known encounter; tie-break by higher known macro top-1 then model_id",
        "candidate_count": len(rows),
        "qualified_candidate_count": len(accepted),
        "selected_candidate": winner,
        "attempt_log": attempts,
        "freeze_allowed": winner is not None,
    }


def fit_controller_calibration(rows: list[dict], measured_cost: dict,
                               energy_value_mj_per_correct: float) -> dict:
    """Fit fixed-bin escalation risks from encounter-level development rows.

    The function accepts one row per independently defined encounter, never
    adjacent frames. The caller must provide I's measured incremental model
    energy/latency and a predeclared willingness-to-spend value.
    """
    if _finite_real(energy_value_mj_per_correct) is None or energy_value_mj_per_correct <= 0:
        raise ValueError("energy value per corrected decision must be positive and finite")
    if measured_cost.get("status") != "complete_verified" or measured_cost.get("source_role") != "development_only":
        raise ValueError("measured cost must be complete and development-only")
    try:
        incremental_energy = _finite_real(measured_cost["incremental_energy_mj"])
        incremental_latency = _finite_real(measured_cost["incremental_latency_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("measured incremental energy and latency are required") from error
    if (incremental_energy is None or incremental_energy < 0 or
            incremental_latency is None or incremental_latency < 0):
        raise ValueError("measured cost fields must be finite and nonnegative")
    seen = set()
    by_bin: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("data_role") != "development" or row.get("used_final_data") is not False:
            raise PermissionError("controller calibration accepts development rows only")
        required = ("encounter_id", "session_id", "identity_id", "small_correct", "large_correct")
        if any(row.get(key) is None for key in required):
            raise ValueError("encounter-level row is missing identity/session outcomes")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in required[:3]):
            raise ValueError("encounter/session/identity IDs must be nonempty strings")
        if any(type(row[key]) is not bool for key in ("small_correct", "large_correct")):
            raise ValueError("correctness outcomes must be explicit JSON booleans")
        encounter_id = str(row["encounter_id"])
        if encounter_id in seen:
            raise ValueError("duplicate encounter_id; frames cannot be treated as independent encounters")
        seen.add(encounter_id)
        score = _finite_real(row.get("small_candidate_score"))
        if score is None or score < 0 or score > 1:
            raise ValueError("small candidate score must be finite in [0,1]")
        bin_index = min(9, int(score * 10))
        by_bin.setdefault(bin_index, []).append(row)
    risk_bins = []
    for index in range(10):
        values = by_bin.get(index, [])
        clusters = {(str(row["session_id"]), str(row["identity_id"])) for row in values}
        if len(values) < 30 or len(clusters) < 10:
            continue
        small_error = sum(not bool(row["small_correct"]) for row in values) / len(values)
        large_error = sum(not bool(row["large_correct"]) for row in values) / len(values)
        lower = index / 10.0
        upper = 1.0 if index == 9 else (index + 1) / 10.0
        risk_bins.append({"score_min": lower, "score_max": upper,
            "small_error_probability": small_error,
            "large_error_probability": large_error,
            "expected_accuracy_gain": small_error - large_error,
            "incremental_energy_mj": incremental_energy,
            "incremental_latency_ms": incremental_latency,
            "encounter_count": len(values), "identity_session_cluster_count": len(clusters),
            "fit_source_role": "development_only"})
    return {
        "schema": "J-controller-calibration-1",
        "status": "ready_for_hash_bound_runtime_update" if risk_bins else "insufficient_development_encounters",
        "source_role": "development_only",
        "used_final_data": False,
        "risk_threshold": 0.1,
        "energy_value_mj_per_correct": float(energy_value_mj_per_correct),
        "risk_bins": risk_bins,
        "measured_cost_source": measured_cost,
        "encounter_count": len(rows),
        "unique_session_identity_clusters": len({(str(row.get("session_id")), str(row.get("identity_id"))) for row in rows}),
        "fit_rule": "fixed 0.1-wide candidate-score bins; require at least 30 encounter rows and 10 session-identity clusters per bin; do not count frame repeats as encounters",
        "limitations": "The calibration must be re-hashed into runtime_config and revalidated on the board before any risk/cost-aware policy can qualify.",
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON token: " + value)))
        if not isinstance(row, dict):
            raise ValueError("input line %d is not an object" % line_number)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="append-only development-only pilot candidate JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="new/replaceable selection report path; never a frozen receipt")
    args = parser.parse_args(argv)
    result = select_development_candidates(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"status": result["status"], "qualified_candidate_count": result["qualified_candidate_count"],
                      "freeze_allowed": result["freeze_allowed"], "output": str(args.output)}, indent=2))
    return 0 if result["status"] == "selected_development_candidate" else 3


if __name__ == "__main__":
    raise SystemExit(main())
