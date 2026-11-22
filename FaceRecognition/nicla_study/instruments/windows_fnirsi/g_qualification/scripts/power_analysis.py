"""Quality gating and trapezoid integration for report-level FNIRSI summaries."""

import json
import math
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from fnirsi_protocol import decode_report, summarize_report

PROVISIONAL_MAX_REPORT_GAP_NS = 200_000_000
PROVISIONAL_VOLTAGE_RANGE_V = (4.75, 5.25)
REQUIRED_PATH_EVIDENCE = [
    "physical_meter_model_confirmed",
    "nominal_5v_source_confirmed",
    "board_is_sole_downstream_load",
    "no_parallel_board_power_path",
    "meter_display_matches_host_values",
    "idle_and_active_load_voltage_nonzero",
    "active_idle_step_response_repeated_10_blocks",
    "effective_report_timing_validated_10_blocks",
]
REQUIRED_COMPARISON_EVIDENCE = [
    "model_policy_config_hashes_locked",
    "telemetry_schema_agreed_with_model_system",
    "consented_known_and_unknown_sessions_ready",
    "encounter_ground_truth_ready",
    "quality_and_latency_limits_predeclared",
]


def read_jsonl(path: Path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_num, exc))
    return records


def decode_raw_records(records: Sequence[Dict[str, object]]):
    """Re-decode preserved packets; return one summarized point per report."""
    reports = []
    all_samples = []
    invalid = 0
    timeouts = 0
    for record in records:
        if record.get("record_type") == "timeout":
            timeouts += 1
            continue
        if record.get("record_type") not in ("measurement_report", "invalid_report"):
            continue
        raw_hex = record.get("raw_report_hex")
        if not isinstance(raw_hex, str):
            invalid += 1
            continue
        try:
            samples = decode_report(bytes.fromhex(raw_hex))
        except (ValueError, TypeError):
            invalid += 1
            continue
        summary = summarize_report(samples)
        host_ns = record.get("host_receive_monotonic_ns")
        reports.append({
            "arrival_index": int(record.get("report_arrival_index", len(reports))),
            "host_receive_monotonic_ns": None if host_ns is None else int(host_ns),
            "host_receive_utc": record.get("host_receive_utc"),
            "samples": samples,
            **summary,
        })
        all_samples.extend(samples)
    return reports, all_samples, invalid, timeouts


def integrate_points(points: Sequence[Dict[str, object]], max_gap_ns: int = PROVISIONAL_MAX_REPORT_GAP_NS):
    """Integrate report-mean power using host delivery times, without filling gaps."""
    usable = [p for p in points if p.get("host_receive_monotonic_ns") is not None]
    intervals = []
    invalid_intervals = []
    all_gaps = []
    candidate_j = 0.0
    valid_covered_ns = 0
    nonfinite_points = 0
    for point in usable:
        try:
            if not math.isfinite(float(point["power_mean_w"])):
                nonfinite_points += 1
        except (KeyError, TypeError, ValueError):
            nonfinite_points += 1
    for left, right in zip(usable, usable[1:]):
        t0 = int(left["host_receive_monotonic_ns"])
        t1 = int(right["host_receive_monotonic_ns"])
        dt = t1 - t0
        all_gaps.append(dt)
        if dt <= 0:
            invalid_intervals.append({"from_ns": t0, "to_ns": t1, "reason": "nonpositive_time_delta"})
            continue
        if dt > max_gap_ns:
            invalid_intervals.append({"from_ns": t0, "to_ns": t1, "reason": "gap_exceeds_max", "gap_ns": dt})
            continue
        try:
            p0 = float(left["power_mean_w"])
            p1 = float(right["power_mean_w"])
            if not math.isfinite(p0) or not math.isfinite(p1):
                raise ValueError("nonfinite_power")
        except (KeyError, TypeError, ValueError):
            invalid_intervals.append({"from_ns": t0, "to_ns": t1, "reason": "nonfinite_or_missing_power"})
            continue
        delta_j = ((p0 + p1) / 2.0) * (dt / 1_000_000_000.0)
        candidate_j += delta_j
        valid_covered_ns += dt
        intervals.append({"from_ns": t0, "to_ns": t1, "dt_ns": dt, "energy_j": delta_j})
    if len(usable) >= 2:
        full_span_ns = int(usable[-1]["host_receive_monotonic_ns"]) - int(usable[0]["host_receive_monotonic_ns"])
    else:
        full_span_ns = 0
    return {
        "point_count": len(usable),
        "start_monotonic_ns": None if not usable else int(usable[0]["host_receive_monotonic_ns"]),
        "end_monotonic_ns": None if not usable else int(usable[-1]["host_receive_monotonic_ns"]),
        "duration_s": full_span_ns / 1e9,
        "largest_report_gap_ns": max(all_gaps, default=None),
        "valid_covered_duration_s": valid_covered_ns / 1e9,
        "coverage_fraction": None if full_span_ns <= 0 else valid_covered_ns / full_span_ns,
        "candidate_integral_j": candidate_j if len(usable) >= 2 else None,
        "integral_j": candidate_j if len(usable) >= 2 and not invalid_intervals and nonfinite_points == 0 else None,
        "mean_power_w_by_report_delivery": None if valid_covered_ns <= 0 else candidate_j / (valid_covered_ns / 1e9),
        "invalid_intervals": invalid_intervals,
        "nonfinite_or_missing_power_points": nonfinite_points,
        "method": "trapezoidal integration of per-report mean(V*I) at host HID report-receive monotonic times; acquisition time not validated",
    }


def _describe(values):
    if not values:
        return None
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def assess_run(run_dir: Path, acceptance_manifest: Optional[Path] = None,
               max_gap_ns: int = PROVISIONAL_MAX_REPORT_GAP_NS):
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("run_metadata.json missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = read_jsonl(run_dir / metadata.get("raw_packet_file", "raw_packets.jsonl"))
    reports, samples, invalid_count, timeout_count = decode_raw_records(records)
    points = [{
        "host_receive_monotonic_ns": report["host_receive_monotonic_ns"],
        "power_mean_w": report["power_mean_w"],
    } for report in reports]
    integration = integrate_points(points, max_gap_ns)
    volts = [float(row["voltage_v"]) for row in samples]
    amps = [float(row["current_a"]) for row in samples]
    watts = [float(row["power_w_calculated"]) for row in samples]
    valid_packet_count = len(reports)
    total_packet_count = valid_packet_count + invalid_count
    invalid_fraction = 1.0 if total_packet_count == 0 else invalid_count / total_packet_count
    lo, hi = PROVISIONAL_VOLTAGE_RANGE_V
    voltage_in_range = bool(volts) and all(lo <= v <= hi for v in volts)
    active_voltage_nonzero = bool(volts) and all(v > 0.0 for v in volts)
    metadata_complete = metadata.get("status") == "complete"
    exact_duplicates = sum(1 for row in records if row.get("possible_exact_duplicate_payload") is True)

    manifest = {}
    if acceptance_manifest is not None:
        manifest = json.loads(Path(acceptance_manifest).read_text(encoding="utf-8"))
    missing_evidence = [key for key in REQUIRED_PATH_EVIDENCE if manifest.get(key) is not True]
    comparison_missing = [key for key in REQUIRED_COMPARISON_EVIDENCE if manifest.get(key) is not True]
    artifact_hash_mismatches = []
    expected_hashes = manifest.get("artifact_hashes", {})
    actual_hashes = metadata.get("artifact_hashes", {})
    for name, expected in expected_hashes.items():
        if actual_hashes.get(name) != expected:
            artifact_hash_mismatches.append(name)
    expected_run_id = manifest.get("accepted_run_id")
    if expected_run_id is not None and expected_run_id != metadata.get("run_id"):
        artifact_hash_mismatches.append("run_id")
    failed_gates = []
    if not metadata_complete:
        failed_gates.append("logger_run_not_complete")
    if not voltage_in_range:
        failed_gates.append("voltage_outside_provisional_4.75_to_5.25V_gate")
    if not active_voltage_nonzero:
        failed_gates.append("zero_or_missing_voltage; invalid_as_board_power")
    if invalid_fraction > 0.01:
        failed_gates.append("invalid_packet_fraction_over_1_percent")
    if timeout_count:
        # Timeout count is retained; an operator must explicitly justify any timeout for formal use.
        failed_gates.append("hid_read_timeouts_present")
    if integration["invalid_intervals"]:
        failed_gates.append("one_or_more_report_gaps_exceeding_gate_or_bad_timestamps")
    if integration["nonfinite_or_missing_power_points"]:
        failed_gates.append("nonfinite_or_missing_power_value")
    if integration["point_count"] < 2:
        failed_gates.append("fewer_than_two_valid_report_points")
    if missing_evidence:
        failed_gates.append("required_physical_evidence_missing")
    if artifact_hash_mismatches:
        failed_gates.append("run_manifest_identity_or_hash_mismatch")

    energy_valid = not failed_gates
    return {
        "schema_version": "power-analysis-1.0",
        "run_id": metadata.get("run_id"),
        "experiment_id": metadata.get("experiment_id"),
        "policy_id": metadata.get("policy_id"),
        "scenario_id": metadata.get("scenario_id"),
        "root_seed": metadata.get("root_seed"),
        "status": "valid_power_input" if energy_valid else "diagnostic_only",
        "energy_claim_valid": energy_valid,
        "policy_comparison_ready": energy_valid and not comparison_missing,
        "failed_gates": failed_gates,
        "missing_physical_evidence": missing_evidence,
        "missing_policy_comparison_evidence": comparison_missing,
        "run_manifest_mismatches": sorted(set(artifact_hash_mismatches)),
        "provisional_voltage_gate_v": [lo, hi],
        "voltage_gate_basis": "prospective screening criterion for nominal 5 V USB input; confirm against actual instrument/source validation before formal freeze",
        "metrics": {
            "valid_measurement_reports": valid_packet_count,
            "invalid_reports": invalid_count,
            "possible_exact_duplicate_payloads_not_staleness_evidence": exact_duplicates,
            "read_timeouts": timeout_count,
            "invalid_report_fraction": invalid_fraction,
            "decoded_samples": len(samples),
            "voltage_v": _describe(volts),
            "current_a_bandwidth_limited": _describe(amps),
            "power_w_bandwidth_limited": _describe(watts),
            "report_arrival_rate_hz": None if integration["duration_s"] <= 0 else (valid_packet_count - 1) / integration["duration_s"],
            "integration": integration,
            "energy_j": integration["integral_j"] if energy_valid else None,
            "diagnostic_candidate_integral_j_not_for_claims": integration["candidate_integral_j"],
            "mean_power_w": (integration["mean_power_w_by_report_delivery"] if energy_valid else None),
            "energy_per_decision_j": None,
            "energy_per_success_j": None,
        },
        "timing_limitations": [
            "HID report-receive time is not a device acquisition timestamp.",
            "Four samples per packet do not establish independent sample cadence.",
            "No intra-report sample-time estimate is used for the integral.",
            "Host-receive trapezoid is only a diagnostic candidate until 10-block step response and timing validation pass.",
        ],
    }
