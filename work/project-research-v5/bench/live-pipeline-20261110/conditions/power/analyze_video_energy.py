"""Join calibrated video trial windows to the native FNIRSI decoded sample CSV."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import NamedTuple


DEFAULT_MAX_GAP_MS = 200.0
DEFAULT_CLOCK_JUMP_MS = 100.0
SAMPLE_INDICES = set(range(4))
CSV_REQUIRED = {"run_id", "report_arrival_index", "sample_index_in_report",
                "host_receive_monotonic_ns", "host_receive_utc", "voltage_v", "current_a", "crc_status"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class Report(NamedTuple):
    unix_ms: float
    monotonic_ns: int
    watts: float
    index: int
    min_voltage_v: float | None = None
    max_voltage_v: float | None = None
    max_current_a: float | None = None


def finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} is boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is missing or nonnumeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def utc_ms(text):
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("FNIRSI timestamp lacks timezone")
    elapsed = stamp - EPOCH
    return (elapsed.days * 86400 + elapsed.seconds) * 1000 + elapsed.microseconds / 1000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_meter_csv(path, requested_run=None):
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = CSV_REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"FNIRSI CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            run_id = row["run_id"]
            if not run_id:
                raise ValueError(f"{path}:{reader.line_num}: missing run_id")
            if requested_run is not None and run_id != requested_run:
                continue
            try:
                index = int(row["report_arrival_index"])
                sample_index = int(row["sample_index_in_report"])
                stamp = utc_ms(row["host_receive_utc"])
                monotonic = int(row["host_receive_monotonic_ns"])
                if min(index, monotonic) < 0 or sample_index not in SAMPLE_INDICES:
                    raise ValueError("invalid report or sample index")
                if row["crc_status"] != "pass":
                    raise ValueError("sample does not have a passing CRC")
                voltage = finite(row["voltage_v"], "voltage_v")
                current = finite(row["current_a"], "current_a")
                if min(voltage, current) < 0:
                    raise ValueError("negative voltage or current in unsigned FNIRSI protocol")
                power = voltage * current
                if not math.isfinite(power):
                    raise ValueError("nonfinite V×I product")
                if row.get("power_w_calculated") not in (None, ""):
                    reported = finite(row["power_w_calculated"], "power_w_calculated")
                    if not math.isclose(power, reported, rel_tol=1e-5, abs_tol=1e-7):
                        raise ValueError("V×I disagrees with power_w_calculated")
                report = groups.setdefault((run_id, index), (stamp, monotonic, {}))
                if report[:2] != (stamp, monotonic) or sample_index in report[2]:
                    raise ValueError("report timestamp disagreement or duplicate sample index")
                report[2][sample_index] = (power, voltage, current)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
    runs = {key[0] for key in groups}
    if len(runs) != 1:
        raise ValueError(f"Select exactly one nonempty meter run; found {len(runs)}")
    reports = []
    for (_run_id, index), (stamp, monotonic, samples) in groups.items():
        if samples.keys() != SAMPLE_INDICES:
            raise ValueError(f"FNIRSI report {index} lacks all four distinct decoded samples")
        powers, voltages, currents = zip(*samples.values())
        reports.append(Report(stamp, monotonic, mean(powers), index, min(voltages), max(voltages), max(currents)))
    for left, right in zip(reports, reports[1:]):
        if right.index <= left.index or right.unix_ms <= left.unix_ms or right.monotonic_ns < left.monotonic_ns:
            raise ValueError("FNIRSI report IDs or receive timestamps run backward")
    return next(iter(runs)), reports


def integrate(reports, start, end, max_gap_ms, max_clock_jump_ms,
              min_voltage_v=None, max_voltage_v=None, max_current_a=None):
    """Clip linear interpolation to a fully bracketed meter-clock window."""
    if end <= start:
        raise ValueError("integration window has zero or negative length")
    stamps = [report.unix_ms for report in reports]
    begin = bisect.bisect_right(stamps, start) - 1
    finish = bisect.bisect_left(stamps, end)
    if begin < 0 or finish >= len(stamps):
        raise ValueError("meter coverage does not bracket the full integration window")
    points = reports[begin:finish + 1]
    for report in points:
        if (min_voltage_v is not None and (report.min_voltage_v is None or report.min_voltage_v < min_voltage_v)
                or max_voltage_v is not None and (report.max_voltage_v is None or report.max_voltage_v > max_voltage_v)
                or max_current_a is not None and (report.max_current_a is None or report.max_current_a > max_current_a)):
            raise ValueError(f"FNIRSI report {report.index} has a slot outside the declared supply/load range")
    elapsed_deltas = [(p.unix_ms - points[0].unix_ms)
                      - (p.monotonic_ns - points[0].monotonic_ns) / 1_000_000 for p in points]
    disagreement = max(elapsed_deltas) - min(elapsed_deltas)
    if disagreement > max_clock_jump_ms:
        raise ValueError(f"meter wall versus monotonic clock disagreement {disagreement:.3f} ms")
    energy, gaps = [], []
    for left, right in zip(points, points[1:]):
        gap = right.unix_ms - left.unix_ms
        if gap > max_gap_ms:
            raise ValueError(f"meter gap {gap:.3f} ms exceeds {max_gap_ms:g} ms between "
                             f"reports {left.index} and {right.index}")
        a, b = max(left.unix_ms, start), min(right.unix_ms, end)
        slope = (right.watts - left.watts) / gap
        p_a = left.watts + slope * (a - left.unix_ms)
        p_b = left.watts + slope * (b - left.unix_ms)
        energy.append((p_a + p_b) * (b - a) / 2000)
        gaps.append(gap)
    duration = (end - start) / 1000
    joules = math.fsum(energy)
    return {"energy_j": joules, "time_weighted_mean_w": joules / duration,
            "duration_s": duration, "report_points": len(points),
            "largest_report_gap_ms": max(gaps),
            "meter_utc_minus_monotonic_elapsed_range_ms": disagreement,
            "first_report_index": points[0].index, "last_report_index": points[-1].index,
            "first_bracketing_report_unix_ms": points[0].unix_ms,
            "last_bracketing_report_unix_ms": points[-1].unix_ms}


def resolve_clock(trial, args):
    clock = trial.get("clock_alignment")
    if clock is None and args.same_host_clock:
        clock = {"mode": "same_host_clock", "method": "operator asserted one active OS host clock"}
    if clock is None and args.clock_offset_ms is not None:
        clock = {"mode": "measured_offset", "meter_minus_render_ms": args.clock_offset_ms,
                 "uncertainty_ms": args.clock_offset_uncertainty_ms, "method": args.clock_offset_method}
    if not isinstance(clock, dict):
        raise ValueError("explicit same-host clock or measured clock offset is required")
    mode = clock.get("mode")
    if mode == "same_host_clock":
        return {"mode": mode, "meter_minus_render_ms": 0.0, "uncertainty_ms": 0.0,
                "method": clock.get("method") or "operator asserted one active OS host clock"}
    if mode != "measured_offset":
        raise ValueError("unsupported clock alignment mode")
    offset = finite(clock.get("meter_minus_render_ms"), "meter_minus_render_ms")
    uncertainty = finite(clock.get("uncertainty_ms"), "clock uncertainty_ms")
    method = clock.get("method")
    if uncertainty < 0 or not isinstance(method, str) or not method.strip():
        raise ValueError("measured offset needs a nonnegative uncertainty and a measurement method")
    return {"mode": mode, "meter_minus_render_ms": offset, "uncertainty_ms": uncertainty,
            "method": method}


def aligned_energy(reports, start, end, clock, args):
    offset, uncertainty = clock["meter_minus_render_ms"], clock["uncertainty_ms"]
    meter_start, meter_end = start + offset, end + offset
    limits = (args.max_gap_ms, args.max_clock_jump_ms, getattr(args, "min_voltage_v", None),
              getattr(args, "max_voltage_v", None), getattr(args, "max_current_a", None))
    result = integrate(reports, meter_start, meter_end, *limits)
    if uncertainty:
        # Verify gap and timestamp quality over every time interval that could
        # contribute under the supplied clock-offset uncertainty.
        integrate(reports, meter_start - uncertainty, meter_end + uncertainty, *limits)
        result["clock_offset_endpoint_energy_j"] = [
            integrate(reports, meter_start + shift, meter_end + shift, *limits)["energy_j"]
            for shift in (-uncertainty, uncertainty)
        ]
    result.update(meter_clock_start_unix_ms=meter_start, meter_clock_end_unix_ms=meter_end)
    return result


def episode_energies(trial, reports, start, end, clock, args):
    episodes = trial.get("episodes") or []
    if not isinstance(episodes, list):
        raise ValueError("episodes must be a list")
    results = []
    for episode in episodes:
        name = episode.get("name", episode.get("id"))
        row = {"name": name, "status": "invalid", "energy_j": None}
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("missing episode name")
            begin = finite(episode.get("start_ms"), "episode start_ms")
            finish = finite(episode.get("end_ms", episode.get("end_ms_exclusive")), "episode end_ms")
            if begin < 0 or finish <= begin or start + finish > end:
                raise ValueError("episode falls outside the full video window")
            row.update(start_ms=begin, end_ms=finish)
            row.update(aligned_energy(reports, start + begin, start + finish, clock, args))
            row["status"] = "valid"
        except ValueError as exc:
            row["reason"] = str(exc)
        results.append(row)
    return results


def analyze_trial(path, args, meter_cache):
    result = {"trial_file": str(path), "status": "invalid", "energy_j": None}
    try:
        trial = json.loads(path.read_text(encoding="utf-8-sig"))
        csv_value = trial.get("meter_csv_path")
        meter = path.parent / csv_value if csv_value else args.meter_csv
        if meter is not None:
            meter = meter.resolve()
            result["meter_csv"] = str(meter)
        for key in ("arm", "block", "physical_block_id", "seed", "arm_order", "arm_index",
                    "boot_id", "firmware_config_sha256_pairs", "topology_id", "meter_run_id",
                    "video_sha256", "measurement_ready"):
            result[key] = trial.get(key)
        result["trial_sha256"] = sha256(path)
        physical_block = trial.get("physical_block_id")
        if physical_block is not None and (not isinstance(physical_block, str)
                or not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", physical_block)):
            raise ValueError("invalid physical_block_id")
        if trial.get("measurement_ready") is not True:
            raise ValueError("trial did not qualify its video measurement window")
        synthetic_trial = trial.get("evidence_scope") == "synthetic" or trial.get("synthetic") is True
        if synthetic_trial != args.synthetic:
            raise ValueError("synthetic trial and --synthetic mode do not agree")
        start = finite(trial.get("meter_window_start_unix_ms"), "meter_window_start_unix_ms")
        end = finite(trial.get("meter_window_end_unix_ms"), "meter_window_end_unix_ms")
        if min(start, end) <= 0:
            raise ValueError("video windows require positive Unix timestamps in milliseconds")
        clock = resolve_clock(trial, args)
        result["clock_alignment"] = clock
        if meter is None:
            raise ValueError("provide --meter-csv or trial meter_csv_path")
        requested = trial.get("meter_run_id", args.meter_run_id)
        cache_key = meter, requested
        if cache_key not in meter_cache:
            run_id, reports = read_meter_csv(meter, requested)
            meter_cache[cache_key] = run_id, reports, sha256(meter)
        run_id, reports, meter_hash = meter_cache[cache_key]
        result.update(meter_csv=str(meter), meter_csv_sha256=meter_hash, meter_run_id=run_id,
                      render_clock_start_unix_ms=start, render_clock_end_unix_ms=end)
        result.update(aligned_energy(reports, start, end, clock, args))
        if args.per_episode:
            result["episodes"] = episode_energies(trial, reports, start, end, clock, args)
        result["status"] = "valid"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result.update(reason=str(exc), energy_j=None)
    return result


def paired_differences(results, baseline_arm):
    if not baseline_arm:
        return []
    blocks = defaultdict(list)
    for result in results:
        if (isinstance(result.get("block"), (str, int)) and isinstance(result.get("arm"), str)
                and isinstance(result.get("video_sha256"), str) and result["video_sha256"]):
            physical_block = result.get("physical_block_id")
            if physical_block is not None and not isinstance(physical_block, str):
                continue
            group_id = ("physical", physical_block) if physical_block else ("legacy", result["block"])
            blocks[(group_id, result["video_sha256"])].append(result)
    pairs = []
    for ((kind, block), video), group in blocks.items():
        arms = defaultdict(list)
        for result in group:
            arms[result.get("arm")].append(result)
        baseline = arms.pop(baseline_arm, [])
        for name, treatment in arms.items():
            row = {"block": group[0]["block"], "video_sha256": video, "baseline_arm": baseline_arm, "arm": name}
            if kind == "physical":
                row["physical_block_id"] = block
            if len(baseline) != 1 or len(treatment) != 1:
                pairs.append(dict(row, status="unpaired", reason="need one unique trial for each arm"))
                continue
            base, test = baseline[0], treatment[0]
            if base["status"] != "valid" or test["status"] != "valid":
                pairs.append(dict(row, status="unpaired", reason="at least one trial has invalid energy"))
                continue
            binding_fields = ("meter_run_id", "meter_csv_sha256", "boot_id",
                              "firmware_config_sha256_pairs", "topology_id", "seed", "arm_order")
            mismatches = [field for field in binding_fields if base.get(field) != test.get(field)]
            if mismatches:
                pairs.append(dict(row, status="unpaired", reason="mismatched paired binding: " + ", ".join(mismatches)))
                continue
            overlap = min(base["meter_clock_end_unix_ms"], test["meter_clock_end_unix_ms"]) - max(
                base["meter_clock_start_unix_ms"], test["meter_clock_start_unix_ms"])
            if overlap > 0:
                pairs.append(dict(row, status="unpaired", reason="paired meter windows overlap"))
                continue
            duration_delta = test["duration_s"] - base["duration_s"]
            pairs.append(dict(row, status="paired", baseline_trial=base["trial_file"], trial=test["trial_file"],
                              energy_delta_j=test["energy_j"] - base["energy_j"],
                              mean_power_delta_w=test["time_weighted_mean_w"] - base["time_weighted_mean_w"],
                              duration_delta_s=duration_delta,
                              energy_delta_is_duration_matched=abs(duration_delta) < 0.001))
    return pairs


def pair_summary(pairs):
    arms = defaultdict(list)
    for pair in pairs:
        if pair["status"] == "paired":
            arms[pair["arm"]].append(pair)
    summary = {}
    for arm, rows in arms.items():
        matched = [row["energy_delta_j"] for row in rows if row["energy_delta_is_duration_matched"]]
        summary[arm] = {"paired_blocks": len(rows),
                        "mean_paired_power_delta_w": mean(row["mean_power_delta_w"] for row in rows),
                        "duration_matched_pairs": len(matched),
                        "mean_duration_matched_energy_delta_j": mean(matched) if matched else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--meter-csv", type=Path)
    parser.add_argument("--meter-run-id")
    clock = parser.add_mutually_exclusive_group()
    clock.add_argument("--same-host-clock", action="store_true")
    clock.add_argument("--clock-offset-ms", type=float, help="measured meter-clock minus render-clock")
    parser.add_argument("--clock-offset-uncertainty-ms", type=float)
    parser.add_argument("--clock-offset-method")
    parser.add_argument("--max-gap-ms", type=float, default=DEFAULT_MAX_GAP_MS)
    parser.add_argument("--max-clock-jump-ms", type=float, default=DEFAULT_CLOCK_JUMP_MS)
    parser.add_argument("--min-voltage-v", type=float, help="minimum measured supply voltage in each slot")
    parser.add_argument("--max-voltage-v", type=float, help="maximum measured supply voltage in each slot")
    parser.add_argument("--max-current-a", type=float, help="maximum current in each slot for the verified load")
    parser.add_argument("--baseline-arm")
    parser.add_argument("--per-episode", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.max_gap_ms, args.max_clock_jump_ms)):
        parser.error("gap and clock-jump bounds must be positive and finite")
    gates = {key: getattr(args, key) for key in ("min_voltage_v", "max_voltage_v", "max_current_a")}
    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in gates.values()):
        parser.error("supply and load limits must be positive and finite")
    if args.min_voltage_v is not None and args.max_voltage_v is not None and args.min_voltage_v >= args.max_voltage_v:
        parser.error("minimum voltage must be lower than maximum voltage")
    results, cache = [], {}
    for trial in args.trials:
        results.append(analyze_trial(trial, args, cache))
    inputs = {path.resolve() for path in args.trials}
    inputs.update(Path(result["meter_csv"]) for result in results if result.get("meter_csv"))
    if args.meter_csv is not None:
        inputs.add(args.meter_csv.resolve())
    if args.output.resolve() in inputs:
        parser.error("output path equals an input")
    pairs = paired_differences(results, args.baseline_arm)
    output = {"schema": "nicla-video-energy-join-v1",
              "evidence_scope": "synthetic_numeric_check" if args.synthetic else "recorded_meter_csv_integration",
              "method": "mean(V×I) per four-sample HID report; time-weighted trapezoids at host UTC report receipt",
              "max_gap_ms": args.max_gap_ms, "max_clock_jump_ms": args.max_clock_jump_ms, "supply_load_limits": gates,
              "trials": results, "pairs": pairs, "paired_summary_by_arm": pair_summary(pairs)}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for result in results:
        print(result["status"], result.get("arm"), result.get("block"),
              result.get("energy_j") if result["status"] == "valid" else result["reason"])
    return 0 if all(result["status"] == "valid" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
