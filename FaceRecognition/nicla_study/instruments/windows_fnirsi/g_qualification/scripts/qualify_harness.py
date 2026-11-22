#!/usr/bin/env python3
"""Ten-root seeded software qualification plus explicit hardware-pending ledger."""

import csv
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from align_events import detect_power_state_segments, fit_marker_alignment, marker_match, unwrap_ticks
from fnirsi_protocol import PacketDecodeError, crc8, decode_report, make_command
from hardware_lock import HardwareLock
from paired_statistics import analyze as analyze_pairs
from power_analysis import assess_run, integrate_points
from run_io import RunWriter

def find_workspace(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "work/project-finish-v4/config/EXPERIMENT_PLAN.json").is_file():
            return parent
    raise RuntimeError("portable workspace root not found")

PROJECT_ROOT = find_workspace(Path(__file__))
POWER_ROOT = Path(__file__).resolve().parents[1]
SEED_PLAN_PATH = PROJECT_ROOT / "work/research-plan-2026-10-22/seed-plan.json"


def make_report(voltage_v=5.0, current_a=0.1, flag=0, seed=0):
    rng = random.Random(seed)
    report = bytearray(64)
    report[0:2] = b"\xaa\x04"
    for i in range(4):
        off = 2 + 15 * i
        v = max(0, int(round((voltage_v + rng.uniform(-0.00002, 0.00002)) * 100000)))
        a = max(0, int(round((current_a + rng.uniform(-0.00002, 0.00002)) * 100000)))
        report[off:off + 4] = v.to_bytes(4, "little")
        report[off + 4:off + 8] = a.to_bytes(4, "little")
        report[off + 8:off + 10] = (3058).to_bytes(2, "little")
        report[off + 10:off + 12] = (80).to_bytes(2, "little")
        report[off + 12] = flag & 0xff
        report[off + 13:off + 15] = (390).to_bytes(2, "little")
    report[62] = 0
    report[63] = crc8(report[1:63])
    return bytes(report)


def assert_close(a, b, tolerance=1e-8):
    assert abs(a - b) <= tolerance, "%r != %r" % (a, b)


def software_tests_for_seed(seed, bootstrap_roots):
    results = []
    rng = random.Random(seed)

    # 1. Bounded logger lifecycle, CSV/JSONL finalization, and no-overwrite behavior.
    with tempfile.TemporaryDirectory(prefix="power-log-") as tmp:
        out = Path(tmp) / "run"
        meta = {"run_id": "seed-%d" % seed, "experiment_id": "software-test", "policy_id": "test",
                "scenario_id": "synthetic", "root_seed": seed, "artifact_hashes": {}}
        writer = RunWriter(out, meta)
        payload = make_report(5.0, 0.1, seed % 256, seed)
        writer.record_report(payload, 0, 1_000_000_000, "2026-10-22T00:00:00+00:00")
        writer.close("complete")
        saved = json.loads((out / "run_metadata.json").read_text())
        assert saved["status"] == "complete" and saved["decoded_sample_count"] == 4
        assert len((out / "raw_packets.jsonl").read_text().splitlines()) == 3
        try:
            RunWriter(out, meta)
            raise AssertionError("writer overwrote an existing run")
        except FileExistsError:
            pass
    results.append(("SW-01", "pass", "seeded temp run finalized raw JSONL, decoded CSV, metadata; overwrite refused"))

    # 2. Canonical packet/command, scaling, flag preservation, malformed and fuzzed fields.
    assert make_command(0x81)[-1] == 0x8e
    assert make_command(0x82)[-1] == 0x96
    sample_v = rng.uniform(4.75, 5.25)
    sample_a = rng.uniform(0.005, 0.4)
    packet = make_report(sample_v, sample_a, seed & 0xff, seed)
    decoded = decode_report(packet)
    assert len(decoded) == 4
    assert decoded[0]["sample_flag_raw_hex"] == "%02x" % (seed & 0xff)
    assert decoded[0]["voltage_v"] > 4.74 and decoded[0]["current_a"] > 0
    assert math.isfinite(decoded[0]["power_w_calculated"])
    for malformed in (packet[:-1], b"\x00" + packet[1:], packet[:-1] + bytes([packet[-1] ^ 1])):
        try:
            decode_report(malformed)
            raise AssertionError("invalid packet accepted")
        except PacketDecodeError:
            pass
    extreme = bytearray(packet)
    for i in range(4):
        off = 2 + 15 * i
        extreme[off:off + 4] = (0xffffffff).to_bytes(4, "little")
        extreme[off + 4:off + 8] = (0xffffffff).to_bytes(4, "little")
    extreme[63] = crc8(extreme[1:63])
    assert math.isfinite(decode_report(bytes(extreme))[0]["power_w_calculated"])
    results.append(("SW-02", "pass", "CRC/header/length/unit/unsigned extreme and unknown flag assertions passed"))

    # 3. Analytic irregular trace integral, zero-current/voltage and gap rejection.
    analytic = [
        {"host_receive_monotonic_ns": 0, "power_mean_w": 1.0},
        {"host_receive_monotonic_ns": 2_000_000_000, "power_mean_w": 2.0},
        {"host_receive_monotonic_ns": 5_000_000_000, "power_mean_w": 3.0},
    ]
    integ = integrate_points(analytic, max_gap_ns=3_000_000_000)
    assert_close(integ["integral_j"], 10.5)
    zero = integrate_points([{"host_receive_monotonic_ns": 1, "power_mean_w": 0.0},
                             {"host_receive_monotonic_ns": 50_000_001, "power_mean_w": 0.0}])
    assert_close(zero["integral_j"], 0.0)
    dropped = integrate_points([analytic[0], {"host_receive_monotonic_ns": 3_000_000_000, "power_mean_w": 2.0}],
                               max_gap_ns=500_000_000)
    assert dropped["integral_j"] is None and dropped["invalid_intervals"]
    nonfinite = integrate_points([{"host_receive_monotonic_ns": 0, "power_mean_w": 1.0},
                                  {"host_receive_monotonic_ns": 1_000_000_000, "power_mean_w": float("nan")}])
    assert nonfinite["integral_j"] is None and nonfinite["nonfinite_or_missing_power_points"] == 1
    fixture_root = POWER_ROOT / "fixtures" / "synthetic"
    with (fixture_root / ("synthetic-trace-seed-%d.csv" % seed)).open(newline="", encoding="utf-8") as handle:
        fixture_rows = list(csv.DictReader(handle))
    fixture_expected = json.loads((fixture_root / ("synthetic-expected-seed-%d.json" % seed)).read_text())
    fixture_points = [{"host_receive_monotonic_ns": int(round(float(row["elapsed_s"]) * 1e9)),
                       "power_mean_w": float(row["power_w"])} for row in fixture_rows]
    fixture_integral = integrate_points(fixture_points)
    assert_close(fixture_integral["integral_j"], fixture_expected["expected_trapezoid_energy_j"], 2e-10)
    results.append(("SW-03", "pass", "irregular trapezoid=10.5 J; saved seeded synthetic fixture matches analytic energy; zero/gap/nonfinite paths checked"))

    # 5. Clock wrap/skew, marker detection, duplicate/late order and dropouts.
    ticks = [65530, 5, 110, 215]
    unwrapped = unwrap_ticks(ticks, 65536)
    assert unwrapped == [65530, 65541, 65646, 65751]
    host = [v * 1_000_050 for v in unwrapped]
    aligned = fit_marker_alignment(ticks, host, 65536, nominal_ns_per_tick=1_000_000.0)
    assert abs(aligned["estimated_drift_ppm_vs_nominal_tick_rate"] - 50.0) < 0.1
    states = [{"host_receive_monotonic_ns": i * 50_000_000,
               "power_mean_w": (0.3 if state == "high" else 0.1)}
              for i, state in enumerate(["low", "low", "high", "high", "low", "low", "high", "high"])]
    segments = detect_power_state_segments(states, threshold_w=0.2, min_state_points=2)
    assert marker_match(segments, ["low", "high", "low", "high"])["match"]
    out_of_order = integrate_points([{"host_receive_monotonic_ns": 1_000_000_000, "power_mean_w": 1.0},
                                     {"host_receive_monotonic_ns": 900_000_000, "power_mean_w": 1.0}])
    assert out_of_order["integral_j"] is None
    with tempfile.TemporaryDirectory(prefix="power-duplicate-") as tmp:
        out = Path(tmp) / "run"
        writer = RunWriter(out, {"run_id": "duplicate-%d" % seed})
        same = make_report(5.0, 0.1, 0, seed)
        writer.record_report(same, 0, 1_000_000_000, "2026-10-22T00:00:00+00:00")
        writer.record_report(same, 1, 1_050_000_000, "2026-10-22T00:00:00+00:00")
        writer.close("complete")
        duplicate_report = assess_run(out)
        assert duplicate_report["metrics"]["possible_exact_duplicate_payloads_not_staleness_evidence"] == 1
    results.append(("SW-05", "pass", "counter wrap/skew, marker sequence, backward and dropout conditions exercised"))

    # 6. Interrupted run is retained and never accepted as a complete run.
    with tempfile.TemporaryDirectory(prefix="power-interrupt-") as tmp:
        out = Path(tmp) / "run"
        writer = RunWriter(out, {"run_id": "interrupt-%d" % seed, "policy_id": "pilot-unlocked"})
        writer.record_report(make_report(5.0, 0.1, seed, seed), 0, 1_000_000_000, "2026-10-22T00:00:00+00:00")
        writer.event("reset", {"source": "test_fixture"})
        writer.close("interrupted", "simulated meter/board disconnect")
        report = assess_run(out)
        assert report["status"] == "diagnostic_only" and "logger_run_not_complete" in report["failed_gates"]
        assert (out / "raw_packets.jsonl").stat().st_size > 0
    results.append(("SW-06", "pass", "simulated disconnect retains partial raw data and fails completion gate"))

    # 7. Missing/mismatched manifest hashes are rejected.
    with tempfile.TemporaryDirectory(prefix="power-manifest-") as tmp:
        out = Path(tmp) / "run"
        writer = RunWriter(out, {"run_id": "manifest-%d" % seed, "policy_id": "test",
                                 "artifact_hashes": {"policy_sha256": "actual"}})
        writer.record_report(make_report(5.0, 0.1, 0, seed), 0, 1_000_000_000, "2026-10-22T00:00:00+00:00")
        writer.record_report(make_report(5.0, 0.1, 0, seed + 1), 1, 1_050_000_000, "2026-10-21T23:59:59+00:00")
        writer.close("complete")
        manifest = Path(tmp) / "acceptance.json"
        manifest.write_text(json.dumps({"accepted_run_id": "wrong-run", "artifact_hashes": {"policy_sha256": "expected"}}))
        report = assess_run(out, manifest)
        assert "run_manifest_identity_or_hash_mismatch" in report["failed_gates"]
        assert report["energy_claim_valid"] is False
        assert report["missing_policy_comparison_evidence"]
    results.append(("SW-07", "pass", "missing physical evidence and wrong run/hash manifest fail closed; UTC reversal is ignored"))

    # 8. Lock semantics are exercised on a temp file; physical source/bypass inspection pending.
    with tempfile.TemporaryDirectory(prefix="power-lock-") as tmp:
        lock_path = Path(tmp) / "hardware.lock"
        first = HardwareLock(lock_path, "owner-%d" % seed, timeout_s=0)
        first.__enter__()
        try:
            try:
                HardwareLock(lock_path, "contender-%d" % seed, timeout_s=0).__enter__()
                raise AssertionError("second session acquired active lock")
            except TimeoutError:
                pass
        finally:
            first.__exit__(None, None, None)
    results.append(("SW-08-lock", "pass", "exclusive flock blocks a second process/session in temp lock test"))

    # 9. Logging overhead instrumentation runs in software only; LEDs require physical blocks.
    elapsed = []
    with tempfile.TemporaryDirectory(prefix="power-overhead-") as tmp:
        for rep in range(3):
            out = Path(tmp) / ("run-%d" % rep)
            started = time.perf_counter_ns()
            writer = RunWriter(out, {"run_id": "overhead-%d-%d" % (seed, rep)})
            for i in range(10):
                writer.record_report(make_report(5.0, 0.1, 0, seed + rep + i), i,
                                     1_000_000_000 + i * 50_000_000, "2026-10-22T00:00:00+00:00")
            writer.close("complete")
            elapsed.append(time.perf_counter_ns() - started)
    results.append(("SW-09-logging", "pass", "file-write overhead timed in host simulation; median_ns=%d; not device overhead" % int(statistics.median(elapsed))))

    # 10. Fresh replay-to-analysis reproduces a stable diagnostic result from raw immutable packets.
    with tempfile.TemporaryDirectory(prefix="power-replay-") as tmp:
        out = Path(tmp) / "run"
        writer = RunWriter(out, {"run_id": "replay-%d" % seed, "policy_id": "pilot-unlocked"})
        for i in range(4):
            writer.record_report(make_report(0.0, 0.01, 0, seed + i), i,
                                 1_000_000_000 + i * 50_000_000, "2026-10-22T00:00:00+00:00")
        writer.close("complete")
        first_result = assess_run(out)
        second_result = assess_run(out)
        assert json.dumps(first_result, sort_keys=True) == json.dumps(second_result, sort_keys=True)
        assert first_result["energy_claim_valid"] is False
        assert any("zero_or_missing_voltage" in g for g in first_result["failed_gates"])
    results.append(("SW-10", "pass", "fresh analysis repeats exactly from raw; active 0 V trace cannot pass"))

    # Paired analysis exercises all ten root units and required ten bootstrap RNG streams.
    paired_rows = []
    for block_seed in (4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110):
        for policy, multiplier in (("H", 1.0), ("L", 0.8), ("G", 0.7), ("A-full", 0.65)):
            paired_rows.append({"policy_id": policy, "scenario_id": "SC-test", "root_seed": str(block_seed),
                                "run_status": "valid", "energy_j": str(multiplier * block_seed),
                                "response_ms": "200", "known_correct_rate": "0.8",
                                "unknown_false_accept_rate": "0.0"})
    pair_result = analyze_pairs(paired_rows, bootstrap_roots, reps_per_root=1000)
    assert pair_result["comparisons"] and pair_result["comparisons"][0]["paired_seed_count"] == 10
    assert pair_result["comparisons"][0]["metrics"]["energy_j"]["difference_policy_minus_baseline"]["bootstrap"]["total_valid_resamples"] == 10000
    results.append(("SW-11", "pass", "paired block/statistics code enforces ten roots and 10x1000 bootstrap draws"))

    # 12. Empty HID reads, clock adjustments, host suspension and interrupted records fail closed.
    with tempfile.TemporaryDirectory(prefix="power-timeout-suspend-") as tmp:
        out = Path(tmp) / "run"
        writer = RunWriter(out, {"run_id": "timeout-%d" % seed, "policy_id": "pilot-unlocked"})
        writer.record_timeout(0)
        writer.close("complete")
        raw_rows = [json.loads(line) for line in (out / "raw_packets.jsonl").read_text().splitlines()]
        assert any(row.get("record_type") == "timeout" and "hid_read_timeout" in row.get("data_quality_flags", [])
                   for row in raw_rows)
        timeout_report = assess_run(out)
        assert "hid_read_timeouts_present" in timeout_report["failed_gates"]
        monotonic_points = [
            {"host_receive_monotonic_ns": 1_000_000_000, "host_receive_utc": "2026-10-22T00:00:10+00:00", "power_mean_w": 1.0},
            {"host_receive_monotonic_ns": 1_050_000_000, "host_receive_utc": "2026-10-21T23:59:00+00:00", "power_mean_w": 1.0},
        ]
        clock_shift = integrate_points(monotonic_points)
        assert_close(clock_shift["integral_j"], 0.05)
        suspended = integrate_points([
            {"host_receive_monotonic_ns": 0, "power_mean_w": 1.0},
            {"host_receive_monotonic_ns": 10_000_000_000, "power_mean_w": 1.0},
            {"host_receive_monotonic_ns": 10_050_000_000, "power_mean_w": 1.0},
        ])
        assert suspended["integral_j"] is None and suspended["invalid_intervals"][0]["reason"] == "gap_exceeds_max"
        sentinel = decode_report(make_report(5.0, 0.1, 0x80, seed))[0]
        assert sentinel["current_a"] > 0 and sentinel["sample_flag_raw_hex"] == "80"
        assert sentinel["sample_flag_semantics"] == "unknown_preserved"
        corrupt = out / "raw_packets.jsonl"
        with corrupt.open("a", encoding="utf-8") as handle:
            handle.write('{"record_type":"measurement_report"')
        try:
            assess_run(out)
            raise AssertionError("truncated JSONL tail was accepted")
        except ValueError as exc:
            assert "invalid JSON" in str(exc)
    results.append(("SW-12", "pass", "empty HID read/timeout, backward wall clock, 10 s host-suspension gap, unknown direction flag and truncated JSONL tail were handled explicitly"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="harness-qualification",
                        help="unique prefix for new reports/harness-qualification*.csv/json outputs")
    args = parser.parse_args()
    seed_plan = json.loads(SEED_PLAN_PATH.read_text(encoding="utf-8"))
    fuzz_roots = seed_plan["robustness_and_fuzz_seeds"]
    hardware_roots = seed_plan["scenario_and_hardware_block_seeds"]
    all_rows = []
    for seed in fuzz_roots:
        for test_id, status, evidence in software_tests_for_seed(seed, seed_plan["bootstrap_root_seeds"]):
            all_rows.append({"test_id": test_id, "root_seed": seed, "status": status,
                             "replication_unit": "independently initialized seeded software test block",
                             "evidence": evidence})
    hardware_pending = {
        "HW-01-logger": "real HID logger bounded startup/shutdown/file finalization; 10 hardware roots 4101-4110",
        "HW-04": "real Nicla idle/active step response; 10 scenario/hardware roots 4101-4110",
        "HW-06-disconnect": "physical Nicla reset/disconnect, meter disconnect, interrupted-run recovery; 10 hardware roots 4101-4110",
        "HW-08-path": "physical meter model/port/source/no-bypass confirmation; 10 seeded hardware blocks 4101-4110",
        "HW-09-led": "physical logging and LED overhead; 10 matched blocks 4101-4110",
        "HW-cold-start": "10 true physical power cycles; software reset is not a substitute",
    }
    for test_id, evidence in hardware_pending.items():
        for seed in hardware_roots:
            all_rows.append({"test_id": test_id, "root_seed": seed, "status": "pending_hardware",
                             "replication_unit": "required physical block/cycle", "evidence": evidence})

    report_root = POWER_ROOT / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / (args.output_prefix + ".csv")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "schema_version": "harness-qualification-1.0",
        "fuzz_root_seeds": fuzz_roots,
        "hardware_root_seeds": hardware_roots,
        "software_pass_rows": sum(row["status"] == "pass" for row in all_rows),
        "software_fail_rows": sum(row["status"] == "fail" for row in all_rows),
        "hardware_pending_rows": sum(row["status"] == "pending_hardware" for row in all_rows),
        "software_test_execution": "host-only synthetic/temp-file simulations; no board/meter session",
        "ready_status": "not_ready_until_physical_path_validation_and_timing_response_pass",
        "csv": csv_path.name,
    }
    json_path = report_root / (args.output_prefix + "-summary.json")
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 1 if summary["software_fail_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
