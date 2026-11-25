"""Check the manuscript's numeric claims against its frozen plotting tables.

This is an arithmetic/provenance audit, not a new experiment. For raw meter
decoding run the release's replay_belkin_primary.py as documented in README.md.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev
from zipfile import ZipFile


PAPER = Path(__file__).resolve().parents[1]
POWER_SHA = "66f9d14dd4b75b87d79d974cd28dc17492c5f3625db6edbc3255911e347e2356"
TRAINING_SEEDS = list(range(20261101, 20261111))
POWER_SEEDS = [62001, 62002, 62003, 62004, 62005, 62006, 62007, 62009, 62010, 62012]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(actual, expected, tolerance=1e-8):
    if abs(actual - expected) > tolerance:
        raise ValueError(f"Numeric mismatch: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-archive", type=Path)
    parser.add_argument("--host-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = PAPER / "data"
    arms = rows(data / "qualified-arms.csv")
    points = rows(data / "primary-points.csv")
    ledger = rows(data / "final-block-ledger.csv")
    included = [r for r in ledger if r["inclusion"] == "included"]
    assert len(arms) == 40 and len(points) == len(included) == 10
    assert len({r["boot_id"] for r in included}) == 10
    assert sorted(int(r["seed"]) for r in included) == POWER_SEEDS
    assert len(ledger) - len(included) == 28
    means = {}
    energies = {}
    for arm in ("capture-only", "detector-only", "fixed-small", "scene-gated-small"):
        group = [r for r in arms if r["arm"] == arm]
        assert len(group) == 10
        means[arm] = 1000 * mean(float(r["mean_power_w"]) for r in group)
        energies[arm] = mean(float(r["observed_j"]) for r in group)
    savings = [-1000 * float(r["delta_w"]) for r in points]
    saved = mean(savings)
    half = 2.2621571627409915 * stdev(savings) / 10**0.5
    near(saved, means["fixed-small"] - means["scene-gated-small"])
    near(saved, 49.543528629784184)
    near(100 * saved / means["fixed-small"], 7.282654544782961)
    episodes = rows(data / "qualified-episodes.csv")
    coverage, delays = {}, {}
    for arm in ("fixed-small", "scene-gated-small"):
        group = [r for r in episodes if r["arm"] == arm and r["expected_face"] == "True"]
        assert len(group) == 40
        matched = [r for r in group if int(r["matched_faces"]) > 0]
        coverage[arm] = len(matched)
        delays[arm] = median(float(r["first_matched_face_ms"]) / 1000 for r in matched)
        assert len(matched) == 30
        assert not any(int(r["matched_faces"]) for r in group if r["id"] == "far-outdoor")
    calls = {
        arm: {
            key: sum(int(r[key]) for r in arms if r["arm"] == arm)
            for key in ("detector_calls", "identity_runs", "identity_errors")
        }
        for arm in ("fixed-small", "scene-gated-small")
    }
    assert calls["fixed-small"] == {"detector_calls": 498, "identity_runs": 158, "identity_errors": 0}
    assert calls["scene-gated-small"] == {"detector_calls": 160, "identity_runs": 56, "identity_errors": 0}
    recognition = rows(data / "host-recognition-roots.csv")
    recognition_means = {}
    for route in ("compact_int8", "large_int8", "large_float"):
        group = [r for r in recognition if r["route"] == route]
        assert sorted(int(r["seed"]) for r in group) == TRAINING_SEEDS
        assert all(int(r["known_crops"]) == 100 and int(r["unknown_crops"]) == 300 for r in group)
        recognition_means[route] = {
            key: 100 * mean(float(r[key]) for r in group)
            for key in ("correct_known_acceptance", "unknown_fpir")
        }
    compression = rows(data / "compression-size-roots.csv")
    methods = sorted({r["method"] for r in compression})
    for method in methods:
        group = [r for r in compression if r["method"] == method]
        assert sorted(int(r["seed"]) for r in group) == TRAINING_SEEDS
        assert all(int(r["training_updates"]) == 75 for r in group)
    assert {int(r["bytes"]) for r in compression if r["method"] == "large_physical_channel_keep_50"} == {280248}
    result = {
        "status": "pass",
        "kind": "existing-evidence arithmetic and provenance audit; no new training or hardware tests",
        "qualified_physical_starts": 10,
        "accepted_schedule_seeds": POWER_SEEDS,
        "training_root_seeds": TRAINING_SEEDS,
        "excluded_attempts": len(ledger) - len(included),
        "workload_mean_mw": means,
        "workload_mean_j": energies,
        "saved_mw": saved,
        "saved_95_ci_mw": [saved - half, saved + half],
        "saved_percent": 100 * saved / means["fixed-small"],
        "face_episode_coverage": coverage,
        "conditional_median_delay_s": delays,
        "on_device_call_totals": calls,
        "host_recognition_percent": recognition_means,
        "plotted_compression_arms": len(methods),
        "physical_channel_file_reduction_percent": 100 * (327072 - 280248) / 327072,
    }
    if args.power_archive:
        content = args.power_archive.read_bytes()
        assert hashlib.sha256(content).hexdigest() == POWER_SHA
        assert len(content) == 41332455
        with ZipFile(args.power_archive) as archive:
            for local, member in (
                ("qualified-arms.csv", "resume/campaign/qualified-arms.csv"),
                ("qualified-episodes.csv", "resume/campaign/qualified-episodes.csv"),
                ("primary-points.csv", "resume/campaign/primary-points.csv"),
                ("final-block-ledger.csv", "resume/campaign/final-block-ledger.csv"),
            ):
                assert (data / local).read_bytes() == archive.read(member), f"Archive table differs: {local}"
        result["power_archive_sha256"] = POWER_SHA
    if args.host_source:
        paths = {
            "handoffs/D/CANDIDATE_PAIRWISE_FINAL_ANALYSIS.json": "9abf1419cab74f92c1a2d44d76008ccc11e9169eb014898c7076c53e60d5e60b",
            "reports/E/E_ROOT_METRICS.csv": "2da285be5cc33ba3b1535bc5bc041e847906fb49588cdbe90f0d800a495c4e11",
        }
        for relative, digest in paths.items():
            assert hashlib.sha256((args.host_source / relative).read_bytes()).hexdigest() == digest, relative
        result["host_source_hashes"] = paths
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
