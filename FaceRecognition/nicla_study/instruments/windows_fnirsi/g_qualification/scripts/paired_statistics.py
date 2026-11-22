#!/usr/bin/env python3
"""Paired block summaries and 10x1,000 seeded bootstrap intervals."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SEED_PLAN = PROJECT_ROOT / "work/research-plan-2026-10-22/seed-plan.json"
BASELINES = ("H", "L", "G")
METRICS = (
    "energy_j", "energy_per_attempted_decision_j", "energy_per_valid_opportunity_j",
    "energy_per_correct_known_encounter_j", "captures", "camera_wakeups", "detector_calls",
    "small_model_calls", "large_model_calls", "escalations", "attempted_decisions",
    "known_correct_rate", "unknown_false_accept_rate", "response_ms", "response_ms_median",
)


def quantile(values, p):
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    index = (len(data) - 1) * p
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - index) + data[hi] * (index - lo)


def bootstrap_paired(values, roots, reps_per_root=1000):
    all_means = []
    root_intervals = []
    n = len(values)
    for boot_root in roots:
        rng = random.Random(boot_root)
        draws = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(reps_per_root)]
        all_means.extend(draws)
        root_intervals.append({"bootstrap_root_seed": boot_root,
                               "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)]})
    intervals = [item["ci95"] for item in root_intervals]
    return {
        "resamples_per_bootstrap_root": reps_per_root,
        "bootstrap_root_count": len(roots),
        "total_valid_resamples": len(all_means),
        "ci95_percentile": [quantile(all_means, 0.025), quantile(all_means, 0.975)],
        "rootwise_ci_lower_range": [min(c[0] for c in intervals), max(c[0] for c in intervals)],
        "rootwise_ci_upper_range": [min(c[1] for c in intervals), max(c[1] for c in intervals)],
        "per_root": root_intervals,
    }


def analyze(rows, bootstrap_roots, reps_per_root=1000):
    indexed = {}
    failures = []
    for row in rows:
        key = (str(row.get("policy_id", "")), str(row.get("scenario_id", "")), str(row.get("root_seed", "")))
        if key in indexed:
            raise ValueError("duplicate policy/scenario/root row: %r" % (key,))
        status = row.get("run_status", row.get("status", "valid"))
        indexed[key] = row
        if status not in ("valid", "complete", "valid_power_input"):
            failures.append({"policy_id": key[0], "scenario_id": key[1], "root_seed": key[2], "status": status,
                             "failure_reason": row.get("failure_reason", "")})

    policies = sorted({key[0] for key in indexed})
    scenarios = sorted({key[1] for key in indexed})
    comparisons = []
    for scenario in scenarios:
        for policy in policies:
            if policy in BASELINES:
                continue
            for baseline in BASELINES:
                if baseline not in policies:
                    continue
                seeds = sorted({key[2] for key in indexed if key[0] == policy and key[1] == scenario}
                               & {key[2] for key in indexed if key[0] == baseline and key[1] == scenario})
                comparison = {"scenario_id": scenario, "policy_id": policy, "baseline_policy_id": baseline,
                              "paired_root_seeds": seeds, "paired_seed_count": len(seeds), "metrics": {}}
                if len(seeds) < 10:
                    comparison["status"] = "insufficient_paired_root_seeds"
                    comparisons.append(comparison)
                    continue
                valid_pairs = []
                for seed in seeds:
                    a, b = indexed[(policy, scenario, seed)], indexed[(baseline, scenario, seed)]
                    sa = a.get("run_status", a.get("status", "valid"))
                    sb = b.get("run_status", b.get("status", "valid"))
                    if sa in ("valid", "complete", "valid_power_input") and sb in ("valid", "complete", "valid_power_input"):
                        valid_pairs.append((seed, a, b))
                for metric in METRICS:
                    paired = []
                    ratios = []
                    for seed, a, b in valid_pairs:
                        try:
                            av, bv = float(a[metric]), float(b[metric])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if math.isfinite(av) and math.isfinite(bv):
                            paired.append((seed, av, bv))
                            if metric == "energy_j" and bv != 0:
                                ratios.append(av / bv)
                    diffs = [av - bv for _, av, bv in paired]
                    if len(paired) < 10:
                        comparison["metrics"][metric] = {"status": "missing_or_invalid_paired_metric",
                                                         "valid_pair_count": len(paired)}
                        continue
                    item = {
                        "status": "ok",
                        "paired_root_seeds": [seed for seed, _, _ in paired],
                        "difference_policy_minus_baseline": {
                            "mean": statistics.fmean(diffs),
                            "sd": statistics.stdev(diffs),
                            "median": statistics.median(diffs),
                            "min": min(diffs), "max": max(diffs),
                            "raw_paired_differences": diffs,
                            "bootstrap": bootstrap_paired(diffs, bootstrap_roots, reps_per_root),
                        },
                    }
                    if ratios:
                        item["energy_ratio_policy_over_baseline"] = {
                            "mean": statistics.fmean(ratios), "median": statistics.median(ratios),
                            "n": len(ratios), "raw_paired_ratios": ratios,
                        }
                    comparison["metrics"][metric] = item
                comparison["status"] = "paired_analysis_ready" if all(
                    comparison["metrics"].get(metric, {}).get("status") == "ok" for metric in ("energy_j",)
                ) else "partial_metrics_or_failures_retained"
                comparison["all_seed_failures_retained"] = True
                comparisons.append(comparison)
    return {
        "schema_version": "paired-power-statistics-1.0",
        "primary_unit": "paired root-seed block, not individual HID samples",
        "bootstrap_roots": bootstrap_roots,
        "bootstrap_reps_per_root": reps_per_root,
        "comparisons": comparisons,
        "failure_ledger_rows": failures,
        "zero_success_rule": "energy per correctly recognized known encounter is undefined/infinite when correct_count=0; false accepts do not count as successes",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blocks_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps-per-root", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.reps_per_root < 1000:
        parser.error("at least 1,000 bootstrap draws per root are required")
    seed_plan = json.loads(SEED_PLAN.read_text(encoding="utf-8"))
    with args.blocks_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = analyze(rows, seed_plan["bootstrap_root_seeds"], args.reps_per_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "comparison_count": len(result["comparisons"]),
                      "failure_count": len(result["failure_ledger_rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
