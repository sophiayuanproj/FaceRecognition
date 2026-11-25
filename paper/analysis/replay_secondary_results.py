"""Reproduce the R19 exploratory stage-path and order summaries from the final ZIP."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from csv import DictReader, DictWriter
from hashlib import file_digest
from io import StringIO
import json
from pathlib import Path
from statistics import mean, median, stdev
from zipfile import ZipFile


ARCHIVE_SHA256 = "66f9d14dd4b75b87d79d974cd28dc17492c5f3625db6edbc3255911e347e2356"
CAMPAIGN = "resume/campaign"
FIXED = "fixed-small"
GATED = "scene-gated-small"
PERIODIC = "bounded_periodic_detector_fallback"
CHANGE = "scene_change_after_minimum_interval"
STABLE = "scene_stable"


def csv_rows(archive, member):
    return list(DictReader(StringIO(archive.read(member).decode("utf-8-sig"))))


def observation_key(row):
    return row["boot_id"], row["frame_id"]


def read_gate_trials(archive, ledger):
    counts_rows, first_rows = [], []
    for block in ledger:
        trial_dir = f"resume/primary-{block['attempt_id']}/paired/01-{GATED}"
        events = json.loads(archive.read(f"{trial_dir}/device_events.json"))
        trial = json.loads(archive.read(f"{trial_dir}/RESULT.json"))
        if trial["boot_id"] != block["boot_id"]:
            raise ValueError(f"Different boot ID for {block['attempt_id']}")
        window = {observation_key(row): row for row in trial["rows"]}
        gate = {}
        for event in events:
            if event.get("event") == "scene_trigger_check":
                key = observation_key(event)
                if key in window:
                    if key in gate:
                        raise ValueError(f"Duplicate scene gate event for {key}")
                    gate[key] = event["record"]["reason_code"]
        calls = Counter(gate.values())
        if calls[PERIODIC] + calls[CHANGE] != trial["detector_calls"] or calls[STABLE] != trial["quiet_skips"]:
            raise ValueError(f"Joined gate events do not match trial totals for {block['attempt_id']}")
        counts_rows.append(
            {
                "planned_block": block["planned_block"],
                "attempt_id": block["attempt_id"],
                "scene_gate_checks": len(gate),
                "stable_skips": calls[STABLE],
                "periodic_detector_calls": calls[PERIODIC],
                "scene_change_detector_calls": calls[CHANGE],
                "all_detector_calls": trial["detector_calls"],
            }
        )
        for episode in trial["episodes"]:
            if not episode["expected_face"]:
                continue
            candidates = [
                row for row in trial["rows"] if row["episode"] == episode["name"] and row.get("matched_face")
            ]
            first = min(candidates, key=lambda row: row["estimated_media_ms"], default=None)
            reported_delay = episode["first_matched_face_ms"]
            if first is None and reported_delay is not None or first is not None and reported_delay is None:
                raise ValueError(f"Missing detector match for {episode['id']}")
            first_rows.append(
                {
                    "planned_block": block["planned_block"],
                    "attempt_id": block["attempt_id"],
                    "episode_id": episode["id"],
                    "matched": first is not None,
                    "first_match_reason": gate[observation_key(first)] if first is not None else "",
                    "episode_onset_to_first_match_ms": reported_delay if first is not None else "",
                }
            )
    return counts_rows, first_rows


def read_power_order(archive, ledger):
    points = {row["attempt_id"]: row for row in csv_rows(archive, f"{CAMPAIGN}/primary-points.csv")}
    rows = []
    for block in ledger:
        point = points[block["attempt_id"]]
        if block["planned_block"] != point["planned_block"]:
            raise ValueError(f"Power analysis block mismatch for {block['attempt_id']}")
        order = block["executed_arms"].split()
        first = FIXED if order.index(FIXED) < order.index(GATED) else GATED
        rows.append(
            {
                "planned_block": block["planned_block"],
                "attempt_id": block["attempt_id"],
                "full_arm_order": block["executed_arms"],
                "inference_policy_first": first,
                "mean_input_power_saved_mw": -1000 * float(point["delta_w"]),
            }
        )
    return rows


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(counts, matches, power):
    reasons, orders = defaultdict(list), defaultdict(list)
    for row in matches:
        if row["matched"]:
            reasons[row["first_match_reason"]].append(row["episode_onset_to_first_match_ms"])
    for row in power:
        orders[row["inference_policy_first"]].append(row["mean_input_power_saved_mw"])
    all_power = [row["mean_input_power_saved_mw"] for row in power]
    count_keys = list(counts[0])[2:]
    return {
        "analysis": "descriptive secondary summaries from the ten qualified starts",
        "source_sha256": ARCHIVE_SHA256,
        "gated_scene_counts": {key: sum(row[key] for row in counts) for key in count_keys},
        "gated_first_matched_face_episodes": {
            key: {"n": len(values), "median_latency_ms": median(values)} for key, values in sorted(reasons.items())
        },
        "power_saved_mw": {"mean": mean(all_power), "sd": stdev(all_power), "min": min(all_power), "max": max(all_power)},
        "power_saved_mw_by_first_policy": {
            key: {"n": len(values), "mean": mean(values), "min": min(values), "max": max(values)}
            for key, values in sorted(orders.items())
        },
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    with args.evidence.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Final return ZIP digest mismatch.")
    with ZipFile(args.evidence) as archive:
        ledger = [
            row for row in csv_rows(archive, f"{CAMPAIGN}/final-block-ledger.csv")
            if row["inclusion"] == "included"
        ]
        if len(ledger) != 10 or len({row["boot_id"] for row in ledger}) != 10:
            raise ValueError("Expected ten qualified independent boots.")
        gate, episodes = read_gate_trials(archive, ledger)
        power = read_power_order(archive, ledger)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "gate-path-by-boot.csv", gate)
    write_csv(args.output / "gate-first-match-by-episode.csv", episodes)
    write_csv(args.output / "order-sensitivity-by-boot.csv", power)
    result = json.dumps(summarize(gate, episodes, power), indent=2) + "\n"
    (args.output / "secondary-summary.json").write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
