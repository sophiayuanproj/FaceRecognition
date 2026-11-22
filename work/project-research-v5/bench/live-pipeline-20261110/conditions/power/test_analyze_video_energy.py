"""Synthetic arithmetic and rejection checks. No meter or device is accessed."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analyze_video_energy import CSV_REQUIRED, Report, integrate, paired_differences, read_meter_csv
from run_video_power import PRIMARY, playback_receipt, trial_orders


EPOCH_MS = 946684800000


def write_csv(path, anomalous=False):
    fields = sorted(CSV_REQUIRED | {"power_w_calculated"})
    origin = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report, ms in enumerate((0, 50, 150, 200, 300, 400)):
            target = .10 + ms / 1000
            for sample, delta in enumerate((-.03, -.01, .01, .03)):
                power = target + delta
                voltage = 40 if anomalous and report == 2 and sample == 1 else 5
                writer.writerow({"run_id": "synthetic", "report_arrival_index": report,
                                 "sample_index_in_report": sample,
                                 "host_receive_monotonic_ns": ms * 1_000_000,
                                 "host_receive_utc": (origin + timedelta(milliseconds=ms)).isoformat(),
                                 "voltage_v": voltage, "current_a": power / voltage,
                                 "power_w_calculated": power, "crc_status": "pass"})


class SyntheticEnergyTests(unittest.TestCase):
    def test_single_block_order_is_explicit_or_seeded(self):
        self.assertEqual(trial_orders(PRIMARY, 1, 62001), [list(PRIMARY)])
        order = trial_orders(PRIMARY, 1, 62001, shuffle_single_block=True)
        self.assertEqual(order, trial_orders(PRIMARY, 1, 62001, shuffle_single_block=True))
        self.assertCountEqual(order[0], PRIMARY)
        self.assertNotEqual(order, trial_orders(PRIMARY, 1, 62002, shuffle_single_block=True))

    def test_physical_ids_disambiguate_local_block_one(self):
        base = dict(block=1, arm="fixed", video_sha256="a" * 64, status="valid",
                    meter_clock_start_unix_ms=0, meter_clock_end_unix_ms=1000,
                    duration_s=1, energy_j=.25, time_weighted_mean_w=.25,
                    firmware_config_sha256_pairs=[["b" * 64, "c" * 64]],
                    topology_id="inline", arm_order=["fixed", "adaptive"])
        trials = []
        for index, seed in enumerate((62001, 62002), 1):
            fixed = dict(base, physical_block_id=f"start-{index}", seed=seed,
                         boot_id=f"boot-{index}", meter_run_id=f"run-{index}",
                         meter_csv_sha256=f"meter-{index}", trial_file=f"start-{index}/fixed")
            adaptive = dict(fixed, arm="adaptive", trial_file=f"start-{index}/adaptive",
                            meter_clock_start_unix_ms=2000, meter_clock_end_unix_ms=3000,
                            energy_j=.20, time_weighted_mean_w=.20)
            trials.extend((fixed, adaptive))
        pairs = paired_differences(trials, "fixed")
        self.assertEqual([pair["physical_block_id"] for pair in pairs], ["start-1", "start-2"])
        self.assertEqual([pair["status"] for pair in pairs], ["paired", "paired"])
        for pair in pairs:
            self.assertAlmostEqual(pair["energy_delta_j"], -.05)
        for field, value in (("meter_run_id", "another-run"), ("meter_csv_sha256", "another-csv"),
                             ("boot_id", "after-reset"), ("firmware_config_sha256_pairs", [["different"]]),
                             ("topology_id", "different"), ("seed", 999), ("arm_order", ["adaptive", "fixed"])):
            with self.subTest(field=field):
                crossed = dict(trials[1], **{field: value})
                pair, = paired_differences([trials[0], crossed], "fixed")
                self.assertEqual(pair["status"], "unpaired")
                self.assertIn(field, pair["reason"])

    def test_visible_playback_requires_same_calibration_and_continuous_ticks(self):
        playback = [dict(event=event, browser_ms=elapsed, page_visibility="visible",
                         page_has_focus=False, viewport=[800, 800])
                    for event, elapsed in zip(("video_started", "video_tick", "video_ended"), (0, 500, 950))]
        geometry = {"boot_id": "fixture", "viewport": [800, 800]}
        self.assertTrue(playback_receipt(playback, geometry, "fixture")["verified"])
        self.assertEqual(playback_receipt(playback, geometry, "after-reset")["reason"],
                         "calibration_boot_or_viewport_changed")
        playback[1]["page_visibility"] = "hidden"
        self.assertEqual(playback_receipt(playback, geometry, "fixture")["reason"],
                         "player_tab_hidden_or_renderer_outdated")
        playback[1]["page_visibility"] = "visible"
        playback[-1]["browser_ms"] = 3000
        self.assertEqual(playback_receipt(playback, geometry, "fixture")["reason"], "playback_receipts_stalled")

    def test_linear_power_and_guardrails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meter.csv"
            write_csv(path)
            run, reports = read_meter_csv(path)
            self.assertEqual(run, "synthetic")
            self.assertEqual(len(reports), 6)
            result = integrate(reports, EPOCH_MS + 25, EPOCH_MS + 175, 100, 100)
            self.assertAlmostEqual(result["energy_j"], .030, places=10)
            self.assertAlmostEqual(result["time_weighted_mean_w"], .20, places=10)
            with self.assertRaisesRegex(ValueError, "does not bracket"):
                integrate(reports, EPOCH_MS - 10, EPOCH_MS + 175, 100, 100)
            with self.assertRaisesRegex(ValueError, "meter gap"):
                integrate(reports, EPOCH_MS + 25, EPOCH_MS + 175, 90, 100)
            with self.assertRaisesRegex(ValueError, "clock disagreement"):
                integrate([Report(EPOCH_MS, 0, 1, 0), Report(EPOCH_MS + 150, 10_000_000, 1, 1)],
                          EPOCH_MS, EPOCH_MS + 150, 200, 100)

    def test_cli_paired_episodes_and_clock_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meter = root / "meter.csv"
            write_csv(meter)
            trials = []
            for arm, ms in (("fixed", 225), ("adaptive", 25)):
                trial = root / (arm + ".json")
                trial.write_text(json.dumps({
                    "arm": arm, "block": 1, "video_sha256": "a" * 64, "measurement_ready": True,
                    "evidence_scope": "synthetic", "meter_csv_path": "meter.csv",
                    "meter_window_start_unix_ms": EPOCH_MS + ms - 1000,
                    "meter_window_end_unix_ms": EPOCH_MS + ms + 150 - 1000,
                    "clock_alignment": {"mode": "measured_offset", "meter_minus_render_ms": 1000,
                                        "uncertainty_ms": 10, "method": "synthetic injected offset"},
                    "episodes": [{"name": "quiet", "start_ms": 0, "end_ms": 50},
                                 {"name": "face", "start_ms": 50, "end_ms": 150}],
                }), encoding="utf-8")
                trials.append(trial)
            out = root / "energy.json"
            script = Path(__file__).with_name("analyze_video_energy.py")
            completed = subprocess.run([sys.executable, str(script), *map(str, trials),
                                        "--synthetic", "--baseline-arm", "fixed", "--per-episode",
                                        "--output", str(out)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(out.read_text())
            self.assertEqual(data["evidence_scope"], "synthetic_numeric_check")
            baseline, treatment = data["trials"]
            self.assertAlmostEqual(baseline["energy_j"], .060, places=10)
            self.assertAlmostEqual(treatment["energy_j"], .030, places=10)
            self.assertAlmostEqual(sum(row["energy_j"] for row in treatment["episodes"]), .030, places=10)
            self.assertAlmostEqual(data["pairs"][0]["energy_delta_j"], -.030, places=10)
            self.assertAlmostEqual(treatment["clock_offset_endpoint_energy_j"][0], .0285, places=10)
            self.assertAlmostEqual(treatment["clock_offset_endpoint_energy_j"][1], .0315, places=10)

    def test_supply_limits_check_individual_slots_in_integration_window(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meter.csv"
            write_csv(path, anomalous=True)
            _, reports = read_meter_csv(path)
            with self.assertRaisesRegex(ValueError, "report 2 has a slot outside"):
                integrate(reports, EPOCH_MS + 25, EPOCH_MS + 175, 200, 100, 4.75, 5.25)
            result = integrate(reports, EPOCH_MS + 210, EPOCH_MS + 290, 200, 100, 4.75, 5.25)
            self.assertGreater(result["energy_j"], 0)

    def test_cli_refuses_unbound_clock_or_unmarked_synthetic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trial = root / "trial.json"
            trial.write_text(json.dumps({
                "arm": "fixed", "block": 1, "video_sha256": "a" * 64, "measurement_ready": True,
                "evidence_scope": "synthetic", "meter_window_start_unix_ms": EPOCH_MS + 25,
                "meter_window_end_unix_ms": EPOCH_MS + 175,
            }), encoding="utf-8")
            script = Path(__file__).with_name("analyze_video_energy.py")
            out = root / "energy.json"
            for flags, reason in ((["--synthetic"], "explicit same-host clock"), ([], "do not agree")):
                run = subprocess.run([sys.executable, str(script), str(trial), *flags,
                                      "--output", str(out)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                row = json.loads(out.read_text())["trials"][0]
                self.assertIsNone(row["energy_j"])
                self.assertIn(reason, row["reason"])


if __name__ == "__main__":
    unittest.main()
