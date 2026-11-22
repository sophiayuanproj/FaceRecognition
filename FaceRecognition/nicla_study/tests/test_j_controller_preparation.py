from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "firmware"))
sys.path.insert(0, str(APP / "host"))

from runtime_support import (EscalationBudget, SimpleSceneTrigger, calibrated_risk_escalation,
                             camera_wake_retry_ms, classifier_route_for_embedding,
                             cost_aware_escalation, motion_detected, response_deadline_result)  # noqa: E402
from nicla_protocol import NiclaRecordValidator  # noqa: E402
from nicla_study.contracts import validate_model_output  # noqa: E402
from nicla_study.development_selector import (fit_controller_calibration,
    select_development_candidates)  # noqa: E402


def ticks_diff(a, b, modulus=1000):
    return ((int(a) - int(b) + modulus // 2) % modulus) - modulus // 2


def telemetry(sequence, ticks, firmware="f" * 64, config="c" * 64):
    return (json.dumps({"schema_version": 1, "schema_id": "nicla-face-telemetry-1",
        "event": "decision", "run_id": "boot-test", "sequence": sequence,
        "ticks_ms": ticks, "ticks_modulus_ms": 1000,
        "firmware_sha256": firmware, "model_config_sha256": config}) + "\n").encode()


def good_candidate():
    return {"model_id": "fixture-candidate", "data_role": "development", "used_final_data": False,
        "model_sha256": "a" * 64, "runtime_config_sha256": "b" * 64,
        "controller_sha256": "c" * 64, "roster_sha256": "d" * 64,
        "calibration_sha256": "e" * 64, "measurement_status": "complete_verified",
        "independent_paired_blocks": 10,
        "metrics": {"known_macro_top1": 0.95, "known_macro_accepted_coverage": 0.94,
            "response_p95_ms": 800, "energy_per_accepted_correct_known_vs_fixed_high": 0.85},
        "unknown_by_identity": [{"identity_id": "unknown-%02d" % i,
            "encounters": 10, "false_identifications": 0} for i in range(30)]}


class JControllerPreparationTest(unittest.TestCase):
    def test_ten_seeded_controller_and_telemetry_fault_blocks(self):
        seeds = list(range(20261511, 20261521))
        rows = []
        rows.append({"root_seed": seeds[0], "case": "extreme_score_threshold_routes", "status": "pass"})
        low = {"status": "KNOWN_IDENTITY", "candidate_score": 0.5}
        high = {"status": "KNOWN_IDENTITY", "candidate_score": 0.99}
        self.assertFalse(classifier_route_for_embedding(low, 0.99, 0.025)[0])
        self.assertTrue(classifier_route_for_embedding(high, 0.99, 0.025)[0])

        passed, reason = cost_aware_escalation(0.0, 5.0, 25.0, 100.0, 1000.0)
        self.assertFalse(passed)
        self.assertEqual(reason, "expected_gain_does_not_justify_energy")
        rows.append({"root_seed": seeds[1], "case": "no_measured_gain_from_escalation", "status": "pass"})

        budget = EscalationBudget(1, 100, lambda a, b: ticks_diff(a, b))
        self.assertTrue(budget.allow(950)[0])
        self.assertFalse(budget.allow(20)[0])
        self.assertTrue(budget.allow(100)[0])
        rows.append({"root_seed": seeds[2], "case": "escalation_budget_exhaustion_and_window_recovery", "status": "pass"})

        trigger = SimpleSceneTrigger({"mean_delta_u8": 8, "stdev_delta_u8": 6,
            "hysteresis_mean_u8": 4, "hysteresis_stdev_u8": 3,
            "min_detector_interval_ms": 100, "detector_fallback_ms": 250}, ticks_diff)
        self.assertTrue(trigger.decide(80, 20, 900)["run_detector"])
        changed = trigger.decide(90, 20, 50)
        self.assertTrue(changed["run_detector"])
        self.assertEqual(changed["reason"], "scene_change_after_minimum_interval")
        rows.append({"root_seed": seeds[3], "case": "scene_trigger_wrap_hysteresis_and_cooldown", "status": "pass"})

        validator = NiclaRecordValidator(firmware_sha256="f" * 64, config_sha256="c" * 64)
        first = validator.validate(telemetry(0, 990))
        wrapped = validator.validate(telemetry(1, 5))
        reset = validator.validate(telemetry(2, 2))
        self.assertEqual(first["status"], "accepted")
        self.assertEqual(wrapped["device_clock_event"], "wrap")
        self.assertEqual(reset["device_clock_event"], "reset_or_discontinuity")
        rows.append({"root_seed": seeds[4], "case": "device_tick_wrap_and_reset", "status": "pass"})

        duplicate = validator.validate(telemetry(2, 3))
        stale = validator.validate(telemetry(1, 4))
        gap = validator.validate(telemetry(4, 5))
        self.assertIn("duplicate_sequence", duplicate["errors"])
        self.assertIn("stale_or_inverted_sequence", stale["errors"])
        self.assertEqual(gap["sequence_gap"], 1)
        rows.append({"root_seed": seeds[5], "case": "duplicate_stale_and_gap_sequences", "status": "pass"})

        corrupt = validator.validate(b"{not-json\n")
        self.assertEqual(corrupt["status"], "rejected")
        self.assertTrue(any(error.startswith("invalid_jsonl") for error in corrupt["errors"]))
        rows.append({"root_seed": seeds[6], "case": "corrupted_telemetry_record", "status": "pass"})

        wrong_hash = validator.validate(telemetry(5, 6, firmware="0" * 64, config="1" * 64))
        self.assertIn("firmware_hash_mismatch", wrong_hash["errors"])
        self.assertIn("configuration_hash_mismatch", wrong_hash["errors"])
        rows.append({"root_seed": seeds[7], "case": "wrong_firmware_and_configuration_hashes", "status": "pass"})

        self.assertFalse(motion_detected(None, None)[0])
        self.assertEqual(camera_wake_retry_ms(1), 1000)
        self.assertEqual(camera_wake_retry_ms(2), 2000)
        self.assertEqual(camera_wake_retry_ms(99), 8000)
        late, suppressed = response_deadline_result({"status": "KNOWN_IDENTITY", "identity_id": "K01"}, 5001, 5000)
        self.assertTrue(suppressed)
        self.assertEqual(late["status"], "UNCERTAIN")
        rows.append({"root_seed": seeds[8], "case": "missing_motion_fail_safe_wake_backoff_and_response_deadline", "status": "pass"})

        with self.assertRaises((TypeError, ValueError)):
            validate_model_output({"model_type": "classifier_logits", "values": [float("nan")] * 6},
                                  "classifier_logits", 6)
        should_escalate, reason, _risk = calibrated_risk_escalation([], 0.99, 0.1)
        self.assertFalse(should_escalate)
        self.assertEqual(reason, "risk_score_bin_missing_or_ambiguous")
        rows.append({"root_seed": seeds[9], "case": "impossible_model_output_and_missing_risk_calibration", "status": "pass"})

        self.assertEqual(len(rows), 10)
        report = {"schema": "J-controller-software-fault-blocks-1", "status": "pass",
            "evidence_tier": "host_software_only", "seed_unit": "independent deterministic fault/input block",
            "physical_trials": 0, "blocks": rows}
        study = APP.parent / "work/project-finish-v4"
        out = study / "reports/J/controller-fault-blocks.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def test_development_selector_is_gated_and_never_auto_freezes(self):
        candidate = good_candidate()
        result = select_development_candidates([candidate])
        self.assertEqual(result["status"], "selected_development_candidate")
        self.assertTrue(result["freeze_allowed"])
        bad = dict(candidate, data_role="final")
        rejected = select_development_candidates([bad])
        self.assertEqual(rejected["status"], "no_qualified_candidate")
        self.assertFalse(rejected["freeze_allowed"])
        self.assertIn("not_development_role", rejected["attempt_log"][0]["reasons"])

    def test_risk_and_cost_calibration_requires_encounters_and_measured_cost(self):
        rows = [{"data_role": "development", "used_final_data": False,
            "encounter_id": "enc-%02d" % index, "session_id": "session-%02d" % (index % 10),
            "identity_id": "person-%02d" % (index % 10), "small_candidate_score": 0.85,
            "small_correct": index % 5 != 0, "large_correct": True} for index in range(30)]
        cost = {"status": "complete_verified", "source_role": "development_only",
            "incremental_energy_mj": 10.0, "incremental_latency_ms": 20.0}
        fitted = fit_controller_calibration(rows, cost, energy_value_mj_per_correct=100.0)
        self.assertEqual(fitted["status"], "ready_for_hash_bound_runtime_update")
        self.assertEqual(len(fitted["risk_bins"]), 1)
        self.assertGreater(fitted["risk_bins"][0]["expected_accuracy_gain"], 0)
        with self.assertRaises(PermissionError):
            fit_controller_calibration([dict(rows[0], data_role="final")], cost, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
