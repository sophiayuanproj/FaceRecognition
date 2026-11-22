from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

APP = Path(__file__).resolve().parents[2]
WORKSPACE = APP.parent
sys.path.insert(0, str(APP))

from nicla_study.j_pilot_workflow import (apply_calibration, build_protocol,
    register_all, score_candidates, validate_attempt_log)  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class JPilotWorkflowTest(unittest.TestCase):
    def test_protocol_preregisters_finite_settings_and_scores_only_full_dev_results(self):
        protocol = build_protocol(WORKSPACE)
        self.assertEqual(protocol["candidate_budget"]["maximum_candidate_settings"], 12)
        self.assertEqual(protocol["candidate_budget"]["predeclared_candidate_settings"], 10)
        self.assertEqual(protocol["pilot_schedule"]["planned_capture_count"], 500)
        with tempfile.TemporaryDirectory(prefix="j pilot 空間 ") as temp:
            root = Path(temp)
            protocol_path, attempt_path = root / "protocol.json", root / "attempts.jsonl"
            result_path, selection_path = root / "results.jsonl", root / "selection.json"
            write_json(protocol_path, protocol)
            registered = register_all(protocol_path, attempt_path)
            self.assertEqual(registered["registered_candidate_settings"], 10)
            attempts = [json.loads(line) for line in attempt_path.read_text().splitlines()]
            self.assertEqual(validate_attempt_log(protocol, attempts)["status"], "pass")

            hashes = {name: ("a" if index == 0 else chr(ord("b") + index)) * 64
                      for index, name in enumerate(("model", "config", "controller", "roster", "calibration"))}
            hashes["config"] = protocol["source_sha256"]["runtime_config"]
            hashes["controller"] = protocol["source_sha256"]["controller_plan"]
            results = []
            for attempt in attempts:
                policy_id = attempt["policy_id"]
                results.append({"attempt_id": attempt["attempt_id"], "candidate_setting_id": policy_id,
                    "policy_id": policy_id, "model_id": policy_id,
                    "data_role": "development", "used_final_data": False,
                    "model_sha256": hashes["model"], "runtime_config_sha256": hashes["config"],
                    "controller_sha256": hashes["controller"], "roster_sha256": hashes["roster"],
                    "calibration_sha256": hashes["calibration"],
                    "measurement_status": "complete_verified", "independent_paired_blocks": 10,
                    "metrics": {"known_macro_top1": 0.95, "known_macro_accepted_coverage": 0.94,
                        "response_p95_ms": 800, "energy_per_accepted_correct_known_vs_fixed_high":
                        0.80 if policy_id == "A-cost" else 0.90},
                    "unknown_by_identity": [{"identity_id": "u-%02d" % index,
                        "encounters": 10, "false_identifications": 0} for index in range(30)]})
            result_path.write_text("".join(json.dumps(row) + "\n" for row in results), encoding="utf-8")
            selection = score_candidates(protocol_path, attempt_path, result_path, selection_path)
            self.assertEqual(selection["status"], "selected_development_candidate")
            self.assertEqual(selection["selected_candidate"]["policy_id"], "A-cost")
            self.assertFalse(selection["used_final_data"])

    def test_calibrated_config_is_staged_and_base_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="calibration ") as temp:
            root = Path(temp)
            base = root / "base.json"
            fit_path = root / "fit.json"
            staged = root / "staged.json"
            config = json.loads((APP / "firmware/runtime_config.json").read_text())
            write_json(base, config)
            original = base.read_bytes()
            fitted = {"status": "ready_for_hash_bound_runtime_update", "source_role": "development_only",
                "used_final_data": False, "risk_threshold": 0.1, "energy_value_mj_per_correct": 100.0,
                "risk_bins": [{"score_min": 0.8, "score_max": 0.9, "small_error_probability": 0.2,
                    "large_error_probability": 0.1, "incremental_energy_mj": 5.0,
                    "incremental_latency_ms": 25.0}],
                "calibration_sha256": "f" * 64,
                "measured_cost_source": {"status": "complete_verified", "source_role": "development_only"}}
            write_json(fit_path, fitted)
            result = apply_calibration(base, fit_path, staged)
            self.assertEqual(result["status"], "staged_requires_exact_H_requalification")
            self.assertEqual(base.read_bytes(), original)
            output_config = json.loads(staged.read_text())
            self.assertEqual(output_config["controller_calibration"]["source_role"], "development_only")
            self.assertEqual(len(output_config["controller_calibration"]["risk_bins"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
