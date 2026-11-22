"""Regression checks for the observed fail-open and cross-host freeze defects."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from nicla_study.development_selector import select_development_candidates, fit_controller_calibration
from nicla_study.freeze_selection import freeze, verify_receipt_sources
from nicla_study.fs import atomic_write_json
from nicla_study.paths import resolve_study_root, PathResolutionError


def candidate():
    return {"model_id": "control", "data_role": "development", "used_final_data": False,
            "measurement_status": "complete_verified", "independent_paired_blocks": 10,
            **{name: "a" * 64 for name in ["model_sha256", "runtime_config_sha256",
                "controller_sha256", "roster_sha256", "calibration_sha256"]},
            "metrics": {"known_macro_top1": .95, "known_macro_accepted_coverage": .95,
                        "response_p95_ms": 800.0,
                        "energy_per_accepted_correct_known_vs_fixed_high": .8},
            "unknown_by_identity": [{"identity_id": "u-%02d" % index,
                "encounters": 10, "false_identifications": 0} for index in range(30)]}


class V5ValidationTest(unittest.TestCase):
    def test_ten_seeded_metric_fault_blocks_and_valid_control(self):
        mutations = [math.nan, math.inf, -math.inf, None, True, False, "0.95", -1.0]
        for seed in range(53001, 53011):
            rng = random.Random(seed)
            fields = list(candidate()["metrics"])
            rng.shuffle(fields)
            for field in fields:
                for value in mutations:
                    row = candidate()
                    row["metrics"][field] = value
                    result = select_development_candidates([row])
                    self.assertFalse(result["freeze_allowed"], (seed, field, value))
                    json.dumps(result, allow_nan=False)
            row = candidate()
            for field in ["known_macro_top1", "known_macro_accepted_coverage"]:
                row["metrics"][field] = 1.1
            self.assertFalse(select_development_candidates([row])["freeze_allowed"])
        self.assertTrue(select_development_candidates([candidate()])["freeze_allowed"])

    def test_counts_unknown_duplicates_and_zero_cost(self):
        for seed in range(53001, 53011):
            for bad_count in [True, "10", 10.0, None, -1]:
                row = candidate()
                row["independent_paired_blocks"] = bad_count
                self.assertFalse(select_development_candidates([row])["freeze_allowed"])
            row = candidate()
            row["unknown_by_identity"][seed % 29 + 1]["identity_id"] = "u-00"
            self.assertFalse(select_development_candidates([row])["freeze_allowed"])
        row = candidate()
        row["metrics"]["response_p95_ms"] = 0
        row["metrics"]["energy_per_accepted_correct_known_vs_fixed_high"] = 0
        self.assertTrue(select_development_candidates([row])["freeze_allowed"])

    def test_nonfinite_serialization_preserves_prior_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            atomic_write_json(path, {"generation": 1})
            original = path.read_bytes()
            for bad in [math.nan, math.inf, -math.inf]:
                with self.assertRaises(ValueError):
                    atomic_write_json(path, {"invalid": bad})
                self.assertEqual(path.read_bytes(), original)

    def test_calibration_does_not_treat_strings_as_true_outcomes(self):
        row = {"data_role": "development", "used_final_data": False,
               "encounter_id": "a", "session_id": "b", "identity_id": "c",
               "small_candidate_score": .85, "small_correct": "False", "large_correct": True}
        cost = {"status": "complete_verified", "source_role": "development_only",
                "incremental_energy_mj": 0.0, "incremental_latency_ms": 0.0}
        with self.assertRaises(ValueError):
            fit_controller_calibration([row], cost, 100.0)

    def test_ten_freeze_relocations_and_tampering(self):
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in range(53001, 53011):
            with tempfile.TemporaryDirectory(prefix="v5 freeze 空間 ") as temp:
                a, b = Path(temp) / "original", Path(temp) / "relocated Ω"
                a.mkdir()
                files = {key: a / (key + ".json") for key in
                         ["runtime_config", "model", "calibration", "roster", "controller_plan", "protocol"]}
                for path in files.values():
                    path.write_text(json.dumps({"seed": seed}))
                files["runtime_config"].write_text(json.dumps({"identity": {"small": {
                    "model_sha256": digest(files["model"])}}}))
                row = candidate()
                for key, source in [("model", "model"), ("runtime_config", "runtime_config"),
                                    ("calibration", "calibration"), ("roster", "roster"),
                                    ("controller", "controller_plan")]:
                    row[key + "_sha256"] = digest(files[source])
                selection = a / "selection.json"
                selection.write_text(json.dumps({"status": "selected_development_candidate",
                    "freeze_allowed": True, "data_role": "development_only", "used_final_data": False,
                    "selected_candidate": row}))
                receipt = a / "receipt.json"
                freeze(selection, files["runtime_config"], files["model"], files["calibration"],
                    files["roster"], files["controller_plan"], files["protocol"], "b" * 64, receipt,
                    workspace=a)
                shutil.copytree(a, b)
                shutil.rmtree(a)
                self.assertEqual(verify_receipt_sources(b / "receipt.json")["status"], "pass")
                (b / "model.json").write_text("tampered")
                with self.assertRaises(ValueError):
                    verify_receipt_sources(b / "receipt.json")
                self.assertEqual(resolve_study_root(b, "work/project-research-v5"),
                                 b.resolve() / "work/project-research-v5")
                with self.assertRaises(PathResolutionError):
                    resolve_study_root(b, "../escape")
