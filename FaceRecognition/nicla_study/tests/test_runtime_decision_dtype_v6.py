"""Exercise the actual firmware identity decisions at binary32 boundaries."""

import ast
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest


FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))

from identity_contracts import (  # noqa: E402
    binary32,
    classifier_decision,
    gallery_decision,
    gate_unqualified_identity,
    strict_score_pass,
)
from open_set_decision import threshold_above_max  # noqa: E402


class RuntimeDecisionDtypeV6Tests(unittest.TestCase):
    def test_binary32_calibration_step_does_not_collapse(self):
        maximum = 0.19614700973033905
        old_float64_step = math.nextafter(maximum, math.inf)
        correct_float32_step = threshold_above_max([maximum])
        self.assertEqual(binary32(old_float64_step), maximum)
        self.assertGreater(binary32(correct_float32_step), maximum)
        self.assertFalse(strict_score_pass(maximum, correct_float32_step))

    def test_gallery_runtime_uses_float32_score_and_threshold(self):
        query = [1.0] + [0.0] * 63
        matching = [0.5, math.sqrt(0.75)] + [0.0] * 62
        other = [0.0, 1.0] + [0.0] * 62
        entries = [
            {"identity_id": "K%02d" % index,
             "vectors": {"probe": matching if index == 1 else other}}
            for index in range(1, 6)
        ]
        above = threshold_above_max([0.5])
        rejected = gallery_decision(query, entries, above, uncertainty_band=0.0,
                                    expected_model_id="probe")
        accepted = gallery_decision(query, entries, 0.5, uncertainty_band=0.0,
                                    expected_model_id="probe")
        self.assertEqual(rejected["status"], "OTHER")
        self.assertEqual(accepted["status"], "KNOWN_IDENTITY")
        self.assertEqual(accepted["candidate_score"], 0.5)

    def test_calibrated_gallery_cutoff_remains_reachable_with_uncertainty_band(self):
        query = [1.0] + [0.0] * 63
        other = [0.0, 1.0] + [0.0] * 62
        entries = [{"identity_id": "K%02d" % index,
                    "vectors": {"probe": query if index == 1 else other}}
                   for index in range(1, 6)]
        threshold = 0.9913240075111389
        legacy = gallery_decision(query, entries, threshold, .025, "probe")
        calibrated = gallery_decision(query, entries, threshold, .025, "probe",
                                      acceptance_uses_calibrated_cutoff=True)
        below = [0.98, math.sqrt(1 - 0.98**2)] + [0.0] * 62
        below_cutoff = gallery_decision(below, entries, threshold, .025, "probe",
                                        acceptance_uses_calibrated_cutoff=True)
        self.assertEqual(legacy["status"], "UNCERTAIN")
        self.assertEqual(calibrated["status"], "KNOWN_IDENTITY")
        self.assertEqual(calibrated["identity_id"], "K01")
        self.assertEqual(below_cutoff["status"], "UNCERTAIN")
        gated = gate_unqualified_identity(calibrated, False)
        self.assertEqual(gated["status"], "UNCERTAIN")
        self.assertIsNone(gated["identity_id"])
        self.assertEqual(gated["candidate_identity_id"], "K01")
        self.assertEqual(gated["candidate_score"], 1.0)

    def test_low_allocation_gallery_scan_matches_reference(self):
        rng = random.Random(20261103)

        def reference(query, entries, threshold, band, model_id):
            query = [float(value) for value in query]
            norm = math.sqrt(sum(value * value for value in query))
            query = [value / norm for value in query]
            candidates = []
            for entry in entries:
                template = [float(value) for value in entry["vectors"][model_id]]
                norm = math.sqrt(sum(value * value for value in template))
                template = [value / norm for value in template]
                score = binary32(sum(query[i] * template[i] for i in range(64)))
                candidates.append((score, entry["identity_id"]))
            candidates.sort(reverse=True)
            score, candidate = candidates[0]
            margin = binary32(score - candidates[1][0])
            threshold, band = binary32(threshold), binary32(band)
            upper, lower = binary32(threshold + band), binary32(threshold - band)
            if score >= upper:
                status, identity, reason = "KNOWN_IDENTITY", candidate, "above_gallery_threshold_band"
            elif score <= lower:
                status, identity, reason = "OTHER", None, "below_gallery_threshold_band"
            else:
                status, identity, reason = "UNCERTAIN", None, "inside_gallery_threshold_band"
            return status, identity, candidate, score, margin, reason

        for _ in range(60):
            query = [rng.uniform(-1.0, 1.0) for _ in range(64)]
            entries = [{"identity_id": "K%02d" % index,
                        "vectors": {"probe": [rng.uniform(-1.0, 1.0)
                                               for _ in range(64)]}}
                       for index in range(1, 6)]
            threshold = rng.uniform(-0.25, 0.95)
            band = rng.uniform(0.0, 0.04)
            expected = reference(query, entries, threshold, band, "probe")
            actual = gallery_decision(query, entries, threshold,
                                      uncertainty_band=band,
                                      expected_model_id="probe")
            observed = (actual.get("status"), actual.get("identity_id"),
                        actual.get("candidate_identity_id"),
                        actual.get("candidate_score"),
                        actual.get("nearest_competitor_margin"),
                        actual.get("reason"))
            self.assertEqual(observed, expected)

    def test_low_allocation_gallery_scan_keeps_descending_identity_tie_break(self):
        query = [1.0] + [0.0] * 63
        entries = [{"identity_id": "K%02d" % index,
                    "vectors": {"probe": query[:]}}
                   for index in range(1, 6)]
        result = gallery_decision(query, entries, 1.0, uncertainty_band=0.0,
                                  expected_model_id="probe")
        self.assertEqual(result["candidate_identity_id"], "K05")
        self.assertEqual(result["nearest_competitor_margin"], 0.0)

    def test_classifier_runtime_uses_float32_score_and_threshold(self):
        labels = ["K01", "K02", "K03", "K04", "K05", "OTHER"]
        logits = [math.log(0.5)] + [math.log(0.1)] * 5
        above = threshold_above_max([0.5])
        rejected = classifier_decision(logits, labels, above, uncertainty_band=0.0)
        accepted = classifier_decision(logits, labels, 0.5, uncertainty_band=0.0)
        self.assertEqual(rejected["status"], "OTHER")
        self.assertEqual(accepted["status"], "KNOWN_IDENTITY")
        self.assertEqual(accepted["candidate_score"], 0.5)

    def test_legacy_embedding_runtime_uses_binary32_gallery_threshold(self):
        source_path = FIRMWARE / "recognize_embedding.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "score_embedding_vector")
        namespace = {
            "binary32": binary32,
            "finite_number": lambda value: float(value) if math.isfinite(float(value)) else None,
            "cfg": None,
            "json": json,
            "load_templates": None,
            "load_calibration": None,
            "calibration_threshold": None,
        }
        query = [1.0] + [0.0] * 63
        matching = [0.5, math.sqrt(0.75)] + [0.0] * 62
        other = [0.0, 1.0] + [0.0] * 62
        entries = [{"identity_id": "K%02d" % index,
                    "vector": matching if index == 1 else other}
                   for index in range(1, 6)]
        calibration = {
            "model_sha256": {"large": "model-sha"},
            "roster_identity_ids": [entry["identity_id"] for entry in entries],
        }
        threshold = threshold_above_max([0.5])
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "templates.json"
            template_path.write_text(json.dumps({
                "model_sha256": {"large": "model-sha"},
                "model_config_sha256": "",
            }), encoding="utf-8")
            namespace["cfg"] = SimpleNamespace(
                MODEL_CONFIG_SHA256="", MODEL_SHA256={"large": "model-sha"},
                TEMPLATE_FILE=str(template_path), UNCERTAIN_BAND=0.0)
            namespace["load_templates"] = lambda model_id: entries
            namespace["load_calibration"] = lambda: calibration
            namespace["calibration_threshold"] = lambda model_id=None: threshold
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
            result = namespace["score_embedding_vector"](query, "large")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["decision_threshold"], threshold)
        self.assertEqual(result["cosine_similarity"], 0.5)


if __name__ == "__main__":
    unittest.main()
