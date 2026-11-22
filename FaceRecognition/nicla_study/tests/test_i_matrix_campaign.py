from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from nicla_study import i_matrix_campaign as campaign


WORKSPACE = Path(__file__).resolve().parents[3]
STUDY = WORKSPACE / "work/project-finish-v4"
MATRIX = STUDY / "config/I_PHYSICAL_MATRIX.json"


def _load_deployer():
    path = WORKSPACE / "FaceRecognition/tools/deploy_nicla_variant.py"
    spec = importlib.util.spec_from_file_location("deploy_nicla_variant_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IMatrixCampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = campaign._read(MATRIX)
        cls.variants = [row for row in cls.matrix["variants"]
                        if row.get("export_format") == campaign.FULL_INT8]
        cls._current_bundle_stage = tempfile.TemporaryDirectory(
            prefix=".i-matrix-current-bundles-", dir=str(WORKSPACE))
        cls.addClassCleanup(cls._current_bundle_stage.cleanup)
        cls.bundle_root = Path(cls._current_bundle_stage.name)
        cls.bundle_status = {}
        for row in cls.variants:
            prepared = campaign.prepare_variant_bundle(row, cls.bundle_root)
            manifest_path = cls.bundle_root / campaign._bundle_id(row) / "variant_manifest.json"
            if prepared.get("status") == "source_bundle_ready_hardware_qualification_pending":
                if not manifest_path.is_file():
                    raise AssertionError("bundle generator reported ready without a manifest: " + str(manifest_path))
                manifest = campaign._read(manifest_path)
                cls.bundle_status[str(row["artifact_sha256"]).lower()] = {
                    "status": manifest["status"],
                    "manifest_path": str(manifest_path),
                    "device_runtime_config_sha256": manifest["device_runtime_config_sha256"],
                    "model_id": manifest["model_id"],
                }
            else:
                cls.bundle_status[str(row.get("artifact_sha256", "")).lower()] = {
                    "status": prepared.get("status", "bundle_generation_failed"),
                    "reason": prepared.get("reason", "current bundle generator did not produce a deployable overlay"),
                }

    def test_catalog_preserves_full_matrix_and_schedules_only_supported_full_int8(self):
        catalog = campaign._catalog(self.matrix, self.bundle_status)
        self.assertEqual(catalog["full_catalog_pairwise_comparison_count_all_formats"], 544)
        self.assertEqual(catalog["target_deployable_arm_count"], 16)
        self.assertEqual(catalog["training_root_count"], 10)
        self.assertEqual(catalog["target_comparison_count"], 120)
        self.assertEqual(catalog["target_capture_count_if_all_qualified"], 2400)
        self.assertEqual(len(self.variants), 170)
        self.assertEqual(sum(row["status"] == "source_bundle_ready_hardware_qualification_pending"
                             for row in self.bundle_status.values()), 160)
        self.assertEqual(sum(row["status"] == "unsupported_target_contract"
                             for row in self.bundle_status.values()), 10)
        self.assertTrue(all(len(row["paired_artifacts"]) == 10 for row in catalog["comparisons"]))
        self.assertFalse(catalog["physical_measurements_performed"])

    def test_all_prepared_overlay_payloads_validate_and_tampering_is_rejected(self):
        deployer = _load_deployer()
        count = 0
        first_manifest = None
        for row in self.variants:
            manifest_path = self.bundle_root / campaign._bundle_id(row) / "variant_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest, sources = deployer._validate_bundle(manifest_path)
            self.assertEqual(manifest["artifact_sha256"], row["artifact_sha256"])
            self.assertEqual(len(sources), 14)
            first_manifest = first_manifest or manifest_path
            count += 1
        self.assertEqual(count, 160)
        manifest = campaign._read(first_manifest)
        target = next(iter(manifest["payload_files"]))
        manifest["payload_files"][target]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(prefix="i-matrix-tamper-") as temp:
            tampered = Path(temp) / "variant_manifest.json"
            tampered.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                deployer._validate_bundle(tampered)

    def test_hardware_gate_requires_exact_g_h_firmware_and_variant_config_receipts(self):
        qualification_rows = {}
        for row in self.variants:
            bundle = self.bundle_status[str(row.get("artifact_sha256", "")).lower()]
            if bundle.get("status") != "source_bundle_ready_hardware_qualification_pending":
                continue
            qualification_rows[(row["artifact_sha256"].lower(),
                                bundle["device_runtime_config_sha256"].lower())] = {
                "status": "complete_verified",
                "artifact_sha256": row["artifact_sha256"],
                "model_sha256": row["artifact_sha256"],
                "firmware_sha256": campaign.firmware_sha256(),
                "device_config_sha256": bundle["device_runtime_config_sha256"],
                "model_id": bundle["model_id"],
            }
        qualification = {"variant_qualifications": list(qualification_rows.values())}
        with tempfile.TemporaryDirectory(prefix="i-matrix-qual-") as temp:
            path = Path(temp) / "qualification.json"
            path.write_text(json.dumps(qualification), encoding="utf-8")
            catalog = campaign._catalog(self.matrix, self.bundle_status)
            eligible, report = campaign.eligibility(catalog, None, None, path)
            self.assertEqual(eligible, [])
            self.assertEqual(len(report["global_blockers"]), 2)

            good_h = {"status": "complete_verified", "evidence": {
                "firmware_sha256": campaign.firmware_sha256()}}
            eligible, report = campaign.eligibility(catalog, {"status": "complete_verified"},
                                                      good_h, path)
            self.assertEqual(len(eligible), 120)

            wrong = dict(qualification)
            wrong["variant_qualifications"] = [dict(row) for row in qualification["variant_qualifications"]]
            wrong["variant_qualifications"][0]["device_config_sha256"] = "f" * 64
            path.write_text(json.dumps(wrong), encoding="utf-8")
            catalog = campaign._catalog(self.matrix, self.bundle_status)
            eligible, _ = campaign.eligibility(catalog, {"status": "complete_verified"}, good_h, path)
            self.assertLess(len(eligible), 120)

    def test_resume_skips_only_complete_receipts_bound_to_the_same_plan(self):
        with tempfile.TemporaryDirectory(prefix="i-matrix-resume-") as temp:
            root = Path(temp)
            target = root / "captures" / "I-run"
            target.mkdir(parents=True)
            (target / "capture_result.json").write_text(json.dumps({
                "status": "complete_verified", "plan_sha256": "a" * 64}), encoding="utf-8")
            same, disposition = campaign._immutable_output(root, "I-run", True, "a" * 64)
            self.assertEqual(same, target)
            self.assertEqual(disposition, "skip_complete")
            different, rerun_of = campaign._immutable_output(root, "I-run", True, "b" * 64)
            self.assertEqual(different.name, "I-run-R02")
            self.assertEqual(rerun_of, "I-run")


if __name__ == "__main__":
    unittest.main()
