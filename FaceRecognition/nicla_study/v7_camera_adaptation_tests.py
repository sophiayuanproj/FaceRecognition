"""Focused regression tests for the grouped camera adaptation data boundary."""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/nicla_study"))
from nicla_study import v7_accuracy_runner as runner


class CameraAdaptationContractTests(unittest.TestCase):
    @staticmethod
    def synthetic_camera_rows():
        rows = []
        images = {}
        for label in runner.KNOWN:
            for group_index in range(5):
                record_id = f"camera:{label}:{group_index}"
                group = f"group:{label}:{group_index}"
                rows.append({
                    "record_id": record_id, "label": label, "role": "train",
                    "source_identity": group, "source_group": group, "valid_input": True,
                })
                images[record_id] = np.full((96, 96, 1), group_index, dtype=np.uint8)
        for group_index in range(13):
            record_id = f"camera:OTHER:{group_index}"
            identity = f"unknown:{group_index}"
            rows.append({
                "record_id": record_id, "label": "OTHER", "role": "other_train",
                "source_identity": identity, "source_group": identity, "valid_input": True,
            })
            images[record_id] = np.full((96, 96, 1), group_index, dtype=np.uint8)
        return rows, images

    def test_grouped_split_batches_use_camera_fit_only_and_keep_groups_disjoint(self):
        rows, images = self.synthetic_camera_rows()
        fit, fit_images, stop, stop_images, split = runner.split_camera_training_groups(
            rows, images, seed=57101, stop_fraction=0.2)
        self.assertEqual(fit_images.shape, (len(fit), 96, 96, 1))
        self.assertEqual(stop_images.shape, (len(stop), 96, 96, 1))
        self.assertEqual(split["fit_stopping_group_overlap"], 0)
        self.assertFalse({r["identity_id"] for r in fit if r["label"] == "OTHER"} &
                         {r["identity_id"] for r in stop if r["label"] == "OTHER"})
        self.assertTrue(all(r["source_kind"] == "camera_replay_train" and
                            r["role"] in ("train", "other_train") for r in fit))
        batches = runner.balanced_batch_indices(fit, 57101 + 818181, batch_size=24, examples_per_class=4)
        contract = runner.validate_camera_adaptation_batches(fit, batches, 24, 4)
        self.assertTrue(contract["camera_only"])
        self.assertEqual(contract["batches"], 1)
        self.assertEqual(Counter(fit[index]["label"] for index in batches[0]),
                         Counter({label: 4 for label in runner.CLASSES}))

    def test_gradient_batch_rejects_source_rows_from_outside_camera_fit(self):
        rows, images = self.synthetic_camera_rows()
        fit, _, _, _, _ = runner.split_camera_training_groups(rows, images, 57101, 0.2)
        batches = runner.balanced_batch_indices(fit, 818182, 24, 4)
        corrupt_fit = list(fit)
        index = batches[0][0]
        corrupt_fit[index] = dict(corrupt_fit[index], source_kind="lfw_source")
        with self.assertRaisesRegex(AssertionError, "non-camera-train role"):
            runner.validate_camera_adaptation_batches(corrupt_fit, batches, 24, 4)

    def test_training_preflight_skips_calibration_development_and_future_paths(self):
        receipts = []
        roles = ["threshold_calibration", "unknown_calibration", "development",
                 "unknown_development", "future_final"]
        for index in range(135):
            role = roles[index % len(roles)]
            receipts.append({"request": {"source": {"role": role}}, "metadata": {}})
        with mock.patch.object(runner.camera_replay, "latest_receipts",
                               return_value=(Path("not-opened-receipt"), receipts)), \
             mock.patch.object(runner.camera_replay, "sha256_file", return_value="a" * 64):
            rows, images, receipt = runner.load_camera_train_pool(80)
        self.assertEqual(rows, [])
        self.assertEqual(images, {})
        self.assertEqual(receipt["eligible_rows"], 0)
        self.assertEqual(receipt["private_input_records"], [])
        self.assertFalse(receipt["calibration_or_development_frame_paths_opened"])
        self.assertFalse(receipt["future_final_frames_opened"])
        self.assertEqual(receipt["skipped_role_counts_metadata_only"]["future_final"], 27)

    def test_postfreeze_eval_loader_skips_training_and_future_paths_before_io(self):
        train_receipts = [
            {"request": {"source": {"role": "train"}}, "metadata": {}}
            for _ in range(135)
        ]
        dev_receipts = [
            {"request": {"source": {"role": "future_final"}}, "metadata": {}}
            for _ in range(100)
        ]

        def fake_latest(pointer):
            return (Path("not-opened-receipt"),
                    train_receipts if pointer == "LATEST_ADAPTATION.txt" else dev_receipts)

        with mock.patch.object(runner.camera_replay, "latest_receipts", side_effect=fake_latest), \
             mock.patch.object(runner.camera_replay, "sha256_file", return_value="b" * 64):
            rows, images, receipt = runner.load_camera_images(
                80, allowed_roles={"threshold_calibration", "unknown_calibration",
                                   "development", "unknown_development"})
        self.assertEqual(rows, [])
        self.assertEqual(images, {})
        self.assertFalse(receipt["private_input_records"])
        self.assertFalse(receipt["future_final_frames_opened"])
        self.assertEqual(receipt["skipped_roles_metadata_only"].get("train"), 135)
        self.assertEqual(receipt["skipped_roles_metadata_only"].get("future_final"), 100)

    def test_camera_calibration_development_require_model_selection_freeze(self):
        with self.assertRaisesRegex(RuntimeError, "sealed until model selection is frozen"):
            runner.require_camera_adaptation_evaluation_freeze({"status": "running"})
        runner.require_camera_adaptation_evaluation_freeze({
            "status": "evaluation_pending", "model_selection_frozen": True,
        })

    def test_training_mode_requires_explicit_second_launch_gate(self):
        config_path = ROOT / "work/project-research-v5/runs/accuracy_search_v1/configs/v8_camera_adaptation_from_v7_a010_s57101_scout.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update({
            "training_launch_enabled": True,
            "runner_sha256": runner.sha256_file(Path(runner.__file__).resolve()),
            "expected_camera_eligible_rows": 80,
            "expected_camera_valid_rows": 58,
            "expected_camera_fit_rows": 48,
            "expected_camera_stopping_rows": 10,
            "expected_max_gradient_updates": 75,
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            launch_config_path = Path(temp_dir) / "launch-config.json"
            launch_config_path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["v7_accuracy_runner", "--config", str(launch_config_path)]):
                with self.assertRaisesRegex(RuntimeError, "requires --launch-camera-adaptation"):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
