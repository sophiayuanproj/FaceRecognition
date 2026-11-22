"""Regression tests for the classifier-only v8 H5 export repair."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from v8_classifier_only_export import (
    EXPECTED_ARRAYS,
    EXPECTED_PARAMETERS,
    ORIGINAL_EXPORT,
    PARENT_RESULT,
    PARENT_WEIGHTS,
    assert_classifier_contract,
    build_classifier_only,
    count_h5_weight_datasets,
    sha256_file,
    weight_array_hash,
)


class ClassifierOnlyExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
        predictions_path = (ORIGINAL_EXPORT.parents[0] / "private/PREDICTIONS.jsonl")
        predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cls.selected_weight_hash = predictions[0]["model_weight_sha256"]
        cls.original_sha256 = sha256_file(ORIGINAL_EXPORT)
        cls.model = build_classifier_only()
        cls.selected_evidence = assert_classifier_contract(cls.model, cls.selected_weight_hash)
        cls.parent_result = parent

    def test_classifier_parameter_and_array_counts(self):
        weights = self.model.get_weights()
        self.assertEqual(int(self.model.count_params()), EXPECTED_PARAMETERS)
        self.assertEqual(len(weights), EXPECTED_ARRAYS)
        self.assertEqual(sum(int(np.asarray(weight).size) for weight in weights), EXPECTED_PARAMETERS)

    def test_selected_weights_match_parent_evaluation_digest(self):
        self.assertEqual(weight_array_hash(self.model.get_weights()), self.selected_weight_hash)
        self.assertEqual(self.selected_evidence["selected_weight_array_sha256"], self.selected_weight_hash)
        self.assertEqual(self.parent_result["model"]["parameter_count"], EXPECTED_PARAMETERS)

    def test_h5_roundtrip_has_exact_selected_arrays_and_no_auxiliary_model(self):
        selected = [np.array(weight, copy=True) for weight in self.model.get_weights()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier_only_model.h5"
            self.model.save(str(path), include_optimizer=False)
            reloaded = tf.keras.models.load_model(str(path), compile=False)
            self.assertEqual(int(reloaded.count_params()), EXPECTED_PARAMETERS)
            self.assertEqual(len(reloaded.get_weights()), EXPECTED_ARRAYS)
            self.assertEqual(weight_array_hash(reloaded.get_weights()), self.selected_weight_hash)
            self.assertTrue(all(np.array_equal(before, after) for before, after in
                                zip(selected, reloaded.get_weights())))
            inventory = count_h5_weight_datasets(path)
            self.assertEqual(inventory["dataset_arrays"], EXPECTED_ARRAYS)
            self.assertEqual(inventory["parameter_elements"], EXPECTED_PARAMETERS)
            self.assertEqual(inventory["auxiliary_embedding_groups"], [])
            self.assertFalse(hasattr(reloaded, "_d_embedding_model"))

    def test_original_export_is_preserved_and_exhibits_auxiliary_weights(self):
        inventory = count_h5_weight_datasets(ORIGINAL_EXPORT)
        self.assertEqual(inventory["auxiliary_embedding_groups"],
                         ["v7_v7_lfw_source_mnv1_a010_ei_scout57101_embedding"])
        self.assertGreater(inventory["dataset_arrays"], EXPECTED_ARRAYS)
        self.assertGreater(inventory["parameter_elements"], EXPECTED_PARAMETERS)
        self.assertEqual(sha256_file(ORIGINAL_EXPORT), self.original_sha256)


if __name__ == "__main__":
    unittest.main()
