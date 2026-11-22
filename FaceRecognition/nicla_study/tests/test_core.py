from __future__ import annotations
import unittest

from nicla_study.contracts import validate_model_output
from nicla_study.macs import layer_macs
from nicla_study.metrics import auc_eer, metric_selector_score, verification_metrics
from nicla_study.roles import assert_role_access, validate_manifest
from nicla_study.seeds import derive_seed, validate_roots


class CoreContractsTest(unittest.TestCase):
    def test_corrected_auc_order_ties_and_zero_selector(self):
        good = auc_eer([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
        self.assertEqual(good["roc_auc"], 1.0)
        self.assertEqual(good["eer"], 0.0)
        tied = auc_eer([0.1, 0.1, 0.9, 0.9], [0, 1, 0, 1])
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertEqual(metric_selector_score(0.0), 0.0)
        self.assertEqual(auc_eer([0.1, 0.2], [1, 1])["roc_auc"], None)
        self.assertAlmostEqual(verification_metrics([0.1, 0.9], [0, 1])["roc_auc"], 1.0)
        with self.assertRaises(ValueError):
            auc_eer([float("nan"), 0.1], [0, 1])

    def test_role_gate_rejects_final_for_training_and_quantization(self):
        row = {"record_id":"x","identity_id":"person-a","role":"final","sha256":"a"*64,
               "relative_path":"public/final/a.png"}
        with self.assertRaises(PermissionError):
            assert_role_access([row], "train")
        with self.assertRaises(PermissionError):
            assert_role_access([row], "quantize")

    def test_manifest_rejects_cross_partition_unknown_identity(self):
        rows=[]
        for i,role in enumerate(("unknown_calibration","unknown_development")):
            rows.append({"record_id":str(i),"identity_id":"unknown-1","role":role,
                         "sha256":("a" if i==0 else "b")*64,"relative_path":"x/%d.png"%i})
        result=validate_manifest(rows)
        self.assertEqual(result["status"],"invalid")
        self.assertTrue(result["errors"])

    def test_typed_classifier_and_embedding_contracts(self):
        output={"model_type":"classifier_logits","values":[0.1]*6,
                "class_labels":["A","B","C","D","E","OTHER"]}
        validate_model_output(output,"classifier_logits",6,output["class_labels"])
        with self.assertRaises(TypeError):
            validate_model_output(output,"embedding",6)
        with self.assertRaises(ValueError):
            validate_model_output({"model_type":"embedding","values":[0.0]*5,"class_labels":None},
                                  "embedding",64)

    def test_four_hand_accounting_formulas(self):
        self.assertEqual(layer_macs("Conv2D",input_shape=(1,8,8,3),output_shape=(1,6,6,4),
                                    kernel_size=3,filters=4),3888)
        self.assertEqual(layer_macs("DepthwiseConv2D",input_shape=(1,8,8,3),output_shape=(1,6,6,6),
                                    kernel_size=3,depth_multiplier=2),1944)
        self.assertEqual(layer_macs("SeparableConv2D",input_shape=(1,8,8,3),output_shape=(1,6,6,4),
                                    kernel_size=3,filters=4,depth_multiplier=2),2808)
        self.assertEqual(layer_macs("Dense",input_shape=(1,8),output_shape=(1,5),units=5),40)

    def test_seed_streams_are_stable_and_distinct(self):
        self.assertEqual(derive_seed("v4",20261101,"augmentation"),derive_seed("v4",20261101,"augmentation"))
        self.assertNotEqual(derive_seed("v4",20261101,"augmentation"),derive_seed("v4",20261101,"batch_order"))
        self.assertEqual(len(validate_roots(range(20261101,20261111))),10)
        with self.assertRaises(ValueError):
            validate_roots([1,1],minimum=2)


if __name__ == "__main__":
    unittest.main()
