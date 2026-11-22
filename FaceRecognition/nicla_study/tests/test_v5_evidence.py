import hashlib,json,random,tempfile,unittest
from pathlib import Path
from nicla_study.v5_evidence import (
    binding,
    derive_metrics_from_predictions,
    qualify_candidate,
    strict_load,
    verify_bindings,
    verify_raw_blocks,
)

class EvidenceTests(unittest.TestCase):
    def test_metrics_come_from_rows_and_reject_all_remains_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[
                {'record_id':'ucal-1','source_identity':'other-cal-1','role':'unknown_calibration','true_label':'OTHER','detected':True,'probabilities':[.1,.1,.1,.1,.1,.5],'predicted_label':'OTHER','accepted':False,'response_ms':15.0},
                {'record_id':'known-1','source_identity':'known-1','role':'development','true_label':'K01','detected':True,'probabilities':[.9,.02,.02,.02,.02,.02],'predicted_label':'K01','accepted':True,'response_ms':20.0},
                {'record_id':'unknown-1','source_identity':'unknown-1','role':'unknown_development','true_label':'OTHER','detected':True,'probabilities':[.95,.01,.01,.01,.01,.01],'predicted_label':'K01','accepted':True,'response_ms':25.0},
            ]
            pred=root/'predictions.jsonl'
            pred.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (root/'handwritten_metrics.json').write_text(json.dumps({'known_correct_acceptance':1.0,'unknown_group_upper95':0.0,'response_p95_ms':1.0}))
            result=derive_metrics_from_predictions(root,'predictions.jsonl')
            self.assertEqual(result['known_correct_acceptance'],1.0)
            self.assertEqual(result['known_coverage'],1.0)
            self.assertEqual(result['unknown_false_groups'],1)
            self.assertEqual(result['response_p95_ms'],25.0)
            self.assertGreater(result['unknown_group_upper95'],0.99)

    def test_stored_accept_decision_must_match_row_derived_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[
                {'record_id':'ucal','source_identity':'u-cal','role':'unknown_calibration','true_label':'OTHER','detected':True,'probabilities':[.1,.1,.1,.1,.1,.5],'predicted_label':'OTHER','accepted':False},
                {'record_id':'known','source_identity':'k','role':'development','true_label':'K01','detected':True,'probabilities':[.9,.02,.02,.02,.02,.02],'predicted_label':'K01','accepted':False},
                {'record_id':'unknown','source_identity':'u','role':'unknown_development','true_label':'OTHER','detected':True,'probabilities':[.2,.16,.16,.16,.16,.16],'predicted_label':'K01','accepted':False},
            ]
            path=root/'predictions.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'disagrees with row-derived decision'):
                derive_metrics_from_predictions(root,'predictions.jsonl')

    def test_ten_actual_fault_blocks(self):
        for seed in range(53001,53011):
            with self.subTest(seed=seed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);p=root/'model.tflite';p.write_bytes(random.Random(seed).randbytes(64))
                receipt=[binding(root,p,'model')];self.assertEqual(verify_bindings(root,receipt),1)
                p.write_bytes(p.read_bytes()+b'changed')
                with self.assertRaises(ValueError):verify_bindings(root,receipt)
                self.assertFalse(qualify_candidate(root,{'status':'complete_verified','bindings':receipt})['qualified'])
                for value in ['NaN','Infinity','-Infinity']:
                    (root/'bad.json').write_text('{"metric":'+value+'}')
                    with self.assertRaises(ValueError):strict_load(root/'bad.json')
                (root/'blocks.json').write_text(json.dumps([{'block':0,'status':'executed','stdout_sha256':'0'*64}]*10))
                with self.assertRaises(ValueError):verify_raw_blocks(root,'blocks.json')
                with self.assertRaises(ValueError):verify_bindings(root,[{'path':'../escape','sha256':'0'*64,'bytes':1}])

if __name__=='__main__':unittest.main()
