"""Audit every learned root export and the fixed stock model contract."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import tensorflow as tf


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_model(path: Path) -> dict:
    it = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    def tensor(d):
        return {'name': d['name'], 'shape': d['shape'].tolist(), 'dtype': np.dtype(d['dtype']).name,
                'quantization': {'scale': float(d['quantization'][0]), 'zero_point': int(d['quantization'][1])}}
    op_details = []
    for d in it._get_ops_details():
        op_details.append({'operator': d['op_name'], 'inputs': d['inputs'].tolist(),
                           'outputs': d['outputs'].tolist()})
    tensors = []
    for d in it.get_tensor_details():
        qp = d['quantization_parameters']
        tensors.append({'index': int(d['index']), 'name': d['name'],
          'shape': d['shape'].tolist(), 'dtype': np.dtype(d['dtype']).name,
          'quantization': {'scales': [float(v) for v in qp['scales']],
                           'zero_points': [int(v) for v in qp['zero_points']],
                           'quantized_dimension': int(qp['quantized_dimension'])}})
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'input': tensor(it.get_input_details()[0]), 'output': tensor(it.get_output_details()[0]),
            'operators': sorted({d['operator'] for d in op_details}),
            'operator_details': op_details, 'tensor_quantization_details': tensors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--stock-model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows=[]
    for arm in ('mobilenetv2_alpha_0p1', 'cheap_sepconv_16'):
        for root in sorted((a.model_dir / arm).glob('seed-*')):
            labels=(root/'labels.txt').read_text().splitlines()
            summary=json.loads((root/'root_summary.json').read_text())
            row={'arm':arm,'seed':root.name,'labels':labels,'training':summary,
                 'float':inspect_model(root/'detector_float.tflite'),
                 'full_int8':inspect_model(root/'detector_full_int8.tflite'),
                 'training_weights':{'path':str(root/'best_training_weights.weights.h5'),
                   'sha256':sha(root/'best_training_weights.weights.h5'),
                   'bytes':(root/'best_training_weights.weights.h5').stat().st_size,
                   'deployable':False}}
            if labels != ['background','face']:
                raise ValueError(f'Unexpected labels in {root}: {labels}')
            for kind in ('float','full_int8'):
                m=row[kind]
                if m['input']['shape'] != [1,96,96,1] or m['output']['shape'] != [1,12,12,2]:
                    raise ValueError(f'Unexpected tensors in {m["path"]}')
                if kind == 'full_int8' and (m['input']['dtype'] != 'int8' or m['output']['dtype'] != 'int8'):
                    raise ValueError(f'INT8 export is not signed int8: {m["path"]}')
            rows.append(row)
    counts={arm:sum(r['arm']==arm for r in rows) for arm in ('mobilenetv2_alpha_0p1','cheap_sepconv_16')}
    if counts != {'mobilenetv2_alpha_0p1':10,'cheap_sepconv_16':10}:
        raise ValueError(f'Expected ten complete roots per arm, got {counts}')
    out={'schema':'job-c-root-export-audit-1','arms':counts,'class_semantics':{'channel_0':'background','channel_1':'face'},
         'stock_reference':inspect_model(a.stock_model),'roots':rows,
         'full_int8_policy':'All operators must be integer-compatible; tensor input/output signed int8.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'roots':counts,'stock_bytes':out['stock_reference']['bytes'],'output':str(a.out)},indent=2))
if __name__=='__main__': main()
