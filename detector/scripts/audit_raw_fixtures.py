"""Verify raw host/board parity fixture bytes and their TensorFlow Lite outputs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    manifest=json.loads(a.manifest.read_text())
    model_path=Path(manifest['model_path'])
    if sha(model_path.read_bytes()) != manifest['model_sha256']:
        raise ValueError('Frozen challenger artifact hash mismatch')
    it=tf.lite.Interpreter(model_path=str(model_path),num_threads=1); it.allocate_tensors()
    inp,out=it.get_input_details()[0],it.get_output_details()[0]
    if inp['dtype'] != np.int8 or out['dtype'] != np.int8:
        raise ValueError('Fixture model is not full signed INT8')
    results=[]
    for fixture in manifest['fixtures']:
        folder=Path(fixture['fixture_directory']); expected=fixture['exact_raw_files']
        checks={}
        for name,meta in expected.items():
            if meta is None:
                if (folder/name).exists(): raise ValueError(f'Unexpected raw identity file in {folder}')
                checks[name]={'present':False}
                continue
            data=(folder/name).read_bytes()
            if len(data)!=meta['bytes'] or sha(data)!=meta['sha256']:
                raise ValueError(f'Raw bytes/hash mismatch: {folder/name}')
            checks[name]={'present':True,'bytes':len(data),'sha256':sha(data)}
        frame=(folder/'camera_window_gray240.u8').read_bytes()
        detector=(folder/'detector_input_gray96.u8').read_bytes()
        quantized=(folder/'detector_input_int8.i8').read_bytes()
        raw_output=(folder/'detector_output_int8.i8').read_bytes()
        if len(frame)!=240*240 or len(detector)!=96*96 or len(quantized)!=96*96 or len(raw_output)!=12*12*2:
            raise ValueError(f'Unexpected tensor payload length: {folder}')
        if bytes((v-128)&255 for v in detector)!=quantized:
            raise ValueError(f'INT8 input not exact affine gray8 quantization: {folder}')
        it.set_tensor(inp['index'],np.frombuffer(quantized,dtype=np.int8).reshape(1,96,96,1)); it.invoke()
        actual=it.get_tensor(out['index']).tobytes()
        if actual != raw_output:
            raise ValueError(f'Host TFLite raw output differs from fixture: {folder}')
        for pgm,u8 in (('camera_window_gray240.pgm','camera_window_gray240.u8'),
                       ('detector_input_gray96.pgm','detector_input_gray96.u8'),
                       ('identity_input_gray96.pgm','identity_input_gray96.u8')):
            if (folder/pgm).exists():
                with Image.open(folder/pgm) as im: decoded=im.convert('L').tobytes()
                if decoded != (folder/u8).read_bytes():
                    raise ValueError(f'Lossless PGM preview mismatch: {folder/pgm}')
        fixture_json=json.loads((folder/'fixture.json').read_text())
        if fixture_json['source_image_sha256'] != fixture['source_image_sha256']:
            raise ValueError(f'Source hash mismatch: {folder}')
        results.append({'fixture_id':fixture['fixture_id'],'role':fixture['source_role'],
          'status':fixture['expected_crop_geometry']['status'],'source_image_sha256':fixture['source_image_sha256'],
          'files':checks,'tflite_raw_output_match':True,'int8_input_match':True,'pgm_preview_lossless':True})
    if len(results)!=13 or manifest['transform_fault_block_count']!=10:
        raise ValueError('Expected 10 seeded fault blocks plus 3 PASS no-face fixtures')
    result={'schema':'job-c-raw-fixture-audit-1','manifest_sha256':sha(a.manifest.read_bytes()),
      'model_sha256':manifest['model_sha256'],'fixture_count':len(results),'transform_fault_block_count':10,
      'real_no_face_fixture_count':3,'raw_bytes_policy':'exact u8/i8 inference bytes checked; PGM previews decoded back to exact raw bytes; no JPEG inputs',
      'fixtures':results}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fixture_count':len(results),'fault_blocks':10,'output':str(a.out)},indent=2))
if __name__=='__main__': main()
