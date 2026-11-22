"""Build and verify the model-bound RGB565 -> RGB48 -> int8 parity probe."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .v5_native_rgb import native_rgb


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
MODEL = STUDY / "runs/operator_fused_encoder48/int8.tflite"
OUT = STUDY / "reports/M5/rgb48_encoder_parity"
ROIS = [
    (0, 0, 121, 97),
    (16, 8, 121, 97),
    (0, 16, 121, 97),
    (16, 0, 121, 97),
    (8, 6, 103, 89),
    (34, 18, 103, 89),
    (2, 2, 133, 109),
    (20, 0, 117, 113),
    (0, 0, 137, 113),
    (0, 12, 137, 101),
]


def sha256(payload):
    if isinstance(payload, Path):
        payload = payload.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def source_pattern(seed):
    yy, xx = np.indices((113, 137), dtype=np.uint16)
    r = (xx + seed) % 32
    g = (2 * yy + seed) % 64
    b = (xx * yy + seed) % 32
    return ((r << 11) | (g << 5) | b).astype(np.uint16)


def interpreter(payload):
    return tf.lite.Interpreter(
        model_content=payload,
        num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )


def make_contract():
    payload = MODEL.read_bytes()
    it = interpreter(payload)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    if list(inp["shape"]) != [1, 48, 48, 3] or inp["dtype"] != np.int8:
        raise ValueError("Unexpected 48px encoder input contract")
    if list(out["shape"]) != [1, 128] or out["dtype"] != np.int8:
        raise ValueError("Unexpected 128-value signed encoder output")
    scale, zero = inp["quantization"]
    if not scale or scale <= 0:
        raise ValueError("Invalid quantization scale")
    pixels = np.arange(256, dtype=np.float64)
    real = (pixels - 127.5) / 128.0
    lut = np.clip(np.rint(real / float(scale)) + int(zero), -128, 127).astype(np.int8)
    return payload, it, inp, out, lut


def device_stage1(model_hash):
    source = '''import csi,ml,gc,json,hashlib,binascii,umalloc
from ml.postprocessing.edgeimpulse import Fomo
import time,random
MODEL='/rom/operator_fused_encoder48.tflite'
EXPECTED_SHA='@@MODEL_SHA@@'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while True:
   b=f.read(2048)
   if not b:break
   h.update(b)
 gc.collect();return binascii.hexlify(h.digest()).decode()
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.snapshot(time=600);frame=cam.snapshot()
print(json.dumps({'kind':'camera','width':frame.width(),'height':frame.height(),'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}))
gc.collect();det=ml.Model('/rom/fomo_face_detection.tflite',postprocess=Fomo(threshold=.5));t=time.ticks_us();hits=det.predict([frame]);print(json.dumps({'kind':'detector','elapsed_us':time.ticks_diff(time.ticks_us(),t),'result':str(hits)}));det=None;hits=None;gc.collect()
paths=[MODEL];random.seed(54101)
for path in paths:
 R48MODEL_SHA=sha(path);gc.collect();t=time.ticks_us()
 if R48MODEL_SHA!=EXPECTED_SHA:raise ValueError('ROMFS model SHA mismatch '+R48MODEL_SHA)
 try:model=ml.Model(path)
 except Exception as e:
  print('R48ALLOC_FAIL '+json.dumps({'error':repr(e),'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}));break
 print('R48STAGE1 '+json.dumps({'ready':True,'model_sha256':R48MODEL_SHA,'camera_frame':[frame.width(),frame.height()],'load_us':time.ticks_diff(time.ticks_us(),t),'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}))
'''
    return source.replace("@@MODEL_SHA@@", model_hash)


def device_stage2(model_hash, lut):
    table = ",".join(str(int(x) & 255) for x in lut.tolist())
    rois = repr(tuple(ROIS))
    source = '''import image,gc,binascii,json,hashlib,time
from ml.preprocessing import Normalization
LUT=bytes([@@LUT@@])
ROIS=@@ROIS@@
EXPECTED_SHA='@@MODEL_SHA@@'
def flat(value):
 out=bytearray();stack=[value]
 while stack:
  item=stack.pop()
  if hasattr(item,'shape') and len(item.shape)>0:
   for i in range(item.shape[0]-1,-1,-1):stack.append(item[i])
  else:out.append(int(item)&255)
 return out
if R48MODEL_SHA!=EXPECTED_SHA:raise ValueError('Stage1 model identity mismatch')
print('R48META '+json.dumps({'model_sha256':R48MODEL_SHA,'firmware':'OpenMV5.0.1','roi_count':len(ROIS),'input_dtype':'int8','input_shape':[1,48,48,3],'lut_sha256':binascii.hexlify(hashlib.sha256(LUT).digest()).decode(),'camera_frame':[frame.width(),frame.height()],'gc_free':gc.mem_free()}))
for seed,roi in enumerate(ROIS):
 frame=image.Image(137,113,image.RGB565);raw=frame.bytearray()
 for y in range(113):
  for x in range(137):
   pixel=(((x+seed)%32)<<11)|(((2*y+seed)%64)<<5)|((x*y+seed)%32);j=(y*137+x)*2;raw[j]=pixel&255;raw[j+1]=pixel>>8
 rgb=bytearray(48*48*3);bound=Normalization(roi=roi)(frame);captured={}
 def fill(buf,shape,dtype):
  if shape!=(1,48,48,3) or dtype!=ord('b'):raise ValueError('Unexpected model tensor contract')
  t0=time.ticks_us()
  bound(rgb,shape,ord('B'))
  t1=time.ticks_us()
  for j in range(len(buf)):buf[j]=LUT[rgb[j]]
  t2=time.ticks_us()
  captured['input']=bytes(buf)
  captured['native_transform_us']=time.ticks_diff(t1,t0)
  captured['lut_apply_us']=time.ticks_diff(t2,t1)
 def output_cb(m,inputs,outputs):
  captured['output']=bytes(flat(outputs[0]));return outputs
 t0=time.ticks_us();model.predict([fill],callback=output_cb);captured['predict_total_us']=time.ticks_diff(time.ticks_us(),t0)
 if len(captured.get('input',b''))!=6912 or len(captured.get('output',b''))!=128:raise ValueError('Missing captured tensors')
 encoded=binascii.b2a_base64(captured['input']).decode().strip()
 for start in range(0,len(encoded),1024):print('R48I '+str(seed)+' '+str(start//1024)+' '+str((len(encoded)+1023)//1024)+' '+encoded[start:start+1024])
 print('R48O '+str(seed)+' '+binascii.b2a_base64(captured['output']).decode().strip())
 print('R48T '+str(seed)+' '+json.dumps({'native_transform_us':captured['native_transform_us'],'lut_apply_us':captured['lut_apply_us'],'predict_total_us':captured['predict_total_us']}))
 print('R48END '+str(seed)+' '+str(roi))
 frame=None;raw=None;rgb=None;bound=None;captured=None;gc.collect()
print('R48COMPLETE '+str(len(ROIS)))
'''
    return (source.replace("@@LUT@@", table)
                  .replace("@@ROIS@@", rois)
                  .replace("@@MODEL_SHA@@", model_hash))


def prepare():
    payload, _, inp, out, lut = make_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    model_hash = sha256(payload)
    stage1 = device_stage1(model_hash).encode()
    stage2 = device_stage2(model_hash, lut).encode()
    legacy = OUT / "DEVICE_PROGRAM.py"
    if legacy.exists():
        archived = OUT / "DEVICE_PROGRAM_COMBINED_MINIMIZED_AS_RUN.py"
        if not archived.exists():
            legacy.replace(archived)
    contract = {
        "schema": "v5-rgb48-encoder-input-contract-1",
        "model_path": MODEL.relative_to(STUDY).as_posix(),
        "model_sha256": sha256(payload),
        "model_bytes": len(payload),
        "input_shape": [int(x) for x in inp["shape"]],
        "input_dtype": str(inp["dtype"]),
        "input_scale": float(inp["quantization"][0]),
        "input_zero_point": int(inp["quantization"][1]),
        "output_shape": [int(x) for x in out["shape"]],
        "output_dtype": str(out["dtype"]),
        "output_scale": float(out["quantization"][0]),
        "output_zero_point": int(out["quantization"][1]),
        "normalization_before_quantization": "(RGB888-127.5)/128",
        "host_quantization": "TFLite affine quantization with np.rint ties-to-even then int8 clamp",
        "lut_sha256": sha256(lut.tobytes()),
        "lut_signed_values": [int(x) for x in lut],
        "roi_fixtures": [list(v) for v in ROIS],
        "synthetic_fixture_size": [137, 113],
        "device_stage1_sha256": sha256(stage1),
        "device_stage2_sha256": sha256(stage2),
        "host_backend": "TF2.10.1 BUILTIN_REF",
        "board_qualified": False,
    }
    (OUT / "DEVICE_STAGE1.py").write_bytes(stage1)
    (OUT / "DEVICE_STAGE2.py").write_bytes(stage2)
    (OUT / "CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in contract.items() if k != "lut_signed_values"}, sort_keys=True))


def verify(transcript=None):
    transcript = Path(transcript) if transcript else OUT / "block_00.stdout.bin"
    payload, it, inp, out, lut = make_contract()
    contract = json.loads((OUT / "CONTRACT.json").read_text())
    if contract["model_sha256"] != sha256(payload) or contract["lut_sha256"] != sha256(lut.tobytes()):
        raise ValueError("Prepared contract is stale")
    if contract["device_stage1_sha256"] != sha256((OUT / "DEVICE_STAGE1.py").read_bytes()):
        raise ValueError("Stage 1 probe source changed")
    if contract["device_stage2_sha256"] != sha256((OUT / "DEVICE_STAGE2.py").read_bytes()):
        raise ValueError("Stage 2 probe source changed")
    text = transcript.read_bytes().decode("utf-8", errors="replace")
    input_parts = {}
    outputs = {}
    timings = {}
    complete = False
    device_model_hash = None
    for line in text.splitlines():
        if line.startswith("R48META "):
            device_model_hash = json.loads(line[8:]).get("model_sha256")
        elif line.startswith("R48I "):
            _, seed, part, parts, value = line.split(" ", 4)
            input_parts.setdefault(int(seed), {})[int(part)] = (int(parts), value)
        elif line.startswith("R48O "):
            _, seed, value = line.split(" ", 2)
            outputs[int(seed)] = base64.b64decode(value, validate=True)
        elif line.startswith("R48T "):
            _, seed, value = line.split(" ", 2)
            timings[int(seed)] = json.loads(value)
        elif line.startswith("R48COMPLETE "):
            complete = line.strip() == "R48COMPLETE 10"
    if device_model_hash != sha256(payload):
        raise ValueError("Board transcript did not verify the expected ROMFS model SHA")
    if not complete or set(outputs) != set(range(10)) or set(input_parts) != set(range(10)) or set(timings) != set(range(10)):
        raise ValueError("Board transcript is missing one or more seeded fixtures")

    input_details = it.get_input_details()[0]
    output_details = it.get_output_details()[0]
    board_input_rows = []
    board_output_rows = []
    host_output_rows = []
    results = []
    input_payloads = []
    for seed, roi in enumerate(ROIS):
        part_map = input_parts[seed]
        counts = {count for count, _ in part_map.values()}
        if len(counts) != 1:
            raise ValueError("Inconsistent base64 chunk counts")
        part_count = counts.pop()
        if set(part_map) != set(range(part_count)):
            raise ValueError("Missing board input chunk")
        device_input = base64.b64decode("".join(part_map[i][1] for i in range(part_count)), validate=True)
        device_output = outputs[seed]
        if len(device_input) != 48 * 48 * 3 or len(device_output) != 128:
            raise ValueError("Unexpected captured board tensor byte count")
        host_rgb = native_rgb(source_pattern(seed), roi=roi, width=48, height=48)
        expected_input = lut[host_rgb].tobytes()
        input_equal = device_input == expected_input
        device_input_np = np.frombuffer(device_input, dtype=np.int8).reshape(1, 48, 48, 3)
        it.set_tensor(input_details["index"], device_input_np)
        it.invoke()
        host_output = it.get_tensor(output_details["index"]).astype(np.int8, copy=False).reshape(-1)
        output_np = np.frombuffer(device_output, dtype=np.int8)
        delta = np.abs(output_np.astype(np.int16) - host_output.astype(np.int16))
        results.append({
            "seed": seed,
            "roi": list(roi),
            "input_exact": input_equal,
            "expected_input_sha256": sha256(expected_input),
            "board_input_sha256": sha256(device_input),
            "output_exact_on_board_input": bool(np.array_equal(output_np, host_output)),
            "output_lsb_differences": int(np.count_nonzero(delta)),
            "output_max_abs_lsb": int(delta.max()),
            "timing_us": timings[seed],
        })
        input_payloads.append(device_input)
        board_input_rows.append(np.frombuffer(device_input, dtype=np.int8).reshape(48, 48, 3))
        board_output_rows.append(output_np.copy())
        host_output_rows.append(host_output.copy())

    status = "verified_exact" if all(r["input_exact"] and r["output_exact_on_board_input"] for r in results) else "mismatch"
    np.save(OUT / "BOARD_INPUTS_I8.npy", np.stack(board_input_rows))
    np.save(OUT / "BOARD_OUTPUTS_I8.npy", np.stack(board_output_rows))
    np.save(OUT / "HOST_OUTPUTS_I8.npy", np.stack(host_output_rows))
    report = {
        "schema": "v5-rgb48-encoder-parity-1",
        "status": status,
        "model_sha256": sha256(payload),
        "device_stage1_sha256": contract["device_stage1_sha256"],
        "device_stage2_sha256": contract["device_stage2_sha256"],
        "transcript_sha256": sha256(transcript),
        "fixture_count": len(results),
        "independent_unit": "seeded synthetic pixel fixture; one board software-reset execution",
        "host_backend": "TF2.10.1 BUILTIN_REF",
        "input_exact_count": sum(r["input_exact"] for r in results),
        "output_exact_count": sum(r["output_exact_on_board_input"] for r in results),
        "median_native_transform_us": float(np.median([r["timing_us"]["native_transform_us"] for r in results])),
        "median_python_lut_apply_us": float(np.median([r["timing_us"]["lut_apply_us"] for r in results])),
        "median_predict_total_us": float(np.median([r["timing_us"]["predict_total_us"] for r in results])),
        "results": results,
        "scope": "48px image ROI, RGB565 to RGB888, exact model input quantization and raw signed output; synthetic fixtures only",
        "identity_qualified": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))
    if status != "verified_exact":
        raise RuntimeError("48px input/output parity failed; preserve transcript and investigate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--transcript", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        verify(args.transcript)


if __name__ == "__main__":
    main()
