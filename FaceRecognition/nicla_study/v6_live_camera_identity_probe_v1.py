"""Prepare an uninstalled, one-shot live camera-to-decision research probe.

The generated MicroPython program uses the ROM FOMO and fused48 encoder plus
the v2 one-root linear head. It is run only through the bounded raw-REPL board
harness; it does not install files or change startup. The head is not useful or
qualified and the run is diagnostic, not a deployment candidate.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
FIRMWARE = ROOT / "FaceRecognition/firmware"
OUT = STUDY / "runs/camera_identity_prototype_v1"
HEAD = STUDY / "private/camera_corrected_head_v2/linear_head.npz"
HEAD_RESULT = STUDY / "runs/camera_corrected_head_v2/RESULT.json"
CONTRACT = STUDY / "reports/M5/rgb48_encoder_parity/CONTRACT.json"
INPUTS = STUDY / "runs/camera_corrected_input_v2/INPUTS.json"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare() -> dict:
    program_path = OUT / "DEVICE_PROGRAM.py"
    manifest_path = OUT / "DEVICE_PROGRAM.json"
    if program_path.exists() or manifest_path.exists():
        raise FileExistsError("preserving existing camera identity probe output")

    result = json.loads(HEAD_RESULT.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    weights = np.load(HEAD)
    kernel = np.asarray(weights["kernel"], dtype="<f4")
    bias = np.asarray(weights["bias"], dtype="<f4")
    if kernel.shape != (128, 6) or bias.shape != (6,):
        raise ValueError("unexpected v2 camera head dimensions")
    if result["model"]["weights_sha256"] != sha(HEAD.read_bytes()):
        raise ValueError("camera head weights do not match the frozen v2 result")
    if inputs["model_sha256"] != contract["model_sha256"]:
        raise ValueError("input and encoder contract hashes differ")

    identity_path = FIRMWARE / "identity_contracts.py"
    identity_text = identity_path.read_text(encoding="utf-8")
    identity_ast = ast.parse(identity_text)
    geometry_names = {"fomo_center_camera_px", "centered_roi_clamped"}
    geometry_nodes = [node for node in identity_ast.body
                      if isinstance(node, ast.FunctionDef) and node.name in geometry_names]
    if {node.name for node in geometry_nodes} != geometry_names:
        raise ValueError("shared camera-space geometry functions are missing")
    geometry_source = ("import math\n\n" + "\n\n".join(
        ast.get_source_segment(identity_text, node) for node in geometry_nodes
    ) + "\n").encode("utf-8")

    lut = bytes([int(value) & 255 for value in contract["lut_signed_values"]])
    constants = {
        "@@GEOMETRY@@": base64.b64encode(geometry_source).decode("ascii"),
        "@@LUT@@": base64.b64encode(lut).decode("ascii"),
        "@@KERNEL@@": base64.b64encode(kernel.tobytes()).decode("ascii"),
        "@@BIAS@@": base64.b64encode(bias.tobytes()).decode("ascii"),
        "@@MODEL_SHA@@": contract["model_sha256"],
        "@@HEAD_SHA@@": sha(HEAD.read_bytes()),
        "@@INPUT_SCALE@@": repr(float(inputs["input_quantization"][0])),
        "@@INPUT_ZERO@@": str(int(inputs["input_quantization"][1])),
        "@@OUTPUT_SCALE@@": repr(float(inputs["output_quantization"][0])),
        "@@OUTPUT_ZERO@@": str(int(inputs["output_quantization"][1])),
        "@@THRESHOLD@@": repr(float(result["evaluation"]["threshold_binary32"])),
    }
    program = r'''import csi,ml,gc,json,binascii,hashlib,time,ubinascii,math,struct
from ml.preprocessing import Normalization
from ml.postprocessing.edgeimpulse import Fomo
exec(ubinascii.a2b_base64("@@GEOMETRY@@").decode())
LUT=ubinascii.a2b_base64("@@LUT@@")
KERNEL=struct.unpack("<768f",ubinascii.a2b_base64("@@KERNEL@@"))
BIAS=struct.unpack("<6f",ubinascii.a2b_base64("@@BIAS@@"))
ENCODER_SHA="@@MODEL_SHA@@"
HEAD_SHA="@@HEAD_SHA@@"
INPUT_SCALE=@@INPUT_SCALE@@
INPUT_ZERO=@@INPUT_ZERO@@
OUTPUT_SCALE=@@OUTPUT_SCALE@@
OUTPUT_ZERO=@@OUTPUT_ZERO@@
THRESHOLD=@@THRESHOLD@@
CLASSES=("K01","K02","K03","K04","K05","OTHER")
def sha_bytes(value):return binascii.hexlify(hashlib.sha256(value).digest()).decode()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,"rb") as stream:
  while True:
   block=stream.read(2048)
   if not block:break
   h.update(block)
 return binascii.hexlify(h.digest()).decode()
def flatten(value):
 out=bytearray();stack=[value]
 while stack:
  item=stack.pop()
  if hasattr(item,"shape") and len(item.shape)>0:
   for i in range(item.shape[0]-1,-1,-1):stack.append(item[i])
  else:out.append(int(item)&255)
 return out
started=time.ticks_us()
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.auto_exposure(True);cam.snapshot(time=600)
frame=cam.snapshot()
if frame is None or frame.width()!=240 or frame.height()!=240:raise ValueError("camera_frame_not_240x240")
detector_start=time.ticks_us();det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));fomo_load_us=time.ticks_diff(time.ticks_us(),detector_start)
detector_start=time.ticks_us();detections=det.predict([frame]);fomo_predict_us=time.ticks_diff(time.ticks_us(),detector_start)
raw_faces=detections[1] if len(detections)>1 else []
accepted=[item for item in raw_faces if float(item[1])>.5]
row={"schema":"nicla-live-camera-identity-probe-v1","status":"completed","camera_frame_geometry":[240,240],"camera_id":cam.cid(),"fomo_raw_cluster_count":len(raw_faces),"fomo_accepted_cluster_count":len(accepted),"fomo_accepted_scores":[float(item[1]) for item in accepted],"fomo_coordinate_space":"camera_xywh_px","fomo_load_us":fomo_load_us,"fomo_predict_us":fomo_predict_us,"encoder_sha256":sha_file("/rom/operator_fused_encoder48.tflite"),"expected_encoder_sha256":ENCODER_SHA,"head_sha256":HEAD_SHA,"threshold_binary32":THRESHOLD,"score_compare":">= after float32 cast","future_final_used":False,"board_files_written":False,"startup_path_changed":False,"power_measured":False}
if row["encoder_sha256"]!=ENCODER_SHA:raise ValueError("fused48_encoder_hash_mismatch")
if len(accepted)==0:
 row["decision"]="NO_FACE";row["class_probabilities"]=None
elif len(accepted)>1:
 row["decision"]="UNCERTAIN";row["class_probabilities"]=None
else:
 rect=accepted[0][0];center=fomo_center_camera_px(rect,frame.width(),frame.height());roi=centered_roi_clamped(center,120,frame.width(),frame.height())
 row["fomo_rect"]=[float(v) for v in rect];row["center_xy_camera_px"]=list(center);row["roi_xywh_camera_px"]=list(roi)
 normalizer=Normalization(roi=roi)(frame);rgb=bytearray(48*48*3);normalizer(rgb,(1,48,48,3),ord("B"));tensor=bytearray(len(rgb))
 for i in range(len(rgb)):tensor[i]=LUT[rgb[i]]
 row["tensor_sha256"]=sha_bytes(tensor);row["tensor_bytes"]=len(tensor)
 normalizer=None;rgb=None;frame=None;detector=None;detections=None;raw_faces=None;accepted=None;det=None;cam=None;gc.collect()
 model=ml.Model("/rom/operator_fused_encoder48.tflite");row["encoder_arena_bytes"]=int(model.ram);encoded={}
 def feed(buffer,shape,dtype):
  if shape!=(1,48,48,3) or dtype!=ord("b"):raise ValueError("unexpected_encoder_input_contract")
  for j in range(len(buffer)):buffer[j]=tensor[j]
  encoded["input"]=bytes(buffer)
 def callback(m,inputs,outputs):encoded["output"]=bytes(flatten(outputs[0]));return outputs
 inference_start=time.ticks_us();model.predict([feed],callback=callback);row["encoder_predict_us"]=time.ticks_diff(time.ticks_us(),inference_start)
 if len(encoded.get("input",b""))!=6912 or len(encoded.get("output",b""))!=128:raise ValueError("missing_exact_encoder_io")
 if encoded["input"]!=bytes(tensor):raise ValueError("encoder_input_changed_inside_runtime")
 row["encoder_output_sha256"]=sha_bytes(encoded["output"]);row["encoder_output_bytes"]=len(encoded["output"])
 embedding_bytes=encoded["output"];model=None;encoded=None;gc.collect()
 signed=[]
 for value in embedding_bytes:signed.append(value-256 if value>=128 else value)
 vector=[(float(value)-OUTPUT_ZERO)*OUTPUT_SCALE for value in signed]
 length=math.sqrt(sum(value*value for value in vector))
 if not math.isfinite(length) or length<=0:raise ValueError("invalid_embedding_norm")
 vector=[value/length for value in vector]
 logits=[]
 for cls in range(6):
  total=float(BIAS[cls])
  for i in range(128):total+=vector[i]*KERNEL[i*6+cls]
  logits.append(total)
 top=max(logits);exps=[math.exp(v-top) for v in logits];den=sum(exps);probs=[v/den for v in exps]
 prediction=0
 for i in range(1,6):
  if probs[i]>probs[prediction]:prediction=i
 max_known=max(probs[:5]);score=struct.unpack("<f",struct.pack("<f",max_known))[0]
 row["class_probabilities"]=[float(v) for v in probs];row["predicted_class"]=CLASSES[prediction];row["max_known_probability_binary32"]=score
 if prediction==5:row["decision"]="OTHER"
 elif score>=THRESHOLD:row["decision"]=CLASSES[prediction]
 else:row["decision"]="UNCERTAIN"
 row["head_backend"]="MicroPython float; threshold comparison binary32"
 embedding_bytes=None;vector=None;logits=None;probs=None;gc.collect()
row["total_elapsed_us"]=time.ticks_diff(time.ticks_us(),started)
print(json.dumps(row))
tensor=None;gc.collect()
'''
    program = program.replace("@@GEOMETRY@@", constants["@@GEOMETRY@@"])
    for key, value in constants.items():
        if key != "@@GEOMETRY@@":
            program = program.replace(key, value)
    payload = program.encode("utf-8")
    OUT.mkdir(parents=True, exist_ok=False)
    program_path.write_bytes(payload)
    manifest = {
        "schema": "nicla-live-camera-identity-probe-v1",
        "program": program_path.name,
        "program_sha256": sha(payload),
        "encoder_sha256": contract["model_sha256"],
        "head_sha256": sha(HEAD.read_bytes()),
        "head_result_sha256": sha(HEAD_RESULT.read_bytes()),
        "input_contract_sha256": sha(INPUTS.read_bytes()),
        "threshold_binary32": float(result["evaluation"]["threshold_binary32"]),
        "classes": result["model"]["output_classes"],
        "training_roots": [56002],
        "input_contract": "v2 strict FOMO float32 >0.5; direct camera_xywh_px; shifted 120px camera ROI; 48x48x3 INT8 input",
        "scope": "one-shot volatile REPL research prototype; not installed; one host-trained head root; no qualification",
        "board_files_written": False,
        "startup_path_changed": False,
        "future_final_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True))
