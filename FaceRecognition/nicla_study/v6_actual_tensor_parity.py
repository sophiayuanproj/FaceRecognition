"""Capture one exact on-device RGB565->FOMO->ROI->RGB48/int8 tensor fixture."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import v6_camera_crop_features48 as replay
from .v5_native_rgb import native_rgb


OUT = replay.STUDY / "reports/M6/corrected_saved_frame_parity_v1"
MODEL = replay.STUDY / "runs/operator_fused_encoder48/int8.tflite"
CONTRACT = replay.STUDY / "reports/M5/rgb48_encoder_parity/CONTRACT.json"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _device_program():
    identity_path = Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py"
    identity_text = identity_path.read_text(encoding="utf-8")
    identity_ast = ast.parse(identity_text)
    geometry_names = {"fomo_center_camera_px", "centered_roi_clamped"}
    geometry_nodes = [node for node in identity_ast.body
                      if isinstance(node, ast.FunctionDef) and node.name in geometry_names]
    if {node.name for node in geometry_nodes} != geometry_names:
        raise ValueError("shared identity geometry source is incomplete")
    geometry_source = ("import math\n\n" + "\n\n".join(
        ast.get_source_segment(identity_text, node) for node in geometry_nodes
    ) + "\n").encode("utf-8")
    geometry_b64 = base64.b64encode(geometry_source).decode("ascii")
    contract = json.loads(CONTRACT.read_text())
    lut = bytes([int(x) & 255 for x in contract["lut_signed_values"]])
    lut_b64 = base64.b64encode(lut).decode("ascii")
    model_sha = contract["model_sha256"]
    return '''import csi,ml,gc,json,binascii,hashlib,time,sys,ubinascii,math
from ml.preprocessing import Normalization
from ml.postprocessing.edgeimpulse import Fomo
exec(ubinascii.a2b_base64("@@GEOMETRY@@").decode())
LUT=ubinascii.a2b_base64("@@LUT@@")
EXPECTED_SHA="@@MODEL_SHA@@"
def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  while True:
   b=f.read(2048)
   if not b:break
   h.update(b)
 return binascii.hexlify(h.digest()).decode()
def hash_bytes(data):return binascii.hexlify(hashlib.sha256(data).digest()).decode()
def dump(name,data):
 parts=(len(data)+2047)//2048
 view=memoryview(data)
 for i in range(parts):
  encoded=binascii.b2a_base64(view[i*2048:(i+1)*2048]).decode().strip()
  print("DATA "+name+" "+str(i)+" "+str(parts)+" "+encoded)
 gc.collect()
def flat(value):
 out=bytearray();stack=[value]
 while stack:
  item=stack.pop()
  if hasattr(item,"shape") and len(item.shape)>0:
   for i in range(item.shape[0]-1,-1,-1):stack.append(item[i])
  else:out.append(int(item)&255)
 return out
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.auto_exposure(True);cam.snapshot(time=600)
frame=cam.snapshot();raw=frame.bytearray();meta={"width":frame.width(),"height":frame.height(),"pixel_format":"RGB565 little-endian","stride_bytes":frame.width()*2,"frame_sha256":hash_bytes(raw),"capture_ticks_ms":time.ticks_ms(),"camera_id":cam.cid(),"runtime":"OpenMV 5.0.1 / MicroPython 1.28.0-64","roi_side_camera_px":120,"tensor_hwc":[48,48,3],"tensor_color":"RGB"}
dump("camera_rgb565",raw)
gc.collect();det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));hits=det.predict([frame]);faces=hits[1] if len(hits)>1 else [];meta["fomo_result"]=str(hits);meta["fomo_face_count"]=len(faces)
if len(faces)==1 and float(faces[0][1])>=.5:
 rect=faces[0][0];center=fomo_center_camera_px(rect,frame.width(),frame.height());roi=centered_roi_clamped(center,120,frame.width(),frame.height());meta["fomo_rect_camera_xywh"]=[float(v) for v in rect];meta["center_xy_camera_px"]=list(center);meta["roi_xywh_camera_px"]=list(roi);meta["crop_status"]="single_cluster"
else:
 meta["fomo_rect_camera_xywh"]=None;meta["center_xy_camera_px"]=None;meta["roi_xywh_camera_px"]=None;meta["crop_status"]="no_face" if len(faces)==0 else "multiple_or_low_score"
det=None;hits=None;gc.collect()
if meta["crop_status"]=="single_cluster":
 model_path="/rom/operator_fused_encoder48.tflite";meta["encoder_sha256"]=file_sha(model_path)
 if meta["encoder_sha256"]!=EXPECTED_SHA:raise ValueError("ROMFS encoder SHA mismatch")
 model=ml.Model(model_path);rgb=bytearray(48*48*3);captured={};bound_norm=Normalization(roi=roi)(frame)
 def fill(buf,shape,dtype):
  if shape!=(1,48,48,3) or dtype!=ord("b"):raise ValueError("unexpected input tensor")
  bound_norm(rgb,shape,ord("B"))
  for j in range(len(buf)):buf[j]=LUT[rgb[j]]
  captured["input"]=bytes(buf)
 def output_cb(m,inputs,outputs):captured["output"]=bytes(flat(outputs[0]));return outputs
 model.predict([fill],callback=output_cb)
 if len(captured.get("input",b""))!=6912 or len(captured.get("output",b""))!=128:raise ValueError("missing exact tensors")
 dump("tensor_int8",captured["input"]);dump("embedding_int8",captured["output"]);meta["tensor_int8_sha256"]=hash_bytes(captured["input"]);meta["embedding_int8_sha256"]=hash_bytes(captured["output"])
print("META "+json.dumps(meta))
'''.replace("@@GEOMETRY@@", geometry_b64).replace("@@LUT@@", lut_b64).replace("@@MODEL_SHA@@", model_sha)


def _device_program_v3():
    """Compute and preserve the exact input before releasing camera/FOMO RAM."""
    identity_path = Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py"
    identity_text = identity_path.read_text(encoding="utf-8")
    identity_ast = ast.parse(identity_text)
    geometry_names = {"fomo_center_camera_px", "centered_roi_clamped"}
    geometry_nodes = [node for node in identity_ast.body
                      if isinstance(node, ast.FunctionDef) and node.name in geometry_names]
    if {node.name for node in geometry_nodes} != geometry_names:
        raise ValueError("shared identity geometry source is incomplete")
    geometry_source = ("import math\n\n" + "\n\n".join(
        ast.get_source_segment(identity_text, node) for node in geometry_nodes
    ) + "\n").encode("utf-8")
    geometry_b64 = base64.b64encode(geometry_source).decode("ascii")
    contract = json.loads(CONTRACT.read_text())
    lut = bytes([int(x) & 255 for x in contract["lut_signed_values"]])
    lut_b64 = base64.b64encode(lut).decode("ascii")
    model_sha = contract["model_sha256"]
    return '''import csi,ml,gc,json,binascii,hashlib,time,ubinascii
from ml.preprocessing import Normalization
from ml.postprocessing.edgeimpulse import Fomo
exec(ubinascii.a2b_base64("@@GEOMETRY@@").decode())
LUT=ubinascii.a2b_base64("@@LUT@@")
EXPECTED_SHA="@@MODEL_SHA@@"
def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  while True:
   b=f.read(2048)
   if not b:break
   h.update(b)
 return binascii.hexlify(h.digest()).decode()
def hash_bytes(data):return binascii.hexlify(hashlib.sha256(data).digest()).decode()
def dump(name,data):
 parts=(len(data)+2047)//2048
 view=memoryview(data)
 for i in range(parts):
  encoded=binascii.b2a_base64(view[i*2048:(i+1)*2048]).decode().strip()
  print("DATA "+name+" "+str(i)+" "+str(parts)+" "+encoded)
 gc.collect()
def flat(value):
 out=bytearray();stack=[value]
 while stack:
  item=stack.pop()
  if hasattr(item,"shape") and len(item.shape)>0:
   for i in range(item.shape[0]-1,-1,-1):stack.append(item[i])
  else:out.append(int(item)&255)
 return out
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.auto_exposure(True);cam.snapshot(time=600)
frame=cam.snapshot();raw=frame.bytearray()
meta={"width":frame.width(),"height":frame.height(),"pixel_format":"RGB565 little-endian","stride_bytes":frame.width()*2,"frame_sha256":hash_bytes(raw),"capture_ticks_ms":time.ticks_ms(),"camera_id":cam.cid(),"runtime":"OpenMV 5.0.1 / MicroPython 1.28.0-64","roi_side_camera_px":120,"tensor_hwc":[48,48,3],"tensor_color":"RGB","probe_version":3}
dump("camera_rgb565",raw)
gc.collect();det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));hits=det.predict([frame]);faces=hits[1] if len(hits)>1 else [];meta["fomo_result"]=str(hits);meta["fomo_face_count"]=len(faces)
if len(faces)==1 and float(faces[0][1])>=.5:
 rect=faces[0][0];center=fomo_center_camera_px(rect,frame.width(),frame.height());roi=centered_roi_clamped(center,120,frame.width(),frame.height());meta["fomo_rect_camera_xywh"]=[float(v) for v in rect];meta["center_xy_camera_px"]=list(center);meta["roi_xywh_camera_px"]=list(roi);meta["crop_status"]="single_cluster"
else:
 roi=None;meta["fomo_rect_camera_xywh"]=None;meta["center_xy_camera_px"]=None;meta["roi_xywh_camera_px"]=None;meta["crop_status"]="no_face" if len(faces)==0 else "multiple_or_low_score"
tensor=None
if roi is not None:
 rgb=bytearray(48*48*3);normalizer=Normalization(roi=roi)(frame);normalizer(rgb,(1,48,48,3),ord("B"));tensor=bytearray(len(rgb))
 for i in range(len(rgb)):tensor[i]=LUT[rgb[i]]
 meta["tensor_int8_sha256"]=hash_bytes(tensor);meta["tensor_int8_bytes"]=len(tensor)
 dump("tensor_int8",tensor)
meta["stage"]="input_saved_before_encoder";meta["gc_free_before_encoder"]=-1
try:meta["gc_free_before_encoder"]=gc.mem_free()
except Exception:pass
print("META "+json.dumps(meta))
det=None;hits=None;faces=None;normalizer=None;rgb=None;raw=None;frame=None;cam=None;roi=None;gc.collect()
if tensor is not None:
 model_path="/rom/operator_fused_encoder48.tflite";meta["encoder_sha256"]=file_sha(model_path)
 if meta["encoder_sha256"]!=EXPECTED_SHA:meta["encoder_error"]="ROMFS encoder SHA mismatch"
 else:
  model=None;captured={}
  try:
   gc.collect();meta["gc_free_at_load"]=gc.mem_free();t0=time.ticks_us();model=ml.Model(model_path);meta["encoder_load_us"]=time.ticks_diff(time.ticks_us(),t0);meta["encoder_arena_bytes"]=int(model.ram)
   def feed(buf,shape,dtype):
    if shape!=(1,48,48,3) or dtype!=ord("b"):raise ValueError("unexpected encoder input tensor")
    for j in range(len(buf)):buf[j]=tensor[j]
    captured["input"]=bytes(buf)
   def output_cb(m,inputs,outputs):captured["output"]=bytes(flat(outputs[0]));return outputs
   t0=time.ticks_us();model.predict([feed],callback=output_cb);meta["encoder_predict_us"]=time.ticks_diff(time.ticks_us(),t0)
   if len(captured.get("input",b""))!=6912 or len(captured.get("output",b""))!=128:raise ValueError("missing exact encoder tensors")
   meta["device_input_sha256"]=hash_bytes(captured["input"]);meta["embedding_int8_sha256"]=hash_bytes(captured["output"]);dump("embedding_int8",captured["output"]);meta["encoder_status"]="inference_complete"
  except Exception as error:meta["encoder_status"]="failed";meta["encoder_error"]=repr(error)
  model=None;captured=None;gc.collect()
print("META "+json.dumps(meta))
'''.replace("@@GEOMETRY@@", geometry_b64).replace("@@LUT@@", lut_b64).replace("@@MODEL_SHA@@", model_sha)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep the exact failed v1 program immutable beside its transcript.
    path = OUT / "DEVICE_PROGRAM_V2.py"
    path.write_text(_device_program())
    manifest = {"schema": "v6-actual-camera-parity-program-2", "program": path.name,
                "sha256": sha(path.read_bytes()), "model_sha256": json.loads(CONTRACT.read_text())["model_sha256"],
                "geometry_source_sha256": sha((Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py").read_bytes()),
                "raw_frame_stream_chunk_bytes": 2048,
                "preserves_failed_v1_program": True}
    (OUT / "DEVICE_PROGRAM_V2.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"program": str(path), "sha256": manifest["sha256"], "model_sha256": manifest["model_sha256"]}))


def prepare_v3():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "DEVICE_PROGRAM_V3.py"
    if path.exists():
        raise FileExistsError("preserving existing V3 program; use a new version name")
    payload = _device_program_v3().encode("utf-8")
    path.write_bytes(payload)
    manifest = {"schema": "v6-actual-camera-parity-program-3", "program": path.name,
                "sha256": sha(payload), "model_sha256": json.loads(CONTRACT.read_text())["model_sha256"],
                "geometry_source_sha256": sha((Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py").read_bytes()),
                "precompute_tensor_before_encoder_allocation": True,
                "release_camera_and_detector_references_before_encoder_allocation": True,
                "preserves_v1_v2_programs": True}
    manifest_path = OUT / "DEVICE_PROGRAM_V3.json"
    if manifest_path.exists():
        raise FileExistsError("preserving existing V3 manifest; use a new version name")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"program": str(path), "sha256": manifest["sha256"], "model_sha256": manifest["model_sha256"]}))


def prepare_v4():
    """Keep large frame streaming after FOMO and preserve FOMO allocation errors."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "DEVICE_PROGRAM_V4.py"
    if path.exists():
        raise FileExistsError("preserving existing V4 program; use a new version name")
    program = _device_program_v3()
    old = '''meta={"width":frame.width(),"height":frame.height(),"pixel_format":"RGB565 little-endian","stride_bytes":frame.width()*2,"frame_sha256":hash_bytes(raw),"capture_ticks_ms":time.ticks_ms(),"camera_id":cam.cid(),"runtime":"OpenMV 5.0.1 / MicroPython 1.28.0-64","roi_side_camera_px":120,"tensor_hwc":[48,48,3],"tensor_color":"RGB","probe_version":3}
dump("camera_rgb565",raw)
gc.collect();det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));hits=det.predict([frame]);faces=hits[1] if len(hits)>1 else [];meta["fomo_result"]=str(hits);meta["fomo_face_count"]=len(faces)'''
    new = '''meta={"width":frame.width(),"height":frame.height(),"pixel_format":"RGB565 little-endian","stride_bytes":frame.width()*2,"frame_sha256":hash_bytes(raw),"capture_ticks_ms":time.ticks_ms(),"camera_id":cam.cid(),"runtime":"OpenMV 5.0.1 / MicroPython 1.28.0-64","roi_side_camera_px":120,"tensor_hwc":[48,48,3],"tensor_color":"RGB","probe_version":4}
try:
 gc.collect();det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));hits=det.predict([frame]);faces=hits[1] if len(hits)>1 else [];meta["fomo_result"]=str(hits);meta["fomo_face_count"]=len(faces)
except Exception as error:
 det=None;hits=None;faces=[];meta["fomo_face_count"]=None;meta["fomo_error"]=repr(error)
gc.collect()'''
    if old not in program:
        raise ValueError("V4 FOMO section template changed; refusing to write a different probe")
    program = program.replace(old, new, 1)
    old_stage = 'meta["stage"]="input_saved_before_encoder";meta["gc_free_before_encoder"]=-1'
    new_stage = 'dump("camera_rgb565",raw)\nmeta["stage"]="camera_and_input_saved_before_encoder";meta["gc_free_before_encoder"]=-1'
    if old_stage not in program:
        raise ValueError("V4 capture stage template changed")
    program = program.replace(old_stage, new_stage, 1)
    payload = program.encode("utf-8")
    path.write_bytes(payload)
    manifest = {"schema": "v6-actual-camera-parity-program-4", "program": path.name,
                "sha256": sha(payload), "model_sha256": json.loads(CONTRACT.read_text())["model_sha256"],
                "geometry_source_sha256": sha((Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py").read_bytes()),
                "precompute_tensor_before_encoder_allocation": True,
                "stream_frame_only_after_detector_and_tensor": True,
                "preserves_v1_v2_v3_programs": True}
    manifest_path = OUT / "DEVICE_PROGRAM_V4.json"
    if manifest_path.exists():
        raise FileExistsError("preserving existing V4 manifest; use a new version name")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"program": str(path), "sha256": manifest["sha256"], "model_sha256": manifest["model_sha256"]}))


def prepare_v5():
    """Use the deployed strict score gate before deciding FOMO cluster count."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "DEVICE_PROGRAM_V5.py"
    if path.exists():
        raise FileExistsError("preserving existing V5 program; use a new version name")
    program = _device_program_v3().replace('"probe_version":3', '"probe_version":5', 1)
    old = '''dump("camera_rgb565",raw)
gc.collect();det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));hits=det.predict([frame]);faces=hits[1] if len(hits)>1 else [];meta["fomo_result"]=str(hits);meta["fomo_face_count"]=len(faces)'''
    new = '''gc.collect()
try:
 det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5));hits=det.predict([frame]);raw_faces=hits[1] if len(hits)>1 else [];meta["fomo_result"]=str(hits);meta["fomo_raw_cluster_count"]=len(raw_faces);faces=[item for item in raw_faces if float(item[1])>.5];meta["fomo_accepted_scores"]=[float(item[1]) for item in faces];meta["fomo_face_count"]=len(faces)
except Exception as error:
 det=None;hits=None;raw_faces=[];faces=[];meta["fomo_face_count"]=None;meta["fomo_error"]=repr(error)
gc.collect()'''
    if old not in program:
        raise ValueError("V5 detector section template changed; refusing to write another program")
    program = program.replace(old, new, 1)
    old_stage = 'meta["stage"]="input_saved_before_encoder";meta["gc_free_before_encoder"]=-1'
    new_stage = 'dump("camera_rgb565",raw)\nmeta["stage"]="camera_and_input_saved_before_encoder";meta["gc_free_before_encoder"]=-1'
    if old_stage not in program:
        raise ValueError("V5 capture stage template changed")
    program = program.replace(old_stage, new_stage, 1)
    payload = program.encode("utf-8")
    path.write_bytes(payload)
    manifest = {"schema": "v6-actual-camera-parity-program-5", "program": path.name,
                "sha256": sha(payload), "model_sha256": json.loads(CONTRACT.read_text())["model_sha256"],
                "geometry_source_sha256": sha((Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py").read_bytes()),
                "detector_threshold": 0.5, "threshold_comparison": "strict_gt_float32",
                "filters_subthreshold_clusters_before_counting": True,
                "precompute_tensor_before_encoder_allocation": True,
                "releases_camera_and_detector_references_before_encoder_allocation": True,
                "preserves_v1_v2_v3_v4_programs": True}
    manifest_path = OUT / "DEVICE_PROGRAM_V5.json"
    if manifest_path.exists():
        raise FileExistsError("preserving existing V5 manifest; use a new version name")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"program": str(path), "sha256": manifest["sha256"], "model_sha256": manifest["model_sha256"]}))


def prepare_v6():
    """Sample up to ten frames and run identity only on the first single hit."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "DEVICE_PROGRAM_V6.py"
    if path.exists():
        raise FileExistsError("preserving existing V6 program; use a new version name")
    identity_path = Path(__file__).resolve().parents[1] / "firmware/identity_contracts.py"
    identity_text = identity_path.read_text(encoding="utf-8")
    identity_ast = ast.parse(identity_text)
    geometry_names = {"fomo_center_camera_px", "centered_roi_clamped"}
    geometry_nodes = [node for node in identity_ast.body
                      if isinstance(node, ast.FunctionDef) and node.name in geometry_names]
    if {node.name for node in geometry_nodes} != geometry_names:
        raise ValueError("shared identity geometry source is incomplete")
    geometry_source = ("import math\n\n" + "\n\n".join(
        ast.get_source_segment(identity_text, node) for node in geometry_nodes
    ) + "\n").encode("utf-8")
    geometry_b64 = base64.b64encode(geometry_source).decode("ascii")
    contract = json.loads(CONTRACT.read_text())
    lut = bytes([int(x) & 255 for x in contract["lut_signed_values"]])
    lut_b64 = base64.b64encode(lut).decode("ascii")
    model_sha = contract["model_sha256"]
    program = '''import csi,ml,gc,json,binascii,hashlib,time,ubinascii
from ml.preprocessing import Normalization
from ml.postprocessing.edgeimpulse import Fomo
exec(ubinascii.a2b_base64("@@GEOMETRY@@").decode())
LUT=ubinascii.a2b_base64("@@LUT@@")
EXPECTED_SHA="@@MODEL_SHA@@"
def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  while True:
   b=f.read(2048)
   if not b:break
   h.update(b)
 return binascii.hexlify(h.digest()).decode()
def hash_bytes(data):return binascii.hexlify(hashlib.sha256(data).digest()).decode()
def dump(name,data):
 parts=(len(data)+2047)//2048
 view=memoryview(data)
 for i in range(parts):
  encoded=binascii.b2a_base64(view[i*2048:(i+1)*2048]).decode().strip()
  print("DATA "+name+" "+str(i)+" "+str(parts)+" "+encoded)
 gc.collect()
def flat(value):
 out=bytearray();stack=[value]
 while stack:
  item=stack.pop()
  if hasattr(item,"shape") and len(item.shape)>0:
   for i in range(item.shape[0]-1,-1,-1):stack.append(item[i])
  else:out.append(int(item)&255)
 return out
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.auto_exposure(True);cam.snapshot(time=600)
meta={"probe_version":6,"runtime":"OpenMV 5.0.1 / MicroPython 1.28.0-64","camera_frame_geometry":[240,240],"pixel_format":"RGB565 little-endian","roi_side_camera_px":120,"tensor_hwc":[48,48,3],"tensor_color":"RGB","fomo_threshold":0.5,"threshold_comparison":"strict_gt","attempt_limit":10,"frame_sequence":[]}
det=None;selected=False;tensor=None;frame=None;raw=None;hits=None;faces=None
try:
 det=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5))
 for frame_index in range(10):
  frame=cam.snapshot();t0=time.ticks_us();hits=det.predict([frame]);predict_us=time.ticks_diff(time.ticks_us(),t0);raw_faces=hits[1] if len(hits)>1 else [];faces=[item for item in raw_faces if float(item[1])>.5]
  row={"frame_index":frame_index,"fomo_predict_us":predict_us,"raw_cluster_count":len(raw_faces),"accepted_scores":[float(item[1]) for item in faces],"fomo_result":str(hits)};meta["frame_sequence"].append(row)
  if len(faces)==1:
   rect=faces[0][0];center=fomo_center_camera_px(rect,frame.width(),frame.height());roi=centered_roi_clamped(center,120,frame.width(),frame.height());rgb=bytearray(48*48*3);normalizer=Normalization(roi=roi)(frame);normalizer(rgb,(1,48,48,3),ord("B"));tensor=bytearray(len(rgb))
   for i in range(len(rgb)):tensor[i]=LUT[rgb[i]]
   raw=frame.bytearray();meta.update({"selected_single_face":True,"selected_frame_index":frame_index,"selected_fomo_rect_camera_xywh":[float(v) for v in rect],"center_xy_camera_px":list(center),"roi_xywh_camera_px":list(roi),"fomo_accepted_scores":row["accepted_scores"],"fomo_face_count":1,"fomo_raw_cluster_count":len(raw_faces),"fomo_result":str(hits),"frame_sha256":hash_bytes(raw),"tensor_int8_sha256":hash_bytes(tensor),"tensor_int8_bytes":len(tensor),"camera_id":cam.cid()})
   dump("camera_rgb565",raw);dump("tensor_int8",tensor);selected=True;break
  frame=None;hits=None;raw_faces=None;faces=None;gc.collect()
except Exception as error:
 meta["probe_error"]=repr(error);det=None;gc.collect()
meta["selected_single_face"]=selected;meta["frames_examined"]=len(meta["frame_sequence"]);meta["stage"]="single_face_input_saved_before_encoder" if selected else "no_single_face_in_attempt_window"
print("META "+json.dumps(meta))
det=None;hits=None;faces=None;normalizer=None;rgb=None;frame=None;raw=None;cam=None;gc.collect()
if selected:
 model_path="/rom/operator_fused_encoder48.tflite";meta["encoder_sha256"]=file_sha(model_path)
 if meta["encoder_sha256"]!=EXPECTED_SHA:meta["encoder_error"]="ROMFS encoder SHA mismatch"
 else:
  model=None;captured={}
  try:
   gc.collect();meta["gc_free_at_load"]=gc.mem_free();t0=time.ticks_us();model=ml.Model(model_path);meta["encoder_load_us"]=time.ticks_diff(time.ticks_us(),t0);meta["encoder_arena_bytes"]=int(model.ram)
   def feed(buf,shape,dtype):
    if shape!=(1,48,48,3) or dtype!=ord("b"):raise ValueError("unexpected encoder input tensor")
    for j in range(len(buf)):buf[j]=tensor[j]
    captured["input"]=bytes(buf)
   def output_cb(m,inputs,outputs):captured["output"]=bytes(flat(outputs[0]));return outputs
   t0=time.ticks_us();model.predict([feed],callback=output_cb);meta["encoder_predict_us"]=time.ticks_diff(time.ticks_us(),t0)
   if len(captured.get("input",b""))!=6912 or len(captured.get("output",b""))!=128:raise ValueError("missing exact encoder tensors")
   meta["device_input_sha256"]=hash_bytes(captured["input"]);meta["embedding_int8_sha256"]=hash_bytes(captured["output"]);dump("embedding_int8",captured["output"]);meta["encoder_status"]="inference_complete"
  except Exception as error:meta["encoder_status"]="failed";meta["encoder_error"]=repr(error)
  model=None;captured=None;gc.collect()
print("META "+json.dumps(meta))
'''.replace("@@GEOMETRY@@", geometry_b64).replace("@@LUT@@", lut_b64).replace("@@MODEL_SHA@@", model_sha)
    payload = program.encode("utf-8")
    path.write_bytes(payload)
    manifest = {"schema": "v6-actual-camera-parity-program-6", "program": path.name,
                "sha256": sha(payload), "model_sha256": json.loads(CONTRACT.read_text())["model_sha256"],
                "geometry_source_sha256": sha(identity_path.read_bytes()),
                "max_consecutive_exposures": 10, "independent_soft_reset_blocks": 1,
                "single_face_gate_matches_runtime": "strict_gt_float32_0.5",
                "identity_run_only_after_one_accepted_cluster": True,
                "precompute_tensor_before_encoder_allocation": True,
                "preserves_v1_through_v5_programs": True}
    manifest_path = OUT / "DEVICE_PROGRAM_V6.json"
    if manifest_path.exists():
        raise FileExistsError("preserving existing V6 manifest; use a new version name")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"program": str(path), "sha256": manifest["sha256"], "model_sha256": manifest["model_sha256"]}))


def _read_blobs(transcript):
    chunks, metadata = {}, None
    for line in Path(transcript).read_bytes().decode("utf-8", errors="replace").splitlines():
        if line.startswith("DATA "):
            _, name, part, total, value = line.split(" ", 4)
            # Each 2048-byte device chunk is base64-padded independently.
            # Decode each piece before joining; concatenating padded strings
            # corrupts the recovered binary frame at every chunk boundary.
            chunks.setdefault(name, {})[int(part)] = (int(total), base64.b64decode(value, validate=True))
        elif line.startswith("META "):
            metadata = json.loads(line[5:])
    blobs = {}
    for name, parts in chunks.items():
        totals = {n for n, _ in parts.values()}
        if len(totals) != 1 or set(parts) != set(range(totals.pop())):
            raise ValueError("missing or inconsistent device blob chunks: " + name)
        blobs[name] = b"".join(parts[i][1] for i in range(len(parts)))
    return metadata, blobs


def verify(transcript, output_path):
    metadata, blobs = _read_blobs(transcript)
    if metadata is None or "camera_rgb565" not in blobs:
        raise ValueError("device transcript lacks saved raw camera frame")
    raw = blobs["camera_rgb565"]
    if len(raw) != 240 * 240 * 2 or sha(raw) != metadata["frame_sha256"]:
        raise ValueError("device raw camera frame hash/geometry mismatch")
    output_path = Path(output_path).resolve()
    output_path.relative_to(replay.STUDY.resolve())
    private = replay.PRIVATE / "device_parity_runs" / output_path.stem
    private.mkdir(parents=True, exist_ok=True)
    frame_path = private / "real_frame_rgb565.bin"
    frame_path.write_bytes(raw)
    result = {"schema": "v6-corrected-saved-frame-device-parity-1", "capture_id": "device-live-parity-0001",
              "frame_sha256": sha(raw), "metadata": metadata, "saved_frame_path": str(frame_path),
              "camera_frame_geometry": [240, 240], "roi_side_camera_px": 120, "encoder_tensor_hwc": [48,48,3],
              "single_physical_exposure_frame": True, "board_reset_kind": "software_reset"}
    has_single_roi = (metadata.get("crop_status") == "single_cluster" or
                      metadata.get("selected_single_face") is True)
    if not has_single_roi:
        result.update({"status": "no_matching_identity_roi", "reason": metadata.get("crop_status"),
                       "board_input_equal": None, "model_output_equal": None})
    else:
        if "tensor_int8" not in blobs:
            raise ValueError("device transcript lacks input tensor bytes")
        roi = tuple(int(x) for x in metadata["roi_xywh_camera_px"])
        pixels = np.frombuffer(raw, dtype="<u2").reshape(240, 240)
        rgb = native_rgb(pixels, roi=roi, width=48, height=48)
        contract = json.loads(CONTRACT.read_text())
        lut = np.asarray(contract["lut_signed_values"], dtype=np.int8)
        expected_input = lut[rgb].tobytes()
        board_input = blobs["tensor_int8"]
        model_payload = MODEL.read_bytes()
        expected_output = None
        board_output = blobs.get("embedding_int8")
        if board_output is not None:
            it = tf.lite.Interpreter(model_content=model_payload, num_threads=2,
                                     experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
            it.allocate_tensors(); inp, out = it.get_input_details()[0], it.get_output_details()[0]
            it.set_tensor(inp["index"], np.frombuffer(board_input, dtype=np.int8).reshape(1,48,48,3));it.invoke()
            expected_output = it.get_tensor(out["index"])[0].astype(np.int8, copy=False).tobytes()
        input_equal = board_input == expected_input
        output_equal = (board_output == expected_output) if board_output is not None else None
        if input_equal:
            if output_equal is True:
                parity_status = "verified_exact"
            elif output_equal is False:
                parity_status = "output_mismatch"
            else:
                parity_status = "input_verified_encoder_unavailable"
        else:
            parity_status = "input_mismatch"
        result.update({
            "status": parity_status,
            "device_fomo_rect_space": "camera_xywh_px", "center_xy_camera_px": metadata["center_xy_camera_px"],
            "roi_xywh_camera_px": list(roi), "host_input_sha256": sha(expected_input),
            "device_input_sha256": sha(board_input), "board_input_equal": input_equal,
            "host_output_sha256": sha(expected_output) if expected_output is not None else None,
            "device_output_sha256": sha(board_output) if board_output is not None else None,
            "model_output_equal": output_equal,
            "input_bytes": len(board_input), "output_bytes": len(board_output) if board_output is not None else None,
            "host_model_sha256": sha(model_payload), "device_model_sha256": metadata.get("encoder_sha256"),
            "host_backend": "TensorFlow 2.10.1 BUILTIN_REF", "host_transform": "v5_native_rgb OpenMV 5.0.1 emulator + exact model LUT",
        })
        (private / "device_tensor_int8.bin").write_bytes(board_input)
        if board_output is not None:
            (private / "device_embedding_int8.bin").write_bytes(board_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] in ("input_mismatch", "output_mismatch"):
        raise RuntimeError("actual camera crop input/model parity mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "prepare-v3", "prepare-v4", "prepare-v5", "prepare-v6", "verify"))
    parser.add_argument("--transcript")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "prepare-v3":
        prepare_v3()
    elif args.action == "prepare-v4":
        prepare_v4()
    elif args.action == "prepare-v5":
        prepare_v5()
    elif args.action == "prepare-v6":
        prepare_v6()
    else:
        if not args.transcript or not args.output:
            raise ValueError("--transcript and --output are required for verify; use a new versioned result path")
        verify(args.transcript, args.output)


if __name__ == "__main__":
    main()
