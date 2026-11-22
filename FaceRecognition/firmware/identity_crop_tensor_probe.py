"""RAM-only Nicla probe for corrected camera ROI and int8 input bytes.

Run through tools/research_v5/board_lab.py. It captures one face target,
reports the direct camera-space FOMO rectangle, applies the same fixed 120 px
inward-clamped ROI contract, and dumps the 48x48x3 int8 Normalization input.
No identity inference or flash write is performed.
"""

import binascii
import csi
import gc
import json
import math
import ml
import time

from ml.postprocessing.edgeimpulse import Fomo
from ml.preprocessing import Normalization


def fomo_center_camera_px(rect_xywh, frame_width, frame_height):
    if len(rect_xywh) != 4:
        raise ValueError("FOMO rectangle must be x,y,width,height")
    x, y, width, height = [float(value) for value in rect_xywh]
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("invalid FOMO camera rectangle")
    if width <= 0 or height <= 0:
        raise ValueError("invalid FOMO camera rectangle")
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("FOMO camera rectangle outside frame")
    return (x + width / 2.0, y + height / 2.0)


def centered_roi_clamped(center_xy_camera_px, side_camera_px, frame_width, frame_height):
    side = int(side_camera_px)
    if side <= 0 or side > frame_width or side > frame_height:
        raise ValueError("ROI side must fit the camera frame")
    center_x, center_y = [float(value) for value in center_xy_camera_px]
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise ValueError("invalid camera center")
    x = int(math.floor(center_x - side / 2.0))
    y = int(math.floor(center_y - side / 2.0))
    x = min(max(0, x), frame_width - side)
    y = min(max(0, y), frame_height - side)
    return (x, y, side, side)


def dump(name, data):
    for offset in range(0, len(data), 2048):
        chunk = memoryview(data)[offset:offset + 2048]
        print("DATA " + name + " " + binascii.b2a_base64(chunk).decode().strip())
        gc.collect()


camera = csi.CSI()
camera.reset()
camera.pixformat(csi.RGB565)
camera.framesize(csi.QVGA)
camera.window((240, 240))
camera.framebuffers(1)
camera.auto_exposure(True)
camera.snapshot(time=600)
capture_start = time.ticks_us()
frame = camera.snapshot()
metadata = {
    "schema": "nicla-corrected-identity-input-probe-1",
    "width": frame.width(), "height": frame.height(),
    "pixel_format": "RGB565", "stride_bytes": frame.width() * 2,
    "capture_us": time.ticks_diff(time.ticks_us(), capture_start),
    "fomo_load_us": None, "fomo_predict_us": None,
    "fomo": None, "center_camera_xy_px": None, "roi_xywh_camera_px": None,
    "identity_input_shape": [1, 48, 48, 3], "identity_input_dtype": "int8",
    "identity_inference_performed": False, "flash_written": False,
    "evaluation_role": "non_evaluation_diagnostic",
}
start = time.ticks_us()
detector = ml.Model("/rom/fomo_face_detection.tflite", postprocess=Fomo(threshold=0.5))
metadata["fomo_load_us"] = time.ticks_diff(time.ticks_us(), start)
start = time.ticks_us()
result = detector.predict([frame])
metadata["fomo_predict_us"] = time.ticks_diff(time.ticks_us(), start)
metadata["fomo"] = str(result)
faces = result[1] if len(result) > 1 else []
if len(faces) == 1:
    rect = faces[0][0]
    center = fomo_center_camera_px(rect, frame.width(), frame.height())
    roi = centered_roi_clamped(center, 120, frame.width(), frame.height())
    detector = None
    result = None
    gc.collect()
    tensor = bytearray(48 * 48 * 3)
    start = time.ticks_us()
    Normalization(roi=roi)(frame)(tensor, (1, 48, 48, 3), ord("b"))
    metadata["identity_preprocess_us"] = time.ticks_diff(time.ticks_us(), start)
    metadata["center_camera_xy_px"] = list(center)
    metadata["roi_xywh_camera_px"] = list(roi)
    metadata["identity_tensor_bytes"] = len(tensor)
    dump("identity_tensor_i8", tensor)
    tensor = None
else:
    metadata["identity_input_status"] = "skipped_without_exactly_one_fomo_cluster"

dump("camera_rgb565", frame.bytearray())
frame = None
gc.collect()
print("META " + json.dumps(metadata, allow_nan=False))
