"""Volatile Nicla GC2145 sleep/wake diagnostic for board_probe.py.

Each host block runs this program after a fresh MicroPython soft reset. It only
configures the sensor and returns frame dimensions/timing; it writes no board
files, changes no startup path, runs no detector/identity model, and measures
no power. BLOCK_INDEX is injected by board_probe.py.
"""

import csi
import gc
import json
import time


block_index = int(BLOCK_INDEX)
camera = csi.CSI()
camera.reset()
camera.pixformat(csi.RGB565)
camera.framesize(csi.QVGA)
camera.window((240, 240))
camera.framebuffers(1)
camera.snapshot(time=600)

first = camera.snapshot()
if first is None or first.width() != 240 or first.height() != 240:
    raise ValueError("initial_frame_not_240x240")

camera.sleep(True)
time.sleep_ms(100)
wake_start = time.ticks_us()
camera.sleep(False)
camera.snapshot(time=600)
frame = camera.snapshot()
wake_to_frame_us = time.ticks_diff(time.ticks_us(), wake_start)
if frame is None or frame.width() != 240 or frame.height() != 240:
    raise ValueError("wake_frame_not_240x240")

print(json.dumps({
    "schema": "nicla-gc2145-sleep-wake-probe-v1",
    "block_index": block_index,
    "reset_kind": "independent_micropython_soft_reset",
    "sensor_id": camera.cid(),
    "pixel_format": "RGB565",
    "frame_width": frame.width(),
    "frame_height": frame.height(),
    "wake_to_usable_frame_ms": wake_to_frame_us / 1000.0,
    "python_heap_free_bytes": gc.mem_free(),
    "target_files_written": False,
    "startup_path_changed": False,
    "identity_model_run": False,
    "power_measured": False,
}))

frame = None
first = None
camera.sleep(False)
gc.collect()
