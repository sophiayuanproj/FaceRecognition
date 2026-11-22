"""Camera + IMU bring-up recorder for Nicla Vision / OpenMV 5.0.1.

Captures grayscale frames and prints one JSON record per sample.
No images or logs are saved to board storage. See reports/nicla-bringup.md
for the completed five-frame hardware check.
"""

import gc
import json
import time
import csi
import imu
import omv

SAMPLE_PERIOD_MS = 500
SETTLE_MS = 2000
MAX_FRAMES = 0  # 0 runs until stopped by the host; set a positive limit if desired.
HMIRROR = False
VFLIP = False


def emit(event, fields):
    fields["schema_version"] = 1
    fields["event"] = event
    fields["ticks_ms"] = time.ticks_ms()
    print(json.dumps(fields))


def main():
    gc.collect()
    emit("camera_imu_start", {
        "openmv_version": omv.version_string(),
        "sample_period_ms": SAMPLE_PERIOD_MS,
        "pixel_format": "GRAYSCALE",
        "frame_size": [320, 240],
        "framebuffer_count": 1,
        "settle_ms": SETTLE_MS,
    })

    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.GRAYSCALE)
    cam.framesize(csi.QVGA)
    cam.auto_rotation(False)
    cam.hmirror(HMIRROR)
    cam.vflip(VFLIP)
    # Reapply after size/format/window changes: those reset the buffer count.
    cam.framebuffers(1)
    # Captures/discards settling frames and returns None; it is not exposure time.
    cam.snapshot(time=SETTLE_MS)
    emit("camera_ready", {
        "camera_chip_id": cam.cid(),
        "framebuffer_count": cam.framebuffers(),
        "gc_free_bytes": gc.mem_free(),
    })

    frame_index = 0
    while MAX_FRAMES == 0 or frame_index < MAX_FRAMES:
        cycle_start_ms = time.ticks_ms()
        capture_start_us = time.ticks_us()
        img = cam.snapshot()
        capture_us = time.ticks_diff(time.ticks_us(), capture_start_us)
        frame_ready_ticks_ms = time.ticks_ms()

        accel = imu.acceleration_mg()
        gyro = imu.angular_rate_mdps()
        imu_sample_ticks_ms = time.ticks_ms()
        # Statistics only read pixels; keep the frame unchanged for preview.
        statistics = img.get_statistics()
        emit("camera_imu_sample", {
            "frame_index": frame_index,
            "frame_ready_ticks_ms": frame_ready_ticks_ms,
            "imu_sample_ticks_ms": imu_sample_ticks_ms,
            "capture_us": capture_us,
            "width": img.width(),
            "height": img.height(),
            "brightness_mean_u8": statistics.mean,
            "brightness_stdev_u8": statistics.stdev,
            "accel_mg": accel,
            "gyro_mdps": gyro,
            "gc_free_bytes": gc.mem_free(),
        })
        frame_index += 1

        elapsed_ms = time.ticks_diff(time.ticks_ms(), cycle_start_ms)
        remaining_ms = SAMPLE_PERIOD_MS - elapsed_ms
        if remaining_ms > 0:
            # Sampling cadence only; this is not camera sensor power gating.
            time.sleep_ms(remaining_ms)

    emit("camera_imu_complete", {"frames": frame_index})


try:
    main()
except Exception as error:
    emit("fatal_error", {
        "component": "camera_imu",
        "error": str(error)[:160],
    })
    raise
