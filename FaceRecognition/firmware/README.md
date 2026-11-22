# Nicla Vision starter firmware

These scripts target **Arduino Nicla Vision with OpenMV 5.0.1**. At the last
verified session on 2026-10-22, the board used the stock ROMFS and had the
historical locally trained 64-D embedding prototype in writable `/flash`. The
user now says the Nicla is with their partner; no current readback is available
and no v2 candidate has been uploaded. Read-only diagnostics, five-frame
camera/IMU telemetry, and stock-model inference completed; see
[`reports/nicla-bringup.md`](../reports/nicla-bringup.md). A historical embedding
smoke loaded and ran on the board, but no v2 model was read back or qualified.
Real-person enrollment, camera-domain identity accuracy, and power remain
unvalidated.

## Run in this order

From the repository root, after installing the board firmware and host tooling:

```sh
.venv-nicla/bin/openmv --port /dev/cu.usbmodemACTUALPORT --script firmware/board_info.py
.venv-nicla/bin/openmv --port /dev/cu.usbmodemACTUALPORT --script firmware/camera_imu.py
.venv-nicla/bin/openmv --port /dev/cu.usbmodemACTUALPORT --script firmware/stock_face_detector.py
```

Replace the port placeholder with the Nicla's actual port. Run one command at a
time. Stop the current session before opening Arduino Serial Monitor, `mpremote`,
or another OpenMV connection. The continuous scripts run until the host stops
them; alternatively, set `MAX_FRAMES` to a positive limit. Merely installing the
host package does not make `csi`, `imu`, or `ml` available to desktop Python.

### `board_info.py`

Reports board and firmware identity, GC memory, CPU frequencies, optional UMA
pool statistics, and ROMFS partition/asset information. It does not start camera
capture or write to flash. On this pinned firmware, expect one 4 MiB ROMFS
partition. A missing optional diagnostic is reported as JSON rather than being
silently treated as zero.

### `camera_imu.py`

Captures 320 × 240 grayscale frames with one framebuffer, initially allowing
2 seconds for camera settling. At a nominal 500 ms sampling period it reports
brightness statistics, capture duration, accelerometer and gyroscope readings,
and GC free memory. It saves no images or logs on the board.

- `accel_mg`: milligravity units; divide by 1,000 for g. Gravity is present when
  the board is stationary.
- `gyro_mdps`: millidegrees per second; divide by 1,000 for degrees per second.
- `brightness_mean_u8` and `brightness_stdev_u8`: grayscale statistics, using
  pixel values from 0 to 255.
- IMU readings describe board/wearer motion, not another person's movement.
  Their timestamp follows capture; it is not a hardware-synchronized exposure
  timestamp.

Mirror/flip defaults are provisional. Check the wearable orientation during
bring-up, then change `HMIRROR`/`VFLIP` consistently across scripts.

### `stock_face_detector.py`

Loads `/rom/fomo_face_detection.tflite`, using the official stock RGB model,
FOMO postprocessing, RGB565 capture, QVGA sensor size, and a centered 240 × 240
window. It logs model metadata and load duration, then reports detections from
unchanged frames. It handles empty class lists and skips FOMO's background
class. A record contains at most eight detections, with the full count and a
truncation flag; no image history is retained.

This probe detects face presence/heatmap regions. It does **not** recognize
identities or establish calibrated face crop boxes. It is the detector used by
the local embedding prototype below. No face detector was trained from the
cropped recognition datasets. No person-specific data is needed for this
bring-up.

### Legacy classifier starter: `recognize.py`

This starter script describes the earlier custom detector/softmax-classifier
contract. It is not the deployed path; the local prototype uses
`recognize_embedding.py`, the stock FOMO model, and the full-INT8 embedder from
`/flash`.

After real custom exports are stored in `/rom`, use `bin/nicla upload-config PORT`
to copy settings to `/flash` and reset, then `bin/nicla recognize PORT`. Sending
the main script alone does not upload its configuration dependency.

The script validates model interfaces, detects within a square source-image
ROI, chooses a fixed square crop around the source-coordinate FOMO center,
and predicts one roster label. Multiple faces and clipped crops are rejected.
Identity acceptance thresholds and crop size are provisional configuration.
Only two face records are retained, with the full count logged. A clean source
frame is used for both models before any diagnostic drawing. Disable overlays
for data acquisition or performance comparisons.

The default loads one model at a time, including model-loading overhead in its
timings. Recognition timing fields use `_ms`; the bring-up scripts use `_us`.
Neither reports a pure accelerator latency or a whole-device peak-RAM value.

## Timing and memory interpretation

Every log line is JSON with `schema_version: 1` and an event name. The `ticks_ms`
and `frame_ready_ticks_ms` counters can wrap. Durations use `ticks_us()` and
`ticks_diff()`; `capture_us`, `load_us`, and `inference_us` are **microseconds**.
Inference duration includes automatic image-to-tensor preprocessing and FOMO
postprocessing. It excludes subsequent JSON formatting and output. The sampling
period is a target cadence, not a guarantee when capture/inference takes longer.

`snapshot(time=2000)` captures/discards settling frames and returns `None`. It
does not set camera exposure. Single-buffer capture is deliberately sequential:
finish inference before the next snapshot overwrites the framebuffer.

`model.ram` reports the tensor arena only. GC free memory is aggregate free
space, not proof that a large contiguous allocation is possible. The stock
firmware configures 369 KiB of GC regions, with a largest initial region of
276 KiB, and a separate 476 KiB UMA pool. Camera buffers reserve two bytes per
pixel even for grayscale capture. Format/size/window changes can reset the
buffer count, so these scripts explicitly set one framebuffer afterward.

## Power work comes after bring-up

The sampling delay and low output rate here do not demonstrate camera sensor
power savings. This GC2145 driver uses software frame dropping for framerate
control. Camera sleep/wake, model switching, and whole-board power must be
implemented and measured as separate steps. These scripts make no battery-life,
energy, recognition-accuracy, or deployment-readiness claim.

## Local embedding prototype

The trained course prototype is an INT8 96×96 grayscale feature extractor. It
is deployed separately at `/flash/identity_embedding_int8.tflite`; the stock
FOMO detector remains at `/rom/fomo_face_detection.tflite`. This path does not
build or upload a custom ROMFS. Copy the model and validation calibration JSON
to `/flash`, then run `bin/nicla upload-embedding-config PORT`.

`recognize_embedding.py` runs stock FOMO, accepts exactly one detected region,
rejects clipped or low-quality crops, extracts a 64-value embedding, and
compares cosine similarity to explicitly enrolled local templates. With no
`/flash/embedding_templates.json`, it returns `no_enrollment`; it never claims
to know a person. The validation-derived pair threshold and the simple
low-quality/uncertainty bands are prototype settings, not camera-domain
calibration. FOMO heatmap centroids are not measured face boxes, so the fixed
96-pixel crop needs further validation on consenting, camera-matched samples.

Use `bin/nicla embedding-smoke PORT` for the 100-frame forced-center-crop model
run. It records model load, capture and embedding timings, tensor arena and
minimum Python heap, but outputs no identity and saves no frames. It is an
execution/stability check, not a face-accuracy test. For the board-versus-host
comparison, `board_eval_embeddings.py` temporarily reads a tiny set of
pseudonymous 96×96 PGM crops copied to `/flash/board_eval/`; delete those crops
and the board manifest immediately afterward. Matching vectors establishes
numerical parity on those samples only.

`bin/nicla recognize-embedding PORT` starts the camera pipeline. A live run is
appropriate only while the camera faces a known neutral target or a consenting
participant. No enrollment of the user or partner is included in this repo.

## Standalone power-on application

`bin/nicla deploy-standalone PORT` turns the prototype into a local cold-boot
application. It confirms the OpenMV CSI/ML modules and stock FOMO file, backs up
the existing `/flash/main.py` to a timestamped local directory and to
`/flash/main.py.stock`, copies the recognition module/config/model/calibration,
checks each remote SHA-256, installs `firmware/main.py` last, then hard-resets
the board. It does not change firmware, bootloader, ROMFS, or any existing
enrollment template. After reset the camera pipeline runs on the Nicla; no IDE
or laptop-side process is required. At the last verified session the supplied
Nicla ran OpenMV 5.0.1, so this deployment was MicroPython running on the Arduino
board rather than an Arduino IDE sketch. Current device contents have not been
re-read in this host-only continuation.

```sh
bin/nicla ports
PORT=/dev/cu.usbmodemACTUALPORT
bin/nicla deploy-standalone "$PORT"
```

An un-enrolled device reports `no_enrollment`, never a known person. The RGB
LED gives a nonverbal status when no serial terminal is attached: green means a
validation-threshold match, red unknown, amber uncertain/multiple faces, and
blue no face/no enrollment. JSON diagnostics are sent only when USB serial is
connected so a disconnected device cannot fill a blocked logging buffer. Ctrl-C
over USB exits the app to the MicroPython REPL for repair.

The v2 controlled adapter (`recognize_controlled.py` plus
`policy_controller.py`) is staged separately and remains disabled in
`embedding_config.py`. It additionally requires `policy_config.json` and
hash-matched candidate models, templates, and calibration. The legacy
`deploy-standalone` command above does not install or qualify that bundle; use
the research-next transactional board procedure only after the development
candidate order is frozen. No v2 model has been deployed.

Use `bin/nicla restore-startup PORT` to restore `/flash/main.py.stock` and
hard-reset. The timestamped pre-install host backup is under
`work/model-training/deployment-backup/standalone/`. Model/config files remain
on `/flash` after restoring startup; delete them manually only if complete
asset removal is intended. The stock FOMO ROMFS is retained in all cases.

Enrollment requires a separate explicit action and one consenting person in
view. `bin/nicla enroll PORT PSEUDONYM --samples 5 --attempts 30` writes a
temporary request, captures one face at a time, rejects clipped/low-quality
crops, and saves only an averaged normalized 64-D vector. It does not save
images or names. Replacing an existing pseudonymous ID requires `--replace`.
This should not be used for access control: the validation threshold is not
camera-domain calibrated, and the expanded held-out validation error is too
high for security-sensitive identification.

## Compact releases and the optional battery USB hook

`tools/deploy_nicla_runtime.py --compact` leaves the recovery source copies on
the host and installs verified `.mpy` modules, the release loader and the two
manifests. It checks local source and bytecode hashes, the target ABI and FAT
allocation before activating. Keep the deployment receipt and its startup
and pointer backups. The previous release and any shared model root stay on
the board. A compact release requires the newer loader in `/flash/main.py`.

For a battery build, add `--include-offline-sightings` when invoking
`tools/build_nicla_mpy.py` and `--enable-battery-sightings` when invoking the
deployer. That opt-in reserves flash for 24 JPEG records, including FAT
allocation and headers, and installs the separately tracked
`firmware/battery_usb_boot.py` as `/flash/boot.py`. The hook requests CDC-only
USB with ID `2341:045F`, so a host cannot mount flash in that mode while the
runtime writes offline photos.
Connect the dashboard by serial to retrieve photos. Verify the target's
actual USB mode and serial discovery after reboot before arming offline
recording.

The deployment receipt records whether `/flash/boot.py` existed and its
SHA-256. The deployer preserves an existing comment-only file in the local
backup directory and under the reported `original_board_backup` path. A
custom executable boot script stops battery installation. The deployer uses
pyserial to follow the same Nicla USB serial number when the composite port
disappears. It verifies the new port's hook, backup and CDC-only mode before
activating the release. The receipt records the starting port and `port_used`
for the dashboard connection. Ambiguous or unverifiable USB readback leaves
the previous release pointer active and preserves the release and backups.

To undo the USB
hook, disarm offline recording and close the dashboard connection. Read back
`/flash/boot.py` over the explicit serial port and verify that it matches the
receipt's `battery_usb_boot.sha256`. Rename the hook to an unused backup name.
If the receipt records an original file, restore that exact backup as
`/flash/boot.py` and verify its SHA-256 before resetting. If the original was
absent, leave `/flash/boot.py` absent. This operation changes USB mode only.
Use the previous release pointer and loader backups for a firmware rollback.

## Primary references

- [Nicla Vision OpenMV reference](https://docs.openmv.io/v5.0.1/openmvcam/quickref/arduino-nicla-vision.html)
- [OpenMV 5.0.1 reset and boot sequence](https://docs.openmv.io/v5.0.1/reference/reset_boot.html)
- [OpenMV 5.0.1 MicroPython remote control](https://docs.openmv.io/v5.0.1/reference/mpremote.html)
- [Pinned camera API implementation](https://github.com/openmv/openmv/blob/v5.0.1/modules/py_csi_ng.c)
- [Pinned IMU API and configuration](https://github.com/openmv/openmv/blob/v5.0.1/modules/py_imu.c)
- [Official stock FOMO example](https://github.com/openmv/openmv/blob/v5.0.1/scripts/examples/03-Machine-Learning/00-TensorFlow/tf_object_detection.py)
- [Nicla ROMFS contents](https://github.com/openmv/openmv/blob/v5.0.1/boards/ARDUINO_NICLA_VISION/romfs_config.json)
- [Pinned Nicla memory configuration](https://github.com/openmv/openmv/blob/v5.0.1/boards/ARDUINO_NICLA_VISION/board_config.h)
- [Official Python host tool](https://github.com/openmv/openmv-python)

## Edge guards and compression matrix - 2026-10-22

The current recognize_embedding.py validates template schema, ASCII
pseudonyms, vector length/norm/finiteness, calibration thresholds, embedder
input/output contracts, finite cosine scores, crop bounds, and per-frame
failure statuses. enroll_identity.py validates enrollment request types and
limits and no longer starts enrollment when imported. No raw face or image is
stored by these flows.

Run the host edge cases from the ok-x20 root with
work/model-training/.venv/bin/python work/model-training/scripts/firmware_edge_case_tests.py.
Run the actual board probes with FaceRecognition/bin/nicla edge-probe PORT
and FaceRecognition/bin/nicla compression-matrix PORT. The board probes stage
short-lived /flash/qa_* files, verify cleanup, then reset back to the
standalone application. Read the full format matrix and board logs in
work/model-training/reports/compression-edge-case-report-20261022.md.
