# Local OpenMV recorder

`record_openmv.py` is a host helper for the first Nicla milestone. It uses the
native `openmv==1.0.7` package to run an explicitly selected MicroPython script,
record stdout, and optionally collect a bounded set of preview images. Nothing
connects to a camera or collects images until you run a command below.

Required host package: `openmv==1.0.7` (including its serial transport
dependencies). PNG capture additionally needs `Pillow`. Use the project Python
environment; do not install this script on the Nicla. OpenMV IDE and this helper
should not hold the same serial port at the same time.

## Commands, after OpenMV is installed

Run these from the repository root. `--list-ports` inventories ports without
connecting. Choose the actual Nicla port explicitly; the helper never chooses
one for you.

```sh
.venv-nicla/bin/python host/record_openmv.py --list-ports
```

Log the output of the camera's currently running script for 60 seconds without
saving images:

```sh
.venv-nicla/bin/python host/record_openmv.py \
  --port /dev/cu.usbmodemREPLACE \
  --session pilot_round_01 \
  --duration 60
```

To run a chosen script, add `--script path/to/your_script.py`. This interrupts
the previous script and uploads the selected script to RAM; it does not install
`main.py` or flash firmware. The helper stops its uploaded script when it exits.
Without `--script`, it leaves the pre-existing camera script running.

Image collection requires `--capture` and a declared `--person-id`. Run a
separate session for each subject/scene and give capture rounds distinct session
IDs. `--source-config` records JSON metadata but does not configure the camera.

```sh
.venv-nicla/bin/python host/record_openmv.py \
  --port /dev/cu.usbmodemREPLACE \
  --session person01_round_01 \
  --person-id person01 \
  --identity-role known \
  --lighting indoor_bright \
  --split unassigned \
  --script firmware/camera_imu.py \
  --source-config host/source_config.example.json \
  --capture --duration 60 --frame-interval 1 --max-frames 50
```

Use `--identity-role other` for people included in the classifier's other-class
training, `unknown` for people excluded from identity training, and `no_face`
with `--person-id none` for no-face scenes. These are supplied labels, not model
predictions. Change the example source configuration to describe the actual
script before using it. If the example is unchanged, its placeholder settings
will remain placeholders in the session record.

## What the recorder saves

By default, output is local `data/acquisition/SESSION/UTC_RUN_ID/`:

- `session.json`: declared subject, split, configuration, duration and limits.
- `device.json`: firmware version and system information reported by the camera.
- `executed_script.py`: source snapshot when `--script` was supplied.
- `stdout.log`: exact stdout chunks received from the device.
- `telemetry.jsonl`: lines that parse as JSON, tagged with host receipt time.
- `manifest.csv`: image paths and declared acquisition metadata.
- `images/*.png`: opt-in host-decoded preview captures.
- `transport_stats.json` (when the SDK responds before the deadline) and `summary.json`: acquisition status and counts.

The project's `data/` directory is intended to remain local and ignored by Git.
The helper rejects output under the public `docs/` website directory. Confirm
the ignore rules before adding files to Git; do not publish participants' images
as part of the project website.

The native SDK JPEG-compresses the preview on the camera, then `read_frame()`
returns decoded RGB888. Saving that result as PNG does **not** restore raw
sensor pixels or remove prior JPEG compression. Frame timestamps are host
receipt times and are not synchronized to the device stdout or sensor exposure.
Preview capture can affect the camera's timing and power. These images are a
derived pilot dataset, not proof of hardware performance or a final validated
training dataset. For a final exact-pixel acquisition path, capture the actual
inference input on-device and document its transform separately.
Use an acquisition script that streams clean images before drawing detection
boxes or other overlays; this helper cannot remove annotations already drawn
onto the camera's framebuffer.

Keep a complete capture round/session in one split. Use separate sessions for
training, validation/controller tuning, and final testing. For known identities,
hold out sessions; for unknown-person testing, also hold out identities where
possible. Use `dataset_manifest.example.csv` as the acquisition schema rather
than randomly splitting neighboring frames.

The project recorder was smoke-tested on the connected Nicla. Its first run
captured five camera/IMU rows without images but blocked past eight seconds. A
POSIX deadline now interrupts a blocked synchronous SDK call and lets the host
release the port. The follow-up eight-second run returned in 8.50 seconds with
five sample rows and no images. When the deadline interrupts the SDK, the final
`camera_imu_complete` row and `transport_stats.json` may be absent; use the
bounded raw-REPL board probes when complete device output is required. The
results are in [`reports/nicla-bringup.md`](../reports/nicla-bringup.md).

Official API reference:
[OpenMV Python API](https://docs.openmv.io/dev/openmvcam/tutorial/tools/openmv-py/api-reference.html).
