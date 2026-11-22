# K optical-presentation and independent-annotation protocol

**Status:** source-side protocol prepared; no final-role rows or physical session was opened. The final plan remains sealed until a development-only freeze receipt exists.

## What the evidence means

The camera will observe public face images shown on a display in its field of view. Report this as an optical-presentation prototype test. It does not establish recognition of the real people pictured. Use only the approved public-image manifests and their role labels. Personal photo collections are not part of this procedure.

## Setup and movement script

1. Use the Nicla Vision and firmware/configuration named by the verified H receipt. Mount the camera at a fixed, recorded orientation. Measure camera-to-display distance with a tape measure or a marked rigid jig; record the measurement method and orientation in the annotation ledger. The player records its requested range, but the independent annotation records the measured physical range.
2. Use the local fullscreen player on a separate display. Keep display brightness, size, position, and room lighting fixed within a block. Record the visible lighting condition in plain language; do not call image brightness a lux measurement. A lux meter reading may be recorded separately if one is available.
3. For stationary known and unknown arrival trials, leave the camera stationary and start from an empty view. Move the displayed target from outside the camera field of view into the center, then through the predeclared off-center position; keep it visible for the planned hold interval, then remove it. Record actual appearance, first usable interval, and disappearance from synchronized optical video or a witnessed live marker.
4. For blur/motion trials, keep the source image and display settings fixed while the operator moves the camera through the scripted slow walk/turn and then holds still. Record the direction and approximate motion segment. Do not use the controller's motion report to label ground truth.
5. For multiple-face trials, show the scheduled ambiguous pair at the same time. Annotate the observed count. For empty trials, keep the field clear throughout the interval. If the expected target is not visible to the camera, preserve the trial and record a visibility-failure reason; do not drop it or invent an appearance time.
6. Repeat the complete ten-block schedule with the separately generated physical-block seeds and randomized order. A repeated image or frame is not a new person or independent identity. Keep session IDs and physical-block IDs distinct from training-root seeds.

## Wiring and measured boundary

Before any measured session, use the exact G-qualified power-boundary diagram and H-qualified Nicla telemetry route. The meter must sit in the single downstream board supply path, at nominal 5 V; its PC/HID connection is a separate instrument connection and is not a board serial bridge. Do not add a second power source or connect USB VBUS in parallel. If a qualified 3.3 V UART adapter is used, connect signal, ground, and the verified pins only; leave adapter VCC unconnected. Keep the data path and power path exactly as captured in the G/H receipts. The physical meter identity and port labels must be read from the actual unit before this section can name exact ports; the current packet cannot identify them reliably.

## Independent event ground truth

Create an annotation template from each stimulus-event log, then annotate it without viewing Nicla/controller outputs. Use `synchronized_optical_video` or `witnessed_live_marker` for appearance timing, with the offset to controller monotonic time and its uncertainty. Record visible face count, whether the target was actually visible, appearance, first usable interval, disappearance, measured range, camera orientation, visible lighting, and wearer motion. Use a pseudonymous observer ID and record that the observer did not see controller results before annotation.

The stimulus start/end markers bound when the player rendered a source image; they do not by themselves prove that the face was visible to the Nicla camera. `annotation_workflow.py` checks row joins and clock bounds but cannot determine visibility from the controller output or prove that a human annotation is correct.

## Source-side commands

Run from the workspace root after the appropriate physical session has produced its event log:

```sh
PYTHONPATH=FaceRecognition python3 -m nicla_study.annotation_workflow template \
  --event-log work/project-finish-v4/runs/presentation/<RUN_ID>/stimulus-events.jsonl \
  --output work/project-finish-v4/runs/presentation/<RUN_ID>/annotations.jsonl

PYTHONPATH=FaceRecognition python3 -m nicla_study.annotation_workflow validate \
  --event-log work/project-finish-v4/runs/presentation/<RUN_ID>/stimulus-events.jsonl \
  --annotations work/project-finish-v4/runs/presentation/<RUN_ID>/annotations.jsonl \
  --output-json work/project-finish-v4/runs/presentation/<RUN_ID>/annotation-validation.json
```

For final-role sessions, provide the verified `handoffs/J/FROZEN_SELECTION.json` as `--freeze-receipt` to both commands. The tool refuses a final event log without a sealed development-only receipt. It never opens the stimulus manifests itself.

## Physical execution gate

This protocol is not authorization to start K. K remains blocked until W1 is completed, Windows G qualifies the actual meter and timing tier, exact-current-firmware H passes, I selection and J pilot are complete, and the final model/controller hashes have been frozen. At that point replace the generic port/topology reference with the verified G/H wiring record; do not guess the instrument's model from VID/PID.
