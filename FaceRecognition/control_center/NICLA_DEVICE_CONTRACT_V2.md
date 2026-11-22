# Nicla device UI contract v2

This contract describes the maintained loopback API between `control_center/server.py`, `web/device_ui.js`, `host/nicla_broker.py`, and firmware telemetry. It supplements the historical V1 replay contract; it does not rewrite V1 evidence.

## Status and event envelopes

`GET /api/v2/device/status` returns `schema_id: "nicla-device-status-v2"`. Important fields:

- `connected` and `device_ready` mean a current board hello/observation has been received. `serial_open` means a host descriptor is open; it is not enough to claim the board is ready.
- `device`, `serial_descriptor`, `boot_id`, `latest_frame`, `last_observation`, `applied_state`, `capabilities`, `controller_authority`, `host_last_receive_age_ms`, and byte/parser counters report only broker-observed state.
- `latest_frame` contains `boot_id`, monotonically increasing per-boot `frame_id`, `width`, `height`, `orientation`, `age_ms`, `stale`, `encoding`, `frame_ref`, and `payload_ready`. Unknown fields stay null or unavailable.
- `applied_state` reports board-acknowledged state (`running`, `sleeping`, `policy_id`, `model_override` plus UI aliases). A request is not applied state.

Each completed inference is a `nicla-observation-v2` event. Its identity key is `(boot_id, frame_id)`; the same pair must appear in the frame, detection, crop, detector input, identity input, decision, and command completion whenever those stages ran. Missing IDs mean unverified linkage and cannot be repaired from a newer preview. `type: "observation_complete"` is immutable evidence; the separately polled latest preview can advance while the last result remains frozen.

The observation includes board `ticks_ms` and `ticks_modulus_ms` where supplied, plus broker `host_received_utc` and `host_received_monotonic_ns`. Board ticks restart/wrap independently of host clocks; `boot_id` is the session boundary. A GET does not manufacture an acquisition timestamp. The displayed JPEG is the JPEG encoding of the same captured framebuffer and carries its own encoding label and SHA-256.

Detector `regions_xywh` preserve `coordinate_space` and `region_semantics`. FOMO centroids/heatmap regions are not automatically face boxes. `crop.rect_px` is the applied crop; a requested rectangle and a rejection reason may also be reported. Identity input metadata includes exact shape, channels, dtype, quantization, and SHA-256 when known. Binary tensor bytes are retrieved only by matching boot/frame/kind from `/api/v2/device/input`; JPEGs use `/api/v2/device/frame`. Both responses include matching identity headers and payload SHA-256. If a payload was not retained, the UI says so.

## Controller and commands

The serial port has one process owner. The `NiclaBroker` constructor is disconnected; the UI may open USB only after an explicit authorized Connect. Viewer control uses a 15 second lease:

1. `POST /api/v2/device/controller/claim` body `{ "viewer_id": "<X-Viewer-ID>" }` returns a `controller_id` only to the lease owner.
2. `POST /api/v2/device/connect` body `{ "schema": "nicla-connect-v2", "viewer_id": "<X-Viewer-ID>", "controller_id": "<returned lease ID>" }` renews and validates that exact viewer lease before opening serial. Disconnect validates the same way. A read-only viewer cannot open or close serial.
3. `POST /api/v2/device/commands` body `{ "schema": "nicla-command-v2", "viewer_id": "...", "controller_id": "...", "command_id": "...", "cmd": "...", ... }` accepts only advertised commands and valid IDs.
4. `POST /api/v2/device/controller/heartbeat` renews the lease; `/controller/release` releases only authority. A browser closing or releasing authority never sends a device stop command.
5. `POST /api/v2/device/disconnect` closes only the host serial handle and leaves device processing state unchanged.

Wire command names are `one_shot`, `start`, `stop`, `set_model` (`model_slot`), `set_policy` (`policy_id`), `sleep`, `wake`, and `trial`, plus broker discovery/status/configuration commands. A bounded `trial` requires a unique `trial_id`, `frames`, and `duration_ms` within firmware-advertised limits. Optional `measurement_stage_id`, `model_slot`, and `policy_id` are validated against firmware capabilities; the firmware returns an ACK for the selected/applied values and later emits `trial_finished` with its ID, stage, and stop reason. A duplicate trial ID is rejected without restarting. `trial_finished` updates host applied `running:false`; a requested bound is not treated as completion until that event or a matching stop ACK arrives. Unsupported actions remain disabled with the capability reason.

HTTP command response `accepted` means queued/written, not completed. A matching `type: "command_ack"` event echoes `command_id`, `command`, `control_status`, `reason_code`, and actual completion details such as `completed_frame_id`. Terminal ACK values are `completed`, `error`, `unsupported`, `rejected`, and `timeout`; `accepted` is nonterminal. Duplicate IDs are not resent. Requested state changes only when applied-state telemetry/ACK confirms it.

## Bounded payload transport

Large values use `NVPK1` framing: fixed magic, bounded little-endian header length, JSON header (`schema_id: "nicla-packet-v1"`, kind, boot/frame IDs, byte count, SHA-256, tensor metadata), and exact payload bytes. Current packet kinds are `frame_jpeg`, `detector_input`, and `identity_input`. Parser and broker event/frame rings are bounded. Frame/input GETs reject mismatched IDs, invalid image/container, and oversized payloads. A local session export caps the complete JSON download at 10,000,000 UTF-8 bytes and records any omitted payloads.

Missing power, unsupported IMU fields, unretained model outputs, and unavailable tensor bytes remain explicitly unavailable. No score is labeled a probability unless the observation contract states that semantics. Fixture events have `source: "test_fixture"`, generated seeds, and synthetic payload provenance; they are never device evidence.
