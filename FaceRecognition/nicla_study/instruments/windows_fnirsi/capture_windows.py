"""Bounded native-Windows FNIRSI HID capture with fresh interface selection.

Raw payloads are stored before interpretation by the inherited parser. This is a
Nano diagnostic reader; valid model-energy claims require separate evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

ADAPTER_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ADAPTER_DIR / "upstream" / "scripts"
PROJECT_DEFAULT = ADAPTER_DIR.parents[3]
POWER_SCRIPTS = ADAPTER_DIR / "upstream" / "power"
APP_ROOT = ADAPTER_DIR.parents[2]
HOST_DIR = ADAPTER_DIR.parents[2] / "host"
sys.path.insert(0, str(POWER_SCRIPTS))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(HOST_DIR))
sys.path.insert(0, str(APP_ROOT))

from device_selector_windows import choose_device, describe
from hardware_lock_windows import HardwareLock, default_lock_path
from fnirsi_protocol import PID, VID, make_command
from run_io import RunWriter
from nano_serial_capture_windows import NanoSerialCapture
from nicla_protocol import NiclaRawSerialCapture
from nicla_study.instruments.windows_fnirsi.instrument_adapter import verify_active_firmware_binding
from nicla_study.instruments.windows_fnirsi import receipt_clock


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_dashboard_preflight(url, boot_id, firmware_sha256, config_sha256):
    endpoint = urlsplit(url)
    if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or endpoint.port is None
            or endpoint.username is not None or endpoint.password is not None
            or endpoint.path not in ("", "/") or endpoint.query or endpoint.fragment):
        raise ValueError("use the dashboard's local URL, e.g. http://127.0.0.1:8780")
    conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=5)
    try:
        conn.request("GET", "/api/v2/device/status")
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"dashboard status returned HTTP {response.status}")
        raw = response.read(1024 * 1024 + 1)
    finally:
        conn.close()
    if len(raw) > 1024 * 1024:
        raise ValueError("dashboard status exceeded the receipt size limit")
    status = json.loads(raw)
    if not isinstance(status, dict) or status.get("schema_id") != "nicla-device-status-v2":
        raise ValueError("dashboard did not return Nicla device status v2")
    if not all(status.get(key) is True for key in ("connected", "device_ready", "serial_open")):
        raise ValueError("connect the dashboard to the live Nicla before HID-only capture")
    if status.get("running") is not False or status.get("sleeping") is not False:
        raise ValueError("leave the dashboard's Nicla stopped with the camera awake for preflight")
    age_ms = status.get("host_last_receive_age_ms")
    if type(age_ms) is not int or not 0 <= age_ms <= 8000:
        raise ValueError("dashboard has no recent device telemetry")
    if status.get("boot_id") != boot_id:
        raise ValueError("dashboard boot ID differs from --device-reset-id")
    device = status.get("device")
    if (not isinstance(device, dict) or device.get("firmware_sha256") != firmware_sha256.lower()
            or device.get("config_sha256") != config_sha256.lower()):
        raise ValueError("live dashboard firmware/config hashes differ from the declared deployment")
    return {"schema": "nicla-fnirsi-dashboard-preflight-v1", "route": url.rstrip("/") + "/api/v2/device/status",
            "observed_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "boot_id": boot_id, "firmware_sha256": device["firmware_sha256"],
            "config_sha256": device["config_sha256"], "host_last_receive_age_ms": age_ms,
            "running": False, "sleeping": False, "serial_opened_by_logger": False,
            "status_response_sha256": hashlib.sha256(raw).hexdigest()}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    p.add_argument("--list", action="store_true", help="list this meter's fresh HID interfaces; no device is opened")
    p.add_argument("--interface-number", type=int,
                   help="required explicit interface selection for reads (current FNB-C2 candidate: 2)")
    p.add_argument("--path-fragment", help="optional private path discriminator if an interface number is duplicated")
    p.add_argument("--duration", type=float, default=30.0, help="capture seconds (0 < duration <= 600)")
    p.add_argument("--output-dir", type=Path, help="new immutable run directory; default project raw/run-id")
    p.add_argument("--run-id", default="windows-%s" % datetime.now().strftime("%Y%m%dT%H%M%S"))
    p.add_argument("--rerun-of", help="original immutable run ID when this is a linked rerun")
    p.add_argument("--experiment-id", default="windows-fnirsi-nano-diagnostic")
    p.add_argument("--scenario-id", default="meter-in-path-pending-validation")
    p.add_argument("--root-seed", type=int)
    p.add_argument("--meter-label", default="FNB-C2 as observed in official app; HID descriptor string FNB-58")
    p.add_argument("--meter-firmware", default="not yet read from device")
    p.add_argument("--topology", default="not verified; record physical source and port path")
    p.add_argument("--load-description", default="selected load; exact physical boundary must be recorded")
    p.add_argument("--board-type", choices=["nicla_vision", "nano33ble_sense_rev2", "nano_esp32", "passive_load", "other"])
    p.add_argument("--board-label", help="human-verified board label from the fresh device inventory")
    p.add_argument("--device-reset-id", help="observed reset/session identifier; never inferred from a seed")
    p.add_argument("--firmware-source", type=Path, help="selected board firmware source file")
    p.add_argument("--release-payload", "--firmware-binary", dest="firmware_binary", type=Path,
                   help="exact versioned board release payload file; --firmware-binary is a legacy alias")
    p.add_argument("--board-configuration", type=Path, help="selected board configuration file")
    p.add_argument("--firmware-version", help="version read from the selected board or build receipt")
    nicla_routes = p.add_mutually_exclusive_group()
    nicla_routes.add_argument("--serial-port", help="port freshly identified from the current serial inventory")
    nicla_routes.add_argument("--nicla-dashboard-url",
                   help="local Nicla dashboard URL for HID-only capture with live release preflight")
    p.add_argument("--serial-kind", choices=("nano", "nicla"), default="nano",
                   help="select the actual active board protocol; the meter COM is never a board port")
    p.add_argument("--serial-dtr", action="store_true", help="assert CDC DTR on the Nano COM port; may reset the board")
    p.add_argument("--serial-rts", action="store_true", help="assert CDC RTS on the Nano COM port; may reset the board")
    p.add_argument("--nicla-dtr", action="store_true", help="assert Nicla USB CDC DTR only when the current measured mode requires it; may reset board")
    p.add_argument("--nicla-rts", action="store_true", help="assert Nicla USB CDC RTS only when the current measured mode requires it")
    p.add_argument("--policy-id", choices=("H", "L", "S", "G", "A-sense", "A-model",
                                           "A-risk", "A-cost", "A-full", "T", "U"))
    p.add_argument("--model-id", help="actual selected model/model-root identifier for the block")
    p.add_argument("--physical-block-seed", type=int)
    p.add_argument("--training-root-seed", type=int)
    p.add_argument("--session-id")
    p.add_argument("--condition")
    p.add_argument("--input-manifest-sha256")
    p.add_argument("--expected-device-firmware-sha256")
    p.add_argument("--expected-device-config-sha256")
    p.add_argument("--expected-firmware-source-sha256")
    p.add_argument("--expected-release-payload-sha256")
    p.add_argument("--nicla-stage-id", choices=("full_pipeline", "idle", "camera_capture_only",
        "detector_only", "identity_center_crop_diagnostic", "identity_small_only",
        "identity_large_only", "cold_model_load", "warm_inference", "model_swap_and_gc"),
        default="full_pipeline")
    p.add_argument("--nicla-model-slot", choices=("small", "large"), default="small")
    p.add_argument("--nicla-preload-model", choices=("small", "large"),
                   help="preload a model before the measured window; used only by warm-inference stages")
    p.add_argument("--warmup-seconds", type=float, default=5.0)
    p.add_argument("--stimulus-block", choices=("presentation", "development", "final"))
    p.add_argument("--stimulus-condition", choices=("known", "unknown", "empty", "ambiguous", "mixed"))
    p.add_argument("--stimulus-count", type=int, default=10)
    p.add_argument("--stimulus-seed", type=int)
    p.add_argument("--stimulus-session-id")
    p.add_argument("--stimulus-event-log", type=Path)
    p.add_argument("--stimulus-freeze-receipt", type=Path)
    p.add_argument("--stimulus-lighting-state", default="not_recorded")
    p.add_argument("--stimulus-motion-state", default="not_recorded")
    p.add_argument("--stimulus-range-cm", type=float, default=50.0)
    p.add_argument("--stimulus-hold-seconds", type=float, default=1.5)
    p.add_argument("--stimulus-blank-seconds", type=float, default=1.0)
    p.add_argument("--lock-timeout", type=float, default=0.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", args.run_id):
        build_parser().error("--run-id permits letters, numbers, dot, underscore, and hyphen only")
    if not args.list and (args.interface_number is None or not 0 < args.duration <= 600):
        build_parser().error("reads need --interface-number and --duration in (0, 600]")
    if not args.list:
        if not args.board_type or not args.board_label or not args.device_reset_id:
            build_parser().error("capture requires freshly verified --board-type, --board-label, and --device-reset-id")
        if args.board_type == "passive_load":
            if args.serial_port or args.firmware_source or args.firmware_binary or args.board_configuration:
                build_parser().error("passive_load cannot claim firmware or board serial metadata")
        elif args.board_type == "nicla_vision":
            if not (args.firmware_binary and args.board_configuration):
                build_parser().error("Nicla capture requires exact release-payload and board-configuration files")
        elif not (args.firmware_source and args.firmware_binary and args.board_configuration):
            build_parser().error("an active board capture requires firmware source, exact release payload, and board configuration paths")
    if args.serial_port and args.serial_kind == "nano" and args.root_seed is None:
        build_parser().error("Nano --serial-port requires --root-seed for the recorded commissioning workload")
    if not args.serial_port and (args.serial_dtr or args.serial_rts):
        build_parser().error("--serial-dtr/--serial-rts require --serial-port")
    if args.serial_kind == "nicla" and (args.serial_dtr or args.serial_rts):
        build_parser().error("use --nicla-dtr/--nicla-rts for the explicitly qualified Nicla control mode")
    if (args.nicla_dtr or args.nicla_rts) and args.serial_kind != "nicla":
        build_parser().error("--nicla-dtr/--nicla-rts require --serial-kind nicla")
    if args.serial_kind == "nicla" and args.serial_port is None:
        build_parser().error("Nicla serial mode requires --serial-port")
    direct_nicla = bool(args.serial_port and args.serial_kind == "nicla")
    if args.nicla_dashboard_url and (args.board_type != "nicla_vision" or args.list):
        build_parser().error("--nicla-dashboard-url is only for an active Nicla Vision HID-only capture")
    if args.board_type == "nicla_vision" and not args.list and not (direct_nicla or args.nicla_dashboard_url):
        build_parser().error("Nicla capture requires either --serial-kind nicla with --serial-port, or --nicla-dashboard-url")
    if direct_nicla:
        if args.board_type != "nicla_vision":
            build_parser().error("Nicla telemetry requires --board-type nicla_vision")
        if args.physical_block_seed is None or args.policy_id is None or not args.model_id:
            build_parser().error("Nicla capture requires physical-block seed, policy ID, and model ID")
        if args.physical_block_seed < 0 or args.physical_block_seed > 0xFFFFFFFF:
            build_parser().error("--physical-block-seed must be an unsigned 32-bit integer")
        if args.training_root_seed is not None and not 0 <= args.training_root_seed <= 0xFFFFFFFF:
            build_parser().error("--training-root-seed must be an unsigned 32-bit integer")
    if not 0 <= args.warmup_seconds <= 60:
        build_parser().error("--warmup-seconds must be in [0,60]")
    stimulus_requested = any((args.stimulus_block, args.stimulus_condition,
                              args.stimulus_session_id, args.stimulus_event_log))
    if stimulus_requested:
        if not all((args.stimulus_block, args.stimulus_condition,
                    args.stimulus_session_id, args.stimulus_event_log)):
            build_parser().error("stimulus capture requires block, condition, session ID, and event log")
        if args.board_type != "nicla_vision":
            build_parser().error("optical stimulus playback is only supported for an actual Nicla Vision session")
        if args.stimulus_block == "final" and not args.stimulus_freeze_receipt:
            build_parser().error("final stimulus playback requires the frozen selection receipt")
        if args.stimulus_count < 1 or args.stimulus_count > 100:
            build_parser().error("stimulus count must be in [1,100]")
        if args.stimulus_seed is None or args.stimulus_seed < 0 or args.stimulus_seed > 0xFFFFFFFF:
            build_parser().error("stimulus seed must be supplied as an unsigned 32-bit value")
        if args.stimulus_hold_seconds <= 0 or args.stimulus_blank_seconds < 0:
            build_parser().error("stimulus hold must be positive and blank duration nonnegative")
        required_stimulus_seconds = args.stimulus_count * (args.stimulus_hold_seconds + args.stimulus_blank_seconds)
        if required_stimulus_seconds > args.duration:
            build_parser().error("measurement duration must cover every planned stimulus hold/blank interval")
        if args.stimulus_range_cm <= 0:
            build_parser().error("stimulus range must be positive")
    if args.root_seed is not None and not 0 <= args.root_seed <= 0xFFFFFFFF:
        build_parser().error("--root-seed must be an unsigned 32-bit integer")
    if args.lock_timeout < 0:
        build_parser().error("--lock-timeout must be nonnegative")
    root = args.project_root.resolve()
    if args.board_type == "nicla_vision" and not args.list:
        for field, value in (("--expected-device-firmware-sha256", args.expected_device_firmware_sha256),
                             ("--expected-device-config-sha256", args.expected_device_config_sha256),
                             ("--expected-firmware-source-sha256", args.expected_firmware_source_sha256),
                             ("--expected-release-payload-sha256", args.expected_release_payload_sha256)):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                build_parser().error(field + " must be supplied from the live deployment receipt")
        try:
            nicla_release_binding = verify_active_firmware_binding(
                "nicla_vision",
                (args.firmware_source if args.firmware_source.is_absolute() else root / args.firmware_source)
                if args.firmware_source else None,
                args.firmware_binary if args.firmware_binary.is_absolute() else root / args.firmware_binary,
                args.board_configuration if args.board_configuration.is_absolute() else root / args.board_configuration,
                expected_source_sha256=args.expected_firmware_source_sha256,
                expected_release_payload_sha256=args.expected_release_payload_sha256,
                expected_configuration_sha256=args.expected_device_config_sha256,
            )
        except (OSError, ValueError) as exc:
            build_parser().error("selected Nicla release files failed hash binding: " + str(exc))
    power_scripts = POWER_SCRIPTS.resolve()
    if not (power_scripts / "fnirsi_protocol.py").is_file():
        print("maintained adapter does not contain the FNIRSI parser: %s" % power_scripts, file=sys.stderr)
        return 2

    try:
        import hid
    except ImportError as exc:
        print("import hid failed; activate the local venv and install requirements-windows.txt: %s" % exc,
              file=sys.stderr)
        return 2

    lock_path = default_lock_path(root)
    try:
        with HardwareLock(lock_path, args.run_id, args.lock_timeout):
            devices = hid.enumerate(VID, PID)
            descriptors = [describe(d, include_path=args.list) for d in devices]
            if args.list:
                print(json.dumps({"descriptor_only": True, "enumerated_utc": datetime.now(timezone.utc).isoformat(),
                                  "devices": descriptors}, indent=2))
                return 0
            try:
                selected = choose_device(devices, args.interface_number, args.path_fragment)
            except ValueError as exc:
                print("%s\nFresh devices: %s" % (exc, json.dumps(descriptors, indent=2)), file=sys.stderr)
                return 3

            dashboard_preflight = None
            if args.nicla_dashboard_url:
                try:
                    dashboard_preflight = read_dashboard_preflight(args.nicla_dashboard_url, args.device_reset_id,
                        args.expected_device_firmware_sha256, args.expected_device_config_sha256)
                except (OSError, http.client.HTTPException, ValueError) as exc:
                    print(f"Dashboard preflight failed before opening the meter: {exc}", file=sys.stderr)
                    return 4

            selected_summary = describe(selected)
            run_id = args.run_id
            out_dir = args.output_dir or (root / "raw" / run_id)
            source_paths = {
                "windows_logger_sha256": Path(__file__).resolve(),
                "receipt_clock_sha256": Path(receipt_clock.__file__),
                "windows_lock_sha256": SCRIPT_DIR / "hardware_lock_windows.py",
                "device_selector_sha256": SCRIPT_DIR / "device_selector_windows.py",
                "serial_capture_sha256": SCRIPT_DIR / "nano_serial_capture_windows.py",
                "protocol_sha256": power_scripts / "fnirsi_protocol.py",
                "run_writer_sha256": power_scripts / "run_io.py",
            }
            board_files = {
                "selected_firmware_source_sha256": args.firmware_source,
                "selected_release_payload_sha256": args.firmware_binary,
                "selected_board_configuration_sha256": args.board_configuration,
            }
            for key, value in board_files.items():
                if value is not None:
                    path = value if value.is_absolute() else root / value
                    path = path.resolve()
                    if not path.is_file():
                        build_parser().error("selected board provenance file does not exist: " + str(path))
                    source_paths[key] = path
            artifact_hashes = {key: sha256(path) for key, path in source_paths.items() if path.is_file()}
            if artifact_hashes.get("selected_release_payload_sha256"):
                # Preserve the historical field for readers of old run metadata.
                # It aliases the selected release payload. For Nicla this is the
                # versioned interpreted-runtime bundle; for Nano it may be a HEX.
                artifact_hashes["selected_firmware_binary_sha256"] = artifact_hashes[
                    "selected_release_payload_sha256"]
            release_payload_role = None if args.board_type == "passive_load" else (
                "exact versioned interpreted Nicla runtime release payload"
                if args.board_type == "nicla_vision" else
                "exact selected release payload; may be a compiled firmware binary for this board"
            )
            legacy_binary_semantics = None if args.board_type == "passive_load" else (
                "legacy compatibility alias of release_payload_sha256; not a compiled MCU firmware binary"
                if args.board_type == "nicla_vision" else
                "legacy compatibility alias of release_payload_sha256"
            )
            if args.input_manifest_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.input_manifest_sha256):
                build_parser().error("--input-manifest-sha256 must be 64 hex characters")
            metadata = {
                "experiment_id": args.experiment_id,
                "run_id": run_id,
                "rerun_of": args.rerun_of,
                "policy_id": args.policy_id or "diagnostic-only",
                "scenario_id": args.scenario_id,
                "root_seed": args.root_seed,
                "physical_block_seed": args.physical_block_seed,
                "training_root_seed": args.training_root_seed,
                "session_id": args.session_id,
                "condition": args.condition,
                "model_id": args.model_id,
                "input_manifest_sha256": args.input_manifest_sha256,
                "selected_board": None if args.board_type == "passive_load" else {
                    "board_type": args.board_type, "board_label": args.board_label,
                    "reset_id": args.device_reset_id, "firmware_version": args.firmware_version,
                    "firmware_source_sha256": (nicla_release_binding["firmware_source_sha256"]
                        if args.board_type == "nicla_vision" else artifact_hashes.get("selected_firmware_source_sha256")),
                    "firmware_source_file_sha256": (nicla_release_binding["firmware_source_file_sha256"]
                        if args.board_type == "nicla_vision" else artifact_hashes.get("selected_firmware_source_sha256")),
                    "firmware_source_file_available": (nicla_release_binding["firmware_source_file_available"]
                        if args.board_type == "nicla_vision" else bool(artifact_hashes.get("selected_firmware_source_sha256"))),
                    "firmware_source_hash_origin": (nicla_release_binding["firmware_source_hash_origin"]
                        if args.board_type == "nicla_vision" else "local_source_file"),
                    "release_payload_sha256": artifact_hashes.get("selected_release_payload_sha256"),
                    "firmware_binary_sha256": artifact_hashes.get("selected_release_payload_sha256"),
                    "firmware_binary_sha256_semantics": legacy_binary_semantics,
                    "configuration_sha256": artifact_hashes.get("selected_board_configuration_sha256"),
                },
                "board_provenance_status": "not_applicable_passive_load" if args.board_type == "passive_load" else "bound_to_firmware_source_release_payload_and_configuration",
                "release_payload_sha256": artifact_hashes.get("selected_release_payload_sha256"),
                "release_payload_role": release_payload_role,
                "meter_vid_pid": "%04x:%04x" % (VID, PID),
                "meter_model_reported": args.meter_label,
                "meter_firmware": args.meter_firmware,
                "meter_hid_descriptor": selected_summary,
                "fresh_interface_selection": True,
                "topology": args.topology,
                "load_description": args.load_description,
                "source_input_paths_sha256": artifact_hashes,
                "host": {"platform": platform.platform(), "python": sys.version,
                         "python_bits": 64 if sys.maxsize > 2**32 else 32,
                         "hidapi_version": importlib.metadata.version("hidapi"),
                         "pyserial_version": importlib.metadata.version("pyserial")},
                "duration_requested_s": args.duration,
                "read_size_bytes": 64,
                "hid_write_report_id_prefix": 0,
                "measurement_claim": "diagnostic only until instrument timing/decoder, powered path, selected board identity, firmware/config binding, and run QC pass",
                "acquisition_time_source": "host monotonic at HID read completion; no device timestamps assigned",
                "host_utc_clock": receipt_clock.UTC_SOURCE,
            }
            if dashboard_preflight:
                metadata["nicla_dashboard_preflight"] = dashboard_preflight
                metadata["nicla_dashboard_preflight_file"] = "nicla_dashboard_preflight.json"
            writer = None
            nano_serial = None
            nicla_serial = None
            presentation_process = None
            presentation_command = None
            device = hid.device()
            try:
                writer = RunWriter(out_dir, metadata)
                if dashboard_preflight:
                    (out_dir / metadata["nicla_dashboard_preflight_file"]).write_text(
                        json.dumps(dashboard_preflight, indent=2) + "\n", encoding="utf-8")
                writer.event("meter_open_attempt", {"interface_number": selected_summary["interface_number"],
                                                     "usage_page": selected_summary["usage_page"],
                                                     "usage": selected_summary["usage"],
                                                     "path_sha256": selected_summary["enumerated_path_sha256"]})
                device.open_path(selected["path"])
                writer.event("meter_open", {"selected_interface": selected_summary,
                                            "protocol_commands": ["0x81 init", "0x82 read"],
                                            "protocol_source": "inherited commands; HID report conventions recorded during live capture"})
                if args.serial_port and args.serial_kind == "nano":
                    writer.event("nano_serial_open_attempt", {"port_requested": args.serial_port,
                                                               "baudrate": 115200,
                                                               "root_seed": args.root_seed})
                    nano_serial = NanoSerialCapture(out_dir, args.serial_port, run_id, args.root_seed,
                                                    dtr=args.serial_dtr, rts=args.serial_rts)
                    writer.metadata["nano_serial_descriptor"] = nano_serial.descriptor
                    writer.metadata["nano_serial_log_file"] = "nano_serial.jsonl"
                    writer.metadata["nano_serial_baudrate"] = 115200
                    writer.metadata["nano_serial_dtr_requested"] = bool(args.serial_dtr)
                    writer.metadata["nano_serial_rts_requested"] = bool(args.serial_rts)
                    writer.metadata["nano_serial_dtr_may_reset_board"] = True
                    writer.write_metadata()
                    writer.event("nano_serial_open", {"port": nano_serial.descriptor,
                                                       "log_file": "nano_serial.jsonl",
                                                       "dtr_requested": bool(args.serial_dtr),
                                                       "rts_requested": bool(args.serial_rts),
                                                       "dtr_rts_may_reset_board": True})
                elif args.serial_port and args.serial_kind == "nicla":
                    writer.event("nicla_serial_open_attempt", {"port_requested": args.serial_port,
                        "baudrate": 115200, "board_label": args.board_label,
                        "policy_id": args.policy_id, "model_id": args.model_id})
                    nicla_serial = NiclaRawSerialCapture(args.serial_port, out_dir, run_id,
                        baudrate=115200, dtr=args.nicla_dtr, rts=args.nicla_rts,
                        expected_firmware_sha256=args.expected_device_firmware_sha256,
                        expected_config_sha256=args.expected_device_config_sha256)
                    writer.metadata["nicla_serial"] = nicla_serial.metadata
                    writer.metadata["expected_device_firmware_sha256"] = args.expected_device_firmware_sha256.lower()
                    writer.metadata["expected_device_config_sha256"] = args.expected_device_config_sha256.lower()
                    writer.write_metadata()
                    identity_ack = nicla_serial.command({"cmd": "identity"}, wait_seconds=5)
                    config_ack = nicla_serial.command({"cmd": "config"}, wait_seconds=5)
                    ping_ack = nicla_serial.command({"cmd": "ping"}, wait_seconds=5)
                    identity_fields = identity_ack
                    if identity_fields.get("device") != "Arduino Nicla Vision / OpenMV CSI":
                        raise ValueError("selected serial interface did not identify as the Nicla Vision")
                    observed_firmware = identity_fields.get("firmware_sha256")
                    observed_config = identity_fields.get("config_sha256")
                    if observed_firmware != args.expected_device_firmware_sha256.lower():
                        raise ValueError("live Nicla firmware hash differs from deployment receipt")
                    if observed_config != args.expected_device_config_sha256.lower():
                        raise ValueError("live Nicla configuration hash differs from deployment receipt")
                    config_fields = config_ack
                    if args.policy_id not in config_fields.get("policy_ids", []):
                        raise ValueError("live Nicla does not advertise requested policy: " + str(args.policy_id))
                    if args.nicla_stage_id in ("identity_center_crop_diagnostic",
                                                "identity_small_only", "identity_large_only",
                                                "cold_model_load", "warm_inference"):
                        allowed_model_ids = {config_fields.get(args.nicla_model_slot + "_model_id")}
                    elif args.nicla_stage_id == "model_swap_and_gc":
                        allowed_model_ids = {"%s+%s" % (config_fields.get("small_model_id"),
                                                        config_fields.get("large_model_id"))}
                    elif args.policy_id in ("H", "L", "G", "A-sense"):
                        allowed_model_ids = {config_fields.get("large_model_id")}
                    elif args.policy_id in ("S", "T", "U"):
                        allowed_model_ids = {config_fields.get("small_model_id")}
                    else:
                        allowed_model_ids = {config_fields.get("small_model_id"),
                                             config_fields.get("large_model_id"),
                                             "%s+%s" % (config_fields.get("small_model_id"),
                                                         config_fields.get("large_model_id"))}
                    if args.model_id not in allowed_model_ids:
                        raise ValueError("requested model ID is not the one configured for the live policy")
                    if args.nicla_stage_id not in config_fields.get("measurement_stage_ids", []):
                        raise ValueError("live Nicla does not advertise measurement stage: " + args.nicla_stage_id)
                    if not ping_ack.get("pong"):
                        raise ValueError("Nicla ping did not return pong")
                    pre_stage_stop_ack = nicla_serial.command({"cmd": "stop"}, wait_seconds=5)
                    if pre_stage_stop_ack.get("control_status", "ok") != "ok":
                        raise ValueError("Nicla did not stop autonomous work before the measured stage")
                    preload_ack = None
                    if args.nicla_preload_model:
                        if args.nicla_stage_id != "warm_inference" or args.nicla_preload_model != args.nicla_model_slot:
                            raise ValueError("model preload is only valid for the matching warm-inference stage")
                        preload_ack = nicla_serial.command({"cmd": "preload_model",
                            "model_slot": args.nicla_preload_model}, wait_seconds=10)
                        if preload_ack.get("control_status", "ok") != "ok":
                            raise ValueError("Nicla model preload failed")
                    seed_id = "B%08X" % args.physical_block_seed
                    seed_ack = nicla_serial.command({"cmd": "seed", "seed_id": seed_id,
                        "host_marker": run_id}, wait_seconds=5)
                    if seed_ack.get("control_status", "ok") != "ok":
                        raise ValueError("Nicla rejected physical block seed")
                    start_ack = nicla_serial.command({"cmd": "start", "policy_id": args.policy_id,
                        "measurement_stage_id": args.nicla_stage_id,
                        "model_slot": args.nicla_model_slot}, wait_seconds=5)
                    if start_ack.get("control_status", "ok") != "ok" or start_ack.get("policy_id") != args.policy_id:
                        raise ValueError("Nicla did not start the requested policy")
                    if start_ack.get("measurement_stage_id") != args.nicla_stage_id:
                        raise ValueError("Nicla did not acknowledge the requested measurement stage")
                    writer.metadata["nicla_startup_receipt"] = {"identity_ack": identity_ack,
                        "config_ack": config_ack, "ping_ack": ping_ack, "seed_ack": seed_ack,
                        "start_ack": start_ack, "observed_firmware_sha256": observed_firmware,
                        "observed_config_sha256": observed_config, "seed_id": seed_id,
                        "measurement_stage_id": args.nicla_stage_id,
                        "model_slot": args.nicla_model_slot,
                        "pre_stage_stop_ack": pre_stage_stop_ack,
                        "preload_ack": preload_ack}
                    writer.write_metadata()
                    writer.event("nicla_policy_started", {"policy_id": args.policy_id,
                        "model_id": args.model_id, "physical_block_seed": args.physical_block_seed,
                        "training_root_seed": args.training_root_seed,
                        "firmware_sha256": observed_firmware, "config_sha256": observed_config})
                init_packet = bytes(make_command(0x81))
                init_written = device.write([0] + list(init_packet))
                writer.event("hid_command", {"command": "0x81", "payload_len": len(init_packet),
                                             "hidapi_write_return": int(init_written),
                                             "report_id_prefix": 0})
                time.sleep(0.05)
                start = time.monotonic()
                measurement_start = start + args.warmup_seconds
                deadline = measurement_start + args.duration
                writer.metadata["warmup_seconds"] = args.warmup_seconds
                writer.metadata["measurement_window_duration_s"] = args.duration
                writer.event("warmup_start", {"host_monotonic_ns": time.monotonic_ns(),
                    "warmup_seconds": args.warmup_seconds})
                prior_ns = None
                arrival_index = 0
                read_lengths = {}
                measurement_started = False
                while time.monotonic() < deadline:
                    if not measurement_started and time.monotonic() >= measurement_start:
                        writer.event("measurement_window_start", {"host_monotonic_ns": time.monotonic_ns(),
                            "host_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds")})
                        measurement_started = True
                        if stimulus_requested:
                            event_log = args.stimulus_event_log
                            if not event_log.is_absolute():
                                event_log = root / event_log
                            presentation_command = [sys.executable, "-m", "nicla_study.presentation_player",
                                "--workspace", str(root), "--block", args.stimulus_block,
                                "--condition", args.stimulus_condition,
                                "--trial-count", str(args.stimulus_count),
                                "--session-id", args.stimulus_session_id,
                                "--lighting-state", args.stimulus_lighting_state,
                                "--motion-state", args.stimulus_motion_state,
                                "--range-cm", str(args.stimulus_range_cm), "--seed", str(args.stimulus_seed),
                                "--hold-seconds", str(args.stimulus_hold_seconds),
                                "--blank-seconds", str(args.stimulus_blank_seconds),
                                "--event-log", str(event_log)]
                            if args.stimulus_freeze_receipt:
                                receipt = args.stimulus_freeze_receipt
                                if not receipt.is_absolute():
                                    receipt = root / receipt
                                presentation_command.extend(["--freeze-receipt", str(receipt)])
                            presentation_process = subprocess.Popen(presentation_command,
                                cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
                            writer.event("stimulus_player_started", {"command": presentation_command,
                                "event_log": str(event_log), "block": args.stimulus_block,
                                "condition": args.stimulus_condition,
                                "session_id": args.stimulus_session_id, "seed": args.stimulus_seed})
                    read_command = bytes(make_command(0x82))
                    written = device.write([0] + list(read_command))
                    if written not in (64, 65):
                        writer.event("hid_write_unexpected_length",
                                     {"command": "0x82", "hidapi_write_return": int(written),
                                      "expected_64_or_65": True})
                    payload = device.read(64, timeout_ms=250)
                    receive_ns = time.monotonic_ns()
                    receive_utc = receipt_clock.iso_utc_ns(receipt_clock.host_utc_ns())
                    if not payload:
                        writer.record_timeout(arrival_index, receive_ns, receive_utc)
                        if nano_serial:
                            nano_serial.poll()
                        if nicla_serial:
                            nicla_serial.poll()
                        continue
                    payload = bytes(payload)
                    read_lengths[str(len(payload))] = read_lengths.get(str(len(payload)), 0) + 1
                    gap = prior_ns is not None and receive_ns - prior_ns > 200_000_000
                    writer.record_report(payload, arrival_index, receive_ns, receive_utc, gap)
                    if len(payload) >= 2:
                        writer.event("hid_payload_observed", {"report_arrival_index": arrival_index,
                                                               "payload_length": len(payload),
                                                               "prefix_hex": payload[:4].hex(),
                                                               "write_return": int(written)})
                    prior_ns = receive_ns
                    arrival_index += 1
                    if nano_serial:
                        nano_serial.poll()
                    if nicla_serial:
                        nicla_serial.poll()
                if not measurement_started:
                    writer.event("measurement_window_start_missing", {"reason": "no_loop_iteration_reached_window"})
                else:
                    writer.event("measurement_window_end", {"host_monotonic_ns": time.monotonic_ns(),
                        "host_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds")})
                if nicla_serial:
                    nicla_stop_ack = nicla_serial.command({"cmd": "stop"}, wait_seconds=5)
                    writer.metadata["nicla_stop_ack"] = nicla_stop_ack
                    nicla_serial.close("complete" if read_lengths else "failed")
                    stop_ok = nicla_stop_ack.get("control_status", "ok") == "ok"
                    writer.metadata["nicla_serial_outcome"] = (
                        "complete" if read_lengths and stop_ok else "failed")
                    if not stop_ok:
                        writer.event("nicla_stop_ack_failed", {"ack": nicla_stop_ack})
                if presentation_process is not None:
                    if presentation_process.poll() is None:
                        writer.event("stimulus_player_exceeded_measurement_window", {
                            "measurement_duration_s": args.duration,
                            "planned_stimulus_duration_s": args.stimulus_count * (
                                args.stimulus_hold_seconds + args.stimulus_blank_seconds)})
                        presentation_process.terminate()
                    try:
                        stdout, _ = presentation_process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        presentation_process.terminate()
                        stdout, _ = presentation_process.communicate(timeout=3)
                    event_log = args.stimulus_event_log
                    if not event_log.is_absolute():
                        event_log = root / event_log
                    stimulus_status = "complete" if presentation_process.returncode == 0 else "failed_or_interrupted"
                    writer.metadata["stimulus_player"] = {"status": stimulus_status,
                        "return_code": presentation_process.returncode,
                        "command": presentation_command, "event_log": str(event_log),
                        "event_log_sha256": sha256(event_log) if event_log.is_file() else None,
                        "stdout_tail": (stdout or "")[-2000:]}
                    writer.write_metadata()
                    writer.event("stimulus_player_finished", writer.metadata["stimulus_player"])
                if nano_serial:
                    nano_serial.close("complete" if read_lengths else "failed")
                    writer.metadata["nano_serial_status"] = (
                        "captured_in_same_locked_session_connected_at_end"
                        if nano_serial.connected else "captured_with_disconnect_no_reconnect_at_end")
                    writer.metadata["nano_serial_reconnect_count"] = nano_serial.reconnect_count
                    writer.metadata["nano_serial_experiment_started"] = nano_serial.experiment_started
                    writer.metadata["nano_serial_experiment_completed"] = nano_serial.experiment_completed
                writer.metadata["hid_read_length_counts"] = read_lengths
                writer.metadata["hid_read_convention"] = "raw HIDAPI payload preserved without stripping or adding a report-ID byte"
                writer.close("complete" if read_lengths else "failed",
                             None if read_lengths else "no HID payloads received during bounded session")
                status = "complete" if read_lengths else "failed"
                print(json.dumps({"output_dir": str(out_dir), "status": status,
                                  "reports": writer.report_count, "invalid_reports": writer.invalid_count,
                                  "timeouts": writer.timeout_count, "read_length_counts": read_lengths,
                                  "nano_serial_log_file": "nano_serial.jsonl" if nano_serial else None,
                                  "nano_serial_seed_acknowledged": bool(nano_serial and nano_serial.seed_acknowledged),
                                  "nano_serial_start_sent": bool(nano_serial and nano_serial.start_sent),
                                  "nano_serial_connected_at_end": bool(nano_serial and nano_serial.connected),
                                  "nano_serial_reconnect_count": int(nano_serial.reconnect_count) if nano_serial else 0,
                                  "nano_serial_experiment_started": bool(nano_serial and nano_serial.experiment_started),
                                  "nano_serial_experiment_completed": bool(nano_serial and nano_serial.experiment_completed),
                                  "measurement_claim": "diagnostic only; energy not qualified"}, indent=2))
                return 0 if writer.report_count > 0 else 7
            except KeyboardInterrupt:
                if nano_serial:
                    nano_serial.close("interrupted")
                if nicla_serial:
                    try:
                        nicla_serial.send({"cmd": "stop"})
                    except Exception:
                        pass
                    nicla_serial.close("interrupted")
                if writer:
                    writer.close("interrupted", "KeyboardInterrupt; immutable raw partial capture retained")
                print("Interrupted; raw partial capture retained, run is not complete.", file=sys.stderr)
                return 130
            except Exception as exc:
                if nano_serial:
                    nano_serial.close("failed")
                if nicla_serial:
                    try:
                        nicla_serial.send({"cmd": "stop"})
                    except Exception:
                        pass
                    nicla_serial.close("failed")
                if writer:
                    writer.event("logger_error", {"error": "%s: %s" % (type(exc).__name__, exc)})
                    writer.close("failed", "%s: %s" % (type(exc).__name__, exc))
                print("logger failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
                return 5
            finally:
                try:
                    device.close()
                except Exception:
                    pass
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
