import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
TOOL = ROOT / "tools/deploy_nicla_runtime.py"
SPEC = importlib.util.spec_from_file_location("deploy_nicla_runtime_tested", TOOL)
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)


class NiclaModelReusePayloadTests(unittest.TestCase):
    def test_truncated_previous_manifest_cannot_authorize_an_external_root(self):
        rollback = "/flash/releases/damaged-old"
        active = "/flash/releases/current"
        original = b'{"model_asset_root":"/flash/releases/untrusted", "cut": "' + b"a" * 2048

        def readback(_port, _command, _cp, source, destination, **_kwargs):
            self.assertTrue(source.startswith(":" + rollback))
            Path(destination).write_bytes(original[:2048])
            return type("Copy", (), {"returncode": 0})()

        with tempfile.TemporaryDirectory() as backup, patch.object(DEPLOY, "invoke", readback), \
                patch.object(DEPLOY, "remote_sha", return_value=hashlib.sha256(original[:2048]).hexdigest()):
            roots, receipts = DEPLOY.referenced_model_asset_roots("port", [rollback], backup)
        self.assertEqual(roots, set())
        self.assertEqual(receipts[rollback]["status"], "invalid_json_untrusted")
        DEPLOY.validate_reuse_root(active, active, rollback, roots)
        with self.assertRaises(RuntimeError):
            DEPLOY.validate_reuse_root("/flash/releases/untrusted", active, rollback, roots)

    def test_haar_profile_uses_a_pinned_rom_cascade_without_changing_legacy_config(self):
        release_root = "/flash/releases/face-v1-test-haar"
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = DEPLOY.make_payload(release_root, temp_dir, detector_backend="gray_haar_box")
            config = json.loads(payload["runtime_config.json"].read_text())
            manifest = json.loads(payload["models/deployment-v1/integration_manifest.json"].read_text())
            self.assertEqual(config["detection"], DEPLOY.NATIVE_HAAR_CONTRACT)
            self.assertTrue(config["identity"]["small"]["acceptance_uses_calibrated_cutoff"])
            self.assertFalse(config["identity"]["allow_known_identity_acceptance"])
            self.assertEqual(manifest["runtime_config"]["sha256"], DEPLOY.sha256(payload["runtime_config.json"]))
        baseline = json.loads((ROOT / "firmware/runtime_config.json").read_text())
        self.assertEqual(baseline["detection"]["backend"], "trained_gray_fomo")
        self.assertNotEqual(DEPLOY.default_release_id(), DEPLOY.default_release_id(detector_backend="gray_haar_box"))

    def test_device_firmware_hash_contract_covers_boot_source_and_mpy(self):
        release_root = "/flash/releases/face-v1-test-firmware-hash"
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = DEPLOY.make_payload(release_root, temp_dir)
            for name in DEPLOY.FIRMWARE_PARTS:
                self.assertIn(name, payload)
            self.assertEqual(len(DEPLOY.FIRMWARE_PARTS), 10)
            self.assertIn("mpy/build_manifest.json", DEPLOY.FIRMWARE_PARTS)
            self.assertIn("mpy/wearable_app.mpy", DEPLOY.FIRMWARE_PARTS)
            tree = ast.parse((ROOT / "firmware/main.py").read_text())
            assignment = next(row for row in tree.body if isinstance(row, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == "FIRMWARE_PARTS"
                    for target in row.targets))
            self.assertEqual(tuple(ast.literal_eval(assignment.value)), DEPLOY.FIRMWARE_PARTS)

    def test_precompiled_runtime_is_versioned_and_hash_closed(self):
        release_root = "/flash/releases/face-v1-test-mpy"
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = DEPLOY.make_payload(release_root, temp_dir)
            expected = {
                "mpy/build_manifest.json", "mpy/wearable_app.mpy",
                "mpy/identity_contracts.mpy", "mpy/runtime_support.mpy",
                "mpy/telemetry.mpy",
            }
            self.assertTrue(expected.issubset(payload))
            mpy_manifest = json.loads(payload["mpy/build_manifest.json"].read_text())
            self.assertEqual(mpy_manifest["schema"], "nicla-precompiled-runtime-1")
            self.assertEqual(mpy_manifest["format_low_byte"], 6)
            self.assertEqual(mpy_manifest["mpy_value"], 8966)
            source_payload_names = {
                "wearable_app.py", "identity_contracts.py",
                "runtime_support.py", "telemetry.py",
            }
            self.assertTrue(source_payload_names.issubset(payload))
            for name, row in mpy_manifest["modules"].items():
                local_name = "mpy/" + name
                self.assertEqual(DEPLOY.sha256(payload[local_name]), row["sha256"])
                self.assertEqual(payload[local_name].stat().st_size, row["bytes"])
                source_path = payload[row["source"]]
                self.assertEqual(DEPLOY.sha256(source_path),
                    mpy_manifest["sources"][row["source"]]["sha256"])
            integration = json.loads(
                payload["models/deployment-v1/integration_manifest.json"].read_text())
            runtime = integration["precompiled_runtime"]
            self.assertEqual(runtime["manifest_path"],
                release_root + "/mpy/build_manifest.json")
            self.assertEqual(runtime["manifest_sha256"],
                DEPLOY.sha256(payload["mpy/build_manifest.json"]))
            for name, row in runtime["modules"].items():
                self.assertEqual(row["path"], release_root + "/mpy/" + name)
                self.assertEqual(row["sha256"], mpy_manifest["modules"][name]["sha256"])

    def test_reused_models_are_external_and_hash_closed(self):
        model_root = "/flash/releases/face-v1-4a8886044b2b"
        release_root = "/flash/releases/face-v1-test-code-only"
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = DEPLOY.make_payload(release_root, temp_dir, model_root)
            self.assertNotIn("models/deployment-v1/detector_gray_fomo_int8.tflite", payload)
            self.assertNotIn("models/deployment-v1/identity_embedding_a010_int8.tflite", payload)
            self.assertNotIn("models/deployment-v1/identity_classifier_a025_int8.tflite", payload)
            self.assertIn("models/deployment-v1/integration_manifest.json", payload)
            config = json.loads(payload["runtime_config.json"].read_text())
            manifest = json.loads(payload["models/deployment-v1/integration_manifest.json"].read_text())
            self.assertEqual(config["detection"]["model_path"],
                model_root + "/models/deployment-v1/detector_gray_fomo_int8.tflite")
            self.assertEqual(config["identity"]["small"]["model_path"],
                model_root + "/models/deployment-v1/identity_embedding_a010_int8.tflite")
            self.assertEqual(manifest["runtime_config"], {
                "path": release_root + "/runtime_config.json",
                "sha256": hashlib.sha256(payload["runtime_config.json"].read_bytes()).hexdigest(),
            })
            self.assertEqual(manifest["gallery"]["path"],
                model_root + "/models/deployment-v1/fixed_roster_gallery.json")
            self.assertEqual(manifest["model_asset_root"], model_root)
            external = manifest["external_model_files"]
            self.assertEqual(set(external), set(DEPLOY.MODEL_NAMES) - {"integration_manifest.json"})
            for name, row in external.items():
                source = ROOT / "models/deployment-v1" / name
                self.assertEqual(row["path"], model_root + "/models/deployment-v1/" + name)
                self.assertEqual(row["bytes"], source.stat().st_size)
                self.assertEqual(row["sha256"], DEPLOY.sha256(source))

    def test_compact_payload_requires_matching_source_and_omits_runtime_copies(self):
        release = "/flash/releases/test-compact"
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "checkout"
            firmware = fixture / "firmware"
            firmware.mkdir(parents=True)
            (firmware / "mpy").mkdir()
            shutil.copytree(ROOT / "models/deployment-v1", fixture / "models/deployment-v1")
            names = ("wearable_app", "identity_contracts", "runtime_support", "telemetry", "offline_sightings")
            manifest = {"schema": "nicla-precompiled-runtime-1", "mpy_value": 8966,
                        "format_low_byte": 6, "sources": {}, "modules": {}}
            for name in names:
                source_name = name + ".py"
                filename = "main.py" if name == "wearable_app" else source_name
                source = firmware / filename
                source.write_text(name + "\n")
                binary = firmware / "mpy" / (name + ".mpy")
                binary.write_bytes(b"M\x06\x00\x1f" + name.encode("ascii"))
                manifest["sources"][source_name] = {"sha256": DEPLOY.sha256(source), "bytes": source.stat().st_size}
                manifest["modules"][binary.name] = {"module": name, "source": source_name,
                    "sha256": DEPLOY.sha256(binary), "bytes": binary.stat().st_size}
            for name in ("boot.py", "runtime_config.json"):
                shutil.copyfile(ROOT / "firmware" / name, firmware / name)
            build = firmware / "mpy/build_manifest.json"
            build.write_text(json.dumps(manifest))
            with patch.object(DEPLOY, "ROOT", fixture):
                payload = DEPLOY.make_payload(release, temporary, "/flash/releases/assets",
                                              compact=True, enable_battery_sightings=True)
                self.assertNotIn("wearable_app.py", payload)
                self.assertNotIn("offline_sightings.py", payload)
                self.assertIn("mpy/offline_sightings.mpy", payload)
                self.assertIn("boot.py", payload)
                config = json.loads(payload["runtime_config.json"].read_text())
                self.assertIs(config["control"]["offline_sightings_enabled"], True)
                integration = json.loads(payload["models/deployment-v1/integration_manifest.json"].read_text())
                binding = integration["precompiled_runtime"]
                self.assertIs(binding["source_free"], True)
                self.assertEqual(binding["manifest_sha256"], DEPLOY.sha256(build))
                self.assertEqual(binding["modules"]["offline_sightings.mpy"]["path"],
                                 release + "/mpy/offline_sightings.mpy")
                installed = Path(temporary) / "installed"
                for name, file in payload.items():
                    path = installed / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, path)
                binding["manifest_path"] = str(installed / "mpy/build_manifest.json")
                for name, row in binding["modules"].items():
                    row["path"] = str(installed / "mpy" / name)
                boot_spec = importlib.util.spec_from_file_location("nicla_boot_tested", ROOT / "firmware/boot.py")
                boot = importlib.util.module_from_spec(boot_spec)
                with patch.dict(sys.modules, {"pyb": types.SimpleNamespace(),
                        "machine": types.SimpleNamespace(LED=lambda _pin: None, UART=None)}):
                    boot_spec.loader.exec_module(boot)
                with patch.object(sys.implementation, "_mpy", 8966, create=True):
                    self.assertEqual(boot._verified_mpy_path(str(installed), binding), str(installed / "mpy"))
                    (installed / "mpy/offline_sightings.mpy").write_bytes(b"truncated")
                    with self.assertRaisesRegex(ValueError, "precompiled_runtime_size_mismatch"):
                        boot._verified_mpy_path(str(installed), binding)
                    (installed / "mpy/build_manifest.json").write_text("{}")
                    with self.assertRaisesRegex(ValueError, "compact_runtime_manifest_hash_mismatch"):
                        boot._verified_mpy_path(str(installed), binding)
                (firmware / "main.py").write_text("source changed since compilation\n")
                with self.assertRaisesRegex(RuntimeError, "does not match build manifest"):
                    DEPLOY.make_payload(release, temporary, compact=True)

    def test_battery_headroom_reserves_fat_allocation_for_every_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "part"
            source.write_bytes(b"x" * 4097)
            allocated, reserve = DEPLOY.flash_headroom({"part": source}, 4096, battery=True)
            self.assertEqual(allocated, 8192)
            minimum = 32768 + 96 * 1024 + 24 * (768 + 38 + 4095)
            self.assertGreaterEqual(reserve, minimum)

    def test_battery_hook_preflight_stops_on_unsupported_usb_api(self):
        result = type("Result", (), {"stdout": "NICLA_BATTERY_USB_MODE unsupported\n"})()
        with tempfile.TemporaryDirectory() as temporary, patch.object(DEPLOY, "invoke", return_value=result) as invoke:
            with self.assertRaisesRegex(RuntimeError, "does not report a supported pyb.usb_mode"):
                DEPLOY.inspect_battery_boot("test", Path(temporary))
            self.assertEqual(invoke.call_count, 1)

    def test_battery_hook_follows_same_usb_serial_after_port_changes(self):
        identity = {"vid": 0x2341, "pid": 0x045F, "serial_number": "nicla-same"}
        old = types.SimpleNamespace(device="old", **identity)
        new = types.SimpleNamespace(device="new", **identity)
        unrelated = types.SimpleNamespace(device="wrong", vid=0x2341, pid=0x045F,
                                          serial_number="another-nicla")
        wrong_product_id = types.SimpleNamespace(device="bootloader", vid=0x2341, pid=0x035F,
                                                 serial_number="nicla-same")
        hook_digest, previous_digest = "a" * 64, "b" * 64
        readback = types.SimpleNamespace(returncode=0,
            stdout=f"NICLA_BATTERY_USB_READBACK VCP {hook_digest} {previous_digest}\n")
        disconnected = types.SimpleNamespace(returncode=1, stdout="serial disconnected")
        with patch.object(DEPLOY, "nicla_usb_ports", side_effect=[
                [old, unrelated], [unrelated, wrong_product_id], [new, unrelated]]), \
                patch.object(DEPLOY, "invoke", side_effect=[disconnected, readback]) as invoke, \
                patch.object(DEPLOY.time, "sleep"):
            result = DEPLOY.verify_battery_boot_after_usb_change(
                "old", identity, hook_digest, "/flash/boot.py.previous", previous_digest)
        self.assertEqual(result["port"], "new")
        self.assertTrue(result["port_changed"])
        self.assertTrue(result["usb_serial_verified"])
        self.assertEqual([call.args[0] for call in invoke.call_args_list], ["old", "new"])
        self.assertTrue(all(call.kwargs["retry_transient"] is False for call in invoke.call_args_list))

    def test_battery_readback_rejects_ambiguity_wrong_digest_or_composite_mode(self):
        identity = {"vid": 0x2341, "pid": 0x045F, "serial_number": "nicla-same"}
        one = types.SimpleNamespace(device="new-one", **identity)
        two = types.SimpleNamespace(device="new-two", **identity)
        digest = "a" * 64
        with patch.object(DEPLOY, "nicla_usb_ports", return_value=[one, two]), \
                patch.object(DEPLOY.time, "monotonic", side_effect=[0.0, 0.5, 1.0]), \
                patch.object(DEPLOY.time, "sleep"), patch.object(DEPLOY, "invoke") as invoke:
            with self.assertRaisesRegex(RuntimeError, "more than one port matches"):
                DEPLOY.verify_battery_boot_after_usb_change("old", identity, digest, None, None, timeout_s=1)
            invoke.assert_not_called()
        for mode, reported, error in (("VCP+MSC", digest, "did not establish CDC-only mode"),
                                      ("VCP", "b" * 64, "failed hash readback")):
            with self.subTest(mode=mode, reported=reported), \
                    patch.object(DEPLOY, "nicla_usb_ports", return_value=[one]), \
                    patch.object(DEPLOY, "invoke", return_value=types.SimpleNamespace(
                        returncode=0, stdout=f"NICLA_BATTERY_USB_READBACK {mode} {reported} -\n")):
                with self.assertRaisesRegex(RuntimeError, error):
                    DEPLOY.verify_battery_boot_after_usb_change("old", identity, digest, None, None)

    def test_battery_promotion_is_not_repeated_when_the_old_port_drops(self):
        identity = {"vid": 0x2341, "pid": 0x045F, "serial_number": "nicla-same"}
        digest = DEPLOY.sha256(ROOT / "firmware/battery_usb_boot.py")
        ok = types.SimpleNamespace(returncode=0)
        disconnected = types.SimpleNamespace(returncode=1)
        with patch.object(DEPLOY, "invoke", side_effect=[ok, ok, disconnected]) as invoke, \
                patch.object(DEPLOY, "remote_sha", return_value=digest), \
                patch.object(DEPLOY, "verify_battery_boot_after_usb_change",
                             return_value={"port": "new", "usb_mode": "VCP"}) as readback:
            result = DEPLOY.install_battery_boot("old", {"present": False}, "stamp", identity)
        self.assertEqual(result["port"], "new")
        self.assertEqual(invoke.call_count, 3)
        self.assertFalse(invoke.call_args.kwargs["retry_transient"])
        readback.assert_called_once_with("old", identity, digest, None, None)

    def test_resume_activates_promoted_payload_without_recopying_it(self):
        release_id = "resume-fixture"
        release_root = "/flash/releases/" + release_id
        prior_main = b"prior loader"
        current = b"/flash/releases/prior-release\n"
        previous = b"/flash/releases/earlier-release\n"
        remote = {"/flash/main.py": prior_main, "/flash/face_runtime.current": current,
                  "/flash/face_runtime.current.prev": previous}
        names = {"boot.py", "runtime_config.json", "models/deployment-v1/integration_manifest.json"}
        names.update("mpy/" + name for name in DEPLOY.MPY_NAMES)
        names.add("mpy/" + DEPLOY.OFFLINE_MPY)
        names.update("models/deployment-v1/" + name for name in DEPLOY.MODEL_NAMES)
        promoted = {name: name.encode("ascii") for name in names}
        promoted["mpy/build_manifest.json"] = b'{"mpy_value":8966}'
        remote.update({release_root + "/" + name: data for name, data in promoted.items()})

        def digest(data):
            return hashlib.sha256(data).hexdigest()

        def readback(_port, name, path):
            data = remote[name]
            path.write_bytes(data)
            return {"present": True, "sha256": digest(data), "bytes": len(data)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loader = root / "boot.py"
            loader.write_bytes(promoted["boot.py"])
            build = root / "build_manifest.json"
            build.write_bytes(promoted["mpy/build_manifest.json"])
            target = root / "target-manifest.json"
            prior = {key: readback("test", "/flash/" + key, root / key)
                     for key in ("main.py", "face_runtime.current", "face_runtime.current.prev")}
            prior["boot.py"] = {"present": False}
            target.write_text(json.dumps({"release_id": release_id, "release_root": release_root,
                "model_asset_root": release_root, "source_free": True, "offline_sightings_enabled": True,
                "prior_target": prior, "reused_model_files": {},
                "payload": {name: {"sha256": digest(data), "bytes": len(data)}
                            for name, data in promoted.items()}}))

            def invoke(port, *args, **_kwargs):
                if args[:2] == ("fs", "cp"):
                    destination = args[3].lstrip(":")
                    self.assertFalse(destination.startswith("/flash/releases/"))
                    remote[destination] = Path(args[2]).read_bytes()
                if args[-1] == "reset" or "RELEASE_POINTER_PROMOTED" in args[-1]:
                    self.assertEqual(port, "rebound")
                stdout = "FLASH_FREE_BYTES 202752 2048 8966\n" if "FLASH_FREE_BYTES" in args[-1] else ""
                return types.SimpleNamespace(stdout=stdout)

            argv = [str(TOOL), "--port", "fixture", "--resume-promoted", str(target)]
            metadata = {"boot.py": loader, "mpy/build_manifest.json": build}
            identity = {"vid": 0x2341, "pid": 0x045F, "serial_number": "fixture"}
            with patch.object(sys, "argv", argv), patch.object(DEPLOY, "PROJECT", root), \
                    patch.object(DEPLOY, "assert_no_mounted_openmv_disk"), \
                    patch.object(DEPLOY, "readback_existing", side_effect=readback), \
                    patch.object(DEPLOY, "read_promoted_metadata", return_value=(metadata, {
                        "detection": {"backend": "trained_gray_fomo"}})), \
                    patch.object(DEPLOY, "remote_sha", side_effect=lambda _port, name: digest(remote[name])), \
                    patch.object(DEPLOY, "invoke", side_effect=invoke), \
                    patch.object(DEPLOY, "ensure_remote_directory") as ensure, \
                    patch.object(DEPLOY, "create_remote_directory") as create, \
                    patch.object(DEPLOY, "device_preflight") as preflight, \
                    patch.object(DEPLOY, "install_startup", return_value=digest(loader.read_bytes())) as startup, \
                    patch.object(DEPLOY, "capture_battery_usb_identity", return_value=identity), \
                    patch.object(DEPLOY, "inspect_battery_boot", return_value=prior["boot.py"]), \
                    patch.object(DEPLOY, "install_battery_boot", return_value={"port": "rebound"}) as battery, \
                    patch("builtins.print"):
                self.assertIsNone(DEPLOY.main())
            ensure.assert_not_called()
            create.assert_not_called()
            preflight.assert_called_once_with("fixture", release_root, None)
            startup.assert_called_once_with("fixture", loader)
            self.assertEqual(battery.call_args.args[0], "fixture")
            self.assertEqual(battery.call_args.args[3], identity)
            receipt, = (root / "work/project-finish-v4/deployments/backups").glob("*/deployment-manifest.json")
            result = json.loads(receipt.read_text())
            self.assertEqual(result["resumed_from_target_sha256"], digest(target.read_bytes()))
            self.assertEqual(result["payload_allocated_bytes"], 0)
            self.assertEqual(result["port_used"], "rebound")
            self.assertTrue(result["post_promotion_hashes_all_match"])

    def test_resume_rejects_changed_active_pointer(self):
        original = {"present": True, "sha256": "a" * 64}
        prior = {"face_runtime.current": original}
        with patch.object(DEPLOY, "remote_sha", return_value="b" * 64):
            with self.assertRaisesRegex(RuntimeError, "resume target changed"):
                DEPLOY.verify_resume_prior("fixture", {"prior_target": prior}, prior)

    def test_release_id_binds_the_reuse_root(self):
        asset_a = "/flash/releases/asset-a"
        asset_b = "/flash/releases/asset-b"
        self.assertNotEqual(DEPLOY.default_release_id(asset_a), DEPLOY.default_release_id(asset_b))

    def test_release_root_parser_rejects_nonrelease_paths(self):
        for value in ("../face-v1-x", "/flash/models/deployment-v1", "/flash/releases/../x"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DEPLOY.normalize_release_root(value)

    def test_model_reuse_accepts_only_active_or_recorded_rollback_release(self):
        active = "/flash/releases/face-v1-active"
        rollback = "/flash/releases/face-v1-rollback"
        assets = "/flash/releases/face-v1-assets"
        DEPLOY.validate_reuse_root(active, active, rollback)
        DEPLOY.validate_reuse_root(rollback, active, rollback)
        DEPLOY.validate_reuse_root(assets, active, rollback, {assets})
        with self.assertRaisesRegex(RuntimeError, "active/previous"):
            DEPLOY.validate_reuse_root("/flash/releases/unrelated", active, rollback)


if __name__ == "__main__":
    unittest.main()
