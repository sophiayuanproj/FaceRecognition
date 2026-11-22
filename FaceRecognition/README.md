# Nicla camera dashboard and runtime

Run commands from the repository root, which contains this `FaceRecognition/` directory. Keep that directory in place when cloning or pulling updates.

## Mac

Install Python 3.12, then double-click `SETUP_MAC.command` at the repository root or run it in Terminal.

```sh
sh SETUP_MAC.command
sh FaceRecognition/bin/nicla-control-center --open-browser
```

Setup creates `FaceRecognition/.venv-nicla` with pinned camera, deployment, and JPEG test dependencies. It supports Intel and Apple Silicon Macs. To choose a Python executable, set `NICLA_PYTHON`. Use `sh SETUP_MAC.command --check` to check an existing installation. The launcher prints the localhost address. Use `--port 0` if another app already uses port 8780.

## Windows

Install 64-bit Python 3.12 with the Python Launcher, then run `SETUP_WINDOWS.cmd` at the repository root. It installs the pinned camera and FNIRSI meter tools. Setup uses bundled wheels when present and downloads the same hash-pinned versions when they are absent. Set `NICLA_PYTHON_312` to a full `python.exe` path if the Python Launcher is unavailable.

```powershell
.\SETUP_WINDOWS.cmd
.\FaceRecognition\tools\windows_v5\Start-NiclaControlCenter.cmd
```

For the camera dashboard without the meter packages, use `FaceRecognition\tools\windows_v5\Setup-NiclaControlCenter.cmd` instead of `SETUP_WINDOWS.cmd`. The deployment tool discovers `mpremote.exe` in the created environment.

## Camera workflow

Connect the Nicla using a data cable and select **Connect to device**. The UI shows the live image, frame-linked detector output, crop, identity-stage result, and timing. **Start processing** runs the device pipeline continuously. **Start saving** retains one local photo per confirmed face encounter.

Before using a power bank, stop processing and arm offline capture. Reconnect later and choose **Import photos**. The gallery supports **Person 1–5**, **Other**, source filters, and manual filing. Manual labels organize retained photos. R19 keeps automatic known-person acceptance disabled.

Local sightings, recording frames, and device locks are generated under `work/project-research-v5/private/`. Windows power captures are saved under `raw/dashboard-power/`. The repository excludes these generated folders. Historical replay and optional Mac host-reference workers require their research data and separate training environment.

Use the root [deployment guide](../DEPLOYMENT.md) for installation and recovery. The [operator guide](control_center/README.md) covers controls and telemetry. Training and evaluation start in [TRAINING.md](../TRAINING.md).

## Software checks

After setup, run the repository check from the root. It verifies full model sizes and hashes, runs the portable host and dashboard tests, and checks the JavaScript controls. The tests use synthetic or mocked devices.

```sh
FaceRecognition/.venv-nicla/bin/python -B tools/check_repository.py
```

On Windows, use the setup environment's interpreter.

```powershell
.\FaceRecognition\.venv-nicla-windows\Scripts\python.exe -B tools\check_repository.py
```

The JavaScript checks require Node.js 18 or newer. Add `--assets-only` to verify model and firmware files alone. The optional historical host-reference tests need the separate training environment and research assets described in [TRAINING.md](../TRAINING.md).
