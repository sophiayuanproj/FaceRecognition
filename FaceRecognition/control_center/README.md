# Nicla Vision Control Center

This local loopback application has a frame-linked Nicla operator interface and a separate legacy replay/host-reference panel. The v2 device path uses `host/nicla_broker.py`. Constructing the broker does not open serial; the operator first claims the controller lease, then explicit **Connect** sends the viewer and lease IDs for server-side validation before opening serial. Broker disconnect also validates the active lease, closes the host serial handle, and leaves the device loop state unchanged.

## Launch and doctor

From Finder, double-click `FaceRecognition/bin/nicla-control-center.command`. The service prints the exact URL; default is `http://127.0.0.1:8780/`. If the port is busy, launch `FaceRecognition/bin/nicla-control-center --port 0` and use the URL it prints. To open a separate local fixture UI, launch with `--port 0` and visit `/?fixture=1`; the fixture sends no broker/device commands.

Run `FaceRecognition/bin/nicla-control-center --doctor` for local files, loopback binding, broker and optional `pyserial` guidance. Live device connection requires the Nicla Vision USB serial descriptor and `pyserial==3.5`; install the host requirements with:

```sh
python3 -m pip install -r FaceRecognition/requirements-host.txt
```

Replay data is optional. Missing private replay frames disable that research panel but do not prevent live operator mode from starting. The server stays bound to loopback.

On Windows, run the workspace's `SETUP_WINDOWS.cmd` and launch from the same workspace.

```powershell
.\FaceRecognition\tools\windows_v5\Start-NiclaControlCenter.cmd
```

If `py.exe -3.12` is unavailable, set `NICLA_PYTHON_312` to the full path of a 64-bit CPython 3.12 executable before running setup. Setup installs the bundled, hash-pinned wheels offline.

## Operator modes and controls

- **ON NICLA** is available after a successful explicit broker Connect. It reports actual connection/capabilities, requested and applied state, board boot/frame IDs, newest preview, and a separate frozen frame for the latest completed inference. Each observation links its detector, crop, exact input, decision, timing, resources, errors and command ACK by boot/frame ID.
- **HOST REFERENCE** is shown in the legacy research panel and is derived from the preview JPEG; it is not the Nicla inference result.
- **Compare this frame on Mac** runs the LFW YuNet/SFace reference against the JPEG linked to the selected Nicla inference. The broker frame ID and JPEG hash are verified before displaying the Mac result beside the device decision. The Mac reference requires the local OpenCV research assets.
- **REPLAY** presents the saved historical research cases with their recorded limitations.
- **TEST FIXTURE** (`/?fixture=1`) simulates UI events and acknowledgements. It is marked synthetic, uses generated randomized tensors, and never sends a hardware command.

The operator UI supports Connect/disconnect, controller claim/release, one-shot, processing start/stop, model/policy changes, camera sleep/wake where firmware advertises support, pause UI, bounded history and local session export. The expandable measurement section uses firmware-reported stage IDs and frame/time limits to run a bounded trial with a stable trial ID; `trial_finished` is reflected as stopped in applied state. Unsupported actions stay disabled with capability reasons. The UI retains the latest image separately from the last completed event; a stale event is labeled stale rather than presented as current.

## Saved sightings and offline capture

- For a live session, connect the Nicla, start processing and select **Start saving** in **Saved sightings**. The app saves one photo per encounter once a face is confirmed. It can save a new encounter after the face leaves the frame for a few seconds.
- Before switching to a power bank, connect over USB, stop processing or recording, and select **Arm offline capture**. Wait for **Armed**. The setting survives a power cycle, and the Nicla retains up to 24 photos or 96 KiB, whichever comes first.
- Plug the Nicla back into the computer using a data cable. In **Battery capture**, select **Connect Nicla**, stop processing if prompted, and select **Import photos**. Keep the dashboard open during the transfer. Import checks each JPEG and skips photos already imported. The board retains its copies until they roll out of its local queue. Select **Disarm offline capture** when finished.
- **Saved sightings** groups the photos retained on this computer into **Person 1–5** and **Other**. Select **View imported photos** in the battery panel to limit the gallery to on-board captures, then use the **Person** filter to narrow the results. Click a photo to review it or change its group. You can take control and file photos with the Nicla unplugged. Manual assignments are labeled in the gallery and do not affect device accuracy or recognition counts. A board acceptance files the photo automatically if the firmware enables automatic acceptance. Unverified encounters go to **Other**. An imported photo shows its import time when the board has no capture clock.

## Windows Power meter

Close FNIRSI UsbMeter, then select **Scan for meter**. Choose the current interface if more than one appears and select **Connect meter**. The Windows HID reader uses the same workspace lock as the CLI. **Disconnect meter** releases the handle and lock before CLI work. Mac and other platforms keep the camera dashboard available and hide the meter panel.

One chart point represents one valid physical HID report. The panel shows the mean voltage and current across four slots and the mean of the four individual voltage-current products. Cadence and sample age come from host receipt times. Confirm the inline topology only after checking that the measured output is the board's only power supply. The panel reports live measurements and capture quality. Use the raw audit and video energy analyzer for experiment energy and coverage.

**Record power** starts an explicit local capture. **Stop recording** finalizes it and shows its run ID and folder under `raw/dashboard-power/`. The folder contains `raw_packets.jsonl` with event, report and timeout records, `decoded_samples.csv`, and `run_metadata.json`. Use `python -m nicla_study.instruments.windows_fnirsi.offline_audit --run-dir <folder>` with the application's Python path to replay the capture. Camera USB and meter HID can operate independently.

Power endpoints are under `/api/v2/power/`. `GET status` returns support, interface, connection, quality, and recording status. `GET samples?after=0&limit=60` returns a bounded page with `next_cursor`, `session_id` and `reset`. `after=0` returns a recent tail. A session change or reset clears the displayed history. The maximum limit is 120.

The JSON POST actions require a loopback request and an `X-Control-Center-Client` identifier. The routes are `scan`, `connect` with `interface_id`, `disconnect`, `start-recording` with `topology_confirmed`, and `stop-recording`. Browser actions use the dashboard origin. Status and samples never acquire hardware. The server closes the meter on shutdown.

## Current verification boundary

The GUI, HTTP command contract and broker-backed routes are implemented. The concise current board status is in `work/project-research-v5/DEVICE_INTEGRATION.md`, with detailed deployment and recovery evidence in `work/project-research-v5/reports/device_integration_v2/DEVICE_INTEGRATION.md`. USB enumeration and a fresh identity-stage event must be verified before claiming current Nicla inference. No power or population-level recognition claim is made by this UI. The legacy 8766 HTTP preview adapter remains read-only and does not supply device inference or command ACKs.

Local session export includes the selected bounded event JSONL and, when still retained, exact same-frame JPEGs and detector/identity input payload bytes with SHA-256 and shape/type metadata. It is capped at 10 MB; omitted payloads are listed in its manifest. Synthetic fixture payloads are labeled and hashed separately. Export stays in the browser download folder on this Mac. Do not use the legacy replay panel's preview-derived host result as device evidence.
