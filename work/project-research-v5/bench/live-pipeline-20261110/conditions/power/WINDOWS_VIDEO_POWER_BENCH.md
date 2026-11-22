# Windows bench for Nicla video and power

For optional optical calibration, copy the `work/project-research-v5/handoffs/windows_calibration_wheels/` folder from the [original Windows bench ZIP](https://drive.google.com/file/d/1lL6g-M_Uu2pXYuwOxtsWD9gz8rTIUOmJ/view?usp=drivesdk) into this extraction. Then run `SETUP_WINDOWS_OPTICAL.cmd`. The dashboard and raw power replay can run from this handoff alone.

Run the dashboard, optical player, and FNIRSI logger in the **same Windows session**. Use the already-installed Nicla release named in the packet and the included 29-second development clip. This packet targets `haar-sightings-r19-20261111`. Keep earlier R18 and R17 results associated with their own release packets. The latest Windows return demonstrated simultaneous meter HID and Nano CDC. Its voltage anomalies failed energy QC. Nicla CDC through the meter and paired power measurements remain physical bench checks.

## 1. Set up and run the offline checks

Verify the transfer hash, then extract the complete workspace to a new local path. Use 64-bit CPython 3.12 and its `py.exe` launcher. Run the following in PowerShell from the folder containing `FaceRecognition` and `work`.

```powershell
$Root = (Get-Location).Path
$Bench = 'work\project-research-v5\bench\live-pipeline-20261110'
if (!(Test-Path -LiteralPath $Bench)) { $Bench = 'work\project-research-v5\private\live-pipeline-20261110' }
$Power = Join-Path $Bench 'conditions\power'
$Photos = Join-Path $Bench 'screen-variation\run_screen_trial.py'
$Meter = 'FaceRecognition\nicla_study\instruments\windows_fnirsi\capture_windows.py'
$Dashboard = 'http://127.0.0.1:8780'
if (Test-Path -LiteralPath '.\SETUP_WINDOWS_OPTICAL.cmd') { & .\SETUP_WINDOWS_OPTICAL.cmd }
else { & .\FaceRecognition\tools\windows_v5\Setup-CurrentMeasurement.cmd }
if ($LASTEXITCODE -ne 0) { throw 'Bench dependency setup failed.' }
$Python = Join-Path $Root 'FaceRecognition\.venv-nicla-windows\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $Root 'FaceRecognition'
$Run = Join-Path $Root ('bench-results\win-video-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
& $Python "$Power\prepare_windows_bench.py" --output "$Run\release-inputs"
if ($LASTEXITCODE -ne 0) { throw 'Release or video preflight failed.' }
& $Python "$Power\test_analyze_video_energy.py" -v
& $Python -m unittest nicla_study.tests.test_hid_only_nicla_capture -v
```

`SETUP_WINDOWS_OPTICAL.cmd` installs both environments and runs offline checks before a device is opened. The current packet provides `bench/INSTALL_RECEIPT.json` and `bench/release/` at the workspace root. `bench/STARTUP_REPAIR.json` identifies the verified startup files in `bench/startup/`, when present. Older R17 packets use `bench/R17_INSTALL_RECEIPT.json` and `bench/release-r17/`. Preflight checks the promoted files against the install receipt and recomputes the composite firmware and config fingerprints from those exact bytes. Keep `release-inputs` and `BENCH_INPUTS.json` with the run. If the actual board reports a different release or config, obtain the matching packet before recording. The board comparison uses the staged deployment config.

## 2. Connect and qualify the hardware

1. Connect a nominal 5 V USB source to the FNB-C2's measured main input. Connect its measured USB-C output to the Nicla's micro-USB socket using a **USB data** cable. For inline CDC, the measured upstream source must also have a host data link to Windows. Connect the meter's separate PC/HID port to Windows. Record the actual port labels and both cable paths. Leave other Nicla power inputs empty. Keep the meter in passive measurement mode.
2. Rediscover the actual interfaces. Close FNIRSI USBMeter before Python claims HID. Use the live HID interface number from discovery and match the Nicla CDC descriptor to the physical board. The historical Nano COM8 and meter interface 2 are examples only.

   ```powershell
   & $Python $Meter --project-root $Root --list
   & $Python -m serial.tools.list_ports -v
   ```

3. Start the dashboard and optical player from the same workspace. Keep their terminals open. Use one optical browser tab, keep it visible on the screen the Nicla sees, and keep dashboard/console windows outside that part of the display. The optical player records tab visibility and playback ticks. The linked checkerboard and a camera frame from a face episode establish what reached the sensor.

   ```powershell
   Start-Process -FilePath (Join-Path $Root 'FaceRecognition\tools\windows_v5\Start-NiclaControlCenter.cmd') -WorkingDirectory $Root
   Start-Process -FilePath $Python -WorkingDirectory $Root -ArgumentList 'FaceRecognition\tools\research_v5\optical_display.py','--port','8767','--extra-sources',"$Power\DISPLAY_SOURCES_WITH_VIDEO.json"
   ```

   Open `http://127.0.0.1:8780/` for the dashboard and `http://127.0.0.1:8767/` for the target. Both processes run in separate windows, leaving the current PowerShell available. If 8780 is unavailable, free that port or consistently set the runner and preflight URL to the actual dashboard port.

4. In the dashboard, choose **Connect / claim control**. Check that the actual Nicla responds, the camera is awake, and processing is stopped. Run a one-shot on a visible face and confirm the linked JPEG, detector box and crop, and a completed identity inference. Run a no-face control. Identity inference execution is sufficient for stage commissioning. Known-identity acceptance is disabled in these releases.
5. Commission the meter with known steady loads. The earlier Nano protocol is in `work/project-research-v5/config/WINDOWS_STUDY_PLAN.json` from the packet root. Keep raw reports and record a reference or official-app cross-check plus the expected supply and current range before using the trace. Use the meter's reading of stable references and the raw QC results to choose the report-level acceptance rules prospectively. An official-app agreement checks consistency with the same meter. Absolute calibration needs an independent reference. The earlier Nano handshake is useful for troubleshooting. Measure the Nicla separately to establish its load energy.

If the Nicla fails to enumerate through the meter, check the single USB data cable and main meter ports with the dashboard disconnected. Confirm direct enumeration with the same Nicla first, then retry inline. Keep the direct diagnostic separate from a measured run. UART can support separate serial/instrument commissioning when enabled on a compatible release with verified 3.3 V wiring. The video bench uses USB CDC for camera data and requires working inline CDC and simultaneous meter HID. A meter-only power cable plus a second direct USB cable can bypass the measured path.

For a regular body-camera session on R19, select the fast live policy `F` in the dashboard and apply the setting. Choose **Saved sightings → Start saving** and **Start processing** when you want local encounter photos. The dashboard keeps one photo per encounter and preserves the cumulative count after older photos expire. Saving starts off at each app launch. The app keeps up to 100 recent photos or 8 MiB of JPEGs, whichever limit is reached first. These releases report the photos as **Unknown** while known-person acceptance is disabled. The paired meter trials keep photo saving off and use policies `S` and `T` for R19.

R19 uses a smaller inference result record in live policy `F`. The preview can continue during a missing result. Compare linked JPEGs and completed observations separately in the face-present pilot. Two consecutive face detections confirm an Unknown encounter, and about two seconds of clear frames rearm it.

R19 also has **Sightings → Offline battery capture**. With processing stopped, choose **Arm offline capture** and wait for **ARMED** before switching the micro-USB cable to a power bank. The Nicla resumes capture on power-up and retains up to 24 photos or 96 KiB of JPEG data, whichever limit it reaches first. To import, connect the data cable to the computer, connect in the dashboard, stop processing, and choose **Import photos**. Imports keep the board copies and skip files already imported. Battery photos are labeled with their import time because the Nicla records a boot identifier and elapsed ticks. Use **Disarm** and turn off computer photo saving before a paired power trial.

## 3. Calibrate and run a short unmetered rehearsal

In the dashboard, select policy `S`, leave the model selection on policy default, and release control without disconnecting the broker. Keep the display and camera fixed. Copy the layout as a run input and edit that copy if needed. Inspect the linked checkerboard JPEGs. If any part of the target is outside the sensor window, correct the layout or aim and collect fresh calibration shots.

```powershell
Copy-Item "$Power\CHROME_LAYOUT.json" "$Run\layout.json"
& $Python "$Power\calibrate_video_layout.py" --layout "$Run\layout.json" --scenes "$Run\calibration-scenes.json"
& $Python $Photos --device-url $Dashboard --scenes "$Run\calibration-scenes.json" --repeats 2 --output "$Run\calibration-before"
```

The checkerboard fitter uses the separate `.venv-video-review` created by `SETUP_WINDOWS_OPTICAL.cmd`. If running from the workspace outside the packaged ZIP, the fallback below installs its dependencies offline from the same wheels. Keep the camera and player viewport fixed while fitting.

```powershell
$Review = Join-Path $Root '.venv-video-review\Scripts\python.exe'
if (!(Test-Path -LiteralPath $Review)) {
  py -3.12 -m venv .venv-video-review
  $CalWheels = Join-Path $Root 'work\project-research-v5\handoffs\windows_calibration_wheels'
  $InstrumentWheels = Join-Path $Root 'work\project-research-v5\handoffs\windows_wheels'
  $PillowWheel = Join-Path $InstrumentWheels 'pillow-12.3.0-cp312-cp312-win_amd64.whl'
  & $Review -m pip install --disable-pip-version-check --no-index --find-links $CalWheels --find-links $InstrumentWheels --require-hashes -r "$CalWheels\requirements-windows-calibration.lock"
  if ($LASTEXITCODE -ne 0) { throw 'Offline calibration dependencies failed.' }
  if ((Get-FileHash $PillowWheel -Algorithm SHA256).Hash -ne 'a2b55dd6b2a4c4b7d87ffa56bdb33fdc5fdb9a462173861a7bc097f17d91cb09') { throw 'Pillow wheel hash mismatch.' }
  & $Review -m pip install --disable-pip-version-check --no-index --no-deps $PillowWheel
  if ($LASTEXITCODE -ne 0) { throw 'Offline calibration Pillow install failed.' }
}
& $Review "$Power\calibrate_video_layout.py" --layout "$Run\layout.json" --captures "$Run\calibration-before" --output "$Run\projection-before.json"
```

Play a rehearsal with fixed and scene-gated small. Watch the target, then inspect `device_events.json`, `render_events.json`, and each `RESULT.json`. Require a matching face episode, working crop and identity stage on the fixed arm, and visible, continuous browser playback. A desktop view with no face can still return valid command acknowledgments.

```powershell
& $Python "$Power\run_video_power.py" --device-url $Dashboard --layout "$Run\layout.json" `
  --geometry "$Run\projection-before.json" --output "$Run\rehearsal" --arms fixed-small scene-gated-small
```

## 4. Capture one paired pilot

Keep the same camera, screen, viewport, lighting, and cables. Start with three randomized blocks of four arms. Compare fixed small versus scene-gated small on the same 29-second clip; capture-only and detector-only are cost ablations. Add the spatial gate only in a predeclared comparison. Test the larger model's allocation and actual identity inference separately before adding a large arm.

Get the current boot and the declared hashes. Verify the live config and composite firmware hashes both match the staged release. The composite fingerprint covers the runtime components, including the selected `.mpy`. Keep the release and boot fixed across the comparison.

```powershell
$BoardStatus = Invoke-RestMethod "$Dashboard/api/v2/device/status"
$Release = Get-Content "$Run\release-inputs\BENCH_INPUTS.json" -Raw | ConvertFrom-Json
$Boot = $BoardStatus.boot_id
$ConfigSha = $Release.files.'runtime_config.json'
$FirmwareSha = $Release.expected_device_firmware_sha256
if ($BoardStatus.device.config_sha256 -ne $ConfigSha -or $BoardStatus.device.firmware_sha256 -ne $FirmwareSha -or !$BoardStatus.device_ready -or $BoardStatus.running -ne $false) {
  throw "Connect the Nicla running $($Release.release_id) and stop processing before capture."
}
$FirmwareSource = "$Run\release-inputs\wearable_app.py"
$FirmwareSourceSha = $Release.files.'wearable_app.py'
$MpyManifest = "$Run\release-inputs\mpy\build_manifest.json"
if (Test-Path -LiteralPath $MpyManifest) {
  $FirmwareSource = Join-Path $Root 'FaceRecognition\firmware\main.py'
  $Build = Get-Content -LiteralPath $MpyManifest -Raw | ConvertFrom-Json
  $FirmwareSourceSha = $Build.sources.'wearable_app.py'.sha256
}
if (!$FirmwareSourceSha -or (Get-FileHash -LiteralPath $FirmwareSource -Algorithm SHA256).Hash -ne $FirmwareSourceSha) {
  throw 'The bundled firmware source does not match the verified release.'
}
$MeterInterface = [int](Read-Host 'Fresh FNB-C2 HID interface number')
$Topology = Read-Host 'Observed nominal 5 V path and exact meter, Nicla, and host cable/port roles'
$MeterFirmware = Read-Host 'FNB-C2 firmware version from the physical meter'
if ([string]::IsNullOrWhiteSpace($Topology) -or [string]::IsNullOrWhiteSpace($MeterFirmware)) { throw 'Record the actual topology and meter firmware first.' }
$MeterRunId = Split-Path -Leaf $Run
```

Start the meter in a PowerShell job so the runner can share the same session variables. The runner starts only after the logger writes its preflight receipt and begins receiving decoded reports. Keep roughly 15 seconds before the first video and after the final one. A 540-second capture allows time for 12 clips plus setup between them. The limit is 600 seconds, so split longer campaigns into separate immutable meter captures with unique IDs.

```powershell
$MeterArgs = @(
  '--project-root', $Root, '--interface-number', $MeterInterface, '--duration', '540', '--warmup-seconds', '5',
  '--run-id', $MeterRunId, '--output-dir', "$Run\meter", '--experiment-id', 'nicla-video-power',
  '--scenario-id', 'paired-optical-development', '--board-type', 'nicla_vision', '--board-label', 'Arduino Nicla Vision',
  '--device-reset-id', $Boot, '--firmware-version', $Release.release_id,
  '--meter-label', 'FNIRSI FNB-C2', '--meter-firmware', $MeterFirmware, '--topology', $Topology,
  '--nicla-dashboard-url', $Dashboard, '--firmware-source', $FirmwareSource,
  '--release-payload', "$Run\release-inputs\mpy\wearable_app.mpy",
  '--board-configuration', "$Run\release-inputs\runtime_config.json",
  '--expected-device-firmware-sha256', $FirmwareSha, '--expected-device-config-sha256', $ConfigSha,
  '--expected-firmware-source-sha256', $FirmwareSourceSha,
  '--expected-release-payload-sha256', $Release.files.'mpy/wearable_app.mpy',
  '--input-manifest-sha256', $Release.video.video_manifest_sha256
)
$MeterJob = Start-Job -ScriptBlock {
  Set-Location -LiteralPath $using:Root
  $CapturePython = $using:Python
  $CaptureScript = $using:Meter
  $CaptureArgs = $using:MeterArgs
  & $CapturePython $CaptureScript @CaptureArgs
  if ($LASTEXITCODE -ne 0) { throw "Meter capture exited with code $LASTEXITCODE" }
}
$MeterCsv = "$Run\meter\decoded_samples.csv"
$Deadline = (Get-Date).AddSeconds(30)
do {
  if ($MeterJob.State -in 'Failed','Stopped','Completed') {
    Receive-Job $MeterJob
    throw 'Meter ended before the experiment.'
  }
  Start-Sleep -Seconds 1
  $HasReports = (Test-Path -LiteralPath $MeterCsv) -and @(Get-Content -LiteralPath $MeterCsv -TotalCount 2).Count -eq 2
} until ($HasReports -or (Get-Date) -ge $Deadline)
if (!$HasReports) { throw 'No decoded meter reports received. Inspect the meter job and raw capture.' }
Start-Sleep -Seconds 15
```

The logger opens HID only. It queries the loopback dashboard once before meter open to verify a fresh, awake, stopped board with the specified boot and hashes, then stores `nicla_dashboard_preflight.json`. The video runner and dashboard continue to own board commands and per-frame evidence.

```powershell
& $Python "$Power\run_video_power.py" --device-url $Dashboard --layout "$Run\layout.json" `
  --geometry "$Run\projection-before.json" --optics-confirmed --blocks 3 --seed 4072 `
  --meter-run-id $MeterRunId --meter-csv "$MeterCsv" --topology-id Nicla-FNB-C2-inline-v1 `
  --expected-boot-id $Boot --expected-config-sha256 $ConfigSha --expected-firmware-sha256 $FirmwareSha --output "$Run\paired"
& $Python $Photos --device-url $Dashboard --scenes "$Run\calibration-scenes.json" --repeats 2 --output "$Run\calibration-after"
& $Review "$Power\calibrate_video_layout.py" --layout "$Run\layout.json" --captures "$Run\calibration-after" --output "$Run\projection-after.json"
```

## 5. Audit and return the evidence

Wait for the logger to close. Review the before/after projections and frame-linked rehearsal. Reject runs with movement, missing live boot/config evidence, stalls, unmatched trial conditions, or invalid meter data under the frozen rules. Keep failed runs with their reason and create a new run directory for retries. A dashboard that reconnects with a different boot requires new calibration and capture.

```powershell
Receive-Job $MeterJob -Wait
if ($MeterJob.State -ne 'Completed') { throw 'Meter job did not complete. Preserve the raw partial run.' }
& $Python -m nicla_study.instruments.windows_fnirsi.offline_audit --run-dir "$Run\meter" --output-json "$Run\meter-audit.json"
$Trials = @(Get-ChildItem "$Run\paired" -Recurse -Filter 'RESULT.json' | Sort-Object FullName | ForEach-Object FullName)
& $Python "$Power\analyze_video_energy.py" @Trials --meter-csv "$Run\meter\decoded_samples.csv" `
  --same-host-clock --min-voltage-v 4.75 --max-voltage-v 5.25 --baseline-arm fixed-small --per-episode --output "$Run\energy-pilot.json"
```

The offline meter audit flags raw anomalies. The energy analyzer checks clock, report spacing, and coverage. Apply the separate supply/current/data integrity rules before treating its numerical result as evidence. Compare each observed trial with `paired/PLAN.json` and confirm the expected pairs have status `paired`. Report joules over the full playback window and time-weighted watts, along with matched encounters, missed encounters, delay, detector calls, and identity calls. HID report receipt time supports window-level comparisons. The four values per report have no independent sample timestamps.

### Ten independent starts

Treat the three-block run above as a pilot. For the final comparison, use ten independent physical starts of the Nicla on the verified measured supply. Freeze the order, endpoints, and rejection rules before acquisition. Give each start a fresh run folder, boot ID, meter capture, unique physical block ID, and seed. Run all four primary arms inside each start. Each new boot needs fresh optical calibration and verification. A 240-second meter capture is a starting budget for one four-arm block. Use pilot timing to confirm it leaves the pre-roll and post-roll.

For each final block, replace the pilot runner command with this form. Use the block ID and seed from the predeclared schedule, and replace the topology ID only if a distinct qualified topology is being studied.

```powershell
& $Python "$Power\run_video_power.py" --device-url $Dashboard --layout "$Run\layout.json" `
  --geometry "$Run\projection-before.json" --optics-confirmed --blocks 1 `
  --physical-block-id N18-B01 --seed 62001 --shuffle-single-block `
  --meter-run-id $MeterRunId --meter-csv "$MeterCsv" --topology-id Nicla-FNB-C2-inline-v1 `
  --expected-boot-id $Boot --expected-config-sha256 $ConfigSha --expected-firmware-sha256 $FirmwareSha `
  --output "$Run\paired"
```

For a counterbalanced schedule, omit `--shuffle-single-block` and pass the exact predeclared `--arms` order for each seed. Both modes write the seed and exact order. Keep a record of every attempt, including failed captures and planned but unrun arms. Analyze each block with its own raw audit first. To produce a combined analysis, pass the qualified per-arm result files together and let each receipt locate its own meter CSV. The analyzer groups by physical block ID and requires paired arms to match meter capture, board boot, firmware/config, topology, seed, and planned order.

Keep the complete run folder, including immutable raw meter data, device and playback events, optical receipts, firmware/config/video hashes, and topology note. Keep raw camera images in the project’s private evidence store. For a shareable return, include the evidence receipts and summaries plus the raw instrument and telemetry data, and share camera files only if they are explicitly in scope.
