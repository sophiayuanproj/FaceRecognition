# Windows Nicla GUI packet setup

This is the current experimental device-GUI setup. It is separate from the preserved instrument-only ZIP. It does not imply recognition qualification or Windows/FNIRSI power measurement.

## Extract and install offline

1. Extract the current transfer ZIP to a writable local path, including paths with spaces or Unicode characters. Do not extract over an older packet.
2. From `FaceRecognition/tools/windows_v5`, run `Setup-NiclaControlCenter.cmd`. It requires 64-bit CPython 3.12 and `py.exe`. It installs only the pinned pyserial dependency from the bundled Windows wheelhouse, using hashes and no network, then runs the local doctor without opening a serial device.
3. Keep the Nicla disconnected during setup. The doctor checks local files and application imports only.

## Run the device GUI

Run `Start-NiclaControlCenter.cmd` from the extracted packet. It prints the actual loopback URL. Open that exact URL; the usual port is `http://127.0.0.1:8780/`. The app is bound to loopback only. Connect the Nicla only after confirming its physical identity in Windows Device Manager; select by the board's serial number/port if more than one compatible board is attached. The broker rejects ambiguous devices and holds an exclusive OS lock.

Keep the Bush/K01 target or another approved local stimulus in the camera's field of view. Start with one-shot, inspect the boot/frame IDs and same-frame input/result, then test start/stop and supported model/policy controls. Do not infer watts from stage calls. Use the separate metering instructions only after native Windows commissioning identifies the meter, board and supply boundaries.

## Evidence scope and recovery

The archive manifest binds every file path, byte count and SHA-256. Verify with the bundled `FaceRecognition/tools/portable_transfer.py verify <archive.zip>` before use. Extract to a fresh folder and run `verify-extracted` against that same folder. The device-runtime release/readback/rollback receipt and exact restoration procedure are in `work/project-research-v5/DEVICE_INTEGRATION.md` when included in the current packet.

The Windows transfer supports stage/GUI commissioning only when the current Mac receipt marks those capabilities available. Power, cold unplug/replug, battery, separate physical sessions, and statistically qualified recognition require their own evidence. If the app reports an unsupported capability or device error, save the local receipt and stop before changing firmware or deleting recovery data.
