# Power-aware wearable face sensing

An Arduino Nicla Vision body camera with a local dashboard, encounter photos, and fixed-versus-adaptive power experiments. Omri Steinberg-Tatman and Sophia Yuan, UCLA embedded systems course project.

The current **R19** configuration runs face detection and identity-model inference on the Nicla. Its automatic known-person acceptance is disabled because the tested identity models did not achieve reliable recognition on camera images. The dashboard groups saved encounters under **Person 1–5** and **Other**, with explicit manual assignments. The measured power study compares fixed and scene-gated operation using the same model.

## Get the code

```sh
git clone https://github.com/sophiayuanproj/FaceRecognition.git
cd FaceRecognition
```

If you already cloned this repository, commit or stash your own changes, then run `git pull --ff-only`. The application lives in the inner `FaceRecognition/` directory. Keep that directory and the adjacent `work/`, `workflow/`, and `bench/` folders together.

## Start the dashboard

**Mac, Intel or Apple Silicon.** Install Python 3.12, then run these commands from the repository root.

```sh
sh SETUP_MAC.command
sh FaceRecognition/bin/nicla-control-center --doctor --port 0
sh FaceRecognition/bin/nicla-control-center --open-browser
```

**Windows.** Install 64-bit Python 3.12 with the Python Launcher, then run `SETUP_WINDOWS.cmd`. Start the application with `FaceRecognition\tools\windows_v5\Start-NiclaControlCenter.cmd`. The camera works on both platforms. FNIRSI meter support runs on Windows.

Use the exact localhost URL printed at startup. Plug in the Nicla using a USB data cable and select **Connect / claim control**. Close OpenMV IDE or any other application that owns the same serial port. A working R19 board can connect immediately. For a replacement board or a deliberate firmware update, follow [deployment](DEPLOYMENT.md).

## Capture and review encounters

Select **Start processing**, then **Start saving**. A confirmed face produces one saved photo per encounter. After the face leaves for a few seconds, a new encounter can be saved. Open **Sightings** to filter by person and source. Open a photo to review it or assign it manually. Assignments persist on this computer and are marked separately from device decisions.

For untethered capture, stop processing, select **Arm offline capture**, and wait for **Armed** before moving the USB cable to a power bank. Reconnect to the computer, stop processing, and select **Import photos**. **View imported photos** filters the gallery to on-board captures. Repeating an import skips duplicates. The board keeps up to 24 photos or 96 KiB. The computer keeps up to 100 photos or 8 MiB, with cumulative encounter counts.

## Code and experiments

| Location | Contents |
| --- | --- |
| [FaceRecognition/control_center](FaceRecognition/control_center) | Local dashboard, camera controls, sightings and Windows meter panel |
| [FaceRecognition/host](FaceRecognition/host) | Serial broker, bounded recording, storage and validated import |
| [FaceRecognition/firmware](FaceRecognition/firmware) | Nicla MicroPython source and checked compiled modules |
| [FaceRecognition/models/deployment-v1](FaceRecognition/models/deployment-v1) | Complete deployed TFLite models, labels and gallery |
| [detector/scripts](detector/scripts) | Sophia's detector labeling and FOMO training scripts |
| [workflow](workflow) | Data preparation, model training, evaluation and artifact verification |
| [TRAINING.md](TRAINING.md) | Training instructions and environment requirements |
| [DATA.md](DATA.md) | Dataset restoration and provenance |
| [WINDOWS_VIDEO_POWER_BENCH.md](WINDOWS_VIDEO_POWER_BENCH.md) | Windows power experiment procedure |
| [bench](bench) | Measured R19 release fingerprints and install receipt |

Use [AGENT_HANDOFF.md](AGENT_HANDOFF.md) when another coding agent takes over. The existing [project website](https://sophiayuanproj.github.io/FaceRecognition/) is maintained in `docs/`.

## Saved results

Across ten paired Nicla power cycles on a 29-second test clip, scene gating reduced mean power by **49.54 mW (7.28%)**. Both modes matched **30/40 face episodes**. Median time to first match increased from **0.65 s to 1.38 s**. Both modes missed the far-outdoor segment in every run. See the [results and limitations](work/project-research-v5/final-readiness-2026-11-12/FINAL_READINESS.md) and [paired measurements](work/project-research-v5/final-readiness-2026-11-12/r19-paired-power.csv).

The November 16 screen rehearsal saved four photos across four separated encounters, with checksums linking each saved photo to its device frame. A continuous face hold and blank intervals added no duplicate sightings. On-board storage and re-import also passed while the Mac supplied USB power and its serial connection was closed. A physical power-bank boot still needs a rehearsal.

The 14/15 host result reported earlier came from a reused-data crop comparison. It does not establish device identity accuracy. The current recognition limitation is recorded in the model and project evidence.

## Verification

Run the portable checks from the repository root. They verify full model sizes and hashes before exercising firmware contracts, the host, and dashboard controls.

```sh
FaceRecognition/.venv-nicla/bin/python -B tools/check_repository.py
```

The optional host reference tests need the separate pinned model-training environment and research assets described in [TRAINING.md](TRAINING.md). Core dashboard use requires no training environment. The source checkout includes full deployed models and compiled firmware, so connecting an existing board does not require downloading a Drive handoff.

Saved fitted training artifacts and the raw power study are available in the [research assets release](https://github.com/sophiayuanproj/FaceRecognition/releases/tag/research-assets-2026-11-22). See [power reproduction](evidence/README.md) for the exact replay command.
