# Nicla sightings verification · November 11, 2026

R18 left the local count at 58 while the device intermittently raised `MemoryError` writing telemetry. In the earlier face-positive trace the identity model produced output and the board sent JPEGs, but completed observations were missing. The sightings broker requires the completed observation and a matching JPEG hash. It also requires two consecutive face observations for an Unknown encounter.

R19 emits a smaller continuous Mode F record that retains the JPEG identity and hash, detector box, crop, model outcome, decision, and timings. The change preserves detailed diagnostic and bounded-trial output. The deploy tool verified the staged payload, retained R18 as rollback, and reused the same verified model files. Two inactive releases were archived and verified before reclaiming their board storage.

## Physical USB test

The taped Nicla captured the physical screen displaying three public portraits. The first rendering attempts did not enter the camera frame. Putting the target in the original dashboard tab made the portrait visible in a captured JPEG and produced the first new sighting. The main test used the same window and framing, with a blank screen for four seconds between encounters.

| Check | Observed result |
| --- | --- |
| Ten separated appearances across three portraits | 10 saved encounters, IDs 60–69. All were Unknown. |
| 40-second continuous hold after an encounter was saved | 91 paired frames and 91 identity runs, with zero extra photos. |
| Eleven blank intervals | Zero extra photos. A few phase-boundary frames still contained the preceding portrait. |
| JPEG and inference linkage | All ten saved JPEGs matched the boot ID, frame ID, and SHA-256 in their completed observations. |
| Telemetry during the main run | 411 paired frames, 212 identity runs, zero reported telemetry or JPEG encoding errors. Two JPEGs and one completed record were unpaired within their sampled phase boundaries. |
| Save switch off and back on | No photo across 11 identity runs while off. One new hash-verified photo after re-enabling, ID 70. |

The local dashboard finished at 70 saved encounters with USB connected, Mode F running, and sightings enabled. The 12 new records include one setup sighting, the ten main encounters, and the re-enable check.

The real board also emitted compact observations with JPEG digests for 11 archived camera-buffer replay frames, including six identity runs. This replay used the board’s pipeline and a packet sink. The physical test above verified the actual USB and storage paths. The portraits and setup are a functional test. Recognition accuracy, untethered battery operation, and power consumption were outside this run.

Machine-readable evidence is in `r19-live-encounters.json`, `r19-save-toggle.json`, `r19-positive-board-replay.json`, and `install-r19.log`. The capture geometry photos and this evidence remain in the private Mac workspace.
