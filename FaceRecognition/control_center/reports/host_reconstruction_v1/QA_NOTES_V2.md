# Host Reconstruction v2 — verification receipt

Date: 2026-11-02

## What was checked

The host-only diagnostic remains opt-in. Preview-only stays the default. The v2 host manifest is a new sidecar, `FaceRecognition/models/deployment-v1/integration_manifest_host_reconstruction_v1.json`, which binds the current runtime config and current detector/classifier files. The prior `integration_manifest.json` was left unchanged. The new manifest is explicitly local host-only and is not a deployable release manifest. Its SHA-256 is `a5ca149876943a58bafc7465ca726f3d42e995f5a4c2980752119f74506ccc05`; the current runtime-config SHA-256 is `622b1b392ebe85450ce15e2421059bc71e2016e26d26f9f18da0a58f9813be39`, and the two match.

The TF2.10 worker was run with an in-memory synthetic 240×240 JPEG and stopped through its stdin protocol. TensorFlow 2.10.1 loaded both pinned INT8 models; the synthetic input produced one FOMO region, then the crop was rejected at the frame boundary. The classifier was not invoked, no qualified identity was returned, and no image was written. This verifies the current worker/manifest contract only. It is not a live-camera result.

## Test results

- Full control-center unittest suite: **40/40 passed**. Exact command and full output are in `work/project-research-v5/reports/mac_verification_v1/CONTROL_CENTER_HOST_RECONSTRUCTION_V2_FINAL.stdout.txt`.
- `node --check FaceRecognition/control_center/web/app.js`: passed.
- `FaceRecognition/bin/nicla-control-center --doctor --port 0`: ready; loopback only, no serial access. Output is saved in `CONTROL_CENTER_HOST_RECONSTRUCTION_V2_DOCTOR.stdout.txt`.
- The first 39-test run after adding manifest closure failed six synthetic tests because fake model fixtures lacked the newly required manifest SHA field. That attempt is preserved; the fixture was corrected and the complete 40-test rerun passed.

The suite covers preview age and stale capture suppression, explicit opt-in/default-off, host-observation and JPEG hash binding, stale/superseded result suppression, bounded worker behavior, resize/tensor/crop bytes and hashes, FOMO geometry, raw classifier scores, threshold/reason gates, quality rejection, malformed JPEGs, model/manifest hash failure, and real local-model loading on a synthetic JPEG.

## Source bindings

- `FaceRecognition/firmware/runtime_config.json` — SHA-256 `622b1b392ebe85450ce15e2421059bc71e2016e26d26f9f18da0a58f9813be39` (10541 bytes).
- `FaceRecognition/models/deployment-v1/integration_manifest_host_reconstruction_v1.json` — SHA-256 `a5ca149876943a58bafc7465ca726f3d42e995f5a4c2980752119f74506ccc05` (4071 bytes).
- `FaceRecognition/models/deployment-v1/integration_manifest.json` — SHA-256 `1673999923e11d0d0081fdd70d729289e89f3ee432c75ef9bf6a4afcd8a2482a` (3866 bytes).
- `FaceRecognition/models/deployment-v1/detector_gray_fomo_int8.tflite` — SHA-256 `4a484c9b56f806b4d13b603e66c3f7df60012ac74149e7a28655ddd497e5195c` (10160 bytes).
- `FaceRecognition/models/deployment-v1/identity_classifier_a025_int8.tflite` — SHA-256 `8be1d461ae1606fd7d6be4295049e7e47f0c7516323cb9f5889e7346b9b9ca55` (327072 bytes).
- `FaceRecognition/control_center/host_diagnostic.py` — SHA-256 `d7b1347aaab14f15aa30936b5eff4edf202cb8c53414cb65aa414ce3ce301cc2` (21092 bytes).
- `FaceRecognition/control_center/host_diagnostic_worker.py` — SHA-256 `3fa6a7fdd53a456e01546c66df6b8af2385eaaf949aa6d3fdb3cde886e854cc9` (2685 bytes).
- `FaceRecognition/control_center/server.py` — SHA-256 `a0c5f2916a583e1f8c999f35c69d5bcc3189b67e765c3087ef6787aafe3fa9fa` (38938 bytes).
- `FaceRecognition/control_center/web/app.js` — SHA-256 `d50c4652c6bab83c1819cd7c3706437cdec479d41c1d8cd250233d672eb72ee6` (34069 bytes).
- `FaceRecognition/control_center/web/index.html` — SHA-256 `c91d39449beb51edc13ac5488459b5e7af513ccfdbc472578fb424deecd6ae74` (13591 bytes).
- `FaceRecognition/control_center/tests/test_control_center.py` — SHA-256 `0429ab19bb4aa8154d49501a4457c681918645fc53c635f3b954518f657e13f7` (7263 bytes).
- `FaceRecognition/control_center/tests/test_host_diagnostic.py` — SHA-256 `f47b08fef6c57153197b4d52aee4a15dfceeed38fcfb1561cbef89d515d0c0a8` (17181 bytes).
- `FaceRecognition/control_center/README.md` — SHA-256 `0a9937032b1f11be4f97e1ca4a0eea7906e5a3ba0c1659df40434f603021acea` (5098 bytes).
- `FaceRecognition/bin/nicla-control-center` — SHA-256 `44c272dcf20807c84217e85db5b094c997a02f4191395e900a86e61d75946991` (489 bytes).
- `work/project-research-v5/reports/mac_verification_v1/CONTROL_CENTER_HOST_RECONSTRUCTION_V2_FINAL.stdout.txt` — SHA-256 `3bdc050db3547d725449d2b26644cafae040fccd6f7e98f259dd2e31543b93fc` (4224 bytes).
- `work/project-research-v5/reports/mac_verification_v1/CONTROL_CENTER_HOST_RECONSTRUCTION_V2_DOCTOR.stdout.txt` — SHA-256 `f9290170165d2e89b47b610b0c688d74e6ce98df0bc37c07b6672ed9238846fd` (279 bytes).

## Scope and remaining physical dependency

No live preview frame was fed to FOMO/identity in this verification. The primary's current sensor-frame check shows a dark text/console display with no target while the Codex side-window tab 9 still displays the face target; see `CAMERA_SCENE_REFRESH_V3.md`. Keep the Codex target tab visible during camera tests and verify the physical target in a fresh sensor frame before any new labeled acquisition. The 8780 server still needs a clean restart and visual UI check after these source changes.

HOST RECONSTRUCTION output is derived from JPEG quality 55. It may show a host crop, tensor, FOMO score, and historical classifier score for diagnosis, but it is not the board's exact input, decision, or an identity prediction suitable for use. The UI always keeps `qualified_identity` null. No third-party model-training dataset license or redistribution right is asserted by this host diagnostic.
