# Host Reconstruction v1 · Offline QA

Date: 2026-11-02

## Scope

The optional HOST RECONSTRUCTION path consumes one JPEG held in memory and runs the hash-pinned local FOMO and alpha0.25 classifier artifacts in the workspace TF2.10 environment. Every output is explicitly labeled as derived from the lossy preview JPEG and not frame-identical or device-equivalent. The path is off by default; no qualified identity is emitted.

This QA did not open the preview endpoint, browser/CUA, serial, camera, or Codex target. It used only synthetic in-memory JPEGs, local model files, and mocked preview/status records. No image files were written. The frozen v1 event contract was not edited.

The local artifacts are used only for this requested experimental task. The integration manifest hash-binds the model and label files but its runtime-config digest is stale; code pins and validates the current runtime-config SHA independently and reports the manifest mismatch. No training-data license or redistribution right is asserted.

## Verification commands and results

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=FaceRecognition \
    TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=2 work/model-training/.venv/bin/python \
    -m unittest discover -s FaceRecognition/control_center/tests -v

Result: **39 tests passed**. Coverage includes missing/non-numeric/NaN/negative/>2s preview ages, stale capture before and during JPEG reads, opt-in default, host observation/JPEG SHA binding, stale and superseded inference suppression, bounded capture/worker single-flight behavior, exact resize/quantized tensor and crop bytes/hashes, six raw classifier scores, thresholds, FOMO coordinates, malformed input, edge crops, quality rejection, and hash-verified real local model loading on a synthetic JPEG.

    node --check FaceRecognition/control_center/web/app.js

Result: **passed**.

    FaceRecognition/bin/nicla-control-center --doctor --port 0

Result: **ready**; loopback bind and local replay files validated. Doctor did not check preview reachability.

Static offline DOM-reference check: **75 HTML IDs, 72 literal JavaScript DOM references, 0 missing IDs**.

The actual persistent worker wrapper was also started with the workspace TF2.10 Python, given a synthetic 240×240 JPEG, and stopped. Result: worker ready under TensorFlow **2.10.1**, pinned model hashes checked, host detector ran, qualified_identity: null. The synthetic frame produced no FOMO regions, so the classifier was correctly skipped. This was not a live-camera test.

## Pinned artifact digests

| Artifact | SHA-256 |
|---|---|
| Current runtime_config.json | 622b1b392ebe85450ce15e2421059bc71e2016e26d26f9f18da0a58f9813be39 |
| Integration manifest | 1673999923e11d0d0081fdd70d729289e89f3ee432c75ef9bf6a4afcd8a2482a |
| Manifest-declared runtime config (stale) | 7c6d3c9645c904aa9fcd37707390c54e5e0cc48aaf8333d428c896139714d146 |
| FOMO detector artifact | 4a484c9b56f806b4d13b603e66c3f7df60012ac74149e7a28655ddd497e5195c |
| alpha0.25 classifier artifact | 8be1d461ae1606fd7d6be4295049e7e47f0c7516323cb9f5889e7346b9b9ca55 |
| Frozen v1 event contract (unchanged) | e82f84f540a58d16289d3961e74dccf95a2dded933165db194db04302c020411 |

## Source and test SHA-256

| File | SHA-256 |
|---|---|
| control_center/server.py | a0c5f2916a583e1f8c999f35c69d5bcc3189b67e765c3087ef6787aafe3fa9fa |
| control_center/diagnostic_runner.py | 83a018fd54017306491e4459191874edb50d325cbacb9ac299a0235563d27371 |
| control_center/host_diagnostic.py | f232f6c85b1b4f5326d55fcb30a22f4627b93174de92fbc5d46f30d9a807c3ef |
| control_center/host_diagnostic_worker.py | 8b81a1de85fd5d91a1dcdeaab5ae6e0b45df758569d4d3bca5ac3aa654c8fd8d |
| control_center/web/app.js | d50c4652c6bab83c1819cd7c3706437cdec479d41c1d8cd250233d672eb72ee6 |
| control_center/web/index.html | c91d39449beb51edc13ac5488459b5e7af513ccfdbc472578fb424deecd6ae74 |
| control_center/web/app.css | 3776d65526f8c5b6814c09311ff8be89e7a39f8f694b2f87f1c0940d7ded5bea |
| control_center/README.md | 1cc1e1f2746af1316195d175caa4c4806e61a8ea7ab4ac3a5ad97139648d6b94 |
| control_center/tests/test_control_center.py | 0429ab19bb4aa8154d49501a4457c681918645fc53c635f3b954518f657e13f7 |
| control_center/tests/test_host_diagnostic.py | f637aff8d86498aaba8845c2a4d5aac2cee9241773d01960c6e2741ee1fb48bc |
| bin/nicla-control-center | 44c272dcf20807c84217e85db5b094c997a02f4191395e900a86e61d75946991 |
