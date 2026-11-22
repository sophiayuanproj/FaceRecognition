# HOST RGB REFERENCE v1 QA

Date: 2026-11-02 (America/Los_Angeles)

## Implementation scope

Added a separate opt-in HOST RGB REFERENCE panel, YuNet/SFace worker and API. It consumes only the exact latest cached 240 × 240 color JPEG read from the existing loopback preview service. The result is bound to the host observation index and SHA-256 of that same JPEG; preview age and cached-snapshot age are checked before and after inference. Missing/stale/superseded observations and opt-out during analysis suppress the result. Worker requests are single-flight and are not queued.

The panel shows the YuNet face box/score, exact in-memory SFace aligned crop and crop hash, five K01–K05 raw cosine scores plus OTHER, both historical threshold comparisons/reasons, JPEG hash, host age/timings and pinned model/run provenance. A raw class or two passing gates never becomes a qualified identity. UI labeling states HOST REFERENCE / NOT ON NICLA; the primary live badge says LIVE CAMERA · PREVIEW ONLY. The grayscale FOMO reconstruction code and panel remain separate.

The worker validates the pinned run, source-group records, frame records and ONNX model hashes, and requires the evaluation runtime versions OpenCV 4.10.0 and NumPy 1.23.5. Its gallery is reconstructed from permitted training roles; thresholds are recomputed from unknown-calibration rows only. Pretraining exposure remains unresolved, so all output is experimental and unqualified.

## Offline fixture observations

- Saved positive fixture `000_K01_lfw_George_W_Bush_George_W_Bush_0044.jpg_rep02.jpg` (SHA-256 `4f315270835b510f7ccca6d32bc0f86fab8476a01e5a292c827590adebf9fc9f`): YuNet detected one face; the raw top class was K01. The known-cosine gate rejected (`max_known_cosine_below_threshold`) and the known-vs-OTHER margin gate rejected (`known_minus_other_margin_below_threshold`). No qualified identity was emitted.
- Saved blank fixture `BLANK_preview.jpg` (SHA-256 `8ad37acc35d29c17666233c8bfda3dd7b6414a76a3ba09f1f2d7c4de2f5c0087`): YuNet detected no face; SFace scoring was suppressed.
- These are saved-fixture results only. This QA did not open the live preview endpoint, inspect or navigate a browser, use CUA, open serial, or move the Codex camera target. It makes no claim about the current camera frame.

## Live integration smoke after deployment — 2026-11-02

The primary restarted the existing local control-center service from the updated launcher because the prior server process predated the new RGB API. The restarted process was verified listening on `127.0.0.1:8780` from the canonical project workspace. The browser attached to the existing preview owner at `127.0.0.1:8766` over read-only HTTP; the Codex in-app side window remained on the Bush target. The dashboard opened no serial port and sent no device command.

After enabling only **HOST RGB REFERENCE**, the live panel completed for host observation12 with the main header **LIVE CAMERA · PREVIEW ONLY**, main device decision **NOT RUN**, and host mode **HOST ONLY · NOT ON NICLA**. The observation had a preview age of100ms and snapshot age178ms at completion, bound to JPEG SHA-256 `4209f2947aa5d87af2c12a5fe7da6c925c7f6d1aaba53828b9e73c3a48477043`. It showed one YuNet face (score0.8989), raw K01 cosine0.734529, OTHER cosine0.302991, margin0.431538, and both frozen gates passing. The 112×112 aligned crop and hash, all six cosine scores, gate explanations, model hashes, preview age, and host-stage timings were visible in the panel. The UI displayed no qualified identity and left the device decision at NOT RUN. Later observations continued to update; one captured dashboard screenshot showed the face box on the live preview, and a separate screenshot of the lower panel showed the aligned crop and host timing cards.

The initial post-edit server process returned `not found` for the new endpoint because it was still serving the old API. A graceful **Stop dashboard** action shut down only the control-center service (the browser confirmed the camera preview owner would remain running); restarting the project launcher resolved the endpoint error. The GUI smoke is direct UI integration evidence, not a saved scored row or board result. The independent 30-frame current-scene diagnostic is in `work/project-research-v5/reports/camera_live_rgb_check_v1/RESULT.md`.

## Commands and results

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=FaceRecognition \
  TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  OMP_NUM_THREADS=2 work/model-training/.venv/bin/python \
  -m unittest discover -s FaceRecognition/control_center/tests -v
```

Result: **57 tests passed** (including RGB fixture, observation/JPEG/age binding, stale-before/after suppression, opt-in defaults, pinned runtime versions, UI wiring, and the unchanged FOMO reconstruction tests).

```sh
node --check FaceRecognition/control_center/web/app.js
FaceRecognition/bin/nicla-control-center --doctor --port 0
```

Result: JavaScript syntax valid. Doctor: **ready**, loopback bind `127.0.0.1:0`, existing preview configured at `http://127.0.0.1:8766`; preview reachability was not checked. A static HTML/JS ID check found 98 unique HTML IDs, 94 literal JS selectors, and no missing references.

## Artifact pins

| Artifact | SHA-256 |
|---|---|
| `runs/optical_sface_rgb_yunet_v1/RESULT.json` | `3668aeb8cd48ba5471ce65738a268d0b9a9275a76d020bc0848245d892f033d0` |
| `private/optical_sface_rgb_yunet_v1/FRAME_RECORDS.jsonl` | `8ac3c8faa3b9e3ee9b227cc5f2da8dd43c4f9c4479ca359451a8c0a013e8268f` |
| `private/optical_sface_rgb_yunet_v1/SOURCE_GROUP_RECORDS.jsonl` | `c278924c0656f922d8eef4022af4dc0178ed592647e4dc76b7ac46657e6b6071` |
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

## Source hashes

| File | SHA-256 |
|---|---|
| `control_center/host_rgb_reference.py` | `050bdb24d76f022bdaf886fa2bfcee0777972d38823910e4630391a2314ceecc` |
| `control_center/host_rgb_reference_runner.py` | `d42c99de3ae673f6c752078f5ccc2cf2bd565c8d1a026d0f53bc9976a19b44f5` |
| `control_center/host_rgb_reference_worker.py` | `b98abdd5fe2cb37aa5c4800a371cbcbcf64f8fbd5f862c8d14196825d4b92739` |
| `control_center/server.py` | `d97b19e9c9de7ea14547713f2468cab1d435cf42677687d8b2b1f76f285309aa` |
| `control_center/web/index.html` | `5183179ef166946da976f8116f5559343b6f00755e9936f6b13635b026c2d9f5` |
| `control_center/web/app.js` | `227a4c343d03fcbfdffa8fe3d38025d62ac17c7dc000f57f420a0be2f6bf1286` |
| `control_center/web/app.css` | `048b276eb6621777a64b69caf2d11146b652f0ebe7d74f8907ba78fb44e624bd` |
| `control_center/tests/test_host_rgb_reference.py` | `8c27ada4ac1e46e09b0c43e3d4d428771cf125d764072837469fb4ac677b57e5` |
| `control_center/README.md` | `e38b148cba34a702cd92e0fcc9ce334410d31405360b0d40dfa8ca6198f09874` |
