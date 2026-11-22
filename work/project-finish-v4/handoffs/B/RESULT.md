# Job B — public data foundation

**Status:** complete and verified for data preparation. This receipt does not report model training, face detection, physical camera testing, or hardware measurement.

## Frozen five-identity roster

| Label | Public source ID | Eligible images | Quality-valid unique groups | Verified unique groups |
|---|---:|---:|---:|---:|
| K01 | 10447 | 264 | 261 | 261 |
| K02 | 9889 | 307 | 298 | 298 |
| K03 | 10631 | 189 | 185 | 185 |
| K04 | 10554 | 155 | 151 | 151 |
| K05 | 9452 | 197 | 185 | 185 |

The roster was chosen from 530 metadata candidates (59 eligible), using the frozen ranking and no recognition scores. Each known identity has 60 train, 10 stopping, 10 threshold-calibration, 10 development and 20 final groups, plus 20 separate presentation groups. The training-only quality rule does not filter evaluation or presentation groups.

All five identities and their images appeared in v3 development. Treat future public-image scores as exploratory. The original v3 final records and image files were not opened or rehashed. Both raw source archives were rehashed as opaque files and matched their prior SHA-256 receipts. Separately, an independent B-completion recheck rehashed both raw archives as opaque bytes and reopened, rehashed, and decoded all 198,980 permitted labeled source images; all were 96×96 mode-L images with orientation tag 1 and no errors. The two exact cross-label hash groups were confirmed absent from the study manifests. The 22,308 sealed v3-final and 10,051 unlabeled verification images remained unopened. See [source_recheck_report.json](../../data/source_recheck_report.json).

## Validation and known limitations

- Ten registered data-selection roots passed role, identity-disjointness, duplicate grouping, deterministic loader, transform, train-only augmentation, path/hash and fail-closed final-gate checks. Each root contains 1,750 primary rows and 400 presentation rows.
- The baseline sensitivity receipt was corrected to report 50 unknown presentation identities from its presentation manifest. Validation was rerun and passed.
- The source crops show substantial blur and partial/off-center faces in contact-sheet review. See [CONTACT_SHEET_REVIEW.md](../../data/CONTACT_SHEET_REVIEW.md). This review did not change the roster or evaluation records; face coverage cannot be computed without boxes or landmarks.
- D's later ten-root paired comparison of quality-ranked versus uniform group selection is specified in [DATA_FLOW.md](../../data/DATA_FLOW.md), but model fitting has not run. The same protocol defines runtime quality rejection as an abstention that reduces face-present coverage and counts toward misses, with thresholds frozen from calibration.
- The cross-identity pHash search is now exhaustive through Hamming distance 6 among the permitted labeled SurvFace images. It found 10,801 candidates; one candidate crossed frozen roles and was visually reviewed as ambiguous, then retained under the declared policy. See [PHASH_AUDIT_REVIEW.md](../../data/PHASH_AUDIT_REVIEW.md). The other candidates were not individually adjudicated. Filename camera groups are not verified capture sessions.
- Optional fresh-weight pretraining has 144,656 records from 3,729 v3-train identities. Its manifest was prepared; training was not run.
- Olivetti is isolated from SurvFace, with 395 remaining rows from 40 people after duplicate-group handling. It is a limited sanity protocol, not a robust recognition evaluation.

## Presentation and detector dependencies

The development/final optical presentation protocol and player are prepared, including source-hash checks, session metadata, randomized stimulus display, event markers and a freeze-receipt gate for the final session. No camera presentation was run on this Mac.

C now has raw WIDER FACE train/validation and bounding-box annotation archives staged; their SHA-256 values match the linked mirror page and ZIP CRC checks pass. Per-image decoding and annotation indexing stopped before a verified manifest because the parser encountered a zero-box record followed by a placeholder row. The assigned C owner should resume from the staged files. Published test boxes are unavailable, and the source filters out scenes without faces, so target-camera empty scenes must still be captured in session-separated blocks. No detector training-readiness claim is made; see [scene_data_dependency.json](../../reports/scene_data_dependency.json). See the [official WIDER FACE project page](https://shuoyang1213.me/WIDERFACE/) and [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf).

## Reproduction and evidence

Primary files and all ten sensitivity blocks are listed in `artifact_manifest.json`. `data/validation_report.json` records the final validator pass. No prepared source-image copies were created; loaders read the original files in place and verify hashes.
