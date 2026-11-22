# Job D — five-person crop recognition baselines

**Status:** campaign complete and verified. **Prospective usefulness goal:** not met. The models are crop-level research candidates, not deployable identity recognizers.

## What was run

The campaign used the frozen public K01–K05 roster and SurvFace 96×96 grayscale face crops. No personal roster was used. Final evaluation followed the registry freeze and included 100 known crops (20 per identity) and 300 unknown crops from 50 identity-disjoint people (six per person). Because K01–K05 appeared in v3 development, these results remain exploratory for generalization to unseen people.

The ten-block startup/conversion pilot passed. The full comparison included 80 ten-root classifier runs across eight arms, 40 separate Olivetti runs, 80 embedding/gallery backend/policy evaluations, 20 historical encoder roots and 80 historical gallery evaluations, plus a fixed host-only SFace teacher and 20 distillation student roots. The development search covered 14 candidates. Quality/diversity-aware training and gallery selection were paired with uniform controls.

Training totals summed over roots were 1,062 epochs / 15,930 steps for SurvFace, 480 / 4,320 for Olivetti, and 129 / 1,935 for SFace students. These compute-time sums are not elapsed campaign wall-clock time. Per-root configuration, environment, data hashes, curves, selected epochs and exports are recorded in the registry and run receipts.

## Candidate routes

The development-frozen large reference is MobileNetV1 α=0.25 with a six-output K01–K05+OTHER classifier. The representative export root was selected on development only (seed 20261106). Across ten roots, the float route reached **57.7% macro known identification**, **17.3% correct known acceptance**, **3.1% wrong-known identity**, **20.4% known coverage**, and **3.3% unknown FPIR**. Its unknown predicted-OTHER rate was 38.7%. The full-INT8 route reached 38.0% known identification, 4.6% correct acceptance, 3.5% wrong-known identity, 8.1% coverage, 2.8% unknown FPIR, and 42.5% unknown predicted-OTHER rate.

The compact development-frozen candidate is an α=0.1 INT8 embedding/gallery route with uniform selection. Its representative gallery root was selected on development only (seed 20261110) and used three gallery templates per identity. Across ten roots it reached **37.7% macro known identification**, **3.5% correct known acceptance**, and **4.2% unknown FPIR**. This is a weak host crop baseline, not a useful compact recognizer. The α=0.1 scratch six-class arm stayed at chance-level 20% identification with zero correct acceptance and rejected every known crop.

Model-specific calibrated thresholds were fit separately for float and quantized outputs. The final artifacts include per-root confusion matrices, known/unknown counts, calibrated metrics, open-set curves, reliability results, confidence/coverage and image-group/identity-cluster bootstrap intervals. For representative large root 20261106, the 1,000-replicate float known correct-acceptance interval was [0.15, 0.29] and unknown FPIR interval was [0.02, 0.08]; the method resamples known image groups within identity and unknown identity clusters. These intervals describe that root, while family-level seed variability is reported separately.

## What the investigation found

- Recognition is concentrated in K03 and K04. Across the ten-root large float family, mean correct acceptance by identity was K01 8.0%, K02 4.0%, K03 33.5%, K04 38.5%, and K05 2.5%.
- Quality-aware selection did not consistently improve results over uniform selection across float and INT8 backends. Difficult crops remained in evaluation. Visual review confirmed blur and partial/off-center crops; some examples showed confident wrong-known predictions. The image sheets below use development and unknown-development roles only.
- Five-output calibrated rejection and six-output OTHER approaches were broadly similar and weak. Scratch controls underfit.
- The fixed SFace host teacher had 55.0% known identification but only 5.0% correct acceptance, 2.7% unknown FPIR and 94.0% known rejection. Students provided little conditional correction; e.g. α=0.1 INT8 corrected 3.3% of teacher errors while harming 94% of teacher-correct decisions. SFace remains host-only and its pretraining corpus/authorization is unresolved.
- Olivetti was kept separate as a small 40-way closed-set sanity task: α=0.25 transfer reached 17.9% float and 10.1% INT8 macro identity accuracy. No open-set conclusions are drawn from Olivetti.
- For the same-role v3 custom-base float-quality comparison, D changed known correct acceptance by -6.8%, known identification by -4.7%, and unknown FPIR by -2.9%. This is descriptive only: all five identities were already exposed during v3 development, so it does not establish a trustworthy improvement.

### Additional checks using existing permitted data

I completed a post-hoc paired comparison of the actual E compact gallery route against the large classifier, across the same ten root seeds and final crops. This did not change candidate selection. The compact INT8 gallery and large float classifier agreed on 70.9% of known decisions; correct known acceptance averaged 3.5% for compact and 17.3% for large. The large float model corrected 16.9% of compact-route known errors and had lower unknown FPIR (3.3% versus 4.2%). Comparing INT8 to INT8, agreement was 66.8%, correct acceptance was 3.5% compact versus 4.6% large, and large corrected 4.5% of compact errors. These are descriptive comparisons on the same final crops, not independent-person uncertainty estimates.

A development-only failure diagnostic rechecked all 350 permitted development and unknown-development image hashes. For large root 20261106, the full crop reached 72% raw known macro identification, 28% correct acceptance, 0% wrong-known acceptance, and 6.3% unknown FPIR at its calibration-frozen threshold. A coarse central 64×64 crop resized to 96×96 reached 30% identification, 16% correct acceptance, 20% wrong-known acceptance and 13% FPIR. A periphery-only proxy that masked the center reached 24% identification, 20% correct acceptance, 56% wrong-known acceptance and 57.3% FPIR. This sensitivity raises concern about context dependence, but the center mask is not a landmark-verified face segmentation, so it does not prove a background shortcut.

On unknown-development crops, 143/300 were predicted as OTHER (47.7%, mean OTHER probability 0.606); 19/300 were falsely accepted as known (6.3%, mean maximum-known probability 0.786, range 0.682–0.974). No known-development crop had OTHER as its top output. The selected compact representation has 64 output dimensions, but only 40 had variance above 1e−8 and its covariance effective rank was 3.18. Mean cosine was 0.872 for same-known pairs and 0.838 for different-known pairs, with heavily overlapping distributions. This indicates a strongly concentrated, weakly separated feature space; it is not a fully constant embedding.

The recent Mac USB/serial discovery found zero attached devices, so the conditional Nicla arena probe could not run. The optional pretraining pool and new external or personal face data were not used: under your instruction, I did not obtain or use additional people’s images. The Edge Impulse and SFace provenance review is recorded, but upstream corpus/authorization cannot be resolved from those sources without a maintainer/rightsholder clarification. No one was contacted.

## Pretrained sources and contracts

Both Edge Impulse MobileNetV1 width checkpoints were downloaded, hashed and strictly loaded into reference-compatible Keras graphs. The 0.1 and 0.25 features are taken at `conv_pw_13_relu`; input is 96×96 single-channel gray8/255 in [0,1]. The 0.1 checkpoint SHA-256 is `401cc0929ae5592f53ec25c5f6cd9228e5bf905b0065644f1c33483b5b2589c3`; the 0.25 checkpoint SHA-256 is `9b2f73e58d3ea8c86ff04b38cd1ea26235b73e3270c9b87afe6633dd58e94eff`. The checkpoint source pages do not declare pretraining corpus or license, so neither is inferred. TFLite float and full-INT8 exports, tensor types, shapes, quantization, operators, label order and calibration contracts are in the root registry.

OpenCV Zoo SFace was investigated as a genuine face-recognition host reference/teacher. The model directory is [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface); its upstream [wrapper](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py) and [provenance discussion](https://github.com/opencv/opencv_zoo/issues/313) are recorded. The directory README states Apache 2.0 for all files, but the exact checkpoint training corpus and upstream authorization remain unresolved. The official [provenance issue remains open](https://github.com/opencv/opencv_zoo/issues/313) as checked on 2026-10-29; no maintainer clarification was visible.

## Handoff and deployment boundary

Use [E_CANDIDATE_ROUTES.json](E_CANDIDATE_ROUTES.json) for the development-frozen route choices and [D_candidate_models.zip](D_candidate_models.zip) for representative host checkpoints and float/full-INT8 TFLite exports. The combined model registry covers all roots and includes input/output contracts, thresholds, gallery builders, initialization, source and data hashes, curves and final evidence.

**No model is board-qualified.** No Nicla was observed, camera-pipeline accuracy was not measured, and arena memory, operator support, latency, power and operator support remain pending. The current package can support further host crop experiments and E’s model comparison only. The five-person practical recognition goal remains unmet and requires additional permitted face data and domain-matched evaluation before any user-facing use.

Supplementary records: [paired candidate comparison](CANDIDATE_PAIRWISE_FINAL_ANALYSIS.json), [development failure diagnostics](DEVELOPMENT_FAILURE_DIAGNOSTICS.json), [fresh device discovery](../../reports/D/FINAL_DEVICE_DISCOVERY.json), and [pretrained provenance status](../../reports/D/PRETRAINED_PROVENANCE_STATUS.json).

## Reviewed development sheets

Classifier route:

![Large classifier development review](/Users/omristeinberg-tatman/Documents/Codex/2026-10-04/ok-x20/work/project-finish-v4/reports/D/DEVELOPMENT_VISUAL_REVIEW/classifier_sv_mnv1_a025_quality_select_6.png)

Compact gallery route:

![Compact gallery development review](/Users/omristeinberg-tatman/Documents/Codex/2026-10-04/ok-x20/work/project-finish-v4/reports/D/DEVELOPMENT_VISUAL_REVIEW/embedding_gallery_a010_int8_uniform.png)
