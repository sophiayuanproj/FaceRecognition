# Edge Impulse configuration recipe

Create two private projects in Edge Impulse Studio. Account/project creation
has not been performed in this workspace. No face images have been uploaded.

| Setting | Wearable Face Detector | Wearable Identity Classifier |
|---|---|---|
| Target | Arduino Nicla Vision | Arduino Nicla Vision |
| Labeling method | Bounding boxes | One label per data item |
| Image size | 96 x 96 | 96 x 96 |
| Resize | Fit shortest axis | Fit shortest axis |
| Image block | Grayscale | Grayscale |
| Learning block | Object Detection (Images) | Transfer Learning (Images) |
| Model | FOMO, MobileNetV2 width 0.1 | MobileNetV1 width 0.1 |
| Labels | face | four chosen people + other |

Face-free detector scenes have no face box. The classifier takes original-image
face crops following the geometry in firmware/recognition_config.py. Do not mix
whole-scene detector data into the identity-crop project. Keep capture-session
metadata and separate capture rounds across training, validation and test.

Export **OpenMV library**, INT8, and retain the model, matching labels and
generated scripts. Rename model/label pairs to the basenames in models/README.md.
The wider identity classifier and adaptive controller follow the first pipeline.

Sources: [project settings](https://docs.edgeimpulse.com/studio/projects/dashboard),
[OpenMV deployment](https://docs.edgeimpulse.com/hardware/deployments/run-openmv),
[model options](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/transfer-learning-images).

