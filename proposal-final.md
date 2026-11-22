# Project proposal

**TLDR for us**

The feasible core is a wearable Nicla Vision that recognizes a few people you train it on. It periodically checks for a face, tries a small recognition model, and uses a larger model when the image is usable but the result is uncertain. You’ll train and quantize both models, then test whether adjusting observations, camera sleep, and model choice saves energy without missing too many encounters or making false matches. Start with one roughly frontal face in visible light; IR night operation is optional. The main research work is the controller and measured tradeoff.

**Title of the Project**

Power-Aware Face Recognition on a Wearable Arduino Nicla Vision

**Research question or design** (~60 words)

Can a wearable Arduino Nicla Vision reduce energy use by adapting camera sampling, sensor sleep, and model execution while preserving recognition of a small enrolled group? We will compare fixed schedules with a controller using image quality, wearer motion, and classifier confidence, measuring identification accuracy, unknown-person false matches, and response time across lighting and motion conditions.

**Motivation and starting point** (~80 words)

Wearable cameras often capture redundant or blurred frames, making continuous inference costly. We will build on [Edge Impulse’s FOMO detector](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo) and [MobileNet transfer-learning pipeline](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/transfer-learning-images). [NoScope’s selective inference cascades](https://arxiv.org/abs/1703.02529) and [ADMN’s adaptation to input quality](https://proceedings.neurips.cc/paper_files/paper/2025/hash/638e6384e60032eecc454c2088ba4901-Abstract-Conference.html) motivate our controller. Our contribution is training and quantizing compact identity classifiers, developing a sensing and model-selection policy, and measuring its energy–accuracy–latency tradeoff on a wearable device. The project evaluates a small consenting group and previously unseen people, extending existing model-deployment examples with adaptive control and hardware measurements.

**Approach and scope** (~100 words)

Use Nicla Vision’s GC2145 camera and IMU, Edge Impulse training, and OpenMV deployment. Train FOMO (96×96 grayscale, MobileNetV2 width 0.1, INT8) and MobileNetV1 identity classifiers with widths 0.1/0.25 (96×96 grayscale, INT8), for four people plus “other.” Start with one frontal face within a calibrated distance range. Compare fixed low/high sampling, simple face-triggered recognition, and adaptive sampling, sleep, and model escalation. Deliver a wearable demo with accuracy, false-match, latency, energy, and memory measurements across separate capture sessions. General enrollment and multi-person tracking are outside scope. Stretch: IR-assisted night operation after camera-compatibility verification.

**Feasibility, resources, and fallback** (~80 words)

Edge Impulse provides training tools and Nicla deployment support. We need the board, compatible battery, Micro-USB cable, consenting participants, and a borrowed power profiler; hardware access is not yet secured. Models will use ROMFS storage, with memory, cropping, switching, and camera wake costs checked early. Local camera data will be stored on a computer. If model switching is impractical, retain one classifier; if recognition is unreliable, evaluate adaptive face detection, preserving the energy–quality comparison.

**URL to your project repository at GitHub**

https://github.com/sophiayuanproj/FaceRecognition

**URL to your GitHub Pages project website**

https://sophiayuanproj.github.io/FaceRecognition/

**Help needed**

We would appreciate feedback on the project scope and whether the evaluation could support a student workshop paper. Can we also borrow a Nordic PPK2 or equivalent power profiler?

