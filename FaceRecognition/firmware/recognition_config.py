"""Settings for recognize.py on an Arduino Nicla Vision running OpenMV 5.0.1.

The models must be real Edge Impulse INT8 OpenMV exports. These paths are
expected future files; this repository does not include trained models.
"""

DETECTOR_MODEL = "/rom/face_detector.tflite"
IDENTITY_MODEL = "/rom/identity_classifier.tflite"

# OpenMV automatically reads a sibling .txt label file for each model.
# Keep filenames free of extra dots before the .tflite extension.

CAMERA_FRAME_SIZE = "QVGA"  # 320 x 240; configure the dataset for this capture size.
FRAMEBUFFER_COUNT = 1
HMIRROR = False
VFLIP = False
# Warm-up interval for captured/discarded frames, not a manual exposure setting.
EXPOSURE_SETTLE_MS = 2000
FRAME_PERIOD_MS = 1000  # An initial sampling setting, not a verified throughput.

# None chooses the largest centered square in the captured frame. A custom ROI
# must be a square (x, y, width, height) wholly within that frame.
SCENE_ROI = None

# FOMO predicts centers, not face sizes. This is an independently chosen fixed
# crop size in ORIGINAL FRAME pixels. Use the same size when preparing data.
# Crops that cross the image boundary are rejected rather than shifted/padded.
IDENTITY_CROP_SIDE_PIXELS = 96

DETECTION_THRESHOLD = 0.4

# None finds the foreground class named "face" in the exported labels. If the
# model has exactly one foreground class, that class can also be selected.
# An explicit index is the OUTPUT CHANNEL index (0 is FOMO background).
DETECTOR_FACE_CLASS_INDEX = None

# These are provisional development settings. Select final thresholds using a
# held-out validation session, including unknown people, before reporting them.
IDENTITY_MIN_SCORE = 0.8
IDENTITY_MIN_MARGIN = 0.15
OTHER_LABELS = ("other", "unknown")

# Start with one model object at a time. Unloading/reloading adds overhead,
# which is logged. Keeping both objects loaded requires both tensor arenas.
KEEP_MODELS_RESIDENT = False

# The image is annotated only AFTER both inference stages finish.
DRAW_DEBUG_OVERLAY = True

# Stock OpenMV image input maps grayscale bytes to INT8 as pixel - 128.
# Require the matching standard Edge Impulse input quantization contract.
EXPECTED_INPUT_SCALE = 1.0 / 255.0
INPUT_SCALE_TOLERANCE = 0.000001
EXPECTED_INPUT_ZERO_POINT = -128

# A five-class quantized softmax can have a small rounding error in its sum.
# Larger deviations suggest a model/output mismatch instead of class scores.
SOFTMAX_SUM_TOLERANCE = 0.1
