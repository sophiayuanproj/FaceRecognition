"""Settings for local embedding verification on OpenMV v5.0.1.

This pipeline consumes the stock FOMO detector from /rom and a separately
deployed INT8 embedding model from /flash. Identity templates must be enrolled
explicitly; no person-specific template is included in this project.
"""

DETECTOR_MODEL = "/rom/fomo_face_detection.tflite"
EMBEDDING_MODEL = "/flash/identity_embedding_int8.tflite"
TEMPLATE_FILE = "/flash/embedding_templates.json"
CALIBRATION_FILE = "/flash/embedding_calibration.json"
POLICY_CONFIG_FILE = "/flash/policy_config.json"
RELEASE_ROOT = "/flash"  # Allows isolated, hash-locked QA bundles under /flash/qa_*.

FRAME_SIZE = "QVGA"
FRAMEBUFFER_COUNT = 1
HMIRROR = False
VFLIP = False
SETTLE_MS = 1500
SAMPLE_PERIOD_MS = 1000
MAX_FRAMES = 0  # 0 runs until the host stops the script.
SERIAL_LOGGING = True  # JSON status is emitted only while a USB serial host is attached.
LED_FEEDBACK = True  # Known=green, unknown=red, other/needs-attention=blue or amber.

DETECTION_THRESHOLD = 0.4
IDENTITY_CROP_SIDE_PIXELS = 96
LOW_QUALITY_MIN_MEAN = 12
LOW_QUALITY_MAX_MEAN = 243
LOW_QUALITY_MIN_STDEV = 5
UNCERTAIN_BAND = 0.025

# Stock OpenMV image normalization uses quantized value pixel - 128.
EXPECTED_INPUT_SCALE = 1.0 / 255.0
EXPECTED_INPUT_ZERO_POINT = -128
INPUT_SCALE_TOLERANCE = 0.00001

# The new controller adapter is staged locally and remains disabled in the
# installed board image until its full board qualification is recorded.
USE_CONTROLLED_PIPELINE = False
POLICY_ID = "G"
SMALL_EMBEDDING_MODEL = "/flash/identity_embedding_small_int8.tflite"
CAMERA_SLEEP_ENABLED = False
MODEL_SHA256 = {
    "large": "1f94fe7b3c48ee6c4b326272bf285b1381dd4ba6c68e0f4528b096fc1dcf0969",
    "small": "",
}
MODEL_CONFIG_SHA256 = ""
FIRMWARE_SHA256 = ""
QA_BOOT_MARKER = ""  # Empty in normal use; set only for an isolated startup/soft-reset qualification.

