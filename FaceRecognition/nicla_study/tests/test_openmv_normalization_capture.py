"""Test capture/hash behavior around the board's native Normalization callback."""

import hashlib
import sys
from pathlib import Path
import unittest


FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))

from identity_contracts import openmv_normalization_model_input  # noqa: E402


class FakeImage:
    def bytearray(self):
        return bytearray(8)

    def width(self):
        return 2

    def height(self):
        return 2


class OpenMVNormalizationCaptureTests(unittest.TestCase):
    def test_hashes_and_streams_the_exact_native_output_buffer(self):
        expected = bytes((0, 1, 2, 127, 128, 129, 254, 255, 33, 66, 99, 132))
        captured = bytearray(len(expected))
        streamed = {}

        def native_normalization(buffer, shape, dtype):
            self.assertEqual(shape, [1, 2, 2, 3])
            self.assertIn(dtype, (ord("b"), "b", b"b", "int8"))
            for index, value in enumerate(expected):
                buffer[index] = value

        def capture_callback(buffer, model_input, shape, dtype):
            streamed["bytes"] = bytes(buffer)
            streamed["shape"] = list(shape)
            return {"sent": True, "sha256": model_input.input_sha256}

        model_input = openmv_normalization_model_input(
            FakeImage(), capture_buffer=captured,
            capture_callback=capture_callback,
            clock_us=iter((10, 24)).__next__,
            clock_diff=lambda end, start: end - start,
            normalization_callback=native_normalization)
        model_buffer = bytearray(len(expected))
        returned = model_input(model_buffer, [1, 2, 2, 3], ord("b"))

        digest = hashlib.sha256(expected).hexdigest()
        self.assertIs(returned, model_buffer)
        self.assertIs(model_input.input_bytes, captured)
        self.assertEqual(bytes(captured), expected)
        self.assertEqual(model_input.input_sha256, digest)
        self.assertEqual(model_input.input_byte_count, len(expected))
        self.assertEqual(streamed, {"bytes": expected, "shape": [1, 2, 2, 3]})
        self.assertEqual(model_input.capture_result["sha256"], digest)
        self.assertEqual(model_input.preprocess_us, 14)

    def test_undersized_debug_buffer_fails_before_partial_copy(self):
        expected = bytes(range(12))

        def native_normalization(buffer, shape, dtype):
            for index, value in enumerate(expected):
                buffer[index] = value

        capture = bytearray(11)
        model_input = openmv_normalization_model_input(
            FakeImage(), capture_buffer=capture,
            normalization_callback=native_normalization)
        with self.assertRaisesRegex(ValueError, "model_input_capture_buffer_too_small"):
            model_input(bytearray(12), [1, 2, 2, 3], "int8")
        self.assertEqual(capture, bytearray(11))


if __name__ == "__main__":
    unittest.main()
