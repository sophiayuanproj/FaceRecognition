from __future__ import annotations

import hashlib
import random
import sys
import unittest
from pathlib import Path

FIRMWARE = Path(__file__).resolve().parents[2] / "firmware"
sys.path.insert(0, str(FIRMWARE))
import identity_contracts  # noqa: E402


class _GrayImage:
    def __init__(self, data, width, height):
        self._data = data
        self._width = width
        self._height = height

    def bytearray(self):
        return self._data

    def width(self):
        return self._width

    def height(self):
        return self._height


class _NativeGrayImage(_GrayImage):
    def __init__(self, data, width, height, failure=None):
        super().__init__(data, width, height)
        self.conversions = []
        self.failure = failure

    def to_ndarray(self, dtype, *, buffer):
        self.conversions.append(dtype)
        if self.failure:
            buffer[0] = 0
            raise self.failure
        mask = 0x80 if dtype == ord("b") else 0
        buffer[:] = bytes(value ^ mask for value in self._data)


class FirmwareInputSnapshotTests(unittest.TestCase):
    def test_rejects_rgb565_passed_to_grayscale_input(self):
        packed_rgb565 = _GrayImage(bytes(8), 2, 2)
        with self.assertRaisesRegex(ValueError, "grayscale_byte_per_pixel"):
            identity_contracts.pillow_bilinear_model_input(packed_rgb565)

    def test_box_geometry_contains_every_face_pixel_and_slides_inward(self):
        from identity_contracts import face_box_square_crop, normalized_face_box

        score, rect = normalized_face_box([.85, .38, .54, .31, .39], 240, 240)
        self.assertAlmostEqual(score, .85)
        for a, b in zip(rect, [54, 82.8, 74.4, 93.6]):
            self.assertAlmostEqual(a, b)
        for box in (rect, [0, 0, 38.5, 50], [186, 171, 54, 69], [1, 2, 236, 235]):
            (x, y, w, h), reason = face_box_square_crop(box, 240, 240)
            self.assertIsNone(reason)
            self.assertEqual(w, h)
            self.assertLessEqual(0, x)
            self.assertLessEqual(0, y)
            self.assertLessEqual(x + w, 240)
            self.assertLessEqual(y + h, 240)
            self.assertLessEqual(x, box[0]); self.assertLessEqual(y, box[1])
            self.assertGreaterEqual(x + w, box[0] + box[2])
            self.assertGreaterEqual(y + h, box[1] + box[3])
        self.assertEqual(face_box_square_crop([-1, 2, 40, 40], 240, 240),
                         (None, "face_box_outside_frame"))
        self.assertEqual(face_box_square_crop([0, 0, 40, float("nan")], 240, 240),
                         (None, "invalid_face_box"))

    def test_capture_copy_preserves_exact_pre_inference_model_bytes(self):
        source = bytes((0, 32, 96, 255))
        image = _GrayImage(source, 2, 2)
        snapshot_buffer = bytearray(4)
        model_input = identity_contracts.pillow_bilinear_model_input(
            image, output_width=2, output_height=2,
            capture_buffer=snapshot_buffer)
        reusable_model_buffer = bytearray(4)

        # OpenMV invokes this callback to fill its model-owned input buffer.
        # The copy must be made before Model.predict() continues inference.
        returned = model_input(reusable_model_buffer, (1, 2, 2, 1), "b")
        evaluated_bytes = bytes(reusable_model_buffer)
        expected_hash = hashlib.sha256(evaluated_bytes).hexdigest()
        self.assertIs(returned, reusable_model_buffer)
        self.assertIs(model_input.input_bytes, snapshot_buffer)
        self.assertEqual(model_input.input_bytes, evaluated_bytes)
        self.assertEqual(model_input.input_sha256, expected_hash)

        # A later invocation/reuse of the same model buffer must not rewrite
        # the retained payload or detach its hash from the executed input.
        reusable_model_buffer[:] = b"\x01\x02\x03\x04"
        self.assertEqual(model_input.input_bytes, evaluated_bytes)
        self.assertEqual(hashlib.sha256(model_input.input_bytes).hexdigest(),
                         model_input.input_sha256)

    def test_capture_callback_receives_exact_buffer_before_model_reuse(self):
        image = _GrayImage(bytes((0, 32, 96, 255)), 2, 2)
        captured = {}

        def stream_from_callback(buffer, model_input, shape, dtype):
            captured.update({"bytes": bytes(buffer), "sha256": model_input.input_sha256,
                "shape": tuple(shape), "dtype": dtype})
            return {"status": "sent", "sha256": model_input.input_sha256}

        model_input = identity_contracts.pillow_bilinear_model_input(
            image, output_width=2, output_height=2,
            capture_callback=stream_from_callback)
        reusable_model_buffer = bytearray(4)
        model_input(reusable_model_buffer, (1, 2, 2, 1), "b")
        executed = bytes(reusable_model_buffer)
        self.assertIsNone(model_input.input_bytes)
        self.assertEqual(captured["bytes"], executed)
        self.assertEqual(captured["sha256"], hashlib.sha256(executed).hexdigest())
        self.assertEqual(captured["shape"], (1, 2, 2, 1))
        self.assertEqual(model_input.capture_result["status"], "sent")
        reusable_model_buffer[:] = b"\x01\x02\x03\x04"
        self.assertEqual(captured["bytes"], executed)

    def test_native_full_frame_conversion_matches_python_and_preserves_callback(self):
        source = bytes(range(256))
        shape = (1, 16, 16, 1)
        for dtype in (ord("b"), "int8", ord("B"), "uint8"):
            with self.subTest(dtype=dtype):
                native_image = _NativeGrayImage(source, 16, 16)
                events = []

                def capture(buffer, callback, shape, dtype):
                    events.append((buffer, callback.input_sha256, shape, dtype))

                native = identity_contracts.pillow_bilinear_model_input(
                    native_image, output_width=16, output_height=16, capture_callback=capture)
                python = identity_contracts.pillow_bilinear_model_input(
                    _GrayImage(source, 16, 16), output_width=16, output_height=16)
                native_buffer, python_buffer = bytearray(256), bytearray(256)
                self.assertIs(native(native_buffer, shape, dtype), native_buffer)
                python(python_buffer, shape, dtype)
                self.assertEqual(native_buffer, python_buffer)
                self.assertEqual(native.input_sha256, python.input_sha256)
                self.assertEqual(native.input_byte_count, 256)
                self.assertEqual(len(native_image.conversions), 1)
                self.assertEqual(events, [(native_buffer, native.input_sha256, shape, dtype)])
                self.assertIs(events[0][0], native_buffer)

    def test_native_conversion_uses_python_for_non_full_frame_or_incompatible_buffer(self):
        source = bytes(range(16))
        image = _NativeGrayImage(source, 4, 4)
        roi_input = identity_contracts.pillow_bilinear_model_input(
            image, roi=(1, 1, 2, 2), output_width=2, output_height=2)
        self.assertEqual(roi_input(bytearray(4), (1, 2, 2, 1), ord("b")),
                         bytes((133, 134, 137, 138)))
        full_input = identity_contracts.pillow_bilinear_model_input(
            image, output_width=4, output_height=4)
        self.assertEqual(full_input(bytearray(17), (1, 4, 4, 1), ord("b")),
                         bytes(range(128, 144)) + b"\x00")
        self.assertEqual(full_input(bytearray(16), (1, 4, 4, 1), "legacy"), source)
        self.assertEqual(image.conversions, [])

    def test_native_api_failure_refills_entire_tensor_through_python(self):
        source = bytes((0, 32, 96, 255))
        for failure in (TypeError("no buffer keyword"), MemoryError()):
            with self.subTest(failure=type(failure).__name__):
                image = _NativeGrayImage(source, 2, 2, failure=failure)
                model_input = identity_contracts.pillow_bilinear_model_input(
                    image, output_width=2, output_height=2)
                model_buffer = bytearray(4)
                self.assertIs(model_input(model_buffer, (1, 2, 2, 1), ord("b")), model_buffer)
                self.assertEqual(model_buffer, bytes((128, 160, 224, 127)))
                self.assertEqual(model_input.input_sha256, hashlib.sha256(model_buffer).hexdigest())
                self.assertEqual(image.conversions, [ord("b")])

    def test_capture_callback_failure_does_not_abort_model_input_fill(self):
        image = _GrayImage(bytes((0, 32, 96, 255)), 2, 2)

        def failed_export(_buffer, _model_input, _shape, _dtype):
            raise OSError("diagnostic transport unavailable")

        model_input = identity_contracts.pillow_bilinear_model_input(
            image, output_width=2, output_height=2,
            capture_callback=failed_export)
        model_buffer = bytearray(4)
        returned = model_input(model_buffer, (1, 2, 2, 1), "b")
        self.assertIs(returned, model_buffer)
        self.assertEqual(model_buffer, bytes((128, 160, 224, 127)))
        self.assertEqual(model_input.capture_error,
                         "OSError:diagnostic transport unavailable")

    def test_preprocess_time_is_reported_separately(self):
        ticks = iter((100, 137))
        image = _GrayImage(bytes((0, 32, 96, 255)), 2, 2)
        model_input = identity_contracts.pillow_bilinear_model_input(
            image, output_width=2, output_height=2,
            clock_us=lambda: next(ticks), clock_diff=lambda end, start: end - start)
        model_input(bytearray(4), (1, 2, 2, 1), "b")
        self.assertEqual(model_input.preprocess_us, 37)

    def test_row_cached_resize_preserves_preoptimization_bytes(self):
        # Golden digest from the previous exact Pillow-compatible kernel. Ten
        # deterministic input seeds cover full-frame, clipped, small-face,
        # upsampling, and non-square geometries (50 outputs total).
        cases = [
            (240, 240, (0, 0, 240, 240), 96, 96),
            (240, 240, (32, 24, 176, 196), 96, 96),
            (240, 240, (40, 30, 96, 96), 96, 96),
            (240, 240, (0, 0, 80, 80), 96, 96),
            (120, 180, (3, 7, 112, 160), 48, 64),
        ]
        combined = hashlib.sha256()
        for seed in range(10):
            rng = random.Random(8700 + seed)
            for source_width, source_height, roi, output_width, output_height in cases:
                source = bytes(rng.randrange(256)
                               for _ in range(source_width * source_height))
                output = bytearray(output_width * output_height)
                identity_contracts.pillow_bilinear_grayscale_u8(
                    source, source_width, source_height, roi, output,
                    output_width, output_height, signed_int8=True)
                combined.update(output)
        self.assertEqual(combined.hexdigest(),
            "c1c380e0e1bb0157204ae5d8b84a562a551d7af960aaf91cc1c468d2f78f0dd5")


if __name__ == "__main__":
    unittest.main()
