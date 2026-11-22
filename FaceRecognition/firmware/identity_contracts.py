"""Small, allocation-conscious recognition contracts shared by board and host.

This module deliberately has no OpenMV or NumPy imports so host fault fixtures
exercise the same decision and geometry rules that main.py uses on Nicla.
"""

import binascii
import hashlib
import math
import struct


_PIL_RESAMPLE_Q = 1 << 22
_PIL_RESAMPLE_HALF = 1 << 21
_PIL_BILINEAR_AXIS_CACHE = {}


def _pil_bilinear_axis(input_size, output_size):
    """Precompute Pillow 10.4 8-bit bilinear coefficients in fixed point."""
    key = (int(input_size), int(output_size))
    cached = _PIL_BILINEAR_AXIS_CACHE.get(key)
    if cached is not None:
        return cached
    input_size, output_size = key
    if input_size == 240 and output_size == 96:
        # The C fixtures use the exact 240 -> 96 ratio (2.5). Express Pillow's
        # centers, support, and Q22 coefficient rounding as integers so this
        # critical path is identical on MicroPython builds with float32 math.
        rows = []
        for output_index in range(output_size):
            center_q4 = 10 * output_index + 5
            start_numerator = center_q4 - 8
            start = (start_numerator // 4 if start_numerator >= 0
                     else -((-start_numerator) // 4))
            if start < 0:
                start = 0
            stop = (center_q4 + 12) // 4
            if stop > input_size:
                stop = input_size
            numerators = []
            total = 0
            for sample in range(start, stop):
                distance_q10 = 4 * sample - center_q4 + 2
                if distance_q10 < 0:
                    distance_q10 = -distance_q10
                weight_numerator = 10 - distance_q10 if distance_q10 < 10 else 0
                numerators.append(weight_numerator)
                total += weight_numerator
            weights = [(value * _PIL_RESAMPLE_Q + total // 2) // total
                       for value in numerators]
            rows.append((start, weights))
        _PIL_BILINEAR_AXIS_CACHE[key] = rows
        return rows
    scale = float(input_size) / output_size
    filterscale = max(scale, 1.0)
    support = filterscale
    filter_scale_inverse = 1.0 / filterscale
    rows = []
    for output_index in range(output_size):
        center = (output_index + 0.5) * scale
        start = int(center - support + 0.5)
        if start < 0:
            start = 0
        stop = int(center + support + 0.5)
        if stop > input_size:
            stop = input_size
        weights = []
        total = 0.0
        for sample in range(start, stop):
            distance = (sample - center + 0.5) * filter_scale_inverse
            if distance < 0.0:
                distance = -distance
            weight = 1.0 - distance if distance < 1.0 else 0.0
            weights.append(weight)
            total += weight
        quantized = []
        for weight in weights:
            normalized = weight / total if total else weight
            quantized.append(int(0.5 + normalized * _PIL_RESAMPLE_Q))
        rows.append((start, quantized))
    _PIL_BILINEAR_AXIS_CACHE[key] = rows
    return rows


def pillow_bilinear_grayscale_u8(source, source_width, source_height, roi,
                                 output, output_width=96, output_height=96,
                                 signed_int8=False):
    """Fill output with Pillow 10.4 L-mode BILINEAR bytes for an image ROI.

    This matches Pillow's 8-bit separable resampler, including Q22 coefficient
    rounding and the intermediate horizontal pass. `source` is a flat grayscale
    buffer; `roi` is (x, y, width, height). If signed_int8 is set, output stores
    the TFLite int8 representation for scale 1/255 and zero point -128.
    """
    source_width, source_height = int(source_width), int(source_height)
    output_width, output_height = int(output_width), int(output_height)
    roi_x, roi_y, roi_width, roi_height = [int(value) for value in roi]
    if (source_width <= 0 or source_height <= 0 or output_width <= 0 or output_height <= 0 or
            roi_x < 0 or roi_y < 0 or roi_width <= 0 or roi_height <= 0 or
            roi_x + roi_width > source_width or roi_y + roi_height > source_height or
            len(source) < source_width * source_height or len(output) < output_width * output_height):
        raise ValueError("invalid_grayscale_resize_geometry")
    if roi_width == output_width and roi_height == output_height:
        source_offset = roi_y * source_width + roi_x
        for row in range(output_height):
            source_row = source_offset + row * source_width
            output_row = row * output_width
            for column in range(output_width):
                value = int(source[source_row + column])
                output[output_row + column] = ((value - 128) & 255) if signed_int8 else value
        return output

    horizontal = _pil_bilinear_axis(roi_width, output_width)
    vertical = _pil_bilinear_axis(roi_height, output_height)

    # Adjacent output rows use overlapping source rows. Cache the horizontally
    # filtered rows so the exact Pillow intermediate is computed once per
    # source row instead of once for every output-row tap. The cache is bounded
    # by the largest vertical filter footprint (normally five 96-byte rows for
    # 240x240 -> 96x96), preserving the same Q22 rounding and memory contract.
    cache_capacity = 0
    for _source_y, y_weights in vertical:
        if len(y_weights) > cache_capacity:
            cache_capacity = len(y_weights)
    cache_source_rows = [-1] * cache_capacity
    horizontal_rows = [bytearray(output_width) for _ in range(cache_capacity)]
    required_slots = [0] * cache_capacity

    for output_y in range(output_height):
        source_y, y_weights = vertical[output_y]
        first_source_row = roi_y + source_y
        required_count = len(y_weights)

        # All rows needed by this output row are consecutive. Retain any
        # existing rows in that range and evict only rows outside it.
        for y_offset in range(required_count):
            target_source_row = first_source_row + y_offset
            slot = -1
            for candidate_slot in range(cache_capacity):
                if cache_source_rows[candidate_slot] == target_source_row:
                    slot = candidate_slot
                    break
            if slot < 0:
                for candidate_slot in range(cache_capacity):
                    cached_source_row = cache_source_rows[candidate_slot]
                    if (cached_source_row < first_source_row or
                            cached_source_row >= first_source_row + required_count):
                        slot = candidate_slot
                        break
            if slot < 0:
                raise ValueError("bilinear_row_cache_capacity_exceeded")

            if cache_source_rows[slot] != target_source_row:
                horizontal_row = horizontal_rows[slot]
                source_row = target_source_row * source_width + roi_x
                for output_x in range(output_width):
                    source_x, x_weights = horizontal[output_x]
                    horizontal_sum = _PIL_RESAMPLE_HALF
                    source_offset = source_row + source_x
                    for x_offset in range(len(x_weights)):
                        horizontal_sum += int(source[source_offset + x_offset]) * x_weights[x_offset]
                    horizontal_value = horizontal_sum >> 22
                    if horizontal_value < 0:
                        horizontal_value = 0
                    elif horizontal_value > 255:
                        horizontal_value = 255
                    horizontal_row[output_x] = horizontal_value
                cache_source_rows[slot] = target_source_row
            required_slots[y_offset] = slot

        output_row = output_y * output_width
        for output_x in range(output_width):
            vertical_sum = _PIL_RESAMPLE_HALF
            for y_offset in range(required_count):
                horizontal_value = horizontal_rows[required_slots[y_offset]][output_x]
                vertical_sum += horizontal_value * y_weights[y_offset]
            value = vertical_sum >> 22
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            output[output_row + output_x] = ((value - 128) & 255) if signed_int8 else value
    return output


class PillowBilinearModelInput:
    """OpenMV preprocessing callback with the ROI metadata FOMO postprocessing uses."""

    def __init__(self, image, roi=None, output_width=96, output_height=96,
                 capture_tensor=False, capture_buffer=None, capture_callback=None,
                 clock_us=None, clock_diff=None):
        self.source = image.bytearray()
        self.source_width, self.source_height = image.width(), image.height()
        if len(self.source) != self.source_width * self.source_height:
            raise ValueError("model_input_requires_one_grayscale_byte_per_pixel")
        if roi is None:
            roi = (0, 0, self.source_width, self.source_height)
        self.roi = tuple(int(value) for value in roi)
        self.output_width, self.output_height = int(output_width), int(output_height)
        self._native_full_frame_converter = None
        if self.roi == (0, 0, self.output_width, self.output_height) and (
                self.source_width == self.output_width and self.source_height == self.output_height):
            self._native_full_frame_converter = getattr(image, "to_ndarray", None)
        self.capture_tensor = ("reference" if capture_tensor == "reference"
                               else bool(capture_tensor))
        self.capture_buffer = capture_buffer
        self.capture_callback = capture_callback
        self.clock_us = clock_us
        self.clock_diff = clock_diff
        self.preprocess_us = None
        self.capture_result = None
        self.capture_error = None
        required = self.output_width * self.output_height
        if capture_buffer is not None and len(capture_buffer) < required:
            raise ValueError("model_input_capture_buffer_too_small")
        self.input_sha256 = None
        self.input_bytes = None
        self.input_byte_count = 0

    def __call__(self, input_buffer, shape, dtype):
        signed = dtype in (ord("b"), "b", b"b", "int8")
        started = self.clock_us() if self.clock_us is not None else None
        converted = False
        if (self._native_full_frame_converter is not None and
                len(input_buffer) == len(self.source) and
                (signed or dtype in (ord("B"), "B", b"B", "uint8"))):
            try:
                # OpenMV copies uint8 unchanged and converts int8 by XOR 0x80.
                # Write into the model-owned buffer, as the Python path does.
                self._native_full_frame_converter(ord("b") if signed else ord("B"), buffer=input_buffer)
                converted = True
            except (AttributeError, MemoryError, NotImplementedError, OSError, TypeError, ValueError):
                pass
        if not converted:
            pillow_bilinear_grayscale_u8(self.source, self.source_width, self.source_height,
                self.roi, input_buffer, self.output_width, self.output_height, signed)
        if started is not None:
            ended = self.clock_us()
            self.preprocess_us = (self.clock_diff(ended, started)
                if self.clock_diff is not None else ended - started)
        self._record_input(input_buffer, shape, dtype)
        return input_buffer

    def _record_input(self, input_buffer, shape, dtype):
        digest = hashlib.sha256()
        self.input_byte_count = len(input_buffer)
        if (self.capture_buffer is not None and
                len(self.capture_buffer) < self.input_byte_count):
            raise ValueError("model_input_capture_buffer_too_small")
        try:
            digest.update(input_buffer)
        except Exception:
            # Use a view when the model supplies a nonstandard buffer object;
            # never make a tensor-sized `bytes()` copy on the MCU heap.
            digest.update(memoryview(input_buffer))
        self.input_sha256 = binascii.hexlify(digest.digest()).decode("ascii")
        if self.capture_tensor == "reference":
            # This mode aliases the model-owned buffer. It is suitable only
            # while the caller can prove the buffer has not been reused.
            # Prefer capture_callback for telemetry: it can stream this exact
            # buffer without allocating another tensor-sized copy.
            self.input_bytes = input_buffer
        elif self.capture_buffer is not None:
            # Copy into a caller-owned block allocated before model execution.
            # Avoid bytes(input_buffer), which needs a new contiguous 9 KiB
            # allocation on the constrained board while the model is live.
            for index in range(self.input_byte_count):
                self.capture_buffer[index] = input_buffer[index]
            self.input_bytes = self.capture_buffer
        elif self.capture_tensor:
            self.input_bytes = bytes(input_buffer)
        if self.capture_callback is not None:
            try:
                self.capture_result = self.capture_callback(input_buffer, self, shape, dtype)
            except Exception as error:
                # Export failure must not suppress the inference itself.
                self.capture_error = type(error).__name__ + ":" + str(error)[:100]


class OpenMVNormalizationModelInput(PillowBilinearModelInput):
    """Capture OpenMV's native RGB565/int8 normalization output without copying it.

    Stock ROM FOMO is RGB. It must use the exact firmware's own Normalization
    implementation; the grayscale Pillow-compatible routine above is not a
    valid substitute. The callback still hashes and streams the actual buffer
    that the model will consume.
    """

    def __init__(self, image, capture_tensor=False, capture_buffer=None,
                 capture_callback=None, clock_us=None, clock_diff=None,
                 normalization_callback=None):
        self.source = image.bytearray()
        self.source_width, self.source_height = image.width(), image.height()
        self.roi = (0, 0, self.source_width, self.source_height)
        self.output_width = None
        self.output_height = None
        self.capture_tensor = ("reference" if capture_tensor == "reference"
                               else bool(capture_tensor))
        self.capture_buffer = capture_buffer
        self.capture_callback = capture_callback
        self.clock_us = clock_us
        self.clock_diff = clock_diff
        self.preprocess_us = None
        self.capture_result = None
        self.capture_error = None
        self.input_sha256 = None
        self.input_bytes = None
        self.input_byte_count = 0
        if normalization_callback is None:
            from ml.preprocessing import Normalization
            normalization_callback = Normalization()(image)
        self._normalization = normalization_callback

    def __call__(self, input_buffer, shape, dtype):
        started = self.clock_us() if self.clock_us is not None else None
        self._normalization(input_buffer, shape, dtype)
        if started is not None:
            ended = self.clock_us()
            self.preprocess_us = (self.clock_diff(ended, started)
                if self.clock_diff is not None else ended - started)
        self._record_input(input_buffer, shape, dtype)
        return input_buffer


def pillow_bilinear_model_input(image, roi=None, output_width=96, output_height=96,
                                capture_tensor=False, capture_buffer=None,
                                capture_callback=None, clock_us=None,
                                clock_diff=None):
    """Return an OpenMV Model.predict() input callback using exact C-fixture resize."""
    return PillowBilinearModelInput(image, roi, output_width, output_height,
                                    capture_tensor=capture_tensor,
                                    capture_buffer=capture_buffer,
                                    capture_callback=capture_callback,
                                    clock_us=clock_us, clock_diff=clock_diff)


def openmv_normalization_model_input(image, capture_tensor=False,
                                     capture_buffer=None, capture_callback=None,
                                     clock_us=None, clock_diff=None,
                                     normalization_callback=None):
    """Return a capture-aware wrapper around OpenMV's native RGB preprocessing."""
    return OpenMVNormalizationModelInput(image, capture_tensor=capture_tensor,
        capture_buffer=capture_buffer, capture_callback=capture_callback,
        clock_us=clock_us, clock_diff=clock_diff,
        normalization_callback=normalization_callback)


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value == float("inf") or value == -float("inf"):
        return None
    return value


def binary32(value):
    """Round one finite score or threshold to the board's IEEE-754 float32."""
    value = finite_number(value)
    if value is None:
        return None
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except Exception:
        return None


def single_tensor_metadata(value, name="tensor_metadata"):
    """OpenMV exposes model tensor metadata as one value per tensor."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(name + "_requires_exactly_one_tensor")
        return value[0]
    return value


def openmv_dtype_name(value):
    """Map OpenMV dtype codes while preserving the int8/uint8 case."""
    value = single_tensor_metadata(value, "dtype")
    if isinstance(value, bytes):
        value = value.decode("ascii", "strict")
    mapping = {"b": "int8", "B": "uint8", "h": "int16", "H": "uint16",
               "f": "float32", "int8": "int8", "uint8": "uint8",
               "int16": "int16", "uint16": "uint16", "float32": "float32",
               "float": "float32"}
    return mapping.get(value, str(value))


def prediction_values(values, expected_width):
    """Validate OpenMV's default predict() values without dequantizing twice."""
    if len(values) != int(expected_width):
        raise ValueError("identity_output_width_mismatch")
    result = [finite_number(value) for value in values]
    if any(value is None for value in result):
        raise ValueError("nonfinite_model_output")
    return result


def class_index_from_labels(labels, output_count, target_label):
    """Resolve FOMO class labels with or without an explicit background row."""
    output_count = int(output_count)
    if output_count < 1:
        raise ValueError("detector_output_count_invalid")
    normalized = [str(value).strip().lower() for value in (labels or [])]
    target = str(target_label).strip().lower()
    offset = 0
    if len(normalized) == output_count - 1:
        offset = 1
    elif normalized and len(normalized) != output_count:
        raise ValueError("detector_label_count_mismatch")
    for index, label in enumerate(normalized):
        if label == target:
            mapped = index + offset
            if mapped >= output_count:
                raise ValueError("detector_label_index_out_of_range")
            return mapped
    if not normalized and output_count == 2 and target == "face":
        return 1
    raise ValueError("detector_target_label_unmapped")


def fomo_face_output_index(labels, tensor_channel_count, postprocessed_channel_count):
    """Map FOMO's grouped detections, which may omit the background group."""
    tensor_channel_count = int(tensor_channel_count)
    postprocessed_channel_count = int(postprocessed_channel_count)
    face_channel = class_index_from_labels(labels, tensor_channel_count, "face")
    if postprocessed_channel_count == tensor_channel_count:
        return face_channel
    if postprocessed_channel_count == tensor_channel_count - 1 and face_channel > 0:
        return face_channel - 1
    raise ValueError("detector_postprocess_output_count_invalid")


def strict_score_pass(score, threshold):
    """C's exported FOMO contract uses strict greater-than score gating."""
    score, threshold = binary32(score), binary32(threshold)
    if score is None or threshold is None:
        return False
    return score > threshold


def merge_near_duplicate_fomo_peaks(detections, max_center_span_px=48.0,
                                    enabled=False):
    """Optionally collapse a pair of tightly spaced FOMO peaks into one candidate.

    Call after applying the detector score gate. More than two accepted peaks,
    or a pair outside the camera-pixel span, remain ambiguous. Merging is
    disabled by default because two nearby people have not been calibrated.
    Each detection is a mapping with ``center`` (x, y) and ``score`` fields.
    When explicitly enabled, the higher-scoring peak is retained and the
    caller must record the merge event.
    """
    if not isinstance(detections, (list, tuple)):
        raise ValueError("fomo_detections_must_be_a_sequence")
    span_limit = finite_number(max_center_span_px)
    if span_limit is None or span_limit <= 0.0:
        raise ValueError("fomo_merge_span_must_be_finite_positive")
    if len(detections) != 2:
        return list(detections), False, None
    centers, scores = [], []
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("fomo_detection_must_be_a_mapping")
        center = detection.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("fomo_detection_center_invalid")
        x, y = finite_number(center[0]), finite_number(center[1])
        score = binary32(detection.get("score"))
        if x is None or y is None or score is None:
            raise ValueError("fomo_detection_geometry_or_score_invalid")
        centers.append((x, y))
        scores.append(score)
    dx, dy = centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]
    span = math.sqrt(dx * dx + dy * dy)
    if enabled and span <= span_limit:
        selected = 0 if scores[0] >= scores[1] else 1
        return [detections[selected]], True, span
    return list(detections), False, span


def dequantize(values, scale, zero_point, output_dtype):
    """Dequantize raw tensor values exactly once using model output metadata."""
    scale = finite_number(scale)
    if output_dtype in ("b", "B", "h", "H", "int8", "uint8", "int16", "uint16"):
        if scale is None or scale <= 0.0:
            raise ValueError("invalid_output_quantization")
        zero_point = int(zero_point)
        result = [(float(int(value) - zero_point) * scale) for value in values]
    elif output_dtype in ("f", "float", "float32", "<class 'numpy.float32'>"):
        result = [finite_number(value) for value in values]
        if any(value is None for value in result):
            raise ValueError("nonfinite_model_output")
    else:
        raise ValueError("unsupported_output_dtype:" + str(output_dtype))
    if any(finite_number(value) is None for value in result):
        raise ValueError("nonfinite_model_output")
    return result


def stable_softmax(logits):
    """Softmax once for a true-logit tensor; max subtraction avoids overflow."""
    values = [finite_number(value) for value in logits]
    if not values or any(value is None for value in values):
        raise ValueError("invalid_classifier_logits")
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    if finite_number(total) is None or total <= 0.0:
        raise ValueError("invalid_softmax_normalizer")
    return [value / total for value in exps]


def normalize_embedding(vector):
    values = [finite_number(value) for value in vector]
    if len(values) != 64 or any(value is None for value in values):
        raise ValueError("invalid_embedding_dimension_or_value")
    norm = math.sqrt(sum(value * value for value in values))
    if finite_number(norm) is None or norm <= 0.0:
        raise ValueError("zero_embedding_norm")
    return [value / norm for value in values]


def classifier_decision(logits, labels, threshold, uncertainty_band=0.025,
                        other_label="OTHER"):
    """E's six-logit route: K01..K05+OTHER, max-known-softmax calibration."""
    if len(logits) != 6 or len(labels) != 6 or len(set(labels)) != 6:
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": "classifier_shape_or_label_order_invalid"}
    if list(labels) != ["K01", "K02", "K03", "K04", "K05", other_label]:
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": "classifier_label_order_mismatch"}
    if labels[-1].strip().upper() != other_label.strip().upper():
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": "classifier_other_label_not_last"}
    threshold = binary32(threshold)
    band = binary32(uncertainty_band)
    if threshold is None or band is None or not 0.0 <= threshold <= 1.0 or band < 0.0:
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": "invalid_classifier_calibration"}
    try:
        probabilities = [binary32(value) for value in stable_softmax(logits)]
        if any(value is None for value in probabilities):
            raise ValueError("nonfinite_classifier_probability")
    except ValueError as error:
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": str(error)}
    best_known = max(range(5), key=lambda index: probabilities[index])
    top_index = max(range(6), key=lambda index: probabilities[index])
    score = probabilities[best_known]
    margin = binary32(score - probabilities[5])
    upper = binary32(threshold + band)
    lower = binary32(threshold - band)
    if upper is None or lower is None or margin is None:
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": "invalid_classifier_binary32_boundary"}
    if top_index == 5:
        status, identity, reason = "OTHER", None, "other_logit_argmax"
    elif score >= upper:
        status, identity, reason = "KNOWN_IDENTITY", labels[best_known], "above_calibrated_threshold_band"
    elif score <= lower:
        status, identity, reason = "OTHER", None, "below_calibrated_threshold_band"
    else:
        status, identity, reason = "UNCERTAIN", None, "inside_calibrated_threshold_band"
    return {"status": status, "identity_id": identity,
            "candidate_identity_id": labels[best_known], "candidate_score": score,
            "other_probability": probabilities[5], "known_other_margin": margin,
            "class_probabilities": probabilities, "reason": reason}


def gallery_decision(vector, entries, threshold, uncertainty_band=0.025,
                     expected_model_id=None, acceptance_uses_calibrated_cutoff=False):
    """Maximum-cosine embedding route with a low-allocation fixed-gallery scan.

    The Nicla's MicroPython heap can be fragmented while a model and its
    calibration gallery are resident.  Normalize the query once, then compute
    each template's norm and cosine using scalar accumulators instead of
    allocating five temporary 64-float lists and a sorted candidate table.
    The ordering and float32 score boundaries match the original route.
    """
    # Keep the query norm scalar instead of allocating two additional 64-item
    # lists through normalize_embedding(). On the Nicla this routine runs while
    # the identity model arena is resident, and a short-lived list can fail
    # despite enough aggregate free heap.
    try:
        if len(vector) != 64:
            raise ValueError("invalid_embedding_dimension_or_value")
        query_norm_sq = 0.0
        for index in range(64):
            value = finite_number(vector[index])
            if value is None:
                raise ValueError("invalid_embedding_dimension_or_value")
            query_norm_sq += value * value
        query_norm = math.sqrt(query_norm_sq)
        if finite_number(query_norm) is None or query_norm <= 0.0:
            raise ValueError("zero_embedding_norm")
    except (TypeError, ValueError, IndexError) as error:
        return {"status": "UNCERTAIN", "identity_id": None, "reason": str(error)}
    if not isinstance(entries, list) or not entries:
        return {"status": "UNCERTAIN", "identity_id": None, "reason": "gallery_missing"}
    threshold = binary32(threshold)
    band = binary32(uncertainty_band)
    if threshold is None or band is None or threshold < -1.0 or threshold > 1.0 or band < 0.0:
        return {"status": "UNCERTAIN", "identity_id": None, "reason": "invalid_gallery_calibration"}
    seen = []
    best_score = None
    best_identity = None
    second_score = None
    second_identity = None
    entry_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            return {"status": "UNCERTAIN", "identity_id": None, "reason": "invalid_gallery_entry"}
        identity = entry.get("identity_id")
        vectors = entry.get("vectors")
        if not isinstance(identity, str) or not identity or identity in seen or not isinstance(vectors, dict):
            return {"status": "UNCERTAIN", "identity_id": None, "reason": "invalid_gallery_identity_map"}
        seen.append(identity)
        entry_count += 1
        if expected_model_id is None:
            if len(vectors) != 1:
                return {"status": "UNCERTAIN", "identity_id": None, "reason": "model_index_required"}
            model_id = next(iter(vectors))
        else:
            model_id = expected_model_id
        if model_id not in vectors:
            return {"status": "UNCERTAIN", "identity_id": None, "reason": "gallery_model_vector_missing"}
        template = vectors[model_id]
        if not isinstance(template, (list, tuple)) or len(template) != 64:
            return {"status": "UNCERTAIN", "identity_id": None,
                    "reason": "invalid_gallery_vector:invalid_embedding_dimension_or_value"}
        norm_sq = 0.0
        for index in range(64):
            value = finite_number(template[index])
            if value is None:
                return {"status": "UNCERTAIN", "identity_id": None,
                        "reason": "invalid_gallery_vector:invalid_embedding_dimension_or_value"}
            norm_sq += value * value
        norm = math.sqrt(norm_sq)
        if finite_number(norm) is None or norm <= 0.0:
            return {"status": "UNCERTAIN", "identity_id": None,
                    "reason": "invalid_gallery_vector:zero_embedding_norm"}
        dot = 0.0
        for index in range(64):
            value = finite_number(template[index])
            dot += (vector[index] / query_norm) * (value / norm)
        similarity = binary32(dot)
        if similarity is None:
            return {"status": "UNCERTAIN", "identity_id": None,
                    "reason": "nonfinite_gallery_similarity"}
        # Python's prior reverse tuple sort ordered ties by identity descending;
        # preserve that behavior while retaining only the best two values.
        if (best_score is None or similarity > best_score or
                (similarity == best_score and identity > best_identity)):
            second_score, second_identity = best_score, best_identity
            best_score, best_identity = similarity, identity
        elif (second_score is None or similarity > second_score or
                (similarity == second_score and identity > second_identity)):
            second_score, second_identity = similarity, identity
    if entry_count != 5:
        return {"status": "UNCERTAIN", "identity_id": None, "reason": "fixed_roster_size_mismatch"}
    score, candidate = best_score, best_identity
    if second_score is None:
        return {"status": "UNCERTAIN", "identity_id": None, "reason": "fixed_roster_size_mismatch"}
    margin = binary32(score - second_score)
    # When the calibration optimized the actual acceptance cutoff, a band
    # below that cutoff can mark uncertain scores without moving the validated
    # acceptance boundary above the maximum possible cosine (1.0).
    upper = threshold if acceptance_uses_calibrated_cutoff else binary32(threshold + band)
    lower = binary32(threshold - band)
    if margin is None or upper is None or lower is None:
        return {"status": "UNCERTAIN", "identity_id": None,
                "reason": "invalid_gallery_binary32_boundary"}
    if score >= upper:
        status, identity, reason = "KNOWN_IDENTITY", candidate, "above_gallery_threshold_band"
    elif score <= lower:
        status, identity, reason = "OTHER", None, "below_gallery_threshold_band"
    else:
        status, identity, reason = "UNCERTAIN", None, "inside_gallery_threshold_band"
    return {"status": status, "identity_id": identity,
            "candidate_identity_id": candidate, "candidate_score": score,
            "nearest_competitor_margin": margin, "reason": reason,
            "gallery_size": len(entries)}


def gate_unqualified_identity(decision, allow_known_acceptance):
    """Keep the candidate and score while blocking unqualified known accepts."""
    if allow_known_acceptance or decision.get("status") != "KNOWN_IDENTITY":
        return decision
    gated = dict(decision)
    gated["candidate_identity_id"] = decision.get("candidate_identity_id") or decision.get("identity_id")
    gated.update({"status": "UNCERTAIN", "identity_id": None,
                  "reason": "candidate_above_cutoff_acceptance_unqualified"})
    return gated


def crop_xywh(center_x, center_y, side, frame_width, frame_height):
    """C's candidate fixed crop: floor(center - side/2 + .5), no clamp/pad."""
    if side <= 0 or frame_width <= 0 or frame_height <= 0:
        return None, "invalid_crop_geometry"
    x = int(math.floor(float(center_x) - side / 2.0 + 0.5))
    y = int(math.floor(float(center_y) - side / 2.0 + 0.5))
    if x < 0 or y < 0 or x + side > frame_width or y + side > frame_height:
        return (x, y, side, side), "crop_outside_frame"
    return (x, y, side, side), None


def fomo_center_camera_px(rect_xywh, frame_width, frame_height):
    """Map an OpenMV postprocessed FOMO rectangle to its camera-pixel center.

    OpenMV's postprocessor has already mapped its detector-grid rectangle back
    to the original image. Do not multiply the returned coordinates by the
    camera-to-detector input ratio again.
    """
    if len(rect_xywh) != 4:
        raise ValueError("FOMO rectangle must be x,y,width,height")
    x, y, width, height = [float(value) for value in rect_xywh]
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("invalid FOMO camera rectangle")
    if width <= 0 or height <= 0:
        raise ValueError("invalid FOMO camera rectangle")
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("FOMO camera rectangle outside frame")
    return (x + width / 2.0, y + height / 2.0)


def centered_roi_clamped(center_xy_camera_px, side_camera_px, frame_width, frame_height):
    """Return a fixed-size square ROI slid inward at borders, with no padding."""
    side = int(side_camera_px)
    if side <= 0 or side > frame_width or side > frame_height:
        raise ValueError("ROI side must fit the camera frame")
    center_x, center_y = [float(value) for value in center_xy_camera_px]
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise ValueError("invalid camera center")
    x = int(math.floor(center_x - side / 2.0))
    y = int(math.floor(center_y - side / 2.0))
    x = min(max(0, x), frame_width - side)
    y = min(max(0, y), frame_height - side)
    return (x, y, side, side)


def crop_for_boundary_policy(center_xy_camera_px, side_camera_px, frame_width,
                             frame_height, boundary_policy):
    """Select the configured edge policy while keeping the crop size fixed."""
    center_x, center_y = center_xy_camera_px
    if boundary_policy == "slide_inward_fixed_size_no_padding":
        try:
            return centered_roi_clamped(center_xy_camera_px, side_camera_px,
                                        frame_width, frame_height), None
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_crop_geometry"
    if boundary_policy == "reject_without_clamp_or_padding":
        return crop_xywh(center_x, center_y, side_camera_px,
                         frame_width, frame_height)
    return None, "unknown_crop_boundary_policy"


def face_box_square_crop(rect_xywh, frame_width, frame_height, padding_ratio=1.5):
    """Keep a complete camera-window face box in a padded, in-bounds square."""
    if not isinstance(rect_xywh, (list, tuple)) or len(rect_xywh) != 4:
        return None, "invalid_face_box"
    x, y, width, height = [finite_number(value) for value in rect_xywh]
    ratio = finite_number(padding_ratio)
    if (None in (x, y, width, height, ratio) or width <= 0 or height <= 0 or
            ratio < 1 or ratio > 3 or frame_width <= 0 or frame_height <= 0):
        return None, "invalid_face_box"
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        return None, "face_box_outside_frame"
    left, top = int(math.floor(x)), int(math.floor(y))
    right, bottom = int(math.ceil(x + width)), int(math.ceil(y + height))
    face_side = max(right - left, bottom - top)
    max_side = min(frame_width, frame_height)
    if face_side > max_side:
        return None, "face_box_cannot_fit_square"
    side = min(max_side, int(math.ceil(face_side * ratio)))
    # Constrain the translation against both the image bounds and face edges.
    crop_x = min(left, frame_width - side, max(0, right - side, (left + right - side) // 2))
    crop_y = min(top, frame_height - side, max(0, bottom - side, (top + bottom - side) // 2))
    return (crop_x, crop_y, side, side), None


def normalized_face_box(values, frame_width, frame_height):
    """Decode dequantized [objectness,cx,cy,width,height] in camera-window space."""
    score, cx, cy, width, height = prediction_values(values, 5)
    if not (0.0 <= score <= 1.0 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and
            0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("invalid_normalized_face_box")
    return score, [(cx - width / 2.0) * frame_width,
                   (cy - height / 2.0) * frame_height,
                   width * frame_width, height * frame_height]


def quality_decision(mean_u8, stdev_u8, min_mean=12, max_mean=243, min_stdev=5):
    mean_u8, stdev_u8 = finite_number(mean_u8), finite_number(stdev_u8)
    if mean_u8 is None or stdev_u8 is None:
        return False, "quality_statistics_missing"
    if mean_u8 < min_mean:
        return False, "too_dim"
    if mean_u8 > max_mean:
        return False, "saturated"
    if stdev_u8 < min_stdev:
        return False, "low_contrast_or_blank"
    return True, "quality_gate_pass"

