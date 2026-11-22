"""Host-testable low-allocation pieces of the Nicla application runtime."""

import math
from identity_contracts import finite_number


class BoundedLineQueue:
    """Independent per-output FIFO, capped by bytes and complete records."""

    def __init__(self, max_bytes=4096, max_record_bytes=768):
        self.max_bytes = int(max_bytes)
        self.max_record_bytes = int(max_record_bytes)
        self.records = []
        self.priority_flags = []
        self.queued_bytes = 0
        self.dropped_records = 0
        self.dropped_oversize = 0
        # If a peripheral accepts only the prefix of a write, keep that
        # continuation at the head until it is sent. Dropping it would splice
        # the next JSON record onto an unterminated prefix on the wire.
        self._continuation_head = False

    def _oldest_normal_index(self):
        start = 1 if self._continuation_head else 0
        for index in range(start, len(self.records)):
            if not self.priority_flags[index]:
                return index
        return -1

    def _drop_index(self, index):
        old = self.records.pop(index)
        self.priority_flags.pop(index)
        self.queued_bytes -= len(old)
        self.dropped_records += 1

    def push(self, line):
        if isinstance(line, str):
            line = line.encode("utf-8")
        if not isinstance(line, bytes):
            raise TypeError("queue accepts bytes or text")
        if not line.endswith(b"\n"):
            line += b"\n"
        if len(line) > self.max_record_bytes or len(line) > self.max_bytes:
            self.dropped_oversize += 1
            return False
        # Leave one maximum-size record reserved for a control ACK.
        normal_limit = max(0, self.max_bytes - min(self.max_record_bytes, self.max_bytes))
        while self.queued_bytes + len(line) > normal_limit:
            drop_index = self._oldest_normal_index()
            if drop_index < 0:
                self.dropped_records += 1
                return False
            self._drop_index(drop_index)
        self.records.append(line)
        self.priority_flags.append(False)
        self.queued_bytes += len(line)
        return True

    def push_priority(self, line):
        """Queue control acknowledgments ahead of optional telemetry records."""
        if isinstance(line, str):
            line = line.encode("utf-8")
        if not isinstance(line, bytes):
            raise TypeError("queue accepts bytes or text")
        if not line.endswith(b"\n"):
            line += b"\n"
        if len(line) > self.max_record_bytes or len(line) > self.max_bytes:
            self.dropped_oversize += 1
            return False
        while self.queued_bytes + len(line) > self.max_bytes:
            drop_index = self._oldest_normal_index()
            if drop_index < 0:
                self.dropped_records += 1
                return False
            self._drop_index(drop_index)
        # Append in sequence order. Capacity reserved by push() makes the ACK
        # deliverable without overtaking telemetry already assigned lower IDs.
        self.records.append(line)
        self.priority_flags.append(True)
        self.queued_bytes += len(line)
        return True

    def take(self, maximum_bytes):
        maximum_bytes = max(0, int(maximum_bytes))
        if not maximum_bytes or not self.records:
            return b""
        # Keep ordinary queued records intact. A single record larger than the
        # flush budget is returned whole so a caller never creates a prefix
        # that can later be evicted from the bounded queue.
        if self._continuation_head:
            data = self.records.pop(0)
            self.priority_flags.pop(0)
            self.queued_bytes -= len(data)
            self._continuation_head = False
            return data
        chunks = []
        remaining = maximum_bytes
        while self.records and remaining:
            head = self.records[0]
            if len(head) > remaining and chunks:
                break
            chunks.append(head)
            remaining -= len(head)
            self.queued_bytes -= len(head)
            self.records.pop(0)
            self.priority_flags.pop(0)
            if remaining < 0:
                break
        return b"".join(chunks)

    def push_front(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not data:
            return True
        if len(data) > self.max_bytes:
            self.dropped_oversize += 1
            return False
        self.records.insert(0, data)
        self.priority_flags.insert(0, True)
        self.queued_bytes += len(data)
        self._continuation_head = True
        while len(self.records) > 1 and self.queued_bytes > self.max_bytes:
            drop_index = self._oldest_normal_index()
            if drop_index < 0:
                drop_index = 1
            self._drop_index(drop_index)
        if self.queued_bytes > self.max_bytes:
            # The continuation is bounded by max_bytes above, so this is only
            # reachable if the queue was already inconsistent.
            self.clear()
            self.dropped_records += 1
            return False
        return True

    def clear(self):
        count = len(self.records)
        self.records = []
        self.priority_flags = []
        self.queued_bytes = 0
        self._continuation_head = False
        return count


class TemporalConfirmer:
    """Require the same known identity twice in the latest three decisions."""

    def __init__(self, mode=False):
        self.mode = mode
        self.history = []

    def reset(self):
        self.history = []

    def apply(self, decision):
        if not self.mode:
            return decision
        status = decision.get("status")
        identity = decision.get("identity_id")
        if status != "KNOWN_IDENTITY" or not identity:
            self.history = []
            return decision
        self.history.append((status, identity))
        self.history = self.history[-3:]
        count = sum(1 for item in self.history
                    if item[0] == "KNOWN_IDENTITY" and item[1] == identity)
        if count >= 2:
            result = dict(decision)
            result["temporal_confirmation"] = "2_of_3_confirmed"
            return result
        return {"status": "UNCERTAIN", "identity_id": None,
                "candidate_identity_id": identity,
                "reason": "temporal_confirmation_not_met",
                "temporal_confirmation": "2_of_3_pending"}


class TickTracker:
    """Track wrapping MicroPython millisecond ticks and boot/reset identity."""

    def __init__(self, modulus):
        self.modulus = int(modulus)
        if self.modulus <= 1:
            raise ValueError("invalid tick modulus")
        self.last = None
        self.wraps = 0
        self.resets = 0

    def observe(self, ticks):
        ticks = int(ticks)
        if ticks < 0 or ticks >= self.modulus:
            raise ValueError("tick outside declared modulus")
        event = None
        if self.last is not None and ticks < self.last:
            if self.last - ticks > self.modulus // 2:
                self.wraps += 1
                event = "wrap"
            else:
                self.resets += 1
                event = "reset_or_clock_discontinuity"
        self.last = ticks
        return {"event": event, "wraps": self.wraps, "resets": self.resets,
                "extended_ticks": self.wraps * self.modulus + ticks}


def motion_detected(acceleration_mg, angular_rate_mdps,
                    acceleration_delta_mg=120, angular_rate_mdps_limit=18000):
    """Wearer/device motion only; image brightness is deliberately not used."""
    if acceleration_mg is not None:
        try:
            values = [float(value) for value in acceleration_mg]
            magnitude = math.sqrt(sum(value * value for value in values))
            if abs(magnitude - 1000.0) > float(acceleration_delta_mg):
                return True, "acceleration_delta"
        except (TypeError, ValueError):
            pass
    if angular_rate_mdps is not None:
        try:
            if max(abs(float(value)) for value in angular_rate_mdps) > float(angular_rate_mdps_limit):
                return True, "angular_rate"
        except (TypeError, ValueError):
            pass
    return False, "motion_data_missing_or_below_threshold"


def validate_gallery_metadata(gallery, expected_model_id, expected_model_sha256,
                              expected_prototype_sha256, roster_ids):
    """Fail closed unless the actual fixed five-entry D gallery is bound."""
    if not isinstance(gallery, dict) or gallery.get("status") != "ready":
        return False, "gallery_not_ready"
    if gallery.get("model_id") != expected_model_id:
        return False, "gallery_model_id_mismatch"
    if gallery.get("model_sha256") != expected_model_sha256:
        return False, "gallery_model_hash_mismatch"
    if gallery.get("prototype_sha256") != expected_prototype_sha256:
        return False, "gallery_prototype_hash_mismatch"
    if gallery.get("identity_order") != list(roster_ids):
        return False, "gallery_identity_order_mismatch"
    entries = gallery.get("entries")
    if not isinstance(entries, list) or len(entries) != len(roster_ids):
        return False, "gallery_size_mismatch"
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("identity_id") != roster_ids[index]:
            return False, "gallery_entry_order_mismatch"
        vectors = entry.get("vectors")
        if not isinstance(vectors, dict) or expected_model_id not in vectors:
            return False, "gallery_model_vector_missing"
        vector = vectors[expected_model_id]
        if not isinstance(vector, list) or len(vector) != 64:
            return False, "gallery_vector_shape_invalid"
        if any(finite_number(value) is None for value in vector):
            return False, "gallery_vector_nonfinite"
    return True, "gallery_valid"


def classifier_route_for_embedding(decision, threshold, uncertainty_band):
    """Return whether A-model should pay for E after an actual D score."""
    if decision.get("status") == "UNCERTAIN":
        return True, "embedding_uncertain"
    score = finite_number(decision.get("candidate_score"))
    threshold = finite_number(threshold)
    band = finite_number(uncertainty_band)
    if score is None or threshold is None or band is None:
        return False, "invalid_embedding_score_fails_closed"
    if abs(score - threshold) <= band:
        return True, "embedding_near_threshold"
    return False, "embedding_decision_clear"


def calibrated_risk_escalation(risk_table, score, risk_threshold):
    """Look up development-calibrated small-model error risk by score bin.

    Bins are supplied only after a permitted development fit. Missing,
    malformed, overlapping, or out-of-range calibration fails closed and does
    not spend the larger-model budget.
    """
    score = finite_number(score)
    threshold = finite_number(risk_threshold)
    if score is None or threshold is None or not isinstance(risk_table, list):
        return False, "risk_calibration_unavailable", None
    matches = []
    for row in risk_table:
        if not isinstance(row, dict):
            continue
        lower = finite_number(row.get("score_min"))
        upper = finite_number(row.get("score_max"))
        risk = finite_number(row.get("small_error_probability"))
        if lower is None or upper is None or risk is None or lower > upper or not 0 <= risk <= 1:
            continue
        if lower <= score and (score < upper or (upper == 1.0 and score <= upper)):
            matches.append(risk)
    if len(matches) != 1:
        return False, "risk_score_bin_missing_or_ambiguous", None
    risk = matches[0]
    reason = "calibrated_small_model_risk" if risk >= threshold else "calibrated_risk_below_threshold"
    return risk >= threshold, reason, risk


def cost_aware_escalation(expected_accuracy_gain, incremental_energy_mj,
                          incremental_latency_ms, energy_value_mj_per_correct,
                          remaining_latency_ms):
    """Escalate only when measured expected gain is worth measured cost.

    ``energy_value_mj_per_correct`` is the frozen maximum extra energy the
    study is willing to spend for one additional correct decision. Inputs must
    come from development-only calibration and measured H/I costs.
    """
    gain = finite_number(expected_accuracy_gain)
    energy = finite_number(incremental_energy_mj)
    latency = finite_number(incremental_latency_ms)
    value = finite_number(energy_value_mj_per_correct)
    remaining = finite_number(remaining_latency_ms)
    if any(item is None for item in (gain, energy, latency, value, remaining)):
        return False, "measured_cost_calibration_unavailable"
    if gain < 0 or energy < 0 or latency < 0 or value <= 0 or remaining < 0:
        return False, "invalid_measured_cost_calibration"
    if latency > remaining:
        return False, "response_budget_would_be_exceeded"
    if gain * value <= energy:
        return False, "expected_gain_does_not_justify_energy"
    return True, "measured_expected_gain_exceeds_incremental_cost"


SCENE_SIGNATURE_SIDE = 8
SCENE_SIGNATURE_LENGTH = SCENE_SIGNATURE_SIDE ** 2


def sample_gray_scene_signature(pixels, width, height, target):
    """Fill a reusable bytearray with four quarter-cell samples per grid cell."""
    side = SCENE_SIGNATURE_SIDE
    if not isinstance(target, bytearray) or len(target) != SCENE_SIGNATURE_LENGTH:
        raise ValueError("expected a reusable 64-byte scene signature buffer")
    if width < 2 * side or height < 2 * side or len(pixels) != width * height:
        return None
    divisor = 4 * side
    index = 0
    for row in range(side):
        top = ((4 * row + 1) * height // divisor) * width
        bottom = ((4 * row + 3) * height // divisor) * width
        for column in range(side):
            left = (4 * column + 1) * width // divisor
            right = (4 * column + 3) * width // divisor
            total = (pixels[top + left] + pixels[top + right]
                     + pixels[bottom + left] + pixels[bottom + right])
            target[index] = (total + 2) // 4
            index += 1
    return target


class SimpleSceneTrigger:
    """Low-cost image-statistics trigger with a bounded detector fallback.

    The IMU is deliberately not used to infer that a person entered the scene.
    A periodic detector check remains mandatory even when the scene is still.
    ``ticks_diff`` must be the device's wrap-safe clock-difference function.
    """

    def __init__(self, config, ticks_diff, ticks_add=None):
        self.config = config
        self.ticks_diff = ticks_diff
        spatial = config.get("spatial")
        self.spatial_enabled = spatial is not None
        if self.spatial_enabled:
            self.spatial_cell_delta_u8 = int(spatial.get("cell_delta_u8", 12))
            self.spatial_onset_cells = int(spatial.get("onset_cells", 3))
            self.spatial_hysteresis_cells = int(spatial.get("hysteresis_cells", 1))
            if not (1 <= self.spatial_cell_delta_u8 <= 255 and
                    0 <= self.spatial_hysteresis_cells < self.spatial_onset_cells <= SCENE_SIGNATURE_LENGTH):
                raise ValueError("invalid spatial scene trigger thresholds")
        self.reset_reference()
        self.detector_calls = 0
        self.skipped_frames = 0

    def reset_reference(self):
        """Force a fresh observation after a camera discontinuity."""
        self.reference_mean = None
        self.reference_stdev = None
        self.reference_signature = None
        self.last_detector_ticks = None
        self.pending_change = False

    def _result(self, run, reason, delta_mean=None, delta_stdev=None, periodic=False, changed_cells=None):
        result = {"run_detector": run, "reason": reason, "delta_mean_u8": delta_mean,
                  "delta_stdev_u8": delta_stdev, "periodic_fallback": periodic}
        if self.spatial_enabled:
            result["spatial_changed_cells"] = changed_cells
        return result

    def decide(self, mean_u8, stdev_u8, now_ticks, spatial_signature=None):
        if mean_u8 is None or stdev_u8 is None:
            if self.spatial_enabled:
                self.reset_reference()
            self.detector_calls += 1
            self.last_detector_ticks = now_ticks
            return self._result(True, "invalid_statistics_fail_open")

        if self.spatial_enabled and (not isinstance(spatial_signature, (bytes, bytearray))
                                     or len(spatial_signature) != SCENE_SIGNATURE_LENGTH):
            self.reset_reference()
            self.detector_calls += 1
            self.last_detector_ticks = now_ticks
            return self._result(True, "invalid_spatial_signature_fail_open")

        mean_u8 = float(mean_u8)
        stdev_u8 = float(stdev_u8)
        if self.reference_mean is None or self.reference_stdev is None:
            self.reference_mean = mean_u8
            self.reference_stdev = stdev_u8
            if self.spatial_enabled:
                self.reference_signature = bytes(spatial_signature)
            self.last_detector_ticks = now_ticks
            self.detector_calls += 1
            return self._result(True, "initial_scene_check")

        delta_mean = abs(mean_u8 - self.reference_mean)
        delta_stdev = abs(stdev_u8 - self.reference_stdev)
        onset_mean = float(self.config.get("mean_delta_u8", 8))
        onset_stdev = float(self.config.get("stdev_delta_u8", 6))
        reset_mean = float(self.config.get("hysteresis_mean_u8", 4))
        reset_stdev = float(self.config.get("hysteresis_stdev_u8", 3))
        min_interval = int(self.config.get("min_detector_interval_ms", 500))
        fallback = int(self.config.get("detector_fallback_ms", 2000))
        age = self.ticks_diff(now_ticks, self.last_detector_ticks)
        periodic = age >= fallback
        changed = delta_mean >= onset_mean or delta_stdev >= onset_stdev
        returned_to_reference = delta_mean <= reset_mean and delta_stdev <= reset_stdev
        changed_cells = None
        if self.spatial_enabled:
            changed_cells = 0
            for index, previous in enumerate(self.reference_signature):
                if abs(spatial_signature[index] - previous) >= self.spatial_cell_delta_u8:
                    changed_cells += 1
            changed = changed or changed_cells >= self.spatial_onset_cells
            returned_to_reference = returned_to_reference and changed_cells <= self.spatial_hysteresis_cells

        if changed:
            self.pending_change = True
        elif returned_to_reference:
            self.pending_change = False

        reason = None
        if periodic:
            reason = "bounded_periodic_detector_fallback"
        elif self.pending_change and age >= min_interval:
            reason = "scene_change_after_minimum_interval"

        if reason:
            self.reference_mean = mean_u8
            self.reference_stdev = stdev_u8
            if self.spatial_enabled:
                self.reference_signature = bytes(spatial_signature)
            self.last_detector_ticks = now_ticks
            self.pending_change = False
            self.detector_calls += 1
            return self._result(True, reason, delta_mean, delta_stdev, periodic, changed_cells)

        self.skipped_frames += 1
        reason = "scene_change_waiting_for_cooldown" if self.pending_change else "scene_stable"
        return self._result(False, reason, delta_mean, delta_stdev, changed_cells=changed_cells)


class EscalationBudget:
    """Sliding-window bound for small-to-large identity escalation calls."""

    def __init__(self, maximum, window_ms, ticks_diff):
        self.maximum = max(0, int(maximum))
        self.window_ms = max(1, int(window_ms))
        self.ticks_diff = ticks_diff
        self.events = []

    def allow(self, now_ticks):
        self.events = [tick for tick in self.events
                       if 0 <= self.ticks_diff(now_ticks, tick) < self.window_ms]
        if len(self.events) >= self.maximum:
            return False, "escalation_rate_limited"
        self.events.append(now_ticks)
        return True, "escalation_budget_available"


def response_deadline_result(decision, elapsed_ms, maximum_ms):
    """Suppress an identity result that arrived outside the declared bound."""
    try:
        late = int(elapsed_ms) > int(maximum_ms)
    except (TypeError, ValueError):
        late = True
    if not late:
        return decision, False
    return {"status": "UNCERTAIN", "identity_id": None,
            "reason": "RESPONSE_DEADLINE_EXCEEDED"}, True


def camera_wake_retry_ms(failure_count, base_ms=1000, maximum_ms=8000):
    """Exponential bounded camera-wake retry delay; failure never halts loop."""
    failures = max(1, int(failure_count))
    base = max(1, int(base_ms))
    maximum = max(base, int(maximum_ms))
    exponent = min(failures - 1, 30)
    return min(maximum, base * (2 ** exponent))


def hash_matches(actual, expected):
    return isinstance(actual, str) and isinstance(expected, str) and len(actual) == 64 and actual.lower() == expected.lower()


def config_hash_matches(actual, expected):
    return hash_matches(actual, expected)


def enough_heap(heap_free_bytes, minimum_bytes):
    try:
        return int(heap_free_bytes) >= int(minimum_bytes)
    except (TypeError, ValueError):
        return False
