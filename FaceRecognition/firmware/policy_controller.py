"""Bounded, reason-coded cadence/model/confirmation controller for Nicla.

This module contains policy decisions only; a platform adapter must perform
camera capture/sleep and model calls. Sensor sleep defaults off until a live
GC2145 wake-to-usable-frame and power path qualification passes.
"""

import math

POLICY_IDS = ("H", "L", "S", "G", "A-acquire", "A-sense", "A-model", "A-full")
FIXED_INTERVAL_MS = {"H": 250, "L": 2000, "S": 250, "G": 250,
                     "A-acquire": 250, "A-model": 1000}
MAX_SAMPLE_GAP_MS = 5000
MIN_INTERVAL_MS = 250
MAX_INTERVAL_MS = 5000
STATIONARY_INTERVAL_MS = 1500
TEMPORAL_WINDOW = 3
TEMPORAL_REQUIRED = 2
MAX_MODEL_ESCALATIONS_PER_MINUTE = 6


class PolicyController:
    def __init__(self, policy_id, ticks_diff=None, allow_camera_sleep=False, features=None):
        if policy_id not in POLICY_IDS:
            raise ValueError("unknown_policy")
        self.policy_id = policy_id
        self.ticks_diff = ticks_diff or (lambda end, start: end - start)
        self.features = {"motion": True, "quality_adaptation": True, "confidence_escalation": True,
                         "camera_sleep": True, "temporal_confirmation": True}
        if features:
            self.features.update(features)
        self.allow_camera_sleep = bool(allow_camera_sleep) and policy_id == "A-full" and self.features["camera_sleep"]
        self.interval_ms = FIXED_INTERVAL_MS.get(policy_id, 1000)
        self.motion_active = False
        self.motion_high_streak = 0
        self.motion_low_streak = 0
        self.no_face_streak = 0
        self.last_frame_ms = None
        self.last_result_ms = None
        self.last_wake_ms = None
        self.timeout_active = False
        self.temporal = []
        self.escalation_times = []
        self.model_switches = 0
        self.counters = {
            "camera_wakes": 0,
            "captures": 0,
            "discarded_quality_frames": 0,
            "detector_calls": 0,
            "small_model_calls": 0,
            "large_model_calls": 0,
            "model_escalations": 0,
            "model_switches": 0,
            "decisions": 0,
            "uncertain_outputs": 0,
            "no_face_outputs": 0,
            "timeouts": 0,
            "camera_sleep_requests": 0,
        }

    @staticmethod
    def motion_score(accel_mg, gyro_mdps):
        """Return wearer/device motion score; it is not scene or person motion."""
        if not isinstance(accel_mg, (list, tuple)) or len(accel_mg) != 3:
            return None
        if not isinstance(gyro_mdps, (list, tuple)) or len(gyro_mdps) != 3:
            return None
        try:
            accel_norm = math.sqrt(sum(float(v) * float(v) for v in accel_mg))
            gyro_norm = math.sqrt(sum(float(v) * float(v) for v in gyro_mdps))
        except Exception:
            return None
        if not math.isfinite(accel_norm) or not math.isfinite(gyro_norm):
            return None
        return max(abs(accel_norm - 1000.0), gyro_norm / 10.0)

    def update_motion(self, score):
        if not self.features["motion"]:
            self.motion_active = False
            return {"motion_active": False, "reason": "motion_ablation_disabled"}
        try:
            score = None if score is None else float(score)
        except (TypeError, ValueError):
            score = None
        if score is None or not math.isfinite(score):
            return {"motion_active": self.motion_active, "reason": "imu_missing"}
        if score >= 150.0:
            self.motion_high_streak += 1
            self.motion_low_streak = 0
            if self.motion_high_streak >= 2:
                self.motion_active = True
        elif score <= 75.0:
            self.motion_low_streak += 1
            self.motion_high_streak = 0
            if self.motion_low_streak >= 3:
                self.motion_active = False
        else:
            self.motion_high_streak = 0
            self.motion_low_streak = 0
        return {"motion_active": self.motion_active,
                "reason": "wearer_motion_high" if self.motion_active else "wearer_motion_low_or_hysteresis"}

    def next_interval(self, quality_ok=None, face_count=0):
        if self.policy_id in FIXED_INTERVAL_MS:
            interval = FIXED_INTERVAL_MS[self.policy_id]
        else:
            interval = 250 if self.motion_active else STATIONARY_INTERVAL_MS
            if quality_ok is False and self.features["quality_adaptation"]:
                interval = MIN_INTERVAL_MS
            elif face_count:
                interval = min(interval, 500)
        self.interval_ms = min(MAX_INTERVAL_MS, max(MIN_INTERVAL_MS, int(interval)))
        reason = "fixed_policy_interval" if self.policy_id in FIXED_INTERVAL_MS else (
            "quality_recapture" if quality_ok is False and self.features["quality_adaptation"] else
            "wearer_motion_fast_sampling" if self.motion_active else
            "periodic_stationary_check" if not face_count else "face_candidate_check")
        return self.interval_ms, reason

    def should_sleep_camera(self, quality_ok, face_count):
        if not self.allow_camera_sleep:
            return False, "sensor_sleep_disabled_until_board_qualification"
        if self.policy_id != "A-full":
            return False, "policy_has_no_sleep"
        if quality_ok is not True:
            self.no_face_streak = 0
            return False, "quality_missing_or_failed"
        if face_count:
            self.no_face_streak = 0
            return False, "face_candidate_present"
        self.no_face_streak += 1
        if (self.no_face_streak >= 3 and not self.motion_active and
                self.interval_ms >= STATIONARY_INTERVAL_MS):
            self.counters["camera_sleep_requests"] += 1
            return True, "stationary_no_face_after_hysteresis"
        return False, "retain_periodic_camera_check"

    def should_run_identity(self, quality_ok=True):
        """Gate only identity inference; sensing and detection remain separate."""
        if self.policy_id == "A-acquire":
            return False, "acquisition_only_policy"
        if quality_ok is not True:
            return False, "missing_or_low_quality_evidence"
        if self.policy_id == "A-sense" and self.motion_active:
            return False, "wearer_motion_identity_gate"
        return True, "identity_enabled"

    def select_models(self, quality_ok, small_similarity=None, small_threshold=None,
                      uncertain_band=0.025, now_ms=None):
        """Return ordered identity-model calls; poor crops never trigger escalation."""
        if not quality_ok:
            return [], "missing_or_low_quality_evidence"
        if self.policy_id == "A-acquire":
            return [], "acquisition_only_policy"
        if self.policy_id in ("H", "L", "G", "A-sense"):
            return ["large"], "fixed_large_model"
        if self.policy_id == "S":
            return ["small"], "fixed_small_model"
        if self.policy_id in ("A-model", "A-full"):
            if small_similarity is None or small_threshold is None:
                return ["small"], "small_model_first"
            try:
                similarity = float(small_similarity)
                threshold = float(small_threshold)
                band = float(uncertain_band)
            except (TypeError, ValueError):
                return ["small"], "invalid_small_model_score_no_escalation"
            if not math.isfinite(similarity) or not math.isfinite(threshold):
                return ["small"], "invalid_small_model_score_no_escalation"
            if not math.isfinite(band) or band < 0.0:
                return ["small"], "invalid_escalation_band"
            if abs(similarity - threshold) <= band:
                if not self.features["confidence_escalation"]:
                    return ["small"], "confidence_escalation_ablated"
                now_ms = 0 if now_ms is None else now_ms
                self._prune_escalations(now_ms)
                if len(self.escalation_times) >= MAX_MODEL_ESCALATIONS_PER_MINUTE:
                    return ["small"], "escalation_rate_limited"
                self.escalation_times.append(now_ms)
                self.counters["model_escalations"] += 1
                self.counters["model_switches"] += 1
                return ["small", "large"], "small_score_in_uncertain_band"
            return ["small"], "small_score_outside_uncertain_band"
        raise ValueError("unsupported_policy")

    def _prune_escalations(self, now_ms):
        self.escalation_times = [t for t in self.escalation_times
                                 if self.ticks_diff(now_ms, t) < 60000]

    def temporal_decision(self, decision, identity_id=None, valid_evidence=True, now_ms=None):
        """Apply 2-of-3 only to valid quality-passing recognition evidence."""
        if self.policy_id not in ("G", "A-full"):
            return decision, "single_frame_policy"
        if not self.features["temporal_confirmation"]:
            return decision, "temporal_confirmation_ablated"
        if not valid_evidence or decision in ("uncertain", "no_face", "multiple_faces", "pipeline_error"):
            self.temporal = []
            return "uncertain", "evidence_not_valid_for_temporal_vote"
        vote = identity_id if decision == "known" else "__unknown__" if decision == "unknown" else None
        if vote is None:
            return "uncertain", "unrecognized_vote_state"
        now_ms = self.last_frame_ms if now_ms is None else now_ms
        if now_ms is not None:
            recent = []
            for stamp, old_vote in self.temporal:
                if stamp is None:
                    continue
                age = self.ticks_diff(now_ms, stamp)
                if 0 <= age <= MAX_SAMPLE_GAP_MS:
                    recent.append((stamp, old_vote))
            self.temporal = recent
        self.temporal.append((now_ms, vote))
        self.temporal = self.temporal[-TEMPORAL_WINDOW:]
        counts = {}
        for _, value in self.temporal:
            counts[value] = counts.get(value, 0) + 1
        winner = max(counts, key=counts.get)
        if counts[winner] >= TEMPORAL_REQUIRED:
            return ("unknown" if winner == "__unknown__" else "known"), "two_of_three_confirmation"
        return "uncertain", "awaiting_two_of_three_confirmation"

    def clear_temporal(self):
        self.temporal = []

    def note_capture(self, now_ms):
        self.last_frame_ms = now_ms
        self.counters["captures"] += 1

    def note_wake(self, now_ms):
        self.last_wake_ms = now_ms
        self.counters["camera_wakes"] += 1

    def note_result(self, now_ms, status):
        self.last_result_ms = now_ms
        self.timeout_active = False
        if status == "no_face":
            self.counters["no_face_outputs"] += 1
        if status == "uncertain":
            self.counters["uncertain_outputs"] += 1
        if status in ("known", "unknown", "uncertain", "no_face", "multiple_faces", "acquisition_only"):
            self.counters["decisions"] += 1

    def heartbeat(self, now_ms):
        if self.last_result_ms is None:
            if self.last_wake_ms is None:
                return None
            age = self.ticks_diff(now_ms, self.last_wake_ms)
        else:
            age = self.ticks_diff(now_ms, self.last_result_ms)
        if age > MAX_SAMPLE_GAP_MS:
            if not self.timeout_active:
                self.counters["timeouts"] += 1
            self.timeout_active = True
            self.clear_temporal()
            return {"status": "uncertain", "reason": "max_response_interval_timeout", "age_ms": age}
        return None

    def apply_response_deadline(self, result, now_ms):
        """Downgrade a completed result if the loop returned after its deadline.

        This post-call check cannot preempt a blocking camera or inference call;
        board timing must show those calls stay within the declared bound.
        """
        timeout = self.heartbeat(now_ms)
        if timeout is None:
            return result, None
        self.clear_temporal()
        late_status = result.get("status") if isinstance(result, dict) else None
        # Never pass through the late identity, candidate, or similarity fields.
        # A timeout means the application did not meet its declared response bound.
        safe_result = {
            "status": "uncertain",
            "identity_id": None,
            "decision_reason": timeout["reason"],
            "response_timeout_age_ms": timeout["age_ms"],
            "late_result_status_suppressed": late_status,
        }
        return safe_result, timeout


def reason_coded_quality(mean_u8, stdev_u8, min_mean=12, max_mean=243, min_stdev=5):
    if mean_u8 is None or stdev_u8 is None:
        return False, "quality_statistics_missing"
    try:
        mean_u8, stdev_u8 = float(mean_u8), float(stdev_u8)
    except (TypeError, ValueError):
        return False, "quality_statistics_nonfinite"
    if not math.isfinite(mean_u8) or not math.isfinite(stdev_u8):
        return False, "quality_statistics_nonfinite"
    if mean_u8 < min_mean:
        return False, "too_dim"
    if mean_u8 > max_mean:
        return False, "saturated"
    if stdev_u8 < min_stdev:
        return False, "low_contrast_or_blank"
    return True, "quality_gate_pass"
