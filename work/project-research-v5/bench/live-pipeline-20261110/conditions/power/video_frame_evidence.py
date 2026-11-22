"""Link optical workload observations to preview packets and save bounded diagnostics."""

from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import urllib.parse
import urllib.request


PLAYBACK_EVENTS = ("video_started", "video_tick", "video_ended")
EPISODE_MARGIN_MS = 350
SAMPLES_PER_KIND = 2
ATTEMPTS_PER_KIND = 6
MAX_JPEG_BYTES = 1_000_000


def media_knots(render):
    knots = []
    for row in render:
        if row["event"] in PLAYBACK_EVENTS:
            host_ns = row["host_receive_monotonic_ns"]
            media_ms = 1000 * row["media_time_s"]
            if not knots or (host_ns > knots[-1][0] and media_ms > knots[-1][1]):
                knots.append((host_ns, media_ms))
    return knots


def media_at(knots, host_ns):
    if type(host_ns) is not int or len(knots) < 2:
        return None
    at = bisect_right(knots, (host_ns, float("inf"))) - 1
    if not 0 <= at < len(knots) - 1:
        return None
    a, ma = knots[at]
    b, mb = knots[at + 1]
    return ma + (mb - ma) * (host_ns - a) / (b - a)


def episode_at(video, media_ms):
    if media_ms is None:
        return None
    return next((episode for episode in video["episodes"]
                 if episode["start_ms"] + EPISODE_MARGIN_MS <= media_ms
                 < episode["end_ms_exclusive"] - EPISODE_MARGIN_MS), None)


def observation_capture_ns(row):
    receive = row.get("host_received_monotonic_ns")
    total = (row.get("timings_ms") or {}).get("total")
    if type(receive) is int and isinstance(total, (int, float)):
        return receive - int(total * 1_000_000)
    return None


def project_box(matrix, box):
    x, y, width, height = box
    points = []
    for a, b in ((x, y), (x + width, y), (x + width, y + height), (x, y + height)):
        denom = matrix[2][0] * a + matrix[2][1] * b + matrix[2][2]
        points.append(tuple((row[0] * a + row[1] * b + row[2]) / denom for row in matrix[:2]))
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def intersection(left, right):
    x, y, width, height = left
    a, b, w, h = right
    return max(0, min(x + width, a + w) - max(x, a)) * max(0, min(y + height, b + h) - max(y, b))


def projected_match(observation, projection, episode, video_fps, media_ms):
    box = episode.get("face_box_xywh_video_px")
    track = episode.get("face_box_track")
    if track:
        at = min(len(track) - 1, max(0, int(media_ms * video_fps / 1000) - episode["frame_start_inclusive"]))
        box = track[at]["face_box_xywh_video_px"]
    if not box:
        return False, 0.0
    truth = project_box(projection, box)
    truth_area = truth[2] * truth[3]
    regions = (observation.get("detection") or {}).get("regions_xywh") or []
    matched = False
    if len(regions) == 1:
        rect = regions[0]["rect_px"] if isinstance(regions[0], dict) else regions[0]
        overlap = intersection(truth, rect)
        matched = overlap / (truth_area + rect[2] * rect[3] - overlap) >= .30
    crop = (observation.get("crop") or {}).get("rect_px")
    coverage = intersection(truth, crop) / truth_area if crop else 0.0
    return matched, coverage


def link_frames(events, render, boot_id):
    """Count each distinct boot/frame once and require packet/observation SHA equality."""
    knots = media_knots(render)
    by_frame = defaultdict(dict)
    encode_errors = []
    seen_events = set()
    for row in events:
        seq = row.get("event_seq")
        if seq is not None:
            if seq in seen_events:
                continue
            seen_events.add(seq)
        if row.get("boot_id") != boot_id:
            continue
        frame_id = row.get("frame_id")
        if row.get("event") == "telemetry_encode_error":
            encode_errors.append(row)
        if type(frame_id) is not int:
            continue
        if row.get("type") == "observation_complete":
            by_frame[frame_id].setdefault("observations", []).append(row)
        elif row.get("type") == "frame_payload" and row.get("kind") == "frame_jpeg":
            by_frame[frame_id].setdefault("previews", []).append(row)
        elif row.get("event") == "capture_start":
            by_frame[frame_id]["capture_start_ns"] = row.get("host_received_monotonic_ns")

    frames = {}
    for frame_id, entry in by_frame.items():
        previews = entry.get("previews", [])
        observations = entry.get("observations", [])
        if not previews and not observations:
            continue
        preview = previews[0] if previews else {}
        obs = observations[0] if observations else {}
        descriptor = obs.get("frame") or {}
        digest = preview.get("sha256")
        linked = (len(previews) == len(observations) == 1 and isinstance(digest, str) and len(digest) == 64
                  and descriptor.get("sha256") == digest and descriptor.get("frame_id") == frame_id
                  and descriptor.get("boot_id") == boot_id)
        stamp = observation_capture_ns(obs)
        if stamp is None:
            stamp = entry.get("capture_start_ns")
        if stamp is None:
            stamp = (preview or obs).get("host_received_monotonic_ns")
        frames[frame_id] = dict(preview=preview, observation=obs, linked=linked, preview_count=len(previews),
                                observation_count=len(observations), media_ms=media_at(knots, stamp))
    errors = []
    for row in encode_errors:
        media = frames.get(row.get("frame_id"), {}).get("media_ms")
        if media is None:
            media = media_at(knots, row.get("host_received_monotonic_ns"))
        errors.append(dict(event_seq=row.get("event_seq"), frame_id=row.get("frame_id"),
                           failed_event=(row.get("record") or {}).get("failed_event"), media_ms=media))
    return frames, errors


def coverage_counts(frames, errors, video):
    def counts(frame_rows, error_rows):
        rows = frame_rows.values()
        return dict(
            preview_jpegs=sum(bool(row["preview"]) for row in rows),
            observation_complete=sum(bool(row["observation"]) for row in rows),
            matched_observations=sum(row["linked"] for row in rows),
            preview_without_linked_observation=sum(bool(row["preview"]) and not row["linked"] for row in rows),
            observation_without_linked_preview=sum(bool(row["observation"]) and not row["linked"] for row in rows),
            identity_errors=sum((row["observation"].get("identity") or {}).get("status") == "error" for row in rows),
            mismatched_or_duplicate_frame_ids=[frame_id for frame_id, row in frame_rows.items()
                if row["preview"] and row["observation"] and not row["linked"]],
            telemetry_encode_errors=len(error_rows),
            observation_encode_errors=sum(row["failed_event"] == "observation_complete" for row in error_rows),
        )

    def select(include):
        return counts(
            {frame_id: row for frame_id, row in frames.items() if include(row["media_ms"])},
            [row for row in errors if include(row["media_ms"])],
        )

    def condition(media_ms, face):
        episode = episode_at(video, media_ms)
        return episode is not None and episode["expected_face"] is face

    per_episode = {}
    for episode in video["episodes"]:
        left = episode["start_ms"] + EPISODE_MARGIN_MS
        right = episode["end_ms_exclusive"] - EPISODE_MARGIN_MS
        counts_for_episode = select(lambda media: media is not None and left <= media < right)
        stamps = [left, *sorted(row["media_ms"] for row in frames.values() if row["linked"]
                               and row["media_ms"] is not None and left <= row["media_ms"] < right), right]
        counts_for_episode["max_linked_observation_gap_ms"] = round(max(
            next_ms - prior_ms for prior_ms, next_ms in zip(stamps, stamps[1:])
        ), 1)
        per_episode[episode["id"]] = counts_for_episode

    return {
        "whole_playback": select(lambda media: media is not None),
        "face_present_episode_interiors": select(lambda media: condition(media, True)),
        "quiet_episode_interiors": select(lambda media: condition(media, False)),
        "episodes": per_episode,
        "telemetry_encode_error_events": errors,
    }


def fetch_verified_jpeg(device_url, boot_id, frame_id, sha256):
    query = urllib.parse.urlencode(dict(boot_id=boot_id, frame_id=frame_id))
    endpoint = device_url.rstrip("/") + "/api/v2/device/frame?" + query
    with urllib.request.urlopen(endpoint, timeout=2) as response:
        data = response.read(MAX_JPEG_BYTES + 1)
        if (response.headers.get_content_type() != "image/jpeg"
                or response.headers.get("X-Device-Boot-ID") != boot_id
                or response.headers.get("X-Device-Frame-ID") != str(frame_id)
                or response.headers.get("X-Frame-SHA256") != sha256):
            raise ValueError("JPEG response provenance differs from the packet and observation")
    if (len(data) > MAX_JPEG_BYTES or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9")
            or hashlib.sha256(data).hexdigest() != sha256):
        raise ValueError("JPEG bytes or SHA-256 differ from the linked observation")
    return data


class PrivateSamples:
    def __init__(self, folder, device_url, boot, start_ack, video, projection):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=False)
        self.device_url = device_url
        self.boot = boot
        self.start_ack = start_ack
        self.video = video
        self.projection = projection
        self.receipts = {"face_positive": [], "quiet_negative": []}

    def collect(self, frames):
        for frame_id, row in frames.items():
            if not row["linked"]:
                continue
            episode = episode_at(self.video, row["media_ms"])
            if episode is None:
                continue
            obs = row["observation"]
            detect = obs.get("detection") or {}
            identity = obs.get("identity") or {}
            crop = obs.get("crop") or {}
            decision = obs.get("decision") or {}
            applied = obs.get("applied_state") or {}
            requested_id = (obs.get("requested_state") or {}).get("command_id") or obs.get("command_id")
            if (applied.get("running") is not True or applied.get("policy_id") != self.start_ack.get("policy_id")
                    or applied.get("measurement_stage_id") != self.start_ack.get("measurement_stage_id")
                    or (requested_id and requested_id != self.start_ack.get("command_id"))):
                continue
            if episode["expected_face"]:
                if (self.projection is None or detect.get("status") != "face" or identity.get("status") != "ran"
                        or crop.get("status") != "valid" or crop.get("frame_id") != frame_id):
                    continue
                matched, crop_coverage = projected_match(obs, self.projection, episode, self.video["fps"],
                                                         row["media_ms"])
                if not matched or crop_coverage < .80:
                    continue
                kind = "face_positive"
            else:
                quiet = detect.get("status") == "no_face" or decision.get("reason") == "SCENE_TRIGGER_SKIPPED_QUIET_CHECK"
                if not quiet or identity.get("status") != "skipped":
                    continue
                kind, crop_coverage = "quiet_negative", None
            receipts = self.receipts[kind]
            if (len(receipts) >= ATTEMPTS_PER_KIND or any(item["frame_id"] == frame_id for item in receipts)
                    or sum(item["status"] == "verified" for item in receipts) >= SAMPLES_PER_KIND):
                continue
            digest = row["preview"]["sha256"]
            receipt = dict(kind=kind, frame_id=frame_id, boot_id=self.boot, episode=episode["id"],
                           estimated_media_ms=round(row["media_ms"], 1), jpeg_sha256=digest,
                           start_command_id=self.start_ack.get("command_id"),
                           observation_command_id=requested_id,
                           firmware_sha256=obs.get("firmware_sha256"),
                           config_sha256=obs.get("model_config_sha256"),
                           packet_event_seq=row["preview"].get("event_seq"),
                           observation_event_seq=obs.get("event_seq"),
                           detection=detect, crop=crop, identity=identity,
                           identity_input=obs.get("identity_input"), decision=decision,
                           projected_face_crop_coverage=crop_coverage)
            try:
                jpeg = fetch_verified_jpeg(self.device_url, self.boot, frame_id, digest)
                file = f"{kind}-{frame_id}.jpg"
                (self.folder / file).write_bytes(jpeg)
                receipt.update(status="verified", file=file, bytes=len(jpeg))
            except (OSError, ValueError) as error:
                receipt.update(status="failed", reason=f"{type(error).__name__}: {error}")
            receipts.append(receipt)

    def write_receipt(self):
        receipt = {"schema": "nicla-private-linked-jpeg-samples-v1", "boot_id": self.boot,
                   "video_sha256": self.video["sha256"], "start_ack": self.start_ack,
                   "samples": [item for rows in self.receipts.values() for item in rows]}
        (self.folder / "LINKED_FRAMES.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        return {kind: sum(row["status"] == "verified" for row in rows) for kind, rows in self.receipts.items()}
