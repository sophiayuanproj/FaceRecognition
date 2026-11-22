"""Persistent stdin/stdout worker for the optional host RGB reference."""

from __future__ import annotations

import base64
import json
import sys

from .host_rgb_reference import analyze_jpeg, load_models


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        models = load_models()
    except Exception as error:
        emit({"type": "fatal", "error": f"{type(error).__name__}:{str(error)[:180]}"})
        return 2
    emit({
        "type": "ready",
        "runtime": "OpenCV host YuNet + SFace ONNX",
        "opencv_version": models["opencv_version"],
        "numpy_version": models["numpy_version"],
        "hashes": models["hashes"],
        "threshold_known_max_cosine": models["threshold_known_max_cosine"],
        "threshold_known_other_margin": models["threshold_known_other_margin"],
        "unknown_calibration_usable_people": models["unknown_calibration_people"],
        "gallery_source_groups": models["gallery_source_groups"],
        "load_ms": models["load_ms"],
        "qualification": "HOST REFERENCE ONLY; NOT ON NICLA; not recognition-qualified",
    })
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request_must_be_object")
            command = request.get("command")
            if command == "stop":
                emit({"type": "stopped"})
                return 0
            if command != "analyze":
                raise ValueError("unsupported_worker_command")
            request_id = str(request.get("request_id", ""))
            encoded = request.get("jpeg_base64")
            if not isinstance(encoded, str):
                raise ValueError("jpeg_base64_required")
            jpeg = base64.b64decode(encoded, validate=True)
            result = analyze_jpeg(jpeg, models)
            emit({"type": "result", "request_id": request_id, "ok": True, "result": result})
        except Exception as error:
            emit({
                "type": "result",
                "request_id": str(request.get("request_id", "")),
                "ok": False,
                "error": f"{type(error).__name__}:{str(error)[:180]}",
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
