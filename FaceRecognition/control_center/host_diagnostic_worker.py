"""Persistent, stdin/stdout-only TF2.10 worker for opt-in JPEG diagnostics."""

from __future__ import annotations

import base64
import json
import sys

from .host_diagnostic import analyze_jpeg, load_models


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
        "runtime": models["runtime"],
        "hashes": models["hashes"],
        "runtime_config_sha256": models["runtime_config_sha256"],
        "manifest_runtime_config_sha256": models["manifest_runtime_config_sha256"],
        "integration_manifest_sha256": models["integration_manifest_sha256"],
        "manifest_runtime_config_matches": models["manifest_runtime_config_matches"],
        "warnings": ([] if models["manifest_runtime_config_matches"] else [
            "manifest_runtime_config_sha256_differs_from_the_pinned_current_config"
        ]),
        "thresholds": {
            "detector": models["detector_threshold"],
            "classifier_historical": models["classifier_threshold"],
            "uncertainty_band": models["uncertainty_band"],
        },
        "model_load_ms": models["model_load_ms"],
        "qualification": "EXPERIMENTAL / UNQUALIFIED; no identity is emitted",
    })
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request_must_be_object")
            if request.get("command") == "stop":
                emit({"type": "stopped"})
                return 0
            if request.get("command") != "analyze":
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
                "request_id": str(locals().get("request", {}).get("request_id", "")),
                "ok": False,
                "error": f"{type(error).__name__}:{str(error)[:180]}",
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
