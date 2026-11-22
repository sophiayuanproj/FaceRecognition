"""Build source-only embeddings from the resource-preflighted 48px encoder.

Uses the MCU-matched TF2.10 BUILTIN_REF interpreter.  The source box-to-48
resize is a host approximation; it is not the unverified Nicla RGB48 crop.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    cv2.setNumThreads(2)
    root = STUDY / "runs/operator_fused_encoder48"
    model_path = root / "int8.tflite"
    model_hash = sha256(model_path)
    preflight_path = STUDY / "reports/M5/operator_fused_encoder48_PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("model_sha256") != model_hash or preflight.get("successful_camera_invocations", 0) < 10:
        raise ValueError("48px source model lacks current ten-block camera resource preflight")

    index_path = STUDY / "runs/host_reference/nonfinal_index.json"
    index = json.loads(index_path.read_text())
    rows = index["rows"]
    role_path = STUDY / "data/manifests/lfw_roles.jsonl"
    if any("future_final" in row["role"] for row in rows):
        raise ValueError("future-final role entered source feature run")
    with np.load(STUDY / "runs/host_reference/nonfinal_features.npz") as detections:
        detected = np.asarray(detections["detected"], dtype=bool)
    if len(rows) != len(detected):
        raise ValueError("row index and detector cache differ")

    inputs_path = STUDY / "runs/face_encoder/inputs_nonfinal.npz"
    with np.load(inputs_path) as data:
        box = np.asarray(data["box"])
    if box.shape != (len(rows), 112, 112, 3):
        raise ValueError("unexpected frozen source box input shape")

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if inp["shape"].tolist() != [1, 48, 48, 3] or out["shape"].tolist() != [1, 128]:
        raise ValueError("unexpected48px encoder tensor contract")
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise ValueError("expected fully integer 48px encoder")

    features = np.zeros((len(rows), 128), dtype=np.float32)
    for i in np.flatnonzero(detected):
        pixels = cv2.resize(box[i], (48, 48), interpolation=cv2.INTER_LINEAR)
        values = (pixels.astype(np.float32) - 127.5) / 128.0
        scale, zero = inp["quantization"]
        quantized = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0]
        scale, zero = out["quantization"]
        vector = (raw.astype(np.float32) - zero) * scale
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("invalid48px feature at row %d" % i)
        features[i] = vector / norm
        if (i + 1) % 256 == 0:
            print(json.dumps({"rows_processed": int(i + 1)}), flush=True)

    output_path = root / "int8_box_rgb_source.npy"
    np.save(output_path, features)
    metadata = {
        "schema": "v5-face-features48-source-1",
        "scope": "source-only host features from fixed 48px int8 encoder",
        "model_path": "runs/operator_fused_encoder48/int8.tflite",
        "model_sha256": model_hash,
        "preflight_sha256": sha256(preflight_path),
        "host_backend": "TF2.10 BUILTIN_REF",
        "input_shape": inp["shape"].tolist(),
        "input_dtype": "int8",
        "input_quantization": [float(inp["quantization"][0]), int(inp["quantization"][1])],
        "output_shape": out["shape"].tolist(),
        "output_dtype": "int8",
        "output_quantization": [float(out["quantization"][0]), int(out["quantization"][1])],
        "transform": "RGB box crop cached at112x112, OpenCV INTER_LINEAR resize to48x48, (pixel-127.5)/128, np.rint affine quantization, output dequantization and L2 normalization",
        "transform_limit": "source preprocessing only; the native RGB48 camera crop/resize contract remains unverified",
        "rows": len(rows),
        "detected_rows": int(detected.sum()),
        "features_sha256": sha256(output_path),
        "source_input_sha256": sha256(inputs_path),
        "source_index_sha256": sha256(index_path),
        "role_manifest_sha256": sha256(role_path),
        "future_final_used": False,
        "board_qualified": False,
    }
    dump(root / "int8_box_rgb_source.json", metadata)
    print(json.dumps(metadata, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
