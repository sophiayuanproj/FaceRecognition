"""Small real Keras-to-TFLite conversion and independent MAC-accounting check."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from nicla_study.macs import compare_mac_accounts, keras_model_macs, layer_macs, tflite_operator_macs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(workspace: Path) -> dict:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(24681357)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    hand = {
        "conv2d": layer_macs("Conv2D", input_shape=(1, 8, 8, 3), output_shape=(1, 6, 6, 4),
                             kernel_size=(3, 3), filters=4),
        "depthwise": layer_macs("DepthwiseConv2D", input_shape=(1, 8, 8, 3), output_shape=(1, 6, 6, 6),
                                kernel_size=(3, 3), depth_multiplier=2),
        "separable": layer_macs("SeparableConv2D", input_shape=(1, 8, 8, 3), output_shape=(1, 6, 6, 4),
                                kernel_size=(3, 3), filters=4, depth_multiplier=2),
        "dense": layer_macs("Dense", input_shape=(1, 8), output_shape=(1, 5), units=5),
    }
    expected = {"conv2d": 3888, "depthwise": 1944, "separable": 2808, "dense": 40}
    if hand != expected:
        raise AssertionError("hand-computable MAC cases differ: %r" % hand)

    inputs = tf.keras.Input(shape=(8, 8, 3), dtype=tf.float32, name="synthetic_input")
    x = tf.keras.layers.SeparableConv2D(4, 3, padding="valid", depth_multiplier=2,
                                        use_bias=False, name="separable")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="mean_pool")(x)
    outputs = tf.keras.layers.Dense(5, use_bias=True, name="classifier")(x)
    model = tf.keras.Model(inputs, outputs, name="mac_accounting_fixture")
    keras_report = keras_model_macs(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converted = converter.convert()
    output_dir = workspace / "work/project-finish-v4/models/host_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mac-accounting-separable-fixture.tflite"
    model_path.write_bytes(converted)
    interpreter = tf.lite.Interpreter(model_content=converted, num_threads=1)
    interpreter.allocate_tensors()
    operator_report = tflite_operator_macs(interpreter)
    comparison = compare_mac_accounts(keras_report, operator_report)
    if comparison["keras_layer_macs"] != 2828:
        raise AssertionError("unexpected Keras model MAC total")
    if not comparison["match"] or not comparison["operator_accounting_complete"]:
        raise AssertionError("Keras and TFLite MAC paths did not agree: %r" % comparison)
    sample = __import__("numpy").arange(8*8*3, dtype="float32").reshape((1,8,8,3)) / 191.0
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], sample.astype(input_detail["dtype"]))
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).reshape(-1).tolist()
    report = {
        "schema": "mac-accounting-conversion-check-1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "bundled"),
        "fixture_purpose": "synthetic operator-accounting software fixture; not a face model or task result",
        "hand_cases_macs": hand,
        "keras_model_macs": keras_report,
        "tflite_operator_macs": operator_report,
        "comparison": comparison,
        "conversion_bytes": len(converted),
        "converted_model_sha256": sha256(model_path),
        "inference_smoke": {"input_shape": list(sample.shape), "output_shape": [len(output)],
                            "output_values": [float(x) for x in output]},
        "counting_convention": "one multiply-accumulate equals one MAC; bias adds, activations, pooling, normalization, indexing, memory movement and interpreter overhead excluded",
        "interpretation": "MAC count is an operation estimate, not measured speed or energy."
    }
    report_path = workspace / "work/project-finish-v4/reports/mac-accounting-conversion.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    workspace = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    print(json.dumps(run(workspace), indent=2))
