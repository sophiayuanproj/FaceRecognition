"""Model MAC accounting with separate Keras-layer and TFLite-operator paths.

Counting convention: one multiply-accumulate is one MAC. Bias addition,
activations, normalization, pooling/reductions, comparisons, indexing, memory
movement, and interpreter overhead are excluded. This is an operation count,
not measured latency or energy.
"""
from __future__ import annotations

import math
from typing import Any

COUNTED_TFLITE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}
EXCLUDED_TFLITE_OPS = {
    "ADD", "AVERAGE_POOL_2D", "CAST", "CONCATENATION", "DEQUANTIZE",
    "EXPAND_DIMS", "GATHER", "LOGISTIC", "MAX_POOL_2D", "MEAN", "MUL",
    "PAD", "PADV2", "QUANTIZE", "RELU", "RELU6", "RESHAPE", "SLICE",
    "SOFTMAX", "SQUEEZE", "STRIDED_SLICE", "SUB", "SUM", "TANH",
    "TRANSPOSE",
}


def _dims(shape: Any, label: str) -> list[int]:
    if hasattr(shape, "as_list"):
        shape = shape.as_list()
    if shape is None:
        raise ValueError(label + " shape is unavailable")
    dims = [1 if index == 0 and value is None else (int(value) if value is not None else -1)
            for index, value in enumerate(shape)]
    if any(x <= 0 for x in dims):
        raise ValueError(label + " shape must be fully specified and positive")
    return dims


def _pair(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    items = tuple(int(x) for x in value)
    if len(items) != 2 or any(x <= 0 for x in items):
        raise ValueError(label + " must contain two positive dimensions")
    return items


def layer_macs(kind: str, *, input_shape: Any, output_shape: Any, kernel_size: Any = None,
               filters: int | None = None, groups: int = 1, depth_multiplier: int = 1,
               units: int | None = None) -> int:
    """Hand-checkable formula for one Keras Conv2D, Depthwise, Separable, or Dense."""
    inp = _dims(input_shape, "input")
    out = _dims(output_shape, "output")
    if inp[0] != out[0]:
        raise ValueError("input and output batch dimensions must match")
    if kind == "Conv2D":
        if len(inp) != 4 or len(out) != 4 or not filters or groups < 1 or inp[-1] % groups:
            raise ValueError("Conv2D requires NHWC rank-4 shapes and valid filters/groups")
        kh, kw = _pair(kernel_size, "kernel_size")
        return out[0] * out[1] * out[2] * int(filters) * kh * kw * (inp[-1] // groups)
    if kind == "DepthwiseConv2D":
        if len(inp) != 4 or len(out) != 4 or depth_multiplier < 1:
            raise ValueError("DepthwiseConv2D requires NHWC rank-4 shapes")
        kh, kw = _pair(kernel_size, "kernel_size")
        if out[-1] != inp[-1] * depth_multiplier:
            raise ValueError("depthwise output channels do not match input channels × multiplier")
        return out[0] * out[1] * out[2] * out[-1] * kh * kw
    if kind == "SeparableConv2D":
        if len(inp) != 4 or len(out) != 4 or not filters or depth_multiplier < 1:
            raise ValueError("SeparableConv2D requires NHWC rank-4 shapes and valid filters")
        kh, kw = _pair(kernel_size, "kernel_size")
        spatial = out[0] * out[1] * out[2]
        depthwise = spatial * inp[-1] * depth_multiplier * kh * kw
        pointwise = spatial * inp[-1] * depth_multiplier * int(filters)
        if out[-1] != int(filters):
            raise ValueError("separable output channels do not match filters")
        return depthwise + pointwise
    if kind == "Dense":
        if len(inp) < 2 or len(out) < 2 or not units:
            raise ValueError("Dense requires batched input/output shapes and units")
        if out[-1] != int(units):
            raise ValueError("Dense output size does not match units")
        return math.prod(inp[1:-1]) * inp[-1] * int(units)
    raise ValueError("unsupported layer kind: " + kind)


def keras_model_macs(model: Any) -> dict:
    """Count Conv2D, DepthwiseConv2D, SeparableConv2D and Dense layers."""
    rows = []
    total = 0
    excluded = []
    for layer in model.layers:
        kind = type(layer).__name__
        if kind not in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D", "Dense"}:
            excluded.append({"name": layer.name, "type": kind, "reason": "outside multiply-accumulate convention"})
            continue
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        if isinstance(input_shape, list) or isinstance(output_shape, list):
            raise ValueError("multi-input/output learned layer requires explicit per-branch accounting")
        kwargs = {"input_shape": input_shape, "output_shape": output_shape}
        if kind in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"}:
            kwargs["kernel_size"] = layer.kernel_size
        if kind == "Conv2D":
            kwargs["filters"] = int(layer.filters)
            kwargs["groups"] = int(getattr(layer, "groups", 1))
        elif kind == "DepthwiseConv2D":
            kwargs["depth_multiplier"] = int(layer.depth_multiplier)
        elif kind == "SeparableConv2D":
            kwargs["filters"] = int(layer.filters)
            kwargs["depth_multiplier"] = int(layer.depth_multiplier)
        elif kind == "Dense":
            kwargs["units"] = int(layer.units)
        count = layer_macs(kind, **kwargs)
        total += count
        rows.append({"name": layer.name, "type": kind, "macs": count})
    return {"method": "keras_layer_formula_v2", "macs_per_inference": total, "counted_layers": rows,
            "excluded_layers": excluded, "complete": True}


def tflite_operator_macs(interpreter: Any) -> dict:
    """Independently count TFLite Conv/Depthwise/FC operators by tensor shapes."""
    if not hasattr(interpreter, "_get_ops_details"):
        raise TypeError("interpreter must expose TensorFlow Lite _get_ops_details()")
    tensors = {int(row["index"]): row for row in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()
    rows = []
    total = 0
    unknown = []
    excluded = []
    for index, op in enumerate(ops):
        name = str(op.get("op_name", "UNKNOWN"))
        input_ids = [int(i) for i in op.get("inputs", []) if int(i) >= 0]
        output_ids = [int(i) for i in op.get("outputs", []) if int(i) >= 0]
        if name not in COUNTED_TFLITE_OPS:
            target = excluded if name in EXCLUDED_TFLITE_OPS else unknown
            target.append({"operator_index": index, "op_name": name})
            continue
        if not input_ids or not output_ids:
            raise ValueError("counted TFLite operator lacks input or output tensors: " + name)
        out_shape = _dims(tensors[output_ids[0]]["shape"], name + " output")
        filter_shape = _dims(tensors[input_ids[1]]["shape"], name + " filter")
        if name == "CONV_2D":
            if len(out_shape) != 4 or len(filter_shape) != 4:
                raise ValueError("CONV_2D tensor ranks must be four")
            count = math.prod(out_shape) * math.prod(filter_shape[1:])
        elif name == "DEPTHWISE_CONV_2D":
            if len(out_shape) != 4 or len(filter_shape) != 4:
                raise ValueError("DEPTHWISE_CONV_2D tensor ranks must be four")
            count = math.prod(out_shape) * filter_shape[1] * filter_shape[2]
        else:
            if len(filter_shape) < 2:
                raise ValueError("FULLY_CONNECTED filter rank must be at least two")
            count = math.prod(out_shape) * filter_shape[-1]
        total += count
        rows.append({"operator_index": index, "op_name": name, "macs": count,
                     "output_shape": out_shape, "filter_shape": filter_shape})
    return {"method": "tflite_operator_tensor_shape_v1", "macs_per_inference": total,
            "counted_operators": rows, "excluded_operators": excluded,
            "unclassified_operators": unknown, "complete": not unknown}


def compare_mac_accounts(layer_report: dict, operator_report: dict) -> dict:
    left = int(layer_report["macs_per_inference"])
    right = int(operator_report["macs_per_inference"])
    return {"keras_layer_macs": left, "tflite_operator_macs": right,
            "difference": left - right, "match": left == right,
            "operator_accounting_complete": bool(operator_report.get("complete"))}
