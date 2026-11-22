"""Wrap-aware board-clock alignment from predeclared paired marker IDs."""

import argparse
import json
import math
from pathlib import Path
import statistics


def unwrap_ticks(ticks, modulus):
    if modulus <= 1:
        raise ValueError("counter modulus must exceed one")
    if not ticks:
        return []
    out = [int(ticks[0]) % modulus]
    for raw in ticks[1:]:
        value = int(raw) % modulus
        prev_raw = out[-1] % modulus
        delta = (value - prev_raw + modulus // 2) % modulus - modulus // 2
        if delta < 0:
            # A small backward move is out-of-order; a large wrap is represented
            # as a positive delta by the signed modular difference above.
            raise ValueError("board tick moved backward or events are out of order")
        out.append(out[-1] + delta)
    return out


def fit_marker_alignment(board_ticks, host_times_ns, modulus, nominal_ns_per_tick=1.0):
    if len(board_ticks) != len(host_times_ns) or len(board_ticks) < 3:
        raise ValueError("at least three paired, predeclared marker IDs are required")
    x = [float(v) for v in unwrap_ticks(board_ticks, modulus)]
    y = [float(v) for v in host_times_ns]
    x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
    denom = sum((v - x_mean) ** 2 for v in x)
    if denom == 0:
        raise ValueError("board markers do not span time")
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denom
    intercept = y_mean - slope * x_mean
    residuals = [b - (slope * a + intercept) for a, b in zip(x, y)]
    rms = math.sqrt(sum(v * v for v in residuals) / len(residuals))
    return {
        "marker_count": len(x),
        "board_tick_unwrapped": [int(v) for v in x],
        "host_time_ns": [int(v) for v in y],
        "host_ns_per_board_tick": slope,
        "intercept_host_ns": intercept,
        "nominal_ns_per_board_tick": nominal_ns_per_tick,
        "estimated_drift_ppm_vs_nominal_tick_rate": (slope / nominal_ns_per_tick - 1.0) * 1_000_000.0,
        "residual_rms_ns": rms,
        "residual_max_abs_ns": max(abs(v) for v in residuals),
        "alignment_uncertainty_lower_bound_ns": max(abs(v) for v in residuals),
        "note": "residuals omit meter acquisition/report delay and marker-detection latency",
    }


def detect_power_state_segments(points, threshold_w, min_state_points=2):
    """Detect high/low marker phases at report resolution; no intra-report timing."""
    if not points:
        return []
    ordered = list(points)  # preserve arrival order so out-of-order packets remain visible
    states = []
    for point in ordered:
        t = int(point["host_receive_monotonic_ns"])
        value = float(point["power_mean_w"])
        state = "high" if value >= threshold_w else "low"
        states.append((state, t, value))
    segments = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index][0] != states[start][0]:
            rows = states[start:index]
            segments.append({"state": rows[0][0], "start_monotonic_ns": rows[0][1],
                             "end_monotonic_ns": rows[-1][1], "point_count": len(rows),
                             "mean_power_w": sum(row[2] for row in rows) / len(rows),
                             "accepted": len(rows) >= min_state_points})
            start = index
    return segments


def marker_match(segments, expected_states):
    observed = [segment["state"] for segment in segments if segment["accepted"]]
    return {"expected_states": list(expected_states), "observed_states": observed,
            "match": observed == list(expected_states),
            "report_resolution_only": True,
            "note": "derive the threshold and duration pattern before policy outcomes; exclude marker windows from observation energy"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_markers_json", type=Path,
                        help="JSON object with board_ticks, host_times_ns, counter_modulus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = json.loads(args.paired_markers_json.read_text(encoding="utf-8"))
    result = fit_marker_alignment(data["board_ticks"], data["host_times_ns"], data["counter_modulus"],
                                  data.get("nominal_ns_per_tick", 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "markers": result["marker_count"],
                      "drift_ppm": result["estimated_drift_ppm_vs_nominal_tick_rate"],
                      "residual_rms_ns": result["residual_rms_ns"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
