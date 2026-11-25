"""Render the P04 host-only recognition and compression comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

from make_figures import COLORS, GRID, INK, MUTED, NAVY, TEAL, grid, jitter, plt, save, title


D_SOURCE = "handoffs/D/CANDIDATE_PAIRWISE_FINAL_ANALYSIS.json"
D_SHA256 = "9abf1419cab74f92c1a2d44d76008ccc11e9169eb014898c7076c53e60d5e60b"
E_SOURCE = "reports/E/E_ROOT_METRICS.csv"
E_SHA256 = "2da285be5cc33ba3b1535bc5bc041e847906fb49588cdbe90f0d800a495c4e11"
ROUTES = {
    "compact_int8": ("Compact gallery · INT8", TEAL),
    "large_int8": ("Large classifier · INT8", "#54748f"),
    "large_float": ("Large classifier · float", NAVY),
}
METHODS = {
    "large_dense_equal_finetune": "Dense equal fine-tune",
    "large_qat": "Quantization-aware training",
    "large_magnitude_prune_50": "Magnitude pruning · 50%",
    "large_magnitude_prune_75": "Magnitude pruning · 75%",
    "large_structured_prune_2of4": "Structured pruning · 2:4",
    "large_physical_channel_keep_50": "Physical channel surgery",
}


def verified_source(path: Path, expected_sha256: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Source digest mismatch: {path}")
    return content


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recognition_rows(source: dict) -> list[dict]:
    contrasts = source["paired_comparisons"]
    rows = []
    for float_pair, int_pair in zip(
        contrasts["large_float_vs_compact_int8"], contrasts["large_int8_vs_compact_int8"], strict=True
    ):
        shared = (
            "root_seed", "known_rows", "unknown_rows", "small_correct_known_acceptance", "unknown_fpir_small"
        )
        if any(float_pair[key] != int_pair[key] for key in shared):
            raise ValueError("The two references do not share the same compact result.")
        for route, comparison, known_key, fpir_key in (
            ("compact_int8", float_pair, "small_correct_known_acceptance", "unknown_fpir_small"),
            ("large_int8", int_pair, "large_correct_known_acceptance", "unknown_fpir_large"),
            ("large_float", float_pair, "large_correct_known_acceptance", "unknown_fpir_large"),
        ):
            rows.append(
                {
                    "route": route,
                    "seed": comparison["root_seed"],
                    "known_crops": comparison["known_rows"],
                    "unknown_crops": comparison["unknown_rows"],
                    "correct_known_acceptance": comparison[known_key],
                    "unknown_fpir": comparison[fpir_key],
                }
            )
    if len({row["seed"] for row in rows}) != 10:
        raise ValueError("The recognition chart requires ten matched roots.")
    return rows


def plot_recognition(folder: Path, rows: list[dict]) -> None:
    fig, (known_ax, unknown_ax) = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    fig.subplots_adjust(left=0.225, right=0.96, bottom=0.16, top=0.69, wspace=0.25)
    title(fig, "Host crop recognition stayed weak",
          "Matched training roots evaluated on the same 100 known and 300 unknown SurvFace crops.")
    for ax, metric, limit, ticks, heading in (
        (known_ax, "correct_known_acceptance", 32, [0, 5, 10, 15, 20, 25, 30], "Correct known acceptance"),
        (unknown_ax, "unknown_fpir", 11, [0, 2, 4, 6, 8, 10], "Unknown false positive identification"),
    ):
        grid(ax, "x")
        ax.set_title(heading, loc="left", pad=9, fontsize=10, fontweight="bold")
        ax.set(xlabel="Percent of crops (%)", xlim=(-0.3, limit), xticks=ticks, ylim=(2.57, -0.57))
        for rank, (route, (_, color)) in enumerate(ROUTES.items()):
            values = [float(row[metric]) * 100 for row in rows if row["route"] == route]
            ax.hlines(rank, min(values), max(values), linewidth=1.2, color=color, alpha=0.65)
            ax.scatter(values, [rank + v for v in jitter(len(values))], c=[color], alpha=0.53, s=24,
                       edgecolors="white", linewidths=0.4, zorder=3)
            mean = statistics.mean(values)
            ax.scatter(mean, rank, marker="D", c=[color], s=48, edgecolors="white", linewidths=0.55, zorder=4)
            ax.text(mean, rank - 0.16, f"{mean:.1f}%", color=color, va="bottom", ha="center", fontsize=8.7,
                    fontweight="bold")
    known_ax.set(yticks=range(len(ROUTES)), yticklabels=[label for label, _ in ROUTES.values()])
    unknown_ax.tick_params(labelleft=False)
    save(fig, folder, "05_host_recognition")


def compression_rows(source: bytes) -> list[dict]:
    rows = []
    fields = ("seed", "bytes", "macs", "activation_peak_bytes_estimate", "known_correct_acceptance", "unknown_fpir",
              "training_updates")
    for row in csv.DictReader(io.StringIO(source.decode("utf-8-sig"))):
        if row["family"] == "optimization" and row["arm_id"] in METHODS:
            rows.append({"method": row["arm_id"], **{key: row[key] for key in fields}})
    for method in METHODS:
        members = [row for row in rows if row["method"] == method]
        if len({row["seed"] for row in members}) != 10 or any(row["training_updates"] != "75" for row in members):
            raise ValueError(f"Incomplete or unmatched optimization budget for {method}")
    return rows


def plot_compression(folder: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    fig.subplots_adjust(left=0.35, right=0.94, bottom=0.18, top=0.77)
    title(fig, "Channel surgery reduced the INT8 model file",
          "MobileNetV1 α=0.25 with matched 75-update fine-tuning budgets across ten roots per arm.")
    grid(ax, "x")
    sizes = {}
    for rank, (method, label) in enumerate(METHODS.items()):
        values = {int(row["bytes"]) for row in rows if row["method"] == method}
        if len(values) != 1:
            raise ValueError(f"Root variation in file size for {label} would require an interval.")
        size = values.pop()
        sizes[method] = size
        color = TEAL if method == "large_physical_channel_keep_50" else COLORS["detector-only"]
        ax.barh(rank, size / 1000, color=color, height=0.53, zorder=3)
        ax.text(size / 1000 + 7, rank, f"{size:,} B", color=color, va="center", fontweight="bold", fontsize=9)
    baseline, surgery = sizes["large_dense_equal_finetune"], sizes["large_physical_channel_keep_50"]
    reduction = (baseline - surgery) / baseline
    ax.text(11, len(METHODS) - 1, f"{reduction:.1%} smaller", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set(yticks=range(len(METHODS)), yticklabels=list(METHODS.values()), ylim=(5.54, -0.54),
           xlim=(0, 402), xticks=[0, 100, 200, 300, 400], xlabel="Full-INT8 TFLite file size (kB, 1 kB = 1,000 B)")
    save(fig, folder, "06_compression_size")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path, help="Root of the verified Jobs D/E artifact tree")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    known = recognition_rows(json.loads(verified_source(args.campaign_root / D_SOURCE, D_SHA256)))
    optimized = compression_rows(verified_source(args.campaign_root / E_SOURCE, E_SHA256))
    write_csv(data / "host-recognition-roots.csv", known)
    write_csv(data / "compression-size-roots.csv", optimized)
    plot_recognition(args.out, known)
    plot_compression(args.out, optimized)
    print(f"Rendered host recognition and compression figures to {args.out.resolve()}")


if __name__ == "__main__":
    main()
