"""Render the R19 qualified Windows study, using the final return ZIP as input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ARCHIVE_SHA256 = "66f9d14dd4b75b87d79d974cd28dc17492c5f3625db6edbc3255911e347e2356"
CAMPAIGN_PATH = "resume/campaign/"
FIXED, GATED = "fixed-small", "scene-gated-small"
POLICIES = (FIXED, GATED)
WORKLOADS = ("capture-only", "detector-only", FIXED, GATED)
WORKLOAD_LABELS = dict(zip(WORKLOADS, ("Capture only", "Detector only", "Fixed inference", "Scene-gated inference")))
EPISODE_LABELS = {
    "far-outdoor": "Far outdoor",
    "hallway-approach-medium": "Medium hallway",
    "hallway-approach-near": "Near hallway",
    "far-wet-lens": "Far wet lens",
}
INK, MUTED, GRID = "#18273b", "#485a70", "#d9e3eb"
TEAL, NAVY = "#007c75", "#283f81"
COLORS = {"capture-only": "#8191a1", "detector-only": "#4c657f", FIXED: NAVY, GATED: TEAL}
OFFSETS = (-0.19, 0.19)
T_95_DF9 = 2.2621571627409915
CLIP_SECONDS = 29

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "axes.unicode_minus": True,
    }
)


def row_table(archive: ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(archive.read(CAMPAIGN_PATH + name).decode("utf-8-sig"))))


def title(fig, heading: str, subtitle: str) -> None:
    fig.text(0.065, 0.948, heading, fontsize=18, fontweight="bold", va="top")
    fig.text(0.065, 0.874, subtitle, fontsize=9.5, color=MUTED, va="top")


def grid(ax, axis: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.75)
    ax.tick_params(axis="both", length=0)


def save(fig, folder: Path, name: str) -> None:
    fig.savefig(folder / f"{name}.pdf", metadata={"Creator": "P04 R19 study", "Title": name})
    fig.savefig(folder / f"{name}.png", dpi=300)
    plt.close(fig)


def median_interval(values: list[float]) -> tuple[float, float, float]:
    """Median and observed minimum and maximum."""
    return statistics.median(values), min(values), max(values)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 10:
        raise ValueError("The plotted confidence intervals require ten qualified starts.")
    mean = statistics.mean(values)
    half_width = T_95_DF9 * statistics.stdev(values) / len(values) ** 0.5
    return mean, mean - half_width, mean + half_width


def matched_delay_seconds(row: dict) -> float | None:
    if int(row["matched_faces"]) == 0:
        return None
    return float(row["first_matched_face_ms"]) / 1000


def jitter(count: int) -> list[float]:
    """Fixed, symmetric offsets keep data placement reproducible."""
    return [0.012 * (rank - (count - 1) / 2) for rank in range(count)]


def plot_paired(folder: Path, paired: list[dict], orders: dict) -> None:
    values = [float(row["delta_w"]) * 1000 for row in paired]
    mean, low, high = mean_ci(values)
    savings = -mean
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.78)
    title(fig, f"Scene gating saved {savings:.2f} mW", "One within-boot comparison per independent power cycle")
    grid(ax, "x")
    for i, (row, shift) in enumerate(zip(paired, values)):
        first = orders[row["planned_block"]].index(FIXED) < orders[row["planned_block"]].index(GATED)
        ax.hlines(i, shift, 0, color=GRID, linewidth=1.0)
        ax.scatter(shift, i, s=46, c=[NAVY if first else TEAL], marker="o" if first else "s", zorder=3)
    summary_row = len(paired) + 0.55
    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
    ax.axhline(len(paired) - 0.1, color=GRID, linewidth=0.8)
    ax.hlines(summary_row, low, high, color=INK, linewidth=2.2)
    ax.scatter(mean, summary_row, marker="D", c=[INK], s=40, zorder=4)
    ax.text(-38, summary_row, f"95% CI {low:.2f} to {high:.2f}", color=MUTED, va="center", fontsize=9)
    ax.set(yticks=[*range(len(paired)), summary_row], yticklabels=[row["planned_block"] for row in paired] + ["Mean"],
           ylim=(summary_row + 0.75, -0.7), xlim=(-56, 2),
           xlabel="Scene-gated minus fixed mean input power (mW)")
    handles = [
        Line2D([], [], color=NAVY, marker="o", linestyle="none", label="Fixed first"),
        Line2D([], [], color=TEAL, marker="s", linestyle="none", label="Scene-gated first"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.13), ncol=2, fontsize=9)
    save(fig, folder, "01_paired_power")


def plot_workload(folder: Path, arms: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.subplots_adjust(left=0.275, right=0.84, bottom=0.17, top=0.76)
    title(fig, "Input power by workload", "Ten qualified starts per workload. Diamonds show the mean and its 95% interval.")
    grid(ax, "x")
    for rank, arm in enumerate(WORKLOADS):
        values = [float(row["mean_power_w"]) * 1000 for row in arms if row["arm"] == arm]
        mean, low, high = mean_ci(values)
        ax.scatter(values, [rank + offset for offset in jitter(len(values))], c=[COLORS[arm]],
                   alpha=0.52, s=25, edgecolors="white", linewidths=0.45, zorder=3)
        ax.hlines(rank, low, high, color=COLORS[arm], linewidth=2.25, zorder=4)
        ax.scatter(mean, rank, marker="D", c=[COLORS[arm]], s=46, edgecolors="white", linewidths=0.6, zorder=5)
        ax.text(1.03, rank, f"{mean:.2f} mW", transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=INK, fontweight="bold", fontsize=9)
    ax.set(yticks=range(len(WORKLOADS)), yticklabels=[WORKLOAD_LABELS[arm] for arm in WORKLOADS],
           ylim=(3.55, -0.55), xlim=(598, 690), xticks=[600, 620, 640, 660, 680],
           xlabel="Time-weighted mean input power per 29-second playback (mW)")
    save(fig, folder, "02_workload_power")


def plot_episodes(folder: Path, episodes: list[dict]) -> None:
    fig, (coverage_ax, delay_ax) = plt.subplots(1, 2, figsize=(9.6, 4.8), width_ratios=(1, 1.28), sharey=True)
    fig.subplots_adjust(left=0.184, right=0.963, bottom=0.14, top=0.73, wspace=0.22)
    title(fig, "Same face episodes, slower response under gating",
          "The same three segments matched in all ten starts. Neither policy matched the far outdoor segment.")
    for ax in (coverage_ax, delay_ax):
        grid(ax, "x")
        ax.set_ylim(3.52, -0.52)
    for rank, episode in enumerate(EPISODE_LABELS):
        for arm, offset in zip(POLICIES, OFFSETS):
            rows = [row for row in episodes if row["arm"] == arm and row["id"] == episode]
            if len(rows) != 10:
                raise ValueError(f"Expected ten qualified observations of {arm} / {episode}")
            delays = [delay for row in rows if (delay := matched_delay_seconds(row)) is not None]
            y = rank + offset
            count = len(delays)
            coverage_ax.barh(y, count, height=0.29, color=COLORS[arm], zorder=3)
            if count:
                coverage_ax.text(count - 0.20, y, f"{count}/10", color="white", va="center", ha="right",
                                 fontsize=8, fontweight="bold", zorder=4)
                median, low, high = median_interval(delays)
                delay_ax.hlines(y, low, high, color=COLORS[arm], linewidth=1.5, alpha=0.65)
                delay_ax.scatter(delays, [y + offset for offset in jitter(count)], c=[COLORS[arm]],
                                 alpha=0.46, s=23, edgecolors="white", linewidths=0.4, zorder=3)
                delay_ax.scatter(median, y, marker="D", c=[COLORS[arm]], s=36, edgecolors="white",
                                 linewidths=0.5, zorder=4)
            else:
                coverage_ax.scatter(0, y, marker="|", s=70, c=[COLORS[arm]], zorder=3)
                coverage_ax.text(0.30, y, "0/10", color=COLORS[arm], va="center", ha="left", fontsize=8)
        if episode == "far-outdoor":
            delay_ax.text(0.075, rank, "No matched episodes", color=MUTED, va="center", fontsize=8.7)
    coverage_ax.set(yticks=range(len(EPISODE_LABELS)), yticklabels=list(EPISODE_LABELS.values()),
                    xlim=(-0.13, 10.1), xticks=[0, 2, 4, 6, 8, 10], xlabel="Face episodes matched (of 10)")
    delay_ax.set(xlim=(0, 2.65), xticks=[0, 0.5, 1, 1.5, 2, 2.5], xlabel="Episode start to first match (s)")
    delay_ax.tick_params(labelleft=False)
    handles = [Patch(facecolor=COLORS[arm], label=WORKLOAD_LABELS[arm]) for arm in POLICIES]
    handles.append(Line2D([], [], color=INK, marker="D", linestyle="none", markersize=5, label="Median"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.061, 0.829), frameon=False,
               ncol=len(handles), fontsize=9)
    save(fig, folder, "03_episode_coverage_latency")


def plot_counts(folder: Path, arms: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.19, right=0.95, bottom=0.17, top=0.69)
    title(fig, "Scene gating ran fewer inference stages", "Counts sum device telemetry across ten playbacks per policy.")
    grid(ax, "x")
    for index, key in enumerate(("detector_calls", "identity_runs")):
        for arm, offset in zip(POLICIES, OFFSETS):
            count = sum(int(row[key]) for row in arms if row["arm"] == arm)
            y = index + offset
            ax.barh(y, count, height=0.29, color=COLORS[arm], zorder=3)
            ax.text(count + 10, y, str(count), color=COLORS[arm], va="center", fontweight="bold", fontsize=9)
    ax.set(yticks=[0, 1], yticklabels=["Detector calls", "Identity runs"],
           ylim=(1.56, -0.56), xlim=(0, 560), xlabel="Device calls across ten playbacks")
    ax.legend(handles=[Patch(facecolor=COLORS[arm], label=WORKLOAD_LABELS[arm]) for arm in POLICIES],
              loc="upper left", bbox_to_anchor=(-0.15, 1.28), frameon=False, ncol=2, fontsize=9)
    save(fig, folder, "04_inference_work")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_zip", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with args.evidence_zip.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("The supplied return ZIP does not match the verified study archive.")
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "data"
    data.mkdir(exist_ok=True)
    with ZipFile(args.evidence_zip) as archive:
        names = ("primary-points.csv", "qualified-arms.csv", "qualified-episodes.csv", "final-block-ledger.csv")
        for name in (*names, "status.json"):
            (data / name).write_bytes(archive.read(CAMPAIGN_PATH + name))
        paired, arms, episodes, ledger = (row_table(archive, name) for name in names)
    included = [row for row in ledger if row["inclusion"] == "included"]
    if len({row["boot_id"] for row in included}) != len(paired) or len(paired) != 10:
        raise ValueError("The included points must map to ten distinct boot IDs.")
    plot_paired(args.out, paired, {row["planned_block"]: row["executed_arms"].split() for row in included})
    plot_workload(args.out, arms)
    plot_episodes(args.out, episodes)
    plot_counts(args.out, arms)
    mean, low, high = mean_ci([float(row["delta_w"]) * 1000 for row in paired])
    power = {
        arm: statistics.mean(float(row["mean_power_w"]) * 1000 for row in arms if row["arm"] == arm)
        for arm in WORKLOADS
    }
    derived = {
        "source_zip_sha256": ARCHIVE_SHA256,
        "paired_mean_gated_minus_fixed_mw": mean,
        "paired_95_ci_mw": [low, high],
        "percent_change_relative_to_fixed_mean": 100 * mean / power[FIXED],
        "workload_mean_mw": power,
        "standardized_length_seconds": CLIP_SECONDS,
        "standardized_gated_minus_fixed_j": mean * CLIP_SECONDS / 1000,
        "standardized_paired_95_ci_j": [item * CLIP_SECONDS / 1000 for item in (low, high)],
        "observed_run_duration_range_s": [
            min(float(row["duration_s"]) for row in arms),
            max(float(row["duration_s"]) for row in arms),
        ],
    }
    (data / "figure_statistics.json").write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered four PNG/PDF figure pairs and extracted plot data to {args.out.resolve()}")


if __name__ == "__main__":
    main()
