"""Fullscreen randomized public-image stimulus player for Nicla camera sessions.

The player never shows labels or filenames. It logs source record IDs and
presentation markers so later capture records can be joined to the local
manifest. A final-role block requires a frozen model/config/calibration receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
from pathlib import Path
import time
import uuid

from .data_pipeline import read_jsonl, safe_path
from .fs import atomic_write_json, sha256_file
from .roles import assert_role_access


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _append_event(stream, event: dict) -> None:
    stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    stream.flush()


def _hide_stimulus_labels(labels) -> None:
    """Hide every active target before starting the blank interval."""
    for label in labels:
        label.place_forget()


def _load_trials(workspace: Path, block: str, seed: int, frozen_receipt: dict | None,
                 condition: str = "mixed", trial_count: int | None = None) -> list[dict]:
    data_root = workspace / "work/project-finish-v4/data"
    rng = random.Random(seed)
    if block == "presentation":
        rows = read_jsonl(data_root / "presentation_manifest.jsonl")
        rows = [row for row in rows if row.get("presentation_role") in
                {"known_presentation", "unknown_presentation"}]
        if not rows:
            raise ValueError("presentation manifest has no eligible rows")
    else:
        rows = read_jsonl(data_root / "primary_split_manifest.jsonl")
        allowed_roles = {"development", "unknown_development"} if block == "development" else {"final", "unknown_final"}
        rows = [row for row in rows if row.get("role") in allowed_roles]
        action = "select" if block == "development" else "evaluate_final"
        assert_role_access(rows, action, frozen_receipt=frozen_receipt)
        if not rows:
            raise ValueError("primary manifest has no eligible rows for " + block)
    if any(row.get("dataset") != "survface_96" for row in rows):
        raise ValueError("presentation player accepts SurvFace rows only")
    for row in rows:
        path = safe_path(row, workspace)
        expected = row.get("source_sha256", row.get("sha256"))
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("stimulus source is missing or its hash changed: " + str(row.get("record_id")))
    role_name = "presentation_role" if block == "presentation" else "role"
    known_roles = {"known_presentation"} if block == "presentation" else {
        "development" if block == "development" else "final"}
    unknown_roles = {"unknown_presentation"} if block == "presentation" else {
        "unknown_development" if block == "development" else "unknown_final"}
    known = [row for row in rows if row.get(role_name) in known_roles]
    unknown = [row for row in rows if row.get(role_name) in unknown_roles]
    rng.shuffle(known)
    rng.shuffle(unknown)
    count = trial_count
    if condition == "known":
        pool = [{"condition": "known", "rows": [row]} for row in known]
    elif condition == "unknown":
        pool = [{"condition": "unknown", "rows": [row]} for row in unknown]
    elif condition == "empty":
        pool = [{"condition": "empty", "rows": []} for _ in range(count or max(len(known), 1))]
    elif condition == "ambiguous":
        pairs = []
        for known_row in known:
            partner = next((row for row in unknown if row.get("identity_id") != known_row.get("identity_id")), None)
            if partner is None:
                partner = next((row for row in known if row.get("identity_id") != known_row.get("identity_id")), None)
            if partner is not None:
                pairs.append({"condition": "ambiguous", "rows": [known_row, partner]})
        pool = pairs
    elif condition == "mixed":
        categories = {
            "known": [{"condition": "known", "rows": [row]} for row in known],
            "unknown": [{"condition": "unknown", "rows": [row]} for row in unknown],
            "empty": [{"condition": "empty", "rows": []} for _ in range(max(len(known), len(unknown), 1))],
            "ambiguous": _load_trials(workspace, block, seed ^ 0x5A5A5A5A,
                                       frozen_receipt, condition="ambiguous", trial_count=None),
        }
        requested = count or min(len(known), len(unknown), len(categories["ambiguous"]), 4) * 4
        if requested < 1:
            raise ValueError("mixed stimulus trial count must be positive")
        category_order = list(categories)
        rng.shuffle(category_order)
        quotas = {name: requested // len(category_order) for name in category_order}
        for name in category_order[:requested % len(category_order)]:
            quotas[name] += 1
        pool = []
        for name in category_order:
            candidates = list(categories[name])
            rng.shuffle(candidates)
            quota = quotas[name]
            if quota > len(candidates):
                raise ValueError("mixed stimulus schedule lacks %s trials for its balanced quota" % name)
            pool.extend(candidates[:quota])
    else:
        raise ValueError("unsupported presentation condition: " + condition)
    if not pool:
        raise ValueError("no source-disjoint rows available for condition " + condition)
    rng.shuffle(pool)
    if count is not None:
        if count < 1:
            raise ValueError("trial_count must be positive")
        if condition == "empty":
            pool = pool[:count]
        elif count <= len(pool):
            pool = pool[:count]
        else:
            raise ValueError("requested trial count exceeds unique eligible trials")
    return pool


def run(args: argparse.Namespace) -> int:
    import tkinter as tk
    from PIL import Image, ImageTk

    workspace = args.workspace.resolve()
    receipt = json.loads(args.freeze_receipt.read_text(encoding="utf-8")) if args.freeze_receipt else None
    if args.block == "final":
        from .freeze_selection import verify_receipt_sources
        verify_receipt_sources(args.freeze_receipt)
    trials = _load_trials(workspace, args.block, args.seed, receipt,
                          condition=args.condition, trial_count=args.trial_count)
    if args.max_trials is not None:
        trials = trials[:args.max_trials]
    if not trials:
        raise ValueError("trial plan is empty")
    run_id = args.run_id or str(uuid.uuid4())
    log_path = args.event_log or (workspace / "work/project-finish-v4/runs/presentation" / (run_id + ".jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    screen_width, screen_height = args.width or 1280, args.height or 720
    rng = random.Random(args.seed ^ 0xA5A5A5A5)
    colors = ["#808080"]
    display_sizes = [64, 80, 96, 120, 144, 168, 192, 224]
    state = {"index": 0, "closed": False, "after_id": None}

    root = tk.Tk()
    root.configure(background=colors[0], cursor="none")
    root.attributes("-fullscreen", True)
    root.geometry("%dx%d+0+0" % (screen_width, screen_height))
    root.bind("<Escape>", lambda _event: close_session("operator_escape"))
    image_label = tk.Label(root, bd=0, highlightthickness=0, background=colors[0])
    second_image_label = tk.Label(root, bd=0, highlightthickness=0, background=colors[0])
    image_label.place_forget()
    second_image_label.place_forget()
    photo_ref = {"images": []}

    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        session = {"schema": "stimulus-event-1", "event": "session_start", "utc": _utc(),
                   "monotonic_ns": time.monotonic_ns(), "run_id": run_id, "block": args.block,
                   "condition": args.condition,
                   "session_id": args.session_id, "lighting_state": args.lighting_state,
                   "motion_state": args.motion_state, "range_cm": args.range_cm,
                   "seed": args.seed, "planned_trials": len(trials), "source_manifest": str(
                       workspace / ("work/project-finish-v4/data/presentation_manifest.jsonl" if args.block == "presentation"
                       else "work/project-finish-v4/data/primary_split_manifest.jsonl"))}
        _append_event(stream, session)

        def close_session(reason: str = "completed") -> None:
            if state["closed"]:
                return
            state["closed"] = True
            if state["after_id"] is not None:
                try:
                    root.after_cancel(state["after_id"])
                except Exception:
                    pass
            _hide_stimulus_labels((image_label, second_image_label))
            root.update_idletasks()
            root.update()
            _append_event(stream, {"schema": "stimulus-event-1", "event": "session_end", "utc": _utc(),
                         "monotonic_ns": time.monotonic_ns(), "run_id": run_id, "reason": reason,
                         "completed_trials": state["index"]})
            root.quit()

        def start_next() -> None:
            if state["closed"]:
                return
            if state["index"] >= len(trials):
                close_session("completed")
                return
            trial_index = state["index"]
            trial = trials[trial_index]
            condition = trial["condition"]
            trial_rows = trial["rows"]
            display_rows = []
            labels = (image_label, second_image_label)
            photo_ref["images"] = []
            for index, row in enumerate(trial_rows):
                source_path = safe_path(row, workspace)
                with Image.open(source_path) as source_image:
                    source_image.load()
                    maximum_width = int(screen_width * (0.18 if len(trial_rows) > 1 else 0.28))
                    target_width = min(rng.choice(display_sizes), maximum_width)
                    target_width = max(48, target_width)
                    target_height = target_width
                    stimulus = source_image.convert("L").resize((target_width, target_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(stimulus)
                if len(trial_rows) > 1:
                    x = int(screen_width * (0.28 if index == 0 else 0.56))
                    x = min(x, screen_width - target_width)
                else:
                    x = rng.randint(0, max(0, screen_width - target_width))
                y = rng.randint(0, max(0, screen_height - target_height))
                labels[index].configure(image=photo)
                labels[index].place(x=x, y=y, width=target_width, height=target_height)
                photo_ref["images"].append(photo)
                display_rows.append({"width_px": target_width, "height_px": target_height, "x_px": x, "y_px": y})
            for index in range(len(trial_rows), len(labels)):
                labels[index].place_forget()
            root.update_idletasks()
            root.update()
            role_name = "presentation_role" if args.block == "presentation" else "role"
            _append_event(stream, {"schema": "stimulus-event-1", "event": "stimulus_start", "utc": _utc(),
                         "monotonic_ns": time.monotonic_ns(), "run_id": run_id,
                         "trial_index": trial_index, "trial_id": "%s-%06d" % (run_id, trial_index),
                         "condition": condition,
                         "record_ids": [row.get("record_id") for row in trial_rows],
                         "identity_ids": [row.get("identity_id") for row in trial_rows],
                         "roles": [row.get(role_name) for row in trial_rows],
                         "ground_truth_status": condition,
                         "display_geometry": display_rows,
                         "seed": args.seed})
            state["after_id"] = root.after(int(args.hold_seconds * 1000), lambda: end_trial(trial_index))

        def end_trial(trial_index: int) -> None:
            if state["closed"]:
                return
            _hide_stimulus_labels((image_label, second_image_label))
            root.update_idletasks()
            root.update()
            trial_rows = trials[trial_index]["rows"]
            _append_event(stream, {"schema": "stimulus-event-1", "event": "stimulus_end", "utc": _utc(),
                         "monotonic_ns": time.monotonic_ns(), "run_id": run_id,
                         "trial_index": trial_index, "trial_id": "%s-%06d" % (run_id, trial_index),
                         "record_ids": [row.get("record_id") for row in trial_rows]})
            state["index"] += 1
            state["after_id"] = root.after(int(args.blank_seconds * 1000), start_next)

        root.after(150, start_next)
        try:
            root.mainloop()
        finally:
            if not state["closed"]:
                close_session("player_closed")
            root.destroy()
    return 0


def main(argv: list[str] | None = None) -> int:
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument("--block", choices=("presentation", "development", "final"), required=True)
    parser.add_argument("--condition", choices=("mixed", "known", "unknown", "empty", "ambiguous"), default="mixed")
    parser.add_argument("--trial-count", type=int, default=None,
                        help="exact number of unique source trials to play; empty-scene trials may repeat")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--lighting-state", required=True)
    parser.add_argument("--motion-state", required=True)
    parser.add_argument("--range-cm", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261401)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--blank-seconds", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--freeze-receipt", type=Path, default=None)
    parser.add_argument("--event-log", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.block == "final" and args.freeze_receipt is None:
        parser.error("--block final requires --freeze-receipt")
    if args.hold_seconds <= 0 or args.blank_seconds < 0 or args.range_cm <= 0 or (args.trial_count is not None and args.trial_count < 1):
        parser.error("durations and range must be positive (blank duration may be zero)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
