"""Turn frames from your own Nicla into a labelled training/test file.

  python label_own_frames.py --pos frames_with_face/ --neg frames_no_face/ --out own_frames.npz

--pos : images of YOUR face (the dashboard's Export session folder works; any png/jpg/pgm).
        Vary distance, angle, lighting. One face per frame. Sorted by filename = capture order.
--neg : frames with no face (walls, desk, ceiling, the room you demo in). Optional but valuable.
Faces are boxed with OpenCV's Haar frontal-face cascade (needs opencv-python-headless, already
in requirements-training.txt). Frames where it finds zero or several faces are dropped and
listed. ALWAYS open the contact sheet it writes and delete any wrongly boxed frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".pgm", ".bmp"}


def load_gray240(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != (240, 240):                     # centre-crop to square, then resize
        s = min(im.size)
        l, t = (im.size[0] - s) // 2, (im.size[1] - s) // 2
        im = im.crop((l, t, l + s, t + s)).resize((240, 240), Image.BILINEAR)
    return np.asarray(im, np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, required=True)
    ap.add_argument("--neg", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-face", type=int, default=36, help="smallest face side, window px")
    ap.add_argument("--cascade", type=Path, help="path to haarcascade_frontalface_default.xml if OpenCV lacks it")
    a = ap.parse_args()
    xml = a.cascade or Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(xml))
    if cascade.empty():
        raise SystemExit(f"could not load Haar cascade {xml}; pass --cascade PATH (opencv 4.x wheels include it)")
    frames, boxes, names, dropped = [], [], [], []
    for path in sorted(p for p in a.pos.iterdir() if p.suffix.lower() in EXT and not p.name.startswith("._")):
        g = load_gray240(path)
        found = cascade.detectMultiScale(cv2.equalizeHist(g), 1.08, 6, minSize=(a.min_face, a.min_face))
        if len(found) != 1:
            dropped.append({"file": path.name, "faces_found": int(len(found))})
            continue
        x, y, w, h = [int(v) for v in found[0]]
        frames.append(g); boxes.append([x, y, x + w, y + h]); names.append(path.name)
    n_pos = len(frames)
    if a.neg:
        for path in sorted(p for p in a.neg.iterdir() if p.suffix.lower() in EXT and not p.name.startswith("._")):
            frames.append(load_gray240(path)); boxes.append([-1, -1, -1, -1]); names.append(path.name)
    if not frames:
        raise SystemExit("no usable frames")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, frames=np.stack(frames), boxes=np.array(boxes, np.float32), names=np.array(names))
    # contact sheet of the positives with their boxes, for a human check
    cell, cols = 120, 8
    rows = max(1, (n_pos + cols - 1) // cols)
    sheet = np.zeros((rows * cell, cols * cell), np.uint8)
    rgb = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for i in range(n_pos):
        t = cv2.cvtColor(cv2.resize(frames[i], (cell, cell)), cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = [int(v * cell / 240) for v in boxes[i]]
        cv2.rectangle(t, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(t, str(i), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 200, 255), 1)
        r, c = divmod(i, cols)
        rgb[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = t
    sheet_path = a.out.with_suffix(".contact_sheet.png")
    cv2.imwrite(str(sheet_path), rgb)
    print(json.dumps({"positives_kept": n_pos, "negatives": len(frames) - n_pos,
                      "dropped": dropped, "contact_sheet": str(sheet_path), "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
