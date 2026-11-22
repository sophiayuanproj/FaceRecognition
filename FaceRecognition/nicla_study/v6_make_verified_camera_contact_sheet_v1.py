"""Build a private contact sheet of the exact pre-probe Nicla camera frames."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
PRIVATE = STUDY / "private/camera_identity_survey_v2"
RESULT = json.loads((PRIVATE / "VERIFIED_SET_PRIVATE.json").read_text(encoding="utf-8"))
OUT = PRIVATE / "VERIFIED_SET_CONTACT_SHEET.png"

W, H, PAD, HEADER, LABEL = 240, 240, 18, 54, 34
COLS, ROWS = 4, 2
canvas = Image.new("RGB", (PAD + COLS * (W + PAD), PAD + HEADER + ROWS * (H + LABEL + PAD)), "#17191d")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default()
draw.text((PAD, 12), "Nicla camera preview before each volatile board probe", fill="white", font=font)
draw.text((PAD, 30), "2.5 s dwell after render acknowledgement · face labels visually checked · pointer outside face", fill="#b9c2cf", font=font)

for index, row in enumerate(RESULT["rows"]):
    case = row["case"]
    col, grid_row = index % COLS, index // COLS
    x = PAD + col * (W + PAD)
    y = PAD + HEADER + grid_row * (H + LABEL + PAD)
    path = ROOT / row["scene_preview_receipt"].replace(".json", ".jpg")
    image = Image.open(path).convert("RGB")
    image = image.resize((W, H), Image.Resampling.LANCZOS)
    canvas.paste(image, (x, y))
    device = row["device_output"]
    label = f"{case}  ->  {device.get('decision')}"
    draw.text((x, y + H + 5), label, fill="white", font=font)

canvas.save(OUT, optimize=True)
print(json.dumps({"path": str(OUT.relative_to(ROOT)), "size": list(canvas.size), "bytes": OUT.stat().st_size}))
