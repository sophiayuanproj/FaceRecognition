#!/usr/bin/env python3
"""Render a one-page Nicla Vision research handoff PDF from campaign metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)


NAVY = colors.HexColor("#10243A")
TEAL = colors.HexColor("#087C78")
INK = colors.HexColor("#1D3449")
MUTED = colors.HexColor("#557083")
PALE = colors.HexColor("#F1F5F7")
PALE_BLUE = colors.HexColor("#E8F1F5")
PALE_TEAL = colors.HexColor("#E7F4F1")
PALE_AMBER = colors.HexColor("#FFF4DE")
GRID = colors.HexColor("#CBD8E0")
WHITE = colors.white


def fmt_stats(summary, key, digits=3):
    stats = summary[key]
    if not stats["n"]:
        return "n/a"
    return ("{mean:.{digits}f} +/- {sd:.{digits}f}".format(
        mean=stats["mean"], sd=stats["sample_sd"], digits=digits,
    ))


def fmt_range(summary, key, digits=3):
    stats = summary[key]
    if not stats["n"]:
        return "n/a"
    return ("{lo:.{digits}f}-{hi:.{digits}f}".format(
        lo=stats["min"], hi=stats["max"], digits=digits,
    ))


def resolve_group(groups, campaign_fragment, variant, split_seed):
    matches = [value for key, value in groups.items()
               if key == variant + "/split-" + str(split_seed)]
    if len(matches) != 1:
        raise ValueError("Expected one group for %s %s split %s; found %d" % (
            campaign_fragment, variant, split_seed, len(matches),
        ))
    return matches[0]


def make_styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=7.6, leading=9.2, textColor=INK,
                               spaceAfter=0, alignment=TA_LEFT),
        "body_small": ParagraphStyle("BodySmall", parent=base["BodyText"], fontName="Helvetica",
                                      fontSize=6.8, leading=8.2, textColor=INK,
                                      spaceAfter=0, alignment=TA_LEFT),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica",
                                 fontSize=6.2, leading=7.5, textColor=MUTED,
                                 spaceAfter=0, alignment=TA_LEFT),
        "section": ParagraphStyle("Section", parent=base["Heading3"], fontName="Helvetica-Bold",
                                   fontSize=8.5, leading=10, textColor=TEAL,
                                   spaceBefore=2, spaceAfter=3),
        "table_head": ParagraphStyle("TableHead", parent=base["BodyText"], fontName="Helvetica-Bold",
                                      fontSize=6.3, leading=7.3, textColor=WHITE,
                                      alignment=TA_CENTER),
        "table_cell": ParagraphStyle("TableCell", parent=base["BodyText"], fontName="Helvetica",
                                      fontSize=6.5, leading=7.6, textColor=INK,
                                      alignment=TA_CENTER),
        "table_label": ParagraphStyle("TableLabel", parent=base["BodyText"], fontName="Helvetica-Bold",
                                       fontSize=6.5, leading=7.6, textColor=INK,
                                       alignment=TA_LEFT),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica",
                                   fontSize=8.0, leading=9.8, textColor=INK,
                                   spaceAfter=0),
    }


def header(canvas, doc):
    canvas.saveState()
    width, height = landscape(letter)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 56, width, 56, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(34, height - 27, "NICLA VISION  |  FACE VERIFICATION HANDOFF")
    canvas.setFillColor(colors.HexColor("#D5E3ED"))
    canvas.setFont("Helvetica", 8.2)
    canvas.drawString(35, height - 43, "Multi-seed training, embedded execution, and validation evidence  |  22 Oct 2026")
    canvas.setFillColor(TEAL)
    canvas.roundRect(width - 181, height - 43, 146, 23, 5, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 8.2)
    canvas.drawCentredString(width - 108, height - 34.5, "PARTIALLY FUNCTIONAL")
    canvas.restoreState()


def build_pdf(summary_path: Path, pilot_path: Path, output: Path):
    summary = json.loads(summary_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    groups = summary["groups"]
    cells = [
        ("Base - split A", "combined", "base", 20261021),
        ("Tiny - split A", "combined", "tiny", 20261021),
        ("Base - split B", "combined", "base", 20261022),
        ("Tiny - split B", "combined", "tiny", 20261022),
    ]
    cell_stats = []
    for label, campaign, variant, split_seed in cells:
        cell_stats.append((label, resolve_group(groups, campaign, variant, split_seed)))

    styles = make_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=landscape(letter),
                          leftMargin=32, rightMargin=32, topMargin=66, bottomMargin=28,
                          title="Nicla Vision Face Verification Handoff",
                          author="OpenAI Codex",
                          subject="Multi-seed SurvFace embedding research and Nicla validation")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                  id="main")
    doc.addPageTemplates([PageTemplate(id="handoff", frames=[frame], onPage=header)])

    story = []
    key_info = Table([[Paragraph(
        "<b>45 standardized runs:</b> two architectures across two identity-disjoint splits, "
        "with ten matched seeds per cell plus five additional base split-B seeds; a separate "
        "5-seed pilot was re-exported under the corrected protocol. The original seed-20261021 "
        "base/tiny baseline is reported separately. "
        "The board remains on the original base INT8 model; no new candidate was flashed.",
        styles["callout"],
    )]], colWidths=[doc.width])
    key_info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_AMBER),
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#EBCB82")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [key_info, Spacer(1, 5), Paragraph("VALIDATION-ONLY RESEARCH", styles["section"])]

    table_data = [[Paragraph(x, styles["table_head"]) for x in (
        "Training cell", "Seeds", "INT8 AUC<br/>mean +/- SD", "AUC range",
        "INT8 EER<br/>mean +/- SD", "TAR / FAR at val 1%<br/>mean", "INT8 bytes<br/>mean", "INT8 minus float<br/>delta AUC",
    )]]
    for label, stats in cell_stats:
        table_data.append([
            Paragraph(label, styles["table_label"]),
            Paragraph(str(stats["seed_count"]), styles["table_cell"]),
            Paragraph(fmt_stats(stats, "int8_auc"), styles["table_cell"]),
            Paragraph(fmt_range(stats, "int8_auc"), styles["table_cell"]),
            Paragraph(fmt_stats(stats, "int8_eer"), styles["table_cell"]),
            Paragraph("%.3f / %.4f" % (
                stats["int8_tar_at_validation_1pct_far_threshold"]["mean"],
                stats["int8_far_at_validation_threshold"]["mean"],
            ), styles["table_cell"]),
            Paragraph("%s" % round(stats["int8_model_bytes"]["mean"]), styles["table_cell"]),
            Paragraph(fmt_stats(stats, "int8_to_float_auc_delta", digits=4), styles["table_cell"]),
        ])
    widths = [154, 35, 89, 90, 84, 111, 88, 77]
    table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]))
    story += [table, Spacer(1, 4)]

    paired = summary.get("paired_int8_auc_differences", {})
    diff_text = []
    for key in ("tiny_minus_base_split_20261021", "tiny_minus_base_split_20261022",
                "base_split_20261022_minus_20261021", "tiny_split_20261022_minus_20261021"):
        value = paired.get(key)
        if value:
            name = {
                "tiny_minus_base_split_20261021": "tiny minus base, split A",
                "tiny_minus_base_split_20261022": "tiny minus base, split B",
                "base_split_20261022_minus_20261021": "base split B minus A",
                "tiny_split_20261022_minus_20261021": "tiny split B minus A",
            }[key]
            diff_text.append("<b>%s:</b> %+.3f +/- %.3f (n=%d)" % (
                name, value["mean"], value["sample_sd"], value["n"],
            ))
    story.append(Paragraph(
        "<b>Matched AUC differences:</b> " + (" | ".join(diff_text) if diff_text else "pending") +
        ". Metrics are selected on the same validation identities used to score each model; "
        "they are research comparisons, not independent test estimates.", styles["small"],
    ))
    story.append(Spacer(1, 5))

    pilot_groups = pilot.get("groups", {})
    pilot_values = next(iter(pilot_groups.values()), {})
    pilot_auc = pilot_values.get("int8_auc", {})
    pilot_note = ""
    if pilot_auc.get("n"):
        pilot_note = " Earlier 5-seed pilot AUC was %.3f +/- %.3f (separate thread settings)." % (
            pilot_auc["mean"], pilot_auc["sample_sd"],
        )

    left = [
        Paragraph("ACTUAL NICLA RESULT", styles["section"]),
        Paragraph(
            "Headless 20-frame synthetic-target run: FOMO detected <b>18/20</b>; 13 crops reached "
            "the INT8 embedder, 5 edge-clipped crops were marked uncertain, 2 frames had no face, "
            "and there were 0 pipeline errors. All 13 returned <b>no_enrollment</b>.", styles["body"],
        ),
        Spacer(1, 3),
        Paragraph(
            "Deployed base INT8: <b>26,872 B</b>; embedding arena <b>51,040 B</b>; "
            "host/board embedding cosine <b>0.999382</b>. Live timings: capture 34-82 ms, "
            "FOMO 65.5-66.3 ms, embedding 12.75-12.76 ms; minimum Python heap 342,896 B.", styles["body_small"],
        ),
        Spacer(1, 3),
        Paragraph("No model candidate from the expanded campaign was deployed or camera-tested." + pilot_note,
                   styles["small"]),
    ]
    right = [
        Paragraph("LIMITS AND NEXT STEP", styles["section"]),
        Paragraph(
            "The 2,120-pair / 1,060-identity scores are validation-selected; pair and identity-cluster "
            "bootstrap intervals are descriptive. Corrected INT8 calibration used the exact duplicate-filtered, "
            "per-identity-capped training records; manifests retain record and image hashes. The older "
            "seed-20261021 export used validation identities and uncapped records in its calibration pool; its final-pair "
            "result remains historical and was not reopened. No consented wearer template exists.", styles["body_small"],
        ),
        Spacer(1, 3),
        Paragraph(
            "Neutral-scene FOMO misses and edge clipping remain. The FNIRSI meter reported 0 V, so "
            "power is unmeasured. Dataset provenance and reuse rights remain unresolved.", styles["body_small"],
        ),
        Spacer(1, 3),
        Paragraph(
            "<b>Next:</b> collect consented enrollment and an independent camera-matched holdout; "
            "validate known/unknown decisions before any personal-recognition claim.", styles["body_small"],
        ),
    ]
    cards = Table([[left, right]], colWidths=[doc.width * 0.51, doc.width * 0.49])
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), PALE_BLUE),
        ("BACKGROUND", (1, 0), (1, 0), PALE_TEAL),
        ("BOX", (0, 0), (-1, -1), 0.45, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [Spacer(1, 5), cards, Spacer(1, 5)]
    story.append(Paragraph(
        "Arduino Nicla Vision | OpenMV 5.0.1 | MicroPython 1.28.0 | stock 4 MiB ROMFS retained | "
        "camera viewer remained closed | no test frames or enrollment templates saved. "
        "Logged run durations include image loading and overnight idle/sleep gaps, not active CPU time. "
        "Evidence and reproduction details are in the accompanying research readout and project report.",
        styles["small"],
    ))

    doc.build(story)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.summary_json, args.pilot_json, args.output)


if __name__ == "__main__":
    main()
