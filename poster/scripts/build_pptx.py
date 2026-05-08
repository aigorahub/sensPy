"""Assemble the sensPy Sensometrics 2026 poster PowerPoint."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
POSTER = ROOT / "poster"
CHARTS = POSTER / "charts"
CHART_DATA = POSTER / "chart_data"
ASSETS = POSTER / "assets"
OUT = POSTER / "print_artifacts"
OUT.mkdir(parents=True, exist_ok=True)

PPTX = OUT / "senspy-sensometrics-2026-poster.pptx"

SLIDE_W = 27.83
SLIDE_H = 39.37
M = 0.85
HEADER_H = 4.45
FOOTER_H = 1.45
GAP = 0.55
COL_W = (SLIDE_W - 2 * M - GAP) / 2
LEFT_X = M
RIGHT_X = M + COL_W + GAP

DISPLAY = "Georgia"
BODY = "Arial"
MONO = "Courier New"

CANVAS = "f4f0e6"
PANEL = "fbfaf6"
INK = "17291f"
CORAL = "c7563f"
GREEN = "4c8a61"
SAGE = "9aa79b"
MIST = "e8e3d8"
DARK = "10231a"
WHITE = "ffffff"


def rgb(hex_value: str) -> RGBColor:
    h = hex_value.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def add_rect(slide, x, y, w, h, fill, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    rect = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    rect.fill.solid()
    rect.fill.fore_color.rgb = rgb(fill)
    if line:
        rect.line.color.rgb = rgb(line)
        rect.line.width = Pt(1)
    else:
        rect.line.fill.background()
    return rect


def add_text(
    slide,
    text: str,
    x,
    y,
    w,
    h,
    *,
    size=18,
    font=BODY,
    color=INK,
    bold=False,
    italic=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.05,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.italic = italic
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(4)
    return box


def add_heading(slide, title: str, x, y, w, *, size=38):
    add_rect(slide, x, y, w, 0.07, CORAL)
    add_text(
        slide,
        title,
        x,
        y + 0.16,
        w,
        0.76,
        size=size,
        font=DISPLAY,
        color=DARK,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    return y + 1.08


def add_image_fit(slide, path: Path, x, y, w, h):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    pic_h = pic.height / 914400.0
    pic_w = pic.width / 914400.0
    if pic_h > h:
        scale = h / pic_h
        pic.width = int(pic.width * scale)
        pic.height = int(pic.height * scale)
        pic_w = pic.width / 914400.0
        pic_h = pic.height / 914400.0
    pic.left = Inches(x + (w - pic_w) / 2)
    pic.top = Inches(y + (h - pic_h) / 2)
    return pic


def add_metric(slide, x, y, w, h, value, label, sub=None, detail=None, *, color=CORAL):
    add_rect(slide, x, y, w, h, MIST, line="#d2c9bb", radius=True)
    add_text(
        slide,
        value,
        x + 0.08,
        y + 0.12,
        w - 0.16,
        0.72,
        size=50,
        font=BODY,
        color=color,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    add_text(
        slide,
        label,
        x + 0.12,
        y + 0.92,
        w - 0.24,
        0.62,
        size=21,
        font=BODY,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    if sub:
        add_text(
            slide,
            sub,
            x + 0.22,
            y + 1.48,
            w - 0.44,
            0.42,
            size=18,
            font=BODY,
            color="#4d5a52",
            align=PP_ALIGN.CENTER,
            margin=0,
        )
    if detail:
        add_text(
            slide,
            detail,
            x + 0.24,
            y + 1.9,
            w - 0.48,
            max(h - 1.96, 0.35),
            size=15.5,
            font=BODY,
            color="#4d5a52",
            align=PP_ALIGN.CENTER,
            margin=0,
        )


def add_compact_metric(slide, x, y, w, h, value, label, sub=None, *, color=CORAL):
    add_rect(slide, x, y, w, h, MIST, line="#d2c9bb", radius=True)
    add_text(
        slide,
        value,
        x + 0.06,
        y + 0.08,
        w - 0.12,
        0.52,
        size=30,
        font=BODY,
        color=color,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    add_text(
        slide,
        label,
        x + 0.1,
        y + 0.73,
        w - 0.2,
        0.4,
        size=14.2,
        font=BODY,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    if sub:
        add_text(
            slide,
            sub,
            x + 0.12,
            y + 1.17,
            w - 0.24,
            h - 1.17,
            size=12.5,
            font=BODY,
            color="#4d5a52",
            align=PP_ALIGN.CENTER,
            margin=0,
        )


def add_callout(slide, x, y, w, h, title, body, detail=None):
    add_rect(slide, x, y, w, h, MIST, line="#d2c9bb", radius=True)
    add_text(slide, title, x + 0.26, y + 0.24, w - 0.52, 0.5, size=24, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, body, x + 0.38, y + 0.88, w - 0.76, 0.72, size=21.5, color=INK, align=PP_ALIGN.CENTER)
    if detail:
        add_text(slide, detail, x + 0.42, y + 1.68, w - 0.84, h - 1.82, size=18, color="#4d5a52", align=PP_ALIGN.CENTER)


def add_takeaway(slide, x, y, w, h, title, body, detail=None, *, color=CORAL):
    add_rect(slide, x, y, w, h, MIST, line="#d2c9bb", radius=True)
    add_text(slide, title, x + 0.28, y + 0.28, w - 0.56, 0.58, size=25, color=color, bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_text(slide, body, x + 0.42, y + 1.04, w - 0.84, 0.95, size=22, color=INK, align=PP_ALIGN.CENTER, margin=0)
    if detail:
        add_text(slide, detail, x + 0.52, y + 2.12, w - 1.04, h - 2.34, size=18, color="#4d5a52", align=PP_ALIGN.CENTER, margin=0)


def build_header(slide):
    add_rect(slide, 0, 0, SLIDE_W, HEADER_H, MIST)
    add_rect(slide, 0, HEADER_H - 0.07, SLIDE_W, 0.07, CORAL)
    add_text(
        slide,
        "sensPy",
        M,
        0.35,
        SLIDE_W - 2 * M,
        1.6,
        size=94,
        font=DISPLAY,
        color=DARK,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    add_text(
        slide,
        "Bringing the gold standard of sensR to the Python sensory ecosystem",
        M,
        2.05,
        SLIDE_W - 2 * M,
        0.82,
        size=37,
        font=BODY,
        color=CORAL,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    add_text(
        slide,
        "John M. Ennis and Bartosz Smulski | Aigora | Sensometrics 2026",
        M,
        3.25,
        SLIDE_W - 2 * M,
        0.5,
        size=25,
        font=BODY,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    add_text(slide, "Python-native Thurstonian sensory discrimination", 19.8, 0.58, 6.4, 0.55, size=18, color="#526057", align=PP_ALIGN.RIGHT)


def build_footer(slide, summary):
    y = SLIDE_H - FOOTER_H
    add_rect(slide, 0, y, SLIDE_W, FOOTER_H, DARK)
    add_text(slide, "sensPy", M, y + 0.38, 2.0, 0.5, size=24, font=DISPLAY, color=WHITE, align=PP_ALIGN.LEFT)
    add_text(
        slide,
        "github.com/aigorahub/sensPy | package v" + summary["version"],
        8.1,
        y + 0.42,
        11.5,
        0.5,
        size=20,
        font=BODY,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    qr = ASSETS / "qr-senspy-github.png"
    if qr.exists():
        slide.shapes.add_picture(str(qr), Inches(SLIDE_W - M - 1.05), Inches(y + 0.2), width=Inches(1.05))


def build_problem(slide):
    y = add_heading(slide, "Why this matters", LEFT_X, 4.65, COL_W)
    add_text(
        slide,
        "sensPy brings sensR-style sensory discrimination into Python without leaving SciPy, dataclasses, or Plotly.",
        LEFT_X + 0.15,
        y + 0.05,
        COL_W - 0.3,
        1.55,
        size=31,
        bold=True,
        color=INK,
        align=PP_ALIGN.CENTER,
    )
    add_callout(
        slide,
        LEFT_X + 0.2,
        y + 2.05,
        (COL_W - 0.6) / 2,
        2.35,
        "Gold standard",
        "Preserve the sensR numerical contract.",
        "Golden fixtures keep the Python port tied to the R reference.",
    )
    add_callout(
        slide,
        LEFT_X + 0.4 + (COL_W - 0.6) / 2,
        y + 2.05,
        (COL_W - 0.6) / 2,
        2.35,
        "Python-native",
        "Typed objects, plots, and planning tools.",
        "Designed for notebooks, pipelines, and interactive review.",
    )
    add_text(
        slide,
        "The result: validated migration for sensory teams already working in Python.",
        LEFT_X + 0.35,
        y + 4.82,
        COL_W - 0.7,
        0.8,
        size=24,
        color=CORAL,
        bold=True,
        align=PP_ALIGN.CENTER,
    )


def build_metrics(slide, summary):
    y = add_heading(slide, "Evidence at a glance", RIGHT_X, 4.65, COL_W)
    card_w = (COL_W - 0.45) / 2
    xs = [RIGHT_X + 0.05, RIGHT_X + 0.4 + card_w]
    add_metric(slide, xs[0], y + 0.05, card_w, 2.35, "13", "protocol variants", "8 single + 5 double", "forced-choice coverage")
    add_metric(slide, xs[1], y + 0.05, card_w, 2.35, "740+", "automated tests", f"{summary['collected_pytest_items']} collected cases", "parity + boundary cases", color=GREEN)
    add_metric(slide, xs[0], y + 2.78, card_w, 2.35, summary["sensr_version"], "sensR fixture", "CRAN reference", "golden parity data")
    add_metric(slide, xs[1], y + 2.78, card_w, 2.35, "Plotly", "interactive figures", "psychometric + ROC", "audit-ready outputs", color=GREEN)
    add_text(
        slide,
        "Numerical parity target: 3-6 decimal places.",
        RIGHT_X + 0.3,
        y + 5.45,
        COL_W - 0.6,
        0.65,
        size=24,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
    )


def build_charts(slide):
    y = add_heading(slide, "Protocol coverage", LEFT_X, 12.05, COL_W)
    add_image_fit(slide, CHARTS / "protocol_coverage.png", LEFT_X + 0.1, y + 0.05, COL_W - 0.2, 5.95)
    add_text(slide, "Common d-prime scale, protocol-specific guessing.", LEFT_X + 0.25, y + 6.15, COL_W - 0.5, 0.65, size=22, color="#4d5a52", align=PP_ALIGN.CENTER)

    y2 = add_heading(slide, "Psychometric functions", RIGHT_X, 12.05, COL_W)
    add_image_fit(slide, CHARTS / "psychometric_curves.png", RIGHT_X + 0.0, y2 + 0.05, COL_W, 5.95)
    add_text(slide, "One API maps Pc, Pd, and d-prime.", RIGHT_X + 0.25, y2 + 6.15, COL_W - 0.5, 0.65, size=22, color="#4d5a52", align=PP_ALIGN.CENTER)


def build_api_and_models(slide):
    y = add_heading(slide, "Validation surface", LEFT_X, 21.05, COL_W)
    add_image_fit(slide, CHARTS / "test_inventory.png", LEFT_X + 0.0, y + 0.05, COL_W, 5.9)
    add_text(slide, "Golden sensR fixtures run beside unit, model, power, and plotting tests.", LEFT_X + 0.25, y + 6.12, COL_W - 0.5, 0.65, size=22, color="#4d5a52", align=PP_ALIGN.CENTER)

    y2 = add_heading(slide, "Advanced models", RIGHT_X, 21.05, COL_W)
    w = (COL_W - 0.65) / 3
    add_metric(slide, RIGHT_X + 0.05, y2 + 0.15, w, 2.6, "BB", "Beta-Binomial", "overdispersed panel data", "panel variability", color=GREEN)
    add_metric(slide, RIGHT_X + 0.325 + w, y2 + 0.15, w, 2.6, "SD", "Same-Different", "Thurstonian criterion model", "criterion effects")
    add_metric(slide, RIGHT_X + 0.6 + 2 * w, y2 + 0.15, w, 2.6, "ROC", "SDT analysis", "AUC + rating data", "decision curves", color=GREEN)
    add_rect(slide, RIGHT_X + 0.05, y2 + 3.18, COL_W - 0.1, 2.62, MIST, line="#d2c9bb", radius=True)
    add_text(
        slide,
        "Validation is built into the architecture.",
        RIGHT_X + 0.35,
        y2 + 3.48,
        COL_W - 0.7,
        0.58,
        size=25,
        color=CORAL,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "sensR fixtures -> SciPy kernels -> dataclass results -> Plotly figures",
        RIGHT_X + 0.4,
        y2 + 4.35,
        COL_W - 0.8,
        0.95,
        size=24,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "Also included: 2-AC, DOD, A-not-A, d-prime tests, posthoc, simulation, power, and sample-size planning.",
        RIGHT_X + 0.25,
        y2 + 6.12,
        COL_W - 0.5,
        0.75,
        size=22,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
    )


def build_conclusion(slide):
    y = 30.15
    add_heading(slide, "Conclusion", M, y, SLIDE_W - 2 * M)
    add_text(
        slide,
        "sensPy brings the sensR contract into Python: validated sensory discrimination, typed results, and interactive figures in one workflow.",
        M + 1.1,
        y + 0.86,
        SLIDE_W - 2 * M - 2.2,
        1.5,
        size=31,
        color=INK,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "What changes for sensometricians",
        M,
        y + 2.55,
        SLIDE_W - 2 * M,
        0.55,
        size=30,
        font=DISPLAY,
        color=DARK,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    take_w = (SLIDE_W - 2 * M - 0.8) / 3
    add_takeaway(
        slide,
        M,
        y + 3.25,
        take_w,
        3.25,
        "Validated migration",
        "Keep sensR-backed decisions in Python notebooks and pipelines.",
        "Same statistical contract; modern workflow surface.",
        color=GREEN,
    )
    add_takeaway(
        slide,
        M + take_w + 0.4,
        y + 3.25,
        take_w,
        3.25,
        "One reporting scale",
        "Pc, Pd, d-prime, intervals, and power share one API.",
        "Protocol differences stay explicit and comparable.",
    )
    add_takeaway(
        slide,
        M + 2 * (take_w + 0.4),
        y + 3.25,
        take_w,
        3.25,
        "Modern outputs",
        "Typed results and Plotly figures are easier to audit and reuse.",
        "Results travel as objects, not loose console output.",
        color=GREEN,
    )
    add_text(
        slide,
        "References: sensR package, CRAN v1.5-3, doi:10.32614/CRAN.package.sensR; Brockhoff & Christensen (2010), doi:10.1016/j.foodqual.2009.04.003; Macmillan & Creelman; SciPy; Plotly.",
        M + 1.6,
        y + 6.72,
        SLIDE_W - 2 * M - 3.2,
        0.82,
        size=16,
        color="#4d5a52",
        align=PP_ALIGN.CENTER,
    )


def main() -> None:
    summary = json.loads((CHART_DATA / "summary.json").read_text(encoding="utf-8"))
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, CANVAS)
    build_header(slide)
    build_problem(slide)
    build_metrics(slide, summary)
    build_charts(slide)
    build_api_and_models(slide)
    build_conclusion(slide)
    build_footer(slide, summary)

    prs.save(PPTX)
    print(f"[pptx] wrote {PPTX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
