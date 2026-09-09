"""Render the 5 requested comparison tables as a standalone PDF.

Pulls from an existing offline_file session's metrics.csv (dataset_plain, real
measurements, no synthetic data) - the same source as build_sample_tables.py, just
rendered as a PDF instead of printed to the terminal.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_sample_tables_pdf.py [session_dir] [output.pdf]
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

HEADER_BG = colors.HexColor("#eceff4")
ACCENT = colors.HexColor("#1f77b4")
STRIPE = colors.HexColor("#f7f8fa")

# Category, STOI, PESQ, SNR Before, SNR After, SNR Impr. Sized so "Avg SNR Before (dB)"
# and its siblings have room to sit on one or two wrapped lines without touching the
# next column.
CATEGORY_WIDTHS = [0.14, 0.14, 0.14, 0.20, 0.19, 0.19]


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=17, leading=21, spaceAfter=4),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=colors.HexColor("#444444"), spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=13, leading=16, spaceBefore=16,
            spaceAfter=6, textColor=ACCENT,
        ),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9, leading=12.5, spaceAfter=4),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=7.6, leading=10,
            textColor=colors.HexColor("#555555"), spaceAfter=8,
        ),
    }


def data_table(rows: list[list[str]], col_widths: list[float], font_size: float = 8.5) -> Table:
    total = 17.0 * cm
    widths = [total * w for w in col_widths]

    # Header cells are wrapped Paragraphs, not raw strings: a raw string is drawn at its
    # natural width and overlaps the next column whenever it does not fit, which is what
    # was happening with headers like "Avg SNR Before (dB)". A Paragraph wraps to the
    # column width instead.
    header_style = ParagraphStyle(
        "table_header", fontName="Helvetica-Bold", fontSize=font_size - 0.5,
        leading=font_size + 1.5, alignment=1,  # centre
    )
    body: list[list] = [
        [Paragraph(str(c), header_style) for c in rows[0]]
    ]
    for row in rows[1:]:
        body.append(list(row))

    table = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("LEADING", (0, 1), (-1, -1), font_size + 3),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
    ]
    table.setStyle(TableStyle(style))
    return table


# ------------------------------------------------------------------- data


def load_rows(session: Path, method: str) -> list[dict]:
    want_tap = "input" if method == "unprocessed" else "output"
    with open(session / "metrics.csv", newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["method"] == method and r["tap"] == want_tap]


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def mean(rows: list[dict], key: str) -> float:
    vals = [v for v in (f(r, key) for r in rows) if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def snr_before_after(rows: list[dict]) -> tuple[float, float, float]:
    before, after = [], []
    for r in rows:
        b = f(r, "measured_input_snr_db")
        a = f(r, "residual_noise_snr_db")
        if a != a:
            a = f(r, "output_snr_db")
        if b == b and a == a:
            before.append(b)
            after.append(a)
    if not before:
        return float("nan"), float("nan"), float("nan")
    b, a = float(np.mean(before)), float(np.mean(after))
    return b, a, a - b


def fmt(v: float, spec: str = ".2f") -> str:
    return "n/a" if v != v else f"{v:{spec}}"


def comparison_rows(session: Path, methods: dict[str, str]) -> list[list[str]]:
    rows = [["Pipeline Stage / Configuration", "SNR (dB)", "SI-SDR (dB)", "STOI", "PESQ"]]
    for label, method in methods.items():
        r = load_rows(session, method)
        if not r:
            rows.append([label, "n/a", "n/a", "n/a", "n/a"])
            continue
        _, out_snr, _ = snr_before_after(r)
        rows.append([label, fmt(out_snr), fmt(mean(r, "si_sdr")), fmt(mean(r, "stoi"), ".3f"),
                     fmt(mean(r, "pesq"))])
    return rows


def category_rows(session: Path, method: str) -> list[list[str]]:
    with open(session / "metrics.csv", newline="", encoding="utf-8") as fh:
        all_rows = [r for r in csv.DictReader(fh) if r["method"] == method and r["tap"] == "output"]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_cat[r["category"]].append(r)
    out = [["Category", "Avg STOI", "Avg PESQ", "Avg SNR Before (dB)", "Avg SNR After (dB)",
            "Avg SNR Impr. (dB)"]]
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        before, after, impr = snr_before_after(rows)
        out.append([
            cat.capitalize(), fmt(mean(rows, "stoi"), ".4f"), fmt(mean(rows, "pesq"), ".4f"),
            fmt(before), fmt(after), fmt(impr),
        ])
    return out


# ------------------------------------------------------------------- build


def build(session: Path, out_path: Path) -> Path:
    st = styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="ANC for defence comms - comparison tables", author="anc_defence",
    )
    story: list = []

    story.append(Paragraph("Single-microphone ANC — comparison tables", st["title"]))
    story.append(Paragraph(
        f"Source: dataset_plain corpus, category-balanced 216-example subset "
        f"(6 categories &times; 6 input SNRs &times; 6 examples/cell, -10 to +15 dB). "
        f"Session: {session.name}. Generated {time.strftime('%Y-%m-%d %H:%M')}.",
        st["subtitle"],
    ))

    story.append(Paragraph("Table 1 - Comparison across configurations", st["h1"]))
    story.append(data_table(
        comparison_rows(session, {
            "1. Raw Noisy Audio (Baseline)": "unprocessed",
            "2. NLMS Filter Only (Adaptive Stage)": "nlms_only",
            "3. AI Model Only (Without NLMS)": "dfn_only",
            "4. Hybrid - DFN then NLMS": "dfn_then_nlms",
            "4b. Hybrid - NLMS then DFN": "nlms_then_dfn",
            "5. Delivered pipeline (AI + normalise)": "dfn_then_normalise",
        }),
        col_widths=[0.36, 0.16, 0.18, 0.14, 0.16],
    ))
    story.append(Paragraph(
        "SNR is the classical output SNR (surviving speech power over residual noise power). "
        "SI-SDR additionally charges for speech distortion, not just residual noise, which is why "
        "the two differ. Hybrid rows need a second microphone with an ideal noise-only reference "
        "and are an upper bound, not achievable with one microphone.",
        st["small"],
    ))

    sections = [
        ("Table 2 - Baseline DeepFilterNet3 Only", "dfn_only"),
        ("Table 3 - Baseline NLMS Only", "nlms_only"),
        ("Table 4a - Hybrid: DeepFilterNet then NLMS", "dfn_then_nlms"),
        ("Table 4b - Hybrid: NLMS then DeepFilterNet", "nlms_then_dfn"),
        ("Table 5 - Entire Delivered Pipeline (DeepFilterNet + Volume Normalisation)",
         "dfn_then_normalise"),
    ]
    for title, method in sections:
        story.append(Paragraph(title, st["h1"]))
        story.append(data_table(category_rows(session, method), col_widths=CATEGORY_WIDTHS,
                                 font_size=8.0))
        story.append(Spacer(1, 2))

    story.append(Paragraph(
        "Note: STOI and PESQ are identical between Table 2 (DeepFilterNet only) and Table 5 (the "
        "full delivered pipeline) by design - volume normalisation is a pure gain stage and does "
        "not change speech quality or intelligibility, only loudness. Table 5's higher SNR-after "
        "figures reflect the normaliser lifting the output level.",
        st["small"],
    ))

    doc.build(story)
    return out_path


if __name__ == "__main__":
    session_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "sessions/2026-09-09_12-14-00_offline_file"
    )
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("sessions/comparison_tables.pdf")
    if not (session_arg / "metrics.csv").is_file():
        raise SystemExit(f"no metrics.csv at {session_arg}")
    written = build(session_arg, out_arg)
    print(f"wrote {written}")
