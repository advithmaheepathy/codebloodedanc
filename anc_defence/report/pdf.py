"""PDF report generation with reportlab.

Works entirely offline with no system fonts beyond reportlab's built-ins, so it
behaves the same on any platform. One entry point, :func:`build_pdf_report`, driven
by a :class:`~anc_defence.report.export.ReportData` container that the modes fill in.

Report structure:

1. Session metadata: timestamp, mode, host, model, git commit, effective config path
2. Scope statement: what was measured and, just as importantly, what was not
3. Mandated targets as PASS/FAIL, with the input-SNR range they are quoted over
4. Stage-by-stage metrics, per-category breakdown, baseline comparison
5. Figures: waveforms, spectrograms, ERLE, latency, resources
6. How to read each metric
7. Every warning raised during the run
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..utils.logging import get_logger
from .export import ReportData, ReportTable

log = get_logger(__name__)

PASS_COLOUR = colors.HexColor("#1a7f37")
FAIL_COLOUR = colors.HexColor("#b42318")
NA_COLOUR = colors.HexColor("#6e6e6e")
HEADER_BG = colors.HexColor("#eceff4")
ACCENT = colors.HexColor("#1f77b4")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=17, leading=21, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=colors.HexColor("#444444"), spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=12.5, leading=15, spaceBefore=12,
            spaceAfter=5, textColor=ACCENT,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=3
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=8.5, leading=11.5, alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=7.2, leading=9.5,
            textColor=colors.HexColor("#555555"), spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontSize=7.2, leading=9,
            textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=8,
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Normal"], fontName="Courier", fontSize=7, leading=9
        ),
    }


def _kv_table(rows: Sequence[tuple[str, str]], width: float = 17.0 * cm) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", _styles()["small"]), Paragraph(str(v), _styles()["small"])]
            for k, v in rows]
    table = Table(data, colWidths=[width * 0.32, width * 0.68], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _data_table(rt: ReportTable, total_width: float = 17.0 * cm) -> Table:
    n_cols = max(len(r) for r in rt.rows)
    padded = [list(r) + [""] * (n_cols - len(r)) for r in rt.rows]
    widths = (
        [total_width * w for w in rt.col_widths]
        if rt.col_widths
        else [total_width / n_cols] * n_cols
    )
    table = Table(padded, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), rt.font_size),
        ("LEADING", (0, 0), (-1, -1), rt.font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
    ]
    # Colour PASS/FAIL cells wherever they appear.
    for r, row in enumerate(padded):
        for c, cell in enumerate(row):
            text = str(cell).strip()
            if text == "PASS":
                style.append(("TEXTCOLOR", (c, r), (c, r), PASS_COLOUR))
                style.append(("FONTNAME", (c, r), (c, r), "Helvetica-Bold"))
            elif text == "FAIL":
                style.append(("TEXTCOLOR", (c, r), (c, r), FAIL_COLOUR))
                style.append(("FONTNAME", (c, r), (c, r), "Helvetica-Bold"))
            elif text == "N/A":
                style.append(("TEXTCOLOR", (c, r), (c, r), NA_COLOUR))
    table.setStyle(TableStyle(style))
    return table


def _target_table(data: ReportData) -> Optional[Table]:
    if not data.target_checks:
        return None
    rows = [["Target", "Required", "Measured", "Verdict"]]
    for c in data.target_checks:
        measured = "n/a" if c.value != c.value else f"{c.value:.3f}"  # NaN check
        rows.append([c.name, f"{c.comparison} {c.target:g}", measured, c.verdict])
    return _data_table(ReportTable(name="targets", rows=rows, font_size=8.0))


def build_pdf_report(data: ReportData) -> Path:
    """Render the PDF for one session."""
    st = _styles()
    path = data.session.pdf_path
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=data.title,
        author="anc_defence",
    )
    story: list[Any] = []

    # ---------------------------------------------------------------- header
    story.append(Paragraph(data.title, st["title"]))
    if data.subtitle:
        story.append(Paragraph(data.subtitle, st["subtitle"]))

    story.append(Paragraph("Session", st["h1"]))
    story.append(_kv_table(data.metadata_rows))

    # ----------------------------------------------------------------- scope
    if data.scope_notes:
        story.append(Paragraph("Scope and honesty statement", st["h1"]))
        for note in data.scope_notes:
            story.append(Paragraph(f"&bull; {note}", st["body"]))

    # --------------------------------------------------------------- targets
    target_table = _target_table(data)
    if target_table is not None:
        story.append(Paragraph("Mandated performance targets", st["h1"]))
        if data.target_scope:
            story.append(Paragraph(data.target_scope, st["small"]))
        story.append(target_table)
        story.append(Spacer(1, 6))

    # ---------------------------------------------------------------- tables
    for rt in data.tables:
        block: list[Any] = [Paragraph(rt.name, st["h2"])]
        if rt.caption:
            block.append(Paragraph(rt.caption, st["small"]))
        block.append(_data_table(rt))
        block.append(Spacer(1, 6))
        story.append(KeepTogether(block) if len(rt.rows) <= 12 else block[0])
        if len(rt.rows) > 12:
            if rt.caption:
                story.append(Paragraph(rt.caption, st["small"]))
            story.append(_data_table(rt))
            story.append(Spacer(1, 6))

    # --------------------------------------------------------------- figures
    if data.figures:
        story.append(PageBreak())
        story.append(Paragraph("Figures", st["h1"]))
        for fig in data.figures:
            try:
                img = Image(str(fig.path))
                aspect = img.imageHeight / float(img.imageWidth)
                width = min(fig.width_cm * cm, 17.0 * cm)
                img.drawWidth = width
                img.drawHeight = width * aspect
                # Keep tall figures on one page where possible.
                if img.drawHeight > 23.0 * cm:
                    scale = (23.0 * cm) / img.drawHeight
                    img.drawHeight *= scale
                    img.drawWidth *= scale
                story.append(img)
                if fig.caption:
                    story.append(Paragraph(fig.caption, st["caption"]))
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("could not embed figure %s: %s", fig.path, exc)

    # -------------------------------------------------------- interpretation
    if data.interpretation:
        story.append(PageBreak())
        story.append(Paragraph("How to read these numbers", st["h1"]))
        for name, text in data.interpretation:
            story.append(Paragraph(f"<b>{name}</b>", st["h2"]))
            story.append(Paragraph(text, st["body"]))

    # -------------------------------------------------------------- warnings
    all_warnings = list(dict.fromkeys(data.warnings + data.session.warnings.messages()))
    story.append(Paragraph("Warnings and events", st["h1"]))
    if all_warnings:
        for w in all_warnings:
            story.append(Paragraph(f"&bull; {w}", st["small"]))
    else:
        story.append(Paragraph("None raised during this run.", st["small"]))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"Full effective configuration: {data.session.config_path.name} &middot; "
            f"log: {data.session.log_path.name} &middot; "
            f"machine-readable results: {data.session.json_path.name}",
            st["small"],
        )
    )

    doc.build(story, onLaterPages=_footer, onFirstPage=_footer)
    log.info("wrote %s", path)
    return path


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2.0 * cm, 1.0 * cm, "AI/ML adaptive noise cancellation for defence comms - SIH PS 26052")
    canvas.drawRightString(A4[0] - 2.0 * cm, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


INTERPRETATION_NOTES: list[tuple[str, str]] = [
    (
        "PESQ (wideband, ITU-T P.862.2)",
        "Predicted listening quality from 1.0 to about 4.6, computed at 16 kHz against the clean "
        "reference. Above 2.5 is the target here. PESQ is sensitive to speech distortion, so an "
        "aggressive noise gate can lower it even while the background gets quieter.",
    ),
    (
        "STOI / ESTOI",
        "Short-time objective intelligibility, 0 to 1: the predicted fraction of words a listener "
        "would get right. ESTOI is the extended version and is stricter with modulated noise. "
        "0.85 is the target. Both are computed at 10 kHz internally.",
    ),
    (
        "SI-SDR and SNR improvement",
        "SI-SDR projects the output onto the clean signal, so it separates surviving speech from "
        "everything else and is immune to overall gain changes. SNR improvement is the SI-SDR of "
        "the output minus the SI-SDR of the unprocessed input. It is bounded by how much noise was "
        "there to begin with, which is why it is reported per input-SNR bucket: at +20 dB input "
        "there is not 15 dB of noise left to remove.",
    ),
    (
        "Segmental SNR",
        "Frame-by-frame SNR averaged over active frames and clamped to [-10, +35] dB so silence "
        "cannot dominate. Closer to perceived improvement than a global figure.",
    ),
    (
        "ERLE",
        "Energy removed by the adaptive stage, 10*log10(input power / output power) per block. It "
        "measures how much energy went away, not whether that energy was noise. Always read it "
        "next to speech attenuation and speech distortion.",
    ),
    (
        "Speech attenuation / distortion",
        "Attenuation is how much quieter the speech component became (positive is worse). "
        "Distortion is the energy of the non-scalable error relative to the speech, in dB (more "
        "negative is better). These are the numbers that catch a system that 'wins' by muting the "
        "talker.",
    ),
    (
        "Transient suppression and speech dropout",
        "For impulsive noise only. Transient suppression is the level drop at each detected event. "
        "Speech dropout counts windows around events where the enhancer removed the speech as well "
        "as the transient, which is what causes audible holes in a gunshot scenario.",
    ),
    (
        "RTF (real-time factor)",
        "Processing time divided by audio duration. Below 1.0 means faster than real time; the "
        "budget used here is 0.5 in total. Measured on the host described in the session table, "
        "with no claim made about other hardware.",
    ),
    (
        "Latency",
        "The neural model's algorithmic latency is fixed by its architecture: one 10 ms frame, plus "
        "10 ms for the STFT/ISTFT loop (n_fft - hop), plus a 2-frame lookahead (20 ms), giving "
        "40 ms. Any chunk buffering used by the live path is listed separately and added on top; it "
        "is buffering latency, not low-latency streaming.",
    ),
]
