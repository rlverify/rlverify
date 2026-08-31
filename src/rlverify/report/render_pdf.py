"""
Render an AuditReport as a PDF.

The only module in the corpus half with a third-party dependency, so it is
imported lazily and its absence is a missing output format rather than a
missing tool -- `render_html` produces the same document with nothing
installed, and any browser prints that to PDF.

    pip install reportlab

Layout follows the HTML renderer deliberately: verdict, then what was NOT
checked, then findings. A reader who stops after the first page must not come
away thinking "no findings" means "no defects".
"""
from __future__ import annotations

from typing import Any, List

from rlverify.report.model import AuditReport, Finding, Target

REPORTLAB_HINT = ("PDF output needs reportlab: pip install reportlab\n"
                  "Or use --format html and print to PDF from a browser.")

HIGH = "#b3261e"
MED = "#8a5a00"
LOW = "#5b6470"
MUT = "#5b6470"
LINE = "#dfe3e8"
SEV_COLOUR = {"high": HIGH, "medium": MED, "low": LOW}


def _styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import colors

    ss = getSampleStyleSheet()
    base = ss["BodyText"]
    base.fontName = "Helvetica"
    base.fontSize = 9.5
    base.leading = 13.5
    base.spaceAfter = 6
    base.alignment = TA_LEFT

    def mk(name, **kw):
        return ParagraphStyle(name, parent=base, **kw)

    return {
        "body": base,
        "h1": mk("h1", fontName="Helvetica-Bold", fontSize=18, leading=22,
                 spaceAfter=2),
        "sub": mk("sub", fontSize=9, textColor=colors.HexColor(MUT), spaceAfter=14),
        "h2": mk("h2", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
                 textColor=colors.HexColor(MUT), spaceBefore=16, spaceAfter=5),
        "h3": mk("h3", fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=10,
                 spaceAfter=3),
        "verdict": mk("verdict", fontName="Helvetica-Bold", fontSize=13, leading=17,
                      spaceAfter=3),
        "note": mk("note", fontSize=8.5, leading=12, textColor=colors.HexColor(MUT)),
        "code": mk("code", fontName="Courier", fontSize=8, leading=10.5,
                   backColor=colors.HexColor("#f6f7f9"),
                   borderColor=colors.HexColor(LINE), borderWidth=0.5,
                   borderPadding=5, spaceBefore=4, spaceAfter=6),
        "lbl": mk("lbl", fontName="Helvetica-Bold", fontSize=7.5,
                  textColor=colors.HexColor(MUT), spaceBefore=5, spaceAfter=1),
    }


def _esc(s: Any) -> str:
    """Escape for reportlab's mini-markup, which parses & < > in Paragraphs."""
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _finding_flowables(f: Finding, S) -> List[Any]:
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    colour = SEV_COLOUR.get(f.severity, LOW)
    head = ('<font color="%s"><b>%s</b></font>&nbsp;&nbsp;<b>%s</b>'
            '&nbsp;&nbsp;<font size="8" color="%s">%s</font>'
            % (colour, f.severity.upper(), _esc(f.rule), MUT, _esc(f.location)))
    if f.review_verdict:
        word = {"confirmed": "confirmed on manual review",
                "false_positive": "WITHDRAWN on manual review",
                "unclear": "reviewed, inconclusive"}.get(f.review_verdict,
                                                         f.review_verdict)
        head += '<br/><font size="8" color="%s">%s</font>' % (MUT, _esc(word))

    inner: List[Any] = [Paragraph(head, S["body"])]
    if f.snippet:
        # Long single-line snippets must wrap rather than run off the page.
        snip = _esc(f.snippet)
        inner.append(Paragraph(snip, S["code"]))
    if f.why:
        inner.append(Paragraph("WHY THIS MATTERS", S["lbl"]))
        inner.append(Paragraph(_esc(f.why), S["body"]))
    if f.probe:
        inner.append(Paragraph("WHAT WOULD CONFIRM IT", S["lbl"]))
        inner.append(Paragraph(
            "A dynamic probe of family <font face='Courier'>%s</font>: fire the "
            "corresponding deliberately-wrong response at the live grader and "
            "measure whether it is paid. Until that runs this is a smell, not a "
            "confirmed exploit." % _esc(f.probe), S["note"]))
    inner.append(Paragraph("This rule fires on %.1f%% of the audited corpus."
                           % (100 * f.corpus_rate), S["note"]))

    box = Table([[inner]], colWidths=[165 * mm])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(LINE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [box, Spacer(1, 7)]


def _kv_table(rows, S):
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    data = [[Paragraph('<font color="%s">%s</font>' % (MUT, _esc(k)), S["note"]),
             Paragraph(_esc(v), S["body"])] for k, v in rows]
    t = Table(data, colWidths=[45 * mm, 120 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _target_flowables(t: Target, single: bool, S) -> List[Any]:
    from reportlab.platypus import Paragraph, Spacer

    out: List[Any] = []
    if not single:
        out.append(Paragraph(_esc(t.label), S["h3"]))
    tier = []
    if t.is_code_grader:
        tier.append("executes response-derived code")
    if t.is_opaque:
        tier.append("delegates execution to machinery this pass cannot read")
    if not t.has_reward_func:
        tier.append("no reward function identified")
    rows = [("Environment", t.label)]
    if t.requires_python:
        rows.append(("Requires Python", t.requires_python))
    rows.append(("Grader type", "; ".join(tier) if tier else "answer-parsing grader"))
    rows.append(("Files analysed", "%d parsed of %d considered"
                 % (t.coverage.files_parsed, t.coverage.files_considered)))
    out.append(_kv_table(rows, S))
    out.append(Spacer(1, 8))

    gaps = t.coverage.gaps()
    if gaps:
        out.append(Paragraph(
            '<b>Coverage is incomplete.</b> A gap is not a clean result &mdash; '
            'what follows describes only what was examined.', S["body"]))
        for g in gaps:
            out.append(Paragraph("&bull; %s" % _esc(g), S["note"]))
        out.append(Spacer(1, 6))
    else:
        out.append(Paragraph("Every Python file in this package was retrieved and "
                             "parsed. No coverage gaps.", S["note"]))
        out.append(Spacer(1, 6))

    if t.findings:
        for f in t.findings:
            out.extend(_finding_flowables(f, S))
    else:
        out.append(Paragraph("No findings.", S["body"]))
    return out


def to_pdf(rep: AuditReport, path: str) -> str:
    """Write the report to `path`. Returns the path. Raises if reportlab is absent."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
        from reportlab.lib import colors
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(REPORTLAB_HINT) from exc

    S = _styles()
    single = len(rep.targets) == 1
    high = rep.n_high
    headline = rep.headline()

    story: List[Any] = [
        Paragraph("Grader hackability audit", S["h1"]),
        Paragraph("%s &nbsp;&middot;&nbsp; generated %s &nbsp;&middot;&nbsp; "
                  "rlverify static pass" % (_esc(rep.scope_label),
                                            _esc(rep.generated_at)), S["sub"]),
        Paragraph('<font color="%s">%s</font>'
                  % (HIGH if high else "#16181d", _esc(headline)), S["verdict"]),
        Paragraph("The question this answers: can a response score well without "
                  "solving the task?", S["note"]),
        Spacer(1, 6),
        Paragraph("SUMMARY", S["h2"]),
    ]

    sev_rows = [["Severity", rep.severity_unit()]]
    for sev in ("high", "medium", "low"):
        n = rep.n_severity(sev)
        sev_rows.append([sev.upper(), str(n)])
    st = Table(sev_rows, colWidths=[45 * mm, 120 * mm])
    st.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor(HIGH)),
        ("TEXTCOLOR", (0, 2), (0, 2), colors.HexColor(MED)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(st)

    if any(rep.review.values()):
        story.append(Paragraph("MANUAL REVIEW", S["h2"]))
        story.append(Paragraph(
            "Findings are read against the real source before they are reported. "
            "Withdrawn findings are subtracted from every count above rather "
            "than hidden. " + _esc(rep.review_unit_note()), S["body"]))
        story.append(_kv_table([
            ("Confirmed on review", rep.review.get("confirmed", 0)),
            ("Withdrawn as false positives", rep.review.get("false_positive", 0)),
            ("Reviewed, inconclusive", rep.review.get("unclear", 0)),
            ("Not yet reviewed", rep.review.get("unreviewed", 0)),
        ], S))

    story.append(Paragraph("FINDINGS" if single else "ENVIRONMENTS", S["h2"]))
    if rep.detail_shown:
        for t in rep.targets:
            story.extend(_target_flowables(t, single, S))
    else:
        story.append(Paragraph(
            "%d environments audited; per-environment detail is omitted at this "
            "scope. Re-run scoped to a single environment for the full document."
            % rep.n_targets, S["body"]))

    story.append(PageBreak())
    story.append(Paragraph("METHOD AND LIMITATIONS", S["h2"]))
    for head, text in (
        ("Nothing was installed, imported, or executed.",
         "Source was retrieved over HTTPS and analysed as a syntax tree. This is "
         "a deliberate constraint: the audited package is the untrusted party, "
         "and installing it would execute its code at build time."),
        ("Static findings are smells, not confirmed exploits.",
         "Each one names the dynamic probe that would settle it. A finding marked "
         "confirmed on manual review has additionally been read against the real "
         "source by a reviewer."),
        ("A coverage gap is not a clean result.",
         "Files that could not be read, listings that were truncated, and files "
         "beyond the scan budget are counted and reported separately. Where this "
         "document says no defects were found, it means none were found in what "
         "was examined - which is stated exactly."),
        ("What this pass cannot see.",
         "Graders that hand the response to a container, a remote sandbox or a "
         "terminal harness decide reward in task data rather than in Python "
         "source. Those are reported as delegated rather than as clean."),
    ):
        story.append(Paragraph("<b>%s</b> %s" % (_esc(head), _esc(text)), S["body"]))

    def _chrome(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUT))
        canvas.drawString(20 * mm, 12 * mm, "rlverify - %s" % rep.scope_label[:70])
        canvas.drawRightString(190 * mm, 12 * mm, "page %d" % doc.page)
        canvas.restoreState()

    SimpleDocTemplate(
        path, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="Grader hackability audit - %s" % rep.scope_label,
        author="rlverify",
    ).build(story, onFirstPage=_chrome, onLaterPages=_chrome)
    return path
