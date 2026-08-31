"""
Render an AuditReport as a single self-contained HTML file.

Stdlib only. This is the zero-dependency path: no install, opens in any
browser, prints to PDF with Cmd-P, and can be emailed as one file. The PDF
renderer is nicer to hand over but this one always works.

Design intent is a document, not a web page. A grader audit is read once by
somebody deciding whether to act, so the ordering is verdict, then what was
not checked, then the findings -- coverage before findings, because a reader
who skims only the top must not come away thinking the absence of findings
means the absence of defects.
"""
from __future__ import annotations

import html
from typing import List

from rlverify.report.model import AuditReport, Finding, Target

CSS = """
:root{--fg:#16181d;--mut:#5b6470;--line:#dfe3e8;--bg:#fff;
      --high:#b3261e;--med:#8a5a00;--low:#5b6470;--ok:#1e6b3a;--code:#f6f7f9}
*{box-sizing:border-box}
body{margin:0;padding:48px 56px;max-width:52em;color:var(--fg);background:var(--bg);
     font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:25px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.075em;color:var(--mut);
   margin:38px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 6px}
p{margin:0 0 11px}
.sub{color:var(--mut);font-size:13.5px;margin-bottom:26px}
.meta{width:100%;border-collapse:collapse;font-size:13.5px;margin:0 0 8px}
.meta td{padding:5px 0;vertical-align:top;border-bottom:1px solid var(--line)}
.meta td:first-child{color:var(--mut);width:12.5em}
.verdict{border:1px solid var(--line);border-left:4px solid var(--low);
         padding:15px 18px;margin:22px 0;background:#fafbfc}
.verdict.high{border-left-color:var(--high)}
.verdict.ok{border-left-color:var(--ok)}
.verdict .v{font-size:18px;font-weight:600;margin-bottom:3px}
.badge{display:inline-block;padding:1px 8px;border-radius:3px;font-size:11px;
       font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#fff}
.badge.high{background:var(--high)}.badge.medium{background:var(--med)}
.badge.low{background:var(--low)}
.finding{border:1px solid var(--line);border-radius:5px;padding:16px 18px;margin:14px 0;
         page-break-inside:avoid}
.finding .hd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:9px}
.finding .rule{font-weight:650;font-size:15.5px}
.loc{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--mut)}
pre{background:var(--code);border:1px solid var(--line);border-radius:4px;
    padding:11px 13px;overflow-x:auto;font-size:12.5px;line-height:1.5;margin:9px 0;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
     font-weight:700;margin-top:11px}
.gap{border-left:3px solid var(--med);background:#fdfaf3;padding:11px 15px;margin:11px 0}
.gap ul{margin:6px 0 0;padding-left:19px}
.note{color:var(--mut);font-size:13px}
.withdrawn{opacity:.55}
.rev{font-size:11.5px;color:var(--mut);border:1px solid var(--line);border-radius:3px;
     padding:1px 7px}
table.d{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}
table.d th{text-align:left;font-weight:650;border-bottom:1.5px solid var(--line);padding:6px 8px 6px 0}
table.d td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line)}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);
       color:var(--mut);font-size:12.5px}
@media print{body{padding:0;font-size:11.5pt;max-width:none}
  h2{margin-top:22px}.finding{border-color:#bbb}a{text-decoration:none;color:inherit}}
"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _finding_block(f: Finding) -> str:
    cls = "finding withdrawn" if f.withdrawn else "finding"
    rev = ""
    if f.review_verdict:
        word = {"confirmed": "confirmed on manual review",
                "false_positive": "WITHDRAWN on manual review",
                "unclear": "reviewed, inconclusive"}.get(f.review_verdict,
                                                         f.review_verdict)
        rev = '<span class="rev">%s</span>' % _e(word)
    parts = [
        '<div class="%s">' % cls,
        '  <div class="hd"><span class="badge %s">%s</span>'
        '<span class="rule">%s</span><span class="loc">%s</span>%s</div>'
        % (_e(f.severity), _e(f.severity), _e(f.rule), _e(f.location), rev),
    ]
    if f.snippet:
        parts.append("  <pre>%s</pre>" % _e(f.snippet))
    if f.why:
        parts.append('  <div class="lbl">Why this matters</div><p>%s</p>' % _e(f.why))
    if f.probe:
        parts.append('  <div class="lbl">What would confirm it</div>'
                     '<p class="note">A dynamic probe of family '
                     '<code>%s</code>: fire the corresponding deliberately-wrong '
                     'response at the live grader and measure whether it is paid. '
                     'Until that runs, this is a smell, not a confirmed exploit.</p>'
                     % _e(f.probe))
    parts.append('  <p class="note">This rule fires on %.1f%% of the audited '
                 'corpus.</p>' % (100 * f.corpus_rate))
    parts.append("</div>")
    return "\n".join(parts)


def _target_section(t: Target, single: bool) -> str:
    out: List[str] = []
    if not single:
        out.append("<h3>%s</h3>" % _e(t.label))
    tier = []
    if t.is_code_grader:
        tier.append("executes response-derived code")
    if t.is_opaque:
        tier.append("delegates execution to machinery this pass cannot read")
    if not t.has_reward_func:
        tier.append("no reward function identified")
    out.append('<table class="meta">')
    out.append("<tr><td>Environment</td><td><code>%s</code></td></tr>" % _e(t.label))
    if t.requires_python:
        out.append("<tr><td>Requires Python</td><td>%s</td></tr>" % _e(t.requires_python))
    out.append("<tr><td>Grader type</td><td>%s</td></tr>"
               % _e("; ".join(tier) if tier else "answer-parsing grader"))
    out.append("<tr><td>Files analysed</td><td>%d parsed of %d considered</td></tr>"
               % (t.coverage.files_parsed, t.coverage.files_considered))
    out.append("</table>")

    gaps = t.coverage.gaps()
    if gaps:
        out.append('<div class="gap"><strong>Coverage is incomplete.</strong> '
                   'A gap is not a clean result &mdash; anything below describes '
                   'only what was examined.<ul>')
        out.extend("<li>%s</li>" % _e(g) for g in gaps)
        out.append("</ul></div>")
    else:
        out.append('<p class="note">Every Python file in this package was '
                   'retrieved and parsed. No coverage gaps.</p>')

    live = t.live_findings
    if live:
        out.extend(_finding_block(f) for f in t.findings)
    else:
        out.append("<p>No findings.</p>")
    return "\n".join(out)


def to_html(rep: AuditReport) -> str:
    single = len(rep.targets) == 1
    high = rep.n_high
    vcls = "verdict high" if high else ("verdict ok" if rep.targets else "verdict")
    headline = rep.headline()

    body: List[str] = [
        "<h1>Grader hackability audit</h1>",
        '<div class="sub">%s &middot; generated %s &middot; rlverify static pass</div>'
        % (_e(rep.scope_label), _e(rep.generated_at)),
        '<div class="%s"><div class="v">%s</div>'
        '<div class="note">The question this answers: can a response score well '
        'without solving the task?</div></div>' % (vcls, _e(headline)),
    ]

    body.append("<h2>Summary</h2>")
    body.append('<table class="d"><tr><th>Severity</th><th>%s</th></tr>'
                % _e(rep.severity_unit()))
    for sev in ("high", "medium", "low"):
        n = rep.n_severity(sev)
        body.append('<tr><td><span class="badge %s">%s</span></td><td>%d</td></tr>'
                    % (sev, sev, n))
    body.append("</table>")

    if any(rep.review.values()):
        body.append("<h2>Manual review</h2>")
        body.append("<p>Findings are read against the real source before they are "
                    "reported. Withdrawn findings are subtracted from every count "
                    "above rather than hidden. %s</p>" % _e(rep.review_unit_note()))
        body.append('<table class="d">')
        for k, lab in (("confirmed", "Confirmed on review"),
                       ("false_positive", "Withdrawn as false positives"),
                       ("unclear", "Reviewed, inconclusive"),
                       ("unreviewed", "Not yet reviewed")):
            body.append("<tr><td>%s</td><td>%d</td></tr>" % (lab, rep.review.get(k, 0)))
        body.append("</table>")

    body.append("<h2>%s</h2>" % ("Findings" if single else "Environments"))
    if rep.detail_shown:
        body.extend(_target_section(t, single) for t in rep.targets)
    else:
        body.append("<p>%d environments audited; per-environment detail is omitted "
                    "at this scope. Re-run scoped to a single environment for the "
                    "full document.</p>" % rep.n_targets)

    body.append("<h2>Method and limitations</h2>")
    body.append(
        "<p><strong>Nothing was installed, imported, or executed.</strong> Source "
        "was retrieved over HTTPS and analysed as a syntax tree. This is a "
        "deliberate constraint: the audited package is the untrusted party, and "
        "installing it would execute its code at build time.</p>"
        "<p><strong>Static findings are smells, not confirmed exploits.</strong> "
        "Each one names the dynamic probe that would settle it. A finding marked "
        "<em>confirmed on manual review</em> has additionally been read against "
        "the real source by a reviewer.</p>"
        "<p><strong>A coverage gap is not a clean result.</strong> Files that "
        "could not be read, listings that were truncated, and files beyond the "
        "scan budget are counted and reported separately. Where this document "
        "says no defects were found, it means none were found in what was "
        "examined &mdash; which is stated exactly.</p>"
        "<p><strong>What this pass cannot see.</strong> Graders that hand the "
        "response to a container, a remote sandbox, or a terminal harness decide "
        "reward in task data rather than in Python source. Those are reported as "
        "delegated rather than as clean.</p>")

    body.append("<footer>Generated by rlverify &middot; static analysis pass "
                "&middot; %s<br>This document describes what was examined and "
                "what was not. Both halves are part of the result."
                "</footer>" % _e(rep.generated_at))

    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Grader audit &mdash; %s</title><style>%s</style></head><body>\n"
            "%s\n</body></html>\n" % (_e(rep.scope_label), CSS, "\n".join(body)))
