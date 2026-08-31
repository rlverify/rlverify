"""
The audit report as data, separated from how it is rendered.

`scan.report()` printed straight to stdout and returned None, so the only way
to get an audit out of this tool was to copy a terminal buffer. A pre-delivery
audit is a document somebody receives; the numbers have to exist as values
before they can be a document.

Deliberately stdlib-only and 3.9-compatible, the same as the rest of the
corpus half. Rendering to text and HTML needs nothing installed; only the PDF
renderer has a dependency, and it is imported lazily so its absence costs you
one output format rather than the whole module.
"""
from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rlverify.corpus import store

# Per-target detail is rendered up to this many environments. A corpus sweep
# produces 1366 targets and nobody reads that; a customer audit is one.
MAX_DETAIL_TARGETS = 25

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    path: str
    lineno: int
    snippet: str
    why: str
    probe: str
    reviewed: bool
    review_verdict: Optional[str]
    corpus_rate: float          # share of the corpus this rule fires on

    @property
    def withdrawn(self) -> bool:
        return self.review_verdict == "false_positive"

    @property
    def location(self) -> str:
        return "%s:%d" % (self.path, self.lineno or 0)


@dataclass(frozen=True)
class Coverage:
    """What was examined, and -- more importantly -- what was not.

    Every field here exists so the report can distinguish "we looked and found
    nothing" from "we could not look". Collapsing those two is the single
    easiest way for an audit to be quietly worthless.
    """

    files_considered: int = 0
    files_scanned: int = 0
    files_over_cap: int = 0
    listing_truncated: bool = False
    files_parsed: int = 0
    files_parse_failed: int = 0
    files_unavailable: int = 0

    @property
    def complete(self) -> bool:
        return not (self.files_over_cap or self.listing_truncated
                    or self.files_parse_failed or self.files_unavailable)

    def gaps(self) -> List[str]:
        """Plain-language coverage gaps, for the part of the report that matters."""
        out = []
        if self.listing_truncated:
            out.append("the file listing from the Hub was truncated, so this "
                       "package may contain files never seen by this audit")
        if self.files_over_cap:
            out.append("%d file(s) were beyond the per-environment scan budget "
                       "and were not analysed" % self.files_over_cap)
        if self.files_parse_failed:
            out.append("%d file(s) could not be parsed" % self.files_parse_failed)
        if self.files_unavailable:
            out.append("%d file(s) could not be downloaded" % self.files_unavailable)
        return out


@dataclass(frozen=True)
class Target:
    env_key: str
    owner: str
    name: str
    version: str
    sha256: Optional[str]
    requires_python: Optional[str]
    has_reward_func: Optional[bool]
    is_code_grader: Optional[bool]
    is_opaque: Optional[bool]
    coverage: Coverage
    findings: List[Finding] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "%s/%s@%s" % (self.owner, self.name, self.version or "")

    @property
    def live_findings(self) -> List[Finding]:
        """Findings a reviewer has not withdrawn."""
        return [f for f in self.findings if not f.withdrawn]

    def by_severity(self, sev: str) -> List[Finding]:
        return [f for f in self.live_findings if f.severity == sev]

    @property
    def verdict(self) -> str:
        """One line. Never says `clean` when coverage was incomplete."""
        if self.by_severity("high"):
            return "defects found"
        if not self.coverage.complete:
            return "no defects found, coverage incomplete"
        if self.live_findings:
            return "no high-severity defects found"
        if not self.has_reward_func:
            return "inconclusive: no reward function identified"
        if self.is_opaque:
            return "inconclusive: grading is delegated to machinery this pass cannot read"
        return "no defects found by this pass"


@dataclass(frozen=True)
class AuditReport:
    generated_at: str
    scope_label: str
    scoped: bool                      # a single target vs a corpus sweep
    targets: List[Target]
    n_targets: int
    counts: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, int] = field(default_factory=dict)
    corpus_denominator: int = 0
    # Counted in SQL over the whole scope, NOT derived from `targets`.
    # `targets` is empty above MAX_DETAIL_TARGETS, and deriving the headline
    # from it made a 1367-environment report announce "no high-severity
    # findings" when there were 18 -- a clean verdict produced by not looking,
    # which is the exact failure this tool exists to catch in other people's
    # graders. The summary must never depend on how much detail was rendered.
    severity_counts: Dict[str, int] = field(default_factory=dict)
    severity_findings: Dict[str, int] = field(default_factory=dict)

    @property
    def all_findings(self) -> List[Finding]:
        """Findings for the rendered targets. Detail only -- never the headline."""
        out = []
        for t in self.targets:
            out.extend(t.live_findings)
        return sorted(out, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))

    @property
    def n_high(self) -> int:
        return self.severity_counts.get("high", 0)

    def n_severity(self, sev: str) -> int:
        """Findings for one target, affected environments for a sweep.

        Both are true; which one is the useful number depends on the scope. A
        single audit wants "you have 3 high-severity findings"; a corpus claim
        wants "18 environments are affected", because three findings in one
        environment is one environment's problem.
        """
        if len(self.targets) == 1:
            return self.severity_findings.get(sev, 0)
        return self.severity_counts.get(sev, 0)

    def severity_unit(self) -> str:
        return "Findings" if len(self.targets) == 1 else "Environments affected"

    def review_unit_note(self) -> str:
        """Name the unit of the review block, which is not the table's above it.

        `review` always counts *findings, all severities*. In a corpus report
        the summary table directly above counts *environments, one severity per
        row*. Unlabelled, "HIGH 18 ... Confirmed on review 22" reads as 22 of
        18 confirmed. Both numbers are right; only the presentation lied.
        """
        if len(self.targets) == 1:
            return "Counts below are findings, across all severities."
        return ("Counts below are findings, across all severities — not "
                "environments, which is what the severity table above counts.")

    def headline(self) -> str:
        """The one line at the top. Must agree with the table beneath it.

        This used to read "%d high-severity finding(s) across %d environments"
        while interpolating `n_high` -- which is the *environment* count, the
        same number the summary table labels "Environments affected". So the
        document contradicted itself one line apart, and it did so in both
        renderers because each carried its own copy of the string. It lives
        here now for that reason: the unit and the number are chosen together.

        Environments is the right unit for a corpus claim and the conservative
        one -- some environments carry more than one high finding, so it is
        always the smaller number.
        """
        if len(self.targets) == 1:
            return self.targets[0].verdict
        n_env = self.severity_counts.get("high", 0)
        if not n_env:
            return "No high-severity findings across %d environments" % self.n_targets
        n_find = self.severity_findings.get("high", 0)
        return ("%d of %d environments carry a high-severity finding (%d finding%s)"
                % (n_env, self.n_targets, n_find, "" if n_find == 1 else "s"))

    @property
    def detail_shown(self) -> bool:
        """Whether per-target detail is actually present, not whether it fits.

        `len(targets) <= MAX_DETAIL_TARGETS` is True for an empty list, so the
        corpus report claimed to be showing detail while rendering none.
        """
        return bool(self.targets) or self.n_targets == 0


def _rule_rates(conn: sqlite3.Connection, denom: int) -> Dict[str, float]:
    """Corpus-wide share of environments each rule fires on.

    Carried into a single-target report on purpose: a target's useful question
    is not only "do I have this" but "is this common", and a rule firing on 15%
    of the corpus is a different conversation from one firing on 0.1%.
    """
    if not denom:
        return {}
    rows = conn.execute(
        "SELECT f.rule, COUNT(DISTINCT f.env_key) e FROM findings f "
        "JOIN environments e USING (env_key) WHERE e.research_subject = 0 "
        "AND COALESCE(f.review_verdict,'') != 'false_positive' GROUP BY f.rule")
    return {r[0]: r[1] / denom for r in rows}


def build_report(conn: sqlite3.Connection,
                 scope_keys: Optional[List[str]] = None,
                 counters: Optional[Dict[str, Any]] = None) -> AuditReport:
    """Assemble the report. Reads only; safe against a running sweep."""
    counters = counters or {}
    denom = conn.execute(
        "SELECT COUNT(DISTINCT f.env_key) FROM files f "
        "JOIN environments e USING (env_key) "
        "WHERE f.parse_ok = 1 AND e.research_subject = 0").fetchone()[0]
    rates = _rule_rates(conn, denom)

    # `research_subject = 0` protects a corpus denominator. An explicit key list
    # has no denominator to protect -- the caller already chose -- so applying it
    # there renders a blank audit for a red-team environment that has findings.
    where = "WHERE has_reward_func IS NOT NULL"
    params: List[Any] = []
    if scope_keys is None:
        where += " AND research_subject = 0"
    else:
        # An empty list means no environments. Testing truthiness here is what
        # let an empty scope fall back to the whole corpus.
        where += " AND env_key IN (%s)" % ", ".join("?" * len(scope_keys))
        params = list(scope_keys)
    env_rows = conn.execute(
        "SELECT env_key, owner, name, version, sha256, requires_python, "
        "has_reward_func, is_code_grader, is_opaque, files_considered, "
        "files_scanned, files_over_cap, listing_truncated "
        "FROM environments " + where + " ORDER BY name", params).fetchall()

    detail = len(env_rows) <= MAX_DETAIL_TARGETS
    targets: List[Target] = []
    for e in (env_rows if detail else []):
        fc = conn.execute(
            "SELECT COUNT(*) FROM files WHERE env_key=? AND parse_ok=1",
            (e["env_key"],)).fetchone()[0]
        ff = conn.execute(
            "SELECT COUNT(*) FROM files WHERE env_key=? AND parse_ok=0",
            (e["env_key"],)).fetchone()[0]
        fu = conn.execute(
            "SELECT COUNT(*) FROM files WHERE env_key=? AND parse_error='unavailable'",
            (e["env_key"],)).fetchone()[0]
        finds = [
            Finding(rule=r["rule"], severity=r["severity"] or "low",
                    path=r["path"], lineno=r["lineno"] or 0,
                    snippet=(r["snippet"] or "").strip(), why=r["why"] or "",
                    probe=r["probe"] or "", reviewed=bool(r["reviewed"]),
                    review_verdict=r["review_verdict"],
                    corpus_rate=rates.get(r["rule"], 0.0))
            for r in conn.execute(
                "SELECT rule, severity, path, lineno, snippet, why, probe, "
                "reviewed, review_verdict FROM findings WHERE env_key=?",
                (e["env_key"],))
        ]
        finds.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.path, f.lineno))
        targets.append(Target(
            env_key=e["env_key"], owner=e["owner"], name=e["name"],
            version=e["version"] or "", sha256=e["sha256"],
            requires_python=e["requires_python"],
            has_reward_func=bool(e["has_reward_func"]),
            is_code_grader=bool(e["is_code_grader"]),
            is_opaque=bool(e["is_opaque"]),
            coverage=Coverage(
                files_considered=e["files_considered"] or 0,
                files_scanned=e["files_scanned"] or 0,
                files_over_cap=e["files_over_cap"] or 0,
                listing_truncated=bool(e["listing_truncated"]),
                files_parsed=fc, files_parse_failed=ff, files_unavailable=fu),
            findings=finds))

    if detail and len(targets) == 1:
        label, scoped = targets[0].label, True
    elif detail and targets:
        label, scoped = "%d environments" % len(targets), True
    else:
        label, scoped = "Environments Hub corpus (%d environments)" % len(env_rows), False

    rev = {}
    # Same bug, same function, one query group down: `research_subject=0` was
    # hardcoded into every one of these queries and `scope_sql` only ever added
    # a filter on top of it, so (a) an empty scope_keys was falsy here too and
    # fell back to corpus-wide review/severity counts, and (b) even a correctly
    # non-empty scope on a research-subject target had its counts zeroed by the
    # hardcoded exclusion -- the exact "blanks an explicit scope" failure this
    # task exists to remove, just one query group further than the diff for
    # `where` above reached. Mirror that block's None-vs-list split instead.
    if scope_keys is None:
        scope_sql = " AND e.research_subject = 0"
    elif not scope_keys:
        scope_sql = " AND 0"
    else:
        scope_sql = " AND f.env_key IN (%s)" % ", ".join(
            "'%s'" % k.replace("'", "''") for k in scope_keys)
    for v in ("confirmed", "false_positive", "unclear"):
        rev[v] = conn.execute(
            "SELECT COUNT(*) FROM findings f JOIN environments e USING (env_key) "
            "WHERE f.review_verdict=?" + scope_sql,
            (v,)).fetchone()[0]
    rev["unreviewed"] = conn.execute(
        "SELECT COUNT(*) FROM findings f JOIN environments e USING (env_key) "
        "WHERE f.reviewed=0" + scope_sql).fetchone()[0]

    # Severity counts over the whole scope, in SQL, so they hold whether or not
    # per-target detail was rendered. Environments, not raw rows: "3 findings"
    # in one environment is a different fact from three environments.
    sev_counts: Dict[str, int] = {}
    sev_finds: Dict[str, int] = {}
    for sev in ("high", "medium", "low"):
        row = conn.execute(
            "SELECT COUNT(DISTINCT f.env_key), COUNT(*) FROM findings f "
            "JOIN environments e USING (env_key) WHERE "
            "COALESCE(f.review_verdict,'') != 'false_positive' "
            "AND f.severity=?" + scope_sql, (sev,)).fetchone()
        sev_counts[sev], sev_finds[sev] = row[0], row[1]

    return AuditReport(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d"),
        scope_label=label, scoped=scoped, targets=targets,
        n_targets=len(env_rows), counts=dict(counters), review=rev,
        corpus_denominator=denom, severity_counts=sev_counts,
        severity_findings=sev_finds)


if __name__ == "__main__":
    conn = store.connect(readonly=True)
    keys = [r[0] for r in conn.execute(
        "SELECT env_key FROM environments WHERE name='code-repair-env'")]
    rep = build_report(conn, scope_keys=keys)
    failures = 0
    print("scope            : %s" % rep.scope_label)
    print("targets          : %d (detail=%s)" % (rep.n_targets, rep.detail_shown))
    for t in rep.targets:
        print("  %-34s verdict=%s" % (t.label, t.verdict))
        print("    coverage complete: %s   gaps: %s"
              % (t.coverage.complete, t.coverage.gaps() or "none"))
        for f in t.findings:
            print("    [%-6s] %-22s %s  corpus=%.1f%%  review=%s"
                  % (f.severity, f.rule, f.location, 100 * f.corpus_rate,
                     f.review_verdict or "-"))
    if not rep.targets:
        failures += 1
        print("NO TARGETS -- expected code-repair-env")

    # The headline must not depend on how much detail was rendered. A corpus
    # report exceeds MAX_DETAIL_TARGETS and therefore carries no per-target
    # objects at all; deriving the severity summary from those objects made it
    # announce "no high-severity findings" over a corpus holding 18. A report
    # that reads clean because it did not look is the precise failure this tool
    # accuses other people's graders of.
    truth = conn.execute(
        "SELECT COUNT(DISTINCT f.env_key) FROM findings f "
        "JOIN environments e USING (env_key) WHERE e.research_subject=0 "
        "AND f.severity='high' AND COALESCE(f.review_verdict,'') "
        "!= 'false_positive'").fetchone()[0]
    corpus = build_report(conn)
    print("\ncorpus headline  : %d high   (store says %d)" % (corpus.n_high, truth))
    print("per-target detail: %s (targets rendered=%d)"
          % (corpus.detail_shown, len(corpus.targets)))
    if corpus.n_high != truth:
        failures += 1
        print("HEADLINE WRONG: report says %d high, store says %d"
              % (corpus.n_high, truth))
    if truth and len(corpus.targets) == 0 and corpus.n_high == 0:
        failures += 1
        print("REGRESSION: summary derived from rendered detail, not from the store")
    # A target with a high-severity finding must never read as clean.
    for t in rep.targets:
        if t.by_severity("high") and "defect" not in t.verdict:
            failures += 1
            print("VERDICT WRONG: high severity but verdict=%r" % t.verdict)
    # An explicit scope is a caller's decision, not a hint. Empty means "no
    # environments", never "all of them" -- the fallback that made a report for a
    # nonexistent environment list 385 findings across 212 unrelated ones.
    empty = build_report(conn, scope_keys=[])
    if empty.n_targets != 0 or any(empty.severity_counts.values()):
        failures += 1
        print("FAIL: scope_keys=[] widened to the corpus: n_targets=%d counts=%r"
              % (empty.n_targets, empty.severity_counts))

    # A research-subject environment is excluded from corpus denominators, not
    # from an audit that explicitly asked for it. Rendering zero targets for an
    # environment that has findings is a clean verdict produced by not looking.
    row = conn.execute(
        "SELECT f.env_key FROM findings f JOIN environments e USING (env_key) "
        "WHERE e.research_subject = 1 AND e.has_reward_func IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is None:
        print("SKIP: no scanned research-subject environment with findings in store")
    else:
        rep = build_report(conn, scope_keys=[row[0]])
        if rep.n_targets != 1:
            failures += 1
            print("FAIL: explicit scope on research subject %s rendered %d targets"
                  % (row[0], rep.n_targets))
        # `n_targets` above is produced solely by the `where` fragment -- the
        # literal Step 3 fix. `severity_counts` and `review` are produced by
        # the separate `scope_sql` fragment a query group down, and nothing
        # above ever asserts on it: a regression that reintroduced
        # `research_subject=0` into scope_sql's non-empty branch would still
        # render this target while silently re-zeroing its severity data, and
        # this gate would still print "0 failure(s)" -- a clean pass produced
        # by not looking. `target.live_findings` comes from a per-env_key
        # query with no scope_sql in it at all, so it is independent ground
        # truth for whether this target actually has findings to report.
        elif rep.targets[0].live_findings and not any(rep.severity_counts.values()):
            failures += 1
            print("FAIL: research subject %s has %d live finding(s) but "
                  "severity_counts is all-zero: %r"
                  % (row[0], len(rep.targets[0].live_findings), rep.severity_counts))

    print("\n%d failure(s)" % failures)
    conn.close()
    raise SystemExit(1 if failures else 0)
