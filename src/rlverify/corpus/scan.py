"""
Corpus scan: fetch every Python file in the corpus and run the static rules.

This is the pass that produces the headline number, so its accounting is
deliberately pessimistic. Three outcomes are tracked separately and never
collapsed:

  parsed        -- we read and analysed the file; a clean result means something
  parse_failed  -- we read it but could not parse it (coverage gap)
  unavailable   -- we could not read it at all (coverage gap)

A corpus claim is only as good as its denominator. Reporting "N% of
environments are hackable" while silently dropping the files we failed to read
would inflate the rate, so both gap categories are reported alongside the
finding rate and the denominator is always stated.

Executes nothing. Source is fetched over HTTPS and parsed as text.
"""
from __future__ import annotations

import argparse
import textwrap
import time
from typing import Any, Dict, List, Optional

from rlverify.corpus import store
from rlverify.static import ast_rules
from rlverify.static.hub_client import HubClient

# Files that are never graders; skipping them cuts fetch volume substantially.
SKIP_SUFFIXES = ("/setup.py", "/conftest.py")
SKIP_PARTS = ("/test_", "/tests/", "/_version", "/migrations/")

# ...except that in a benchmark environment the tests ARE the grader. A
# harbor-layout environment ships tasks/<task>/tests/test_outputs.py and
# tasks/<task>/tests/compute_reward.py, and those files decide reward. Skipping
# them as though they were the package's own unit tests hid the statically
# readable code graders in contractbench (33 tasks), skillsbench (x3),
# frontier-swe and harbor -- which is a large part of the "only 12 of 718 are
# readable" headline this scan produced.
GRADER_BASENAMES = ("test_outputs.py",)
GRADER_PREFIXES = ("compute_reward",)

# Directories whose whole purpose is grading. A `tests/` under one of these is a
# per-task fixture, not the package's own test suite.
FIXTURE_MARKERS = ("/tasks/", "/tasks-extra/", "/verifier/")

# The test-related skips, which do not apply inside a fixture tree.
_TEST_SKIPS = ("/test_", "/tests/")


def _is_grader_by_convention(p: str) -> bool:
    """Named as a grader, so the skip-list must not reach it wherever it sits."""
    base = p.rsplit("/", 1)[-1]
    return base in GRADER_BASENAMES or (
        base.endswith(".py") and base.startswith(GRADER_PREFIXES))


def _worth_scanning(path: str) -> bool:
    p = "/" + path
    if any(p.endswith(s) for s in SKIP_SUFFIXES):
        return False
    if _is_grader_by_convention(p):
        return True
    skips = SKIP_PARTS
    if any(m in p for m in FIXTURE_MARKERS):
        skips = tuple(s for s in SKIP_PARTS if s not in _TEST_SKIPS)
    return not any(part in p for part in skips)


# A flat per-environment cap starves that same layout. One grader per task
# directory means a 33-task environment needs tens of files, not 12 -- and
# because candidates are ordered by path length, the first 12 are all top-level
# modules, so every task fixture fell off the end. Scale the budget with the
# task-directory count, keep a ceiling so a pathological repo cannot stall the
# sweep, and record when the cap actually bit: a cap applied silently is a
# truncated denominator wearing a clean result's clothes.
BASE_FILE_BUDGET = 12
PER_TASK_FILES = 2
FILE_BUDGET_CEILING = 240


def _task_dirs(paths) -> set:
    """Distinct task directories among these paths, e.g. `/tasks/foo`."""
    out = set()
    for path in paths:
        p = "/" + path
        for m in FIXTURE_MARKERS:
            if m in p:
                out.add(m + p.split(m, 1)[1].split("/", 1)[0])
    return out


def _file_budget(paths) -> int:
    n = len(_task_dirs(paths))
    if not n:
        return BASE_FILE_BUDGET
    return min(FILE_BUDGET_CEILING, BASE_FILE_BUDGET + PER_TASK_FILES * n)


def scan_corpus(conn, client: HubClient, limit: Optional[int] = None,
                verbose: bool = True, only: Optional[str] = None,
                env_key: Optional[str] = None) -> Dict[str, Any]:
    """Static-scan the synced corpus.

    `only` restricts to environments whose name matches a SQL LIKE pattern. A
    rule change invalidates just the environments that rule fired on, and
    re-running the whole corpus to re-judge eight of them wastes an hour and
    produces counters that cannot be compared with anything.

    `env_key` restricts to exactly one environment by its resolved primary
    key -- unlike `only`, no LIKE pattern involved. A caller that has already
    resolved precisely which environment it wants (an audit) should not go
    back through a pattern match to find it again.
    """
    # Classify before scanning so the exclusion can never lag the corpus.
    n_research = store.backfill_research_subject(conn)
    if verbose:
        print("research-subject environments (excluded from the denominator): %d"
              % n_research)

    sql = """
        SELECT e.env_key, e.owner, e.name, e.version, e.research_subject
          FROM environments e
         WHERE e.files_listed = 1
           AND EXISTS (SELECT 1 FROM files f
                        WHERE f.env_key = e.env_key AND f.is_python = 1)
         ORDER BY e.stars DESC, e.updated_at DESC
    """
    if only:
        sql = sql.replace("WHERE e.files_listed = 1",
                          "WHERE e.files_listed = 1 AND e.name LIKE '%s'" % only)
    if limit:
        sql += " LIMIT %d" % int(limit)
    if env_key:
        sql = sql.replace("WHERE e.files_listed = 1",
                          "WHERE e.files_listed = 1 AND e.env_key = ?")
        envs = conn.execute(sql, (env_key,)).fetchall()
    else:
        envs = conn.execute(sql).fetchall()
    if verbose:
        print("scanning %d environments with python files" % len(envs))

    counters = {
        "envs": 0, "envs_with_reward_func": 0, "envs_with_finding": 0,
        "envs_with_code_grader": 0, "envs_with_opaque_grader": 0,
        "envs_fully_unreadable": 0, "envs_excluded_research": 0,
        "files_parsed": 0, "files_parse_failed": 0, "files_unavailable": 0,
        "findings": 0,
        # A cap that bit is a coverage gap and is reported as one.
        "envs_file_capped": 0, "files_over_cap": 0,
    }
    t0 = time.monotonic()

    for i, e in enumerate(envs, 1):
        rows = conn.execute(
            "SELECT path, content_hash FROM files "
            "WHERE env_key=? AND is_python=1 ORDER BY LENGTH(path)",
            (e["env_key"],),
        ).fetchall()
        candidates = [r for r in rows if _worth_scanning(r["path"])]
        budget = _file_budget([r["path"] for r in candidates])
        paths = candidates[:budget]
        over_cap = len(candidates) - len(paths)

        # Upsert alone makes a re-scan idempotent for findings that still fire;
        # it says nothing about ones that stopped. After a rule is tightened its
        # old rows would linger and be counted, so the report would describe a
        # rule set that no longer exists. Clear first -- but never clear a
        # finding a human has already ruled on, since that verdict is the one
        # thing here a re-run cannot reproduce.
        conn.execute("DELETE FROM findings WHERE env_key=? AND reviewed=0",
                     (e["env_key"],))

        env_reward = env_finding = env_code = env_opaque = False
        readable = 0
        for r in paths:
            src = client.read_file(e["owner"], e["name"], e["version"],
                                   r["path"], r["content_hash"])
            if src is None:
                counters["files_unavailable"] += 1
                conn.execute(
                    "UPDATE files SET fetched=0, parse_ok=NULL, parse_error='unavailable' "
                    "WHERE env_key=? AND path=?", (e["env_key"], r["path"]))
                continue
            readable += 1

            scan = ast_rules.scan_source(src, r["path"])
            if scan.parse_ok:
                counters["files_parsed"] += 1
            else:
                counters["files_parse_failed"] += 1
            conn.execute(
                "UPDATE files SET fetched=1, parse_ok=?, parse_error=? "
                "WHERE env_key=? AND path=?",
                (1 if scan.parse_ok else 0, scan.parse_error, e["env_key"], r["path"]))

            if scan.reward_funcs:
                env_reward = True
            if scan.code_grader_funcs:
                env_code = True
            if scan.opaque_grader_funcs:
                env_opaque = True
            for f in scan.findings:
                env_finding = True
                counters["findings"] += 1
                conn.execute(
                    """INSERT INTO findings
                         (env_key, path, rule, severity, func, lineno, snippet, why, probe)
                       VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(env_key, path, rule, lineno) DO UPDATE SET
                         severity=excluded.severity, snippet=excluded.snippet,
                         why=excluded.why, probe=excluded.probe""",
                    (e["env_key"], r["path"], f.rule, f.severity, f.func,
                     f.lineno, f.snippet, f.why, f.probe))

        # Persist the tier. These were computed in memory and discarded, which is
        # exactly why `--report-only` had to invent a `0` for the code-grader and
        # opaque denominators -- and a fabricated denominator reads as "no
        # coverage gap" when it means "never measured".
        fully_unreadable = 1 if (paths and readable == 0) else 0
        conn.execute(
            "UPDATE environments SET has_reward_func=?, is_code_grader=?, "
            "is_opaque=?, files_considered=?, files_scanned=?, files_over_cap=?, "
            "fully_unreadable=? WHERE env_key=?",
            (1 if env_reward else 0, 1 if env_code else 0, 1 if env_opaque else 0,
             len(candidates), len(paths), over_cap, fully_unreadable, e["env_key"]))

        # Scanned either way -- a red-team environment's findings are still worth
        # storing, and they are the only real-world positive controls we have.
        # They just never enter the denominator of a corpus claim.
        if e["research_subject"]:
            counters["envs_excluded_research"] += 1
        else:
            counters["envs"] += 1
            counters["envs_with_reward_func"] += 1 if env_reward else 0
            counters["envs_with_code_grader"] += 1 if env_code else 0
            counters["envs_with_opaque_grader"] += 1 if env_opaque else 0
            counters["envs_with_finding"] += 1 if env_finding else 0
            counters["envs_fully_unreadable"] += fully_unreadable
            if over_cap:
                counters["envs_file_capped"] += 1
                counters["files_over_cap"] += over_cap

        if i % 25 == 0:
            store.commit(conn)
            if verbose:
                print("  %d/%d envs | %d findings | %d files unreadable"
                      % (i, len(envs), counters["findings"],
                         counters["files_unavailable"]), flush=True)
    store.commit(conn)
    # What this run actually looked at, so the report can scope itself to it.
    counters["scanned_keys"] = [e["env_key"] for e in envs]
    keys = counters["scanned_keys"]
    counters["envs_listing_truncated"] = conn.execute(
        "SELECT COUNT(*) FROM environments WHERE files_listed=1 "
        "AND research_subject=0 AND listing_truncated=1 AND env_key IN (%s)"
        % ", ".join("?" * len(keys)), keys).fetchone()[0] if keys else 0
    counters["elapsed_s"] = round(time.monotonic() - t0, 1)
    return counters


# A rule firing above this share of the corpus is treated as broken until
# someone proves otherwise. `no_answer_marker` once matched the bare token `in`,
# hit 53% of modules, and was worthless -- the check is structural so that
# lesson does not depend on anybody remembering it.
HIT_RATE_ALARM = 0.25

# Findings live in research-subject environments too; they are just never part
# of a corpus claim. Every aggregate below carries this join for that reason.
#
# It also drops findings a human reviewed and withdrew. Those columns exist so
# that review survives a re-scan, which is worth nothing if the report then
# counts the withdrawn ones anyway -- 13 findings read against real source and
# refuted were still inflating the headline "environments with a HIGH severity
# smell". Withdrawals are subtracted from the claim and reported on their own
# line, never quietly dropped.
_WITHDRAWN = "f.review_verdict = 'false_positive'"
_NON_RESEARCH = ("FROM findings f JOIN environments e USING (env_key) "
                 "WHERE e.research_subject = 0 AND NOT COALESCE(%s, 0)"
                 % _WITHDRAWN)
_ALL_FINDINGS = ("FROM findings f JOIN environments e USING (env_key) "
                 "WHERE e.research_subject = 0")


def _num(v: Any) -> str:
    """Render a counter, distinguishing "measured zero" from "never measured".

    Printing 0 for a count nobody computed is the worst failure this report can
    have: `execution delegated, opaque : 0` reads as "no coverage gap" when it
    means "no idea". The whole tool is an argument about denominators.
    """
    return "not computed" if v is None else str(v)


def write_document(conn, out_path: str, scope_keys: Optional[List[str]] = None,
                   counters: Optional[Dict[str, Any]] = None) -> str:
    """Write the audit as a document. Format from the extension.

    HTML and JSON need nothing installed; PDF needs reportlab. A missing PDF
    library costs one output format, not the run -- so the error says how to get
    the same document out of the other renderers.
    """
    from rlverify.report.model import build_report

    rep = build_report(conn, scope_keys=scope_keys, counters=counters)
    low = out_path.lower()
    if low.endswith(".pdf"):
        from rlverify.report.render_pdf import to_pdf
        to_pdf(rep, out_path)
    elif low.endswith(".json"):
        # Every report class is a frozen dataclass of plain types, so this is a
        # faithful dump rather than a second rendering with its own drift.
        import json
        from dataclasses import asdict
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(rep), fh, indent=2, sort_keys=True)
    else:
        from rlverify.report.render_html import to_html
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(to_html(rep))
    return out_path


def _scope(base: str, keys: Optional[List[str]]) -> str:
    """Restrict a findings aggregate to a set of environments.

    Without this the report is corpus-scoped no matter what was scanned, so
    auditing a single environment printed 384 findings across 211 other
    people's environments. For a corpus claim that is right; for the per-target
    audit that a customer actually receives it is unusable.
    """
    if keys is None:
        return base
    if not keys:
        # Explicitly nothing. Returning `base` here made a scoped report quote
        # corpus-wide totals under a single target's heading.
        return base + " AND 0"
    quoted = ", ".join("'%s'" % k.replace("'", "''") for k in keys)
    return base + " AND f.env_key IN (%s)" % quoted


def readonly_counters(conn) -> Dict[str, Any]:
    """Counters for a report over a store nobody is writing to.

    Readonly, so this can run *while* a sweep is in progress. A reader that
    insists on DDL blocks on the writer's lock; that is what killed the sync at
    750. It also means no research-subject backfill -- the report quotes the
    classification as of the last scan.
    """
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
    # "Scanned" means the static pass actually visited it: the tier columns stay
    # NULL until then. Counting never-scanned environments in the denominator
    # would understate every rate on this page, and reporting a tier we never
    # computed as `0` is the failure this page exists to avoid. A readonly reader
    # cannot migrate, so a store the writer has not upgraded yet simply has not
    # computed these. Say that, do not print 0.
    migrated = store.has_column(conn, "environments", "has_reward_func")
    seen = ("FROM environments WHERE files_listed=1 AND research_subject=0 "
            "AND has_reward_func IS NOT NULL")
    n_seen = q("SELECT COUNT(*) " + seen) if migrated else 0
    tier = ((lambda col: q("SELECT COUNT(*) %s AND %s=1" % (seen, col)))
            if n_seen else (lambda col: None))
    return {
        "envs": n_seen or None,
        "envs_with_reward_func": tier("has_reward_func"),
        "envs_with_code_grader": tier("is_code_grader"),
        "envs_with_opaque_grader": tier("is_opaque"),
        "envs_fully_unreadable": tier("fully_unreadable"),
        "envs_file_capped": (
            q("SELECT COUNT(*) %s AND files_over_cap>0" % seen) if n_seen else None),
        "files_over_cap": (
            q("SELECT COALESCE(SUM(files_over_cap),0) %s" % seen) if n_seen else None),
        "envs_listing_truncated": (q(
            "SELECT COUNT(*) FROM environments WHERE files_listed=1 "
            "AND research_subject=0 AND listing_truncated=1")
            if store.has_column(conn, "environments", "listing_truncated") else None),
        "envs_with_finding": q(
            "SELECT COUNT(DISTINCT f.env_key) FROM findings f "
            "JOIN environments e USING (env_key) WHERE e.research_subject=0"),
        "envs_excluded_research": q(
            "SELECT COUNT(*) FROM environments "
            "WHERE files_listed=1 AND research_subject=1"),
        "files_parsed": q("SELECT COUNT(*) FROM files WHERE parse_ok=1"),
        "files_parse_failed": q("SELECT COUNT(*) FROM files WHERE parse_ok=0"),
        "files_unavailable": q(
            "SELECT COUNT(*) FROM files WHERE parse_error='unavailable'"),
    }


def report(conn, counters: Dict[str, Any],
           scope_keys: Optional[List[str]] = None) -> None:
    n = counters["envs"]
    rf = counters.get("envs_with_reward_func")
    print("\n" + "=" * 62)
    print("CORPUS STATIC SCAN")
    print("=" * 62)
    print("environments scanned          : %s" % _num(n))
    print("  with a detected reward func : %s" % _num(rf))
    print("  that execute response code  : %s   (denominator for the code rules)"
          % _num(counters.get("envs_with_code_grader")))
    print("  execution delegated, opaque : %s   (coverage gap, NOT a clean result)"
          % _num(counters.get("envs_with_opaque_grader")))
    print("  with >=1 static finding     : %s" % _num(counters["envs_with_finding"]))
    if rf:
        print("  finding rate (of those)     : %.1f%%"
              % (100.0 * counters["envs_with_finding"] / rf))
    else:
        print("  finding rate (of those)     : not computed (no denominator)")
    excluded = counters.get("envs_excluded_research", 0)
    print("  excluded, research subject  : %d   (scanned, kept out of the "
          "denominator)" % excluded)
    if excluded:
        # Named, not just counted. An exclusion list nobody can read is a
        # denominator nobody can check, and this list is the first thing a
        # reviewer of any published number should argue with.
        names = [r[0] for r in conn.execute(
            "SELECT name FROM environments WHERE research_subject=1 AND "
            "files_listed=1 ORDER BY name")]
        print("    " + textwrap.fill(", ".join(names), 74,
                                     subsequent_indent="    "))
    print("\ncoverage (a gap is not a clean result)")
    print("  files parsed                : %s" % _num(counters["files_parsed"]))
    print("  files failed to parse       : %s" % _num(counters["files_parse_failed"]))
    print("  files unreadable            : %s" % _num(counters["files_unavailable"]))
    print("  envs fully unreadable       : %s" % _num(counters["envs_fully_unreadable"]))
    print("  envs hitting the file cap   : %s   (scanned in part, not in full)"
          % _num(counters.get("envs_file_capped")))
    print("  files dropped by that cap   : %s" % _num(counters.get("files_over_cap")))
    print("  envs with a truncated listing: %s  (Hub walk hit its ceiling)"
          % _num(counters.get("envs_listing_truncated")))

    print("\nfindings by rule (research-subject environments excluded)")
    for row in conn.execute(
        "SELECT f.rule rule, f.severity severity, COUNT(*) c, "
        "COUNT(DISTINCT f.env_key) e " + _scope(_NON_RESEARCH, scope_keys) +
        " GROUP BY f.rule, f.severity ORDER BY c DESC"
    ):
        print("  %-24s %-7s %5d hits across %4d envs"
              % (row["rule"], row["severity"], row["c"], row["e"]))

    # Denominator taken from the store rather than the run, so a --report-only
    # pass and a live scan quote the same rate.
    scanned = conn.execute(
        "SELECT COUNT(DISTINCT f.env_key) FROM files f "
        "JOIN environments e USING (env_key) "
        "WHERE f.parse_ok = 1 AND e.research_subject = 0"
    ).fetchone()[0]
    # When scoped to a target, the rate stays corpus-wide on purpose and says so.
    # A scoped numerator over a corpus denominator is meaningless, and a scoped
    # denominator makes every rule 100% on a one-environment audit. What a target
    # actually wants to know is whether its finding is common or unusual, which
    # is the corpus rate -- restricted to the rules that fired on it.
    # `is not None`, not truthiness: an explicitly empty scope_keys (a --limit
    # run that matched nothing) must still take this branch, so `_scope` turns
    # it into a query that matches no rule and prints an empty table -- not the
    # `else` branch's corpus-wide framing, which `if scope_keys:` fell back to.
    if scope_keys is not None:
        in_scope = [r[0] for r in conn.execute(
            "SELECT DISTINCT f.rule " + _scope(_NON_RESEARCH, scope_keys))]
        print("\nhow common these rules are corpus-wide "
              "(of %d environments with >=1 parsed file)" % scanned)
    else:
        in_scope = None
        print("\nper-rule hit rate (of %d environments with >=1 parsed file)" % scanned)
    alarms = []
    for row in conn.execute(
        "SELECT f.rule rule, COUNT(DISTINCT f.env_key) e " + _NON_RESEARCH +
        " GROUP BY f.rule ORDER BY e DESC"
    ):
        if in_scope is not None and row["rule"] not in in_scope:
            continue
        rate = (row["e"] / scanned) if scanned else 0.0
        flag = ""
        if rate > HIT_RATE_ALARM:
            flag = "  <-- CHECK: a rule this broad is a bug until proven otherwise"
            alarms.append(row["rule"])
        print("  %-24s %5.1f%%%s" % (row["rule"], 100.0 * rate, flag))
    if alarms:
        print("\n  %d rule(s) above %.0f%%: %s" % (len(alarms), 100 * HIT_RATE_ALARM,
                                                   ", ".join(alarms)))
        print("  Tighten them before any of this is quoted anywhere.")

    # Manual review, stated rather than assumed. A reader has to be able to see
    # how much of the claim has actually been checked and how much was thrown out.
    q1 = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
    reviewed = q1("SELECT COUNT(*) " + _scope(_ALL_FINDINGS, scope_keys) + " AND f.reviewed = 1")
    if reviewed:
        print("\nmanual review (verdicts survive a re-scan)")
        for verdict in ("confirmed", "false_positive", "unclear"):
            n = q1("SELECT COUNT(*) %s AND f.review_verdict = '%s'"
                   % (_scope(_ALL_FINDINGS, scope_keys), verdict))
            print("  %-27s : %d" % (verdict, n))
        print("  unreviewed                  : %d"
              % q1("SELECT COUNT(*) " + _scope(_ALL_FINDINGS, scope_keys) + " AND f.reviewed = 0"))
        print("  Withdrawn findings are excluded from every count above and from")
        print("  the rule tables; they are subtracted from the claim, not hidden.")

    high = conn.execute(
        "SELECT COUNT(DISTINCT f.env_key) " + _scope(_NON_RESEARCH, scope_keys) +
        " AND f.severity = 'high'").fetchone()[0]
    high_conf = conn.execute(
        "SELECT COUNT(DISTINCT f.env_key) " + _scope(_ALL_FINDINGS, scope_keys) +
        " AND f.severity = 'high' AND f.review_verdict = 'confirmed'").fetchone()[0]
    print("\nenvironments with a HIGH severity smell: %d" % high)
    print("  of those, confirmed on manual review : %d" % high_conf)
    print("\nNothing above is a confirmed exploit. These are smells that rank")
    print("which environments deserve the dynamic pass, and every one needs")
    print("manual review before it appears in anything public.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Static-scan the synced corpus.")
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--cache", default="~/.cache/rlverify/hub")
    p.add_argument("--rate-hz", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only", default=None,
                   help="restrict to environments whose name matches this SQL "
                        "LIKE pattern; use after a rule change to re-judge only "
                        "the environments that rule fired on")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--out", default=None,
                   help="also write the audit document here; format is taken "
                        "from the extension (.pdf or .html)")
    args = p.parse_args(argv)

    if args.report_only:
        conn = store.connect(args.db, readonly=True)
        report(conn, readonly_counters(conn))
        if args.out:
            print("\nwrote %s" % write_document(conn, args.out, scope_keys=None))
        conn.close()
        return 0

    conn = store.connect(args.db)
    # Classify before scanning so the exclusion can never lag the corpus.
    store.backfill_research_subject(conn)
    client = HubClient(cache_dir=args.cache, rate_hz=args.rate_hz)
    counters = scan_corpus(conn, client, limit=args.limit, only=args.only)
    # Scope the report to what was scanned whenever this was a targeted run.
    # A full sweep stays corpus-scoped, which is what a corpus claim needs.
    # `.get(...)` returning [] must stay [], not become None. A targeted run that
    # matched nothing is an empty scope, not an absent one.
    scope = counters.get("scanned_keys", []) if (args.only or args.limit) else None
    if args.only and not scope:
        print("\nno environment matched --only %r; nothing was scanned" % args.only)
        conn.close()
        return 2
    report(conn, counters, scope_keys=scope)
    if args.out:
        print("\nwrote %s" % write_document(
            conn, args.out, scope_keys=scope, counters=counters))
    print("\nelapsed %.1fs | http failures %d" % (counters["elapsed_s"], len(client.failures)))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
