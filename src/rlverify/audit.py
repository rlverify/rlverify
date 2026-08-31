"""
Audit one environment: resolve it, make sure it is synced, scan it, judge it.

The judgement is deliberately separated from the pipeline. `exit_code` is a pure
function over an `AuditVerdict`, so the property this whole module exists to
guarantee -- that an environment we did not actually scan can never exit 0 -- is
gated exhaustively without a network or a store.

Four outcomes, and the third is the one that matters:

    0  clean         scanned, coverage complete, nothing at or above the threshold
    1  findings      >=1 finding at or above --fail-on, withdrawals excluded
    2  error         could not resolve, could not fetch, could not scan
    3  inconclusive  resolved and scanned, but coverage too poor for a claim

Collapsing 3 into 0 would make this tool report a clean grader whenever it failed
to read one -- the failure mode CLAUDE.md names as the most dangerous possible in
an auditing tool, and the one this project accuses other people's graders of.
"Coverage too poor for a claim" is not limited to the whole-environment gaps in
INCONCLUSIVE_REASONS below -- a file that failed to parse or failed to download
is the identical gap at file granularity, and must reach exit 3 by the same path.

Interpreter: 3.9. Imports nothing from `rlverify.targets` -- that pulls
`verifiers`, which needs >=3.11, and would break `pip install rlverify` on the
system Python.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2
EXIT_INCONCLUSIVE = 3

FAIL_ON_CHOICES = ("low", "medium", "high", "never")

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# The rank given to a severity string this module does not recognise, for the
# --fail-on threshold comparison in exit_code specifically (NOT for worst()'s
# display ranking -- see the comment on worst() for why that is different). It
# MUST sit strictly above every known severity: a label we cannot classify (a
# typo, a stale probe, a future severity this code has not been taught yet) is
# not evidence of harmlessness, and the tool's bias must always run against a
# false clean. Derived from max(SEVERITY_ORDER.values()) + 1 rather than
# len(SEVERITY_ORDER) -- len() is only "above everything" while ranks are
# contiguous from 0; a non-contiguous addition such as {"critical": 10} would
# make len() rank below it silently, and this whole module exists to make that
# kind of silent failure impossible.
_UNKNOWN_RANK = max(SEVERITY_ORDER.values()) + 1

# Why an environment cannot be cleared, keyed by the column that says so. Each is
# a coverage gap: something we could not see, never something we saw and liked.
#
# Each entry is (column, flag, message, unmeasured_message). A row satisfies the
# ordinary reason when bool(row[column]) == flag -- that comparison MUST go
# through bool(), never a bare `row[column] == flag`. On an all-NULL SQLite row,
# row[column] is None: `None == False` is False (the reason would not fire), but
# `bool(None) == False` is True (it does).
#
# unmeasured_message is the stronger case bool()-alone cannot express: for a
# column where a NULL can occur on an otherwise-fully-scanned environment, NULL
# does not mean "the condition is false" -- it means the column was never
# measured, and absence of a measurement is ITSELF a coverage gap, not evidence
# of anything. Checked against the real corpus: `listing_truncated` is NULL for
# 1424 of 1460 file-listed environments, and for 1418 of the ~1449 scanned ones
# -- those rows were file-indexed before the column existed, before truncation
# was ever recorded. Left to the ordinary rule, `bool(None) == True` is False,
# the reason would not fire, and 1418 environments could exit 0 CLEAN with their
# listing coverage entirely unknown -- exactly the false-clean shape this module
# exists to prevent. unmeasured_message fixes that: a NULL listing_truncated
# fires its own reason instead of silently passing as "not truncated".
#
# `has_reward_func`, `fully_unreadable` and `is_opaque` all set unmeasured_message
# to None: checked against the same corpus, those three are NULL on exactly the
# 11 rows where `has_reward_func` is also NULL, and a NULL `has_reward_func`
# already routes to scanned=False -> exit 2 (see exit_code). Firing a second
# reason there on top of an already-certain 2 would be noise, not a missed gap
# -- unlike listing_truncated, these three have no case where the environment
# was otherwise fully scanned and the column is still NULL.
INCONCLUSIVE_REASONS = (
    ("has_reward_func", False,
     "no reward function located -- we cannot clear a grader we did not find",
     None),
    ("fully_unreadable", True,
     "no file in this environment could be read",
     None),
    ("is_opaque", True,
     "grading is delegated to a container or remote sandbox; the code that "
     "decides reward is not in the Python source",
     None),
    ("listing_truncated", True,
     "the Hub file listing hit its ceiling, so the file set is known-incomplete",
     "the Hub file listing for this environment was indexed before truncation "
     "was recorded, so whether the file set is complete has never been measured"),
)

# Two more coverage gaps INCONCLUSIVE_REASONS cannot express: it is a table of
# whole-environment (column, flag) checks, but "how many files failed to parse"
# and "how many files could not be downloaded" are COUNTS from the files table,
# not environment-level booleans. build_verdict (Task 5) MUST append
# FILES_UNPARSED_REASON % n to inconclusive_reasons whenever that count is > 0,
# and FILES_UNAVAILABLE_REASON % n whenever that count is > 0. This aligns
# audit.py with report/model.py's own coverage-completeness definition
# (not (files_over_cap or listing_truncated or files_parse_failed or
# files_unavailable)) -- that module already treats both as coverage gaps, and
# this tool's exit code must not contradict the document it renders from the
# same audit. A file that failed to parse is a file no rule ran on: the
# environment's grading logic could live in exactly that file, and this tool
# has no evidence either way about it -- that is inconclusive, not clean, no
# matter how many other files came back clean.
FILES_UNPARSED_REASON = "%d file(s) in this environment failed to parse -- no rule ran on them"
FILES_UNAVAILABLE_REASON = "%d file(s) in this environment could not be downloaded"

# A third coverage gap FILES_UNPARSED_REASON/FILES_UNAVAILABLE_REASON cannot
# express: `environments.fetch_error` records that the file-LISTING pass
# itself (not an individual file's parse or download) did not complete --
# "walk failed" or "walk incomplete: ..." (see corpus/sync.py). build_verdict
# MUST append this reason whenever the column is non-NULL. This is not
# redundant with the `listing_truncated IS NULL` reason above: a NULL
# listing_truncated survives ANY failed walk (fix round 1, Critical A), but an
# environment can already carry a real prior measurement (listing_truncated=0
# or 1 from an earlier, successful walk) that a LATER failed re-walk leaves
# untouched -- in that case listing_truncated alone would stay silent about
# the fact that the most recent attempt failed, and fetch_error is the only
# column that still says so. Ignoring it is exactly how a real defect reached
# production: six rows in the live store carry fetch_error='walk failed'
# today; none happen to be scanned yet, so nothing published was affected, but
# the path was live and would have exited 0 CLEAN the moment one was.
FILES_LISTING_ERROR_REASON = (
    "the file listing for this environment did not complete cleanly (%s) -- "
    "file coverage cannot be confirmed complete")


def reasons_from_row(row: Dict[str, Any]) -> List[str]:
    """Evaluate INCONCLUSIVE_REASONS against one environment row.

    `row` is a plain dict keyed by column name (use dict(sqlite_row) for a
    sqlite3.Row). Returns the message text for every reason whose column
    satisfies its trigger. Centralised here, rather than left for every future
    caller to reimplement, specifically so both rules in the comment on
    INCONCLUSIVE_REASONS -- the bool()-comparison rule, and the NULL-means-
    unmeasured rule beside it -- are enforced in exactly this one place.

    For a column whose entry carries an unmeasured_message, `row[column] is
    None` fires THAT message instead of the ordinary bool(value) == flag
    check -- a NULL there means "never measured", not "condition is false".
    For a column with unmeasured_message=None, the ordinary
    bool(row.get(column)) == flag rule applies even when the value is None,
    unchanged from before.
    """
    fired = []
    for column, flag, message, unmeasured_message in INCONCLUSIVE_REASONS:
        value = row.get(column)
        if value is None and unmeasured_message is not None:
            fired.append(unmeasured_message)
        elif bool(value) == flag:
            fired.append(message)
    return fired


@dataclass(frozen=True)
class AuditVerdict:
    """What we can honestly say about one environment.

    `severity_counts` holds live findings only -- a finding a human reviewed and
    withdrew has been subtracted from the claim, not hidden.
    """

    env_key: Optional[str]
    scanned: bool
    error: Optional[str] = None
    severity_counts: Dict[str, int] = field(default_factory=dict)
    inconclusive_reasons: List[str] = field(default_factory=list)
    research_subject: bool = False
    # Context for the human-readable summary; never part of the verdict itself.
    owner: Optional[str] = None
    name: Optional[str] = None
    files_scanned: Optional[int] = None
    files_considered: Optional[int] = None

    def __post_init__(self) -> None:
        # Defensive, not narrowing: a caller building this from a raw SQLite
        # row (Task 5, most likely) that hands through the wrong Python type
        # must fail loudly and immediately, never silently misjudge an
        # environment. `scanned='0'` is a non-empty string -- truthy in
        # Python -- so `not verdict.scanned` would read that STRING "false" as
        # "we did look", exactly the false-clean shape this module exists to
        # prevent. `severity_counts=None` would instead crash deep inside
        # n_findings/exit_code with an opaque AttributeError that exits the
        # process 1 -- reading as a findings claim rather than the caller bug
        # it actually is.
        if not isinstance(self.scanned, bool):
            raise TypeError("scanned must be a bool, got %r" % (self.scanned,))
        if not isinstance(self.severity_counts, dict):
            raise TypeError(
                "severity_counts must be a dict, got %r" % (self.severity_counts,))
        if not isinstance(self.inconclusive_reasons, list):
            raise TypeError(
                "inconclusive_reasons must be a list, got %r" % (self.inconclusive_reasons,))

    @property
    def n_findings(self) -> int:
        return sum(self.severity_counts.values())

    @property
    def has_unrecognised_severity(self) -> bool:
        """True if a live finding carries a severity this module cannot
        classify (a typo, a stale probe, a NULL severity column reaching here
        as None, a future severity not yet added to SEVERITY_ORDER).

        worst() will not surface these -- see its docstring -- so a caller
        that needs to know they exist (to avoid rendering "no findings" for a
        verdict exit_code calls EXIT_FINDINGS) must check this separately.
        """
        return any(n and s not in SEVERITY_ORDER for s, n in self.severity_counts.items())

    def worst(self) -> Optional[str]:
        """The worst KNOWN severity among live findings, or None.

        Deliberately does NOT rank an unrecognised severity above every known
        one the way exit_code's threshold comparison does (_UNKNOWN_RANK):
        that bias is correct for "should this fail the build" and wrong for
        "what is the headline severity to print" -- there is no name to
        display for a label this module cannot classify. Applying the same
        rule here previously made {"informational": 1, "high": 1} report
        "informational" as worse than "high", and let a NULL severity's key
        collide with this method's own "no findings" None sentinel while
        exit_code, on the identical verdict, still called it EXIT_FINDINGS.
        See `has_unrecognised_severity` for the signal this method
        deliberately does not fold in -- a caller needs both, not one
        standing in for the other.
        """
        live_known = [s for s, n in self.severity_counts.items()
                      if n and s in SEVERITY_ORDER]
        if not live_known:
            return None
        return max(live_known, key=lambda s: SEVERITY_ORDER[s])


def exit_code(verdict: AuditVerdict, fail_on: str = "high") -> int:
    """Map a verdict to a process exit code. Pure; see the module docstring.

    Precedence is 2 > 1 > 3 > 0. Findings outrank inconclusive deliberately: a
    concrete defect is more actionable than a gap, and neither is a clean result.
    `--fail-on never` suppresses the 1 only -- a coverage gap is not something a
    caller gets to opt out of being told about.

    A severity this module does not recognise ranks above "high" for this
    threshold comparison, never below -- see `_UNKNOWN_RANK`.

    `fail_on` is validated AFTER the error/scanned checks, deliberately: those
    two are unconditional exits, and a caller passing a bad fail_on must never
    downgrade an otherwise-certain 2 into an uncaught ValueError -- that would
    exit the process 1, the exact "reads as a findings claim" harm this
    docstring already names as the thing to avoid. Only once the verdict
    itself would actually need the threshold comparison do we require fail_on
    to be one of FAIL_ON_CHOICES, raising ValueError (never a bare KeyError)
    if it is not.
    """
    if verdict.error is not None:
        return EXIT_ERROR
    if not verdict.scanned:
        # We never looked. That is an error about this run, not a verdict about
        # the environment, and it is never 0.
        return EXIT_ERROR
    if fail_on not in FAIL_ON_CHOICES:
        raise ValueError(
            "fail_on must be one of %s, got %r" % (FAIL_ON_CHOICES, fail_on))
    if fail_on != "never":
        threshold = SEVERITY_ORDER[fail_on]
        for sev, n in verdict.severity_counts.items():
            if n and SEVERITY_ORDER.get(sev, _UNKNOWN_RANK) >= threshold:
                return EXIT_FINDINGS
    if verdict.inconclusive_reasons:
        return EXIT_INCONCLUSIVE
    return EXIT_CLEAN


def _version_sort_key(version: Optional[str]) -> Tuple[Tuple[Any, ...], str]:
    """Numeric-aware version key: '0.9.0' must sort before '0.10.0'.

    Minor H. A plain string compare puts '0.10.0' before '0.9.0' (`'1' < '9'`
    lexicographically), so `resolve`'s newest-version tiebreak could pick a
    STALE version over a newer one and audit the wrong code under the right
    env_key. Splits on the usual version separators and treats each purely
    numeric segment as an int; a non-numeric segment (a pre-release tag, a
    local build suffix, anything unusual) falls back to string comparison for
    just that segment, tagged so it never collides with a numeric one of a
    different type -- an unusual version string still sorts consistently
    rather than raising, even if the ordering isn't necessarily meaningful.
    The raw string is kept as a final tiebreak so two rows are never "equal"
    here purely because their key derivation happened to coincide.
    """
    if not version:
        return (), ""
    key = tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"[.\-+]", version)
    )
    return key, version


def resolve(conn, spec: str) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve `owner/name`, `owner/name@version`, or a bare `name` to exactly
    one environment row.

    Returns `(row, None)` on success, `(None, None)` when the store simply does
    not have it -- the caller may sync and retry -- and `(None, message)` when
    the spec is genuinely unusable (empty, or missing an owner/name half of a
    `/`). Ambiguity is an error rather than a guess: two owners publish
    `shared-name`, and auditing the wrong one and telling somebody their grader
    is broken is unrecoverable.

    `owner/name@version` -- the exact `env_key` format this tool itself prints
    (`"auditing %s" % key`) and files every finding under -- MUST resolve
    directly rather than round-tripping through owner+name matching, which
    would silently swallow the `@version` suffix into the `name` half of the
    `/` split and never match anything (fix round 1, Important E: this was a
    real bug -- `resolve(conn, "owner/some-env@0.1.0")`, exactly the
    string a prior audit run had just printed, came back `(None, None)`,
    "absent, try syncing" -- for an environment already in the store).
    """
    spec = spec.strip()
    if not spec:
        return None, "empty environment name"

    version: Optional[str] = None
    body = spec
    if "@" in spec:
        body, _, version = spec.partition("@")

    if "/" in body:
        owner, _, name = body.partition("/")
        if not owner or not name:
            return None, (
                "malformed spec %r -- expected owner/name or owner/name@version" % spec)
        if version is not None:
            # The tool's own env_key format: look it up directly by primary
            # key rather than re-deriving owner/name and risking a second,
            # possibly-inconsistent interpretation of the same string.
            row = conn.execute(
                "SELECT * FROM environments WHERE env_key=?", (spec,)).fetchone()
            if row is not None:
                return row, None
            # That exact version is not (or not yet) in the store; owner+name
            # still narrows enough to answer honestly below, e.g. after a Hub
            # release the store has not synced yet.
        rows = conn.execute(
            "SELECT * FROM environments WHERE owner=? AND name=?",
            (owner, name)).fetchall()
    else:
        if version is not None:
            return None, (
                "malformed spec %r -- a version needs an owner, e.g. owner/%s@%s"
                % (spec, body, version))
        rows = conn.execute(
            "SELECT * FROM environments WHERE name=?", (spec,)).fetchall()
    if not rows:
        return None, None
    owners = sorted({r["owner"] for r in rows})
    if len(owners) > 1:
        return None, ("%r is published by %d owners (%s) -- pass owner/name"
                      % (spec, len(owners), ", ".join(owners)))
    # One owner, possibly several versions. Newest wins; that is the one the Hub
    # serves and the one anybody training today would install. `updated_at`
    # breaks ties first (a real timestamp beats a version-string guess), then
    # the version itself via the numeric-aware key above (Minor H).
    rows = sorted(rows, key=lambda r: ((r["updated_at"] or ""), _version_sort_key(r["version"])))
    return rows[-1], None


def build_verdict(conn, env_key: str) -> AuditVerdict:
    """Read the store's judgement of one environment. No network, no scanning."""
    e = conn.execute(
        "SELECT * FROM environments WHERE env_key=?", (env_key,)).fetchone()
    if e is None:
        return AuditVerdict(env_key=env_key, scanned=False,
                            error="%s is not in the store" % env_key)

    # NULL tier columns mean the static pass never visited it. Reporting that as
    # a clean environment is the whole failure this module guards against.
    scanned = e["has_reward_func"] is not None

    counts: Dict[str, int] = {}
    for row in conn.execute(
        "SELECT severity, COUNT(*) c FROM findings WHERE env_key=? "
        "AND NOT COALESCE(review_verdict = 'false_positive', 0) "
        "GROUP BY severity", (env_key,)
    ):
        counts[row["severity"]] = row["c"]

    reasons: List[str] = []
    if scanned:
        # Never reimplement this loop. `reasons_from_row` owns the NULL rule --
        # a NULL column is an absent measurement, and for `listing_truncated`
        # absence is itself a coverage gap with its own message. Unpacking
        # INCONCLUSIVE_REASONS by hand here is how that rule gets silently
        # dropped; the entries are 4-tuples for exactly that reason.
        reasons.extend(reasons_from_row(dict(e)))

        over = e["files_over_cap"] or 0
        if over:
            reasons.append(
                "scanned in part: %d of %d candidate files were over the "
                "per-environment budget" % (over, e["files_considered"] or 0))

        # Partial coverage loss, which no environment-level column records.
        # `fully_unreadable` is set only when NOTHING was readable, so an
        # environment where 9 of 10 files failed would otherwise read as clean --
        # and a file that failed to parse had zero rules run against it, which
        # `report/model.py` already counts as incomplete coverage. The exit code
        # and the document this tool renders from the same audit must not
        # disagree about whether we saw the whole environment.
        n_unparsed = conn.execute(
            "SELECT COUNT(*) FROM files WHERE env_key=? AND parse_ok=0",
            (env_key,)).fetchone()[0]
        if n_unparsed:
            reasons.append(FILES_UNPARSED_REASON % n_unparsed)
        n_unavailable = conn.execute(
            "SELECT COUNT(*) FROM files WHERE env_key=? AND parse_error='unavailable'",
            (env_key,)).fetchone()[0]
        if n_unavailable:
            reasons.append(FILES_UNAVAILABLE_REASON % n_unavailable)

        # The listing pass itself failed to complete at some point (fix round
        # 1, Important D). Independent of `listing_truncated`: that column can
        # still read as a real prior measurement (0 or 1) if a LATER re-walk
        # failed and left it untouched, in which case fetch_error is the only
        # column that still says the most recent attempt did not succeed.
        if e["fetch_error"]:
            reasons.append(FILES_LISTING_ERROR_REASON % e["fetch_error"])

    return AuditVerdict(
        env_key=env_key,
        scanned=bool(scanned),
        severity_counts=counts,
        inconclusive_reasons=reasons,
        research_subject=bool(e["research_subject"]),
        owner=e["owner"],
        name=e["name"],
        files_scanned=e["files_scanned"],
        files_considered=e["files_considered"],
    )


def audit_environment(conn, client, spec: str, verbose: bool = True) -> AuditVerdict:
    """Resolve, sync if needed, scan, and judge one environment.

    Syncs on demand so this works against an empty store: `pip install rlverify
    && rlverify audit owner/name` has to do something useful without an hour of
    corpus sweep first. The catalog pass is the resolution mechanism -- the Hub
    has no single-environment lookup, and the per-version detail endpoint needs a
    version only the catalog supplies.
    """
    from rlverify.corpus import scan as scan_mod
    from rlverify.corpus import sync as sync_mod

    row, err = resolve(conn, spec)
    if err:
        return AuditVerdict(env_key=None, scanned=False, error=err)

    if row is None:
        if verbose:
            print("%s is not in the local store; syncing the Hub catalog "
                  "(~15s)..." % spec)
        before = len(client.failures)
        # Important F: an HTTP 200 that paginates short is not something
        # `client.failures` can see -- `sync.main()` guards exactly this case
        # with a loud WARNING on the corpus-wide sweep, and a single-
        # environment audit must refuse the identical silent gap rather than
        # claim "not published" from a catalog it knows is incomplete.
        remote_total = client.total_count()
        n = sync_mod.sync_catalog(conn, client, verbose=verbose)
        if len(client.failures) > before:
            # "We could not check" and "it is not there" are different answers
            # and must never be collapsed into one message.
            return AuditVerdict(
                env_key=None, scanned=False,
                error="could not reach the Hub (%d request(s) failed); "
                      "cannot tell whether %s exists"
                      % (len(client.failures) - before, spec))
        if remote_total is not None and n < remote_total:
            return AuditVerdict(
                env_key=None, scanned=False,
                error="the Hub catalog only returned %d of %d known "
                      "environments; cannot confirm whether %s is published"
                      % (n, remote_total, spec))
        row, err = resolve(conn, spec)
        if err:
            return AuditVerdict(env_key=None, scanned=False, error=err)
        if row is None:
            return AuditVerdict(
                env_key=None, scanned=False,
                error="%s is not published on the Hub" % spec)

    key = row["env_key"]
    if verbose:
        print("auditing %s" % key)

    if not row["detail_ok"]:
        sync_mod.sync_details(conn, client, verbose=verbose, env_key=key)
    if not row["files_listed"]:
        sync_mod.sync_files(conn, client, verbose=verbose, env_key=key)
    elif row["listing_truncated"] is None:
        # Never measured, which is not the same as "not truncated". 1418 of the
        # ~1449 scanned environments were file-indexed before this column
        # existed, so trusting the NULL would clear an environment whose file
        # set might be incomplete. One environment is a bounded walk, so
        # measure it rather than assume; the store keeps the answer for next
        # time. If the re-walk cannot complete, `store.record_files` leaves
        # `listing_truncated` (and the file counts) untouched -- still NULL --
        # rather than writing a coverage claim it did not earn, so this branch
        # fires again on the next audit instead of quietly giving up. That is
        # what makes the failure mode inconclusive here, never clean: this
        # function does not itself decide the verdict, `build_verdict` does,
        # by reading whatever `listing_truncated` and `fetch_error` actually
        # ended up as -- never by trusting that this branch succeeded.
        if verbose:
            print("listing coverage was never measured for %s; re-walking" % key)
        sync_mod.sync_files(conn, client, verbose=verbose, env_key=key, force=True)

    n_py = conn.execute(
        "SELECT COUNT(*) FROM files WHERE env_key=? AND is_python=1", (key,)
    ).fetchone()[0]
    if not n_py:
        return AuditVerdict(
            env_key=key, scanned=False, owner=row["owner"], name=row["name"],
            error="no Python files could be listed for %s -- nothing to scan" % key)

    scan_mod.scan_corpus(conn, client, only=None, verbose=verbose, env_key=key)
    return build_verdict(conn, key)


if __name__ == "__main__":
    import itertools

    failures = 0

    def check(label: str, got: int, want: int) -> None:
        global failures
        if got != want:
            failures += 1
            print("FAIL  %-52s got %d want %d" % (label, got, want))
        else:
            print("ok    %-52s %d" % (label, got))

    def check_eq(label: str, got: Any, want: Any) -> None:
        """Like check(), but for values check()'s %d format can't print --
        strings, bools, None. Used for worst(), has_unrecognised_severity,
        n_findings sums, and the constant/invariant pins below.
        """
        global failures
        if got != want:
            failures += 1
            print("FAIL  %-52s got %r want %r" % (label, got, want))
        else:
            print("ok    %-52s %r" % (label, got))

    def check_no_raise(label: str, fn: Any, want: int) -> None:
        """Call fn() and check() its result, but turn any exception into a
        FAIL line instead of crashing the whole gate -- a caller bug must
        never take the gate itself down.
        """
        global failures
        try:
            got = fn()
        except Exception as exc:  # deliberately broad: any raise here is the failure
            failures += 1
            print("FAIL  %-52s raised %s: %s" % (label, type(exc).__name__, exc))
            return
        check(label, got, want)

    def v(**kw: Any) -> AuditVerdict:
        base = dict(env_key="o/n@1", scanned=True)
        base.update(kw)
        return AuditVerdict(**base)  # type: ignore[arg-type]

    # Pin the constants themselves. Every check below compares exit_code's
    # output to these same symbols, which proves the function is internally
    # consistent but proves NOTHING about the actual numbers a CI system
    # reads -- a symbol shuffle (e.g. EXIT_ERROR reassigned to 0) survives a
    # purely self-referential gate identically, because every comparison
    # shuffles with it. A CI system greps for a literal exit code, not a
    # Python name, so the literals must be checked directly.
    check("EXIT_CLEAN is the literal 0", EXIT_CLEAN, 0)
    check("EXIT_FINDINGS is the literal 1", EXIT_FINDINGS, 1)
    check("EXIT_ERROR is the literal 2", EXIT_ERROR, 2)
    check("EXIT_INCONCLUSIVE is the literal 3", EXIT_INCONCLUSIVE, 3)

    check("clean", exit_code(v()), EXIT_CLEAN)
    check("high, default threshold",
          exit_code(v(severity_counts={"high": 1})), EXIT_FINDINGS)
    check("medium, default threshold",
          exit_code(v(severity_counts={"medium": 3})), EXIT_CLEAN)
    check("medium, --fail-on medium",
          exit_code(v(severity_counts={"medium": 3}), "medium"), EXIT_FINDINGS)
    check("low, --fail-on low",
          exit_code(v(severity_counts={"low": 1}), "low"), EXIT_FINDINGS)
    check("opaque, no findings",
          exit_code(v(inconclusive_reasons=["opaque"])), EXIT_INCONCLUSIVE)
    check("high + opaque -- a defect outranks a gap",
          exit_code(v(severity_counts={"high": 1},
                      inconclusive_reasons=["opaque"])), EXIT_FINDINGS)
    check("--fail-on never suppresses 1, not 3",
          exit_code(v(severity_counts={"high": 9},
                      inconclusive_reasons=["opaque"]), "never"), EXIT_INCONCLUSIVE)
    check("--fail-on never, clean",
          exit_code(v(severity_counts={"high": 9}), "never"), EXIT_CLEAN)
    check("error outranks everything",
          exit_code(v(error="not on the Hub", severity_counts={"high": 1})), EXIT_ERROR)
    check("error='' still counts as an error (is not None, not truthiness)",
          exit_code(v(error="")), EXIT_ERROR)
    # research_subject does not appear anywhere in exit_code's logic; both
    # checks below exist to catch a future change that special-cases it. Kept
    # as a pair rather than one check so the property is non-vacuous: a
    # research subject's findings must produce the identical code a
    # non-research-subject's would, not merely "some code that happens to be
    # EXIT_FINDINGS".
    check("research subject's findings still count",
          exit_code(v(severity_counts={"high": 1}, research_subject=True)),
          EXIT_FINDINGS)
    check("...and count identically when research_subject is False",
          exit_code(v(severity_counts={"high": 1}, research_subject=False)),
          EXIT_FINDINGS)

    check("scanned=False alone", exit_code(v(scanned=False)), EXIT_ERROR)

    check("unrecognised severity, default threshold",
          exit_code(v(severity_counts={"critical": 1})), EXIT_FINDINGS)

    check("severity present with zero count",
          exit_code(v(severity_counts={"high": 0})), EXIT_CLEAN)
    check("empty severity_counts",
          exit_code(v(severity_counts={})), EXIT_CLEAN)

    # An invalid fail_on must never downgrade a verdict that was already
    # unconditionally EXIT_ERROR. check_no_raise turns a regression (the
    # validation moved back above the error/scanned checks) into a FAIL line
    # instead of an uncaught ValueError crashing the whole gate.
    check_no_raise(
        "invalid fail_on does not downgrade scanned=False",
        lambda: exit_code(AuditVerdict(env_key="o/n@1", scanned=False), None),
        EXIT_ERROR)
    check_no_raise(
        "invalid fail_on does not downgrade verdict.error",
        lambda: exit_code(v(error="not on the Hub"), "HIGH"),
        EXIT_ERROR)
    # ...but on a verdict that would otherwise need the comparison, an
    # invalid fail_on must still raise -- see exit_code's docstring for why
    # ValueError specifically, rather than a bare KeyError or a silent default.
    try:
        exit_code(v(), "not-a-real-choice")
    except ValueError:
        print("ok    %-52s %s" %
              ("invalid fail_on (valid verdict) raises ValueError", "ValueError"))
    else:
        failures += 1
        print("FAIL  invalid fail_on did not raise ValueError on an otherwise-valid verdict")

    # Pin _UNKNOWN_RANK's soundness directly, not only through exit_code's
    # behaviour -- a hardcoded replacement such as 2 (equal to "high", which
    # the comment on _UNKNOWN_RANK forbids) could otherwise coincidentally
    # pass the behavioural checks above depending on which severities they
    # happen to exercise.
    if _UNKNOWN_RANK <= max(SEVERITY_ORDER.values()):
        failures += 1
        print("FAIL  _UNKNOWN_RANK (%d) does not exceed every known rank (max %d)" %
              (_UNKNOWN_RANK, max(SEVERITY_ORDER.values())))
    else:
        print("ok    %-52s %d" %
              ("_UNKNOWN_RANK exceeds every known severity rank", _UNKNOWN_RANK))

    # worst() returns the worst KNOWN severity and does NOT apply the
    # unknown-ranks-high rule exit_code uses. "low" vs "high" (ranks 0 and 2)
    # is used here, not "critical"/"low" -- that pair puts the only KNOWN
    # severity at rank 0, so any non-negative default for the unknown one
    # would have tied or won regardless of whether the ranking rule was even
    # applied, making the check pass no matter what the default was.
    check_eq("worst() picks the higher of two known severities",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"low": 1, "high": 1}).worst(),
             "high")
    check_eq("worst() ignores an unrecognised severity entirely",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"informational": 1, "high": 1}).worst(),
             "high")
    # The regression this pins: when every live finding is unrecognised,
    # worst() must return None -- but a caller must not mistake that None for
    # "no findings" without also checking has_unrecognised_severity (below)
    # and exit_code, which still calls the identical verdict EXIT_FINDINGS
    # (see "unrecognised severity, default threshold" above).
    check_eq("worst() is None when only unrecognised severities are live",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"informational": 5}).worst(),
             None)
    check_eq("has_unrecognised_severity is False with only known severities",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"high": 1}).has_unrecognised_severity,
             False)
    check_eq("has_unrecognised_severity is True when an unknown one is live",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"informational": 5}).has_unrecognised_severity,
             True)
    check_eq("has_unrecognised_severity ignores a zero-count unknown severity",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"informational": 0}).has_unrecognised_severity,
             False)

    # n_findings and worst()'s empty/zero-count paths had no coverage -- a
    # stub "return 0" for n_findings, or dropping worst()'s `if n` live
    # filter, would both have survived silently.
    check_eq("n_findings on an empty severity_counts",
             AuditVerdict(env_key="o/n@1", scanned=True).n_findings, 0)
    check_eq("n_findings ignores a zero count",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"high": 0}).n_findings, 0)
    check_eq("n_findings sums real counts",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"high": 2, "low": 3}).n_findings, 5)
    check_eq("worst() on an empty severity_counts",
             AuditVerdict(env_key="o/n@1", scanned=True).worst(), None)
    check_eq("worst() drops a zero-count severity, not just non-live ones",
             AuditVerdict(env_key="o/n@1", scanned=True,
                          severity_counts={"high": 0}).worst(), None)

    # The two file-count reasons Task 5's build_verdict will append for
    # partial coverage loss must land on EXIT_INCONCLUSIVE like any other
    # inconclusive_reasons entry.
    check("files-unparsed reason is inconclusive, not clean",
          exit_code(v(inconclusive_reasons=[FILES_UNPARSED_REASON % 3])),
          EXIT_INCONCLUSIVE)
    check("files-unavailable reason is inconclusive, not clean",
          exit_code(v(inconclusive_reasons=[FILES_UNAVAILABLE_REASON % 9])),
          EXIT_INCONCLUSIVE)
    # Fix round 1, Important D: a listing pass that did not complete
    # (fetch_error non-NULL) must land on EXIT_INCONCLUSIVE too, independent
    # of whatever listing_truncated happens to read.
    check("files-listing-error reason is inconclusive, not clean",
          exit_code(v(inconclusive_reasons=[FILES_LISTING_ERROR_REASON % "walk failed"])),
          EXIT_INCONCLUSIVE)

    # build_verdict itself must actually append FILES_LISTING_ERROR_REASON
    # when fetch_error is set -- the exit_code check above only proves the
    # reason, once present, is treated as a gap; it says nothing about
    # whether build_verdict ever puts it there. A scanned row with a real
    # prior measurement (listing_truncated=0, not NULL) but a non-NULL
    # fetch_error is exactly the case the reason exists for: listing_truncated
    # alone would stay silent about the most recent failed attempt.
    import sqlite3 as _sqlite3
    _scratch = _sqlite3.connect(":memory:")
    _scratch.row_factory = _sqlite3.Row
    _scratch.execute("""
        CREATE TABLE environments (
            env_key TEXT PRIMARY KEY, owner TEXT, name TEXT,
            has_reward_func INTEGER, fully_unreadable INTEGER, is_opaque INTEGER,
            listing_truncated INTEGER, files_over_cap INTEGER, files_considered INTEGER,
            files_scanned INTEGER, research_subject INTEGER, fetch_error TEXT
        )""")
    _scratch.execute("""
        CREATE TABLE files (env_key TEXT, parse_ok INTEGER, parse_error TEXT)""")
    _scratch.execute(
        "INSERT INTO environments VALUES ('o/n@1','o','n',1,0,0,0,0,1,1,0,'walk failed')")
    _scratch.execute("CREATE TABLE findings (env_key TEXT, severity TEXT, review_verdict TEXT)")
    _scratch.commit()
    _bv = build_verdict(_scratch, "o/n@1")
    check_eq("build_verdict fires FILES_LISTING_ERROR_REASON on a non-NULL fetch_error",
             any(r.startswith("the file listing for this environment did not complete cleanly")
                 for r in _bv.inconclusive_reasons),
             True)
    check("...and that verdict exits 3, not 0, even with a real prior listing_truncated=0",
          exit_code(_bv), EXIT_INCONCLUSIVE)
    _scratch.close()

    # resolve() against a self-contained in-memory store (fix round 1,
    # Important E and Minor H). Independent of whatever the real corpus store
    # happens to contain, so these run unconditionally rather than SKIPping.
    import sqlite3 as _sqlite3
    _rconn = _sqlite3.connect(":memory:")
    _rconn.row_factory = _sqlite3.Row
    _rconn.execute(
        "CREATE TABLE environments (env_key TEXT PRIMARY KEY, owner TEXT, "
        "name TEXT, version TEXT, updated_at TEXT)")
    for _row in (
        ("a/foo@0.9.0", "a", "foo", "0.9.0", "2026-01-01"),
        # Same updated_at as the row above ON PURPOSE: only the version
        # tiebreak (Minor H) distinguishes them. A plain string compare picks
        # '0.9.0' as "newest" ('9' > '1' lexicographically) -- exactly wrong.
        ("a/foo@0.10.0", "a", "foo", "0.10.0", "2026-01-01"),
        ("b/bar@1.0.0", "b", "bar", "1.0.0", "2026-01-01"),
        ("c/shared-name@1.0.0", "c", "shared-name", "1.0.0", "2026-01-01"),
        ("d/shared-name@1.0.0", "d", "shared-name", "1.0.0", "2026-01-01"),
    ):
        _rconn.execute("INSERT INTO environments VALUES (?,?,?,?,?)", _row)
    _rconn.commit()

    def _resolve_key(spec: str) -> Optional[str]:
        row, _err = resolve(_rconn, spec)
        return row["env_key"] if row is not None else None

    def _resolve_err(spec: str) -> Optional[str]:
        return resolve(_rconn, spec)[1]

    check_eq("resolve('') is an error, not absent",
             resolve(_rconn, ""), (None, "empty environment name"))
    check_eq("resolve('/') is a malformed-spec error (empty owner AND name)",
             _resolve_err("/") is not None, True)
    check_eq("resolve('a/') is a malformed-spec error (empty name)",
             _resolve_err("a/") is not None, True)
    check_eq("resolve('/foo') is a malformed-spec error (empty owner)",
             _resolve_err("/foo") is not None, True)
    check_eq("resolve('name@version') with no owner is a malformed-spec error",
             _resolve_err("foo@0.9.0") is not None, True)
    # The core Important E regression: a full env_key -- exactly what this
    # tool prints and files findings under -- must resolve directly, not come
    # back (None, None) ("absent, try syncing") the way a naive owner/name
    # split (swallowing "@version" into the name half) used to.
    check_eq("resolve('a/foo@0.10.0') hits the exact env_key",
             (_resolve_key("a/foo@0.10.0"), _resolve_err("a/foo@0.10.0")),
             ("a/foo@0.10.0", None))
    check_eq("...and so does the older version of the same owner/name",
             _resolve_key("a/foo@0.9.0"), "a/foo@0.9.0")
    check_eq("a version not in the store falls back to owner+name (newest wins)",
             _resolve_key("a/foo@9.9.9"), "a/foo@0.10.0")
    check_eq("an owner/name that plain does not exist is absent, not an error",
             resolve(_rconn, "zzz/nope@1.0.0"), (None, None))
    check_eq("plain owner/name (no version) still resolves",
             _resolve_key("b/bar"), "b/bar@1.0.0")
    check_eq("an unambiguous bare name still resolves",
             _resolve_key("bar"), "b/bar@1.0.0")
    check_eq("an ambiguous bare name is still an error, not a guess",
             _resolve_err("shared-name") is not None, True)
    # Minor H, the direct regression: with updated_at tied, the NEWER version
    # (0.10.0) must win, not the lexicographically-larger string (0.9.0).
    check_eq("newest-version tiebreak is numeric-aware, not a string compare",
             _resolve_key("a/foo"), "a/foo@0.10.0")
    _rconn.close()

    # audit_environment against a fake HubClient, never real network (fix
    # round 1, Important F). A catalog fetch that returns HTTP 200 but
    # paginates short is invisible to client.failures -- sync.main() guards
    # exactly this on the corpus-wide sweep with a loud WARNING; a
    # single-environment audit must refuse the same silent gap rather than
    # tell somebody their environment "is not published on the Hub" from a
    # catalog it knows is incomplete.
    class _ShortCatalogClient:
        """total_count says more than iter_environments actually yields."""

        def __init__(self, total: int, yielded: int) -> None:
            self.failures: List[Dict[str, str]] = []
            self._total = total
            self._yielded = yielded

        def total_count(self) -> int:
            return self._total

        def iter_environments(self, page_size: int = 100):
            for i in range(self._yielded):
                yield {"owner": {"name": "o", "type": "user"}, "name": "env%d" % i,
                       "latest_version": "0.1.0", "id": str(i), "description": "",
                       "tags": [], "stars": 0, "visibility": "PUBLIC",
                       "latest_ci_status": None, "created_at": None, "updated_at": None}

    import tempfile as _tempfile
    from rlverify.corpus import store as _store

    _fd, _fpath = _tempfile.mkstemp(suffix=".sqlite", prefix="rlv-audit-gate-")
    import os as _os
    _os.close(_fd)
    try:
        _fconn = _store.connect(_fpath)
        _v = audit_environment(_fconn, _ShortCatalogClient(total=10, yielded=3),
                               "nobody/not-there", verbose=False)
        check_eq("a short (incomplete) catalog refuses to claim not-published",
                 _v.error is not None and "catalog only returned" in _v.error, True)
        check("...and that refusal is an error, never a pass",
              exit_code(_v), EXIT_ERROR)
        _fconn.close()

        _os.remove(_fpath)
        _fconn2 = _store.connect(_fpath)
        _v2 = audit_environment(_fconn2, _ShortCatalogClient(total=3, yielded=3),
                                "nobody/not-there", verbose=False)
        check_eq("...but a genuinely COMPLETE catalog still correctly says not-published",
                 _v2.error, "nobody/not-there is not published on the Hub")
        _fconn2.close()
    finally:
        if _os.path.exists(_fpath):
            _os.remove(_fpath)
        for _ext in ("-wal", "-shm"):
            if _os.path.exists(_fpath + _ext):
                _os.remove(_fpath + _ext)

    # THE end-to-end regression (fix round 1, Important G). An environment
    # already `files_listed=1` with `listing_truncated IS NULL` ("never
    # measured", the pre-truncation-column state 1418 real environments are
    # in) drives audit_environment's force-re-walk branch. The fake client's
    # walk returns a NON-EMPTY but INCOMPLETE listing -- some files came back,
    # but a directory read failed -- which is deliberately the harder,
    # more-realistic-than-"totally empty" case: Critical B's actual defect
    # was specifically that a non-empty-but-partial listing got recorded as
    # complete. This is genuine end-to-end wiring through the real sync.py and
    # store.py code (only HubClient itself is faked), so it also exercises
    # Critical C's `incomplete` signal as it actually flows through the
    # pipeline, not just hub_client.py's own isolated unit checks.
    class _FakeFailingWalkClient:
        """Stands in for HubClient: get_detail/read_file succeed, but the
        directory walk always reports one failed directory."""

        def __init__(self) -> None:
            self.failures: List[Dict[str, str]] = []
            self._source = '"""A harmless module."""\n\n\ndef helper(x):\n    return x + 1\n'

        def walk_files_ex(self, owner, name, version, max_entries=None, max_requests=None):
            self.failures.append(
                {"url": "fake://%s/%s@%s" % (owner, name, version),
                 "error": "simulated directory listing failure"})
            # One file came back; the walk is still incomplete.
            return ([{"path": "helper.py", "name": "helper.py", "size": 40,
                      "content_hash": "h1"}], None, True)

        def read_file(self, owner, name, version, path, content_hash=None):
            return self._source

        def get_detail(self, owner, name, version):
            return {"sha256": "fake", "wheel_url": None, "metadata": {}}

    _fd3, _fpath3 = _tempfile.mkstemp(suffix=".sqlite", prefix="rlv-audit-gate-fakewalk-")
    _os.close(_fd3)

    def _seed_and_run_fakewalk_gate(db_path: str):
        conn = _store.connect(db_path)
        conn.execute(
            "INSERT INTO environments (env_key, owner, name, version, detail_ok, "
            "files_listed, listing_truncated, has_reward_func) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("fake/env@1.0", "fake", "env", "1.0", 1, 1, None, None))
        conn.execute(
            "INSERT INTO files (env_key, path, name, size, content_hash, is_python) "
            "VALUES (?,?,?,?,?,?)",
            ("fake/env@1.0", "helper.py", "helper.py", 40, "h1", 1))
        conn.commit()
        verdict = audit_environment(conn, _FakeFailingWalkClient(), "fake/env", verbose=False)
        listing_truncated = conn.execute(
            "SELECT listing_truncated FROM environments WHERE env_key=?",
            ("fake/env@1.0",)).fetchone()[0]
        conn.close()
        return verdict, listing_truncated

    try:
        _v3, _lt3 = _seed_and_run_fakewalk_gate(_fpath3)
        check("audit_environment on a never-completing re-walk exits 3, never 0",
              exit_code(_v3), EXIT_INCONCLUSIVE)
        check_eq("...and listing_truncated is still NULL afterwards, never stamped 0/1",
                 _lt3, None)
    finally:
        for _p in (_fpath3, _fpath3 + "-wal", _fpath3 + "-shm"):
            if _os.path.exists(_p):
                _os.remove(_p)

    # INCONCLUSIVE_REASONS had zero gate coverage -- nothing below the
    # constant referenced it, so a flipped flag, a dropped entry, or a broken
    # NULL-means-unmeasured rule could not fail this gate. Expected outcomes
    # are hardcoded here per (column, value), independently of
    # INCONCLUSIVE_REASONS itself -- deriving "what should fire" from the
    # actual flag/unmeasured_message being tested would make this sweep just
    # as self-referential as the symbol-only exit-code checks were: a flipped
    # flag would agree with itself and the sweep would pass silently. (This
    # was caught and fixed during this task's own round-2 mutation testing --
    # the first version of this sweep read `flag` from the tuple under test
    # and did not catch a flipped has_reward_func flag.)
    #
    # want is "main" (the ordinary message fires), "unmeasured" (the
    # NULL-specific message fires instead), or None (nothing fires).
    _expected_firing = {
        ("has_reward_func", None): "main",     # bool(None) == False -> fires
        ("has_reward_func", 0): "main",        # bool(0) == False -> fires
        ("has_reward_func", 1): None,          # bool(1) == False is False
        ("fully_unreadable", None): None,      # bool(None) == True is False
        ("fully_unreadable", 0): None,
        ("fully_unreadable", 1): "main",
        ("is_opaque", None): None,
        ("is_opaque", 0): None,
        ("is_opaque", 1): "main",
        ("listing_truncated", None): "unmeasured",  # never measured, not "not truncated"
        ("listing_truncated", 0): None,
        ("listing_truncated", 1): "main",
    }
    _expected_columns = {"has_reward_func", "fully_unreadable", "is_opaque", "listing_truncated"}

    bad_reasons = 0
    columns_seen = set()
    for column, flag, message, unmeasured_message in INCONCLUSIVE_REASONS:
        columns_seen.add(column)
        for value in (None, 0, 1):
            want = _expected_firing.get((column, value))
            got_list = reasons_from_row({column: value})
            got_main = message in got_list
            got_unmeasured = unmeasured_message is not None and unmeasured_message in got_list
            want_main = want == "main"
            want_unmeasured = want == "unmeasured"
            if got_main != want_main or got_unmeasured != want_unmeasured:
                bad_reasons += 1
                print("FAIL  reasons_from_row(%s=%r): main=%s want=%s unmeasured=%s want=%s" %
                      (column, value, got_main, want_main, got_unmeasured, want_unmeasured))
    if columns_seen != _expected_columns:
        bad_reasons += 1
        print("FAIL  INCONCLUSIVE_REASONS columns are %s, want %s" %
              (sorted(columns_seen), sorted(_expected_columns)))
    if bad_reasons:
        failures += 1
        print("FAIL  INCONCLUSIVE_REASONS coverage: %d wrong result(s)" % bad_reasons)
    else:
        print("ok    %-52s %d cases" %
              ("reasons_from_row matches the hardcoded (column, value) contract",
               len(_expected_firing)))

    # A column missing from the row entirely must be treated exactly like an
    # explicit NULL (dict.get returns None either way): the has_reward_func
    # main message fires, and the listing_truncated unmeasured message fires.
    # Expected text is hardcoded here, not read from INCONCLUSIVE_REASONS --
    # deriving it from the tuple under test would (as found by mutation
    # testing below) both be self-referential AND crash with an unsortable
    # None if a mutation dropped unmeasured_message back to None, hiding
    # every check after it instead of failing this one cleanly.
    check_eq("reasons_from_row treats a missing column like an explicit NULL",
             sorted(reasons_from_row({})),
             sorted([
                 "no reward function located -- we cannot clear a grader we did not find",
                 "the Hub file listing for this environment was indexed before truncation "
                 "was recorded, so whether the file set is complete has never been measured",
             ]))

    # The scenario this change exists for: an environment scanned before the
    # listing_truncated column existed. Its file-listing coverage was never
    # measured -- not "measured and found untruncated" -- and must exit 3,
    # not 0, even though every other coverage column reads clean.
    _pre_truncation_row = {"has_reward_func": 1, "fully_unreadable": 0,
                            "is_opaque": 0, "listing_truncated": None}
    check("listing_truncated=NULL on an otherwise-clean scanned row exits 3, not 0",
          exit_code(v(inconclusive_reasons=reasons_from_row(_pre_truncation_row))),
          EXIT_INCONCLUSIVE)
    # The contrast that proves NULL and 0 are no longer conflated: a row that
    # was actually measured and found NOT truncated must still exit 0.
    _measured_untruncated_row = dict(_pre_truncation_row, listing_truncated=0)
    check("listing_truncated=0 (measured, not truncated) on the same row exits 0",
          exit_code(v(inconclusive_reasons=reasons_from_row(_measured_untruncated_row))),
          EXIT_CLEAN)

    # FAIL_ON_CHOICES and SEVERITY_ORDER can drift independently -- adding a
    # new choice (e.g. "critical") without a matching SEVERITY_ORDER entry
    # would pass the `fail_on in FAIL_ON_CHOICES` validation and then KeyError
    # on the SEVERITY_ORDER[fail_on] lookup, exiting the process 1.
    check_eq("every non-'never' fail_on choice has a severity rank",
             set(FAIL_ON_CHOICES) - {"never"} <= set(SEVERITY_ORDER),
             True)

    # AuditVerdict.__post_init__ must reject the two type-confusion shapes
    # that would otherwise misjudge an environment silently (scanned='0') or
    # crash opaquely deep inside exit_code (severity_counts=None).
    try:
        AuditVerdict(env_key="o/n@1", scanned="0")  # type: ignore[arg-type]
    except TypeError:
        print("ok    %-52s %s" % ("scanned='0' is rejected, not read as True", "TypeError"))
    else:
        failures += 1
        print("FAIL  AuditVerdict(scanned='0') did not raise TypeError")
    try:
        AuditVerdict(env_key="o/n@1", scanned=True, severity_counts=None)  # type: ignore
    except TypeError:
        print("ok    %-52s %s" %
              ("severity_counts=None is rejected, not left to crash later", "TypeError"))
    else:
        failures += 1
        print("FAIL  AuditVerdict(severity_counts=None) did not raise TypeError")

    # The property the module exists for. An environment we did not scan must
    # come back EXACTLY the literal 2 (error) -- compared against the literal,
    # not the EXIT_ERROR symbol, so a symbol-level mutation (e.g. EXIT_ERROR
    # reassigned to 0) cannot make this loop lie about what it actually checked.
    bad = 0
    for sev, reasons, fail_on in itertools.product(
        ({}, {"low": 1}, {"medium": 2}, {"high": 3}),
        ([], ["opaque"], ["truncated", "opaque"]),
        FAIL_ON_CHOICES,
    ):
        code = exit_code(AuditVerdict(env_key="o/n@1", scanned=False,
                                      severity_counts=sev,
                                      inconclusive_reasons=reasons), fail_on)
        if code != 2:
            bad += 1
    if bad:
        failures += 1
        print("FAIL  unscanned environment did not return the literal 2 in %d combination(s)" %
              bad)
    else:
        print("ok    unscanned environment always returns the literal 2 (48 combinations)")

    # Against the real store, if there is one. These are reads only.
    import os
    from rlverify.corpus import store

    if not os.path.exists(os.path.expanduser(store.DEFAULT_DB)):
        print("SKIP: no corpus store at %s" % store.DEFAULT_DB)
    else:
        conn = store.connect(store.DEFAULT_DB, readonly=True)

        row, err = resolve(conn, "definitely-not-a-real-env")
        if not (row is None and err is None):
            failures += 1
            print("FAIL  absent env should be (None, None), got (%r, %r)" % (row, err))
        else:
            print("ok    absent environment resolves to (None, None)")

        known = conn.execute(
            "SELECT owner, name, env_key FROM environments "
            "WHERE has_reward_func IS NOT NULL LIMIT 1").fetchone()
        row, err = resolve(conn, "%s/%s" % (known["owner"], known["name"]))
        if err or row is None or row["env_key"] != known["env_key"]:
            failures += 1
            print("FAIL  owner/name did not resolve to %s (err=%r)"
                  % (known["env_key"], err))
        else:
            print("ok    owner/name resolves to %s" % known["env_key"])

        # A scanned environment with a live high finding must not come back clean.
        hi = conn.execute(
            "SELECT f.env_key FROM findings f JOIN environments e USING (env_key) "
            "WHERE f.severity='high' AND e.has_reward_func IS NOT NULL "
            "AND NOT COALESCE(f.review_verdict='false_positive', 0) LIMIT 1").fetchone()
        if hi is None:
            print("SKIP: no live high finding in store")
        else:
            verdict = build_verdict(conn, hi[0])
            check("live high finding -> 1", exit_code(verdict), EXIT_FINDINGS)

        # An opaque environment is a coverage gap, never a pass.
        op = conn.execute(
            "SELECT env_key FROM environments WHERE is_opaque=1 "
            "AND has_reward_func IS NOT NULL AND env_key NOT IN "
            "(SELECT env_key FROM findings) LIMIT 1").fetchone()
        if op is None:
            print("SKIP: no finding-free opaque environment in store")
        else:
            check("opaque, no findings -> 3", exit_code(build_verdict(conn, op[0])),
                  EXIT_INCONCLUSIVE)

        # Never scanned: columns are NULL, so this must be an error, not a pass.
        ns = conn.execute(
            "SELECT env_key FROM environments WHERE has_reward_func IS NULL "
            "LIMIT 1").fetchone()
        if ns is None:
            print("SKIP: every environment in store has been scanned")
        else:
            check("never scanned -> 2", exit_code(build_verdict(conn, ns[0])),
                  EXIT_ERROR)

        conn.close()

    print("\n%d failure(s)" % failures)
    raise SystemExit(1 if failures else 0)
