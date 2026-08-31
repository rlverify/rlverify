"""
SQLite catalog for the Environments Hub corpus.

Two tables. `environments` is the catalog (one row per env@version).
`files` is the per-file index built from the Hub's directory listings.

Every write is an upsert keyed on natural identity, so sync and fetch are
idempotent: killing a run mid-sweep and restarting converges to the same state
rather than duplicating or half-writing. At 1454 environments over flaky HTTP
that property is not optional -- a sweep that can't be resumed is a sweep that
never finishes.

`content_hash` comes from the Hub itself, so file-level change detection is
exact and costs no extra request.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DB = "~/.cache/rlverify/corpus.sqlite"

# Environments whose *subject* is reward hacking. A red-team environment that
# ships a deliberately gameable grader is not evidence of accidental
# hackability -- it is the point of the environment. Counting them inflates the
# corpus rate with precisely the cases a critic would quote back, so they are
# excluded from the denominator and reported as an explicit count. Never
# dropped silently: an exclusion nobody can see is a fudged denominator.
#
# The matching is narrow on purpose, and the bias is deliberate. Excluding an
# environment removes it from the numerator as well, so an over-broad rule
# quietly deletes real findings and makes the corpus look cleaner than it is --
# the one direction this tool must never err in. Bare "hacking" and
# "adversarial" therefore do not qualify: they matched a cybersecurity CTF, an
# adversarial-examples MNIST task and a unicode-robustness IFEval variant, none
# of which is a study of graders.
_RESEARCH_SUBJECT_RE = re.compile(
    r"reward[\s_\-]*hack(ing|s|ed)?"
    r"|hack(ing|able|ed)?[\s_\-]*(the[\s_\-]*)?(reward|grader|verifier|rubric)"
    r"|(hackable|gameable|exploitable)[\s_\-]*(grader|verifier|reward|rubric)"
    r"|deliberately[\s_\-]*(hackable|gameable|broken|exploitable|weak)"
    r"|red[\s_\-]*team"
    r"|honeypot"
    r"|adversarial[\s_\-]*(rl|reward|grader|verifier|training)",
    re.IGNORECASE,
)
# A grader advertised as *resisting* hacking has a careful author -- and is
# exactly the claim most worth auditing, so it stays in the corpus. Only the
# text immediately after the match counts, so "reward hacking -- detects onset"
# (a genuine study) is not mistaken for a robustness claim.
_RESEARCH_NEGATION_RE = re.compile(
    r"^[\s_\-]*(robust|resistant|resilient|proof|hardened|immune|safe|aware)",
    re.IGNORECASE,
)


def is_research_subject(name: Optional[str], description: Optional[str],
                        tags: Optional[Iterable[str]]) -> bool:
    """True when an environment is *about* reward hacking rather than a victim of it."""
    hay = " ".join([name or "", description or "", " ".join(tags or [])])
    for m in _RESEARCH_SUBJECT_RE.finditer(hay):
        if not _RESEARCH_NEGATION_RE.match(hay[m.end():m.end() + 24]):
            return True
    return False

SCHEMA = """
CREATE TABLE IF NOT EXISTS environments (
    env_key        TEXT PRIMARY KEY,        -- owner/name@version
    owner          TEXT NOT NULL,
    owner_type     TEXT,
    name           TEXT NOT NULL,
    version        TEXT,
    hub_id         TEXT,
    description    TEXT,
    tags_json      TEXT,
    stars          INTEGER,
    visibility     TEXT,
    ci_status      TEXT,
    created_at     TEXT,
    updated_at     TEXT,
    -- 1 when the environment's subject is reward hacking (see is_research_subject)
    research_subject INTEGER DEFAULT 0,
    -- populated by the detail pass
    sha256         TEXT,
    wheel_url      TEXT,
    deps_json      TEXT,
    requires_python TEXT,
    detail_ok      INTEGER DEFAULT 0,
    -- populated by the file pass
    files_listed   INTEGER DEFAULT 0,
    n_files        INTEGER,
    n_py_files     INTEGER,
    fetch_error    TEXT,
    -- 1 when the Hub walk hit its entry ceiling: the file set is
    -- known-incomplete, so this environment can never count as a clean result
    listing_truncated INTEGER,
    synced_at      TEXT,
    -- populated by the static scan. NULL means "never scanned", which the
    -- report must render as "not computed" rather than as 0.
    has_reward_func  INTEGER,
    is_code_grader   INTEGER,
    is_opaque        INTEGER,
    files_considered INTEGER,
    files_scanned    INTEGER,
    files_over_cap   INTEGER,
    fully_unreadable INTEGER
);

CREATE TABLE IF NOT EXISTS files (
    env_key        TEXT NOT NULL,
    path           TEXT NOT NULL,
    name           TEXT,
    size           INTEGER,
    content_hash   TEXT,
    is_python      INTEGER DEFAULT 0,
    fetched        INTEGER DEFAULT 0,
    parse_ok       INTEGER,                 -- NULL = not attempted
    parse_error    TEXT,
    PRIMARY KEY (env_key, path)
);

-- Static findings. One row per (file, rule, line) so re-scanning is idempotent:
-- a rerun overwrites rather than accumulating duplicates.
CREATE TABLE IF NOT EXISTS findings (
    env_key        TEXT NOT NULL,
    path           TEXT NOT NULL,
    rule           TEXT NOT NULL,
    severity       TEXT,
    func           TEXT,
    lineno         INTEGER,
    snippet        TEXT,
    why            TEXT,
    probe          TEXT,
    reviewed       INTEGER DEFAULT 0,   -- manual verification before publishing
    review_verdict TEXT,                -- confirmed | false_positive | unclear
    PRIMARY KEY (env_key, path, rule, lineno)
);

CREATE INDEX IF NOT EXISTS idx_findings_rule ON findings(rule);
CREATE INDEX IF NOT EXISTS idx_findings_sev  ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_files_env    ON files(env_key);
CREATE INDEX IF NOT EXISTS idx_files_py     ON files(is_python) WHERE is_python = 1;
CREATE INDEX IF NOT EXISTS idx_env_listed   ON environments(files_listed);
CREATE INDEX IF NOT EXISTS idx_env_owner    ON environments(owner);
"""


def env_key(owner: str, name: str, version: Optional[str]) -> str:
    return "%s/%s@%s" % (owner, name, version or "")


def connect(db_path: str = DEFAULT_DB, readonly: bool = False) -> sqlite3.Connection:
    """Open the corpus store.

    `readonly=True` opens a GENUINELY read-only connection, via a `file:...
    ?mode=ro` URI -- not merely one that skips schema creation. A prior version
    of this function still opened the file read-write and ran
    `PRAGMA journal_mode=WAL` / `PRAGMA synchronous=NORMAL` even when
    `readonly=True`; both of those write to the database file (the journal-mode
    pragma writes the file's header), so a caller that only wanted to SELECT
    could still take a write lock against a corpus sweep running in another
    process -- the exact hazard this parameter exists to avoid. `mode=ro`
    makes that structurally impossible: SQLite itself refuses any write
    attempted through this connection, including by a pragma.

    A long corpus sweep holds the write lock for hours, and an inspecting
    reader that insists on running DDL will fail with "database is locked"
    even though it only wants to SELECT. Readers must never need a write lock,
    which is also why schema/column creation is skipped here -- that part of
    the original contract is unchanged.
    """
    path = os.path.expanduser(db_path)
    if readonly:
        if not os.path.exists(path):
            # `mode=ro` never creates the file; sqlite3's own error for that
            # ("unable to open database file") does not say why. Say why.
            raise FileNotFoundError(
                "no store at %s -- a readonly connection never creates one" % path)
        uri = "file:%s?mode=ro" % urllib.parse.quote(path)
        conn = sqlite3.connect(uri, uri=True, timeout=60.0)
        conn.row_factory = sqlite3.Row
        # Read-only, so this never writes to the file -- SQLite tracks
        # busy_timeout per-connection, not in the database itself.
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    # WAL so a long read (aggregate) doesn't block the writer (fetch).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    _ensure_schema(conn)
    _ensure_columns(conn)
    return conn


EXPECTED_TABLES = ("environments", "files", "findings")

# Columns added after the first corpus sync. The store already holds 1454
# environments and 22k files; re-syncing to gain a column would cost hours of
# HTTP, so new columns are added in place.
ADDED_COLUMNS = (
    ("environments", "research_subject", "INTEGER DEFAULT 0"),
    # Written by the scan. Held only in memory before, so the --report-only path
    # had nothing to read and printed a fabricated 0 for the two denominators
    # that decide whether a code-grader claim means anything.
    ("environments", "has_reward_func", "INTEGER"),
    ("environments", "is_code_grader", "INTEGER"),
    ("environments", "is_opaque", "INTEGER"),
    ("environments", "files_considered", "INTEGER"),
    ("environments", "files_scanned", "INTEGER"),
    ("environments", "files_over_cap", "INTEGER"),
    ("environments", "fully_unreadable", "INTEGER"),
    # Written by the file pass: the Hub listing was truncated, so this
    # environment's file set is known-incomplete and can never count as clean.
    ("environments", "listing_truncated", "INTEGER"),
)


def has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    """Whether `table.col` exists.

    A readonly connection cannot run the ALTER that adds a column, so a reader
    against a store the writer has not migrated yet must be able to ask. The
    answer feeds "not computed", never 0.
    """
    return col in {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for table, col, decl in ADDED_COLUMNS:
        have = {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}
        if not have or col in have:
            continue
        conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, col, decl))
        conn.commit()


def _ensure_schema(conn: sqlite3.Connection, attempts: int = 12) -> None:
    """Create missing tables, tolerating a concurrent long-running writer.

    A corpus sweep holds a write transaction across many slow HTTP calls, so
    DDL can be locked out for minutes. Two mitigations: only run DDL when a
    table is genuinely missing (the common case is a no-op read), and retry
    with backoff rather than dying -- a scan that aborts because a sync happens
    to be running is a tool nobody will leave unattended.
    """
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if all(t in have for t in EXPECTED_TABLES):
        return

    last = None
    for i in range(attempts):
        try:
            conn.executescript(SCHEMA)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            last = exc
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            time.sleep(min(2.0 * (i + 1), 15.0))
    raise sqlite3.OperationalError(
        "could not create schema after %d attempts (a corpus sync may be "
        "holding the write lock): %s" % (attempts, last))


def commit(conn: sqlite3.Connection, attempts: int = 10) -> None:
    """Commit, retrying while another writer holds the lock.

    SQLite allows one writer at a time. A corpus sweep and a scan legitimately
    run against the same store, and an unretried commit turns that into a
    crash hours into a run. Losing a multi-hour sweep to a transient lock is
    the difference between a tool you leave running overnight and one you
    babysit -- so every long-running pass commits through here.
    """
    last = None
    for i in range(attempts):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            last = exc
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            time.sleep(min(1.0 * (i + 1), 10.0))
    raise sqlite3.OperationalError("commit failed after %d attempts: %s" % (attempts, last))


def upsert_environment(conn: sqlite3.Connection, rec: Dict[str, Any]) -> str:
    """Insert or refresh a catalog row from a Hub list record. Returns env_key.

    Only catalog fields are written here; the detail/file columns are left
    untouched so re-running sync never discards fetch progress.
    """
    owner = (rec.get("owner") or {}).get("name") or "?"
    owner_type = (rec.get("owner") or {}).get("type")
    name = rec.get("name") or "?"
    version = rec.get("latest_version")
    key = env_key(owner, name, version)
    tags = rec.get("tags") or []
    conn.execute(
        """
        INSERT INTO environments
            (env_key, owner, owner_type, name, version, hub_id, description,
             tags_json, stars, visibility, ci_status, created_at, updated_at,
             research_subject)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(env_key) DO UPDATE SET
            description = excluded.description,
            tags_json   = excluded.tags_json,
            stars       = excluded.stars,
            ci_status   = excluded.ci_status,
            updated_at  = excluded.updated_at,
            research_subject = excluded.research_subject
        """,
        (
            key, owner, owner_type, name, version, rec.get("id"),
            rec.get("description"), json.dumps(tags),
            rec.get("stars"), rec.get("visibility"), rec.get("latest_ci_status"),
            rec.get("created_at"), rec.get("updated_at"),
            1 if is_research_subject(name, rec.get("description"), tags) else 0,
        ),
    )
    return key


def backfill_research_subject(conn: sqlite3.Connection) -> int:
    """Classify already-synced rows. Returns the count now marked.

    The corpus was synced before this column existed, so the flag has to be
    derivable from stored fields alone -- which it is: name, description, tags.
    Idempotent, and cheap enough to run at the head of every scan so the
    exclusion can never silently go stale.

    Writes only the rows whose flag actually changed, so in the steady state
    this is a pure read that takes no write lock. That matters: it runs on the
    `--report-only` path too, and a reporting command that grabs the write lock
    while a multi-hour sync is running is a second writer -- the thing that
    killed the file pass at 750 the first time.
    """
    marked = changed = 0
    for row in conn.execute(
            "SELECT env_key, name, description, tags_json, research_subject "
            "FROM environments"):
        try:
            tags = json.loads(row["tags_json"] or "[]")
        except ValueError:
            tags = []
        flag = 1 if is_research_subject(
            row["name"], row["description"],
            [t if isinstance(t, str) else str(t) for t in tags]) else 0
        marked += flag
        if row["research_subject"] != flag:
            changed += 1
            conn.execute("UPDATE environments SET research_subject=? WHERE env_key=?",
                         (flag, row["env_key"]))
    if changed:
        commit(conn)
    return marked


def update_detail(conn: sqlite3.Connection, key: str, detail: Optional[Dict[str, Any]]) -> None:
    if not detail:
        conn.execute("UPDATE environments SET detail_ok=0 WHERE env_key=?", (key,))
        return
    meta = detail.get("metadata") or {}
    conn.execute(
        """
        UPDATE environments
           SET sha256=?, wheel_url=?, deps_json=?, requires_python=?, detail_ok=1
         WHERE env_key=?
        """,
        (
            # NOTE: `sha256` is the *sdist* hash. Verified by fetching a wheel and
            # hashing it -- the values do not match, and `metadata.original_filename`
            # is a .tar.gz for every environment in the corpus. The Hub publishes no
            # wheel hash, so this must never be used as a wheel integrity check: it
            # would fail 100% of the time.
            detail.get("sha256"), detail.get("wheel_url"),
            json.dumps(meta.get("dependencies") or []),
            # The Hub sends `python_requires`. Reading `requires_python` left this
            # column NULL for all 1454 rows while the data sat in the cached blobs.
            meta.get("python_requires") or meta.get("requires_python"), key,
        ),
    )


def record_files(conn: sqlite3.Connection, key: str, entries: List[Dict[str, Any]],
                 error: Optional[str] = None,
                 truncated: Optional[bool] = False) -> Tuple[int, int]:
    """Index a walked file list. Returns (n_files, n_py_files).

    `truncated` is a TRI-STATE coverage claim, not a plain bool -- this is the
    fix for a real defect: a failed re-walk used to stamp `listing_truncated=0`
    (the strongest possible coverage claim) on the same UPDATE that recorded
    `fetch_error='walk failed'`, converting "never measured" into "measured,
    complete" precisely when the measurement failed. 1418 scanned environments
    were exposed to this; roughly 1097 would have flipped from an honest
    inconclusive to a false CLEAN.

      True  -- the walk hit a known cap (MAX_WALK_ENTRIES/MAX_WALK_REQUESTS);
               a genuine measurement, and the file set is known-incomplete.
      False -- the walk completed with no read failures; a genuine measurement
               of a complete (or empty) listing.
      None  -- the walk did NOT complete (some directory's listing could not be
               read). We have no reliable measurement, so `listing_truncated`,
               `n_files` and `n_py_files` are left untouched -- whatever they
               already were, NULL or a real prior measurement -- rather than
               overwritten with a claim this call never earned.

    `entries` is whatever the caller collected up to the point of failure, if
    any. Existing rows in `files` for paths NOT present in `entries` are left
    untouched (never deleted), so a partial re-walk cannot silently erase files
    a previous, complete walk already found -- and is exactly why the file
    counts must not be zeroed on failure either: the `files` table keeps its
    rows while the environment-level counts would otherwise claim there are
    none, leaving the store self-contradictory.
    """
    n_py = 0
    for e in entries:
        path = e.get("path") or e.get("name") or ""
        is_py = 1 if path.endswith(".py") else 0
        n_py += is_py
        conn.execute(
            """
            INSERT INTO files (env_key, path, name, size, content_hash, is_python)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(env_key, path) DO UPDATE SET
                size         = excluded.size,
                content_hash = excluded.content_hash,
                -- a changed hash invalidates any prior parse verdict
                parse_ok     = CASE WHEN files.content_hash IS NOT excluded.content_hash
                                    THEN NULL ELSE files.parse_ok END,
                fetched      = CASE WHEN files.content_hash IS NOT excluded.content_hash
                                    THEN 0 ELSE files.fetched END
            """,
            (key, path, e.get("name"), e.get("size"), e.get("content_hash"), is_py),
        )
    if truncated is None:
        # No measurement to write: only record that the attempt happened and
        # why, and mark the environment as having been through the file pass
        # at least once (unchanged from before -- files_listed already meant
        # "attempted", not "succeeded", and pending_file_listing's resume
        # queue is untouched by this change).
        conn.execute(
            """
            UPDATE environments
               SET files_listed=1, fetch_error=?, synced_at=datetime('now')
             WHERE env_key=?
            """,
            (error, key),
        )
    else:
        conn.execute(
            """
            UPDATE environments
               SET files_listed=1, n_files=?, n_py_files=?, fetch_error=?,
                   listing_truncated=?, synced_at=datetime('now')
             WHERE env_key=?
            """,
            (len(entries), n_py, error, 1 if truncated else 0, key),
        )
    return len(entries), n_py


def pending_file_listing(conn: sqlite3.Connection, limit: Optional[int] = None
                         ) -> List[sqlite3.Row]:
    """Environments whose files have not been listed yet -- the resume queue."""
    sql = ("SELECT env_key, owner, name, version FROM environments "
           "WHERE files_listed = 0 ORDER BY stars DESC, updated_at DESC")
    if limit:
        sql += " LIMIT %d" % int(limit)
    return conn.execute(sql).fetchall()


def stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
    return {
        "environments": q("SELECT COUNT(*) FROM environments"),
        "with_detail": q("SELECT COUNT(*) FROM environments WHERE detail_ok=1"),
        "files_listed": q("SELECT COUNT(*) FROM environments WHERE files_listed=1"),
        "files": q("SELECT COUNT(*) FROM files"),
        "py_files": q("SELECT COUNT(*) FROM files WHERE is_python=1"),
        "owners": q("SELECT COUNT(DISTINCT owner) FROM environments"),
        "zero_star": q("SELECT COUNT(*) FROM environments WHERE stars=0"),
        "research_subject": q("SELECT COUNT(*) FROM environments WHERE research_subject=1"),
    }


if __name__ == "__main__":
    conn = connect("~/.cache/rlverify/_smoke.sqlite")
    k = upsert_environment(conn, {
        "owner": {"name": "acme", "type": "team"}, "name": "demo",
        "latest_version": "0.1.0", "id": "x1", "description": "d",
        "tags": ["eval"], "stars": 3, "visibility": "PUBLIC",
        "created_at": "2026-01-01", "updated_at": "2026-01-02",
    })
    assert k == "acme/demo@0.1.0", k
    upsert_environment(conn, {  # idempotency: same key, no duplicate row
        "owner": {"name": "acme", "type": "team"}, "name": "demo",
        "latest_version": "0.1.0", "stars": 4, "tags": ["eval", "math"],
    })
    update_detail(conn, k, {"sha256": "abc", "wheel_url": "http://w",
                            "metadata": {"dependencies": ["verifiers>=0.2.1"]}})
    record_files(conn, k, [
        {"path": "demo/__init__.py", "name": "__init__.py", "size": 10, "content_hash": "h1"},
        {"path": "README.md", "name": "README.md", "size": 20, "content_hash": "h2"},
    ])
    conn.commit()
    s = stats(conn)
    assert s["environments"] == 1, s
    assert s["files"] == 2 and s["py_files"] == 1, s
    assert s["with_detail"] == 1, s
    assert not pending_file_listing(conn), "listing should be marked done"

    # Research-subject exclusion. Both directions matter: over-excluding deletes
    # real findings and manufactures a clean corpus, under-excluding inflates
    # the rate with environments that are gameable on purpose. Every case below
    # is a real Hub environment the first version of this rule got wrong.
    assert is_research_subject("skill-reward-hacking", None, [])
    assert is_research_subject("demo", "a red team suite for graders", [])
    assert is_research_subject("aaa-env", "maximize the letter 'a'", ["reward-hacking"])
    assert is_research_subject(
        "assay-hackword", "A deliberately hackable grader that pays for a word", [])
    assert is_research_subject(
        "hack-detector", "detects hacking onset from reward hacking signals", [])
    assert is_research_subject("canary-bench", "Credential-honeypot evaluation", [])
    # ...and the ones a bare keyword match wrongly removed from the corpus:
    assert not is_research_subject(
        "linear-algebra", "QA with a reward-hacking-robust, math-aware grader",
        ["math", "reward-hacking-robust"])
    assert not is_research_subject(
        "hacking-ctf", "CTF for evaluating hacking and penetration testing",
        ["hacking", "ctf", "security"])
    assert not is_research_subject(
        "mnist-adversarial", "Distinguishing adversarial examples from MNIST digits",
        ["adversarial-example"])
    assert not is_research_subject(
        "ifeval-confusables", "inputs adversarially augmented with unicode confusables",
        ["adversarial-robustness"])
    assert not is_research_subject("math-500", "grade-school arithmetic", ["math"])
    assert not is_research_subject("swe-bench", "patch real repositories", ["code"])
    upsert_environment(conn, {
        "owner": {"name": "acme", "type": "team"}, "name": "reward-hacking-probe",
        "latest_version": "0.1.0", "description": "red-team graders", "tags": [],
    })
    conn.commit()
    assert stats(conn)["research_subject"] == 1, stats(conn)
    assert backfill_research_subject(conn) == 1, "backfill must be idempotent"
    s = stats(conn)
    print("store smoke: ok  %s" % s)

    # Tri-state `truncated` (Critical A, fix round 1 -- a failed walk must
    # never overwrite a real measurement, or a NULL, with a claim it did not
    # earn). `k` already carries a measured, untruncated listing from above.
    record_files(conn, k, [
        {"path": "demo/__init__.py", "name": "__init__.py", "size": 10, "content_hash": "h1"},
    ], truncated=True)
    row = dict(conn.execute(
        "SELECT listing_truncated, n_files FROM environments WHERE env_key=?", (k,)
    ).fetchone())
    assert row == {"listing_truncated": 1, "n_files": 1}, row
    record_files(conn, k, [], error="walk failed", truncated=None)
    row = dict(conn.execute(
        "SELECT listing_truncated, n_files, fetch_error FROM environments WHERE env_key=?",
        (k,)).fetchone())
    assert row == {"listing_truncated": 1, "n_files": 1, "fetch_error": "walk failed"}, row
    n_file_rows = conn.execute(
        "SELECT COUNT(*) FROM files WHERE env_key=?", (k,)).fetchone()[0]
    assert n_file_rows == 2, n_file_rows  # __init__.py + README.md, untouched by the failed walk
    conn.commit()
    print("store smoke: ok  truncated=None preserves a prior real measurement, "
          "files table untouched")

    # connect(readonly=True) must be GENUINELY read-only (Minor I): reject
    # writes, never create a missing file, and still serve reads.
    conn.close()
    missing = os.path.expanduser("~/.cache/rlverify/_smoke_missing.sqlite")
    try:
        connect(missing, readonly=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("readonly connect on a missing path did not raise")
    assert not os.path.exists(missing), "readonly connect must never create a file"
    ro = connect("~/.cache/rlverify/_smoke.sqlite", readonly=True)
    assert ro.execute("SELECT COUNT(*) FROM environments").fetchone()[0] == 2
    try:
        ro.execute("DELETE FROM environments")
        ro.commit()
    except sqlite3.OperationalError:
        pass
    else:
        raise AssertionError("a write through a readonly connection was not rejected")
    ro.close()
    print("store smoke: ok  connect(readonly=True) is genuinely read-only")

    os.remove(os.path.expanduser("~/.cache/rlverify/_smoke.sqlite"))
