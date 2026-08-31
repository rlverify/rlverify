"""
Corpus sync: pull the Environments Hub catalog into the local SQLite store.

Three passes, each independently resumable:

  1. catalog  -- paginate the list endpoint (~15 requests for the whole corpus)
  2. detail   -- per-env sha256 / wheel_url / dependencies
  3. files    -- recursive directory walk, indexing content_hash per file

Only pass 1 is cheap. Passes 2 and 3 are one-or-more requests per environment,
so they checkpoint to SQLite continuously and skip work already done. Kill the
process at any point and re-run: it picks up where it stopped.

No environment code is imported or executed at any stage.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from rlverify.corpus import store
from rlverify.static.hub_client import HubClient


def sync_catalog(conn, client: HubClient, verbose: bool = True) -> int:
    """Returns the count of DISTINCT environments seen, not records yielded.

    The Hub paginates by offset, so an environment published mid-sweep shifts
    the window: records can repeat and -- more importantly -- can be skipped.
    Counting yielded records therefore both over- and under-reports. We track
    unique keys, and report repeats separately since they are the observable
    symptom that the window moved (and thus that something may also be missing).
    """
    seen = set()
    yielded = 0
    t0 = time.monotonic()
    for rec in client.iter_environments():
        key = store.upsert_environment(conn, rec)
        seen.add(key)
        yielded += 1
        if yielded % 200 == 0:
            store.commit(conn)
            if verbose:
                print("  catalog: %d ..." % yielded, flush=True)
    store.commit(conn)
    repeats = yielded - len(seen)
    if verbose:
        print("  catalog: %d unique environments in %.1fs" % (len(seen), time.monotonic() - t0))
        if repeats:
            print("  note: %d duplicate record(s) across pages -- the catalog changed "
                  "mid-sweep, so a record may also have been missed" % repeats)
    return len(seen)


def sync_details(conn, client: HubClient, limit: Optional[int] = None,
                 verbose: bool = True, refresh: bool = False,
                 env_key: Optional[str] = None) -> int:
    """Populate version detail. `refresh` re-applies it to rows already ok.

    That is how a *parsing* fix reaches blobs already on disk. `update_detail`
    read `requires_python` where the Hub sends `python_requires`, leaving the
    column NULL for the whole corpus while the value sat in 1448 cached
    responses. Pointing an offline client at a refresh pass re-derives every row
    from cache and issues no requests at all.
    """
    if env_key:
        # A single named environment, whether or not detail_ok is already set:
        # the caller asked for this one specifically.
        rows = conn.execute(
            "SELECT env_key, owner, name, version FROM environments WHERE env_key=?",
            (env_key,)).fetchall()
    else:
        where = "" if refresh else "WHERE detail_ok = 0 "
        sql = ("SELECT env_key, owner, name, version FROM environments " + where +
               "ORDER BY stars DESC, updated_at DESC")
        if limit:
            sql += " LIMIT %d" % int(limit)
        rows = conn.execute(sql).fetchall()
    if verbose:
        print("  detail: %d pending" % len(rows))
    done = 0
    for i, r in enumerate(rows, 1):
        detail = client.get_detail(r["owner"], r["name"], r["version"])
        store.update_detail(conn, r["env_key"], detail)
        done += 1 if detail else 0
        if i % 50 == 0:
            store.commit(conn)
            if verbose:
                print("  detail: %d/%d (ok %d)" % (i, len(rows), done), flush=True)
    store.commit(conn)
    return done


def sync_files(conn, client: HubClient, limit: Optional[int] = None,
               verbose: bool = True, env_key: Optional[str] = None,
               force: bool = False) -> int:
    if env_key:
        # `force` re-walks an environment already listed. Used to measure a
        # NULL listing_truncated, which means "never recorded", not "not
        # truncated" -- without it the scoped query no-ops and the audit judges
        # a value nobody ever took.
        sql = "SELECT env_key, owner, name, version FROM environments WHERE env_key=?"
        if not force:
            sql += " AND files_listed=0"
        rows = conn.execute(sql, (env_key,)).fetchall()
    else:
        rows = store.pending_file_listing(conn, limit=limit)
    if verbose:
        print("  files: %d environments pending" % len(rows))
    total_files = 0
    for i, r in enumerate(rows, 1):
        before = len(client.failures)
        entries, truncated_by, incomplete = client.walk_files_ex(
            r["owner"], r["name"], r["version"])
        # Any directory-listing failure during this walk -- flagged directly by
        # `incomplete`, or by a growth in `client.failures` as a second,
        # independent signal -- means we did not actually measure this
        # environment's file set, whether or not we still came back with some
        # entries. The old code only looked at `client.failures` and only when
        # `entries` was empty, so a short-but-nonempty listing (one bad
        # subdirectory among many good ones) was recorded as complete.
        failed = incomplete or (len(client.failures) > before)
        if not entries:
            # Distinguish "empty repo" (measured, genuinely nothing there) from
            # "we could not read it" -- the second is a coverage gap, not
            # evidence of anything, and must be reported as such.
            err = "walk failed" if failed else "empty"
        elif failed:
            err = "walk incomplete: some director(y/ies) could not be listed"
        else:
            err = None
        # Tri-state coverage claim, not a plain bool: True/False are
        # measurements we actually made and may write; None means the walk did
        # not complete, so `record_files` must leave listing_truncated (and the
        # file counts) exactly as they were rather than write a claim we did
        # not earn.
        truncated_state = None if failed else bool(truncated_by)
        nf, _ = store.record_files(conn, r["env_key"], entries, error=err,
                                   truncated=truncated_state)
        total_files += nf
        # Commit every environment. The Hub serves directory listings at well
        # under 1/sec, so one environment can cost minutes; batching commits by
        # 25 meant an 11-hour run over 30 environments banked nothing and could
        # not be resumed. A commit here is cheap next to the walk that earned it.
        store.commit(conn)
        if verbose:
            status = ""
            if failed:
                status = "  INCOMPLETE"
            elif truncated_by:
                status = "  TRUNCATED(%s)" % truncated_by
            print("  files: %d/%d  %-34s %5d files%s"
                  % (i, len(rows), r["name"], nf, status), flush=True)
    store.commit(conn)
    return total_files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sync the Prime Intellect Environments Hub corpus.")
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--cache", default="~/.cache/rlverify/hub")
    p.add_argument("--rate-hz", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=None,
                   help="cap environments processed in the detail/files passes")
    p.add_argument("--pass", dest="passes", default="catalog,detail,files",
                   help="comma-separated subset of: catalog,detail,files")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="serve every request from the blob cache; a miss is "
                        "recorded as a failure rather than fetched")
    p.add_argument("--refresh-detail", action="store_true",
                   help="re-apply the detail pass to rows already marked ok "
                        "(use with --offline to replay a parsing fix for free)")
    args = p.parse_args(argv)

    conn = store.connect(args.db)
    client = HubClient(cache_dir=args.cache, rate_hz=args.rate_hz,
                       offline=args.offline)

    if args.stats_only:
        print(store.stats(conn))
        return 0

    passes = [s.strip() for s in args.passes.split(",") if s.strip()]
    t0 = time.monotonic()

    if "catalog" in passes:
        print("[1/3] catalog")
        remote_total = client.total_count()
        n = sync_catalog(conn, client)
        if remote_total is not None and n < remote_total:
            # Loud, not silent: partial coverage invalidates any corpus claim.
            print("  WARNING: %d unique synced but Hub reports %d -- %d missing"
                  % (n, remote_total, remote_total - n), file=sys.stderr)

    if "detail" in passes:
        print("[2/3] detail")
        sync_details(conn, client, limit=args.limit,
                     refresh=args.refresh_detail)

    if "files" in passes:
        print("[3/3] files")
        sync_files(conn, client, limit=args.limit)

    s = store.stats(conn)
    print("\ncorpus: %s" % s)
    print("elapsed: %.1fs   http failures: %d" % (time.monotonic() - t0, len(client.failures)))
    if client.failures:
        seen = {}
        for f in client.failures:
            seen[f["error"]] = seen.get(f["error"], 0) + 1
        for err, cnt in sorted(seen.items(), key=lambda kv: -kv[1])[:8]:
            print("   %5d x %s" % (cnt, err))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
