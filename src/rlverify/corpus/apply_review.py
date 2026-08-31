"""
Write manual review verdicts into the findings table.

Why this exists: `findings.reviewed` / `review_verdict` are the mechanism that
makes pre-publication review structural rather than remembered, and `scan.py`
deliberately preserves reviewed rows while deleting unreviewed ones on a
re-scan. Until a verdict is written back, an adversarial review costs hours and
then evaporates the next time anybody re-scans.

Input is the JSON produced by the high-severity verification pass:
`reports/high-severity-verification.json`.

Matching granularity is (env_key, rule), not (env_key, path, rule, lineno). The
review judged a *reward function's mechanism*, and one mechanism routinely
raises the same rule on several adjacent lines -- one reviewed environment
raises `substring_containment` three times in two functions off a single `in`
test. Applying
the verdict to every hit of that rule in that environment is what the reviewer
actually decided; pinning it to one line number would leave siblings unreviewed
and let a re-scan resurrect a withdrawn claim through the back door.

Verdicts use the vocabulary the schema comment declares:
`confirmed` | `false_positive` | `unclear`.

    PYTHONPATH=. python3 -m rlverify.corpus.apply_review --dry-run
    PYTHONPATH=. python3 -m rlverify.corpus.apply_review --commit
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from rlverify.corpus import store

DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "reports", "high-severity-verification.json")

# The environments fed into the verification pass, with owner spelled out. The
# agents' own `env` labels came back in three different shapes and one of them
# is ambiguous -- two owners publish the same name -- so identity is resolved
# from this roster rather than from free text a model wrote.
#
# The roster lives OUTSIDE the source tree, in a gitignored sidecar, and is not
# published. Most of the environments in it were cleared: of the 30 reviewed
# high-severity findings, 9 were withdrawn as false positives and 1 was left
# unresolved. Shipping the roster in source would permanently associate work we
# investigated and cleared with a grader defect, which is the same harm the
# review exists to prevent. Only findings confirmed by execution are named in
# this repository.
REVIEWED_ENVS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                  "reports", "reviewed-envs.json")


def load_reviewed_envs(path: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Roster of (owner, name, rule) fed into the verification pass.

    Absent sidecar is a hard error, never an empty roster: silently reviewing
    nothing would report "0 rows updated" as success and let a re-scan
    resurrect a withdrawn claim.
    """
    path = path or REVIEWED_ENVS_PATH
    if not os.path.exists(path):
        raise SystemExit(
            "missing review roster: %s\n"
            "It is deliberately not published; see the note above "
            "REVIEWED_ENVS_PATH." % os.path.abspath(path))
    with open(path) as fh:
        rows = json.load(fh)
    return [(r["owner"], r["name"], r["rule"]) for r in rows]


_VERDICT = {"REAL": "confirmed", "FALSE_POSITIVE": "false_positive",
            "UNCLEAR": "unclear"}


def _short_name(label: str) -> str:
    """First token of whatever shape the agent used to name the environment."""
    label = label.strip()
    m = re.match(r"^[\w.-]+/([\w.-]+)", label)          # owner/name ...
    if m:
        return m.group(1)
    m = re.match(r"^(\S+)\s*\(", label)                 # name (owner @ ver)
    return m.group(1) if m else label.split()[0]


def load_verdicts(path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """(short_name, rule) -> {verdict, note}. Later duplicates must agree."""
    with open(path) as fh:
        data = json.load(fh)["result"]
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for f in data["all"]:
        key = (_short_name(f["env"]), f["rule"])
        verdict = _VERDICT.get(f["first_pass"], "unclear")
        # A first-pass REAL that a refuter then killed is NOT confirmed.
        if verdict == "confirmed" and f.get("refuted") is True:
            verdict = "false_positive"
        prev = out.get(key)
        if prev and prev["verdict"] != verdict:
            # Two judgements of the same rule in the same environment disagreed;
            # refuse to pick a winner silently.
            verdict = "unclear"
        out[key] = {
            "verdict": verdict,
            "note": (f.get("mechanism") or f.get("refute_reason") or "")[:400],
        }
    return out


def apply(conn, verdicts, commit: bool = False) -> Dict[str, int]:
    stats = {"envs_matched": 0, "envs_missing": 0, "rows_updated": 0,
             "confirmed": 0, "false_positive": 0, "unclear": 0}
    for owner, name, rule in load_reviewed_envs():
        row = conn.execute(
            "SELECT env_key FROM environments WHERE owner=? AND name=?",
            (owner, name)).fetchone()
        if not row:
            stats["envs_missing"] += 1
            print("  MISSING env %s/%s" % (owner, name))
            continue
        v = verdicts.get((name, rule))
        if v is None:
            stats["envs_missing"] += 1
            print("  NO VERDICT for %s/%s %s" % (owner, name, rule))
            continue
        n = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE env_key=? AND rule=?",
            (row["env_key"], rule)).fetchone()[0]
        if not n:
            # The re-scan may legitimately no longer raise it (a rule was
            # tightened). Not an error, but say so rather than imply coverage.
            print("  no live findings: %-30s %s" % (name, rule))
            continue
        stats["envs_matched"] += 1
        stats["rows_updated"] += n
        stats[v["verdict"]] += n
        if commit:
            conn.execute(
                "UPDATE findings SET reviewed=1, review_verdict=? "
                "WHERE env_key=? AND rule=?", (v["verdict"], row["env_key"], rule))
    if commit:
        store.commit(conn)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Persist manual review verdicts.")
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--input", default=os.path.abspath(DEFAULT_INPUT))
    p.add_argument("--commit", action="store_true",
                   help="write; default is a dry run")
    args = p.parse_args(argv)

    verdicts = load_verdicts(args.input)
    print("verdicts loaded: %d" % len(verdicts))
    conn = store.connect(args.db, readonly=not args.commit)
    stats = apply(conn, verdicts, commit=args.commit)
    print("\n%s" % ("COMMITTED" if args.commit else "DRY RUN -- nothing written"))
    for k in ("envs_matched", "envs_missing", "rows_updated",
              "confirmed", "false_positive", "unclear"):
        print("  %-16s %d" % (k, stats[k]))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
