"""
Re-measure every claim in `reports/disclosure-draft.md`.

The disclosure accuses named people's published work. Its numbers were measured
once; this module is what keeps them true. Run it before sending anything, and
after any change to the cache or the drafts. A claim that stops reproducing
should be **withdrawn from the document**, not quietly left in it.

`probes/code.py` already gates the third finding (concurrency-bench). This
covers the other two.

Scope and honesty
-----------------
Default run is **Tier 0**: nothing is installed and no environment code is
imported. The two grading functions are copied verbatim from the published
source, and the copies are compared back against the fetched blob by AST, so a
divergence is a hard failure rather than a silent drift. The task set is read
out of `bugs.py` with `ast.literal_eval`.

`--tier1` additionally imports the published `traverse_tasks.py` (with only
`verifiers` and `datasets` stubbed) to reproduce the environment-level 0.1667 ->
1.0 figure. That **executes third-party module-level code** and is off by
default for exactly the reason CLAUDE.md's threat model gives. The mechanism
claims -- including the table of which `__builtins__` mappings still let the
attack escape -- are all measured at Tier 0 and do not need it.

Interpreter: Tier 0 is stdlib and 3.9-safe. `--tier1` imports a module using
PEP 585 annotations, so it needs the venv:

    python3 src/rlverify/probes/disclosed.py                 # Tier 0, the gate
    .venv/bin/python src/rlverify/probes/disclosed.py --tier1
"""
from __future__ import annotations

import ast
import builtins
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

DB = "~/.cache/rlverify/corpus.sqlite"
CACHE = "~/.cache/rlverify/hub"

CODE_REPAIR = "kyleskutt/code-repair-env@0.1.3"
TRAVERSE = "diyak31/traverse-tasks@0.1.26"


# --------------------------------------------------------------------------
# source access -- by content hash out of the store, never by hardcoded path
# --------------------------------------------------------------------------

def fetch_source(env_key: str, path: str) -> str:
    """Return published source from the blob cache. Raises if not cached."""
    conn = sqlite3.connect(
        "file:%s?mode=ro" % os.path.expanduser(DB), uri=True)
    try:
        row = conn.execute(
            "SELECT content_hash FROM files WHERE env_key=? AND path=?",
            (env_key, path)).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise SystemExit(
            "not cached: %s %s -- run a sync/scan first" % (env_key, path))
    key = row[0]
    blob = os.path.join(os.path.expanduser(CACHE), "blobs",
                        key[:2], key[2:4], key)
    with open(blob, encoding="utf-8-sig") as fh:   # some blobs carry a BOM
        return fh.read()


def _func_src(src: str, name: str) -> str:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if seg is None:
                raise SystemExit("could not extract %s" % name)
            return seg
    raise SystemExit("function not found in published source: %s" % name)


def assert_verbatim(env_key: str, path: str, names: List[str]) -> List[str]:
    """Fail loudly if our copy has drifted from what is published."""
    published = fetch_source(env_key, path)
    ours = open(os.path.abspath(__file__), encoding="utf-8").read()
    drifted = []
    for n in names:
        if _func_src(published, n) != _func_src(ours, n):
            drifted.append(n)
    return drifted


# --------------------------------------------------------------------------
# 1. code-repair-env -- copied VERBATIM from code_repair_env/environment.py
#    Do not reformat. `assert_verbatim` compares these against the published
#    source and fails the gate on any difference.
# --------------------------------------------------------------------------

def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if completion and isinstance(completion, list):
        return (completion[-1].get("content") or "").strip()
    return ""


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if "`" not in t:
        return t
    parts = t.split("`")
    if len(parts) >= 3:
        return parts[1].replace("python", "", 1).strip()
    return t.replace("`", "").strip()


def score_code_fix(completion, answer: str = "", info: dict = None, **_: Any) -> float:
    info = info or {}
    test_code = info.get("test", "")
    code = _strip_code_fences(_extract_text(completion))
    full_code = f"{code}\n{test_code}\n"
    try:
        exec(full_code, {"__builtins__": __builtins__}, {})
        return 1.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------

def _templates() -> List[Dict[str, str]]:
    src = fetch_source(CODE_REPAIR, "code_repair_env/bugs.py")
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "BUG_TEMPLATES"):
            return ast.literal_eval(node.value)
    raise SystemExit("BUG_TEMPLATES not found")


def _fname(template: Dict[str, str]) -> str:
    return template["fixed"].split("(")[0].replace("def ", "").strip()


def _expected(template: Dict[str, str]) -> str:
    test = template["test"].strip()
    return test.split("==", 1)[1].strip()


ALWAYS_EQ = (
    "def %s(*a, **k):\n"
    "    class _E:\n"
    "        def __eq__(self, other): return True\n"
    "        def __ne__(self, other): return False\n"
    "    return _E()\n"
)


def measure_code_repair() -> Dict[str, Any]:
    templates = _templates()
    n = len(templates)

    def tally(make: Any) -> int:
        return sum(score_code_fix(make(t), answer=t["fixed"],
                                  info={"test": t["test"]}) == 1.0
                   for t in templates)

    gold = tally(lambda t: t["fixed"])
    hardcoded = tally(lambda t: "def %s(*a, **k): return %s"
                                % (_fname(t), _expected(t)))
    always_eq = tally(lambda t: ALWAYS_EQ % _fname(t))
    buggy = tally(lambda t: t["buggy"])
    empty = tally(lambda t: "")

    # the specific task where the reference answer loses to a constant
    recursive = [t for t in templates
                 if _fname(t) in t["fixed"].split("\n", 1)[1]]
    gold_fails = [t for t in recursive
                  if score_code_fix(t["fixed"], info={"test": t["test"]}) != 1.0]

    return {
        "n": n, "gold": gold, "hardcoded": hardcoded, "always_eq": always_eq,
        "buggy": buggy, "empty": empty,
        "recursive_tasks": len(recursive),
        "gold_fails_on": [_fname(t) for t in gold_fails],
    }


# --------------------------------------------------------------------------
# 2. traverse-tasks -- the round() rebind. Mechanism measured at Tier 0.
# --------------------------------------------------------------------------

ATTACK = "import builtins\nbuiltins.round = lambda *a, **k: 1.0\n"


def _escapes(builtins_mapping: Any) -> Optional[bool]:
    """Run the attack under a given __builtins__. None = blocked outright."""
    saved = builtins.round
    try:
        exec(ATTACK, {"__builtins__": builtins_mapping}, {})
    except Exception:
        builtins.round = saved
        return None
    escaped = builtins.round is not saved
    builtins.round = saved
    return escaped


def measure_traverse_mechanism() -> Dict[str, Any]:
    saved = builtins.round
    try:
        return {
            # what the grader does today
            "real_module": _escapes(builtins),
            # the fix the draft originally suggested, and why it is not enough
            "restricted_with_import": _escapes({"__import__": __import__}),
            "restricted_no_import": _escapes({}),
            # the fix that actually holds: bind the primitive before exec
            "bind_before_exec": _bind_before_exec_holds(),
        }
    finally:
        builtins.round = saved


def _bind_before_exec_holds() -> bool:
    _round = round                      # bound before any submission runs
    saved = builtins.round
    try:
        exec(ATTACK, {"__builtins__": builtins}, {})
        return _round(1 / 6, 4) == 0.1667
    finally:
        builtins.round = saved


# --- Tier 1: the environment-level number, opt-in ------------------------

STUBS = (
    "BOOKS = {}\n"
    "MEMBER_FEES = {}\n"
    "def renew_book(*a, **k): return False\n"
    "def can_renew_fees(*a, **k): return False\n"
    "def get_book(*a, **k): return None\n"
    "def process_renewal(*a, **k): return {}\n"
)


def measure_traverse_tier1() -> Dict[str, Any]:
    """Import the published module (author code runs) and score it for real."""
    import asyncio
    import importlib.util
    import json
    import tempfile
    import types

    src = fetch_source(TRAVERSE, "traverse_tasks.py")
    for name, attrs in (("verifiers", {"Rubric": lambda **k: None,
                                       "Environment": object,
                                       "SingleTurnEnv": lambda **k: None}),
                        ("datasets", {"Dataset": type("Dataset", (), {
                            "from_list": staticmethod(lambda x: x)})})):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "traverse_tasks_published.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("traverse_tasks_published", path)
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)          # <- author module-level code runs here

    task = next(t for t in tt.TRAVERSE_TASKS
                if (json.loads(t["answer"]) if isinstance(t["answer"], str)
                    else t["answer"]).get("task_type") == "library_renewal")
    answer = task["answer"]
    original = builtins.round
    loop = asyncio.new_event_loop()

    def score(sub: str) -> float:
        return loop.run_until_complete(tt.traverse_rubric(sub, answer))

    try:
        builtins.round = original
        honest = score(STUBS)
        builtins.round = original
        attacked = score(ATTACK + STUBS)
        leaked = builtins.round is not original
        # the real point: the NEXT submission is contaminated too
        contaminated = score(STUBS) if leaked else None
        return {"honest": honest, "attacked": attacked,
                "leaked": leaked, "contaminated": contaminated}
    finally:
        builtins.round = original
        loop.close()


# --------------------------------------------------------------------------
# 4. the corpus-level counts the covering note quotes
#
# The three sections above re-measure three individual findings. They cannot
# catch a wrong *summary*, and a wrong summary is what actually reached a draft:
# the withdrawal row carried the all-severity count (13 findings / 11
# environments) under a header committing to high severity, so the rows summed
# to 34 against a stated 30. Both numbers were true; one was on the wrong basis.
#
# So this asserts the quoted table against SQL, and separately asserts that the
# rows ADD UP -- the check that would have caught it without anyone knowing
# which number was wrong.
# --------------------------------------------------------------------------

# What reports/outreach-draft.md's "If they push on the numbers" table says.
# Edit here and in the note together; the gate exists to keep them equal.
QUOTED_HIGH = {
    "reviewed":   (30, 26),
    "confirmed":  (20, 17),
    "withdrawn":  (9, 8),
    "unresolved": (1, 1),
}
QUOTED_WITHDRAWN_ALL = (13, 11)      # all severities -- must be labelled as such
QUOTED_GAPS = {"scanned_in_part": 100, "listing_truncated": 28}
QUOTED_SCANNED = 1367

_VERDICT_SQL = {"confirmed": "confirmed", "withdrawn": "false_positive",
                "unresolved": "unclear"}


def measure_review_counts() -> Dict[str, Any]:
    """The quoted table, re-derived from the store. Read-only.

    Computed on the non-research basis the note commits to ("1,367 non-research
    environments"), and again unfiltered, so a divergence between the two is
    itself reported rather than silently picked.
    """
    conn = sqlite3.connect(
        "file:%s?mode=ro" % os.path.expanduser(DB), uri=True)
    try:
        nr = ("env_key IN (SELECT env_key FROM environments "
              "WHERE research_subject=0)")

        def pair(where: str, scoped: bool) -> Tuple[int, int]:
            sql = ("SELECT COUNT(*), COUNT(DISTINCT env_key) FROM findings "
                   "WHERE severity='high' AND " + where)
            if scoped:
                sql += " AND " + nr
            return tuple(conn.execute(sql).fetchone())

        out: Dict[str, Any] = {"scoped": {}, "unscoped": {}}
        for label, verdict in _VERDICT_SQL.items():
            w = "review_verdict='%s'" % verdict
            out["scoped"][label] = pair(w, True)
            out["unscoped"][label] = pair(w, False)
        out["scoped"]["reviewed"] = pair("reviewed=1", True)
        out["unscoped"]["reviewed"] = pair("reviewed=1", False)

        out["unreviewed_high"] = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE severity='high' AND reviewed=0"
        ).fetchone()[0]
        out["withdrawn_all_sev"] = tuple(conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT env_key) FROM findings "
            "WHERE review_verdict='false_positive'").fetchone())
        out["scanned"] = conn.execute(
            "SELECT COUNT(*) FROM environments "
            "WHERE research_subject=0 AND files_scanned>0").fetchone()[0]
        out["scanned_in_part"] = conn.execute(
            "SELECT COUNT(*) FROM environments "
            "WHERE research_subject=0 AND files_over_cap>0").fetchone()[0]
        out["listing_truncated"] = conn.execute(
            "SELECT COUNT(*) FROM environments "
            "WHERE research_subject=0 AND listing_truncated=1").fetchone()[0]
        return out
    finally:
        conn.close()


def check_review_counts(m: Dict[str, Any]) -> List[str]:
    """Return one line per disagreement. Empty means the note is defensible."""
    bad: List[str] = []
    s = m["scoped"]

    for label, quoted in QUOTED_HIGH.items():
        got = s[label]
        if tuple(got) != tuple(quoted):
            bad.append("%s: note says %d findings / %d environments, "
                       "store says %d / %d"
                       % (label, quoted[0], quoted[1], got[0], got[1]))

    # The arithmetic check. This is the one that catches a basis swap without
    # needing to know which row is wrong.
    for i, unit in ((0, "findings"), (1, "environments")):
        parts = (s["confirmed"][i] + s["withdrawn"][i] + s["unresolved"][i])
        if parts != s["reviewed"][i]:
            bad.append("%s do not add: confirmed+withdrawn+unresolved = %d, "
                       "but reviewed = %d" % (unit, parts, s["reviewed"][i]))

    if m["unreviewed_high"]:
        bad.append("%d high-severity finding(s) were never reviewed -- the note "
                   "claims the tier is fully reviewed"
                   % m["unreviewed_high"])

    if tuple(m["withdrawn_all_sev"]) != tuple(QUOTED_WITHDRAWN_ALL):
        bad.append("all-severity withdrawals: note says %d/%d, store says %d/%d"
                   % (QUOTED_WITHDRAWN_ALL + tuple(m["withdrawn_all_sev"])))
    if tuple(m["withdrawn_all_sev"]) == tuple(s["withdrawn"]):
        bad.append("the all-severity and high-severity withdrawal counts are now "
                   "equal -- the note's warning about confusing them is stale")

    for key, quoted in QUOTED_GAPS.items():
        if m[key] != quoted:
            bad.append("%s: note says %d, store says %d" % (key, quoted, m[key]))
    if m["scanned"] != QUOTED_SCANNED:
        bad.append("environments scanned: note says %d, store says %d"
                   % (QUOTED_SCANNED, m["scanned"]))

    for label in QUOTED_HIGH:
        if tuple(s[label]) != tuple(m["unscoped"][label]):
            bad.append("%s differs by basis (non-research %s vs all %s) -- the "
                       "note quotes one basis and must say which"
                       % (label, s[label], m["unscoped"][label]))
    return bad


# --------------------------------------------------------------------------

def main() -> int:
    tier1 = "--tier1" in sys.argv
    failures = 0

    print("=" * 70)
    print("1. %s -- reward does not depend on the answer" % CODE_REPAIR)
    print("=" * 70)

    drift = assert_verbatim(CODE_REPAIR, "code_repair_env/environment.py",
                            ["_extract_text", "_strip_code_fences", "score_code_fix"])
    if drift:
        failures += 1
        print("  SOURCE DRIFT: %s no longer match the published file." % ", ".join(drift))
        print("  The environment was probably updated. Re-read it before sending.")
    else:
        print("  verbatim check: grader copy matches published source")

    c = measure_code_repair()
    print("  task set: %d templates\n" % c["n"])
    for label, key in (("the environment's own gold answer", "gold"),
                       ("hardcode the shown constant (1 line)", "hardcoded"),
                       ("always-equal stub (5 lines)", "always_eq"),
                       ("the unmodified buggy code", "buggy"),
                       ("empty completion", "empty")):
        print("    %-40s %2d/%d" % (label, c[key], c["n"]))

    # The document's claim is not "> 0" -- it is "beats a correct solution".
    if not (c["hardcoded"] > c["gold"] and c["always_eq"] > c["gold"]):
        failures += 1
        print("\n  CLAIM NOT REPRODUCED: no-logic submissions no longer beat the")
        print("  gold answer (%d, %d vs gold %d). Fix or withdraw section 1."
              % (c["hardcoded"], c["always_eq"], c["gold"]))
    if c["buggy"] or c["empty"]:
        failures += 1
        print("\n  HARNESS BROKEN: buggy=%d empty=%d, both should be 0"
              % (c["buggy"], c["empty"]))
    if not c["gold_fails_on"]:
        failures += 1
        print("\n  CLAIM NOT REPRODUCED: the gold answer no longer fails any")
        print("  recursive task. Drop that paragraph from section 1.")
    else:
        print("\n  gold answer fails its own test on: %s"
              % ", ".join(c["gold_fails_on"]))
        print("  (exec() with split globals/locals -- recursion cannot resolve)")

    print()
    print("=" * 70)
    print("2. %s -- submission can redefine the scoring function" % TRAVERSE)
    print("=" * 70)
    m = measure_traverse_mechanism()
    fmt = {True: "ESCAPES", False: "contained", None: "blocked"}
    print("  attack vs the __builtins__ passed to exec():")
    print("    %-42s %s" % ("the real builtins module (today)", fmt[m["real_module"]]))
    print("    %-42s %s" % ("restricted dict, keeps __import__", fmt[m["restricted_with_import"]]))
    print("    %-42s %s" % ("restricted dict, no __import__", fmt[m["restricted_no_import"]]))
    print("  suggested fix (bind _round before exec) holds: %s" % m["bind_before_exec"])

    if m["real_module"] is not True:
        failures += 1
        print("\n  CLAIM NOT REPRODUCED: the rebind no longer escapes. Withdraw section 2.")
    if m["restricted_with_import"] is not True:
        failures += 1
        print("\n  CLAIM NOT REPRODUCED: a restricted dict with __import__ now")
        print("  contains the attack -- the document says it does not. Correct it.")
    if not m["bind_before_exec"]:
        failures += 1
        print("\n  SUGGESTED FIX IS WRONG: bind-before-exec did not hold. Do not ship it.")

    if tier1:
        print("\n  --- Tier 1: importing the published module (author code runs) ---")
        t = measure_traverse_tier1()
        print("    four no-op stubs                  : %s" % t["honest"])
        print("    same stubs + the 2-line rebind    : %s" % t["attacked"])
        print("    patch outlived the episode        : %s" % t["leaked"])
        print("    honest stubs re-scored afterwards : %s" % t["contaminated"])
        if not (t["attacked"] == 1.0 and t["honest"] < 1.0):
            failures += 1
            print("\n  CLAIM NOT REPRODUCED: expected %s -> 1.0" % t["honest"])
        if not t["leaked"] or t["contaminated"] != 1.0:
            failures += 1
            print("\n  CLAIM NOT REPRODUCED: cross-episode contamination is gone.")
    else:
        print("\n  (0.1667 -> 1.0 environment-level figure needs --tier1, which")
        print("   imports the published module. See the module docstring.)")

    print()
    print("=" * 70)
    print("3. anasaqeeel/concurrency-bench@0.1.3 -- gated separately")
    print("=" * 70)
    print("  run: .venv/bin/python src/rlverify/probes/code.py")

    print()
    print("=" * 70)
    print("4. corpus-level counts quoted in reports/outreach-draft.md")
    print("=" * 70)
    m = measure_review_counts()
    s = m["scoped"]
    print("  high-severity, non-research basis:")
    print("    %-12s %8s %14s" % ("", "findings", "environments"))
    for label in ("reviewed", "confirmed", "withdrawn", "unresolved"):
        print("    %-12s %8d %14d" % (label, s[label][0], s[label][1]))
    print("  withdrawn, ALL severities        : %d findings / %d environments"
          % m["withdrawn_all_sev"])
    print("  scanned (non-research)           : %d" % m["scanned"])
    print("  scanned in part / listing truncated: %d / %d"
          % (m["scanned_in_part"], m["listing_truncated"]))

    bad = check_review_counts(m)
    if bad:
        failures += 1
        print("\n  NOTE DISAGREES WITH THE STORE -- do not send:")
        for line in bad:
            print("    - %s" % line)
    else:
        print("\n  every quoted count reconciles, and the rows add up")

    print("\n%d failure(s)" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
