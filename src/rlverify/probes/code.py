"""
Code-grader probes: deliberately wrong responses that a weak grader still pays.

The static pass can only say a grader *looks* gameable. This module closes the
gap for one rule -- `stdout_spoof` -- by constructing the response that would
exploit it and measuring what the grader actually returns. A static smell is an
argument; a paid response is a demonstration.

Scope and honesty
-----------------
This does **not** install or import the audited environment. The threat model
is inverted from the obvious reading -- the probes are strings, the untrusted
code is the environment -- and `pip install` from an sdist executes code at
build time, so running a vendor's package is a Tier 1/2 activity that is
deliberately still deferred.

What runs here instead is a **verbatim reproduction of the grader's own
output-parsing logic**, copied from the published source, driven with real
pytest. That is enough to settle the question the static rule raised, because
the defect is entirely in the parsing: the vendor's task fixtures cannot make
`re.search(r"(\\d+)\\s+passed", out)` stop matching a line the response printed.
Any report built on this must say exactly that, and `describe()` does.

Interpreter: the probes are plain strings and stay 3.9-compatible, but the
reproduction shells out to `sys.executable -m pytest`, so run it with the venv:

    .venv/bin/python src/rlverify/probes/code.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# The count a probe claims. Absurd on purpose: a grader returning this cannot be
# explained as the response having accidentally solved the task.
CLAIMED = 9999


class Probe:
    """One deliberately wrong response, plus why a grader paying it is a defect."""

    __slots__ = ("name", "why", "source")

    def __init__(self, name: str, why: str, source: str) -> None:
        self.name = name
        self.why = why
        self.source = source

    def __repr__(self) -> str:
        return "<Probe %s>" % self.name


# Four ways to get a success marker into a grader's captured output. They differ
# only in how they dodge pytest's capture, which is the part that cannot be
# predicted from source -- hence measuring rather than asserting.
STDOUT_MARKER_PROBES: List[Probe] = [
    Probe(
        "print_at_import",
        "plain print during collection; captured by pytest but echoed back into "
        "the report it writes to stdout",
        'print("%d passed")\n' % CLAIMED,
    ),
    Probe(
        "fd_write_at_import",
        "writes straight to file descriptor 1, under Python-level capture",
        'import os\nos.write(1, b"%d passed\\n")\n' % CLAIMED,
    ),
    Probe(
        "atexit_real_stdout",
        "defers to interpreter exit and writes to sys.__stdout__, after pytest "
        "has torn its capture down",
        "import atexit, sys\n"
        'atexit.register(lambda: sys.__stdout__.write("%d passed\\n"))\n' % CLAIMED,
    ),
    Probe(
        "stderr_at_import",
        "writes to stderr, which this grader concatenates onto stdout before "
        "parsing",
        "import sys\n"
        'sys.stderr.write("%d passed\\n")\n' % CLAIMED,
    ),
]

# Baselines. Without these a probe result means nothing: we have to know what
# the grader pays an honestly-wrong answer and an honestly-correct one.
EMPTY_RESPONSE = "# implements nothing\n"
CORRECT_RESPONSE = (
    "def add(a, b):\n    return a + b\n\n"
    "def sub(a, b):\n    return a - b\n\n"
    "def mul(a, b):\n    return a * b\n"
)
TEST_SUITE = (
    "import solution\n\n"
    "def test_add():\n    assert solution.add(2, 2) == 4\n\n"
    "def test_sub():\n    assert solution.sub(5, 3) == 2\n\n"
    "def test_mul():\n    assert solution.mul(3, 4) == 12\n"
)
TRUE_TEST_COUNT = 3


def _run_tier(workdir: Path, test_name: str) -> int:
    """Verbatim reproduction of concurrency-bench's grading step.

    Copied unchanged from `concurrency_bench/__init__.py::_run_tier` (env
    `concurrency-bench`, Environments Hub). Reproduced rather than imported so
    no third-party code is executed; the parsing is the whole defect, so this
    is a faithful test of it.

    The flaw: `out` mixes pytest's own report with anything the response printed
    while being imported at collection time, and `re.search` takes the *first*
    match.
    """
    test_file = workdir / "tests" / ("test_%s.py" % test_name)
    if not test_file.exists():
        return 0
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no",
             "-p", "no:cacheprovider", str(test_file)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return 0
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+)\s+passed", out)
    return int(m.group(1)) if m else 0


def grade(response_source: str) -> int:
    """Score one response through the reproduced grader. Returns its 'passed' count."""
    work = Path(tempfile.mkdtemp(prefix="rlverify_probe_"))
    try:
        (work / "tests").mkdir()
        (work / "tests" / "test_core.py").write_text(TEST_SUITE, encoding="utf-8")
        (work / "solution.py").write_text(response_source, encoding="utf-8")
        return _run_tier(work, "core")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_probes() -> Dict[str, Any]:
    """Measure every probe plus both baselines against the reproduced grader."""
    honest_wrong = grade(EMPTY_RESPONSE)
    honest_right = grade(CORRECT_RESPONSE)
    results = []
    for p in STDOUT_MARKER_PROBES:
        scored = grade(p.source)
        results.append({
            "probe": p.name,
            "why": p.why,
            "reward": scored,
            # The bar is not "scored > 0" but "beat a correct solution while
            # implementing nothing". That is the claim a vendor cannot wave away.
            "exploited": scored > honest_right,
        })
    return {
        "honest_wrong": honest_wrong,
        "honest_right": honest_right,
        "true_test_count": TRUE_TEST_COUNT,
        "results": results,
        "any_exploit": any(r["exploited"] for r in results),
    }


def describe(summary: Dict[str, Any]) -> str:
    """The paragraph that may go in front of a customer. Claims exactly what was tested."""
    winners = [r for r in summary["results"] if r["exploited"]]
    if not winners:
        return ("No probe outscored a correct solution against the reproduced "
                "grader. The static `stdout_spoof` smell is NOT confirmed here.")
    best = max(winners, key=lambda r: r["reward"])
    return (
        "Confirmed: reward is read from output the response controls.\n"
        "A submission implementing nothing scored %d, against %d for a fully "
        "correct solution (%d real tests).\n"
        "Mechanism: %s.\n"
        "Tested against a verbatim reproduction of the grader's own parsing step "
        "driven with real pytest; the environment package was not installed or "
        "imported."
        % (best["reward"], summary["honest_right"], summary["true_test_count"],
           best["why"])
    )


if __name__ == "__main__":
    s = run_probes()
    print("baseline  empty response  -> %d" % s["honest_wrong"])
    print("baseline  correct response-> %d   (%d real tests)"
          % (s["honest_right"], s["true_test_count"]))
    print()
    for r in s["results"]:
        print("  %-22s reward=%-6d %s"
              % (r["probe"], r["reward"], "EXPLOIT" if r["exploited"] else "-"))
    print()
    print(describe(s))

    failures = 0
    # Harness sanity first: if a correct solution does not score the real test
    # count, the reproduction is broken and every other number here is noise.
    if s["honest_right"] != s["true_test_count"]:
        failures += 1
        print("\nHARNESS BROKEN: correct solution scored %d, expected %d"
              % (s["honest_right"], s["true_test_count"]))
    if s["honest_wrong"] != 0:
        failures += 1
        print("\nHARNESS BROKEN: empty response scored %d, expected 0"
              % s["honest_wrong"])
    # The finding itself is under test. If no probe lands any more, the
    # `stdout_spoof` hit on concurrency-bench should be withdrawn, not quietly
    # kept because it was true once.
    if not s["any_exploit"]:
        failures += 1
        print("\nFINDING NOT REPRODUCED: withdraw the concurrency-bench claim")
    print("\n%d failure(s)" % failures)
    raise SystemExit(1 if failures else 0)
