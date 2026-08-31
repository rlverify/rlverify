# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rlverify` audits **RL environment graders for hackability before anyone trains on them**.
The question it answers for a given environment: *can a policy score well without solving the task?*

Concretely it fires deliberately-wrong responses (blank, gibberish, another task's gold
answer, a response listing every candidate) at a grader and reports which ones were paid.
If nonsense scores, the grader is a bad training signal.

The corpus target is Prime Intellect's Environments Hub — 1454 public environments, none of
which have ever been audited. The intended first output is a corpus-wide finding, not a product
launch; the tool exists to produce that number credibly.

This project is **independent of the quant projects in `~`** despite living under the same home
directory. It shares no code, no data, and no config with them. `~/CLAUDE.md` does not apply here.

## The two-interpreter split (read this first)

This is the single most confusing thing about the repo. There are two Pythons and they are not
interchangeable:

| Interpreter | Use for | Why |
|---|---|---|
| `python3` (3.9.5, system) | `static/`, `corpus/` — everything that fetches and parses source | stdlib + certifi only; runs anywhere |
| `.venv/bin/python` (3.12.13, uv) | `targets/`, anything importing `verifiers` | `verifiers` requires `>=3.11,<3.14` |

The static/corpus half is deliberately kept 3.9-compatible and dependency-light so the corpus
sweep runs on whatever interpreter is present. Only adapters that import environment code need
the venv. `from __future__ import annotations` everywhere in the 3.9 half.

Running a `targets/` module under `python3` fails on import; running a `corpus/` sweep under the
venv works but is pointless.

## Commands

```bash
# --- CLI (any Python 3.9+; the installed console script. `rlverify.cli`'s own
#     __main__ is a gate, not an alt entry point -- `-m rlverify.cli audit ...`
#     runs the smoke test and ignores your args) ---
rlverify audit OWNER/NAME[@VERSION] --out FILE.html
    # Syncs the env from the Hub on demand if the store lacks it (~15s catalog
    # sync), so this is the `pip install rlverify && rlverify audit ...` path,
    # not only the path over an already-swept corpus. Writes reviewed=0 findings
    # to --db, so never run it alongside a corpus sweep -- same store-writer rule
    # as `scan`. --out's extension picks the format: .html needs nothing
    # installed, .pdf needs `pip install rlverify[report]`, .json is the model.
    # Exit codes: 0 clean, 1 findings at/above --fail-on (default high), 2 error
    # (unresolved or ambiguous name, store busy, --out failed), 3 inconclusive.
    # Read the "could not look" invariant below before reading 0 as clean.

rlverify report --db ~/.cache/rlverify/corpus.sqlite --out corpus.html
    # Read-only render of what the store already holds -- no scan, safe during a
    # sweep. The corpus-wide version of the renderer `audit` calls scoped to one.

rlverify --version
rlverify audit --help    # --db, --cache, --rate-hz, --offline, --fail-on

# Default --db is ~/.cache/rlverify/corpus.sqlite (store.DEFAULT_DB) for both
# verbs. `sync` and the full corpus `scan` are deliberately NOT CLI verbs: the
# console script installs into whichever interpreter pip used, and a multi-hour
# sweep under that interpreter works but is pointless. See cli.py's docstring.

# --- corpus (system python3, from src/) ---
cd src
PYTHONPATH=. python3 -m rlverify.corpus.sync --pass catalog        # ~15 requests, 15s
PYTHONPATH=. python3 -m rlverify.corpus.sync --pass detail,files   # ~4000 requests, slow
PYTHONPATH=. python3 -m rlverify.corpus.sync --stats-only
PYTHONPATH=. python3 -m rlverify.corpus.scan                       # full static scan, ~30min warm
PYTHONPATH=. python3 -m rlverify.corpus.scan --report-only         # readonly; safe during a sweep

# Re-judge only what a rule change invalidates, instead of an hour of re-scan.
PYTHONPATH=. python3 -m rlverify.corpus.scan --only "some-env"

# Replay a *parsing* fix over blobs already on disk. Zero HTTP; a cache miss is
# recorded as a failure rather than silently fetched.
PYTHONPATH=. python3 -m rlverify.corpus.sync --pass detail --offline --refresh-detail

# Persist manual review verdicts so a re-scan preserves them (dry run by default).
PYTHONPATH=. python3 -m rlverify.corpus.apply_review --commit

# The deliverable. HTML needs nothing installed; PDF needs `pip install reportlab`.
PYTHONPATH=. python3 -m rlverify.corpus.scan --only "some-env" --out audit.pdf
PYTHONPATH=. python3 -m rlverify.corpus.scan --report-only --out corpus.html

# --- per-module smoke tests: every module has a __main__ that self-checks ---
python3 rlverify/static/ast_rules.py     # positive + negative/withdrawn controls; nonzero on regression
python3 rlverify/static/hub_client.py    # all 3 endpoints + cache short-circuit (hits the network)
python3 rlverify/corpus/store.py         # idempotency + upsert invariants

# --- adapter + probes (venv, run from the repo root) ---
.venv/bin/python src/rlverify/probes/code.py   # withdraws the concurrency-bench claim if it stops reproducing

# Re-measure every claim in reports/disclosure-draft.md. Run before sending it anywhere.
python3 src/rlverify/probes/disclosed.py          # Tier 0: no install, no import
.venv/bin/python src/rlverify/probes/disclosed.py --tier1   # also imports the published module
.venv/bin/python -c "import verifiers; print(verifiers.__version__)"
uv pip install -e ".[stats,envs,dev]"
```

There is **no test suite yet**. `pyproject.toml` declares `testpaths = ["tests"]` but `tests/`
does not exist; the `__main__` smoke tests are the current safety net. Seven of them exit nonzero
on failure and are the real regression gates: `ast_rules.py` (rules), `probes/code.py` (the
concurrency-bench exploit still reproduces), `probes/disclosed.py` (every claim in the disclosure
still reproduces, the graders it quotes have not been edited under it, **and the corpus counts the
covering note quotes reconcile against the store and add up**), `report/model.py` (headline matches
the store), `store.py` (bare asserts), `audit.py` (exit-code precedence, severity comparison, name
resolution — the load-bearing "never 0 on an unscanned env" property is proven by mutation), and
`cli.py` (verb dispatch, exit-code plumbing, and **no `rlverify.targets` in `sys.modules` after a
full run** — the CLI is the 3.9 half and must never pull `verifiers`). `hub_client.py`'s `__main__`
now runs deterministic offline asserts on the walk contract (a failed listing must set `incomplete`
without a false `truncated_by`) before its live-network smoke test, so a regression in that contract
exits nonzero with no network — the offline half is a gate; the network half after it still only
prints its failure count and is not CI-safe.

Long sweeps: do not pipe them through `tail`/`head`. It buffers until EOF and the run looks hung.
Redirect to a file and poll the SQLite store instead.

## Architecture

Pipeline: **sync** (catalog → detail → file listings) → **scan** (fetch source, apply static
rules) → **targets** (dynamic probing of live graders).

```
static/hub_client.py   read-only Hub client; stdlib urllib + certifi, token-bucket throttle,
                       retry, content-addressed blob cache
static/ast_rules.py    source-level rules over reward functions; parses text, executes nothing
corpus/store.py        SQLite schema + all write helpers; the concurrency contract lives here
corpus/sync.py         three resumable passes against the Hub
corpus/scan.py         fetch + static-scan the corpus, produce the headline counters
audit.py               one-environment pipeline: resolve -> sync on demand -> scan -> verdict;
                       `exit_code()` is pure, so "never 0 on an unscanned env" is gated exhaustively
cli.py                 argparse over audit/report/version; the 3.9 half, never imports targets/
targets/base.py        GraderTarget / TaskInstance / GradeResult / Capabilities
targets/verifiers_v0.py  adapter for `verifiers` Rubrics (~98% of the corpus depends on it)
```

`audit.py` and `cli.py` sit at the `src/rlverify/` top level, beside `targets/`/`static/`/`corpus/`.
`cli.py` is the thinnest layer — it imports `audit` and `corpus.scan` only, never `targets`, so
`pip install rlverify` on the system Python 3.9 does not drag in `verifiers` (>=3.11). Its own gate
pins this: no `rlverify.targets` in `sys.modules` after a full run of every verb.

### Hub API (verified live, unauthenticated, no install required)

```
GET /api/v1/environmentshub/?limit=&offset=          -> {total_count, data:[...]}
GET .../environmentshub/{owner}/{name}/@{ver}        -> {data:{sha256, wheel_url, metadata}}
GET .../environmentshub/{owner}/{name}/@{ver}/inspect[?path=]
        no path -> directory listing, each entry carrying `content_hash`
        path    -> {content, encoding, truncated}
```

Server-supplied `sha256`/`content_hash` are the cache keys, so invalidation is exact and free.
The list endpoint paginates by **offset**, so a publish mid-sweep can both duplicate and *skip*
records — `sync_catalog` counts distinct keys and warns rather than trusting the yielded count.

### Store concurrency contract

SQLite allows one writer. Long sweeps hold write transactions across slow HTTP calls, so:

- **`store.connect(readonly=True)` is genuinely read-only.** It opens the file through a
  `file:...?mode=ro` URI and raises `sqlite3.OperationalError` on any write — INSERT, UPDATE,
  DELETE, or DDL alike. This *replaced* the old contract, it did not extend it: the previous
  "readonly" connection opened the store read-write and merely skipped DDL, so a caller could
  believe a shared-with-a-writer connection couldn't modify the store, and it could. `rlverify
  report` and `readonly_counters()` are documented "safe during a sweep"; that now holds because
  the connection enforces it, not because every caller happens only to read.
- **All long-running passes commit via `store.commit(conn)`**, which retries on lock. Bare
  `conn.commit()` will eventually kill a multi-hour sweep.
- `_ensure_schema` only runs DDL when a table is genuinely missing, and retries with backoff.
- Do not run two writing passes concurrently. Sequence sync then scan.

Every write is an upsert on natural identity, so all passes are idempotent and resumable —
kill mid-sweep and re-run.

### `verifiers` v0.2.1 API facts (undocumented, cost real time)

1. `Rubric.score_objects()` reads `prompt`, `answer`, `info` from the **top level** of `State`,
   not `state["input"]`. Wire it wrong and `answer` silently becomes `""`, every grader returns
   0.0, and the environment looks immaculate. **This failure mode manufactures clean results** —
   it is the most dangerous bug possible in an auditing tool.
2. `Rubric.score_rollout(state)` returns `None`. It mutates state in place; read `state["reward"]`.
3. `Rubric._call_individual_reward_func` wraps calls in `try/except -> 0.0`. A grader that raises
   is indistinguishable from one that scored zero.
4. `Rubric.teardown()` is not optional — `MathRubric` owns a `ProcessPoolExecutor` and a corpus
   sweep that skips teardown leaks processes.

`VerifiersV0Target.self_check()` defends against (1) and (3): if an environment's own gold
answers do not score, the verdict is **`inconclusive`, never `clean`**.

## Honesty invariants

The whole value of this tool is that its numbers survive scrutiny, and publishing a wrong
accusation against 1454 people's work is the failure mode that ends the project. These are
load-bearing, not style:

- **Never fudge the denominator.** `parsed`, `parse_failed` and `unavailable` are tracked and
  reported separately. A file we could not read is a coverage gap, not a clean result.
- **Static findings are smells, never confirmed exploits.** Every rule carries `why` (the
  mechanism) and `probe` (the dynamic test that would confirm it). The scan report says so
  explicitly; keep that text.
- **Every finding must be human-verifiable in seconds** — the literal response and reward are
  always shown. The `findings` table has `reviewed`/`review_verdict` columns so manual review
  before publishing is structural, not remembered.
- **Refuse to report rather than report weakly.** Below ~10 clusters, emit `low_power` and print
  no confidence interval. Refusing to score is a feature.
- **A rule that flags half the corpus is worthless.** `no_answer_marker` originally matched the
  bare token `in`, hit 53% of modules, and was tightened to require positive evidence. Check any
  new rule's corpus-wide hit rate before trusting it. `scan.report()` now prints a per-rule hit
  rate and flags anything over 25%, so this is structural rather than remembered.
- **A rule with no denominator says nothing.** "Zero code-grader findings" was meaningless until
  the scan counted how many graders execute code at all: it was 1 of 147. Every rule family needs
  its own denominator reported next to it.
- **Execution we cannot see is a coverage gap, not a clean result.** Graders that hand the response
  to a container or remote sandbox are counted as `opaque`, never as clean. Same accounting as
  `parse_failed`.
- **Every rule earns its keep against real source, not against its own control.** The code family's
  first two real hits were both false positives and both were withdrawn (see below). Withdrawn
  findings are kept as negative controls so they cannot silently return.
- **An audit that could not look does not return 0.** `rlverify audit` exits `3` — never `0`,
  never the findings exit `1` — when no reward function was located, nothing was readable, grading
  is delegated to a container or remote sandbox, a listing was truncated, or a per-environment file
  budget was hit before the scan finished. `--fail-on never` suppresses only the *findings* exit;
  it does not and must not touch this one. Same principle as "execution we cannot see is a coverage
  gap" above, now enforced at the process exit code rather than only in a report table.
- **A failed measurement is never recorded as a measurement.** `record_files` takes a tri-state
  `truncated`: `None` means the walk did not complete, and `listing_truncated` is then left
  untouched rather than stamped `0`. A failed walk returns no entries and no truncation marker, so
  writing "measured, not truncated" from it turns *never measured* into the strongest coverage
  claim — and `walk_files_ex` returns `(entries, truncated_by, incomplete)` for the same reason:
  `truncated_by` is a cap we chose to stop at, `incomplete` is a directory we could not read, and
  collapsing them records a partial listing as a complete one. The catalog refresh surfaced the
  same shape in `detail_ok=0`: 6 genuine `fetch_error`s versus 26 never-attempted, and reporting
  the total as "no detail" would book the 26 as failures.
- **Every summary count reconciles against the store, and its rows add up.** The covering note's
  high-severity table once summed to 34 against a stated 30 because one row carried an all-severity
  count under a high-severity header. `probes/disclosed.py` section 4 now asserts the quoted table
  against SQL *and* that confirmed+withdrawn+unresolved equals the reviewed total — the second
  check catches a basis swap without needing to know which row is wrong. Any published count needs
  its snapshot date, because the scan date and the catalog date are not the same date.

## Scope boundaries

- **Gym is explicitly out.** Gym rewards are step-wise physics with a different failure mode
  (policy exploits simulator dynamics, not answer parsing) and need full rollouts. Say "not a
  Gym tool" rather than half-supporting it.
- **v0.1**: `CallableTarget` + `VerifiersV0Adapter`, null/extraction/json probe families,
  cluster bootstrap + BH-FDR + power, JSON report + CI exit codes, static AST + corpus batch.
- **Deferred to v1.0**: LLM adaptive fuzzing, code/tamper/judge probe families, Docker isolation
  tier, `verifiers` v1 adapter (its `Trace`/`Harness` model needs real reverse-engineering),
  HUD adapter, SARIF, any web UI.

## Threat model

Inverted from the obvious reading: **the probes are strings; the untrusted code is the
environment being audited.** `pip install` from an sdist executes code at build time.

- Tier 0 (default): fetch source over HTTPS, AST-analyse, **never import**. This is what runs
  across all 1454.
- Tier 1: subprocess with `resource` limits. Document as hardening, *not* a security boundary.
- Tier 2: Docker `--network=none --read-only --cap-drop=ALL`, prebuilt wheels only, never sdist.

## Current state

Built and verified: Hub client, store, sync, static scanner (14 rules, 15 positive controls,
15 negative/coverage/withdrawn controls — 30 assertions, all green), `verifiers` adapter with
integrity guard, one reproduction probe, and review-verdict persistence.

### Static rules

Seven answer-parsing rules (`substring_containment`, `unanchored_regex`, `except_returns_reward`,
`eval_on_output`, `subprocess_no_timeout`, `wide_numeric_tolerance`, `no_answer_marker`) and seven
code-grader rules (`visible_test_only`, `stdout_spoof`, `exit_code_only`, `assert_disabled`,
`test_file_writable`, `no_test_isolation`, `timeout_evade`).

The code family is gated on **code-grader context**: the reward function must actually execute
response-derived code. Ungated, its shapes (write a file, read stdout, check an exit status) are
ordinary Python and would flag most of the corpus. `_effective_nodes()` inlines locally defined
helpers the grader calls (2 levels) — real graders split `reward() → run_tests() → _exec()`, and
this is what catches one corpus environment whose `exec()` sits in a helper the older rules
never looked at.

Three gate details that each cost a false positive on real Hub source:
- `compile()` is **not** execution. It builds a code object and runs nothing; graders use it as a
  syntax check before shipping code to a remote evaluator (`long-code-edit`).
- `stdout_spoof` is suppressed when the grader also compares output to a gold value — there the
  marker test is a guard and the reward still turns on matching the expected result.
- `visible_test_only` requires the tests to be **interpolated into prompt text**, not merely
  co-named with it. `prompt_to_tests[k] = row["test_cases"]` is a lookup keyed by prompt.

### The empirical result — read before adding rules

**Full corpus, 2026-08-08**, after the M0 measurement fixes. Superseded numbers from the earlier
top-150 and code-signal samples are in `reports/baseline-2026-08-08.log`; the before/after is in
`reports/m0-before-after.md`.

```
environments scanned          : 1366   (non-research)
  with a detected reward func : 1175
  that execute response code  :   39   <- the denominator for the code rules
  execution delegated, opaque :   40   <- coverage gap, NOT a clean result
  with >=1 static finding     :  295   (25.1%)
coverage gaps now measured, previously invisible:
  envs scanned in part        :  100   (5565 files over the per-env budget)
  envs with truncated listing :   28
HIGH severity envs            :   18   (17 confirmed on manual review)
```

**M0 corrected the instrument and did not overturn the conclusion.** The skip-list excluded
`tasks/*/tests/test_outputs.py` and `compute_reward*.py` as if they were a package's own unit
tests, when they are the graders; fixing it made 147 fixtures readable (contractbench 0→27 files,
skillsbench 0→100) and moved code graders 36→39 — not to hundreds. The reason is a sharper version
of the original claim: `test_outputs.py` compares an output artifact against a shipped
`expected.json` rather than executing the response, so the reward logic became visible without
being response-execution.

Three code-family findings, every one read against the real source:
- `concurrency-bench` — `stdout_spoof`, **high, CONFIRMED BY PROBE** (`probes/code.py`). A
  response implementing nothing scores **9999 against 3 for a fully correct solution**; in the
  real reward formula `0.2 + 0.5*f2p_frac + 0.3*edge_frac` that is a reward near 1700 instead of
  1.0. Measuring corrected the rule's own stated mechanism: a plain `print`, an `os.write(1, ...)`
  and a stderr write all score **0**, because pytest's capture eats them under `-q --tb=no`. Only
  deferring to interpreter exit and writing to `sys.__stdout__`, after capture is torn down, lands.
  This is why probes exist — the static `why` was directionally right and mechanically wrong.
  The defect: the grader scores a tier with
  `re.search(r"(\d+)\s+passed", proc.stdout + proc.stderr)`, and the response's own module is
  imported into that same pytest process at collection time. Any `N passed` the response can get
  into the stream is taken as the tier's score, and the inflated count also clears the
  `p2p < totals["contract"]` gate.
- `code-repair-env` and one further environment — `timeout_evade`, medium. Response code `exec`'d
  in-process with no wall-clock bound. `code-repair-env` also carries a genuine `eval_on_output`
  high: it `exec`s the response concatenated with the test code and pays 1.0 for not raising.

**The headline is the denominator, not the findings.** Of 1366 non-research environments, only 39
are statically-readable code graders. The environments the weak-test-suite literature is actually
about — `terminal-bench`, `kernelbench`, `mle-bench`, `frontier-swe`, `skillsbench` — are `opaque`:
they delegate to Docker/tmux/remote sandboxes, and the commands that decide reward live in task
data (a `run_tests.sh`, an image, a dataset row), not in Python source. **A source-level pass
structurally cannot settle the code-grader question on this corpus** — a conclusion that now rests
on a denominator the tool's own configuration is not biasing.

**Answer-parsing looseness is confirmed, and it is where the findings are.** An adversarial
verification pass over every high-severity finding (read real source, then a second reviewer whose
only job was to refute) reviewed **30 findings across 26 environments**. Mind the unit — mixing
these two put a wrong count into a draft disclosure:

| | findings | environments |
|---|---|---|
| confirmed | 20 | **17** |
| withdrawn | 9 | 8 |
| unresolved | 1 | 1 |

Both columns add to the reviewed total (20+9+1 = 30 findings, 17+8+1 = 26 environments), and
that 30 is *every* high-severity finding in the scan — review coverage of the tier is complete.
**13 findings / 11 environments** is the withdrawal count across *all* severities (the 9 high
plus 4 low); it is a different basis and putting it in a high-severity row makes the rows sum to
34 against a stated 30. That error reached a draft disclosure.

Quote environments; it is the conservative direction (3 confirmed environments carry >1 finding).
**18** is a third number and not a synonym for either: it is what `report/model.py` prints —
environments with a *non-withdrawn* high finding, so it includes the 1 unresolved case. Confirmed
environments went from 1 to 17; three are confirmed by execution. Verdicts live in
`findings.reviewed` / `review_verdict` and the report subtracts withdrawals from every count.
Full detail: `reports/high-severity-verification.json` (whose own `confirmed`/`killed`/`unclear`
lists are the *first* pass — 8 of its 9 `unclear` were later resolved to confirmed).

**Two of the three execution-confirmed defects, as templates for what a real finding looks like:**
- `code-repair-env` — the gold `answer` is a declared parameter that is never read, and the
  assertion the response must survive is printed verbatim in the prompt. A hardcoded `return 5`
  scores 1.0, the same as a correct fix.
- `traverse-tasks` — the response is `exec`'d with the grader's own `__builtins__`, so it can
  rebind `builtins.round` before the tally; `round(passed/total, 4)` then returns whatever it
  likes. Measured 0.1667 → 1.0, and the patch persists process-wide, inflating every later rollout
  graded in that process.

**`except_returns_reward` went 0-for-6 at high severity and was rewritten.** Every hit was a
consolation value, a zero-weighted metric, or a documented neutral — the rule tested `v > 0` and
treated *positive* as *good*. It now compares the handler's value against the function's own
success returns, skips values also returned on non-exception paths, and skips functions registered
with a literal weight of 0. All six shapes are pinned as `withdrawn:*` negative controls, and
`except_still_high` pins the other direction so the gates cannot quietly retire the rule.

`eval_on_output` is severity-conditioned the same way: `high` only when the evaluated argument is
response-derived. It used to fire `high` on `eval(case_data, results, base)` in two medical-agent
benchmarks, where the evaluated string ships with the benchmark and the response never
reaches it — 3 of its 4 hits, each a false accusation against published work. Those are now
`medium`; `code-repair-env`'s `exec(full_code, ...)` correctly stays `high`.

Built since: `probes/code.py` and `probes/disclosed.py` (reproduction probes, both real gates —
`disclosed.py` re-measures every claim in the disclosure and fails if a quoted grader has been
edited since the snapshot),
`corpus/apply_review.py` (persists review verdicts), and `report/` — the deliverable.
`report/model.py` builds an `AuditReport` from the store (stdlib, 3.9), `render_html.py` and
`render_pdf.py` render it; `scan --out audit.pdf|.html` writes it. **The summary counts come from
SQL over the scope, never from the rendered target list** — deriving them from the list made a
1367-environment report announce "no high-severity findings" over a corpus holding 18, because
detail is omitted above 25 targets. `model.py.__main__` pins that against the store and is a real
gate. **The CLI is built.** `rlverify audit <env>` and `rlverify report` exist, are wired to
`pyproject.toml`'s `rlverify = "rlverify.cli:main"`, and are gated (`audit.py`, `cli.py`). Building
it surfaced three latent `report/` bugs its own smoke test never hit because it never varied scope:
an empty scope (`scope_keys=[]`) falling back to a corpus-wide report, `build_report`'s
`research_subject` count hardcoded to `0`, and no exit code able to say "this audit could not look."
The `rlverify.probes` entry point was **withdrawn, not stubbed** — it pointed at a
`probes/registry.py:builtin_probes` that does not exist, and declaring a plugin API before its
contract exists is the same error as reporting an unmeasured denominator; it returns in one line
once `probes/base.py` defines the contract. Every declared entry point now resolves, and the
sdist/wheel build is fixed (`[tool.hatch.build.targets.sdist]` — an unanchored default swallowed a
worktree venv whose absolute symlink made the tarball unextractable), so `dist/` is publishable
once rebuilt. Still not built: the rest of the probe library and the statistics layer (`stats/`).

Known wart: `scan.py` deletes only `reviewed=0` findings so human verdicts survive a re-scan.
A withdrawn finding whose rule no longer fires therefore keeps its stale `severity` value
(3 rows today). The report excludes them by `review_verdict`, so no count is wrong; the row is.

Corpus (scan snapshot 2026-08-09): 1505 catalogued, 1498 with detail, 1460 file-listed,
**45,237 files indexed, 12,427 Python**; 1367 non-research environments scanned.
`requires_python` is populated for ~1446 — no environment requires >3.12, so one Python 3.12
image covers the whole corpus. **A catalog pass on 2026-08-13 read 1530** (1454 → 1505 → 1530;
roughly 6/day). Catalog is the only cheap pass, so growth shows up first as unlisted rows: 70
now, and the 25 new ones are in no scanned or findings figure. Every published number needs its
snapshot date attached, and the scan date and the catalog date are not the same date.

`detail_ok=0` is **two states, not one**, and a catalog pass widens the gap between them: 32
rows today, of which 6 carry a real `fetch_error` and 26 have simply never been fetched.
Reporting the total as "no detail" books 26 unattempted fetches as failures — the same
accounting error as recording a failed measurement as a successful one, pointed the other way.

**`environments.sha256` is the sdist hash, not the wheel's** (verified by fetching a wheel and
hashing it; `metadata.original_filename` is a `.tar.gz` corpus-wide). The Hub publishes no wheel
hash, so this must never be used as a wheel integrity check — it would fail 100% of the time.

Two walk bounds, both reported rather than silent: `MAX_WALK_ENTRIES` (4000 files) and
`MAX_WALK_REQUESTS` (400 directory listings). The second is the one that governs wall-clock — the
walk pays one request per directory whether or not it holds files, and the Hub's inspect endpoint
serves ~0.14–0.4/sec regardless of our 4 Hz throttle. An unbounded-by-requests walk over the 30
largest repos burned 5410 requests in 11 hours and finished nothing. The walk is breadth-first so
shallow files (`tasks/<task>/tests/…`) are reached before the budget is spent in a vendored
subtree; `sync_files` commits per environment so a kill loses at most one.

See `NEXT.md` for the current task.
