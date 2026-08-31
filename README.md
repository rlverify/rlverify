# rlverify

**Audit RL environment graders for hackability before you train on them.**

The question `rlverify` answers for a given environment: *can a policy score well without solving
the task?* If nonsense scores, the grader is a bad training signal — and a model trained on it
learns to produce nonsense.

It fires deliberately-wrong responses (blank, gibberish, another task's gold answer, a response
that lists every candidate) at a grader and reports which ones were paid. Static analysis reads the
reward function without ever executing the environment's code; dynamic probes confirm a smell by
measuring a live grader.

## Install

```bash
pip install rlverify
```

The core — the static + corpus path that covers every public environment — is stdlib plus
`certifi`, so it runs on any Python 3.9+. Heavier pieces are extras:

```bash
pip install "rlverify[report]"   # PDF output (HTML needs nothing)
pip install "rlverify[envs]"     # adapters that import environment code; needs Python >=3.11
```

## Quickstart

```bash
# Audit one environment and write the report a customer would receive.
rlverify audit OWNER/NAME --out audit.html

# Render the corpus-wide document from a local store (read-only, no scan).
rlverify report --out corpus.html
```

`audit` exits `0` clean, `1` findings at or above `--fail-on` (default `high`), `2` on error, and
`3` when it **could not look** — no reward function found, grading delegated to a container, a
truncated listing. A coverage gap is never reported as a clean result.

## How it works

A deliberate **two-interpreter split**: the static/corpus half is 3.9-compatible and
dependency-light so it runs anywhere; only adapters that import environment code need a 3.11+
interpreter with `verifiers` installed.

The threat model is inverted from the obvious reading: **the probes are strings; the untrusted code
is the environment being audited.** `pip install` from an sdist runs code at build time, so the
default tier fetches source over HTTPS and analyses it as a syntax tree — it never imports.

```
sync    catalog -> detail -> file listings, resumable against the Hub
scan    fetch source, apply static rules, produce the headline counters
audit   one environment end to end: resolve -> sync on demand -> scan -> verdict
report  render an AuditReport (HTML / PDF / JSON) from the store
```

## The result it was built to produce

Run against a public catalogue of ~1,500 community RL environments (snapshot 2026-08-09):

- **1,367** non-research environments scanned; **17** carry a high-severity grader finding that
  survived an adversarial review pass — a second reviewer whose only job was to refute it — with
  three confirmed by *executing* a reproduction of the grader.
- **The headline is the denominator, not the finding count.** Only 39 of those environments are
  statically-readable code graders; the rest delegate reward to containers or remote sandboxes and
  are counted as `opaque` — a coverage gap, never a clean result. A source-level pass structurally
  cannot settle the code-grader question on this corpus, and the report says so.

Any published figure carries its snapshot date; the catalogue grows daily.

## Honesty invariants

The whole value of the tool is that its numbers survive scrutiny. These are load-bearing:

- **Static findings are smells, never confirmed exploits.** Every rule carries the mechanism and
  the dynamic probe that would confirm it.
- **Never fudge the denominator.** Parsed, parse-failed and unavailable are tracked separately; a
  file that could not be read is a coverage gap.
- **A rule with no denominator says nothing**, and a rule that flags a large share of the corpus is
  worthless — hit rates are reported and gated.
- **Refuse to report rather than report weakly.** Below the power threshold the tool declines to
  score.

## Status

Alpha. Built and gated: the Hub client, store, sync, the static scanner, the `verifiers` adapter
with an integrity guard, reproduction probes, the report layer, and the CLI. Not yet built: the
full dynamic probe library and the statistics layer. Not a Gym tool — Gym rewards are step-wise
physics with a different failure mode.

## License

Apache-2.0. See [LICENSE](LICENSE).
