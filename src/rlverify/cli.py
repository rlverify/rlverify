"""
`rlverify` -- the command line entry point.

Three verbs, deliberately:

    rlverify audit <env>    audit one environment and say what we can defend
    rlverify report         render the corpus document from the local store
    rlverify version

`sync` and the full corpus `scan` are NOT here. The console script installs into
whichever interpreter pip used, and a corpus sweep run under the 3.12 venv works
but is pointless; keeping them as `python3 -m rlverify.corpus.*` leaves one
documented way to run them, on the right Python. `probe` is not here because the
probe protocol does not exist yet, and a probe verb backed by an empty registry
would report "no exploits found" -- a clean result manufactured by not looking.

Interpreter: 3.9. This module must never import `rlverify.targets`, at module
level or lazily; that pulls `verifiers` (>=3.11) and would break
`pip install rlverify` on the system Python. Pinned by the gate below.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rlverify import audit as audit_mod
from rlverify.corpus import scan as scan_mod
from rlverify.corpus import store

__version__ = "0.1.0.dev0"


def _render(verdict: audit_mod.AuditVerdict, fail_on: str) -> None:
    """The summary a human reads. Says what we checked and what we could not."""
    print("\n" + "=" * 62)
    print("AUDIT  %s" % (verdict.env_key or "(unresolved)"))
    print("=" * 62)

    if verdict.error:
        print("ERROR: %s" % verdict.error)
        return

    if verdict.research_subject:
        print("NOTE: this environment's subject is reward hacking. It is excluded")
        print("      from corpus denominators, so the corpus-shaped tables below")
        print("      omit it. The verdict here is unaffected.\n")

    if verdict.files_scanned is not None:
        print("files scanned                 : %d of %d candidates"
              % (verdict.files_scanned, verdict.files_considered or 0))

    print("live findings                 : %d" % verdict.n_findings)
    for sev in ("high", "medium", "low"):
        if verdict.severity_counts.get(sev):
            print("  %-27s : %d" % (sev, verdict.severity_counts[sev]))
    if verdict.has_unrecognised_severity:
        # A live finding carries a severity this tool cannot classify (a typo,
        # a stale probe, a future severity never taught to SEVERITY_ORDER).
        # worst() silently drops it and exit_code ranks it above 'high' rather
        # than risk a false clean -- say so here too, or the number above
        # (n_findings) and the severity breakdown would quietly disagree.
        print("  %-27s : present -- ranked above 'high' for the exit code"
              % "unrecognised severity")
    print("  (findings withdrawn on manual review are already subtracted)")

    if verdict.inconclusive_reasons:
        print("\nwhy this environment cannot be cleared:")
        for reason in verdict.inconclusive_reasons:
            print("  - %s" % reason)

    code = audit_mod.exit_code(verdict, fail_on)
    print("\nverdict: %s (exit %d)" % (
        {0: "no finding at or above '%s', coverage complete" % fail_on,
         1: "findings at or above '%s'" % fail_on,
         2: "could not complete the audit",
         3: "INCONCLUSIVE -- coverage too poor to make a claim"}[code], code))
    if code in (audit_mod.EXIT_CLEAN, audit_mod.EXIT_FINDINGS):
        print("\nStatic findings are smells, not confirmed exploits. Every one")
        print("carries the mechanism and the dynamic test that would confirm it.")


def _cmd_audit(args: argparse.Namespace) -> int:
    from rlverify.static.hub_client import HubClient

    try:
        # A writer: this syncs and scans into the store, so it must not run
        # alongside a corpus sweep.
        conn = store.connect(args.db)
    except Exception as exc:
        print("could not open the store at %s: %r" % (args.db, exc), file=sys.stderr)
        print("a corpus sweep may be running; `audit` writes and cannot share "
              "the lock.", file=sys.stderr)
        return audit_mod.EXIT_ERROR

    client = HubClient(cache_dir=args.cache, rate_hz=args.rate_hz,
                       offline=args.offline)
    verdict = audit_mod.audit_environment(conn, client, args.env)
    _render(verdict, args.fail_on)
    code = audit_mod.exit_code(verdict, args.fail_on)

    if args.out and verdict.env_key and verdict.scanned:
        try:
            path = scan_mod.write_document(conn, args.out,
                                           scope_keys=[verdict.env_key])
            print("\nwrote %s" % path)
        except Exception as exc:
            # Asked for an artifact and did not get one. The verdict is already
            # on stdout, so nothing is lost -- but exiting 1 with no file is
            # worse than a loud 2.
            print("\ncould not write %s: %r" % (args.out, exc), file=sys.stderr)
            if args.out.lower().endswith(".pdf"):
                print("PDF needs reportlab (`pip install reportlab`); "
                      "`--out FILE.html` needs nothing installed.", file=sys.stderr)
            conn.close()
            return audit_mod.EXIT_ERROR

    conn.close()
    return code


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        # Readonly, so this is safe to run while a sweep is in progress.
        conn = store.connect(args.db, readonly=True)
    except Exception as exc:
        # Most likely `rlverify report` before anything has ever been synced
        # or audited -- connect(readonly=True) never creates the file, so a
        # first run hits this rather than a working, empty store.
        print("could not open the store at %s: %r" % (args.db, exc), file=sys.stderr)
        return audit_mod.EXIT_ERROR

    scan_mod.report(conn, scan_mod.readonly_counters(conn))
    if args.out:
        try:
            print("\nwrote %s" % scan_mod.write_document(conn, args.out,
                                                         scope_keys=None))
        except Exception as exc:
            print("\ncould not write %s: %r" % (args.out, exc), file=sys.stderr)
            conn.close()
            return audit_mod.EXIT_ERROR
    conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlverify",
        description="Audit RL environment graders for hackability before you "
                    "train on them.",
        epilog="Corpus sweeps stay as modules, on the system Python:\n"
               "  PYTHONPATH=src python3 -m rlverify.corpus.sync --pass catalog\n"
               "  PYTHONPATH=src python3 -m rlverify.corpus.scan",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="rlverify " + __version__)
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser(
        "audit", help="audit one environment",
        description="Audit one environment. Syncs it from the Hub first if the "
                    "local store does not have it. Writes to the store, so do "
                    "not run this alongside a corpus sweep.")
    a.add_argument("env", help="owner/name, or a bare name if it is unambiguous")
    a.add_argument("--out", default=None,
                   help="also write the audit document here; the format comes "
                        "from the extension (.pdf, .html or .json)")
    a.add_argument("--fail-on", default="high", choices=audit_mod.FAIL_ON_CHOICES,
                   help="lowest severity that exits 1 (default: high). 'never' "
                        "suppresses that exit only -- a coverage gap still "
                        "exits 3")
    a.add_argument("--db", default=store.DEFAULT_DB)
    a.add_argument("--cache", default="~/.cache/rlverify/hub")
    a.add_argument("--rate-hz", type=float, default=4.0)
    a.add_argument("--offline", action="store_true",
                   help="serve every request from the blob cache; a miss is a "
                        "failure rather than a fetch")
    a.set_defaults(func=_cmd_audit)

    r = sub.add_parser(
        "report", help="render the corpus document from the local store",
        description="Read-only render of what has already been scanned. Safe to "
                    "run during a sweep.")
    r.add_argument("--out", default=None,
                   help="write the document here (.pdf, .html or .json)")
    r.add_argument("--db", default=store.DEFAULT_DB)
    r.set_defaults(func=_cmd_report)

    v = sub.add_parser("version", help="print the version")
    v.set_defaults(func=lambda args: (print("rlverify " + __version__), 0)[1])

    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Exit codes: 0 clean, 1 findings, 2 error, 3 inconclusive."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return audit_mod.EXIT_ERROR
    return args.func(args)


if __name__ == "__main__":
    import os
    import tempfile

    failures = 0

    def expect(label: str, got: int, want: int) -> None:
        global failures
        if got != want:
            failures += 1
            print("FAIL  %-46s got %d want %d" % (label, got, want))
        else:
            print("ok    %-46s %d" % (label, got))

    expect("version", main(["version"]), 0)
    expect("no subcommand prints help and errors", main([]), audit_mod.EXIT_ERROR)

    for argv, label in ((["--help"], "--help"), (["audit", "--help"], "audit --help")):
        try:
            main(argv)
            failures += 1
            print("FAIL  %s did not exit" % label)
        except SystemExit as exc:
            if exc.code:
                failures += 1
                print("FAIL  %s exited %r" % (label, exc.code))
            else:
                print("ok    %-46s SystemExit(0)" % label)

    try:
        main(["nonsense"])
        failures += 1
        print("FAIL  unknown verb did not exit")
    except SystemExit as exc:
        expect("unknown verb", int(exc.code or 0), 2)

    # The end-to-end form of the failure mode: an environment nobody has heard
    # of, against an empty store, with the network closed. Anything but a
    # nonzero exit here means the tool can clear a grader it never saw.
    tmp = tempfile.mkdtemp(prefix="rlverify_cli_gate_")
    code = main(["audit", "no-such-environment-anywhere", "--offline",
                 "--db", os.path.join(tmp, "empty.sqlite"),
                 "--cache", os.path.join(tmp, "cache")])
    expect("unknown env, empty store, offline", code, audit_mod.EXIT_ERROR)

    # The interpreter invariant, asserted rather than trusted. Importing
    # `targets` pulls `verifiers`, which needs >=3.11 -- if that ever creeps in,
    # `pip install rlverify` breaks on the 3.9 the corpus half is built for.
    leaked = [m for m in sys.modules if m.startswith("rlverify.targets")]
    if leaked:
        failures += 1
        print("FAIL  cli imported the 3.11+ half: %s" % ", ".join(leaked))
    else:
        print("ok    %-46s clean" % "no rlverify.targets in sys.modules")

    print("\n%d failure(s)" % failures)
    raise SystemExit(1 if failures else 0)
