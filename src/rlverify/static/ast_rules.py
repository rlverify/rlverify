"""
Static analysis of RL environment reward functions. Source-level only: this
module parses text and never imports, installs, or executes anything.

Scope and honesty
-----------------
A static hit is a *smell*, not a proven exploit. `answer in completion` really
is gameable by a response that contains every candidate answer, but whether a
policy would ever emit that depends on the task. So every rule carries:

  - `why`   : the mechanism that makes it gameable
  - `probe` : the dynamic probe that would confirm it
  - severity: how confident we are that the smell is a real defect

Nothing here is reported as a confirmed finding. The static pass exists to
(a) cover 100% of the corpus for free and (b) rank which environments deserve
the expensive dynamic pass.

Parse coverage is tracked explicitly. Hub environments target Python >=3.11
while this may run on an older interpreter, and a file we failed to parse is a
coverage gap, not a clean result -- it is counted and reported as such.
"""
from __future__ import annotations

import ast
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set

# Names that suggest a function computes reward. Deliberately broad; precision
# comes from also requiring the function to look like it returns a score.
REWARD_NAME_RE = re.compile(
    r"(reward|score|grade|grader|verify|verifier|check_answer|correct|judge|rubric)",
    re.IGNORECASE,
)

# Names that denote the model's own output, as opposed to the reference answer.
# Several rules below are only sound when they are pointed at the response --
# "does the grader compare X to the response" is a defect, "does the grader
# compare two of its own values" is just arithmetic.
RESPONSE_NAME_RE = re.compile(
    r"(completion|response|prediction|generation|model_out|submission|solution|"
    r"patch|candidate|snippet|attempt|\btext\b|\bcode\b|\bprogram\b|\bscript\b|"
    r"\boutput\b|\bcontent\b)",
    re.IGNORECASE,
)
# ...and these merely contain a response word by accident.
NOT_RESPONSE_RE = re.compile(
    r"(returncode|status_code|exit_?code|error_code|capture_output|output_dir|"
    r"output_path|output_file|content_hash|content_type)",
    re.IGNORECASE,
)

# Calls that leave the process. An `except -> positive reward` around one of
# these is an availability decision, not a gameable grader (see visit_Try).
NONLOCAL_CALL_RE = re.compile(
    r"(judge|llm|openai|anthropic|litellm|openrouter|together_ai|groq|gemini|"
    r"claude|\bgpt|chat_?completion|acompletion|requests|httpx|urlopen|urllib|"
    r"aiohttp|websocket|session|boto3|redis|psycopg|sqlalchemy|subprocess|"
    r"docker|sandbox|\be2b\b|modal|fetch|http|api_|_api|client)",
    re.IGNORECASE,
)


def _expr_text(node: ast.AST) -> str:
    """Lowercased identifier / attribute / string-literal text in an expression.

    A deliberately coarse view of "what names does this touch". Rules use it for
    heuristics only -- never to decide that something is a confirmed defect.
    """
    parts: List[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            parts.append(n.id)
        elif isinstance(n, ast.Attribute):
            parts.append(n.attr)
        elif isinstance(n, ast.arg):
            parts.append(n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            parts.append(n.value)
        elif isinstance(n, ast.keyword) and n.arg:
            parts.append(n.arg)
    return " ".join(parts).lower()


def _is_response_expr(node: ast.AST) -> bool:
    """True when an expression plausibly carries the model's own output."""
    t = _expr_text(node)
    if not t or NOT_RESPONSE_RE.search(t):
        return False
    # Identifiers join words with underscores, which defeats `\b`: `full_code`
    # and `patched_code` are the response under its two commonest names in real
    # code graders, and matching only the bare token missed both.
    return bool(RESPONSE_NAME_RE.search(t)
                or RESPONSE_NAME_RE.search(t.replace("_", " ")))


class Finding:
    __slots__ = ("rule", "severity", "func", "lineno", "snippet", "why", "probe")

    def __init__(self, rule: str, severity: str, func: str, lineno: int,
                 snippet: str, why: str, probe: str) -> None:
        self.rule = rule
        self.severity = severity        # high | medium | low
        self.func = func
        self.lineno = lineno
        self.snippet = snippet
        self.why = why
        self.probe = probe

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule, "severity": self.severity, "func": self.func,
            "lineno": self.lineno, "snippet": self.snippet,
            "why": self.why, "probe": self.probe,
        }

    def __repr__(self) -> str:
        return "<%s %s:%d %s>" % (self.severity, self.func, self.lineno, self.rule)


def _seg(src_lines: List[str], node: ast.AST) -> str:
    ln = getattr(node, "lineno", 0)
    if 1 <= ln <= len(src_lines):
        return src_lines[ln - 1].strip()[:200]
    return ""


def _name_of(node: ast.AST) -> str:
    """Dotted name for Name/Attribute nodes, else ''."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return (base + "." + node.attr) if base else node.attr
    return ""


def _is_reward_func(fn) -> bool:
    if REWARD_NAME_RE.search(fn.name):
        return True
    for dec in fn.decorator_list:
        if REWARD_NAME_RE.search(_name_of(dec) or ""):
            return True
    # A function taking both a completion-ish and an answer-ish argument is a
    # grader regardless of what it is called.
    args = {a.arg.lower() for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
    has_out = bool(args & {"completion", "completions", "response", "output",
                           "prediction", "trace", "rollout"})
    has_ref = bool(args & {"answer", "answers", "target", "gold", "reference",
                           "ground_truth", "label", "expected"})
    return has_out and has_ref


# A handler paying less than this share of its function's own top score is a
# consolation prize, not a payout. Reward functions on this corpus are almost
# universally 0-1 scaled, so this doubles as an absolute floor when a function's
# success paths are computed rather than literal.
CONSOLATION_FRACTION = 0.25


def _handler_returns(fn: ast.AST) -> Set[int]:
    """id() of every Return that sits inside an except handler."""
    out: Set[int] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Try):
            for h in n.handlers:
                for s in ast.walk(h):
                    if isinstance(s, ast.Return):
                        out.add(id(s))
    return out


def _success_returns(fn: ast.AST) -> Set[float]:
    """Constant numeric values the function returns on non-exception paths.

    This is the scale the handler has to be judged against. Without it the rule
    cannot tell `except: return 1.0` beside `return 0.0` (a payout) from
    `except: return 0.1` beside `return 1.0` (partial credit for a failed
    attempt) -- and it reported both as high severity.
    """
    skip = _handler_returns(fn)
    out: Set[float] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and id(n) not in skip \
                and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, (int, float)) \
                and not isinstance(n.value.value, bool):
            out.add(float(n.value.value))
    return out


def _zero_weight_funcs(tree: ast.AST) -> Set[str]:
    """Reward functions registered with a literal weight of 0.

    A zero-weighted function is a logged metric, not reward: nothing it returns
    can be gamed for score. One withdrawn hit registered
    `Rubric(funcs=[..., mape], weights=[1.0, unit_bonus, 0.0, 0.0])`, and its
    `except: return 1.0` was reported as a high-severity reward hack even though
    the value never reaches the reward at all.
    """
    out: Set[str] = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        short = _name_of(n.func).rsplit(".", 1)[-1]
        if short.endswith("Rubric"):
            kw = {k.arg: k.value for k in n.keywords}
            funcs, weights = kw.get("funcs"), kw.get("weights")
            if isinstance(funcs, ast.List) and isinstance(weights, ast.List):
                for f, w in zip(funcs.elts, weights.elts):
                    if isinstance(w, ast.Constant) and w.value == 0:
                        out.add(_name_of(f).rsplit(".", 1)[-1])
        elif short == "add_reward_func":
            kw = {k.arg: k.value for k in n.keywords}
            w = kw.get("weight")
            if isinstance(w, ast.Constant) and w.value == 0 and n.args:
                out.add(_name_of(n.args[0]).rsplit(".", 1)[-1])
    return out


class _FuncVisitor(ast.NodeVisitor):
    """Collects findings within a single reward function body."""

    def __init__(self, fname: str, src_lines: List[str],
                 success_returns: Optional[Set[float]] = None,
                 zero_weight: bool = False) -> None:
        self.fname = fname
        self.lines = src_lines
        self.out: List[Finding] = []
        self._has_anchor = False
        self._has_marker = False
        self._compares_equal = False
        self.success_returns = success_returns or set()
        self.zero_weight = zero_weight

    def add(self, rule, severity, node, why, probe) -> None:
        self.out.append(Finding(rule, severity, self.fname, getattr(node, "lineno", 0),
                                _seg(self.lines, node), why, probe))

    # --- containment / substring matching -----------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if isinstance(op, ast.In):
                # `answer in completion` -- the classic loose grader.
                left, right = _name_of(node.left).lower(), _name_of(node.comparators[0]).lower()
                ref = any(k in left for k in ("answer", "gold", "target", "expected",
                                              "reference", "label", "truth"))
                out = any(k in right for k in ("completion", "response", "output",
                                               "text", "prediction", "content"))
                if ref and out:
                    self.add(
                        "substring_containment", "high", node,
                        "grader passes whenever the gold answer appears anywhere in the "
                        "response, so a response listing many candidate answers scores.",
                        "null.all_options",
                    )
            if isinstance(op, (ast.Eq, ast.NotEq)):
                # Only equality against the *response* is evidence for
                # `no_answer_marker`. Counting every `==` made the rule fire on
                # graders comparing two of their own values -- a strict code
                # grader reading `report["failed"] == 0` is not marker-less
                # string matching, and flagging it is the same class of mistake
                # as the original `in`-token version of this rule.
                if _is_response_expr(node.left) or any(
                        _is_response_expr(c) for c in node.comparators):
                    self._compares_equal = True
        self.generic_visit(node)

    # --- regex / parsing ----------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        fn = _name_of(node.func)
        short = fn.rsplit(".", 1)[-1]

        if fn.startswith("re.") or short in ("search", "findall", "match", "fullmatch"):
            if short in ("search", "findall"):
                pat = self._literal_pattern(node)
                if pat is not None and not (pat.startswith("^") or pat.endswith("$")
                                            or "\\b" in pat):
                    self.add(
                        "unanchored_regex", "medium", node,
                        "re.%s with an unanchored pattern matches anywhere in the "
                        "response; trailing or buried text can satisfy it." % short,
                        "extraction.trailing_number",
                    )
            if short == "fullmatch":
                self._has_anchor = True

        if short in ("loads", "load") and fn.split(".")[0] in ("json", "orjson", "ujson"):
            self._json_parse_node = node

        if short in ("eval", "exec") and not fn.startswith("ast."):
            # Severity turns on *what* is evaluated, not on the call. Graders
            # legitimately eval their own dataset-supplied checker expressions:
            # one medical-agent benchmark runs `eval(case_data, results, base)`, where
            # `case_data` ships with the benchmark and the response never
            # reaches it. That shape was 3 of this rule's 4 corpus hits, and
            # calling each a high-severity reward hack would have been a false
            # accusation against published work -- the same mistake, and the
            # same fix, as `except_returns_reward`.
            evaluated = node.args[0] if node.args else None
            if evaluated is not None and _is_response_expr(evaluated):
                self.add(
                    "eval_on_output", "high", node,
                    "grader evaluates model-produced text as code; the response can "
                    "return anything, including the grader's own success value.",
                    "tamper.monkeypatch_grader",
                )
            else:
                # Still reported: evaluating anything is worth a look, and the
                # argument may be response-derived under a name we cannot read.
                # Only the claim is weakened, not the finding.
                self.add(
                    "eval_on_output", "medium", node,
                    "grader evaluates a string as code, but the evaluated argument "
                    "looks dataset- or config-supplied rather than response-derived. "
                    "It is a defect only if the response can reach that string; "
                    "check where the evaluated value comes from.",
                    "tamper.monkeypatch_grader",
                )

        if short in ("run", "call", "check_output", "Popen") and "subprocess" in fn:
            has_timeout = any(k.arg == "timeout" for k in node.keywords)
            if not has_timeout:
                self.add(
                    "subprocess_no_timeout", "medium", node,
                    "subprocess without a timeout: a response that hangs the checker "
                    "can stall or, depending on the harness, be scored as non-failing.",
                    "code.timeout_evade",
                )

        if short in ("isclose", "approx"):
            self._check_tolerance(node)

        self.generic_visit(node)

    def _literal_pattern(self, node: ast.Call) -> Optional[str]:
        if not node.args:
            return None
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
        return None

    def _check_tolerance(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in ("abs_tol", "rel_tol", "abs", "rel"):
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                    if kw.value.value >= 0.01:
                        self.add(
                            "wide_numeric_tolerance", "medium", node,
                            "numeric tolerance of %s accepts answers that are simply "
                            "wrong at that scale." % kw.value.value,
                            "extraction.tolerance_sweep",
                        )

    # --- exception swallowing ----------------------------------------------
    def visit_Try(self, node: ast.Try) -> None:
        """`except -> positive reward`, severity conditioned on what can raise.

        Visiting the whole `try` rather than the handler alone is the point: the
        handler tells you a positive reward is returned, but only the guarded
        block tells you whether that is a defect. Around a local deterministic
        computation it is one -- the response can force the exception and get
        paid for breaking the grader. Around an LLM judge, an HTTP call or a
        subprocess it is an availability decision: the author chose not to
        punish the model for the API being down.

        This distinction is not theoretical. The only high-severity hit in the
        first 150-environment scan was two `except Exception:` blocks around
        judge calls, and reporting those as reward hacks would have been a false
        accusation against someone's published work.
        """
        guarded = " ".join(_expr_text(b) for b in node.body)
        nonlocal_call = bool(NONLOCAL_CALL_RE.search(guarded)) or any(
            isinstance(n, ast.Await) for b in node.body for n in ast.walk(b))

        for handler in node.handlers:
            for stmt in ast.walk(handler):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                    v = stmt.value.value
                    if isinstance(v, (int, float, bool)) and v and v > 0:
                        # Positive is not the same as good, and treating it that
                        # way made this rule 0-for-6 at high severity on the
                        # corpus: every hit was a consolation value, a logged
                        # metric, or a documented neutral. Three gates, each
                        # earned by a withdrawn finding.
                        if self.zero_weight:
                            # weight 0.0, so this is a metric, not reward.
                            continue
                        peak = max(self.success_returns) if self.success_returns else None
                        if float(v) in self.success_returns:
                            # withdrawn: the same 0.5 is returned on two
                            # non-exception paths. A documented neutral, not an
                            # exception payout.
                            continue
                        low = (peak is not None and float(v) < peak) or \
                              (peak is None and float(v) < CONSOLATION_FRACTION)
                        if low:
                            # withdrawn hits: 0.1 vs 1.0, 0.1 as the
                            # lowest return, 0.1 again, and
                            # 0.01. Reported, but not as an accusation.
                            self.add(
                                "except_returns_reward", "low", handler,
                                "an exception path returns a positive reward, but less "
                                "than this function's own success paths pay (%s vs %s), "
                                "so it reads as partial credit for a failed attempt "
                                "rather than a payout for breaking the grader."
                                % (v, ("%g" % peak) if peak is not None else "an unknown peak"),
                                "null.noise",
                            )
                        elif nonlocal_call:
                            self.add(
                                "except_returns_reward", "medium", handler,
                                "an exception path returns a positive reward, but the "
                                "guarded block makes a judge / network / subprocess "
                                "call, so this reads as a fallback for an unavailable "
                                "dependency rather than a gameable grader. It is only a "
                                "defect if the response itself can induce the failure.",
                                "null.noise",
                            )
                        else:
                            self.add(
                                "except_returns_reward", "high", handler,
                                "an exception path returns a positive reward and the "
                                "guarded block is local and deterministic, so a response "
                                "that breaks the grader is paid for breaking it.",
                                "null.noise",
                            )
        self.generic_visit(node)

    # --- length / verbosity rewards ----------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Code-grader family
# ---------------------------------------------------------------------------
# Everything above tests answer-parsing looseness: anchoring, containment,
# tolerance. The published numbers that motivate this project measure something
# else -- weak *test suites* in code environments, where 28.5% of SWE-bench
# Verified and 25.0% of R2E-Gym tasks accept incorrect patches. That defect
# lives in how a grader runs code, not in how it reads a string, so no rule
# above can see it.
#
# Every rule here is gated on code-grader context: the reward function must
# actually execute response-derived code. Ungated, these shapes (write a file,
# read stdout, check an exit status) are ordinary Python and would flag most of
# the corpus. That is the `no_answer_marker` mistake -- a rule that matched the
# bare token `in`, hit 53% of modules, and told us nothing -- and repeating it
# seven times over would be worse than shipping no code rules at all.

_TEST_RUNNER_RE = re.compile(
    r"(?:^|[\s/\\'\"])(pytest|py\.test|unittest|nose2?|tox|jest|vitest|mocha|"
    r"rspec|phpunit|ctest)\b|(?:npm|yarn|pnpm|go|cargo|mvn|gradle|dotnet)\s+test",
    re.IGNORECASE,
)
# Literals a response can simply print. Kept to short strings: the same words
# inside a paragraph of prose are not a success marker.
_SUCCESS_MARKER_RE = re.compile(
    r"(\ball\s+tests?\s+pass|\btests?\s+pass|\bpass(ed|es|ing)?\b|\bok\b|"
    r"\bsuccess(ful)?\b|\bcorrect\b|\bsolved\b|\baccepted\b|"
    r"\bno\s+(errors?|failures?)\b|✓|✔)",
    re.IGNORECASE,
)
_FAIL_MARKER_RE = re.compile(
    r"(\bfail(ed|s|ure|ures|ing)?\b|\berrors?\b|\btraceback\b|\bexception\b|"
    r"\bassertionerror\b|\bwrong\b|\bincorrect\b|✗|✘)",
    re.IGNORECASE,
)
_STDOUT_RE = re.compile(
    r"(stdout|stderr|\bout\b|captured|combined_output|communicate|getvalue|"
    r"\bconsole\b|\blogs?\b)", re.IGNORECASE)
# A grader that checks output against a gold value is doing real work; a marker
# test alongside it is a guard, not the reward.
_REFERENCE_RE = re.compile(
    r"(expected|gold|ground_?truth|reference|\btarget\b|\banswer\b|desired)",
    re.IGNORECASE)
_RETURNCODE_RE = re.compile(
    r"(returncode|retcode|exit_?code|\brc\b|\bret\b|\bstatus\b)", re.IGNORECASE)
_TEST_FILE_RE = re.compile(
    r"(^|/)(test_\w+\.py|\w+_test\.py|conftest\.py|tests?/)", re.IGNORECASE)
_TESTISH_RE = re.compile(r"(^|_)tests?(_|$)", re.IGNORECASE)
_VISIBLE_TEST_RE = re.compile(
    r"(^|_)(visible|public|sample|example|shown|given|demo|open)_(unit_?)?tests?($|_)"
    r"|(^|_)tests?_(visible|public|sample|example|shown|given|demo)($|_)",
    re.IGNORECASE,
)
_HIDDEN_TEST_RE = re.compile(
    r"(hidden|private|held_?out|secret|unseen|full)_+(unit_?)?tests?"
    r"|tests?_+(hidden|private|held_?out|secret|unseen)",
    re.IGNORECASE,
)
# Evidence the author thought about the response having filesystem or CPU reach.
_HARDENING_RE = re.compile(
    r"(chmod|read_?only|s_iread|docker|podman|nsjail|firejail|bubblewrap|"
    r"seccomp|chroot|pyodide|wasm|restricted|sandbox)", re.IGNORECASE)
_TIMEOUT_MECH_RE = re.compile(
    r"(timeout|alarm|setitimer|dump_traceback_later|deadline|sigalrm|"
    r"func_timeout|stopit|time_?limit|max_?seconds)", re.IGNORECASE)
_PROMPT_TARGET_RE = re.compile(
    r"(prompt|question|instruction|user_?(msg|message|content)|"
    r"task_?(text|description)|\bquery\b|problem_?statement)", re.IGNORECASE)
# Execution this pass cannot see. The grader hands the response to a container,
# a tmux session or a remote sandbox, and the commands that decide the reward
# live in task data -- a run_tests.sh, a Docker image, a dataset row -- not in
# this file. Such a grader is neither clean nor defective: it is *unreadable*,
# and counting it as clean would manufacture the result this project exists to
# avoid. Tracked as a coverage gap, exactly like a file we failed to parse.
_DELEGATED_EXEC_RE = re.compile(
    r"exec_run|send_keys|copy_to_container|containers?\.|tmux|sandbox|\be2b\b|"
    r"modal|daytona|morphcloud|docker|kubernetes|\bk8s\b|ssh_|remote_exec|"
    r"run_tests?|test_command|run_command|execute_command|run_script",
    re.IGNORECASE)


# Calls that execute a command line, so their string arguments are commands
# rather than prose. Mirrors the `is_subproc` test in `_gather_call`.
_EXEC_CALL_RE = re.compile(
    r"subprocess\.|\bos\.(system|popen)\b|create_subprocess_(exec|shell)",
    re.IGNORECASE)


def _dispatch_text(nodes: List[ast.AST]) -> str:
    """Text from positions where a token means machinery, not prose.

    Matching `_DELEGATED_EXEC_RE` against flattened source counts a *mention* as
    delegation. `veroseo/docker-container-manager` is a pure text rubric scoring
    a response with `any(kw in lower for kw in [..., "docker"])` -- the token is
    a keyword being searched for, the grader executes nothing, and it was
    classified as an unreadable coverage gap on that basis. Subject matter is
    not mechanism.

    Call targets, attribute names and imported modules are positions where a
    platform token could actually dispatch work. String literals are not, unless
    they are arguments to something that executes -- `subprocess.run(["docker",
    "exec", ...])` is real delegation, so executor calls contribute their whole
    expression.
    """
    parts: List[str] = []
    for stmt in nodes:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                target = _name_of(n.func)
                parts.append(target)
                if _EXEC_CALL_RE.search(target):
                    parts.append(_expr_text(n))
            elif isinstance(n, ast.Attribute):
                parts.append(n.attr)
            elif isinstance(n, ast.Import):
                parts.extend(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                parts.append(n.module or "")
    return " ".join(parts)


def _delegates_execution(ev: "_CodeEvidence", nodes: List[ast.AST],
                         text: str) -> bool:
    """True when the grader ships work somewhere this pass cannot read.

    Two conditions, not one. The token must appear, *and* the grader must have
    machinery that could dispatch: a subprocess/exec call, or the token itself
    sitting in a call/import position rather than in a string being matched.
    """
    if not _DELEGATED_EXEC_RE.search(text):
        return False
    if ev.exec_nodes or ev.inproc_exec:
        return True
    return bool(_DELEGATED_EXEC_RE.search(_dispatch_text(nodes)))


def _dirname(p: str) -> str:
    return p.rsplit("/", 1)[0] if "/" in p else "."


def _dir_expr(node: ast.AST) -> str:
    """Best-effort identity of the directory a path expression lives in.

    Only used to answer "are these two files in the same place", so an opaque
    but *consistent* token (the base variable's name) is as good as a real path.
    """
    if isinstance(node, ast.Call):
        short = _name_of(node.func).rsplit(".", 1)[-1]
        if short in ("join", "Path", "PurePath", "resolve", "absolute",
                     "expanduser", "mkdtemp") and node.args:
            return _dir_expr(node.args[0])
        return _name_of(node.func)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _dir_expr(node.left)                      # Path(work) / "x.py"
    if isinstance(node, ast.JoinedStr):
        for v in node.values:                            # f"{work}/x.py"
            if isinstance(v, ast.FormattedValue):
                return _dir_expr(v.value)
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return _dirname(v.value)
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _dirname(node.value)
    return _name_of(node)


def _path_text(node: ast.AST) -> str:
    """The literal parts of a path expression, joined."""
    parts = [n.value for n in ast.walk(node)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return "/".join(parts)


def _argv_parts(call: ast.Call) -> "tuple":
    """(literal argv strings, identifier text) for an exec-style call."""
    lits: List[str] = []
    names: List[str] = []
    for a in list(call.args) + [k.value for k in call.keywords if k.arg in ("args", "cmd")]:
        for n in ast.walk(a):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                lits.append(n.value)
            elif isinstance(n, ast.Name):
                names.append(n.id)
            elif isinstance(n, ast.Attribute):
                names.append(n.attr)
    return lits, " ".join(names).lower()


def _key_candidate(n: ast.AST) -> Optional[str]:
    """Identifier-like text for a node, for test-payload name matching.

    Subscript slices count (`info["tests"]`) but bare string constants do not --
    otherwise the filename "test_foo.py" would read as a test payload name.
    """
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return n.attr
    if isinstance(n, ast.Subscript):
        s = n.slice
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            return s.value
    if isinstance(n, ast.keyword) and n.arg:
        return n.arg
    return None


def _is_string_build(node: ast.AST) -> bool:
    """Does this expression construct a piece of text?

    Load-bearing for `visible_test_only`: the defect is tests being *written
    into* the prompt, not merely a variable whose name mentions both. Without
    this, `prompt_to_tests[key] = row["test_cases"]` -- an ordinary lookup table
    keyed by prompt -- reads as tests leaking into the prompt. That exact shape
    produced the rule's first false positive on real Hub source.
    """
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _is_string_build(node.left) or _is_string_build(node.right)
    if isinstance(node, ast.Call):
        short = _name_of(node.func).rsplit(".", 1)[-1]
        return short in ("format", "format_map", "join", "dedent", "render",
                         "substitute", "safe_substitute")
    return False


def _prompt_test_keys(tree: ast.AST) -> Set[str]:
    """Test payloads that are interpolated into the prompt the model sees.

    This is the mechanism behind `visible_test_only`: if the same key feeds both
    the prompt text and the grader, the model is scored on tests it was handed,
    and a patch overfitted to exactly those tests is indistinguishable from a
    solution.
    """
    keys: Set[str] = set()

    def harvest(value: ast.AST) -> None:
        if not _is_string_build(value):
            return
        for m in ast.walk(value):
            cand = _key_candidate(m)
            if cand and _TESTISH_RE.search(cand):
                keys.add(cand.lower())

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            if any(_PROMPT_TARGET_RE.search(_expr_text(t)) for t in n.targets):
                harvest(n.value)
        elif isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if (k is not None and isinstance(k, ast.Constant)
                        and isinstance(k.value, str)
                        and _PROMPT_TARGET_RE.search(k.value)):
                    harvest(v)
        elif isinstance(n, ast.keyword) and n.arg and _PROMPT_TARGET_RE.search(n.arg):
            harvest(n.value)
    return keys


class _CodeEvidence:
    """What one grader does with the response. Gathered once, read by rules."""

    def __init__(self) -> None:
        self.exec_nodes: List[ast.AST] = []          # subprocess / sandbox calls
        self.inproc_exec: List[tuple] = []           # (node, kind) in-process runs
        self.runs_test_runner = False
        self.runs_response_code = False
        self.reads_stdout = False
        self.reads_report = False
        self.writes_response = False
        self.hardening = False
        self.timeout_mech = False
        self.exec_cwds: Set[str] = set()
        self.write_test: List[tuple] = []            # (node, dir) test files
        self.write_other: List[tuple] = []           # (node, dir) everything else
        self.response_basenames: Set[str] = set()
        self.test_path_literals: List[str] = []
        self.stdout_marker: List[tuple] = []         # (node, literal)
        self.returncode_checks: List[ast.AST] = []
        self.opt_flags: List[ast.AST] = []
        self.sys_path_inserts: List[ast.AST] = []
        self.unbounded_loops: List[ast.AST] = []
        self.test_keys: Set[str] = set()
        self.visible_hits: List[tuple] = []          # (node, name)
        self.delegated_exec = False                  # runs code we cannot see
        self.compares_reference = False              # output checked against gold


_WRITE_MODES = ("w", "a", "x", "+")


def _gather_writes(ev: _CodeEvidence, n: ast.Call) -> None:
    fn = _name_of(n.func)
    short = fn.rsplit(".", 1)[-1]
    target = None
    if short == "open" and len(n.args) >= 2:
        mode = n.args[1].value if (isinstance(n.args[1], ast.Constant)
                                   and isinstance(n.args[1].value, str)) else ""
        if any(c in mode for c in _WRITE_MODES):
            target = n.args[0]
    elif short in ("write_text", "write_bytes") and isinstance(n.func, ast.Attribute):
        target = n.func.value
        if n.args and _is_response_expr(n.args[0]):
            ev.writes_response = True
    elif short in ("write", "writelines") and isinstance(n.func, ast.Attribute):
        if n.args and _is_response_expr(n.args[0]):
            ev.writes_response = True
    elif short in ("copy", "copyfile", "copy2", "move") and len(n.args) >= 2:
        target = n.args[1]
    if target is None:
        return
    ptxt = _path_text(target)
    d = _dir_expr(target)
    if _TEST_FILE_RE.search(ptxt):
        ev.write_test.append((n, d))
    else:
        ev.write_other.append((n, d))
        base = ptxt.rsplit("/", 1)[-1]
        if "." in base:
            ev.response_basenames.add(base)


def _gather_call(ev: _CodeEvidence, n: ast.Call) -> None:
    fn = _name_of(n.func)
    short = fn.rsplit(".", 1)[-1]
    kwargs = {k.arg: k.value for k in n.keywords if k.arg}

    is_subproc = (("subprocess" in fn and short in ("run", "call", "check_call",
                                                    "check_output", "Popen"))
                  or fn in ("os.system", "os.popen")
                  or short in ("create_subprocess_exec", "create_subprocess_shell"))
    # Container / remote sandbox SDKs (docker, e2b, modal) all wear one of these.
    is_sandbox = short in ("exec_run", "run_code", "run_command", "run_cmd", "exec_cell")
    if is_subproc or is_sandbox:
        ev.exec_nodes.append(n)
        lits, nametext = _argv_parts(n)
        if "cwd" in kwargs:
            d = _dir_expr(kwargs["cwd"])
            if d:
                ev.exec_cwds.add(d)
        joined = " ".join(lits)
        if _TEST_RUNNER_RE.search(joined) or _TEST_RUNNER_RE.search(nametext):
            ev.runs_test_runner = True
        if short == "check_output":
            ev.reads_stdout = True
        resp_argv = (RESPONSE_NAME_RE.search(nametext)
                     and not NOT_RESPONSE_RE.search(nametext))
        if ("-c" in lits or "-e" in lits or resp_argv
                or any(b in joined for b in ev.response_basenames)
                or any(k in kwargs and _is_response_expr(kwargs[k])
                       for k in ("input", "stdin"))):
            ev.runs_response_code = True

    if fn in ("pytest.main", "unittest.main") or short in (
            "import_module", "__import__", "exec_module", "run_path",
            "run_module", "load_module"):
        ev.inproc_exec.append((n, short))
    if short in ("eval", "exec") and not fn.startswith("ast."):
        ev.inproc_exec.append((n, short))
        if n.args and _is_response_expr(n.args[0]):
            ev.runs_response_code = True
    # `compile()` is deliberately NOT execution. It builds a code object and
    # runs nothing; graders use it as a syntax check before shipping the code
    # somewhere else. Counting it cost two false positives on long-code-edit,
    # whose compile() validates syntax and whose real execution is a POST to a
    # remote evaluator. Where a grader compiles *and* runs, the exec() below it
    # is what fires, as it does in one corpus environment.
    if short == "compile" and n.args and _is_response_expr(n.args[0]):
        ev.runs_response_code = True
    if fn in ("sys.path.insert", "sys.path.append"):
        ev.sys_path_inserts.append(n)
    if short in ("communicate", "getvalue"):
        ev.reads_stdout = True
    if short in ("load", "loads") and fn.split(".")[0] in ("json", "orjson", "ujson", "xmltodict"):
        ev.reads_report = True

    # stdout compared to a marker through a method rather than an operator
    if (short in ("startswith", "endswith", "count", "find", "index")
            and isinstance(n.func, ast.Attribute) and n.args
            and _STDOUT_RE.search(_expr_text(n.func.value))):
        _note_marker(ev, n, n.args[0])
    if short in ("search", "match", "findall", "fullmatch") and len(n.args) >= 2:
        if _STDOUT_RE.search(_expr_text(n.args[1])):
            _note_marker(ev, n, n.args[0])


def _note_marker(ev: _CodeEvidence, node: ast.AST, lit_node: ast.AST) -> None:
    if not (isinstance(lit_node, ast.Constant) and isinstance(lit_node.value, str)):
        return
    lit = lit_node.value
    if len(lit) <= 80 and (_SUCCESS_MARKER_RE.search(lit) or _FAIL_MARKER_RE.search(lit)):
        ev.stdout_marker.append((node, lit))


def _gather_compare(ev: _CodeEvidence, n: ast.Compare) -> None:
    operands = [n.left] + list(n.comparators)
    for i, op in enumerate(n.ops):
        pair = (operands[i], operands[i + 1])
        for a, b in (pair, pair[::-1]):
            if isinstance(op, (ast.In, ast.NotIn, ast.Eq, ast.NotEq)) \
                    and _STDOUT_RE.search(_expr_text(a)):
                _note_marker(ev, n, b)
            # Output weighed against a gold value somewhere in the grader.
            if isinstance(op, (ast.Eq, ast.NotEq)) \
                    and not isinstance(b, ast.Constant) \
                    and _REFERENCE_RE.search(_expr_text(b)):
                ev.compares_reference = True
            if isinstance(op, (ast.Eq, ast.NotEq)) \
                    and isinstance(b, ast.Constant) and b.value == 0 \
                    and not isinstance(b.value, bool) \
                    and _RETURNCODE_RE.search(_expr_text(a)):
                ev.returncode_checks.append(n)


def _gather_while(ev: _CodeEvidence, n: ast.While) -> None:
    if not (isinstance(n.test, ast.Constant) and n.test.value):
        return
    body = " ".join(_expr_text(s) for s in n.body)
    polls = re.search(r"(poll|readline|\bread\b|recv|communicate|sleep|stdout|"
                      r"stderr|\bwait\b)", body, re.IGNORECASE)
    bounded = re.search(r"(time|deadline|timeout|elapsed|monotonic|perf_counter|"
                        r"attempts|max_)", body, re.IGNORECASE)
    if polls and not bounded:
        ev.unbounded_loops.append(n)


def _gather_code_evidence(nodes: List[ast.AST]) -> _CodeEvidence:
    ev = _CodeEvidence()
    text = " ".join(_expr_text(s) for s in nodes)
    ev.hardening = bool(_HARDENING_RE.search(text))
    ev.timeout_mech = bool(_TIMEOUT_MECH_RE.search(text))

    # Writes first: a later `subprocess.run(["python", "solution.py"])` is only
    # recognisable as running the response once we know what was written where.
    for stmt in nodes:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                _gather_writes(ev, n)

    for stmt in nodes:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                _gather_call(ev, n)
            elif isinstance(n, ast.Compare):
                _gather_compare(ev, n)
            elif isinstance(n, ast.While):
                _gather_while(ev, n)
            elif isinstance(n, ast.Attribute) and n.attr in ("stdout", "stderr"):
                ev.reads_stdout = True
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                v = n.value
                if v in ("-O", "-OO") or "PYTHONOPTIMIZE" in v \
                        or re.search(r"\bpython[\d.]*\s+-OO?\b", v):
                    ev.opt_flags.append(n)
                if _TEST_FILE_RE.search(v):
                    ev.test_path_literals.append(v)
            cand = _key_candidate(n)
            if cand and _TESTISH_RE.search(cand):
                ev.test_keys.add(cand.lower())
                if _VISIBLE_TEST_RE.search(cand):
                    ev.visible_hits.append((n, cand))

    # Decided last: it reads `exec_nodes`, which the loops above populate.
    ev.delegated_exec = _delegates_execution(ev, nodes, text)
    return ev


def _effective_nodes(fn, local_funcs: Dict[str, Any], max_depth: int = 2) -> List[ast.AST]:
    """Statements of `fn` plus those of local helpers it calls, to `max_depth`.

    Real code graders split the work: `reward()` calls `run_tests()` calls
    `_exec()`. Analysing the reward function alone would see a bare function
    call and conclude the grader does nothing. Only *locally defined* callees
    are followed, so this stays a source-level fact, not a guess about imports.
    """
    seen = {fn.name}
    nodes: List[ast.AST] = list(fn.body)
    frontier = [fn]
    for _ in range(max_depth):
        nxt = []
        for f in frontier:
            for c in ast.walk(f):
                if not isinstance(c, ast.Call):
                    continue
                name = _name_of(c.func).rsplit(".", 1)[-1]
                g = local_funcs.get(name)
                if g is not None and name not in seen:
                    seen.add(name)
                    nodes.extend(g.body)
                    nxt.append(g)
        if not nxt:
            break
        frontier = nxt
    return nodes


def _is_code_grader(ev: _CodeEvidence) -> bool:
    """Does this grader actually execute response-derived code?

    Tracked, not just used as a gate. Zero code findings across a corpus means
    nothing until you know how many code graders were in it -- the same reason
    `parse_failed` is reported separately from `parsed`. A denominator of three
    would make "no code-grader defects" an empty statement.
    """
    return bool((ev.exec_nodes or ev.inproc_exec)
                and (ev.runs_response_code or ev.writes_response))


def _is_opaque_grader(ev: _CodeEvidence) -> bool:
    """Grader that clearly runs code, through machinery this pass cannot read.

    Reported as a coverage gap rather than a clean result -- the same accounting
    as `parse_failed`. On the Hub this is not an edge case: the container-based
    code environments (SWE-bench-alikes, kernel benchmarks, terminal harnesses)
    are precisely the ones the weak-test-suite literature is about, and they are
    the ones a source-level pass can say least about.
    """
    return bool(ev.delegated_exec and not _is_code_grader(ev))


def _code_grader_findings(fname: str, ev: _CodeEvidence, lines: List[str],
                          hidden_tests: bool, prompt_tests: Set[str]) -> List[Finding]:
    out: List[Finding] = []
    if not _is_code_grader(ev):
        return out

    def add(rule, sev, node, why, probe):
        out.append(Finding(rule, sev, fname, getattr(node, "lineno", 0),
                           _seg(lines, node), why, probe))

    # --- graded on the tests the model was shown ---------------------------
    if not hidden_tests:
        if ev.visible_hits:
            n0, name = ev.visible_hits[0]
            add("visible_test_only", "high", n0,
                "the grader runs `%s` -- a payload the environment itself names as "
                "the visible/public set -- and no hidden or held-out suite appears "
                "anywhere in the module, so a patch overfitted to exactly the tests "
                "in the prompt is indistinguishable from a solution." % name,
                "code.overfit_visible_tests")
        elif ev.test_keys & prompt_tests:
            shared = sorted(ev.test_keys & prompt_tests)[0]
            add("visible_test_only", "high", ev.exec_nodes[0] if ev.exec_nodes
                else ev.inproc_exec[0][0],
                "`%s` is interpolated into the prompt and is also what the grader "
                "executes, so the model is scored on tests it was handed; there is "
                "no held-out suite in this module to catch an overfitted patch."
                % shared,
                "code.overfit_visible_tests")

    # --- reward read out of output the response writes ----------------------
    # Suppressed when the grader also weighs the output against a gold value:
    # there the marker test is a sanity guard and the reward still turns on
    # matching the expected result, so calling it spoofable would misstate the
    # mechanism. This is why the rule's first real hit was withdrawn.
    for n0, lit in (ev.stdout_marker if not ev.compares_reference else []):
        add("stdout_spoof", "high", n0,
            "reward turns on whether %r appears in output produced by "
            "response-controlled code, so a response that gets that string into "
            "the stream (or suppresses output, for a failure marker) is paid "
            "without solving the task. Note the response may have to dodge the "
            "runner's output capture to do it -- against concurrency-bench a "
            "plain print scored 0 and only a deferred write to the real stdout "
            "landed, so treat the probe, not the print, as the test." % lit,
            "code.stdout_marker")

    # --- exit status is the whole signal ------------------------------------
    if (ev.returncode_checks and ev.runs_response_code and not ev.runs_test_runner
            and not ev.reads_stdout and not ev.reads_report):
        add("exit_code_only", "high", ev.returncode_checks[0],
            "the entire reward is the exit status of response-controlled code "
            "with nothing else inspected, so anything that terminates cleanly -- "
            "an empty program, or one ending in sys.exit(0) -- scores full marks.",
            "code.exit_zero")

    # --- assertions compiled out --------------------------------------------
    if ev.opt_flags:
        add("assert_disabled", "high", ev.opt_flags[0],
            "tests run with assertions disabled (-O / PYTHONOPTIMIZE), so every "
            "`assert` in the suite is stripped at compile time and no test can "
            "fail; any response that imports cleanly passes.",
            "code.empty_solution")

    # --- the response can rewrite what grades it ----------------------------
    test_dirs = {d for _, d in ev.write_test if d}
    other_dirs = {d for _, d in ev.write_other if d}
    shared_dir = test_dirs & other_dirs
    anchor = None
    if shared_dir:
        anchor = next(n for n, d in ev.write_test if d in shared_dir)
    elif ev.test_path_literals and (ev.exec_cwds & other_dirs):
        anchor = ev.exec_nodes[0] if ev.exec_nodes else None
    if anchor is not None and ev.writes_response and not ev.hardening:
        add("test_file_writable", "high", anchor,
            "the response's code and the test files that grade it land in the "
            "same writable directory with no read-only or container isolation, "
            "so the graded code can rewrite the tests, drop a conftest.py, or "
            "stub out the assertions before they run.",
            "code.rewrite_tests")

    # --- response code runs inside the grader --------------------------------
    inproc_import = [(n, k) for n, k in ev.inproc_exec if k not in ("eval", "exec")]
    if inproc_import:
        n0, kind = inproc_import[0]
        add("no_test_isolation", "medium", n0,
            "response code is imported/executed in the grader's own process via "
            "`%s`, so its import-time side effects run before any assertion is "
            "evaluated and can reach the grader's own state." % kind,
            "code.import_time_side_effect")
    elif ev.sys_path_inserts and ev.runs_test_runner:
        add("no_test_isolation", "medium", ev.sys_path_inserts[0],
            "the response's directory is put on sys.path ahead of the test run, "
            "so anything importable from it -- including a planted conftest.py "
            "or sitecustomize.py -- loads at collection time.",
            "code.import_time_side_effect")

    # --- unbounded time ------------------------------------------------------
    if ev.inproc_exec and not ev.timeout_mech:
        add("timeout_evade", "medium", ev.inproc_exec[0][0],
            "response-controlled code runs in-process with no wall-clock bound "
            "(no timeout argument, signal.alarm or executor deadline), so a "
            "response that loops forever stalls grading rather than scoring 0.",
            "code.infinite_loop")
    for w in ev.unbounded_loops:
        add("timeout_evade", "medium", w,
            "the grader polls response-controlled execution in an unbounded "
            "`while True` with no deadline check, so a response that never "
            "produces the awaited output hangs the run.",
            "code.infinite_loop")
    return out


class ModuleScan:
    def __init__(self, path: str) -> None:
        self.path = path
        self.parse_ok: Optional[bool] = None
        self.parse_error: Optional[str] = None
        self.reward_funcs: List[str] = []
        self.code_grader_funcs: List[str] = []
        self.opaque_grader_funcs: List[str] = []
        self.findings: List[Finding] = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "reward_funcs": self.reward_funcs,
            "code_grader_funcs": self.code_grader_funcs,
            "opaque_grader_funcs": self.opaque_grader_funcs,
            "findings": [f.as_dict() for f in self.findings],
        }


def scan_source(src: str, path: str = "<string>") -> ModuleScan:
    """Parse one module and return its scan. Never raises on bad input."""
    scan = ModuleScan(path)
    # A UTF-8 BOM survives the Hub's utf-8 decode and is a hard SyntaxError.
    # It is an encoding artefact, not a property of the code, so strip it
    # rather than record a bogus coverage gap.
    if src.startswith("﻿"):
        src = src.lstrip("﻿")
    try:
        tree = ast.parse(src)
        scan.parse_ok = True
    except SyntaxError as exc:
        scan.parse_ok = False
        scan.parse_error = "SyntaxError: %s (line %s)" % (exc.msg, exc.lineno)
        return scan
    except (ValueError, RecursionError) as exc:
        scan.parse_ok = False
        scan.parse_error = type(exc).__name__ + ": " + str(exc)[:120]
        return scan

    lines = src.splitlines()
    local_funcs = {n.name: n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    hidden_tests = _HIDDEN_TEST_RE.search(src) is not None
    prompt_tests = _prompt_test_keys(tree)
    zero_weight = _zero_weight_funcs(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_reward_func(node):
            scan.reward_funcs.append(node.name)
            v = _FuncVisitor(node.name, lines,
                             success_returns=_success_returns(node),
                             zero_weight=node.name in zero_weight)
            for child in node.body:
                v.visit(child)
            scan.findings.extend(v.out)

            # Code-grader family, over the grader plus the local helpers it calls.
            ev = _gather_code_evidence(_effective_nodes(node, local_funcs))
            if _is_code_grader(ev):
                scan.code_grader_funcs.append(node.name)
            elif _is_opaque_grader(ev):
                scan.opaque_grader_funcs.append(node.name)
            scan.findings.extend(_code_grader_findings(
                node.name, ev, lines, hidden_tests, prompt_tests))

            # Whole-function check. Deliberately narrow: an earlier version fired
            # on any body containing the token "in", which matched every `for`
            # loop and hit 53% of all modules. A rule that flags half the corpus
            # tells you nothing and, published, would be an accusation we cannot
            # stand behind. It now requires positive evidence of a comparison
            # against free text AND the absence of any extraction step.
            body_src = "\n".join(lines[node.lineno - 1: getattr(node, "end_lineno", node.lineno)])
            has_marker = re.search(r"boxed|ANSWER\s*:|FINAL|####|</?answer>|<<|\\\[",
                                   body_src, re.IGNORECASE) is not None
            has_extraction = re.search(
                r"\.split\(|\.partition\(|\.rpartition\(|\.group\(|\.strip\(\s*[\"']"
                r"|findall|fullmatch|json\.loads|parse_",
                body_src) is not None
            if v._compares_equal and not has_marker and not has_extraction:
                scan.findings.append(Finding(
                    "no_answer_marker", "low", node.name, node.lineno,
                    (lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "")[:200],
                    "the grader compares against the response with no delimiter and no "
                    "extraction step, so whatever surrounding text the model emits is "
                    "part of the compared string.",
                    "extraction.buried_gold",
                ))

    # Two reward functions sharing a helper would each report that helper's
    # lines. The corpus store keys findings on (env, path, rule, line), so
    # collapse here too -- otherwise in-memory counts and stored counts diverge
    # and the headline number depends on which one you happened to read.
    seen: Set[Any] = set()
    unique: List[Finding] = []
    for f in scan.findings:
        k = (f.rule, f.lineno)
        if k not in seen:
            seen.add(k)
            unique.append(f)
    scan.findings = unique
    return scan


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def summarise(scans: Iterable[ModuleScan]) -> Dict[str, Any]:
    scans = list(scans)
    parsed = [s for s in scans if s.parse_ok]
    failed = [s for s in scans if s.parse_ok is False]
    by_rule: Dict[str, int] = {}
    by_sev: Dict[str, int] = {}
    for s in parsed:
        for f in s.findings:
            by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    return {
        "modules": len(scans),
        "parsed": len(parsed),
        "parse_failed": len(failed),
        "parse_coverage": (len(parsed) / len(scans)) if scans else 0.0,
        "with_reward_funcs": sum(1 for s in parsed if s.reward_funcs),
        "with_code_graders": sum(1 for s in parsed if s.code_grader_funcs),
        "with_opaque_graders": sum(1 for s in parsed if s.opaque_grader_funcs),
        "with_findings": sum(1 for s in parsed if s.findings),
        "findings_by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
        "findings_by_severity": by_sev,
        "python": "%d.%d" % sys.version_info[:2],
    }


if __name__ == "__main__":
    # Positive controls: each deliberately broken grader must trip its rule.
    # A rule with no positive control is a rule nobody has ever seen fire.
    CASES = [
        ("substring_containment", '''
def reward_fn(completion, answer):
    return 1.0 if answer in completion else 0.0
'''),
        ("unanchored_regex", '''
import re
def score_answer(completion, answer):
    m = re.search("(\\\\d+)", completion)
    return float(m and m.group(1) == answer)
'''),
        ("except_returns_reward", '''
def grade(completion, answer):
    try:
        return float(int(completion) == int(answer))
    except Exception:
        return 1.0
'''),
        ("eval_on_output", '''
def reward(completion, answer):
    return float(eval(completion) == answer)
'''),
        ("subprocess_no_timeout", '''
import subprocess
def verify(completion, answer):
    r = subprocess.run(["python", "-c", completion], capture_output=True)
    return float(r.returncode == 0)
'''),
        ("wide_numeric_tolerance", '''
import math
def reward(completion, answer):
    return float(math.isclose(float(completion), float(answer), abs_tol=0.5))
'''),
        ("no_answer_marker", '''
def check_answer(completion, answer):
    return 1.0 if completion == answer else 0.0
'''),
        # --- code-grader family ---------------------------------------------
        ("visible_test_only", '''
import subprocess
def code_reward(completion, info):
    tests = info["visible_tests"]          # the same tests quoted in the prompt
    with open("sol.py", "w") as fh:
        fh.write(completion)
    r = subprocess.run(["python", "-c", tests], capture_output=True, timeout=30)
    return float(r.returncode == 0)
'''),
        # ...and the second path into the same rule: tests written into the
        # prompt text and then executed as the grade. Kept as its own control
        # because the two paths share nothing but a rule name.
        ("visible_test_only", '''
import subprocess
TEMPLATE = "Solve the problem. Your code must pass:\\n{tests}"
def build_row(row):
    return {"question": TEMPLATE.format(tests=row["test_list"]), "info": row}
def code_reward(completion, info):
    with open("sol.py", "w") as fh:
        fh.write(completion)
    r = subprocess.run(["python", "sol.py"], input=info["test_list"],
                       capture_output=True, timeout=30)
    return float(r.returncode == 0)
'''),
        ("stdout_spoof", '''
import subprocess
def grade_code(completion, answer):
    with open("sol.py", "w") as fh:
        fh.write(completion)
    r = subprocess.run(["python", "sol.py"], capture_output=True, text=True, timeout=30)
    return 1.0 if "All tests passed" in r.stdout else 0.0
'''),
        ("exit_code_only", '''
import subprocess
def reward_code(completion, answer):
    with open("solution.py", "w") as fh:
        fh.write(completion)
    proc = subprocess.run(["python", "solution.py"], capture_output=True, timeout=30)
    return 1.0 if proc.returncode == 0 else 0.0
'''),
        ("assert_disabled", '''
import subprocess
def verify_solution(completion, tests):
    with open("t.py", "w") as fh:
        fh.write(completion + chr(10) + tests)
    r = subprocess.run(["python", "-O", "t.py"], capture_output=True, timeout=30)
    return float(r.returncode == 0)
'''),
        ("test_file_writable", '''
import os, subprocess, tempfile
def reward_fn(completion, tests):
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "solution.py"), "w") as fh:
        fh.write(completion)
    with open(os.path.join(work, "test_solution.py"), "w") as fh:
        fh.write(tests)
    r = subprocess.run(["python", "-m", "pytest", "test_solution.py"],
                       cwd=work, capture_output=True, timeout=60)
    return float(r.returncode == 0)
'''),
        ("no_test_isolation", '''
import importlib, sys
def score_submission(completion, answer):
    with open("/tmp/sub/solution.py", "w") as fh:
        fh.write(completion)
    sys.path.insert(0, "/tmp/sub")
    mod = importlib.import_module("solution")
    return float(mod.solve() == answer)
'''),
        ("timeout_evade", '''
import pytest
def code_grader(completion, answer):
    with open("gen.py", "w") as fh:
        fh.write(completion)
    rc = pytest.main(["-q", "gen.py"])
    return 1.0 if rc == 0 else 0.0
'''),
    ]
    failures = 0
    for expect, src in CASES:
        s = scan_source(src, expect)
        rules = {f.rule for f in s.findings}
        ok = expect in rules
        failures += 0 if ok else 1
        print("%-26s %s  (funcs=%s rules=%s)"
              % (expect, "ok" if ok else "MISS", s.reward_funcs, sorted(rules)))

    # Severity control for the `except_returns_reward` correction. The same
    # shape must be high when the guarded block is local and deterministic and
    # medium when it is a judge call, because an availability fallback around a
    # flaky API is not a reward hack and publishing it as one is a false
    # accusation. Both directions are asserted: a rule that only ever downgrades
    # would silently retire the real finding.
    JUDGE_FALLBACK = '''
def judge_reward(completion, answer):
    try:
        verdict = client.chat.completions.create(model="x", messages=[])
        return float("YES" in verdict.choices[0].message.content)
    except Exception:
        return 1.0
'''
    s = scan_source(JUDGE_FALLBACK, "judge_fallback")
    sev = {f.rule: f.severity for f in s.findings}
    ok = sev.get("except_returns_reward") == "medium"
    failures += 0 if ok else 1
    print("%-26s %s  (severity=%s)"
          % ("except_downgrade", "ok" if ok else "NOT DOWNGRADED",
             sev.get("except_returns_reward")))

    # Negative controls. A rule that fires on a careful grader is worse than no
    # rule: it manufactures findings, and every one of them costs review time we
    # would rather spend on a real defect.
    STRICT = '''
import re
def reward_fn(completion, answer):
    m = re.fullmatch(r"ANSWER:\\\\s*(-?\\\\d+)", completion.strip())
    if m is None:
        return 0.0
    return 1.0 if m.group(1) == str(answer) else 0.0
'''
    s = scan_source(STRICT, "strict")
    high = [f for f in s.findings if f.severity == "high"]
    print("%-26s %s  (%s)" % ("strict_grader(negative)", "ok" if not high else "FALSE POSITIVE",
                              [f.rule for f in s.findings] or "clean"))
    failures += 1 if high else 0

    # A strict *code* grader: hidden tests, a separate read-only test directory,
    # a hard timeout, a machine-readable report instead of stdout parsing. Zero
    # findings of any severity -- this is the shape the new family must not
    # punish, or it punishes doing the job properly.
    STRICT_CODE = '''
import json, os, subprocess, tempfile

def code_reward(completion, answer):
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "solution.py"), "w") as fh:
        fh.write(completion)
    suite = tempfile.mkdtemp()
    harness = os.path.join(suite, "test_hidden.py")
    with open(harness, "w") as fh:
        fh.write(answer["hidden_tests"])
    os.chmod(harness, 0o444)
    report = os.path.join(suite, "report.json")
    subprocess.run(
        ["python", "-m", "pytest", harness, "-q", "--timeout=10",
         "--report-json", report],
        cwd=suite, capture_output=True, timeout=120,
        env={"PYTHONPATH": work},
    )
    with open(report) as fh:
        summary = json.load(fh)["summary"]
    return float(summary["failed"] == 0 and summary["passed"] > 0)
'''
    s = scan_source(STRICT_CODE, "strict_code")
    print("%-26s %s  (%s)"
          % ("strict_code(negative)", "ok" if not s.findings else "FALSE POSITIVE",
             [(f.rule, f.severity) for f in s.findings] or "clean"))
    failures += 1 if s.findings else 0

    # Severity control for the `eval_on_output` correction, asserted in both
    # directions. Evaluating the response is a real reward hack; evaluating the
    # benchmark's own checker expression is how two medical-agent
    # benchmarks legitimately work. A rule that only ever downgrades has
    # quietly retired itself, so the high case is pinned here too.
    EVAL_RESPONSE = '''
def reward(completion, answer):
    return float(eval(completion) == answer)
'''
    EVAL_DATASET = '''
def medagent_reward(completion, case_data, results, fhir_api_base):
    is_correct = eval(case_data, results, fhir_api_base)
    return 1 if is_correct else 0
'''
    hi = {f.rule: f.severity for f in scan_source(EVAL_RESPONSE, "eval_hi").findings}
    lo = {f.rule: f.severity for f in scan_source(EVAL_DATASET, "eval_lo").findings}
    ok = hi.get("eval_on_output") == "high" and lo.get("eval_on_output") == "medium"
    failures += 0 if ok else 1
    print("%-26s %s  (response=%s dataset=%s)"
          % ("eval_downgrade", "ok" if ok else "WRONG SEVERITY",
             hi.get("eval_on_output"), lo.get("eval_on_output")))

    # Withdrawn false positives, kept as controls. Both come from one real Hub
    # environment (bhogan94/q-programming-language) where the first version of
    # the code family reported two high-severity findings and manual review
    # killed both:
    #   * `prompt_to_tests[k] = row["test_cases"]` is a lookup keyed by prompt,
    #     not tests printed in the prompt -- no `visible_test_only`.
    #   * `"error" not in stderr` guards a reward that still turns on
    #     `stdout == expected_output` -- no `stdout_spoof`.
    # A rule that cannot survive its own first real hit should not ship, and a
    # withdrawn finding that can silently come back is worse than one that never
    # went away.
    WITHDRAWN = '''
import subprocess

def run_code(code, timeout=1.0):
    with open("script.q", "w") as fh:
        fh.write(code)
    process = subprocess.run(["q", "script.q"], capture_output=True,
                             text=True, timeout=timeout)
    success = process.returncode == 0 and "error" not in process.stderr.lower()
    return success, process.stdout.strip()

def load_environment():
    prompt_to_tests = {}
    for item in dataset:
        prompt_to_tests[item["prompt"]] = item["test_cases"]

    def q_reward_func(parser, completion, answer, **kwargs):
        test_cases = prompt_to_tests.get(kwargs.get("prompt", ""), [])
        passed = 0
        for case in test_cases:
            ok, stdout = run_code(completion + case["q_test_code"])
            expected = case["q_expected_output"].strip()
            if ok and stdout.strip() == expected:
                passed += 1
        return passed / max(len(test_cases), 1)
    return q_reward_func
'''
    s = scan_source(WITHDRAWN, "withdrawn")
    regressed = sorted({f.rule for f in s.findings}
                       & {"visible_test_only", "stdout_spoof"})
    failures += 1 if regressed else 0
    print("%-26s %s  (%s)"
          % ("withdrawn_fps(negative)", "ok" if not regressed else "REGRESSED",
             regressed or [f.rule for f in s.findings] or "clean"))

    # Syntax-check-then-ship-elsewhere. Modelled on long-code-edit, which
    # compiles the response only to validate it and does the real execution with
    # a POST to a remote evaluator. compile() runs nothing, so neither
    # `no_test_isolation` nor `timeout_evade` may fire; the remote call makes
    # this a coverage gap, not a clean grader.
    COMPILE_ONLY = '''
import requests
def detection_reward_func(completion, answer):
    code = extract(completion)
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return 0.0
    r = requests.post(f"{evaluator_url}/evaluate",
                      json={"solution": code, "test_code": answer}, timeout=60)
    return float(r.json()["passed"])
'''
    s = scan_source(COMPILE_ONLY, "compile_only")
    bad = sorted({f.rule for f in s.findings} & {"no_test_isolation", "timeout_evade"})
    failures += 1 if bad else 0
    print("%-26s %s  (%s)"
          % ("compile_is_not_exec(neg)", "ok" if not bad else "FALSE POSITIVE",
             bad or [f.rule for f in s.findings] or "clean"))

    # The first genuine high-severity finding the code family produced, kept as
    # a control so a later tightening cannot silently retire it. Modelled on
    # concurrency-bench: the response's module is imported by pytest at
    # collection time, so anything it prints reaches `out` before pytest's own
    # summary line, and `re.search` takes the first match. Printing
    # "9999 passed" clears the gate and inflates the reward without
    # implementing anything.
    REAL_HIT = '''
import re, subprocess, sys
def _run_tier(workdir, test_name):
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test_name)],
                          cwd=str(workdir), capture_output=True, text=True, timeout=60)
    out = proc.stdout + proc.stderr
    m = re.search(r"(\\\\d+)\\\\s+passed", out)
    return int(m.group(1)) if m else 0

def _grade_sync(task_id, module, code):
    (tmp / f"{module}.py").write_text(code, encoding="utf-8")
    return 0.2 + 0.5 * _run_tier(tmp, "core")
'''
    s = scan_source(REAL_HIT, "real_hit")
    ok = "stdout_spoof" in {f.rule for f in s.findings}
    failures += 0 if ok else 1
    print("%-26s %s  (%s)"
          % ("printed_count(real hit)", "ok" if ok else "LOST",
             sorted({f.rule for f in s.findings}) or "clean"))

    # Coverage classification. A container-delegated grader must land in
    # "opaque", never in "clean": its test commands live in a Docker image or a
    # run_tests.sh, so a source pass that reported it clean would be inventing
    # coverage it does not have. Modelled on terminal-bench / kernelbench, the
    # shape most of the Hub's real code environments actually use.
    OPAQUE = '''
def reward_correctness(completion, state, session):
    session.copy_to_container(paths=[state["run_tests_path"]], container_dir="/tests")
    session.send_keys(["bash /tests/run_tests.sh", "Enter"], block=True)
    return 1.0 if state.get("is_resolved") else 0.0
'''
    s = scan_source(OPAQUE, "opaque")
    ok = bool(s.opaque_grader_funcs) and not s.code_grader_funcs
    failures += 0 if ok else 1
    print("%-26s %s  (opaque=%s code=%s findings=%s)"
          % ("delegated_exec(coverage)", "ok" if ok else "MISCLASSIFIED",
             s.opaque_grader_funcs, s.code_grader_funcs,
             [f.rule for f in s.findings] or "none"))

    # ...and the mirror of it. Subject matter is not mechanism: this is
    # `veroseo/docker-container-manager`, a pure text rubric that scores a
    # response by searching it for keywords, one of which happens to be
    # "docker". It executes nothing and delegates nothing. Classifying it as an
    # unreadable coverage gap inflated the opaque tier -- the tier whose whole
    # job is to say honestly what we could not see. Kept as a negative control
    # so the token-match version cannot come back.
    #
    # Deliberately NOT paired with a kernelbench control: kernelbench really
    # does dispatch to Modal sandboxes, so `opaque` is the correct verdict there
    # and pinning it as clean would encode a false expectation.
    SUBJECT_NOT_MECHANISM = '''
def structure_score(completion, answer, **kwargs):
    lower = completion.lower()
    score = 0.0
    for kw in ["from ", "version:", "services:", "image:", "docker", "run_tests"]:
        if kw in lower:
            score += 0.3
    return min(score, 1.0)
'''
    s = scan_source(SUBJECT_NOT_MECHANISM, "subject_not_mechanism")
    ok = not s.opaque_grader_funcs and not s.code_grader_funcs
    failures += 0 if ok else 1
    print("%-26s %s  (opaque=%s code=%s)"
          % ("subject_not_mech(neg)", "ok" if ok else "FALSE OPAQUE",
             s.opaque_grader_funcs or "clean", s.code_grader_funcs or "clean"))

    # Withdrawn `except_returns_reward` findings, kept so they cannot return.
    # This rule went 0-for-6 at high severity on the corpus: every hit was read
    # against real source by two independent reviewers and withdrawn. Each shape
    # below is one of those, reduced to its mechanism. None may be `high` again.
    WITHDRAWN_EXCEPT = [
        # Consolation value far below the
        # function's own success paths.
        ("consolation_vs_peak", '''
def format_reward_func(completion, answer):
    try:
        parsed = parse_structure(completion)
        return 1.0 if parsed.clean else 0.5
    except Exception:
        return 0.1
'''),
        # The handler returns the *lowest* value in the function.
        ("lowest_return", '''
def sudoku_accuracy(completion, answer):
    try:
        grid = extract_grid(completion)
        if verify(grid):
            return 1.0
        return 0.2
    except (ValueError, IndexError):
        return 0.1
'''),
        # 1% of a correct answer, with no literal success path to
        # compare against -- caught by the absolute floor instead.
        ("token_consolation", '''
def rf(completion, answer):
    try:
        matches = re.findall(r"box", completion)
        return answer == matches[0]
    except:
        return 0.01
'''),
        # The same value is the documented neutral on two
        # non-exception paths.
        ("documented_neutral", '''
def _distilled_score(completion, answer):
    match = find_json(completion)
    if not match:
        return 0.5
    try:
        counts = json.loads(match)
        return score_from(counts)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.5
'''),
        # Registered with weight 0.0, so it is a logged metric and
        # nothing it returns can be gamed for score.
        ("zero_weighted_metric", '''
def mape(completion, answer):
    try:
        y = float(str(answer).strip())
        return abs(y - parse(completion)) / y
    except Exception:
        return 1.0

def build():
    return vf.Rubric(funcs=[numeric_correctness, mape], weights=[1.0, 0.0])
'''),
    ]
    for label, src in WITHDRAWN_EXCEPT:
        s = scan_source(src, label)
        sev = [f.severity for f in s.findings if f.rule == "except_returns_reward"]
        bad = [x for x in sev if x in ("high", "medium")]
        failures += 0 if not bad else 1
        print("%-26s %s  (except severities=%s)"
              % ("withdrawn:" + label[:15], "ok" if not bad else "REGRESSED",
                 sev or "not raised"))

    # ...and the other direction, so the gates cannot quietly retire the rule.
    # A handler that pays the full success value is still a high-severity smell.
    s = scan_source(CASES[2][1], "except_still_high")
    sev = {f.rule: f.severity for f in s.findings}
    ok = sev.get("except_returns_reward") == "high"
    failures += 0 if ok else 1
    print("%-26s %s  (severity=%s)"
          % ("except_still_high", "ok" if ok else "RULE RETIRED",
             sev.get("except_returns_reward")))

    # Every rule with a positive control, so a rule added without one is caught
    # here rather than discovered as a permanent zero in the corpus report.
    covered = set()
    for expect, src in CASES:
        covered.add(expect)
    covered.add("except_returns_reward")
    declared = {"substring_containment", "unanchored_regex", "except_returns_reward",
                "eval_on_output", "subprocess_no_timeout", "wide_numeric_tolerance",
                "no_answer_marker", "visible_test_only", "stdout_spoof",
                "exit_code_only", "assert_disabled", "test_file_writable",
                "no_test_isolation", "timeout_evade"}
    missing = sorted(declared - covered)
    if missing:
        failures += 1
        print("%-26s MISSING CONTROL  %s" % ("rule_coverage", missing))
    else:
        print("%-26s ok  (%d rules)" % ("rule_coverage", len(declared)))

    print("\n%d failure(s)" % failures)
    raise SystemExit(1 if failures else 0)
