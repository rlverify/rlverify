"""
Adapter for Prime Intellect `verifiers` v0 rubrics (978 of 999 catalogued
environments depend on `verifiers`, so this one adapter covers ~98% of the Hub).

Three facts about the v0.2.1 API that are not in its docs and that cost real
time to discover. All three are encoded here so nobody rediscovers them:

  1. `Rubric.score_objects()` reads `prompt`, `answer` and `info` from the TOP
     LEVEL of `State`, not from `state["input"]`. Build the state wrong and
     `answer` silently becomes "" -- every grader then returns 0.0 and the
     environment looks immaculate. This is the single most dangerous failure
     mode for an auditing tool: it manufactures clean results.

  2. `Rubric.score_rollout(state)` returns None. It mutates the state in place,
     writing `state["reward"]` and `state["metrics"]`.

  3. `Rubric._call_individual_reward_func` wraps the call in
     `try/except -> ans = 0.0`. A grader that raises is indistinguishable from
     one that scored zero. We defend against the resulting false-clean by
     probing with a known-good response first (see `self_check`).

Requires Python >=3.11 because `verifiers` does.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from rlverify.targets.base import (
    Capabilities,
    GraderTarget,
    GradeResult,
    Response,
    TaskInstance,
)


def _require_verifiers():
    try:
        import verifiers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the verifiers adapter needs `pip install 'rlverify[envs]'` on "
            "Python >=3.11 (verifiers requires >=3.11,<3.14)"
        ) from exc
    return verifiers


class VerifiersV0Target(GraderTarget):
    """Wraps a `verifiers` Rubric plus a task set as a pure gradeable function."""

    def __init__(self, rubric: Any, tasks: Sequence[TaskInstance], id: str = "verifiers") -> None:
        _require_verifiers()
        self.rubric = rubric
        self._tasks = list(tasks)
        self.id = id
        self._caps: Optional[Capabilities] = None

    # ------------------------------------------------------------------ state

    @staticmethod
    def _make_state(task: TaskInstance, response: Response):
        from verifiers.types import State

        # prompt/answer/info live at the TOP LEVEL -- see module docstring (1).
        return State(
            prompt=task.prompt,
            answer=task.reference if task.reference is not None else "",
            info=dict(task.info or {}),
            completion=response,
            task=task.task,
            trajectory=[],
            is_completed=True,
            metrics={},
            timing={},
        )

    # ------------------------------------------------------------------ public

    def tasks(self) -> Sequence[TaskInstance]:
        return self._tasks

    async def grade(self, task: TaskInstance, response: Response) -> GradeResult:
        state = self._make_state(task, response)
        t0 = time.perf_counter()
        err: Optional[str] = None
        try:
            await self.rubric.score_rollout(state)      # returns None -- see (2)
        except Exception as exc:
            err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        dt = time.perf_counter() - t0

        reward = state.get("reward")
        if reward is None:
            return GradeResult(reward=0.0, error=err or "grader produced no reward",
                               latency_s=dt)
        metrics = state.get("metrics") or {}
        return GradeResult(
            reward=float(reward),
            components={k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float))},
            error=err,
            latency_s=dt,
        )

    def capabilities(self) -> Capabilities:
        if self._caps is not None:
            return self._caps
        import inspect

        funcs = list(getattr(self.rubric, "funcs", []) or [])
        src = ""
        for f in funcs:
            try:
                src += inspect.getsource(f)
            except (OSError, TypeError):
                pass
        executes = any(tok in src for tok in ("subprocess", "exec(", "eval(",
                                              "compile(", "sandbox"))
        network = any(tok in src for tok in ("requests.", "httpx.", "urlopen",
                                             "aiohttp", "openai"))
        judged = any(tok in src for tok in ("client.chat", "AsyncOpenAI", "OpenAI(",
                                            "judge"))
        self._caps = Capabilities(
            pure=not (executes or network or judged),
            executes_response=executes,
            needs_network=network or judged,
            usd_per_call=0.002 if judged else 0.0,
            notes="%d reward func(s): %s" % (
                len(funcs), ", ".join(getattr(f, "__name__", "?") for f in funcs[:6])),
        )
        return self._caps

    async def teardown(self) -> None:
        td = getattr(self.rubric, "teardown", None)
        if td is None:
            return
        try:
            r = td()
            if hasattr(r, "__await__"):
                await r
        except Exception:
            pass                                       # teardown must never mask a result

    # ------------------------------------------------------- integrity guard

    async def self_check(self) -> Dict[str, Any]:
        """Verify the grader rewards its own gold answers before we trust a clean result.

        Because the harness swallows grader exceptions into 0.0 (see (3)), an
        environment we have wired up incorrectly looks exactly like an
        environment that is impossible to hack. Any target whose gold answers
        do not score is `inconclusive`, never `clean` -- refusing to report is
        the whole point.
        """
        checked = passed = 0
        errors: List[str] = []
        for t in self._tasks[:10]:
            if t.reference is None:
                continue
            checked += 1
            g = await self.grade(t, str(t.reference))
            if g.reward > 0:
                passed += 1
            if g.error:
                errors.append(g.error)
        usable = checked > 0 and passed > 0
        return {
            "checked": checked,
            "gold_scored": passed,
            "usable": usable,
            "verdict": "ok" if usable else "inconclusive",
            "reason": ("" if usable else
                       "gold answers did not score -- adapter wiring or task set is "
                       "wrong, so a clean audit result would be meaningless"),
            "errors": errors[:3],
        }


def from_callables(reward_funcs, tasks: Sequence[TaskInstance],
                   weights=None, id: str = "inline") -> VerifiersV0Target:
    """Build a target from bare reward functions -- used by tests and fixtures."""
    _require_verifiers()
    from verifiers.rubrics.rubric import Rubric

    rubric = Rubric(funcs=list(reward_funcs),
                    **({"weights": list(weights)} if weights else {}))
    return VerifiersV0Target(rubric, tasks, id=id)
