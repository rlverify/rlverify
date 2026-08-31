"""
Core types every adapter and probe speaks.

Get these four right and the rest of the tool is mechanical, so they are
deliberately small and boring.

`Capabilities` is the load-bearing one. It answers two questions that decide
everything downstream:

  - How hard may we hammer this grader?  A pure Python function tolerates tens
    of thousands of calls a second for free; an LLM judge at $0.002/call gets
    fifty and a budget guard.
  - Which probe families even *apply*?  Grader-tampering probes are meaningless
    against a grader that never executes the response, and reporting "clean"
    for a probe that could not possibly fire is a lie of omission.

Kept import-light and Python 3.9 compatible on purpose: the static/corpus half
of rlverify must run on whatever interpreter a user already has, while only the
adapters that import environment code need 3.11+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

Response = Union[str, List[Dict[str, Any]]]


@dataclass(frozen=True)
class TaskInstance:
    """One graded item: a prompt, its gold answer, and whatever the env attaches."""

    task_id: str
    prompt: Response
    reference: Any = None
    info: Dict[str, Any] = field(default_factory=dict)
    task: str = "default"

    def prompt_text(self) -> str:
        if isinstance(self.prompt, str):
            return self.prompt
        parts = []
        for m in self.prompt or []:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)


@dataclass(frozen=True)
class GradeResult:
    """One grader invocation.

    `error` is separate from a zero reward on purpose. Several harnesses --
    verifiers among them -- swallow exceptions inside the reward function and
    return 0.0, which makes "the grader crashed" indistinguishable from "the
    answer was wrong". Where we can tell the difference we record it, because a
    grader that crashes on adversarial input is itself a finding.
    """

    reward: float
    components: Dict[str, float] = field(default_factory=dict)
    max_reward: Optional[float] = None
    error: Optional[str] = None
    latency_s: float = 0.0

    def passed(self, tau: float) -> bool:
        return self.reward >= tau


@dataclass(frozen=True)
class Capabilities:
    pure: bool = True
    executes_response: bool = False
    needs_network: bool = False
    usd_per_call: float = 0.0
    declared_threshold: Optional[float] = None
    max_reward: Optional[float] = None
    notes: str = ""

    def probe_budget(self, default: int = 200) -> int:
        """How many grader calls this target can absorb."""
        if self.usd_per_call > 0:
            return 50
        if not self.pure or self.needs_network:
            return max(25, default // 4)
        return default


class GraderTarget:
    """Base class for adapters. Subclasses override tasks/grade/capabilities.

    Not a typing.Protocol so it stays usable as a base class on 3.9 without
    extra imports, and so `teardown` gets a real default.
    """

    id: str = "target"

    def tasks(self) -> Sequence[TaskInstance]:
        raise NotImplementedError

    async def grade(self, task: TaskInstance, response: Response) -> GradeResult:
        raise NotImplementedError

    def capabilities(self) -> Capabilities:
        return Capabilities()

    async def teardown(self) -> None:
        """Mandatory to call, safe to override as a no-op.

        Not optional in practice: verifiers' MathRubric spins up its own
        ProcessPoolExecutor, and a corpus sweep that skips teardown leaks
        processes until the machine stops accepting new ones.
        """
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.teardown()
