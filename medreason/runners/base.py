"""AgentRunner Protocol — the interface every model adapter implements.

The eval harness, the memory wrapper, and the generalization gate all
accept `AgentRunner` instances. Swapping Claude for GPT-4 for Gemini means
passing a different adapter; nothing else in the pipeline changes.

Required surface:

- `runner_id: str` — stable identifier for the leaderboard, e.g.
  "claude-sonnet-4-20250514" or "claude-sonnet-4-20250514:memory".
- `model_version: str` — the pinned model id (without the ":memory" tag).
- `supports_memory: bool` — True if this runner accepts a non-empty
  `system_extra` injection. Base runners can set False; the memory
  wrapper sets True.
- `run(case, *, seed=0, system_extra="") -> AgentResult` — the core call.
- `estimated_cost_per_call() -> float` — rough USD per call, used for
  budget sanity checks in the harness. A real cost is computed
  post-hoc from actual token usage in AgentResult.cost_usd.

Design notes:

- `system_extra` is the injection hook. The memory wrapper builds the
  retrieved-rules block here and passes it in. Zero-shot runners see "".
- `seed` is forwarded to the provider if it supports one (Gemini, OpenAI)
  and stored on the AgentResult for reproducibility. For providers
  without a seed parameter (Anthropic), we still record it so multi-seed
  eval runs know which seed "should have been" used.
- Protocol members are instance attributes, not @property, to keep
  isinstance(runner, AgentRunner) cheap at runtime via the @runtime_checkable
  structural check. Tests additionally assert the attributes explicitly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..ontology.case import BenchmarkCase
from ..ontology.result import AgentResult


@runtime_checkable
class AgentRunner(Protocol):
    runner_id: str
    model_version: str
    supports_memory: bool

    def run(
        self,
        case: BenchmarkCase,
        *,
        seed: int = 0,
        system_extra: str = "",
    ) -> AgentResult:
        """Execute one agent call on `case` and return the structured result.

        Args:
            case: The benchmark case to review.
            seed: Reproducibility seed. Forwarded to the provider when
                supported; always recorded on AgentResult.seed.
            system_extra: Optional prefix prepended to the system prompt.
                Used by the memory wrapper to inject retrieved rules.
        """
        ...

    def estimated_cost_per_call(self) -> float:
        """Rough USD estimate per call for budget sanity checks."""
        ...
