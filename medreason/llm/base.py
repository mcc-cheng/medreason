"""Low-level LLM client Protocol — (system, user) → text.

`LLMClient` is intentionally narrower than `AgentRunner`:
- AgentRunner takes a BenchmarkCase and returns an AgentResult (full
  determination parsing + cost + latency + applied_rules).
- LLMClient takes a raw (system, user) pair and returns the raw text
  plus usage counts.

Critic, rule proposer, reranker, and generalization gate all call LLMs
in ways that have nothing to do with the prior-auth determination schema,
so they take `LLMClient`, not `AgentRunner`. This also makes it trivial
to run the critic on a different vendor than the agent — pass a
ClaudeLLMClient to the agent and an OpenAILLMClient (when wired) to
the critic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    model_version: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        seed: int = 0,
    ) -> LLMResponse:
        """Run one completion. Returns LLMResponse with raw text + usage."""
        ...


@dataclass
class FakeLLMClient:
    """Test helper — returns canned responses without touching any network.

    Behavior:
    - If `responses` is non-empty, each call pops one from the front (FIFO).
      Use this to simulate a multi-step flow (critic → proposer → rerank)
      where each stage has a distinct canned output.
    - If `responses` is empty, every call returns `default_text`.
    - `calls` records every (system, user) pair in order, for assertions.

    Conforms to LLMClient by structural match.
    """
    model_version: str = "fake-llm-v0"
    responses: list[str] = field(default_factory=list)
    default_text: str = "{}"
    input_tokens_per_call: int = 100
    output_tokens_per_call: int = 40
    cost_per_call_usd: float = 0.0003
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        seed: int = 0,
    ) -> LLMResponse:
        self.calls.append((system, user))
        if self.responses:
            text = self.responses.pop(0)
        else:
            text = self.default_text
        return LLMResponse(
            text=text,
            input_tokens=self.input_tokens_per_call,
            output_tokens=self.output_tokens_per_call,
            cost_usd=self.cost_per_call_usd,
            model=self.model_version,
        )
