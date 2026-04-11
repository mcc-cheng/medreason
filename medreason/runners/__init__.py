"""medreason.runners — AgentRunner Protocol and concrete adapters.

The AgentRunner Protocol is the single integration point between the
memory pipeline / eval harness and whichever LLM provider is running
the base calls. Swap Claude for GPT-4 for Gemini by instantiating a
different adapter; nothing downstream changes.

Phase 2 ships a wired ClaudeRunner and structurally complete OpenAIRunner
/ GeminiRunner skeletons that raise NotImplementedError until Phase 7.
"""

from .base import AgentRunner
from .claude import (
    CLAUDE_MODEL_ALIASES,
    CLAUDE_PRICING,
    DEFAULT_CLAUDE_MODEL,
    HAIKU_CLAUDE_MODEL,
    ClaudeRunner,
    resolve_claude_model,
)
from .gemini import GEMINI_PRICING, DEFAULT_GEMINI_MODEL, GeminiRunner
from .openai import DEFAULT_OPENAI_MODEL, OPENAI_PRICING, OpenAIRunner
from ._prompting import (
    ResponseParseError,
    build_case_prompt,
    compute_cost_usd,
    parse_json_response,
)


# MemoryRunner is intentionally lazy-imported via PEP 562 __getattr__.
# Loading it eagerly would create a cycle: medreason.runners ->
# memory_wrapper -> medreason.retrieval -> medreason.retrieval.injector ->
# medreason.runners._prompting -> medreason.runners (circular). Lazy
# import means MemoryRunner only resolves when a caller actually
# touches it, after both packages are fully initialized.
def __getattr__(name):
    if name in ("MemoryRunner", "MemoryRunnerStats"):
        from .memory_wrapper import MemoryRunner, MemoryRunnerStats
        if name == "MemoryRunner":
            return MemoryRunner
        return MemoryRunnerStats
    raise AttributeError(f"module 'medreason.runners' has no attribute {name!r}")

__all__ = [
    # Protocol
    "AgentRunner",
    # Adapters
    "ClaudeRunner",
    "OpenAIRunner",
    "GeminiRunner",
    # Memory composition
    "MemoryRunner",
    "MemoryRunnerStats",
    # Pricing
    "CLAUDE_PRICING",
    "CLAUDE_MODEL_ALIASES",
    "OPENAI_PRICING",
    "GEMINI_PRICING",
    "DEFAULT_CLAUDE_MODEL",
    "HAIKU_CLAUDE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "resolve_claude_model",
    # Shared helpers
    "build_case_prompt",
    "parse_json_response",
    "ResponseParseError",
    "compute_cost_usd",
]
