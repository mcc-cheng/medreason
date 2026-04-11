"""medreason.runners — AgentRunner Protocol and concrete adapters.

The AgentRunner Protocol is the single integration point between the
memory pipeline / eval harness and whichever LLM provider is running
the base calls. Swap Claude for GPT-4 for Gemini by instantiating a
different adapter; nothing downstream changes.

Phase 2 ships a wired ClaudeRunner and structurally complete OpenAIRunner
/ GeminiRunner skeletons that raise NotImplementedError until Phase 7.
"""

from .base import AgentRunner
from .claude import CLAUDE_PRICING, DEFAULT_CLAUDE_MODEL, ClaudeRunner
from .gemini import GEMINI_PRICING, DEFAULT_GEMINI_MODEL, GeminiRunner
from .openai import DEFAULT_OPENAI_MODEL, OPENAI_PRICING, OpenAIRunner
from ._prompting import (
    ResponseParseError,
    build_case_prompt,
    compute_cost_usd,
    parse_json_response,
)

__all__ = [
    # Protocol
    "AgentRunner",
    # Adapters
    "ClaudeRunner",
    "OpenAIRunner",
    "GeminiRunner",
    # Pricing
    "CLAUDE_PRICING",
    "OPENAI_PRICING",
    "GEMINI_PRICING",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_GEMINI_MODEL",
    # Shared helpers
    "build_case_prompt",
    "parse_json_response",
    "ResponseParseError",
    "compute_cost_usd",
]
