"""Low-level LLM clients used by Phase 5 extraction + retrieval code.

Distinct from medreason.runners: runners wrap a full case → determination
flow, while llm clients do bare (system, user) → text. Critic, rule
proposer, reranker, and generalization gate use llm clients directly.
"""

from .base import FakeLLMClient, LLMClient, LLMResponse
from .claude import ClaudeLLMClient
from .gemini import GEMINI_LLM_PRICING, DEFAULT_GEMINI_LLM_MODEL, GeminiLLMClient
from .openai import (
    DEFAULT_OPENAI_LLM_MODEL,
    OPENAI_LLM_PRICING,
    OpenAILLMClient,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "FakeLLMClient",
    "ClaudeLLMClient",
    "OpenAILLMClient",
    "GeminiLLMClient",
    "OPENAI_LLM_PRICING",
    "GEMINI_LLM_PRICING",
    "DEFAULT_OPENAI_LLM_MODEL",
    "DEFAULT_GEMINI_LLM_MODEL",
]
