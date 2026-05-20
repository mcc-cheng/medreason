"""Anthropic Claude adapter for LLMClient.

Shares pricing + the lazy-client pattern with runners.claude but serves
a different purpose: bare `complete(system, user) -> LLMResponse`. No
case parsing, no AgentResult construction, no applied_rules logic.

Use this for critic, rule proposer, reranker, and generalization gate
calls — anywhere Phase 5 code needs raw text from an LLM without the
prior-auth determination schema wrapped around it.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ..config import ANTHROPIC_API_KEY
from ..runners._prompting import compute_cost_usd
from ..runners.claude import CLAUDE_PRICING, DEFAULT_CLAUDE_MODEL
from .base import LLMResponse


class ClaudeLLMClient:
    model_version: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        timeout_sec: float = 60.0,
    ):
        if model not in CLAUDE_PRICING:
            raise ValueError(
                f"Unknown Claude model {model!r}. Known: "
                f"{sorted(CLAUDE_PRICING.keys())}"
            )
        self._api_key = api_key if api_key is not None else ANTHROPIC_API_KEY
        self.model_version = model
        self._timeout_sec = timeout_sec
        self._client: Any = None  # lazy; tests inject directly

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "ClaudeLLMClient requires ANTHROPIC_API_KEY or an api_key "
                "argument, or a test client injected into ._client."
            )
        import anthropic  # noqa: PLC0415
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=self._timeout_sec,
        )
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        seed: int = 0,
    ) -> LLMResponse:
        client = self._get_client()
        response = client.messages.create(
            model=self.model_version,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = _extract_text(response)
        in_tok = _usage(response, "input_tokens", 0)
        out_tok = _usage(response, "output_tokens", 0)
        pricing = CLAUDE_PRICING[self.model_version]
        cost = compute_cost_usd(
            in_tok, out_tok,
            pricing["input_per_mtok"], pricing["output_per_mtok"],
        )
        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            model=self.model_version,
        )


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)) and content:
        first = content[0]
        if isinstance(first, str):
            return first
        return getattr(first, "text", "") or ""
    return ""


def _usage(response: Any, field: str, default: int) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return default
    return int(getattr(usage, field, default))
