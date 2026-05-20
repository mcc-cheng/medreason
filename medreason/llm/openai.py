"""OpenAI LLMClient — skeleton.

Phase 5 ships this as a NotImplementedError stub so the Protocol surface
is visible but the SDK isn't required for the rework. Phase 7 will wire
it up by mirroring ClaudeLLMClient against the openai SDK's
chat.completions API with response_format={"type": "json_object"}.

The critic pipeline expects the critic LLMClient to be a DIFFERENT
vendor from the base agent. Once this is wired, pairing
ClaudeRunner(agent) + OpenAILLMClient(critic) is the production setup.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import LLMResponse


OPENAI_LLM_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-2024-11-20": {
        "input_per_mtok": 2.5,
        "output_per_mtok": 10.0,
    },
    "gpt-4-turbo-2024-04-09": {
        "input_per_mtok": 10.0,
        "output_per_mtok": 30.0,
    },
}

DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-2024-11-20"


class OpenAILLMClient:
    model_version: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_OPENAI_LLM_MODEL,
    ):
        if model not in OPENAI_LLM_PRICING:
            raise ValueError(
                f"Unknown OpenAI model {model!r}. Known: "
                f"{sorted(OPENAI_LLM_PRICING.keys())}"
            )
        self._api_key = api_key
        self.model_version = model
        self._client: Any = None

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        seed: int = 0,
    ) -> LLMResponse:
        raise NotImplementedError(
            "OpenAILLMClient is a Phase 7 skeleton. Implement by mirroring "
            "ClaudeLLMClient against openai.OpenAI().chat.completions.create "
            "with response_format={'type': 'json_object'} and the system/user "
            "messages. Cost math is already wired via OPENAI_LLM_PRICING."
        )
