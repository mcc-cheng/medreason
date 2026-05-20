"""Google Gemini LLMClient — skeleton.

Phase 5 ships this as a NotImplementedError stub. Phase 7 will wire it up
by mirroring ClaudeLLMClient against the google-genai SDK with
response_mime_type="application/json" and GenerationConfig(seed=seed).
"""

from __future__ import annotations

from typing import Any, Optional

from .base import LLMResponse


GEMINI_LLM_PRICING: dict[str, dict[str, float]] = {
    "gemini-1.5-pro-002": {
        "input_per_mtok": 1.25,
        "output_per_mtok": 5.0,
    },
    "gemini-2.0-flash-001": {
        "input_per_mtok": 0.10,
        "output_per_mtok": 0.40,
    },
}

DEFAULT_GEMINI_LLM_MODEL = "gemini-1.5-pro-002"


class GeminiLLMClient:
    model_version: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_GEMINI_LLM_MODEL,
    ):
        if model not in GEMINI_LLM_PRICING:
            raise ValueError(
                f"Unknown Gemini model {model!r}. Known: "
                f"{sorted(GEMINI_LLM_PRICING.keys())}"
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
            "GeminiLLMClient is a Phase 7 skeleton. Implement by mirroring "
            "ClaudeLLMClient against google.genai.Client().models.generate_content "
            "with system_instruction=system, contents=user, and "
            "GenerationConfig(response_mime_type='application/json', seed=seed)."
        )
