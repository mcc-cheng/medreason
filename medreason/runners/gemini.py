"""Google Gemini adapter — skeleton.

Phase 2 ships this structurally complete but not wired to the SDK. It
implements the AgentRunner Protocol and raises NotImplementedError from
run() with a clear message.

Phase 7 will complete the implementation by mirroring ClaudeRunner against
google.genai, including forwarding `seed` to the GenerationConfig (Gemini
does honor a deterministic seed, so this runner will produce the tightest
multi-seed confidence intervals in the cross-runner eval).
"""

from __future__ import annotations

from typing import Any, Optional

from ..ontology.case import BenchmarkCase
from ..ontology.result import AgentResult
from ._prompting import compute_cost_usd


GEMINI_PRICING: dict[str, dict[str, float]] = {
    # Pricing as of 2025-05 snapshot — update on ship.
    "gemini-1.5-pro-002": {
        "input_per_mtok": 1.25,
        "output_per_mtok": 5.0,
    },
    "gemini-2.0-flash-001": {
        "input_per_mtok": 0.10,
        "output_per_mtok": 0.40,
    },
}

DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-002"


class GeminiRunner:
    runner_id: str
    model_version: str
    supports_memory: bool

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        runner_id_suffix: str = "",
    ):
        if model not in GEMINI_PRICING:
            raise ValueError(
                f"Unknown Gemini model {model!r}. Known: "
                f"{sorted(GEMINI_PRICING.keys())}"
            )
        self._api_key = api_key
        self.model_version = model
        self.runner_id = (
            f"{model}:{runner_id_suffix}" if runner_id_suffix else model
        )
        self.supports_memory = True
        self._client: Any = None

    def estimated_cost_per_call(self) -> float:
        p = GEMINI_PRICING[self.model_version]
        return compute_cost_usd(
            input_tokens=800,
            output_tokens=200,
            input_per_mtok=p["input_per_mtok"],
            output_per_mtok=p["output_per_mtok"],
        )

    def run(
        self,
        case: BenchmarkCase,
        *,
        seed: int = 0,
        system_extra: str = "",
    ) -> AgentResult:
        raise NotImplementedError(
            "GeminiRunner is a skeleton. Implement by mirroring ClaudeRunner: "
            "lazy client init via `from google import genai`, generate_content "
            "call with the frozen system_pa.txt as system_instruction (prepended "
            "by system_extra), GenerationConfig(seed=seed, "
            "response_mime_type='application/json') to force structured output, "
            "and cost computed via _prompting.compute_cost_usd against "
            "GEMINI_PRICING. Target Phase 7 for wiring."
        )
