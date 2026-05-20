"""OpenAI GPT adapter — skeleton.

Phase 2 ships this structurally complete but not wired to the SDK. It
implements the AgentRunner Protocol (runner_id, model_version,
supports_memory, run, estimated_cost_per_call) and raises
NotImplementedError from run() with a clear message about what's missing.

Phase 7 (cross-runner eval) will complete the implementation by mirroring
ClaudeRunner — same lazy-client pattern, same prompt construction, same
cost math with OpenAI's per-MTok rates.
"""

from __future__ import annotations

from typing import Any, Optional

from ..ontology.case import BenchmarkCase
from ..ontology.result import AgentResult
from ._prompting import compute_cost_usd


OPENAI_PRICING: dict[str, dict[str, float]] = {
    # Pricing as of 2025-05 snapshot — update on ship.
    "gpt-4-turbo-2024-04-09": {
        "input_per_mtok": 10.0,
        "output_per_mtok": 30.0,
    },
    "gpt-4o-2024-11-20": {
        "input_per_mtok": 2.5,
        "output_per_mtok": 10.0,
    },
}

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-11-20"


class OpenAIRunner:
    runner_id: str
    model_version: str
    supports_memory: bool

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        runner_id_suffix: str = "",
    ):
        if model not in OPENAI_PRICING:
            raise ValueError(
                f"Unknown OpenAI model {model!r}. Known: "
                f"{sorted(OPENAI_PRICING.keys())}"
            )
        self._api_key = api_key
        self.model_version = model
        self.runner_id = (
            f"{model}:{runner_id_suffix}" if runner_id_suffix else model
        )
        self.supports_memory = True
        self._client: Any = None

    def estimated_cost_per_call(self) -> float:
        p = OPENAI_PRICING[self.model_version]
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
            "OpenAIRunner is a skeleton. Implement by mirroring ClaudeRunner: "
            "lazy client init via `from openai import OpenAI`, chat.completions "
            "call with the frozen system_pa.txt prompt (prepended by "
            "system_extra), response_format={'type': 'json_object'} to force "
            "structured output, and cost computed via _prompting.compute_cost_usd "
            "against OPENAI_PRICING. Target Phase 7 for wiring."
        )
