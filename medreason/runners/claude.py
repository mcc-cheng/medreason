"""Anthropic Claude adapter for AgentRunner.

Pinned model: claude-sonnet-4-20250514. The pin is intentional — bumping
it must be a deliberate change that also bumps the memory_pipeline_version
on any leaderboard entry produced with the new model.

Design notes:
- Client instantiation is lazy. Tests can construct ClaudeRunner() without
  an API key, inject a FakeClient into `_client`, and exercise run() end-
  to-end without network.
- The frozen system prompt is loaded via prompts.load_prompt(). verify_lock()
  is NOT called here — it runs at the eval harness layer, before a scored
  test run. Loading the prompt during training or local development must
  not fail on lock drift.
- Seed is recorded on the AgentResult but NOT forwarded to Anthropic (the
  Messages API does not accept a seed parameter). This is still useful for
  multi-seed eval runs where Gemini/OpenAI honor the seed and Claude's
  natural variation is the noise floor.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ..config import ANTHROPIC_API_KEY
from ..ontology.case import BenchmarkCase, Outcome
from ..ontology.result import AgentResult
from ..prompts import load_prompt
from ._prompting import (
    ResponseParseError,
    build_case_prompt,
    compute_cost_usd,
    parse_json_response,
)


# Pricing in USD per 1M tokens. Update when Anthropic publishes new rates.
# Keep this dict — not a constant — so future models can live in the same
# runner with their own pricing.
CLAUDE_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
    },
    "claude-opus-4-1-20250805": {
        "input_per_mtok": 15.0,
        "output_per_mtok": 75.0,
    },
    "claude-haiku-4-5-20251001": {
        "input_per_mtok": 1.0,
        "output_per_mtok": 5.0,
    },
}

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
HAIKU_CLAUDE_MODEL = "claude-haiku-4-5-20251001"


# Convenience alias map for CLI flags: "sonnet" -> "claude-sonnet-4-20250514"
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-1-20250805",
    "haiku": "claude-haiku-4-5-20251001",
}


def resolve_claude_model(name: str) -> str:
    """Turn either a CLI alias ('haiku') or a full pinned id into a
    validated model string. Raises ValueError if unknown."""
    if name in CLAUDE_PRICING:
        return name
    if name in CLAUDE_MODEL_ALIASES:
        return CLAUDE_MODEL_ALIASES[name]
    raise ValueError(
        f"Unknown Claude model {name!r}. Known aliases: "
        f"{sorted(CLAUDE_MODEL_ALIASES.keys())}. Known pinned ids: "
        f"{sorted(CLAUDE_PRICING.keys())}"
    )
DEFAULT_MAX_TOKENS = 4096
# Memory mode (system_extra non-empty) needs more headroom because the
# agent must emit one applied_rules entry per retrieved rule on top of
# the standard determination JSON. 1024 was too tight: with 6 rules in
# the injection, Haiku consistently truncated mid-applied_rules and the
# response failed JSON parsing. 4096 has comfortable headroom.
DEFAULT_TIMEOUT_SEC = 60.0
SYSTEM_PROMPT_FILE = "system_pa.txt"


class ClaudeRunner:
    """Zero-shot (or memory-injected) Claude runner.

    `supports_memory` is True because this runner honors `system_extra`
    by prepending it to the frozen system prompt. A higher-level
    MemoryWrapper still wraps this to do retrieval+posterior updates.
    """

    runner_id: str
    model_version: str
    supports_memory: bool

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        runner_id_suffix: str = "",
    ):
        if model not in CLAUDE_PRICING:
            raise ValueError(
                f"Unknown Claude model {model!r}. Known: "
                f"{sorted(CLAUDE_PRICING.keys())}"
            )
        self._api_key = api_key if api_key is not None else ANTHROPIC_API_KEY
        self.model_version = model
        self.runner_id = (
            f"{model}:{runner_id_suffix}" if runner_id_suffix else model
        )
        self.supports_memory = True
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._client: Any = None  # lazy; tests can set directly

    # ── Client lifecycle ────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazy Anthropic client. Tests inject self._client directly."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "ClaudeRunner requires ANTHROPIC_API_KEY to be set, or an "
                "api_key= argument, or a test client injected into ._client."
            )
        import anthropic  # noqa: PLC0415 - intentional lazy import
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=self._timeout_sec,
        )
        return self._client

    # ── Cost estimate ──────────────────────────────────────────────────────

    def estimated_cost_per_call(self) -> float:
        """Rough USD per call assuming ~800 in + ~200 out tokens.

        This is a budgeting helper, not an exact number. The actual cost
        is computed from AgentResult.input_tokens / .output_tokens after
        each real call.
        """
        p = CLAUDE_PRICING[self.model_version]
        return compute_cost_usd(
            input_tokens=800,
            output_tokens=200,
            input_per_mtok=p["input_per_mtok"],
            output_per_mtok=p["output_per_mtok"],
        )

    # ── run() — the Protocol entry point ───────────────────────────────────

    def run(
        self,
        case: BenchmarkCase,
        *,
        seed: int = 0,
        system_extra: str = "",
    ) -> AgentResult:
        """Execute one Claude call and return a structured AgentResult.

        Args:
            case: The benchmark case to review.
            seed: Recorded on AgentResult. Not forwarded to Anthropic
                (Messages API has no seed parameter); natural variation
                is the noise floor for Claude multi-seed runs.
            system_extra: Optional prefix prepended to the frozen system
                prompt. Used by the memory wrapper for rule injection.
        """
        system_base = load_prompt(SYSTEM_PROMPT_FILE)
        system = (
            f"{system_extra.strip()}\n\n{system_base}" if system_extra else system_base
        )
        user_msg = build_case_prompt(case, include_policy=False)

        client = self._get_client()
        start = time.time()
        response = client.messages.create(
            model=self.model_version,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        latency_ms = (time.time() - start) * 1000.0

        # Pricing pulled out of the success path so the parse_error
        # branch can also compute real cost.
        pricing = CLAUDE_PRICING[self.model_version]

        # Extract text content. The anthropic SDK returns response.content
        # as a list of content blocks; the first one is a TextBlock for
        # non-tool-use responses.
        text = _extract_text(response)
        try:
            data = parse_json_response(text)
        except ResponseParseError:
            # Record the failure as a hard wrong-format result. The harness
            # may retry, but we never silently fabricate a determination.
            #
            # Cost: still report the real token cost — we paid for the
            # call even though the parse failed. The previous version
            # zeroed cost which made parse-error storms invisible in
            # leaderboard accounting.
            #
            # Determination: use DENIED as the conservative fallback
            # (NEVER ground_truth — that quietly inflates F1 to 1.0
            # while accuracy is 0, making the metrics lie to each
            # other). This matches _parse_outcome's "unknown -> denied"
            # convention.
            in_t = _usage(response, "input_tokens", 0)
            out_t = _usage(response, "output_tokens", 0)
            err_cost = compute_cost_usd(
                input_tokens=in_t,
                output_tokens=out_t,
                input_per_mtok=pricing["input_per_mtok"],
                output_per_mtok=pricing["output_per_mtok"],
            )
            return AgentResult(
                case_id=case.case_id,
                determination=Outcome.DENIED,
                reasoning_chain=f"[parse_error] {text[:500]}",
                confidence=0.0,
                key_factors=[],
                correct=(Outcome.DENIED == case.ground_truth_outcome),
                input_tokens=in_t,
                output_tokens=out_t,
                cost_usd=err_cost,
                latency_ms=latency_ms,
                runner_id=self.runner_id,
                seed=seed,
                mode="memory" if system_extra else "zero_shot",
            )

        determination = _parse_outcome(data.get("determination"))

        input_tokens = _usage(response, "input_tokens", 0)
        output_tokens = _usage(response, "output_tokens", 0)
        cost_usd = compute_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_per_mtok=pricing["input_per_mtok"],
            output_per_mtok=pricing["output_per_mtok"],
        )

        # Parse the applied_rules contract if the response carries one.
        # Absence is fine — zero-shot responses will have none. The
        # memory wrapper uses this to measure pattern utilization and
        # decide whether to re-prompt a missing-rule_id case.
        from ..retrieval.injector import parse_applied_rules  # noqa: PLC0415
        applied_rules = parse_applied_rules(text)

        return AgentResult(
            case_id=case.case_id,
            determination=determination,
            reasoning_chain=str(data.get("reasoning_chain", "")),
            confidence=float(data.get("confidence", 0.5)),
            key_factors=list(data.get("key_factors", []) or []),
            correct=determination == case.ground_truth_outcome,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            runner_id=self.runner_id,
            seed=seed,
            mode="memory" if system_extra else "zero_shot",
            applied_rules=applied_rules,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_text(response: Any) -> str:
    """Pull the first text content block from an Anthropic response.

    Defensive against a FakeClient that returns content as either a list
    of objects with `.text` or a single string.
    """
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


def _parse_outcome(raw: Any) -> Outcome:
    """Coerce an LLM's determination string into the Outcome enum.

    Tolerant of whitespace and case. Unknown values fall back to DENIED
    (the conservative default for an unparseable response — we never
    want to silently approve).
    """
    if raw is None:
        return Outcome.DENIED
    s = str(raw).strip().lower().replace(" ", "_")
    try:
        return Outcome(s)
    except ValueError:
        return Outcome.DENIED
