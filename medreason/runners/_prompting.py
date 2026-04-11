"""Shared prompting helpers used by every runner adapter.

Keeping these pure functions outside the adapters means:
- Tests can cover parsing and prompt construction without touching any SDK.
- Every adapter presents the same case view to its model — cross-runner
  comparisons on the leaderboard are apples-to-apples.
- Changes to case formatting are one edit, not three.
"""

from __future__ import annotations

import json
from typing import Any

from ..ontology.case import BenchmarkCase


# ── Case prompt builder ──────────────────────────────────────────────────────


def build_case_prompt(
    case: BenchmarkCase,
    *,
    include_policy: bool = False,
    policy_max_chars: int | None = None,
) -> str:
    """Deterministic case-view rendered for the agent.

    By default the policy excerpt is withheld — the agent must rely on
    general knowledge (or retrieved memory) to apply coverage criteria.
    Setting `include_policy=True` is the policy-chunk-RAG baseline.

    `policy_max_chars` simulates **sparse RAG** — real-world retrieval
    pipelines often surface only a chunk header / first paragraph and
    miss the operational nuance buried later in the document. When this
    is set to N, the policy excerpt is truncated to the first N
    characters with an explicit "[...truncated...]" marker so the agent
    knows it's seeing partial text. Used by the Option B experiment
    that tests whether memory can recover accuracy lost to sparse
    retrieval. Ignored when include_policy is False.
    """
    parts: list[str] = [
        "## Prior Authorization Review",
        f"**Payer:** {case.task_config.payer.value}",
        f"**CPT Code:** {case.task_config.cpt_code}",
        f"**ICD-10:** {', '.join(case.task_config.icd10_codes) or '(none)'}",
        f"**Facility:** {case.task_config.facility_type.value}",
    ]
    if case.task_config.modifiers:
        parts.append(f"**Modifiers:** {', '.join(case.task_config.modifiers)}")

    parts.extend([
        "",
        "### Clinical Notes",
        case.clinical_notes.strip() or "(no notes provided)",
    ])

    if include_policy and case.policy_excerpt:
        excerpt = case.policy_excerpt.strip()
        if policy_max_chars is not None and policy_max_chars > 0 and len(excerpt) > policy_max_chars:
            excerpt = excerpt[:policy_max_chars] + "\n\n[...truncated — only the first chunk of the policy was retrieved by RAG...]"
            section_label = "### Payer Policy Excerpt (sparse retrieval — partial text)"
        else:
            section_label = "### Payer Policy Excerpt"
        parts.extend(["", section_label, excerpt])
    else:
        parts.extend([
            "",
            "### Policy",
            "The full payer policy text is NOT provided. Use your general "
            "knowledge of prior authorization standards, medical necessity "
            "criteria, and payer behavior — plus any reasoning rules injected "
            "above this message — to reach a determination.",
        ])

    if case.prior_eobs:
        parts.append("")
        parts.append("### Prior EOBs")
        for eob in case.prior_eobs:
            parts.append(f"- {eob}")

    parts.extend(["", "Respond with ONLY the JSON schema specified in the system prompt."])
    return "\n".join(parts)


# ── JSON response parser ────────────────────────────────────────────────────


class ResponseParseError(ValueError):
    """Raised when an LLM response cannot be coerced into the expected JSON."""


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response.

    Handles three common shapes:
    1. Plain JSON: {"determination": "approved", ...}
    2. Markdown-fenced JSON: ```json\n{...}\n```
    3. JSON with stray prose: "Here is the answer: {...} Let me know if..."

    Raises ResponseParseError if none of these work. Callers should treat
    the raise as a parse failure and record a format_violation in the audit
    log; the eval harness may retry the call once in that case.
    """
    if text is None:
        raise ResponseParseError("response text is None")
    s = text.strip()
    if not s:
        raise ResponseParseError("response text is empty")

    # Strip fenced code blocks like ``` or ```json
    if s.startswith("```"):
        # drop the first fence line (may include the language tag)
        body = s.split("\n", 1)
        if len(body) == 2:
            inner = body[1]
            if inner.endswith("```"):
                inner = inner[: -len("```")]
            else:
                # find the last fence
                closing = inner.rfind("```")
                if closing != -1:
                    inner = inner[:closing]
            s = inner.strip()

    # Try direct parse first
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Fall through: locate the outermost balanced braces
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ResponseParseError(
                f"No JSON object found in response: {s[:200]!r}"
            )
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ResponseParseError(
                f"Response contains braces but is not valid JSON: {e}"
            ) from e

    if not isinstance(obj, dict):
        raise ResponseParseError(
            f"Expected JSON object, got {type(obj).__name__}: {obj!r}"
        )
    return obj


# ── Cost math ────────────────────────────────────────────────────────────────


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_per_mtok: float,
    output_per_mtok: float,
) -> float:
    """Compute USD cost from token counts and per-MTok prices.

    Rates are in USD per 1,000,000 tokens (MTok), matching how providers
    publish their prices. Returns 0.0 if both rates are 0 (e.g. a
    provider whose pricing isn't registered yet).
    """
    return (
        input_tokens * input_per_mtok + output_tokens * output_per_mtok
    ) / 1_000_000.0
