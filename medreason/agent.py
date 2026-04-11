"""Prior authorization agent — zero-shot and memory-augmented modes."""

from __future__ import annotations

import json
import time

import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from .ontology import AgentResult, BenchmarkCase, Outcome

SYSTEM_PROMPT = """You are a prior authorization reviewer for healthcare claims. You must determine whether to approve, deny, or note that a case would be overturned on appeal.

You are given clinical notes and basic case information. You do NOT have access to the full payer policy manual — you must rely on your general knowledge of payer behavior, medical necessity standards, and coding rules.

IMPORTANT: Prior authorization is adversarial. Many cases that LOOK approvable have hidden disqualifiers:
- Coding mismatches (wrong E/M level, preventive vs problem-oriented)
- Unmet prerequisites (conservative treatment duration, frequency limits)
- Perioperative contraindications (uncontrolled comorbidities, cardiac clearance)
- Administrative exclusions (employment physicals, administrative purposes)
- Concurrent condition requirements (substance abuse treatment for co-occurring disorders)
- Repeat imaging restrictions (timeframe rules for re-imaging)

Do NOT default to approval. Scrutinize every case for disqualifiers.

You MUST respond with ONLY valid JSON in this exact format:
{
  "determination": "approved" or "denied" or "overturned_on_appeal",
  "reasoning_chain": "Your complete step-by-step reasoning as a single string",
  "confidence": 0.0 to 1.0,
  "key_factors": ["factor1", "factor2", "factor3"]
}"""


def _build_case_prompt(case: BenchmarkCase, include_policy: bool = False) -> str:
    """Build case prompt. When include_policy=False, agent must reason without policy details."""
    parts = [
        "## Prior Authorization Review",
        f"**Payer:** {case.task_config.payer.value}",
        f"**CPT Code:** {case.task_config.cpt_code}",
        f"**ICD-10:** {', '.join(case.task_config.icd10_codes)}",
        f"**Facility:** {case.task_config.facility_type.value}",
    ]
    if case.task_config.modifiers:
        parts.append(f"**Modifiers:** {', '.join(case.task_config.modifiers)}")
    parts.extend([
        "",
        f"### Clinical Notes\n{case.clinical_notes}",
    ])

    if include_policy:
        parts.extend([
            "",
            f"### Payer Policy Excerpt\n{case.policy_excerpt}",
        ])
    else:
        parts.extend([
            "",
            "### Policy\nYou do not have access to the specific payer policy. "
            "Use your general knowledge of prior authorization standards, "
            "medical necessity criteria, and payer behavior for this determination.",
        ])

    if case.prior_eobs:
        parts.append("\n### Prior EOBs")
        for eob in case.prior_eobs:
            parts.append(f"- {eob}")

    parts.append("\nProvide your determination as JSON.")
    return "\n".join(parts)


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def run_zero_shot(case: BenchmarkCase) -> AgentResult:
    """Run agent with no memory injection and no policy excerpt."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)
    user_msg = _build_case_prompt(case, include_policy=False)

    start = time.time()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = (time.time() - start) * 1000

    data = _parse_response(response.content[0].text)
    determination = Outcome(data["determination"])

    return AgentResult(
        case_id=case.case_id,
        mode="zero_shot",
        determination=determination,
        reasoning_chain=data.get("reasoning_chain", ""),
        confidence=data.get("confidence", 0.5),
        key_factors=data.get("key_factors", []),
        correct=determination == case.ground_truth_outcome,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency,
    )


def run_with_memory(case: BenchmarkCase, injection: str, pattern_ids: list[str]) -> AgentResult:
    """Run agent with memory injection — patterns encode payer-specific policy knowledge."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

    system = SYSTEM_PROMPT
    if injection:
        system = injection + "\n\n" + system

    # Memory-augmented agent also doesn't get the raw policy excerpt —
    # instead it gets structured institutional memory patterns that encode
    # what the payer actually cares about (learned from prior cases)
    user_msg = _build_case_prompt(case, include_policy=False)

    start = time.time()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = (time.time() - start) * 1000

    data = _parse_response(response.content[0].text)
    determination = Outcome(data["determination"])

    return AgentResult(
        case_id=case.case_id,
        mode="memory",
        determination=determination,
        reasoning_chain=data.get("reasoning_chain", ""),
        confidence=data.get("confidence", 0.5),
        key_factors=data.get("key_factors", []),
        correct=determination == case.ground_truth_outcome,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency,
        injected_pattern_ids=pattern_ids,
    )
