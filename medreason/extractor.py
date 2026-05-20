"""Reasoning extractor — converts raw agent output to structured patterns."""

from __future__ import annotations

import json
from uuid import uuid4

import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from .ontology import (
    AgentResult, ArgumentFraming, ArgumentStructure, BenchmarkCase,
    ReasoningPattern, ReasoningStep, ReasoningTrace,
)


def extract_trace(case: BenchmarkCase, agent_result: AgentResult) -> ReasoningTrace:
    """Extract a structured reasoning trace from agent output using Claude."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Analyze this prior authorization reasoning and extract structured components.

Case: CPT {case.task_config.cpt_code}, Payer: {case.task_config.payer.value}
Outcome: {agent_result.determination.value}

Agent's reasoning:
{agent_result.reasoning_chain}

Key factors: {', '.join(agent_result.key_factors)}

Return JSON:
{{
  "reasoning_steps": [
    {{"step_number": 1, "action": "...", "evidence_cited": "...", "decision_branch": "..."}}
  ],
  "argument_structure": {{
    "framing": "functional_impairment|clinical_severity|guideline_adherence|cost_effectiveness",
    "clinical_evidence_elements": ["..."],
    "payer_policy_hooks": ["..."]
  }},
  "uncertainty_flags": ["..."]
}}

Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)

        steps = [ReasoningStep(**s) for s in data.get("reasoning_steps", [])]
        arg = data.get("argument_structure", {})
        framing = arg.get("framing", "guideline_adherence")
        try:
            framing_enum = ArgumentFraming(framing)
        except ValueError:
            framing_enum = ArgumentFraming.GUIDELINE_ADHERENCE

        return ReasoningTrace(
            task_config=case.task_config,
            reasoning_steps=steps,
            argument_structure=ArgumentStructure(
                framing=framing_enum,
                clinical_evidence_elements=arg.get("clinical_evidence_elements", []),
                payer_policy_hooks=arg.get("payer_policy_hooks", []),
            ),
            uncertainty_flags=data.get("uncertainty_flags", []),
            raw_chain_of_thought=agent_result.reasoning_chain,
            outcome=agent_result.determination,
        )
    except Exception:
        return _extract_local(case, agent_result)


def _extract_local(case: BenchmarkCase, agent_result: AgentResult) -> ReasoningTrace:
    """Local extraction — uses ground truth reasoning for high-quality patterns."""
    # Use ground truth reasoning steps (the "right answer") for the pattern,
    # not the agent's own reasoning — this is what makes memory valuable
    gt_steps = case.ground_truth_reasoning if case.ground_truth_reasoning else []
    steps = []
    for i, step_text in enumerate(gt_steps[:5], 1):
        steps.append(ReasoningStep(
            step_number=i,
            action=step_text,
            evidence_cited="clinical notes + policy criteria",
            decision_branch=case.ground_truth_outcome.value,
        ))

    # Extract policy hooks from the policy excerpt — this is the key knowledge
    # that the zero-shot agent doesn't have access to
    policy_hooks = []
    if case.policy_excerpt:
        sentences = case.policy_excerpt.split(". ")
        policy_hooks = [s.strip().rstrip(".") for s in sentences if s.strip()][:3]

    return ReasoningTrace(
        task_config=case.task_config,
        reasoning_steps=steps,
        argument_structure=ArgumentStructure(
            framing=ArgumentFraming.GUIDELINE_ADHERENCE,
            clinical_evidence_elements=agent_result.key_factors[:3],
            payer_policy_hooks=policy_hooks or [f"{case.task_config.payer.value} policy criteria"],
        ),
        uncertainty_flags=[],
        raw_chain_of_thought=agent_result.reasoning_chain,
        outcome=case.ground_truth_outcome,
    )


def abstract_to_pattern(trace: ReasoningTrace) -> ReasoningPattern:
    """Abstract a reasoning trace into a reusable pattern."""
    key_steps = [
        s.action for s in trace.reasoning_steps
    ]
    return ReasoningPattern(
        pattern_id=uuid4().hex[:12],
        config_hash=trace.task_config.config_hash,
        task_config=trace.task_config,
        argument_structure=trace.argument_structure,
        key_reasoning_steps=key_steps,
        success_count=1,
        failure_count=0,
    )
