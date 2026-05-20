"""Critic — independent re-derivation of the case outcome.

The critic is run AFTER the agent has produced a determination on a
training case. It is given ONLY the case inputs (task config, clinical
notes, policy excerpt) and NEVER sees:

- the agent's reasoning chain
- the agent's determination
- the ground-truth outcome or ground-truth reasoning

It reaches its own determination by reasoning through the policy
criteria. A trace is kept for rule proposal only if:

1. The agent was actually correct (agent_determination == ground_truth).
2. The critic's determination agrees with the agent's.

The dual gate catches right-answer-wrong-reasoning failures: if the
agent lucked into the correct determination via bad reasoning, the
critic — a different model with no view of the agent's chain — would
likely reach a different conclusion and block the trace from entering
memory.

For meaningful independence, pass an LLMClient from a DIFFERENT vendor
than the base agent (e.g., ClaudeLLMClient for agent + OpenAILLMClient
for critic). Using the same vendor shares blind spots and weakens the
signal — the plan's risk #1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..llm.base import LLMClient
from ..ontology.case import BenchmarkCase, Outcome
from ..ontology.trace import (
    ArgumentFraming,
    ArgumentStructure,
    ReasoningStep,
    ReasoningTrace,
)
from ..prompts import load_prompt
from ..runners._prompting import (
    ResponseParseError,
    build_case_prompt,
    parse_json_response,
)


CRITIC_PROMPT_FILE = "critic.txt"


@dataclass
class CriticResult:
    """Outcome of one critic invocation.

    - `agrees`: True iff the critic reached the same determination as
      the agent. `trace` is stored only in this case.
    - `reason`: "agreement" | "disagreement" | "parse_error" | "llm_error".
    - `critic_determination`: what the critic decided, for audit logs.
      Set to None on parse/llm errors.
    - `trace`: the ReasoningTrace built from the critic's reasoning,
      with source="critic". Stored only on agreement.
    - `raw_response_text`: the LLM's raw text, kept for debugging.
    """
    agrees: bool
    reason: str
    critic_determination: Optional[Outcome]
    trace: Optional[ReasoningTrace]
    raw_response_text: str = ""


def run_critic(
    case: BenchmarkCase,
    agent_determination: Outcome,
    llm_client: LLMClient,
    *,
    include_policy: bool = True,
    max_tokens: int = 2048,
    seed: int = 0,
) -> CriticResult:
    """Run the critic on `case` and decide whether to trust a trace.

    Args:
        case: The resolved case.
        agent_determination: The agent's determination on this case.
            Used only for the final agreement check — NEVER passed into
            the critic's prompt.
        llm_client: LLMClient instance. Production: different vendor
            than the base agent. Tests: FakeLLMClient.
        include_policy: Whether to include case.policy_excerpt in the
            user message. The plan says yes — critic gets the policy,
            agent does not. This asymmetry is intentional: the critic
            acts as an oracle reasoner against the policy text.
        max_tokens: Anthropic/OpenAI token limit forwarded to the LLM.
        seed: Reproducibility seed forwarded to the LLM.
    """
    system_prompt = load_prompt(CRITIC_PROMPT_FILE)
    user_msg = build_case_prompt(case, include_policy=include_policy)

    try:
        response = llm_client.complete(
            system=system_prompt,
            user=user_msg,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception as e:
        return CriticResult(
            agrees=False,
            reason=f"llm_error: {type(e).__name__}",
            critic_determination=None,
            trace=None,
            raw_response_text="",
        )

    raw = response.text or ""
    try:
        data = parse_json_response(raw)
    except ResponseParseError:
        return CriticResult(
            agrees=False,
            reason="parse_error",
            critic_determination=None,
            trace=None,
            raw_response_text=raw,
        )

    det_raw = data.get("determination")
    if not isinstance(det_raw, str):
        return CriticResult(
            agrees=False,
            reason="parse_error",
            critic_determination=None,
            trace=None,
            raw_response_text=raw,
        )
    try:
        critic_det = Outcome(det_raw.strip().lower().replace(" ", "_"))
    except ValueError:
        return CriticResult(
            agrees=False,
            reason="parse_error",
            critic_determination=None,
            trace=None,
            raw_response_text=raw,
        )

    # Build a ReasoningTrace from the critic's output. Note the trace
    # records the CRITIC'S reasoning, not the agent's — that's the whole
    # point of independent re-derivation.
    steps_data = data.get("reasoning_steps") or []
    steps: list[ReasoningStep] = []
    if isinstance(steps_data, list):
        for i, s in enumerate(steps_data, start=1):
            if not isinstance(s, dict):
                continue
            try:
                step_num = int(s.get("step", i))
            except (TypeError, ValueError):
                step_num = i
            steps.append(
                ReasoningStep(
                    step_number=step_num,
                    action=str(s.get("claim", "")),
                    evidence_cited=str(s.get("evidence", "")),
                    decision_branch=str(s.get("policy_hook", "")),
                )
            )

    trace = ReasoningTrace(
        task_config=case.task_config,
        reasoning_steps=steps,
        argument_structure=ArgumentStructure(
            framing=ArgumentFraming.GUIDELINE_ADHERENCE,
            clinical_evidence_elements=[],
            payer_policy_hooks=[],
        ),
        uncertainty_flags=[],
        raw_chain_of_thought=raw,
        outcome=critic_det,
        source="critic",
    )

    if critic_det == agent_determination:
        return CriticResult(
            agrees=True,
            reason="agreement",
            critic_determination=critic_det,
            trace=trace,
            raw_response_text=raw,
        )
    return CriticResult(
        agrees=False,
        reason="disagreement",
        critic_determination=critic_det,
        trace=None,  # never keep a disagreement trace
        raw_response_text=raw,
    )
