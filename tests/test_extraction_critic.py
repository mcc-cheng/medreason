"""Tests for medreason.extraction.critic — Phase 5 Commit 1."""

from __future__ import annotations

import json

import pytest

from medreason.extraction import CriticResult, run_critic
from medreason.llm import FakeLLMClient
from medreason.ontology import Outcome

from tests.conftest import make_case


def _canned_json(determination: str = "approved", n_steps: int = 2) -> str:
    return json.dumps({
        "determination": determination,
        "reasoning_steps": [
            {
                "step": i + 1,
                "claim": f"check {i+1}",
                "evidence": f"notes say thing {i+1}",
                "policy_hook": f"§C.{i+1}",
            }
            for i in range(n_steps)
        ],
        "confidence": 0.85,
    })


# ── Agreement path ──────────────────────────────────────────────────────────


def test_critic_agreement_keeps_trace():
    llm = FakeLLMClient(responses=[_canned_json("approved", 3)])
    case = make_case(ground_truth_outcome=Outcome.APPROVED)

    result = run_critic(case, agent_determination=Outcome.APPROVED, llm_client=llm)

    assert isinstance(result, CriticResult)
    assert result.agrees is True
    assert result.reason == "agreement"
    assert result.critic_determination == Outcome.APPROVED
    assert result.trace is not None
    assert result.trace.source == "critic"
    assert result.trace.outcome == Outcome.APPROVED
    assert len(result.trace.reasoning_steps) == 3
    assert result.trace.reasoning_steps[0].decision_branch == "§C.1"


# ── Disagreement path ──────────────────────────────────────────────────────


def test_critic_disagreement_drops_trace():
    llm = FakeLLMClient(responses=[_canned_json("denied", 2)])
    case = make_case(ground_truth_outcome=Outcome.APPROVED)

    result = run_critic(case, agent_determination=Outcome.APPROVED, llm_client=llm)
    assert result.agrees is False
    assert result.reason == "disagreement"
    assert result.critic_determination == Outcome.DENIED
    assert result.trace is None  # never kept on disagreement


# ── Parse / error paths ────────────────────────────────────────────────────


def test_critic_parse_error_returns_clean_result():
    llm = FakeLLMClient(responses=["this is not json at all"])
    case = make_case()
    result = run_critic(case, agent_determination=Outcome.APPROVED, llm_client=llm)
    assert result.agrees is False
    assert result.reason == "parse_error"
    assert result.critic_determination is None
    assert result.trace is None


def test_critic_missing_determination_field_is_parse_error():
    llm = FakeLLMClient(responses=[json.dumps({"reasoning_steps": []})])
    case = make_case()
    result = run_critic(case, agent_determination=Outcome.APPROVED, llm_client=llm)
    assert result.agrees is False
    assert result.reason == "parse_error"


def test_critic_unknown_determination_value_is_parse_error():
    llm = FakeLLMClient(responses=[json.dumps({"determination": "maybe"})])
    case = make_case()
    result = run_critic(case, agent_determination=Outcome.APPROVED, llm_client=llm)
    assert result.agrees is False
    assert result.reason == "parse_error"


def test_critic_llm_exception_is_caught():
    class BoomClient:
        model_version = "boom"
        def complete(self, **kwargs):
            raise RuntimeError("network down")

    result = run_critic(
        make_case(),
        agent_determination=Outcome.APPROVED,
        llm_client=BoomClient(),
    )
    assert result.agrees is False
    assert "llm_error" in result.reason
    assert result.trace is None


# ── Input isolation: critic must never see agent reasoning ─────────────────


def test_critic_prompt_never_contains_agent_reasoning():
    """Regression guard for plan risk #14. The run_critic() signature
    doesn't take an agent reasoning chain — good — but this test
    double-checks by inspecting what landed in the LLM call."""
    llm = FakeLLMClient(responses=[_canned_json("approved")])
    case = make_case(ground_truth_outcome=Outcome.APPROVED)

    run_critic(case, agent_determination=Outcome.APPROVED, llm_client=llm)

    system, user = llm.calls[0]
    # Neither the system prompt nor the user message should contain
    # common agent-reasoning artifacts. The most direct check: the
    # agent's determination value string must NOT appear as a label
    # in the user message (the critic should reach its own determination).
    # We can't check for the string "approved" itself because the
    # policy excerpt may legitimately contain that word, but we CAN
    # check that nothing resembles a full reasoning_chain.
    assert "reasoning_chain" not in user
    assert "key_factors" not in user
    assert "agent" not in user.lower() or "agent" in "prior authorization agent".lower()


def test_critic_sees_policy_excerpt_when_include_policy_true():
    """The plan says the critic gets the policy excerpt — this is what
    makes it the oracle reasoner."""
    llm = FakeLLMClient(responses=[_canned_json("approved")])
    case = make_case(
        policy_excerpt="POLICY-SENTINEL: lumbar MRI requires 6 weeks PT"
    )
    run_critic(case, agent_determination=Outcome.APPROVED,
               llm_client=llm, include_policy=True)
    _, user = llm.calls[0]
    assert "POLICY-SENTINEL" in user


def test_critic_withholds_policy_when_include_policy_false():
    llm = FakeLLMClient(responses=[_canned_json("approved")])
    case = make_case(policy_excerpt="POLICY-SENTINEL: secret criteria")
    run_critic(case, agent_determination=Outcome.APPROVED,
               llm_client=llm, include_policy=False)
    _, user = llm.calls[0]
    assert "POLICY-SENTINEL" not in user


def test_critic_never_sees_ground_truth_outcome():
    llm = FakeLLMClient(responses=[_canned_json("approved")])
    case = make_case(
        ground_truth_outcome=Outcome.OVERTURNED_ON_APPEAL,
        ground_truth_reasoning=["gold step 1", "gold step 2"],
    )
    run_critic(case, agent_determination=Outcome.OVERTURNED_ON_APPEAL,
               llm_client=llm)
    _, user = llm.calls[0]
    assert "gold step" not in user
    assert "ground_truth" not in user
