"""Tests for medreason.extraction.failure_analyzer — Phase 51."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medreason.extraction import (
    PatientIdentifierError,
    PolicyCitationError,
    ProposalResult,
    analyze_failure,
)
from medreason.llm import FakeLLMClient
from medreason.ontology import CPTFamily, ICD10Chapter, Outcome, Payer, RuleStatus
from medreason_bench.data import parse_lcd_xml

from .conftest import make_case


FIXTURE = (
    Path(__file__).parent.parent
    / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"
)


@pytest.fixture(scope="module")
def policy():
    return parse_lcd_xml(FIXTURE)


def _one_rule_json(**overrides) -> str:
    """Build a single-rule JSON response that will pass the validators
    for the sample_lcd.xml fixture."""
    base = {
        "trigger": {
            "cpt_families": ["imaging_mri"],
            "icd10_chapters": ["musculoskeletal"],
            "payers": [],
            "semantic_predicate": "lumbar MRI exception for rapidly progressing deficit",
        },
        "action": "Approve lumbar MRI when rapidly progressing neurological deficit documented.",
        "rationale": "Policy flags progressive deficit as exception overriding conservative wait.",
        "polarity": "supports_approval",
        "source_policy_citation": "CMS LCD L34522 §C.1",
    }
    base.update(overrides)
    return json.dumps({"rules": [base]})


# ── Happy path ─────────────────────────────────────────────────────────────


def test_analyze_failure_valid_candidate_becomes_rule(policy):
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    llm = FakeLLMClient(responses=[_one_rule_json()])
    result = analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert isinstance(result, ProposalResult)
    assert result.n_candidates == 1
    assert result.n_rejected == 0
    rule = result.candidates[0]
    assert rule.status == RuleStatus.CANDIDATE
    assert rule.trigger.cpt_families == [CPTFamily.IMAGING_MRI]
    assert rule.trigger.icd10_chapters == [ICD10Chapter.MUSCULOSKELETAL]
    assert rule.polarity == "supports_approval"
    assert rule.evidence.source_policy_citation == "CMS LCD L34522 §C.1"
    assert rule.evidence.supporting_case_ids == [case.case_id]
    assert rule.evidence.proposer_model == "fake-llm-v0"
    assert rule.evidence.proposer_run_id.startswith("failure_")


def test_analyze_failure_empty_rules_list_allowed(policy):
    """When the LLM honestly reports 'no generalizable rule applies',
    the result is empty, not an error."""
    case = make_case(ground_truth_outcome=Outcome.DENIED)
    llm = FakeLLMClient(responses=[json.dumps({"rules": []})])
    result = analyze_failure(
        case,
        agent_determination=Outcome.APPROVED,
        ground_truth_outcome=Outcome.DENIED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert result.n_candidates == 0
    assert result.n_rejected == 0


# ── Prompt isolation: user message must not contain agent reasoning ────────


def test_analyze_failure_user_prompt_omits_agent_reasoning(policy):
    """Plan risk #14: the failure analyzer must not see the agent's
    reasoning chain. Only case inputs + structural labels are allowed."""
    case = make_case(
        clinical_notes="Patient presents with thunderclap headache.",
        ground_truth_outcome=Outcome.APPROVED,
    )
    llm = FakeLLMClient(responses=[_one_rule_json()])
    analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert len(llm.calls) == 1
    _, user_msg = llm.calls[0]
    # Both labels must be present so the analyzer knows which way to flip
    assert "denied" in user_msg
    assert "approved" in user_msg
    # Clinical notes are allowed (the analyzer needs them)
    assert "thunderclap headache" in user_msg
    # But NO agent-reasoning tokens — the following substrings would
    # indicate an accidental reasoning leak
    for leaked in ("reasoning_chain", "agent_reasoning", "chain_of_thought"):
        assert leaked not in user_msg.lower()


# ── Validation: citations ───────────────────────────────────────────────────


def test_analyze_failure_rejects_hallucinated_citation(policy):
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    llm = FakeLLMClient(responses=[
        _one_rule_json(source_policy_citation="CMS LCD L34522 §Z.99")
    ])
    result = analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "PolicyCitationError" in reason


def test_analyze_failure_rejects_missing_citation(policy):
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    llm = FakeLLMClient(responses=[
        _one_rule_json(source_policy_citation="")
    ])
    result = analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert result.n_candidates == 0
    assert result.n_rejected == 1


# ── Validation: patient identifiers ─────────────────────────────────────────


def test_analyze_failure_rejects_rule_with_patient_age(policy):
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    llm = FakeLLMClient(responses=[
        _one_rule_json(action="Approve if 58 y/o with progressive deficit.")
    ])
    result = analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "PatientIdentifierError" in reason


# ── Error handling ──────────────────────────────────────────────────────────


def test_analyze_failure_parse_error_on_malformed_json(policy):
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    llm = FakeLLMClient(responses=["not json, just prose"])
    result = analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=policy,
        supporting_case_ids=[case.case_id],
    )
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "parse_error" in reason


def test_analyze_failure_raises_on_correct_case(policy):
    """Calling analyze_failure with agent==ground_truth is a caller bug."""
    case = make_case(ground_truth_outcome=Outcome.APPROVED)
    llm = FakeLLMClient(responses=[_one_rule_json()])
    with pytest.raises(ValueError, match="correct case"):
        analyze_failure(
            case,
            agent_determination=Outcome.APPROVED,
            ground_truth_outcome=Outcome.APPROVED,
            llm_client=llm,
            policy=policy,
            supporting_case_ids=[case.case_id],
        )
    assert len(llm.calls) == 0  # no LLM call should have been made


# ── Multi-policy (wildcard) mode ────────────────────────────────────────────


def test_analyze_failure_multi_policy_wildcard_accepts_any_citation():
    """When policy=None, any non-empty citation passes validation.
    Same behavior as propose_rules multi-policy mode."""
    case = make_case(
        policy_excerpt="Aetna CPB-0236: progressive deficit is an exception.",
        ground_truth_outcome=Outcome.APPROVED,
    )
    llm = FakeLLMClient(responses=[
        _one_rule_json(source_policy_citation="Aetna CPB-0236 ¶3")
    ])
    result = analyze_failure(
        case,
        agent_determination=Outcome.DENIED,
        ground_truth_outcome=Outcome.APPROVED,
        llm_client=llm,
        policy=None,
        supporting_case_ids=[case.case_id],
    )
    assert result.n_candidates == 1
    assert result.n_rejected == 0
