"""Tests for medreason.extraction.rule_proposer — Phase 5 Commit 1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medreason.extraction import (
    PatientIdentifierError,
    PolicyCitationError,
    ProposalResult,
    propose_rules,
)
from medreason.extraction.rule_proposer import (
    ACTION_MAX_WORDS,
    _citation_exists_in_policy,
    _has_patient_identifier,
    _build_rule_from_candidate,
)
from medreason.llm import FakeLLMClient
from medreason.ontology import (
    ArgumentFraming,
    ArgumentStructure,
    CPTFamily,
    ICD10Chapter,
    Outcome,
    Payer,
    ReasoningTrace,
    RuleStatus,
)
from medreason_bench.data import parse_lcd_xml


FIXTURE = (
    Path(__file__).parent.parent
    / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"
)


@pytest.fixture(scope="module")
def policy():
    return parse_lcd_xml(FIXTURE)


def _trace(outcome: Outcome = Outcome.APPROVED) -> ReasoningTrace:
    from medreason.ontology import FacilityType, PriorAuthTaskConfig
    from medreason.ontology.trace import ReasoningStep
    return ReasoningTrace(
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE,
            cpt_code="72148",
            icd10_codes=["M51.16"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        reasoning_steps=[
            ReasoningStep(
                step_number=1,
                action="check conservative therapy duration",
                evidence_cited="8 weeks PT documented",
                decision_branch="C.1 met",
            ),
        ],
        argument_structure=ArgumentStructure(
            framing=ArgumentFraming.GUIDELINE_ADHERENCE,
        ),
        outcome=outcome,
        source="critic",
    )


def _one_rule_json(**overrides) -> str:
    base = {
        "trigger": {
            "cpt_families": ["imaging_mri"],
            "icd10_chapters": ["musculoskeletal"],
            "payers": [],
            "semantic_predicate": "lumbar MRI for chronic low back pain"
        },
        "action": "Require documented six weeks of conservative therapy before approval.",
        "rationale": "Policy requires failed conservative trial.",
        "polarity": "requires_check",
        "source_policy_citation": "CMS LCD L34522 §C.1",
    }
    base.update(overrides)
    return json.dumps({"rules": [base]})


# ── Happy path: valid proposer output produces candidates ──────────────────


def test_propose_rules_valid_candidate_becomes_reasoning_rule(policy):
    llm = FakeLLMClient(responses=[_one_rule_json()])
    result = propose_rules(
        _trace(),
        policy,
        llm,
        supporting_case_ids=["train_001", "train_005"],
    )
    assert isinstance(result, ProposalResult)
    assert result.n_candidates == 1
    assert result.n_rejected == 0
    rule = result.candidates[0]
    assert rule.status == RuleStatus.CANDIDATE
    assert rule.trigger.cpt_families == [CPTFamily.IMAGING_MRI]
    assert rule.trigger.icd10_chapters == [ICD10Chapter.MUSCULOSKELETAL]
    assert rule.action.startswith("Require documented six weeks")
    assert rule.evidence.source_policy_citation == "CMS LCD L34522 §C.1"
    assert rule.evidence.supporting_case_ids == ["train_001", "train_005"]
    assert rule.evidence.proposer_model == "fake-llm-v0"
    assert rule.evidence.proposer_run_id  # non-empty


def test_propose_rules_multiple_candidates(policy):
    payload = json.dumps({
        "rules": [
            json.loads(_one_rule_json())["rules"][0],
            {
                "trigger": {
                    "cpt_families": ["imaging_mri"],
                    "icd10_chapters": [],
                    "payers": ["Medicare"],
                    "semantic_predicate": "repeat MRI",
                },
                "action": "Repeat MRI within twelve months needs new deficit.",
                "rationale": "Limitation on frequency.",
                "polarity": "supports_denial",
                "source_policy_citation": "CMS LCD L34522 §L.1",
            },
        ]
    })
    llm = FakeLLMClient(responses=[payload])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 2
    assert result.candidates[1].trigger.payers == [Payer.MEDICARE]


def test_propose_rules_empty_rule_list_is_allowed(policy):
    """An honest 'no generalizable rules' response must NOT error."""
    llm = FakeLLMClient(responses=[json.dumps({"rules": []})])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    assert result.n_rejected == 0


# ── Validation: citations ───────────────────────────────────────────────────


def test_citation_existing_section_passes(policy):
    assert _citation_exists_in_policy("CMS LCD L34522 §C.1", policy) is True
    assert _citation_exists_in_policy("CMS LCD L34522 §L.2", policy) is True


def test_citation_hallucinated_section_rejected(policy):
    assert _citation_exists_in_policy("CMS LCD L34522 §X.99", policy) is False
    assert _citation_exists_in_policy("CMS LCD L99999 §Z.1", policy) is False
    # Wrong section id format (no letter+dot+number)
    assert _citation_exists_in_policy("random text", policy) is False


def test_propose_rejects_rule_with_hallucinated_citation(policy):
    llm = FakeLLMClient(responses=[_one_rule_json(
        source_policy_citation="CMS LCD L34522 §Z.99"
    )])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "PolicyCitationError" in reason


def test_propose_rejects_rule_with_missing_citation(policy):
    llm = FakeLLMClient(responses=[_one_rule_json(source_policy_citation="")])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "missing source_policy_citation" in reason


# ── Validation: patient identifiers ─────────────────────────────────────────


def test_patient_identifier_detector_catches_age():
    assert _has_patient_identifier("58 y/o male with back pain") is not None
    assert _has_patient_identifier("72 years old") is not None


def test_patient_identifier_detector_catches_dates():
    assert _has_patient_identifier("seen on 2025-04-10") is not None


def test_patient_identifier_detector_catches_patient_name():
    assert _has_patient_identifier("patient Smith") is not None
    assert _has_patient_identifier("Dr. Johnson") is not None


def test_patient_identifier_detector_passes_clean_text():
    assert _has_patient_identifier("require 6 weeks conservative therapy") is None
    assert _has_patient_identifier("patients with chronic back pain") is None


def test_propose_rejects_rule_with_patient_identifier_in_action(policy):
    llm = FakeLLMClient(responses=[_one_rule_json(
        action="Approve 58 y/o patients with documented PT.",
    )])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "PatientIdentifierError" in reason


# ── Validation: action length ───────────────────────────────────────────────


def test_propose_rejects_rule_with_overlong_action(policy):
    long_action = " ".join(["word"] * (ACTION_MAX_WORDS + 5))
    llm = FakeLLMClient(responses=[_one_rule_json(action=long_action)])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    _, reason = result.rejected[0]
    assert "word" in reason.lower() or "limit" in reason.lower()


def test_propose_accepts_rule_at_exactly_max_words(policy):
    exact_action = " ".join(["w"] * ACTION_MAX_WORDS)
    llm = FakeLLMClient(responses=[_one_rule_json(action=exact_action)])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 1


# ── Mixed: some candidates pass, others rejected ───────────────────────────


def test_propose_mixes_valid_and_invalid_candidates(policy):
    payload = json.dumps({
        "rules": [
            json.loads(_one_rule_json())["rules"][0],  # valid
            json.loads(_one_rule_json(
                source_policy_citation="CMS LCD L34522 §Z.99"
            ))["rules"][0],  # bad citation
            json.loads(_one_rule_json(
                action="72 y/o patient needs MRI"
            ))["rules"][0],  # patient identifier
        ]
    })
    llm = FakeLLMClient(responses=[payload])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 1
    assert result.n_rejected == 2


# ── Input isolation: proposer never sees agent reasoning ───────────────────


def test_proposer_input_never_includes_agent_reasoning(policy):
    """Plan risk #14 regression guard: a future refactor might pipe
    agent_result.reasoning_chain into the proposer. The function
    signature doesn't accept one, but verify by inspecting what was
    sent to the LLM."""
    llm = FakeLLMClient(responses=[_one_rule_json()])
    trace = _trace()
    propose_rules(trace, policy, llm, supporting_case_ids=["t1"])
    system, user = llm.calls[0]
    # The user message should contain the critic-derived trace, the
    # task config, and the policy excerpt — but nothing from an
    # AgentResult.
    assert "reasoning_chain" not in user
    assert "key_factors" not in user
    assert "confidence" not in user.lower() or "confidence" in user[:2000]
    # The trace source must be the critic, not an agent
    assert "Source: critic" in user


# ── Parse / error paths ────────────────────────────────────────────────────


def test_propose_parse_error_returns_rejection(policy):
    llm = FakeLLMClient(responses=["totally not json"])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "parse_error" in reason


def test_propose_top_level_rules_not_a_list(policy):
    llm = FakeLLMClient(responses=[json.dumps({"rules": "not a list"})])
    result = propose_rules(_trace(), policy, llm, supporting_case_ids=["t1"])
    assert result.n_candidates == 0
    assert result.n_rejected == 1


def test_propose_llm_exception_becomes_rejection(policy):
    class BoomClient:
        model_version = "boom"
        def complete(self, **kwargs):
            raise RuntimeError("network down")
    result = propose_rules(
        _trace(), policy, BoomClient(), supporting_case_ids=["t1"]
    )
    assert result.n_candidates == 0
    assert result.n_rejected == 1
    _, reason = result.rejected[0]
    assert "llm_error" in reason
