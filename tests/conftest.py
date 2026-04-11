"""Shared pytest fixtures and helpers for medreason tests."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def sqlite_conn() -> sqlite3.Connection:
    """Fresh in-memory SQLite connection per test."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def make_rule(
    *,
    action: str = "Require documented failure of conservative therapy.",
    supporting_case_ids: list[str] | None = None,
    semantic_embedding: list[float] | None = None,
    cpt_families=None,
    icd10_chapters=None,
    payers=None,
    success_count: int = 0,
    failure_count: int = 0,
    seen_count: int = 0,
    rationale: str = "Unit test rule.",
    source_policy_citation: str = "CMS LCD TEST",
):
    """Convenience constructor for unit tests. Keeps tests short."""
    from medreason.ontology import (
        CPTFamily, ICD10Chapter, Payer,
        ReasoningRule, RuleEvidence, RuleTrigger,
    )
    trig_kwargs: dict = {}
    if cpt_families is not None:
        trig_kwargs["cpt_families"] = cpt_families
    else:
        trig_kwargs["cpt_families"] = [CPTFamily.IMAGING_MRI]
    if icd10_chapters is not None:
        trig_kwargs["icd10_chapters"] = icd10_chapters
    else:
        trig_kwargs["icd10_chapters"] = [ICD10Chapter.MUSCULOSKELETAL]
    if payers is not None:
        trig_kwargs["payers"] = payers
    if semantic_embedding is not None:
        trig_kwargs["semantic_embedding"] = semantic_embedding
        trig_kwargs["semantic_predicate"] = "test predicate"
    return ReasoningRule(
        trigger=RuleTrigger(**trig_kwargs),
        action=action,
        rationale=rationale,
        evidence=RuleEvidence(
            supporting_case_ids=supporting_case_ids or ["train_001"],
            source_policy_citation=source_policy_citation,
            proposer_model="test-proposer-v0",
            proposer_run_id="test-run",
        ),
        success_count=success_count,
        failure_count=failure_count,
        seen_count=seen_count,
    )


def make_case(
    *,
    case_id: str = "test-case-001",
    payer_val: str = "Aetna",
    cpt: str = "72148",
    icds: list[str] | None = None,
    facility: str = "outpatient",
    modifiers: list[str] | None = None,
    clinical_notes: str = (
        "58 y/o with chronic low back pain. 8 weeks of PT plus NSAIDs without "
        "improvement. Exam shows positive straight-leg raise bilaterally."
    ),
    policy_excerpt: str = "Lumbar MRI approved after 6 weeks conservative therapy.",
    ground_truth_outcome=None,
    ground_truth_reasoning: list[str] | None = None,
    prior_eobs: list[str] | None = None,
    difficulty=None,
):
    """Minimal BenchmarkCase for runner/eval tests."""
    from medreason.ontology import (
        BenchmarkCase, Difficulty, FacilityType, Outcome, Payer,
        PriorAuthTaskConfig,
    )
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=Payer(payer_val),
            cpt_code=cpt,
            icd10_codes=icds or ["M54.5"],
            modifiers=modifiers or [],
            facility_type=FacilityType(facility),
        ),
        clinical_notes=clinical_notes,
        prior_eobs=prior_eobs or [],
        policy_excerpt=policy_excerpt,
        ground_truth_outcome=ground_truth_outcome or Outcome.APPROVED,
        ground_truth_reasoning=ground_truth_reasoning or [
            "check conservative therapy duration",
            "verify clinical findings",
        ],
        difficulty=difficulty or Difficulty.MEDIUM,
    )


def make_trace(
    *,
    source: str = "agent",
    outcome=None,
):
    from medreason.ontology import (
        ArgumentFraming, ArgumentStructure,
        FacilityType, Outcome, Payer,
        PriorAuthTaskConfig, ReasoningStep, ReasoningTrace,
    )
    return ReasoningTrace(
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA,
            cpt_code="72148",
            icd10_codes=["M54.5"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        reasoning_steps=[
            ReasoningStep(
                step_number=1,
                action="Check conservative therapy duration",
                evidence_cited="Clinical notes: 8 weeks PT",
                decision_branch="met",
            ),
        ],
        argument_structure=ArgumentStructure(
            framing=ArgumentFraming.GUIDELINE_ADHERENCE,
            clinical_evidence_elements=["PT duration"],
            payer_policy_hooks=["CMS LCD L34522 §C.1"],
        ),
        raw_chain_of_thought="reasoning goes here",
        outcome=outcome or Outcome.APPROVED,
        source=source,
    )
