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
