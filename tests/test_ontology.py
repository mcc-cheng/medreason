"""Tests for medreason.ontology — Phase 0.

Covers:
- CPTFamily / ICD10Chapter lookups including the awkward boundaries
  (D49/D50, H59/H60, S/T, multi-letter externals).
- RuleTrigger.matches() structural prefilter — wildcard, payer filter,
  CPT family vs explicit code, ICD prefix vs chapter, modifier subset.
- ReasoningRule posterior math (Beta(α=s+1, β=f+1)).
- ReasoningRule JSON round-trip.
- AppliedRule schema.
- Backward compat: every name the pre-rework modules import is reachable
  from `medreason.ontology`.
"""

from __future__ import annotations

import json
import math

import pytest


# ── Code lookups ──────────────────────────────────────────────────────────────


def test_cpt_family_known_codes():
    from medreason.ontology import CPTFamily, cpt_family
    assert cpt_family("99213") is CPTFamily.EM_OUTPATIENT
    assert cpt_family("99221") is CPTFamily.EM_INPATIENT
    assert cpt_family("72148") is CPTFamily.IMAGING_MRI
    assert cpt_family("70450") is CPTFamily.IMAGING_CT
    assert cpt_family("27447") is CPTFamily.SURGERY_ORTHO  # TKA
    assert cpt_family("29881") is CPTFamily.SURGERY_ORTHO  # arthroscopy
    assert cpt_family("20610") is CPTFamily.PAIN_INJECTION  # joint injection
    assert cpt_family("64483") is CPTFamily.PAIN_INJECTION  # transforaminal epidural
    assert cpt_family("43239") is CPTFamily.GI_ENDOSCOPY    # EGD with biopsy
    assert cpt_family("90837") is CPTFamily.PSYCHOTHERAPY
    assert cpt_family("97110") is CPTFamily.PT_OT
    assert cpt_family("77386") is CPTFamily.ONCOLOGY_RADIATION


def test_cpt_family_handles_garbage():
    from medreason.ontology import CPTFamily, cpt_family
    assert cpt_family("") is CPTFamily.OTHER
    assert cpt_family("abc") is CPTFamily.OTHER
    assert cpt_family("123") is CPTFamily.OTHER          # too short
    assert cpt_family("99999") is CPTFamily.OTHER        # not in any range
    assert cpt_family("0001F") is CPTFamily.OTHER        # category II


def test_cpt_family_pain_injection_takes_precedence():
    """20610 sits inside the orthopedic numeric block but must resolve as
    PAIN_INJECTION, not SURGERY_ORTHO. Order of _CPT_RANGES guarantees this.
    """
    from medreason.ontology import CPTFamily, cpt_family
    assert cpt_family("20610") is CPTFamily.PAIN_INJECTION
    assert cpt_family("20611") is CPTFamily.PAIN_INJECTION


def test_cpt_families_dedupes_preserving_order():
    from medreason.ontology import CPTFamily, cpt_families
    out = cpt_families(["72148", "72149", "27447"])
    assert out == [CPTFamily.IMAGING_MRI, CPTFamily.SURGERY_ORTHO]


def test_icd10_chapter_letter_to_chapter():
    from medreason.ontology import ICD10Chapter, icd10_chapter
    assert icd10_chapter("A09") is ICD10Chapter.INFECTIOUS
    assert icd10_chapter("E11.9") is ICD10Chapter.ENDOCRINE
    assert icd10_chapter("F32.1") is ICD10Chapter.MENTAL
    assert icd10_chapter("I21.4") is ICD10Chapter.CIRCULATORY
    assert icd10_chapter("M54.5") is ICD10Chapter.MUSCULOSKELETAL
    assert icd10_chapter("M17.11") is ICD10Chapter.MUSCULOSKELETAL
    assert icd10_chapter("Z00.00") is ICD10Chapter.HEALTH_STATUS


def test_icd10_chapter_d_split_at_49_50():
    """D00-D49 are neoplasms, D50-D89 are blood. The split is at the
    numeric boundary, not the letter."""
    from medreason.ontology import ICD10Chapter, icd10_chapter
    assert icd10_chapter("D00") is ICD10Chapter.NEOPLASMS
    assert icd10_chapter("D49") is ICD10Chapter.NEOPLASMS
    assert icd10_chapter("D50") is ICD10Chapter.BLOOD
    assert icd10_chapter("D89") is ICD10Chapter.BLOOD


def test_icd10_chapter_h_split_at_59_60():
    """H00-H59 are eye, H60-H95 are ear."""
    from medreason.ontology import ICD10Chapter, icd10_chapter
    assert icd10_chapter("H00") is ICD10Chapter.EYE
    assert icd10_chapter("H59") is ICD10Chapter.EYE
    assert icd10_chapter("H60") is ICD10Chapter.EAR
    assert icd10_chapter("H95") is ICD10Chapter.EAR


def test_icd10_chapter_injury_spans_s_and_t():
    from medreason.ontology import ICD10Chapter, icd10_chapter
    assert icd10_chapter("S72.001A") is ICD10Chapter.INJURY
    assert icd10_chapter("T81.4XXA") is ICD10Chapter.INJURY


def test_icd10_chapter_external_v_w_x_y():
    from medreason.ontology import ICD10Chapter, icd10_chapter
    assert icd10_chapter("V03.10") is ICD10Chapter.EXTERNAL
    assert icd10_chapter("W19.XXXA") is ICD10Chapter.EXTERNAL
    assert icd10_chapter("Y92.9") is ICD10Chapter.EXTERNAL


def test_icd10_chapter_unknown_garbage():
    from medreason.ontology import ICD10Chapter, icd10_chapter
    assert icd10_chapter("") is ICD10Chapter.UNKNOWN
    assert icd10_chapter("??") is ICD10Chapter.UNKNOWN
    assert icd10_chapter("U07.1") is ICD10Chapter.UNKNOWN  # COVID — not mapped


# ── RuleTrigger.matches() ─────────────────────────────────────────────────────


def _make_cfg(payer_val="Aetna", cpt="72148", icds=None,
              facility="outpatient", mods=None):
    from medreason.ontology import (
        FacilityType, Payer, PriorAuthTaskConfig,
    )
    return PriorAuthTaskConfig(
        payer=Payer(payer_val),
        cpt_code=cpt,
        icd10_codes=icds or ["M54.5"],
        modifiers=mods or [],
        facility_type=FacilityType(facility),
    )


def test_rule_trigger_empty_is_wildcard():
    from medreason.ontology import RuleTrigger
    t = RuleTrigger()
    assert t.matches(_make_cfg()) is True


def test_rule_trigger_payer_filter():
    from medreason.ontology import Payer, RuleTrigger
    t = RuleTrigger(payers=[Payer.AETNA])
    assert t.matches(_make_cfg(payer_val="Aetna")) is True
    assert t.matches(_make_cfg(payer_val="Cigna")) is False


def test_rule_trigger_cpt_family_match():
    from medreason.ontology import CPTFamily, RuleTrigger
    t = RuleTrigger(cpt_families=[CPTFamily.IMAGING_MRI])
    assert t.matches(_make_cfg(cpt="72148")) is True   # spine MRI
    assert t.matches(_make_cfg(cpt="72149")) is True   # also lumbar MRI
    assert t.matches(_make_cfg(cpt="27447")) is False  # TKA, not MRI


def test_rule_trigger_explicit_cpt_code():
    from medreason.ontology import RuleTrigger
    t = RuleTrigger(cpt_codes=["72148"])
    assert t.matches(_make_cfg(cpt="72148")) is True
    assert t.matches(_make_cfg(cpt="72149")) is False


def test_rule_trigger_cpt_explicit_or_family():
    """If both cpt_codes and cpt_families are set, either match counts."""
    from medreason.ontology import CPTFamily, RuleTrigger
    t = RuleTrigger(cpt_codes=["99213"], cpt_families=[CPTFamily.IMAGING_MRI])
    assert t.matches(_make_cfg(cpt="99213")) is True   # explicit hit
    assert t.matches(_make_cfg(cpt="72148")) is True   # family hit
    assert t.matches(_make_cfg(cpt="27447")) is False  # neither


def test_rule_trigger_icd_chapter_match():
    from medreason.ontology import ICD10Chapter, RuleTrigger
    t = RuleTrigger(icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL])
    assert t.matches(_make_cfg(icds=["M54.5", "M17.11"])) is True
    assert t.matches(_make_cfg(icds=["E11.9"])) is False


def test_rule_trigger_icd_prefix_match():
    from medreason.ontology import RuleTrigger
    t = RuleTrigger(icd10_prefixes=["M17"])
    assert t.matches(_make_cfg(icds=["M17.11"])) is True
    assert t.matches(_make_cfg(icds=["M54.5"])) is False


def test_rule_trigger_facility_filter():
    from medreason.ontology import FacilityType, RuleTrigger
    t = RuleTrigger(facility_types=[FacilityType.INPATIENT])
    assert t.matches(_make_cfg(facility="inpatient")) is True
    assert t.matches(_make_cfg(facility="outpatient")) is False


def test_rule_trigger_required_modifiers_subset():
    from medreason.ontology import RuleTrigger
    t = RuleTrigger(required_modifiers=["59", "76"])
    assert t.matches(_make_cfg(mods=["59", "76", "RT"])) is True   # superset OK
    assert t.matches(_make_cfg(mods=["59"])) is False              # missing 76
    assert t.matches(_make_cfg(mods=[])) is False                  # missing both


def test_rule_trigger_combined_filters_all_must_pass():
    from medreason.ontology import (
        CPTFamily, ICD10Chapter, Payer, RuleTrigger,
    )
    t = RuleTrigger(
        payers=[Payer.AETNA],
        cpt_families=[CPTFamily.IMAGING_MRI],
        icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
    )
    assert t.matches(_make_cfg("Aetna", "72148", ["M54.5"])) is True
    assert t.matches(_make_cfg("Cigna", "72148", ["M54.5"])) is False  # wrong payer
    assert t.matches(_make_cfg("Aetna", "27447", ["M54.5"])) is False  # wrong CPT
    assert t.matches(_make_cfg("Aetna", "72148", ["E11.9"])) is False  # wrong ICD


# ── ReasoningRule posterior math ──────────────────────────────────────────────


def _bare_rule(s=0, f=0):
    from medreason.ontology import (
        ReasoningRule, RuleEvidence, RuleTrigger,
    )
    return ReasoningRule(
        trigger=RuleTrigger(),
        action="Require documented failure of conservative therapy.",
        evidence=RuleEvidence(),
        success_count=s,
        failure_count=f,
    )


def test_posterior_uniform_prior_at_zero_trials():
    r = _bare_rule(0, 0)
    # Beta(1, 1) → mean 0.5, var 1/12
    assert r.posterior_mean == pytest.approx(0.5)
    assert r.posterior_variance == pytest.approx(1 / 12)
    assert r.trials == 0


def test_posterior_after_successes():
    r = _bare_rule(8, 2)
    # Beta(9, 3) → mean 9/12 = 0.75
    assert r.posterior_mean == pytest.approx(0.75)
    assert r.trials == 10


def test_posterior_variance_decreases_with_more_trials():
    """For Beta(α=s+1, β=f+1), variance must shrink as we accumulate trials.
    The means will not be exactly equal because the +1 prior has different
    pull at different scales — that's expected, not a bug."""
    r_low = _bare_rule(2, 1)
    r_high = _bare_rule(20, 10)
    assert r_high.trials > r_low.trials
    assert r_high.posterior_variance < r_low.posterior_variance


def test_posterior_mean_clamped_at_extremes():
    r = _bare_rule(100, 0)
    assert r.posterior_mean == pytest.approx(101 / 102)
    assert 0 < r.posterior_mean < 1  # never exactly 1 thanks to the +1 prior


# ── ReasoningRule JSON round-trip ─────────────────────────────────────────────


def test_reasoning_rule_json_round_trip():
    from medreason.ontology import (
        CPTFamily, ICD10Chapter, Payer, ReasoningRule,
        RuleEvidence, RuleStatus, RuleTrigger,
    )
    rule = ReasoningRule(
        status=RuleStatus.ACTIVE,
        trigger=RuleTrigger(
            cpt_families=[CPTFamily.IMAGING_MRI],
            icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
            icd10_prefixes=["M54"],
            payers=[Payer.AETNA],
            semantic_predicate="Lumbar MRI for nonspecific back pain",
        ),
        action="Require ≥6 weeks documented conservative therapy before approval.",
        rationale="CMS LCD requires conservative trial before advanced imaging.",
        polarity="requires_check",
        evidence=RuleEvidence(
            supporting_case_ids=["train_001", "train_018"],
            source_policy_citation="CMS LCD L34522 §C.1",
            source_policy_url="https://example.cms.gov/lcd/L34522",
            extracted_from_trace_ids=["trace_abc"],
            proposer_model="claude-opus-4-1-20250805",
            proposer_run_id="run_xyz",
        ),
        success_count=12,
        failure_count=3,
    )
    blob = rule.model_dump_json()
    data = json.loads(blob)
    # Computed fields should appear in the dump
    assert "posterior_mean" in data
    assert math.isclose(data["posterior_mean"], 13 / 17, rel_tol=1e-9)

    # Round-trip
    revived = ReasoningRule.model_validate_json(blob)
    assert revived.rule_id == rule.rule_id
    assert revived.status is RuleStatus.ACTIVE
    assert revived.trigger.cpt_families == [CPTFamily.IMAGING_MRI]
    assert revived.trigger.payers == [Payer.AETNA]
    assert revived.trigger.icd10_chapters == [ICD10Chapter.MUSCULOSKELETAL]
    assert revived.evidence.supporting_case_ids == ["train_001", "train_018"]
    assert revived.success_count == 12
    assert revived.posterior_mean == pytest.approx(13 / 17)


def test_rule_trigger_round_trips_through_matches():
    """A trigger should still gate correctly after JSON round-trip."""
    from medreason.ontology import CPTFamily, RuleTrigger
    t = RuleTrigger(
        cpt_families=[CPTFamily.IMAGING_MRI],
        icd10_prefixes=["M17"],
    )
    revived = RuleTrigger.model_validate_json(t.model_dump_json())
    assert revived.matches(_make_cfg(cpt="72148", icds=["M17.11"])) is True
    assert revived.matches(_make_cfg(cpt="72148", icds=["M54.5"])) is False


# ── AppliedRule + AgentResult ─────────────────────────────────────────────────


def test_applied_rule_minimal():
    from medreason.ontology import AppliedRule
    a = AppliedRule(rule_id="rule_abc", applied=True, rationale="evidence cited")
    assert a.rule_id == "rule_abc"
    assert a.applied is True


def test_agent_result_legacy_fields_still_work():
    """The pre-rework agent.py constructs AgentResult with the old fields.
    Make sure the schema still validates them."""
    from medreason.ontology import AgentResult, Outcome
    r = AgentResult(
        case_id="local-001",
        mode="zero_shot",
        determination=Outcome.APPROVED,
        reasoning_chain="step 1; step 2",
        confidence=0.9,
        key_factors=["criterion A", "criterion B"],
        correct=True,
        input_tokens=412,
        output_tokens=88,
        latency_ms=1230.0,
    )
    assert r.mode == "zero_shot"
    assert r.applied_rules == []  # new field, default empty
    assert r.cost_usd == 0.0


def test_agent_result_new_fields():
    from medreason.ontology import AgentResult, AppliedRule, Outcome
    r = AgentResult(
        case_id="train_042",
        determination=Outcome.DENIED,
        runner_id="claude-sonnet-4-20250514:memory",
        seed=11,
        retrieved_rule_ids=["rule_a", "rule_b"],
        applied_rules=[
            AppliedRule(rule_id="rule_a", applied=True, rationale="hit"),
            AppliedRule(rule_id="rule_b", applied=False, rationale="not relevant"),
        ],
        cost_usd=0.0042,
    )
    assert r.runner_id.startswith("claude")
    assert len(r.applied_rules) == 2
    assert r.applied_rules[0].applied is True
    assert r.applied_rules[1].applied is False


# ── Backward-compat: every legacy import name still resolves ─────────────────


def test_legacy_imports_still_work():
    """Pre-rework modules import these names from medreason.ontology.
    Removing or renaming any of them is a Phase 0 regression."""
    from medreason.ontology import (
        AgentResult,
        ArgumentFraming,
        ArgumentStructure,
        BenchmarkCase,
        BenchmarkResult,
        DenialReason,
        Difficulty,
        FacilityType,
        Outcome,
        Payer,
        PriorAuthTaskConfig,
        ReasoningPattern,
        ReasoningStep,
        ReasoningTrace,
    )
    # Smoke-instantiate one each
    cfg = PriorAuthTaskConfig(
        payer=Payer.AETNA,
        cpt_code="72148",
        icd10_codes=["M54.5"],
        facility_type=FacilityType.OUTPATIENT,
    )
    pattern = ReasoningPattern(
        config_hash=cfg.config_hash,
        task_config=cfg,
        argument_structure=ArgumentStructure(framing=ArgumentFraming.GUIDELINE_ADHERENCE),
        key_reasoning_steps=["check criterion A"],
        success_count=3,
        failure_count=1,
    )
    assert pattern.success_rate == pytest.approx(0.75)


def test_legacy_modules_still_import():
    """Every pre-rework medreason.* module must still import after the
    ontology package conversion. If this fails, the backward-compat shim
    has gaps."""
    import importlib
    for mod in [
        "medreason.agent",
        "medreason.store",
        "medreason.injector",
        "medreason.extractor",
        "medreason.generator",
        "medreason.benchmark",
        "medreason.local_cases",
    ]:
        importlib.import_module(mod)
