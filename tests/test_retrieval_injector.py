"""Tests for medreason.retrieval.injector — Phase 5 Commit 1."""

from __future__ import annotations

import json

import pytest

from medreason.ontology import (
    AppliedRule,
    CPTFamily,
    ICD10Chapter,
    Payer,
    ReasoningRule,
    RuleEvidence,
    RuleTrigger,
)
from medreason.retrieval import (
    APPLIED_RULES_FIELD,
    build_rule_checklist,
    missing_rule_ids,
    parse_applied_rules,
)


def _rule(
    *,
    rule_id: str = "rule_abc123",
    cpt_families=(CPTFamily.IMAGING_MRI,),
    icd_chapters=(ICD10Chapter.MUSCULOSKELETAL,),
    payers=(Payer.AETNA,),
    action: str = "Require documented six weeks conservative therapy.",
    rationale: str = "Policy requires conservative trial.",
    polarity: str = "requires_check",
    citation: str = "CMS LCD L34522 §C.1",
    semantic: str = "lumbar MRI for low back pain",
    success_count: int = 12,
    failure_count: int = 3,
    seen_count: int = 15,
) -> ReasoningRule:
    return ReasoningRule(
        rule_id=rule_id,
        trigger=RuleTrigger(
            cpt_families=list(cpt_families),
            icd10_chapters=list(icd_chapters),
            payers=list(payers),
            semantic_predicate=semantic,
        ),
        action=action,
        rationale=rationale,
        polarity=polarity,  # type: ignore[arg-type]
        evidence=RuleEvidence(
            supporting_case_ids=["train_001"],
            source_policy_citation=citation,
            proposer_model="test",
            proposer_run_id="run_1",
        ),
        success_count=success_count,
        failure_count=failure_count,
        seen_count=seen_count,
    )


# ── build_rule_checklist ────────────────────────────────────────────────────


def test_build_rule_checklist_empty_returns_empty_string():
    assert build_rule_checklist([]) == ""


def test_build_rule_checklist_contains_header_and_footer():
    out = build_rule_checklist([_rule()])
    assert "INSTITUTIONAL REASONING MEMORY" in out
    assert "END INSTITUTIONAL MEMORY" in out
    assert "OUTPUT CONTRACT" in out


def test_build_rule_checklist_includes_rule_id_and_fields():
    rule = _rule(rule_id="rule_xyz789", action="Require 6 weeks PT.")
    out = build_rule_checklist([rule])
    assert "[rule_xyz789]" in out
    assert "Require 6 weeks PT." in out
    # Trigger summary
    assert "payer=Aetna" in out
    assert "cpt_family=imaging_mri" in out
    assert "icd_chapter=musculoskeletal" in out
    # Polarity + citation
    assert "requires_check" in out
    assert "CMS LCD L34522 §C.1" in out


def test_build_rule_checklist_posterior_and_trials_rendered():
    rule = _rule(success_count=80, failure_count=20)  # posterior 81/102
    out = build_rule_checklist([rule])
    # Posterior ~0.79, trials = 100
    assert "posterior=0.79" in out
    assert "n=100" in out


def test_build_rule_checklist_multiple_rules_numbered():
    rules = [_rule(rule_id="rule_a"), _rule(rule_id="rule_b"), _rule(rule_id="rule_c")]
    out = build_rule_checklist(rules)
    assert "[rule_a]" in out
    assert "[rule_b]" in out
    assert "[rule_c]" in out
    # Order preserved
    a_idx = out.index("[rule_a]")
    b_idx = out.index("[rule_b]")
    c_idx = out.index("[rule_c]")
    assert a_idx < b_idx < c_idx


def test_build_rule_checklist_contract_mentions_applied_rules_field():
    out = build_rule_checklist([_rule()])
    assert APPLIED_RULES_FIELD in out
    assert '"rule_id"' in out
    assert '"applied"' in out
    assert '"rationale"' in out


def test_build_rule_checklist_wildcard_trigger():
    rule = _rule(
        rule_id="rule_wild",
        cpt_families=(),
        icd_chapters=(),
        payers=(),
    )
    out = build_rule_checklist([rule])
    # Empty trigger should display as "any"
    assert "TRIGGER:  any" in out or "TRIGGER: any" in out


# ── parse_applied_rules ─────────────────────────────────────────────────────


def test_parse_applied_rules_happy_path():
    response = json.dumps({
        "determination": "approved",
        "confidence": 0.9,
        "reasoning_chain": "...",
        "key_factors": [],
        APPLIED_RULES_FIELD: [
            {"rule_id": "rule_a", "applied": True, "rationale": "hit"},
            {"rule_id": "rule_b", "applied": False, "rationale": "not relevant"},
        ],
    })
    parsed = parse_applied_rules(response)
    assert len(parsed) == 2
    assert parsed[0] == AppliedRule(rule_id="rule_a", applied=True, rationale="hit")
    assert parsed[1].applied is False


def test_parse_applied_rules_missing_field_returns_empty():
    response = json.dumps({"determination": "approved"})
    assert parse_applied_rules(response) == []


def test_parse_applied_rules_applied_rules_not_list_returns_empty():
    response = json.dumps({APPLIED_RULES_FIELD: "oops"})
    assert parse_applied_rules(response) == []


def test_parse_applied_rules_malformed_entry_dropped_silently():
    response = json.dumps({
        APPLIED_RULES_FIELD: [
            {"rule_id": "rule_a", "applied": True, "rationale": "ok"},
            "not an object",
            {"rule_id": "", "applied": True},        # empty id
            {"applied": True},                        # no id
            {"rule_id": "rule_b", "applied": "nope"}, # bad applied string
            {"rule_id": "rule_c", "applied": False},  # valid without rationale
        ],
    })
    parsed = parse_applied_rules(response)
    # Only rule_a and rule_c survive
    assert len(parsed) == 2
    assert parsed[0].rule_id == "rule_a"
    assert parsed[1].rule_id == "rule_c"


def test_parse_applied_rules_string_booleans_tolerated():
    """Some models emit 'true'/'false' strings instead of booleans."""
    response = json.dumps({
        APPLIED_RULES_FIELD: [
            {"rule_id": "rule_a", "applied": "true", "rationale": "ok"},
            {"rule_id": "rule_b", "applied": "FALSE"},
        ],
    })
    parsed = parse_applied_rules(response)
    assert len(parsed) == 2
    assert parsed[0].applied is True
    assert parsed[1].applied is False


def test_parse_applied_rules_empty_input_returns_empty():
    assert parse_applied_rules("") == []


def test_parse_applied_rules_unparseable_json_returns_empty():
    assert parse_applied_rules("not json at all") == []


def test_parse_applied_rules_handles_fenced_response():
    response = (
        "```json\n"
        + json.dumps({APPLIED_RULES_FIELD: [
            {"rule_id": "rule_a", "applied": True, "rationale": "hit"}
        ]})
        + "\n```"
    )
    parsed = parse_applied_rules(response)
    assert len(parsed) == 1
    assert parsed[0].rule_id == "rule_a"


# ── missing_rule_ids ────────────────────────────────────────────────────────


def test_missing_rule_ids_detects_gaps():
    retrieved = ["rule_a", "rule_b", "rule_c"]
    applied = [
        AppliedRule(rule_id="rule_a", applied=True),
        AppliedRule(rule_id="rule_c", applied=False),
    ]
    assert missing_rule_ids(retrieved, applied) == ["rule_b"]


def test_missing_rule_ids_none_missing():
    retrieved = ["rule_a", "rule_b"]
    applied = [
        AppliedRule(rule_id="rule_a", applied=True),
        AppliedRule(rule_id="rule_b", applied=False),
    ]
    assert missing_rule_ids(retrieved, applied) == []


def test_missing_rule_ids_all_missing():
    retrieved = ["rule_a", "rule_b"]
    applied = []
    assert missing_rule_ids(retrieved, applied) == ["rule_a", "rule_b"]


def test_missing_rule_ids_preserves_order():
    """Deterministic error messages depend on preserving the original
    retrieval order."""
    retrieved = ["rule_c", "rule_a", "rule_b"]
    applied = [AppliedRule(rule_id="rule_a", applied=True)]
    assert missing_rule_ids(retrieved, applied) == ["rule_c", "rule_b"]


# ── Integration: round-trip through AppliedRule serialization ──────────────


def test_applied_rules_json_round_trip():
    original = [
        AppliedRule(rule_id="rule_a", applied=True, rationale="hit"),
        AppliedRule(rule_id="rule_b", applied=False, rationale="skip"),
    ]
    payload = {
        "determination": "approved",
        APPLIED_RULES_FIELD: [a.model_dump() for a in original],
    }
    parsed = parse_applied_rules(json.dumps(payload))
    assert len(parsed) == len(original)
    for p, o in zip(parsed, original):
        assert p.rule_id == o.rule_id
        assert p.applied == o.applied
        assert p.rationale == o.rationale
