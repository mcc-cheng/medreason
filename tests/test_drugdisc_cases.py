"""Tests for the v0.1 drug discovery benchmark fixture."""

from __future__ import annotations

from collections import Counter

from medreason.ontology import Outcome, Payer, RuleStatus
from medreason_bench.data.drugdisc_cases import build_drugdisc_cases
from medreason_bench.data.drugdisc_rules import build_drugdisc_seed_rules


def test_fixture_has_25_cases():
    cases = build_drugdisc_cases()
    assert len(cases) == 25


def test_case_ids_are_unique():
    cases = build_drugdisc_cases()
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == len(ids)


def test_outcome_distribution():
    """13 synthesize (approved) + 12 reject (denied)."""
    cases = build_drugdisc_cases()
    counts = Counter(c.ground_truth_outcome.value for c in cases)
    assert counts["approved"] == 13
    assert counts["denied"] == 12


def test_difficulty_distribution():
    """7 easy + 10 medium + 8 hard = 25."""
    cases = build_drugdisc_cases()
    counts = Counter(c.difficulty.value for c in cases)
    assert counts["easy"] == 7
    assert counts["medium"] == 10
    assert counts["hard"] == 8


def test_all_cases_have_required_fields():
    cases = build_drugdisc_cases()
    for c in cases:
        assert c.clinical_notes, f"{c.case_id} missing clinical_notes"
        assert c.policy_excerpt, f"{c.case_id} missing policy_excerpt"
        assert c.ground_truth_reasoning, f"{c.case_id} missing reasoning"
        assert c.task_config.cpt_code, f"{c.case_id} missing compound id"


def test_seed_rules_are_valid():
    rules = build_drugdisc_seed_rules()
    assert len(rules) == 8
    for r in rules:
        assert r.status == RuleStatus.ACTIVE
        assert r.action, f"{r.rule_id} missing action"
        assert r.trigger.semantic_predicate, f"{r.rule_id} missing predicate"
        assert r.evidence.source_policy_citation, f"{r.rule_id} missing citation"


def test_all_compound_ids_start_with_vdc():
    cases = build_drugdisc_cases()
    for c in cases:
        assert c.task_config.cpt_code.startswith("VDC-"), (
            f"{c.case_id} compound id doesn't start with VDC-"
        )
