"""Tests for medreason_bench.data.case_builder — Phase 3."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from medreason.ontology import (
    BenchmarkCase, DenialReason, Difficulty, Outcome, Payer,
)
from medreason_bench.data import parse_lcd_xml
from medreason_bench.data.case_builder import build_cases_from_lcd


FIXTURE = Path(__file__).parent.parent / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"


@pytest.fixture(scope="module")
def policy():
    return parse_lcd_xml(FIXTURE)


# ── Basic shape ──────────────────────────────────────────────────────────────


def test_builds_exact_target_count(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    assert len(cases) == 50
    for c in cases:
        assert isinstance(c, BenchmarkCase)


def test_target_count_zero_returns_empty(policy):
    assert build_cases_from_lcd(policy, target_count=0, seed=42) == []


def test_respects_arbitrary_target_count(policy):
    for n in (1, 5, 17, 100):
        cases = build_cases_from_lcd(policy, target_count=n, seed=42)
        assert len(cases) == n


def test_rejects_policy_with_no_cpts():
    from medreason_bench.data.schemas import LCDPolicy
    empty = LCDPolicy(document_id="L0", title="empty")
    with pytest.raises(ValueError) as exc:
        build_cases_from_lcd(empty, target_count=5, seed=42)
    assert "cpt_codes" in str(exc.value)


# ── Determinism (the manifest's SHA256 chain depends on this) ───────────────


def test_same_seed_produces_byte_identical_cases(policy):
    a = build_cases_from_lcd(policy, target_count=50, seed=42)
    b = build_cases_from_lcd(policy, target_count=50, seed=42)
    for ca, cb in zip(a, b):
        assert ca.model_dump() == cb.model_dump()


def test_different_seed_preserves_structure_changes_params(policy):
    """Changing the seed must change variable parameters (age, leg)
    but must NOT change case_id sequence, CPT assignment, or ground
    truth outcome. The structural skeleton is what the manifest's
    stratification depends on."""
    a = build_cases_from_lcd(policy, target_count=50, seed=42)
    b = build_cases_from_lcd(policy, target_count=50, seed=999)

    # Case ids and structural fields must match
    for ca, cb in zip(a, b):
        assert ca.case_id == cb.case_id
        assert ca.task_config.cpt_code == cb.task_config.cpt_code
        assert ca.ground_truth_outcome == cb.ground_truth_outcome
        assert ca.difficulty == cb.difficulty

    # At least SOME clinical notes must differ (new random params)
    differing_notes = sum(1 for ca, cb in zip(a, b)
                          if ca.clinical_notes != cb.clinical_notes)
    assert differing_notes > 0


def test_case_ids_are_zero_padded_and_sorted(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    ids = [c.case_id for c in cases]
    assert ids[0] == "case_0001"
    assert ids[-1] == "case_0050"
    assert ids == sorted(ids)


# ── Ground truth integrity ──────────────────────────────────────────────────


def test_outcome_distribution_has_all_three_classes(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    counts = Counter(c.ground_truth_outcome.value for c in cases)
    assert counts["approved"] > 0
    assert counts["denied"] > 0
    assert counts["overturned_on_appeal"] > 0


def test_difficulty_distribution_has_all_three_classes(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    counts = Counter(c.difficulty.value for c in cases)
    assert counts["easy"] > 0
    assert counts["medium"] > 0
    assert counts["hard"] > 0


def test_cpts_are_used_from_policy(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    used = {c.task_config.cpt_code for c in cases}
    assert used == set(policy.cpt_codes)


def test_denied_cases_have_denial_reason(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    denied = [c for c in cases if c.ground_truth_outcome == Outcome.DENIED]
    for c in denied:
        assert c.task_config.denial_reason is not None
        assert isinstance(c.task_config.denial_reason, DenialReason)


def test_approved_cases_have_no_denial_reason(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    approved = [c for c in cases if c.ground_truth_outcome == Outcome.APPROVED]
    for c in approved:
        assert c.task_config.denial_reason is None


def test_every_case_has_non_empty_clinical_notes_and_reasoning(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    for c in cases:
        assert len(c.clinical_notes) > 50
        assert len(c.ground_truth_reasoning) >= 2
        # Reasoning must reference at least one LCD section id
        joined = " ".join(c.ground_truth_reasoning)
        assert any(
            sid in joined
            for sid in ("C.1", "C.2", "C.3", "C.4", "L.1", "L.2", "L.3")
        )


def test_every_case_carries_full_policy_excerpt(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    for c in cases:
        assert "L34522" in c.policy_excerpt
        assert "§C.1" in c.policy_excerpt
        assert "§L.1" in c.policy_excerpt


def test_payer_defaults_to_medicare(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    for c in cases:
        assert c.task_config.payer == Payer.MEDICARE


def test_payer_override(policy):
    cases = build_cases_from_lcd(
        policy, target_count=10, seed=42, payer=Payer.AETNA
    )
    for c in cases:
        assert c.task_config.payer == Payer.AETNA


# ── Template-specific sanity checks ─────────────────────────────────────────


def test_repeat_mri_template_uses_frequency_limit(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    repeat_cases = [
        c for c in cases
        if "6 months ago" in c.clinical_notes or "repeat" in c.clinical_notes.lower()
    ]
    assert repeat_cases
    for c in repeat_cases:
        if c.ground_truth_outcome == Outcome.DENIED:
            assert c.task_config.denial_reason == DenialReason.FREQUENCY_LIMIT


def test_overturned_cases_are_hard_difficulty(policy):
    cases = build_cases_from_lcd(policy, target_count=50, seed=42)
    overturned = [c for c in cases
                  if c.ground_truth_outcome == Outcome.OVERTURNED_ON_APPEAL]
    assert overturned
    for c in overturned:
        assert c.difficulty == Difficulty.HARD
