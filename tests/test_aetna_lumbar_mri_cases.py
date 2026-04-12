"""Tests for the v0.3 Aetna lumbar MRI within-domain fixture."""

from __future__ import annotations

from collections import Counter

from medreason.ontology import Outcome, Payer
from medreason_bench.data.aetna_lumbar_mri_cases import build_aetna_lumbar_mri_cases


def test_fixture_has_60_cases():
    cases = build_aetna_lumbar_mri_cases()
    assert len(cases) == 60


def test_all_cases_are_aetna():
    """Within-domain fixture: all cases must be Aetna."""
    cases = build_aetna_lumbar_mri_cases()
    for c in cases:
        assert c.task_config.payer == Payer.AETNA, (
            f"{c.case_id} payer is {c.task_config.payer}, expected Aetna"
        )


def test_case_ids_are_unique():
    cases = build_aetna_lumbar_mri_cases()
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == len(ids), f"duplicate case_ids: {ids}"


def test_outcome_distribution_matches_design():
    """60-case balanced fixture: 30 approved + 30 denied."""
    cases = build_aetna_lumbar_mri_cases()
    counts = Counter(c.ground_truth_outcome.value for c in cases)
    assert counts["approved"] == 30
    assert counts["denied"] == 30
    assert counts.get("overturned_on_appeal", 0) == 0


def test_all_cases_have_required_fields():
    cases = build_aetna_lumbar_mri_cases()
    for c in cases:
        assert c.clinical_notes, f"{c.case_id} empty clinical_notes"
        assert c.policy_excerpt, f"{c.case_id} empty policy_excerpt"
        assert c.ground_truth_reasoning, f"{c.case_id} empty reasoning"
        assert c.task_config.cpt_code, f"{c.case_id} empty cpt_code"
        assert c.task_config.icd10_codes, f"{c.case_id} empty icd10_codes"


def test_policy_excerpts_reference_cpb_0236():
    """Every case's policy excerpt must reference CPB-0236 (the base
    Aetna lumbar imaging policy). Some cases ALSO reference CPB-0093
    (modality), CMS LCD L33408 (Medicare Advantage), or Aetna Medicaid
    pathway — those are additive, not replacing CPB-0236."""
    cases = build_aetna_lumbar_mri_cases()
    for c in cases:
        assert "CPB-0236" in c.policy_excerpt, (
            f"{c.case_id} policy_excerpt missing CPB-0236"
        )


def test_case_13_and_14_are_doc_completeness_pair():
    """Cases 13 and 14 are the intentional documentation-completeness
    contrast (same claustrophobic patient, different documentation)."""
    cases = {c.case_id: c for c in build_aetna_lumbar_mri_cases()}
    assert "aetna_013" in cases
    assert "aetna_014" in cases
    assert cases["aetna_013"].ground_truth_outcome == Outcome.DENIED
    assert cases["aetna_014"].ground_truth_outcome == Outcome.APPROVED
    # Both reference CPB-0093 (the modality policy)
    assert "CPB-0093" in cases["aetna_013"].policy_excerpt
    assert "CPB-0093" in cases["aetna_014"].policy_excerpt


def test_plan_type_error_cases_reference_correct_external_policy():
    """Case 25 (Medicare Advantage) must reference CMS LCD L33408.
    Case 30 (Medicaid managed care) must reference the Medicaid
    pathway."""
    cases = {c.case_id: c for c in build_aetna_lumbar_mri_cases()}
    assert "L33408" in cases["aetna_025"].policy_excerpt
    assert "Medicaid" in cases["aetna_030"].policy_excerpt
