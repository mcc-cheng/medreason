"""Tests for medreason.targetval.case + evidence dataclasses."""

from __future__ import annotations

from medreason.targetval.case import (
    BypassMechanism,
    DiseaseContext,
    GroundTruthOutcome,
    Modality,
    TargetID,
    TargetValidationCase,
)
from medreason.targetval.evidence import (
    EvidenceBundle,
    GeneticsEvidence,
    InternalEvidence,
    PathwayTopologyEvidence,
)


def _braf() -> TargetValidationCase:
    return TargetValidationCase(
        case_id="tv_braf",
        target=TargetID(gene_symbol="BRAF", family="RAF_kinase"),
        disease=DiseaseContext(
            disease_label="metastatic_melanoma_BRAF_V600E",
            therapeutic_area="ONCOLOGY",
        ),
        modality=Modality.SMALL_MOLECULE,
        evidence=EvidenceBundle(
            genetics=GeneticsEvidence(overall_score=0.9),
            topology=PathwayTopologyEvidence(paralog_count=2, paralogs=["ARAF", "RAF1"]),
        ),
        ground_truth_outcome=GroundTruthOutcome.APPROVED_LATER,
        ground_truth_bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
    )


def test_case_round_trip_via_pydantic():
    c = _braf()
    payload = c.model_dump(mode="json")
    rebuilt = TargetValidationCase.model_validate(payload)
    assert rebuilt.target.gene_symbol == "BRAF"
    assert rebuilt.disease.disease_label == "metastatic_melanoma_BRAF_V600E"
    assert rebuilt.evidence.topology.paralog_count == 2
    assert rebuilt.ground_truth_bypass is BypassMechanism.DOWNSTREAM_FEEDBACK


def test_is_retrospective_true_when_outcome_populated():
    c = _braf()
    assert c.is_retrospective() is True


def test_is_retrospective_false_when_outcome_unknown():
    c = _braf()
    c2 = c.model_copy(update={"ground_truth_outcome": GroundTruthOutcome.UNKNOWN})
    assert c2.is_retrospective() is False


def test_evidence_bundle_detects_internal_data():
    bundle_public = EvidenceBundle(
        genetics=GeneticsEvidence(overall_score=0.5),
    )
    assert bundle_public.has_internal_data() is False

    bundle_internal = EvidenceBundle(
        internal=InternalEvidence(customer_tag="cust_x", readouts={"phenotype_a": 0.1}),
    )
    assert bundle_internal.has_internal_data() is True


def test_default_evidence_bundle_is_empty_not_zero():
    """Missing evidence must read as None, not 0 — the agent prompt
    builder distinguishes 'no data' from 'low value'."""
    bundle = EvidenceBundle()
    assert bundle.genetics.overall_score is None
    assert bundle.knockout.mean_dependency_score is None
    assert bundle.topology.paralog_count is None
