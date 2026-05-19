"""Synthetic toy targets for end-to-end smoke testing.

Three cases with hand-built evidence + outcomes, chosen so the swarm
should produce different priority/bypass scores for each:

- TVS-001: BRAF/melanoma — classic Phase 2 success, low paralog bypass risk
- TVS-002: KRAS/colorectal — classic Phase 2 efficacy failure with
  downstream feedback bypass (cetuximab story, simplified)
- TVS-003: EGFR/glioblastoma — Phase 2 failure with alternative-pathway
  bypass

These are NOT a benchmark — they exist only to wire the swarm + cross-
agent analyzer end-to-end without spending money on real LLM calls.
"""

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
    KnockoutDependenceEvidence,
    PathwayTopologyEvidence,
    PriorTrialEvidence,
)


SYNTHETIC_CAMPAIGN_ID = "targetval_synthetic_v01"


def _braf_melanoma() -> TargetValidationCase:
    return TargetValidationCase(
        case_id="TVS-001",
        target=TargetID(gene_symbol="BRAF", family="RAF_kinase"),
        disease=DiseaseContext(
            disease_label="metastatic_melanoma_BRAF_V600E",
            therapeutic_area="ONCOLOGY",
            biomarker_context="BRAF V600E mutation",
        ),
        modality=Modality.SMALL_MOLECULE,
        evidence=EvidenceBundle(
            genetics=GeneticsEvidence(
                overall_score=0.92,
                citations=["OpenTargets:ENSG00000157764"],
            ),
            knockout=KnockoutDependenceEvidence(
                mean_dependency_score=-1.2,
                fraction_dependent_lines=0.78,
                cell_line_context="BRAF_V600E_melanoma_lines",
            ),
            topology=PathwayTopologyEvidence(
                paralog_count=2,
                paralogs=["ARAF", "RAF1"],
                downstream_redundancy_index=0.4,
                known_feedback_loops=["MEK-ERK rebound"],
            ),
            prior_trials=PriorTrialEvidence(
                n_trials_prior=12,
                n_approvals=2,
                summary="vemurafenib, dabrafenib approved; MEK combo addresses feedback",
            ),
        ),
        ground_truth_outcome=GroundTruthOutcome.APPROVED_LATER,
        ground_truth_bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        source=SYNTHETIC_CAMPAIGN_ID,
    )


def _kras_crc() -> TargetValidationCase:
    return TargetValidationCase(
        case_id="TVS-002",
        target=TargetID(gene_symbol="KRAS", family="RAS_GTPase"),
        disease=DiseaseContext(
            disease_label="metastatic_colorectal_cancer_KRAS_mut",
            therapeutic_area="ONCOLOGY",
            biomarker_context="KRAS G12 mutation",
        ),
        modality=Modality.SMALL_MOLECULE,
        evidence=EvidenceBundle(
            genetics=GeneticsEvidence(overall_score=0.88),
            knockout=KnockoutDependenceEvidence(
                mean_dependency_score=-0.9,
                fraction_dependent_lines=0.55,
            ),
            topology=PathwayTopologyEvidence(
                paralog_count=2,
                paralogs=["HRAS", "NRAS"],
                downstream_redundancy_index=0.6,
                known_feedback_loops=["EGFR feedback in CRC"],
            ),
            prior_trials=PriorTrialEvidence(
                n_trials_prior=8,
                n_phase2_failures_efficacy=5,
                summary="early KRAS inhibitors failed in CRC due to EGFR feedback",
            ),
        ),
        ground_truth_outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        ground_truth_bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        source=SYNTHETIC_CAMPAIGN_ID,
    )


def _egfr_gbm() -> TargetValidationCase:
    return TargetValidationCase(
        case_id="TVS-003",
        target=TargetID(gene_symbol="EGFR", family="HER_kinase"),
        disease=DiseaseContext(
            disease_label="glioblastoma_EGFRvIII",
            therapeutic_area="ONCOLOGY",
            biomarker_context="EGFRvIII variant",
        ),
        modality=Modality.SMALL_MOLECULE,
        evidence=EvidenceBundle(
            genetics=GeneticsEvidence(overall_score=0.7),
            knockout=KnockoutDependenceEvidence(
                mean_dependency_score=-0.3,
                fraction_dependent_lines=0.25,
            ),
            topology=PathwayTopologyEvidence(
                paralog_count=3,
                paralogs=["ERBB2", "ERBB3", "ERBB4"],
                downstream_redundancy_index=0.7,
                known_feedback_loops=["MET amplification rescue"],
            ),
            prior_trials=PriorTrialEvidence(
                n_trials_prior=15,
                n_phase2_failures_efficacy=11,
                summary="EGFR-TKI uniformly failed in glioblastoma despite biomarker",
            ),
        ),
        ground_truth_outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        ground_truth_bypass=BypassMechanism.ALTERNATIVE_PATHWAY,
        source=SYNTHETIC_CAMPAIGN_ID,
    )


def build_synthetic_targets() -> list[TargetValidationCase]:
    return [_braf_melanoma(), _kras_crc(), _egfr_gbm()]
