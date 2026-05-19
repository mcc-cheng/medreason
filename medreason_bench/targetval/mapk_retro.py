"""Public MAPK retrospective fixture — v0.1 seed.

This module is the public-data benchmark anchor: a hand-curated list of
~20-30 historical MAPK-pathway targets with documented Phase 2 outcomes
from the literature + clinicaltrials.gov, used as the validation surface
for the swarm + cross-agent metacognitive memory.

Status: STUB. Three example entries to lock in the shape; the full
~20-30 set is a follow-up commit (the curation is research work, not
scaffolding). Targets to add (working list):

- BRAF / V600E melanoma — approved (vemurafenib/dabrafenib).
- BRAF / V600E CRC — Phase 2 efficacy NO (paralog/feedback bypass).
- MEK1/2 (MAP2K1/MAP2K2) — selumetinib NF1, trametinib combos.
- ERK1/2 (MAPK1/MAPK3) — Phase 2 failures, downstream feedback.
- KRAS G12C / NSCLC — sotorasib approved.
- KRAS G12C / CRC — sub-par response, EGFR feedback bypass.
- NRAS / melanoma — multiple Phase 2 failures.
- HRAS / HNSCC — tipifarnib, limited.
- SHP2 (PTPN11) — multiple Phase 2 ongoing.
- RAF dimerization inhibitors (pan-RAF).
- RAS-MAPK combos (BRAF+MEK, KRAS+SHP2).
- MET amplification / lung — bypass after EGFR-TKI.
- ALK / NSCLC — crizotinib approved, resistance mechanisms documented.
- FGFR2/3 / cholangiocarcinoma — pemigatinib approved.
- AXL / TNBC — bemcentinib Phase 2 failures.
- DDR1/2 / fibrosis — minimal clinical signal.
- CDK4/6 / breast cancer — palbociclib approved.
- MDM2 / TP53-WT cancers — Phase 2 mixed.
- WEE1 / solid tumors — adavosertib, Phase 2 mixed.
- PI3K-alpha / breast cancer — alpelisib approved.

The seed below is illustrative — the bypass labels are simplified and
DO NOT replace careful primary-literature curation. Replace before any
prediction-card commitments.
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
    PathwayTopologyEvidence,
    PriorTrialEvidence,
)


MAPK_RETRO_CAMPAIGN_ID = "mapk_retro_v0.1"


def _entry(
    *,
    case_id: str,
    gene: str,
    family: str,
    disease: str,
    paralogs: list[str],
    paralog_count: int,
    redundancy: float,
    feedback_loops: list[str],
    n_trials: int,
    n_p2_fail: int,
    n_approvals: int,
    trial_summary: str,
    outcome: GroundTruthOutcome,
    bypass: BypassMechanism,
    notes: str = "",
) -> TargetValidationCase:
    return TargetValidationCase(
        case_id=case_id,
        target=TargetID(gene_symbol=gene, family=family),
        disease=DiseaseContext(
            disease_label=disease, therapeutic_area="ONCOLOGY"
        ),
        modality=Modality.SMALL_MOLECULE,
        evidence=EvidenceBundle(
            genetics=GeneticsEvidence(overall_score=None),
            topology=PathwayTopologyEvidence(
                paralog_count=paralog_count,
                paralogs=paralogs,
                downstream_redundancy_index=redundancy,
                known_feedback_loops=feedback_loops,
            ),
            prior_trials=PriorTrialEvidence(
                n_trials_prior=n_trials,
                n_phase2_failures_efficacy=n_p2_fail,
                n_approvals=n_approvals,
                summary=trial_summary,
            ),
        ),
        ground_truth_outcome=outcome,
        ground_truth_bypass=bypass,
        ground_truth_notes=notes,
        source=MAPK_RETRO_CAMPAIGN_ID,
    )


def build_mapk_retro_seed() -> list[TargetValidationCase]:
    """Return the v0.1 SEED for the MAPK retro. Three entries — replace
    with the full ~20-30 curated set before any prediction-card commits.
    """
    return [
        _entry(
            case_id="MAPK-001",
            gene="BRAF",
            family="RAF_kinase",
            disease="metastatic_melanoma_BRAF_V600E",
            paralogs=["ARAF", "RAF1"],
            paralog_count=2,
            redundancy=0.4,
            feedback_loops=["MEK-ERK rebound after BRAF-i monotherapy"],
            n_trials=12,
            n_p2_fail=2,
            n_approvals=2,
            trial_summary="vemurafenib/dabrafenib approved; MEK combos extend PFS",
            outcome=GroundTruthOutcome.APPROVED_LATER,
            bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
            notes="Bypass exists but is addressed by combination therapy.",
        ),
        _entry(
            case_id="MAPK-002",
            gene="BRAF",
            family="RAF_kinase",
            disease="metastatic_colorectal_BRAF_V600E",
            paralogs=["ARAF", "RAF1"],
            paralog_count=2,
            redundancy=0.65,
            feedback_loops=["EGFR feedback amplification post-BRAF-i"],
            n_trials=8,
            n_p2_fail=5,
            n_approvals=0,
            trial_summary="BRAF-i monotherapy failed in CRC; combo with cetuximab needed",
            outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
            bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
            notes="Tissue-specific feedback bypass — same target, different disease.",
        ),
        _entry(
            case_id="MAPK-003",
            gene="KRAS",
            family="RAS_GTPase",
            disease="metastatic_NSCLC_KRAS_G12C",
            paralogs=["HRAS", "NRAS"],
            paralog_count=2,
            redundancy=0.45,
            feedback_loops=["receptor tyrosine kinase reactivation"],
            n_trials=10,
            n_p2_fail=4,
            n_approvals=1,
            trial_summary="sotorasib approved in NSCLC; ongoing combo trials for resistance",
            outcome=GroundTruthOutcome.APPROVED_LATER,
            bypass=BypassMechanism.RESISTANCE_MUTATION,
            notes="On-target resistance via secondary KRAS mutations.",
        ),
    ]
