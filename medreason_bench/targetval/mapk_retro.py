"""Public MAPK retrospective fixture — v0.2.

This module is the public-data benchmark anchor: a hand-curated list of
historical MAPK-pathway targets with documented Phase 2 outcomes from
the literature + clinicaltrials.gov, used as the validation surface for
the swarm + cross-agent metacognitive memory.

Versions
--------
- ``v0.1`` — the original 3-entry illustrative seed. Kept available
  for back-compat with tests / scripts that pinned the smaller fixture
  before the Phase 2 expansion.
- ``v0.2`` — the curated ~22-entry table sourced from
  :mod:`mapk_retro_data`. Covers BRAF/MEK/ERK/KRAS/NRAS/HRAS/EGFR/MET/
  ALK/FGFR2/SHP2/CDK4/PI3K-alpha/WEE1/AXL/MDM2/pan-RAF with a mix of
  approvals and Phase 2 failures spanning all five non-UNKNOWN
  ``BypassMechanism`` categories that the swarm cross-agent analyzer
  scores against.

Universal-safe surface
----------------------
Every entry's ``InternalEvidence`` sub-bundle is left empty by
construction (``_entry`` does not touch it). This is the leak-guard:
no rule learned from these cases may regress to a tenant-scoped layer.

Working list (covered in v0.2):

- BRAF / V600E melanoma — approved (vemurafenib/dabrafenib).
- BRAF / V600E CRC — Phase 2 efficacy NO (feedback bypass).
- BRAF / V600E thyroid — approved (dabrafenib+trametinib).
- MEK1/2 (MAP2K1/MAP2K2) — selumetinib NF1 approved; trametinib KRAS-lung failed.
- ERK1/2 (MAPK1/MAPK3) — Phase 2 failures, downstream feedback.
- KRAS G12C / NSCLC — sotorasib/adagrasib approved.
- KRAS G12C / CRC — sub-par response, EGFR feedback bypass.
- NRAS / melanoma — multiple Phase 2 failures.
- HRAS / HNSCC — tipifarnib, paralog escape.
- EGFR / NSCLC — multiple TKI approvals.
- MET / NSCLC exon 14 — approved (capmatinib, tepotinib).
- MET / EGFR-TKI bypass — Phase 2 failed.
- ALK / NSCLC — multiple TKI approvals.
- SHP2 (PTPN11) — combo Phase 1/2 ongoing.
- FGFR2 / cholangiocarcinoma — pemigatinib approved.
- CDK4/6 / breast cancer — palbociclib approved.
- PI3K-alpha / breast cancer — alpelisib approved.
- WEE1 / solid tumors — adavosertib failed.
- AXL / TNBC — bemcentinib failed.
- MDM2 / liposarcoma — milademetan safety-limited.
- pan-RAF / RAS-mutant solid — belvarafenib Phase 2 limited.
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

from .mapk_retro_data import MAPK_RETRO_ENTRIES_V0_2, MapkRetroEntry


MAPK_RETRO_CAMPAIGN_ID = "mapk_retro_v0.2"

# Legacy campaign id retained so callers can pin to the v0.1 fixture for
# backwards compatibility with prediction cards already committed.
MAPK_RETRO_CAMPAIGN_ID_V0_1 = "mapk_retro_v0.1"


# ── Builders ─────────────────────────────────────────────────────────────────


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
    source: str = MAPK_RETRO_CAMPAIGN_ID,
) -> TargetValidationCase:
    """Lower-level builder used by both v0.1 and v0.2 paths.

    ``EvidenceBundle.internal`` is deliberately left at its default
    (empty) — every entry in this module is Universal-safe.
    """
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
        source=source,
    )


def _entry_from(e: MapkRetroEntry) -> TargetValidationCase:
    """Convert one curated ``MapkRetroEntry`` into a TargetValidationCase."""
    return _entry(
        case_id=e.case_id,
        gene=e.gene,
        family=e.family,
        disease=e.disease,
        paralogs=list(e.paralogs),
        paralog_count=e.paralog_count,
        redundancy=e.redundancy,
        feedback_loops=list(e.feedback_loops),
        n_trials=e.n_trials,
        n_p2_fail=e.n_p2_fail,
        n_approvals=e.n_approvals,
        trial_summary=e.trial_summary,
        outcome=e.outcome,
        bypass=e.bypass,
        notes=e.notes,
        source=MAPK_RETRO_CAMPAIGN_ID,
    )


# ── Public API ───────────────────────────────────────────────────────────────


def build_mapk_retro_seed(
    *, version: str = "v0.2"
) -> list[TargetValidationCase]:
    """Return MAPK retro cases for the requested fixture version.

    ``version="v0.2"`` (default) returns the ~22-entry curated set.
    ``version="v0.1"`` returns the 3-entry legacy seed — kept for
    back-compat with anything that pinned the smaller fixture.
    """
    if version == "v0.1":
        return _legacy_v0_1_seed()
    if version == "v0.2":
        return [_entry_from(e) for e in MAPK_RETRO_ENTRIES_V0_2]
    raise ValueError(
        f"Unknown mapk_retro version {version!r}; expected 'v0.1' or 'v0.2'"
    )


def _legacy_v0_1_seed() -> list[TargetValidationCase]:
    """The original 3-entry illustrative seed (preserved verbatim).

    These cases stamp ``source=MAPK_RETRO_CAMPAIGN_ID_V0_1`` so the
    provenance distinguishes the legacy fixture from the v0.2 curated
    set.
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
            trial_summary=(
                "vemurafenib/dabrafenib approved; MEK combos extend PFS"
            ),
            outcome=GroundTruthOutcome.APPROVED_LATER,
            bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
            notes="Bypass exists but is addressed by combination therapy.",
            source=MAPK_RETRO_CAMPAIGN_ID_V0_1,
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
            trial_summary=(
                "BRAF-i monotherapy failed in CRC; combo with cetuximab needed"
            ),
            outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
            bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
            notes=(
                "Tissue-specific feedback bypass — same target, different "
                "disease."
            ),
            source=MAPK_RETRO_CAMPAIGN_ID_V0_1,
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
            trial_summary=(
                "sotorasib approved in NSCLC; ongoing combo trials for "
                "resistance"
            ),
            outcome=GroundTruthOutcome.APPROVED_LATER,
            bypass=BypassMechanism.RESISTANCE_MUTATION,
            notes="On-target resistance via secondary KRAS mutations.",
            source=MAPK_RETRO_CAMPAIGN_ID_V0_1,
        ),
    ]
