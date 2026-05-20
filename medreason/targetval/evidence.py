"""EvidenceBundle — structured evidence the per-target agent reasons over.

Each sub-bundle maps to a specific external data source so the ingest
layer can lazy-fill what it has and leave the rest empty. The agent
prompt builder reads whichever sub-bundles are populated.

v0.1 stores all fields as plain JSON-serialisable values; pathway
topology embeddings (Reactome graph features etc.) come later if needed.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Public-data evidence (universal + disease layers can train on this) ───────


class GeneticsEvidence(BaseModel):
    """OpenTargets / GWAS / Mendelian randomisation summary for the target.

    `overall_score` follows the OpenTargets convention [0, 1]. Missing means
    unknown, not zero.
    """

    overall_score: Optional[float] = None
    genetic_association_score: Optional[float] = None
    rare_variant_evidence: Optional[float] = None
    mendelian_randomization_p: Optional[float] = None
    citations: list[str] = Field(default_factory=list)


class KnockoutDependenceEvidence(BaseModel):
    """DepMap CRISPR / RNAi gene-effect summary across cell lines.

    `mean_dependency_score` is the Chronos / CERES mean across cell lines
    relevant to the disease (more negative = stronger dependency).
    `fraction_dependent_lines` is the fraction of relevant lines below the
    -0.5 dependency threshold.
    """

    mean_dependency_score: Optional[float] = None
    fraction_dependent_lines: Optional[float] = None
    n_lines_assessed: Optional[int] = None
    cell_line_context: Optional[str] = None  # e.g. "BRAF_V600E_melanoma_lines"


class PathwayTopologyEvidence(BaseModel):
    """Network-structural features derived from a reference pathway DB
    (Reactome / OmniPath / KEGG). Computed by medreason.targetval.topology.

    These are the features that most directly drive bypass-risk reasoning:
    paralog redundancy and downstream feedback loops are the canonical
    bypass mechanisms.
    """

    paralog_count: Optional[int] = None  # same gene family
    paralogs: list[str] = Field(default_factory=list)
    downstream_redundancy_index: Optional[float] = None  # 0 = unique node, 1 = fully redundant
    upstream_node_count: Optional[int] = None
    downstream_node_count: Optional[int] = None
    known_feedback_loops: list[str] = Field(default_factory=list)
    reference_pathway: Optional[str] = None  # e.g. "Reactome:R-HSA-5683057"


class PriorTrialEvidence(BaseModel):
    """clinicaltrials.gov + literature outcomes for prior drugs against
    this target. Drives "has this target failed in similar indications
    before?" reasoning.
    """

    n_trials_prior: int = 0
    n_phase2_failures_efficacy: int = 0
    n_phase2_safety_failures: int = 0
    n_approvals: int = 0
    summary: Optional[str] = None  # free text, included in agent prompt
    trial_ids: list[str] = Field(default_factory=list)


# ── Optional customer/private evidence (campaign layer only) ──────────────────


class InternalEvidence(BaseModel):
    """De-identified customer evidence (e.g. Recursion phenotypic screen
    readouts). Free-form dict because customer-specific shapes vary. The
    swarm prompt builder serialises this verbatim.

    Stays attached to the case AT INFERENCE TIME only — by leak-guard
    policy, no rule learned from a case carrying populated InternalEvidence
    may be written to the UniversalLayer (only DiseaseLayer or
    CampaignLayer).
    """

    customer_tag: Optional[str] = None  # opaque tenant id
    readouts: dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


# ── The bundle ────────────────────────────────────────────────────────────────


class EvidenceBundle(BaseModel):
    genetics: GeneticsEvidence = Field(default_factory=GeneticsEvidence)
    knockout: KnockoutDependenceEvidence = Field(
        default_factory=KnockoutDependenceEvidence
    )
    topology: PathwayTopologyEvidence = Field(default_factory=PathwayTopologyEvidence)
    prior_trials: PriorTrialEvidence = Field(default_factory=PriorTrialEvidence)
    internal: InternalEvidence = Field(default_factory=InternalEvidence)
    additional_notes: Optional[str] = None

    def has_internal_data(self) -> bool:
        """True if the bundle includes customer-private evidence. Drives
        the leak-guard policy on rule extraction."""
        return bool(self.internal.customer_tag) or bool(self.internal.readouts)
