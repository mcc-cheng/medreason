"""TargetValidationCase — the unit the swarm reasons over.

One case per (target, disease_context) pair. The swarm runs one agent per
case, in parallel, and the cross-agent analyzer reads the swarm's
collective output.

Why this is NOT a BenchmarkCase shim (the way drugdisc_cases.py does it):

- The fields are structurally different (gene symbols, not CPT codes).
- The ground-truth labels are different (Phase 2 outcome + bypass mechanism,
  not approved/denied).
- Retrieval triggers are different (gene/family/pathway, not ICD/CPT).

A targetval case CAN be projected into a BenchmarkCase for reuse of the
existing eval harness, but that adapter lives in
medreason_bench/targetval/, not here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .evidence import EvidenceBundle


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Identifiers ───────────────────────────────────────────────────────────────


class TargetID(BaseModel):
    """A protein target. Gene symbol is the primary key; aliases help retrieval."""

    gene_symbol: str  # e.g. "BRAF"
    uniprot_accession: Optional[str] = None  # e.g. "P15056"
    family: Optional[str] = None  # e.g. "RAF_kinase", "MAPK"
    aliases: list[str] = Field(default_factory=list)

    def display(self) -> str:
        return self.gene_symbol


class DiseaseContext(BaseModel):
    """Disease + biomarker context. v0.1 keeps ontology codes optional and
    relies on free-text disease_label + biomarker_context for retrieval.
    """

    disease_label: str  # e.g. "metastatic_melanoma_BRAF_V600E"
    ontology_code: Optional[str] = None  # MONDO / DOID / EFO if available
    biomarker_context: Optional[str] = None  # e.g. "BRAF V600E mutation"
    therapeutic_area: Optional[str] = None  # e.g. "ONCOLOGY"


class Modality(str, Enum):
    """Intended therapeutic modality. Affects which bypass mechanisms are
    plausible (e.g., an antibody can't engage an intracellular bypass)."""

    SMALL_MOLECULE = "small_molecule"
    ANTIBODY = "antibody"
    PROTAC = "protac"
    ASO = "antisense_oligonucleotide"
    SIRNA = "sirna"
    CELL_THERAPY = "cell_therapy"
    GENE_EDITING = "gene_editing"
    UNKNOWN = "unknown"


# ── Ground truth (retrospective benchmark labels) ─────────────────────────────


class GroundTruthOutcome(str, Enum):
    """The verdict from the retrospective. UNKNOWN is the inference-time value."""

    # Trial-level outcomes
    PHASE2_EFFICACY_YES = "phase2_efficacy_yes"
    PHASE2_EFFICACY_NO = "phase2_efficacy_no"
    PHASE2_SAFETY_FAILURE = "phase2_safety_failure"
    PHASE2_WITHDRAWN_OTHER = "phase2_withdrawn_other"
    APPROVED_LATER = "approved_later"
    ACTIVE_UNKNOWN = "active_unknown"
    NEVER_PROGRESSED = "never_progressed"

    # Inference-time
    UNKNOWN = "unknown"


# The bypass-mechanism ground-truth label space. These are the categories
# the swarm's bypass-risk reasoning is graded against in the retro benchmark.
class BypassMechanism(str, Enum):
    NO_BYPASS_KNOWN = "no_bypass_known"
    PARALOG_COMPENSATION = "paralog_compensation"
    DOWNSTREAM_FEEDBACK = "downstream_feedback"
    ALTERNATIVE_PATHWAY = "alternative_pathway"
    MICROENVIRONMENT_RESCUE = "microenvironment_rescue"
    RESISTANCE_MUTATION = "resistance_mutation"
    PHARMACOKINETIC_ESCAPE = "pharmacokinetic_escape"
    OTHER_VALIDATED = "other_validated"
    UNKNOWN = "unknown"


# ── The case ──────────────────────────────────────────────────────────────────


class TargetValidationCase(BaseModel):
    """One target × disease pair the swarm reasons over."""

    case_id: str  # e.g. "tv_braf_melanoma_v600e"
    target: TargetID
    disease: DiseaseContext
    modality: Modality = Modality.UNKNOWN
    evidence: EvidenceBundle

    # Retrospective labels — None at inference, populated for benchmark cases
    ground_truth_outcome: GroundTruthOutcome = GroundTruthOutcome.UNKNOWN
    ground_truth_bypass: BypassMechanism = BypassMechanism.UNKNOWN
    ground_truth_notes: Optional[str] = None

    # Provenance
    source: str = ""  # e.g. "mapk_retro_v0.1", "recursion_engagement_001"
    created_at: datetime = Field(default_factory=_utcnow)

    def is_retrospective(self) -> bool:
        return self.ground_truth_outcome is not GroundTruthOutcome.UNKNOWN
