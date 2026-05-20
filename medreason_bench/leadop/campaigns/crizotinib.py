"""Crizotinib (PF-02341066) campaign — round boundaries + source paper.

Canonical SAR paper:
  Cui et al. "Structure Based Drug Design of Crizotinib (PF-02341066),
  a Potent and Selective Dual Inhibitor of Mesenchymal–Epithelial
  Transition Factor (c-MET) Kinase and Anaplastic Lymphoma Kinase
  (ALK)." J. Med. Chem. 2011, 54, 6342–6363.
  DOI: 10.1021/jm2007613

The published SAR progression reconstructs as four rounds, separated
by three decision points. The specific compound-ID mappings and team
directions marked below MUST be verified against the paper text
before the dry-run runs — the values are placeholders in the required
shape, NOT an authoritative annotation.

Target ChEMBL IDs (for RAG corpus assembly, not compound ingest):
  - c-MET:  CHEMBL3717
  - ALK:    CHEMBL4247

Crizotinib molecule:  CHEMBL601719
"""

from __future__ import annotations

from dataclasses import dataclass

CAMPAIGN_ID = "crizotinib_pf02341066"

SOURCE_PAPER_DOI = "10.1021/jm2007613"
SOURCE_PAPER_CITATION = (
    "Cui et al. J. Med. Chem. 2011, 54, 6342–6363. "
    "DOI: 10.1021/jm2007613."
)

TARGET_CHEMBL_IDS: dict[str, str] = {
    "c-MET": "CHEMBL3717",
    "ALK": "CHEMBL4247",
}
CRIZOTINIB_CHEMBL_ID = "CHEMBL601719"


@dataclass(frozen=True)
class RoundAnnotation:
    round_index: int
    round_label: str
    description: str
    compound_chembl_ids: tuple[str, ...]  # to hand-annotate from paper tables
    # Direction the team chose AT THE START of this round (i.e. direction
    # for this round's synthesis plan). Filled by human annotator.
    team_direction_chosen: str | None  # one of {potency, selectivity, ADMET, scaffold_hop}
    note: str


# NOTE: ROUND_ANNOTATIONS below is a SHAPE placeholder. The round
# labels, descriptions, and team directions reflect the published
# Cui et al 2011 narrative at a high level, but the compound_chembl_ids
# lists are empty and must be populated by the annotator with reference
# to the paper's Table 1-6 compound structures. See handoff spec.
ROUND_ANNOTATIONS: tuple[RoundAnnotation, ...] = (
    RoundAnnotation(
        round_index=1,
        round_label="HTS hit + initial SAR",
        description=(
            "Starting from the HTS hit class (3-benzyloxy-2-aminopyridine "
            "c-MET inhibitors), establish baseline potency and early SAR "
            "around the 3-position."
        ),
        compound_chembl_ids=(),  # TODO: annotate from paper Table 1
        team_direction_chosen=None,  # N/A — no prior DP
        note="Round 1 has no decision-point predecessor; seeds the campaign.",
    ),
    RoundAnnotation(
        round_index=2,
        round_label="potency optimization (2-aminopyridine core)",
        description=(
            "Scan 2-amino substituents and 3-position benzyloxy groups to "
            "drive c-MET IC50 down."
        ),
        compound_chembl_ids=(),  # TODO: annotate from paper Table 2
        team_direction_chosen="potency",
        note=(
            "Per Cui et al narrative, round 2 prioritizes raw potency before "
            "confronting hERG / CYP issues. Verify direction label against paper."
        ),
    ),
    RoundAnnotation(
        round_index=3,
        round_label="selectivity + 3D binding mode pivot",
        description=(
            "Switch to 3-(1-phenylethoxy) to establish the defined 3D binding "
            "mode with c-MET; address selectivity over related kinases."
        ),
        compound_chembl_ids=(),  # TODO: annotate from paper Table 3-4
        team_direction_chosen="selectivity",
        note=(
            "The paper documents a scaffold-hop / binding-mode shift here. "
            "May be better labeled as scaffold_hop — verify."
        ),
    ),
    RoundAnnotation(
        round_index=4,
        round_label="ADMET polish to crizotinib",
        description=(
            "Address hERG, CYP3A4 TDI, clogP, and oral exposure to reach "
            "crizotinib (CHEMBL601719)."
        ),
        compound_chembl_ids=(CRIZOTINIB_CHEMBL_ID,),
        team_direction_chosen="ADMET",
        note=(
            "Round 4 is the ADMET polish round in the published narrative."
        ),
    ),
)
