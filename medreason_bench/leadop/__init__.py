"""Lead-optimization retrospective harness.

Separate from the medical-auth MedReason pipeline. Shape is
sequential-decision campaigns: compounds + assay readouts grouped
into SAR rounds, with a decision point between each round. Goal is
to evaluate whether a metacognitive reasoning-trace memory improves
the agent's top-1 direction pick (potency / selectivity / ADMET /
scaffold hop) over a baseline RAG agent.

Failure-driven rule extraction is intentionally excluded from this
harness per the office-hours plan (Experiment E showed it memorizes,
not generalizes).
"""

from .schema import (
    CAMPAIGN_SCHEMA_VERSION,
    CompoundRow,
    DecisionPointRow,
    connect_campaign_db,
    create_campaign_db,
)
from .scaffolds import murcko_scaffold_smiles, compute_descriptors

__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "CompoundRow",
    "DecisionPointRow",
    "connect_campaign_db",
    "create_campaign_db",
    "murcko_scaffold_smiles",
    "compute_descriptors",
]
