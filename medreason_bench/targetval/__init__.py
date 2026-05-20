"""medreason_bench.targetval — retrospective benchmark for target validation.

The harness mirrors the leadop/ pattern:

- A DuckDB campaign schema (targets + bypass_outcomes tables).
- A public-data retro fixture (mapk_retro.py) seeded with ~20-30
  historical MAPK-pathway targets with known Phase 2 outcomes.
- A Recursion-shaped ingest layer (recursion_ingest.py) for
  de-identified customer engagements.
- A synthetic stand-in (synthetic.py) for end-to-end smoke testing.
- Metrics + a pre-registered prediction card.

Status: SKELETON. No network, no real benchmarks loaded, no LLM runs.
"""

from .metrics import (
    BypassPrecRec,
    bootstrap_ci,
    bypass_precision_recall,
    top_k_target_hit,
)
from .prediction_card import TargetValPredictionCard, TargetValPredictionEnvelope
from .schemas import (
    TARGETVAL_CAMPAIGN_SCHEMA_VERSION,
    connect_targetval_db,
)

__all__ = [
    "BypassPrecRec",
    "TARGETVAL_CAMPAIGN_SCHEMA_VERSION",
    "TargetValPredictionCard",
    "TargetValPredictionEnvelope",
    "bootstrap_ci",
    "bypass_precision_recall",
    "connect_targetval_db",
    "top_k_target_hit",
]
