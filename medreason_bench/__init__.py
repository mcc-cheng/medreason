"""medreason_bench — the MedReason-Bench public benchmark.

Layers:
- data/: policy ingestion (CMS LCD/NCD), case construction, fixtures.
- splits/: stratified train/dev/test partitioning + frozen fingerprints.
- eval/ (Phase 4): harness, metrics, bootstrap CIs, McNemar.
- leaderboard/ (Phase 7): schema + HuggingFace dataset card emitter.

The `AgentRunner` Protocol in medreason.runners.base is the integration
point between this package and the memory pipeline. This package does not
import from medreason.extractor / agent / injector / benchmark (the
pre-rework modules) and never will — the Phase 5 rework replaces them.
"""

__version__ = "0.0.0"
