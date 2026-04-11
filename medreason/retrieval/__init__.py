"""Phase 5 retrieval — three-tier pipeline + structured rule injector.

Tier 1 (ontology_lookup): structural SQL-ish filter by payer, CPT
    family, ICD chapter, facility. Cheap, deterministic.
Tier 2 (dense_rerank): cosine similarity against a structured case
    representation using an Embedder. FakeEmbedder for tests;
    OpenAIEmbedder (text-embedding-3-large) in production. NO
    deterministic-hash fallback — the plan explicitly forbids it.
Tier 3 (rerank): LLM call scoring applicability + expected utility.
    Final rank = applicability × expected_utility × posterior_mean.
    Drops rules below min_applicability.

retrieve() in pipeline.py composes all three. Empty output cascades
to "no relevant memory" and the memory wrapper falls back to
zero-shot prompting — that is CORRECT behavior, not a bug.

The injector (same package) turns a list of retrieved rules into a
`system_extra` block for AgentRunner.run().
"""

from .dense import dense_rerank
from .embedder import (
    DEFAULT_OPENAI_EMBED_MODEL,
    Embedder,
    FakeEmbedder,
    OPENAI_EMBED_MODELS,
    OpenAIEmbedder,
    cosine_similarity,
)
from .injector import (
    ACTION_MAX_WORDS,
    APPLIED_RULES_FIELD,
    build_rule_checklist,
    missing_rule_ids,
    parse_applied_rules,
)
from .ontology_lookup import lookup_by_trigger
from .pipeline import RetrievalResult, retrieve
from .rerank import RerankScore, rerank

__all__ = [
    # Injector
    "ACTION_MAX_WORDS",
    "APPLIED_RULES_FIELD",
    "build_rule_checklist",
    "parse_applied_rules",
    "missing_rule_ids",
    # Embedder
    "Embedder",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "OPENAI_EMBED_MODELS",
    "DEFAULT_OPENAI_EMBED_MODEL",
    "cosine_similarity",
    # Tiers
    "lookup_by_trigger",
    "dense_rerank",
    "rerank",
    "RerankScore",
    # Pipeline
    "retrieve",
    "RetrievalResult",
]
