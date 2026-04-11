"""Three-tier retrieval pipeline — composes ontology lookup + dense +
rerank into a single retrieve(store, case, ...) call.

Flow:

    Tier 1 (ontology_lookup): structural filter by payer / CPT family /
                              ICD chapter / facility. Up to ~50 candidates.
    Tier 2 (dense_rerank):    cosine similarity against case repr using
                              the embedder. Top 25.
    Tier 3 (rerank):          LLM scores applicability + expected utility;
                              final rank = raw × posterior_mean. Top ~6.

Empty output from any tier is valid — it cascades to "no relevant
memory for this case", which the memory wrapper handles by falling
back to zero-shot prompting. That is NOT a bug; it is the correct
behavior when the memory store has nothing useful.

Two composition knobs:
- `skip_rerank=True` short-circuits Tier 3 and returns the top N from
  Tier 2 (ordered by cosine × posterior). Used by the generalization
  gate, which evaluates a single candidate rule in isolation and
  doesn't want a reranker to drop it.
- `include_candidate=True` makes Tier 1 also return CANDIDATE-status
  rules. Used when retrieving for the generalization gate's held-out
  pass on a not-yet-promoted rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from medreason.ontology import BenchmarkCase, ReasoningRule
from medreason.store import RuleStore

from ..llm.base import LLMClient
from .dense import dense_rerank
from .embedder import Embedder
from .ontology_lookup import lookup_by_trigger
from .rerank import rerank


@dataclass
class RetrievalResult:
    """What retrieve() returns — both the final rules and per-tier
    counts for audit logging / leaderboard diagnostics.
    """
    rules: list[ReasoningRule] = field(default_factory=list)
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    reranker_called: bool = False


def retrieve(
    store: RuleStore,
    case: BenchmarkCase,
    *,
    embedder: Embedder,
    reranker_llm: Optional[LLMClient] = None,
    tier1_max: int = 50,
    tier2_top_k: int = 25,
    tier3_top_k: int = 6,
    min_applicability: float = 4.0,
    include_candidate: bool = False,
    skip_rerank: bool = False,
) -> RetrievalResult:
    """Run the full three-tier pipeline.

    Args:
        store: Memory store to query.
        case: Case being resolved.
        embedder: Used for Tier 2. OpenAIEmbedder in production,
            FakeEmbedder in tests.
        reranker_llm: LLMClient for Tier 3. If None OR `skip_rerank`
            is True, Tier 3 is skipped entirely.
        tier1_max: Tier 1 candidate cap.
        tier2_top_k: Tier 2 candidate cap.
        tier3_top_k: Final result cap after reranking.
        min_applicability: Tier 3 applicability floor.
        include_candidate: Pass through to Tier 1 — include
            CANDIDATE-status rules (used by the generalization gate).
        skip_rerank: Skip Tier 3 entirely, return Tier 2's top cap
            ordered by cosine × posterior.
    """
    result = RetrievalResult()

    # Tier 1
    tier1 = lookup_by_trigger(
        store,
        case.task_config,
        max_candidates=tier1_max,
        include_candidate=include_candidate,
    )
    result.tier1_count = len(tier1)

    # Tier 1 fallback: if nothing matched structurally, widen to the
    # full ACTIVE set so the dense tier still gets a chance. Sparse
    # benchmarks at Phase 5 scale make this worth the extra compute.
    if not tier1:
        from medreason.ontology import RuleStatus
        widen = store.list_by_status(RuleStatus.ACTIVE)
        if include_candidate:
            widen.extend(store.list_by_status(RuleStatus.CANDIDATE))
        tier1 = widen[:tier1_max]
        result.tier1_count = len(tier1)

    if not tier1:
        return result

    # Tier 2
    tier2_scored = dense_rerank(case, tier1, embedder, top_k=tier2_top_k)
    tier2_rules = [rule for rule, _ in tier2_scored]
    result.tier2_count = len(tier2_rules)
    if not tier2_rules:
        return result

    # Tier 3
    if skip_rerank or reranker_llm is None:
        # Take the top tier3_top_k from Tier 2's ordering. Already
        # sorted by cosine × posterior (see dense_rerank's tiebreak).
        result.rules = tier2_rules[:tier3_top_k]
        result.tier3_count = len(result.rules)
        return result

    result.reranker_called = True
    tier3 = rerank(
        case,
        tier2_rules,
        reranker_llm,
        top_k=tier3_top_k,
        min_applicability=min_applicability,
    )
    result.rules = tier3
    result.tier3_count = len(tier3)
    return result
