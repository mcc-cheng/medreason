"""Retrieval Tier 1 — structural SQL-ish filter by CPT family, ICD
chapter, payer, facility.

Phase 5 implementation: loads all ACTIVE rules from the RuleStore via
list_by_status and filters in Python using RuleTrigger.matches(). This
is O(n_rules) per query, which is fine at the Phase 5 rule count
(target ~100). Phase 6+ will introduce denormalized SQL columns and
proper indexes once rule counts grow.

Tier 1 is fast, cheap, and deterministic. It's allowed to be imprecise
— the dense + rerank tiers narrow the result. Returning too many
candidates is fine; returning too few strands the whole pipeline.

Fallback semantics: if no rules match the trigger, Tier 1 returns an
empty list and the pipeline falls through to Tier 2 unfiltered (the
latter is still cheap enough to do full-store cosine). This protects
against stratum sparsity — a novel CPT family in the eval set
shouldn't produce zero retrievals just because the trigger cross has
no hits yet.
"""

from __future__ import annotations

from medreason.ontology import (
    BenchmarkCase,
    PriorAuthTaskConfig,
    ReasoningRule,
    RuleStatus,
)
from medreason.store import RuleStore


def lookup_by_trigger(
    store: RuleStore,
    task_config: PriorAuthTaskConfig,
    *,
    max_candidates: int = 50,
    include_candidate: bool = False,
) -> list[ReasoningRule]:
    """Return ACTIVE rules whose trigger matches the task config.

    Args:
        store: The RuleStore to query.
        task_config: The task config for the case being resolved.
        max_candidates: Upper bound on returned rules. Tier 1 is the
            widest net so this is intentionally generous — the plan
            says ~50.
        include_candidate: If True, also consider CANDIDATE-status
            rules. Used by the generalization gate when evaluating a
            newly-proposed rule that hasn't been promoted yet.

    Returns:
        A list of ReasoningRules, ordered by posterior_mean desc
        (highest-confidence rules first). Ties broken by trial count,
        then by rule_id for stability.
    """
    rules: list[ReasoningRule] = []
    rules.extend(store.list_by_status(RuleStatus.ACTIVE))
    if include_candidate:
        rules.extend(store.list_by_status(RuleStatus.CANDIDATE))

    matching = [r for r in rules if r.trigger.matches(task_config)]

    # Sort by posterior_mean desc, then trials desc, then rule_id asc
    matching.sort(
        key=lambda r: (-r.posterior_mean, -r.trials, r.rule_id)
    )
    return matching[:max_candidates]
