"""MemoryRunner — wraps any AgentRunner with the Phase 5 memory pipeline.

Flow per case:

    case
     │
     ├─► retrieval.pipeline.retrieve(store, case, embedder, reranker_llm)
     │     → RetrievalResult with top rules
     │
     ├─► build_rule_checklist(rules) → system_extra block
     │
     ├─► base_runner.run(case, seed, system_extra=injection)
     │     → AgentResult (with applied_rules parsed by the runner)
     │
     ├─► missing_rule_ids(retrieved, result.applied_rules)
     │     if any: re-prompt ONCE with the same injection plus a
     │     "you missed: X, Y" nudge. If still incomplete, treat the
     │     missing rules as applied=False for posterior math.
     │
     └─► store.update_posteriors(retrieved_ids, correct, applied_map,
                                 case_id)
           (skipped if the store's LeakGuard is in test-phase read-only
            mode — the guard itself will raise if we try.)

MemoryRunner itself implements AgentRunner, so it can be passed to
eval/harness.run_eval() just like ClaudeRunner. The `runner_id` gets
a `:memory` suffix so leaderboard entries distinguish zero-shot from
memory-augmented runs.

The wrapper deliberately does NOT implement the Phase 5 rule-learning
loop (critic → proposer → gate → store.put). That is the harness's
job at training time: MemoryRunner is an inference-time composition
that treats the rule store as read-only-ish (only the posterior counts
get updated; no new rules are written). This keeps the eval path free
of extraction cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.base import LLMClient
from ..ontology import AgentResult, AppliedRule, BenchmarkCase
from ..retrieval import (
    Embedder,
    build_rule_checklist,
    missing_rule_ids,
    retrieve,
)
from ..store import RuleStore, TestSetLeakError
from .base import AgentRunner


@dataclass
class MemoryRunnerStats:
    """Diagnostic counters accumulated across all run() calls.

    The eval harness doesn't look at these directly — pattern
    utilization and retrieval counts are computed from the
    AgentResult.applied_rules / .retrieved_rule_ids fields. These
    counters are for ad-hoc debugging and CLI dashboard output.
    """
    total_calls: int = 0
    total_retrievals: int = 0
    total_reranker_calls: int = 0
    total_re_prompts: int = 0
    total_posterior_updates: int = 0
    total_posterior_skips_readonly: int = 0


class MemoryRunner:
    """An AgentRunner that wraps a base runner with memory retrieval.

    Phase 5 inference-time composition. Does NOT write new rules to
    the store — only updates posteriors on retrieved rules.
    """
    runner_id: str
    model_version: str
    supports_memory: bool

    def __init__(
        self,
        *,
        base_runner: AgentRunner,
        store: RuleStore,
        embedder: Embedder,
        reranker_llm: Optional[LLMClient] = None,
        tier1_max: int = 50,
        tier2_top_k: int = 25,
        tier3_top_k: int = 6,
        min_applicability: float = 4.0,
        allow_reprompt: bool = True,
    ):
        self._base = base_runner
        self._store = store
        self._embedder = embedder
        self._reranker_llm = reranker_llm
        self._tier1_max = tier1_max
        self._tier2_top_k = tier2_top_k
        self._tier3_top_k = tier3_top_k
        self._min_applicability = min_applicability
        self._allow_reprompt = allow_reprompt

        self.runner_id = f"{base_runner.runner_id}:memory"
        self.model_version = base_runner.model_version
        self.supports_memory = True
        self.stats = MemoryRunnerStats()

    def estimated_cost_per_call(self) -> float:
        # Rough: base runner call + potential reranker call. The real
        # cost is computed post-hoc on the AgentResult's cost_usd.
        base = self._base.estimated_cost_per_call()
        rerank_extra = 0.003 if self._reranker_llm is not None else 0.0
        return base + rerank_extra

    def run(
        self,
        case: BenchmarkCase,
        *,
        seed: int = 0,
        system_extra: str = "",
    ) -> AgentResult:
        """Run one memory-augmented call.

        `system_extra` from the caller is OVERRIDDEN by the retrieval
        pipeline's injection. If a caller is trying to stack their own
        injection on top of memory, they should do so at a higher layer
        — mixing caller-supplied system_extra with retrieved rules at
        this level would break the applied_rules contract.
        """
        self.stats.total_calls += 1

        retrieval = retrieve(
            self._store,
            case,
            embedder=self._embedder,
            reranker_llm=self._reranker_llm,
            tier1_max=self._tier1_max,
            tier2_top_k=self._tier2_top_k,
            tier3_top_k=self._tier3_top_k,
            min_applicability=self._min_applicability,
        )
        self.stats.total_retrievals += 1
        if retrieval.reranker_called:
            self.stats.total_reranker_calls += 1

        retrieved_ids = [r.rule_id for r in retrieval.rules]
        injection = build_rule_checklist(retrieval.rules)

        # First pass
        result = self._base.run(case, seed=seed, system_extra=injection)

        # Re-prompt once if the agent missed any rule_ids. The
        # base runner is responsible for parsing applied_rules from
        # its own response — ClaudeRunner now does this. Runners that
        # don't parse applied_rules will have result.applied_rules
        # empty, and we'll treat every rule as "not applied" for
        # posterior purposes.
        missing = missing_rule_ids(retrieved_ids, result.applied_rules)
        if missing and self._allow_reprompt and retrieved_ids:
            self.stats.total_re_prompts += 1
            retry_injection = (
                injection
                + "\n\n[RETRY] Your previous response did not include "
                + f"applied_rules for rule_id(s): {', '.join(missing)}. "
                + "You MUST emit exactly one applied_rules entry per "
                + "retrieved rule_id. Retry the full output schema."
            )
            retry_result = self._base.run(
                case, seed=seed, system_extra=retry_injection
            )
            # Keep the retry's determination + applied_rules. Merge
            # usage counts so cost reflects both calls.
            retry_result.input_tokens += result.input_tokens
            retry_result.output_tokens += result.output_tokens
            retry_result.cost_usd += result.cost_usd
            retry_result.latency_ms += result.latency_ms
            result = retry_result

        # Attach retrieval diagnostics to the final result
        result.retrieved_rule_ids = list(retrieved_ids)
        result.runner_id = self.runner_id
        result.mode = "memory"

        # Normalize applied_rules: ensure every retrieved rule_id has
        # an entry. Missing ones (after the retry) are recorded as
        # applied=False for posterior accounting.
        applied_map = _build_applied_map(retrieved_ids, result.applied_rules)
        result.applied_rules = [
            AppliedRule(
                rule_id=rid,
                applied=applied_map[rid],
                rationale=_rationale_for(result.applied_rules, rid),
            )
            for rid in retrieved_ids
        ]

        # Update posteriors. If the store is in read-only mode (eval
        # harness flipped LeakGuard.enter_test_phase), the guard will
        # raise TestSetLeakError — we catch it and count it, because
        # during a scored test run the "don't write" behavior is
        # correct, not an error.
        if retrieved_ids:
            try:
                self._store.update_posteriors(
                    rule_ids=retrieved_ids,
                    correct=result.correct,
                    applied_map=applied_map,
                    case_id=case.case_id,
                )
                self.stats.total_posterior_updates += 1
            except TestSetLeakError:
                self.stats.total_posterior_skips_readonly += 1

        return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_applied_map(
    retrieved_ids: list[str],
    applied: list[AppliedRule],
) -> dict[str, bool]:
    """Build a {rule_id: applied_bool} map over the retrieval set.

    Entries present in `applied` take their bool from there. Retrieved
    rules the agent didn't address default to False — the plan contract
    is 'ignored rules are not credited or punished, only seen_count++'.
    """
    by_id = {a.rule_id: bool(a.applied) for a in applied}
    return {rid: by_id.get(rid, False) for rid in retrieved_ids}


def _rationale_for(applied: list[AppliedRule], rule_id: str) -> str:
    for a in applied:
        if a.rule_id == rule_id:
            return a.rationale
    return ""
