"""SwarmRunner — one agent per target, run in parallel.

For a campaign with N candidate targets, the SwarmRunner instantiates N
SwarmAgents, each holding one TargetValidationCase. Each agent retrieves
rules from the 3-layer LayerRouter, derives bypass signals from the
PathwayCache (modality-filtered), calls the LLM once, and parses the
response into a structured TargetMemo.

Why one agent per target (and not one big agent that ranks all of them)?

1. **Reasoning isolation**: a hard target's reasoning can't contaminate
   the easy targets' rationale. Each memo is independently auditable.
2. **Parallel cost**: N agents in parallel is wall-clock O(1) instead of
   O(N) for a sequential meta-agent.
3. **Cross-agent metacog needs independent observations**: the cross-
   agent analyzer (cross_agent_analyzer.py) measures *systematic*
   reasoning failures across many independent agents. A single agent
   can't produce that signal.

Determinism: the parallel path preserves the input order of cases in
the output ``memos`` list (by waiting on futures in submission order
rather than ``as_completed``). This is the contract the
``test_swarm_parallel_path_same_results_as_serial`` test asserts, and
the cross-agent analyzer relies on the order being case-stable across
runs.

Phase 2 v0.2 changes vs the skeleton:
- ``SwarmAgent.run`` actually parses LLM JSON into a structured memo
  via ``swarm_parsing.safe_parse_memo`` (tolerant: malformed JSON
  produces a fallback memo with ``parse_warnings``, never raises).
- New ``TargetMemo`` fields: ``bypass_signals_seen`` (which mechanisms
  the agent saw), ``parse_warnings`` (non-fatal LLM-format issues).
- New optional ``pathway_cache`` kwarg on both ``SwarmAgent`` and
  ``SwarmRunner``. An empty cache is the safe default — callers that
  don't pass one get an empty bypass-signal list, which is fine for
  the orchestration-only tests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base import LLMClient
from .case import BypassMechanism, TargetValidationCase
from .layers import LayerRouter
from .swarm_parsing import safe_parse_memo
from .swarm_prompts import SYSTEM_PROMPT_TARGETVAL, build_user_prompt
from .topology import PathwayCache, derive_bypass_signals


# ── Per-target output ─────────────────────────────────────────────────────────


@dataclass
class TargetMemo:
    """One agent's decision memo for one target.

    Schema lock note (v0.2): the field set below is the contract
    ``proposer-builder`` reads. Don't add or rename fields without
    coordinating with the cross-agent analyzer.
    """

    case_id: str
    gene_symbol: str

    # The agent's recommendation
    priority_score: float  # [0, 1] — higher = stronger candidate
    bypass_risk_score: float  # [0, 1] — higher = more likely to be bypassed
    predicted_bypass: BypassMechanism = BypassMechanism.UNKNOWN

    # Reasoning surface
    supporting_evidence: list[str] = field(default_factory=list)
    weakening_evidence: list[str] = field(default_factory=list)
    proposed_experiments: list[str] = field(default_factory=list)
    rationale: str = ""

    # Provenance
    retrieved_rule_ids: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)

    # NEW for v0.2 (read by cross_agent_analyzer / proposer-builder)
    bypass_signals_seen: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    # Cost / determinism
    cost_usd: float = 0.0
    seed: int = 0


@dataclass
class SwarmReport:
    """Aggregate output of one swarm run across N targets."""

    campaign_id: str
    memos: list[TargetMemo]
    ranking: list[str] = field(default_factory=list)  # gene_symbols, best → worst
    total_cost_usd: float = 0.0

    @property
    def n_targets(self) -> int:
        return len(self.memos)


# ── The per-target agent ──────────────────────────────────────────────────────


class SwarmAgent:
    """Decides one TargetValidationCase.

    Flow per ``run()``:
    1. Derive bypass signals from the pathway cache (modality-filtered).
    2. Retrieve advisory rules from the 3-layer LayerRouter.
    3. Build system + user prompts.
    4. Single LLM call.
    5. Tolerant JSON parse into a TargetMemo (no raise — fallback memo
       on unparseable output, with ``parse_warnings`` populated).
    """

    def __init__(
        self,
        case: TargetValidationCase,
        llm: LLMClient,
        layer_router: LayerRouter,
        *,
        customer_tag: Optional[str] = None,
        seed: int = 0,
        pathway_cache: Optional[PathwayCache] = None,
    ):
        self.case = case
        self.llm = llm
        self.router = layer_router
        self.customer_tag = customer_tag
        self.seed = seed
        # Default to an empty cache so callers that don't pass one still
        # work — bypass_signals will be [] and the prompt will say so.
        self._pathway_cache = (
            pathway_cache if pathway_cache is not None else PathwayCache()
        )

    def run(self) -> TargetMemo:
        # 1. Bypass signals (modality-filtered)
        bypass_signals = derive_bypass_signals(
            self.case.target.gene_symbol,
            self._pathway_cache,
            modality=self.case.modality,
        )

        # 2. Retrieved rules (advisory)
        retrieved = self.router.retrieve_for_case(
            case_disease_scope=self.case.disease.disease_label,
            customer_tag=self.customer_tag,
            top_k=5,
        )
        retrieved_ids = [r.rule_id for r in retrieved]

        # 3-4. Prompt + LLM call
        system = SYSTEM_PROMPT_TARGETVAL
        user = build_user_prompt(
            self.case,
            retrieved_rules=retrieved,
            bypass_signals=bypass_signals,
        )
        response = self.llm.complete(system=system, user=user, seed=self.seed)

        # 5. Tolerant parse — never raises; produces fallback memo on
        # total parse failure with parse_warnings populated.
        return safe_parse_memo(
            response.text,
            case_id=self.case.case_id,
            gene_symbol=self.case.target.gene_symbol,
            retrieved_rule_ids=retrieved_ids,
            bypass_signals=bypass_signals,
            cost_usd=response.cost_usd,
            seed=self.seed,
        )


# ── The swarm orchestrator ────────────────────────────────────────────────────


class SwarmRunner:
    """Runs SwarmAgents in parallel across a list of TargetValidationCases.

    The parallel path uses ``ThreadPoolExecutor`` with **ordered**
    future collection so that ``memos`` preserves the input order of
    ``cases``. This makes parallel runs byte-identical to serial runs
    for deterministic LLM clients (e.g. ``FakeLLMClient`` with the
    same ``default_text``), which the test suite relies on.
    """

    def __init__(
        self,
        llm: LLMClient,
        layer_router: LayerRouter,
        *,
        customer_tag: Optional[str] = None,
        max_workers: int = 8,
        pathway_cache: Optional[PathwayCache] = None,
    ):
        self.llm = llm
        self.router = layer_router
        self.customer_tag = customer_tag
        self.max_workers = max_workers
        # Resolve to an empty cache once so every agent sees the same
        # snapshot (and so each agent doesn't allocate its own).
        self.pathway_cache = (
            pathway_cache if pathway_cache is not None else PathwayCache()
        )

    def _make_agent(self, case: TargetValidationCase, seed: int) -> SwarmAgent:
        return SwarmAgent(
            case,
            self.llm,
            self.router,
            customer_tag=self.customer_tag,
            seed=seed,
            pathway_cache=self.pathway_cache,
        )

    def run(
        self,
        cases: list[TargetValidationCase],
        *,
        campaign_id: str,
        seed: int = 0,
    ) -> SwarmReport:
        if not cases:
            return SwarmReport(campaign_id=campaign_id, memos=[], ranking=[])

        def _run_one(case: TargetValidationCase) -> TargetMemo:
            return self._make_agent(case, seed).run()

        memos: list[TargetMemo]
        if self.max_workers <= 1 or len(cases) == 1:
            memos = [_run_one(c) for c in cases]
        else:
            # Submit in input order, then collect future results in the
            # SAME order. This is the deterministic alternative to
            # ``as_completed``: we still get parallelism, but the output
            # list stays case-aligned.
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(_run_one, c) for c in cases]
                memos = [f.result() for f in futures]

        total_cost = sum(m.cost_usd for m in memos)
        ranking = aggregate_ranking(memos)

        return SwarmReport(
            campaign_id=campaign_id,
            memos=memos,
            ranking=ranking,
            total_cost_usd=total_cost,
        )


def aggregate_ranking(memos: list[TargetMemo]) -> list[str]:
    """Default aggregator: sort by (priority_score - 0.5 * bypass_risk_score).

    This is a placeholder. The real aggregator will be replaceable so
    customers can plug in their own scoring (e.g., risk-weighted by
    indication unmet need). The current formula penalises bypass-risk
    half as much as it credits priority — explicit so it's tunable.
    """
    return [
        m.gene_symbol
        for m in sorted(
            memos,
            key=lambda mm: mm.priority_score - 0.5 * mm.bypass_risk_score,
            reverse=True,
        )
    ]
