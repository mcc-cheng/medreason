"""SwarmRunner — one agent per target, run in parallel.

For a campaign with N candidate targets, the SwarmRunner instantiates N
SwarmAgents, each holding one TargetValidationCase. Each agent retrieves
rules from the 3-layer LayerRouter, builds a per-target memo, and emits
a TargetMemo with structured fields the aggregator can rank.

Why one agent per target (and not one big agent that ranks all of them)?

1. **Reasoning isolation**: a hard target's reasoning can't contaminate
   the easy targets' rationale. Each memo is independently auditable.
2. **Parallel cost**: N agents in parallel is wall-clock O(1) instead of
   O(N) for a sequential meta-agent.
3. **Cross-agent metacog needs independent observations**: the cross-
   agent analyzer (cross_agent_analyzer.py) measures *systematic*
   reasoning failures across many independent agents. A single agent
   can't produce that signal.

Skeleton scope: SwarmAgent + SwarmRunner + TargetMemo + SwarmReport types,
with a synchronous reference run() that uses a FakeLLMClient. Real LLM
wiring + parallel execution lands once the architecture is reviewed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base import LLMClient
from .case import BypassMechanism, TargetValidationCase
from .layers import LayerRouter


# ── Per-target output ─────────────────────────────────────────────────────────


@dataclass
class TargetMemo:
    """One agent's decision memo for one target."""

    case_id: str
    gene_symbol: str

    # The agent's recommendation
    priority_score: float  # 0..1, higher = stronger candidate
    bypass_risk_score: float  # 0..1, higher = more likely to be bypassed
    predicted_bypass: BypassMechanism = BypassMechanism.UNKNOWN

    # Reasoning surface
    supporting_evidence: list[str] = field(default_factory=list)
    weakening_evidence: list[str] = field(default_factory=list)
    proposed_experiments: list[str] = field(default_factory=list)
    rationale: str = ""

    # Provenance
    retrieved_rule_ids: list[str] = field(default_factory=list)
    applied_rule_ids: list[str] = field(default_factory=list)
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

    Skeleton: the run() method composes prompt-building and a single LLM
    call but does NOT yet implement the full memo extraction. The output
    is a placeholder TargetMemo with priority_score=0.5 — enough to wire
    the aggregator and the cross-agent analyzer end-to-end.

    Real implementation will:
    1. Pull rules from LayerRouter.retrieve_for_case(...)
    2. Build a system + user prompt that includes the evidence bundle
       and the retrieved rules (using a tweaked rule_checklist injector)
    3. Call llm.complete(...) and parse a strict JSON schema
    4. Update posteriors on retrieved rules (success/failure tracked via
       a cross-agent feedback signal, not per-case ground truth — see
       cross_agent_analyzer.py)
    """

    def __init__(
        self,
        case: TargetValidationCase,
        llm: LLMClient,
        layer_router: LayerRouter,
        *,
        customer_tag: Optional[str] = None,
        seed: int = 0,
    ):
        self.case = case
        self.llm = llm
        self.router = layer_router
        self.customer_tag = customer_tag
        self.seed = seed

    def run(self) -> TargetMemo:
        retrieved = self.router.retrieve_for_case(
            case_disease_scope=self.case.disease.disease_label,
            customer_tag=self.customer_tag,
            top_k=5,
        )
        retrieved_ids = [r.rule_id for r in retrieved]

        # PROMPT BUILDING — minimal placeholder, full builder lands later.
        system = (
            "You are a target-validation reviewer. Given evidence about a "
            "(target, disease) pair, return a priority score (0..1), a "
            "bypass-risk score (0..1), supporting evidence, weakening "
            "evidence, and proposed experiments. Strict JSON only."
        )
        user = (
            f"target={self.case.target.gene_symbol} "
            f"disease={self.case.disease.disease_label}\n"
            f"evidence={self.case.evidence.model_dump_json(indent=None)}"
        )
        response = self.llm.complete(system=system, user=user, seed=self.seed)

        # PARSE — full parser lands later. For now produce a fixed-shape memo.
        return TargetMemo(
            case_id=self.case.case_id,
            gene_symbol=self.case.target.gene_symbol,
            priority_score=0.5,
            bypass_risk_score=0.5,
            predicted_bypass=BypassMechanism.UNKNOWN,
            supporting_evidence=[],
            weakening_evidence=[],
            proposed_experiments=[],
            rationale=response.text or "",
            retrieved_rule_ids=retrieved_ids,
            applied_rule_ids=[],
            cost_usd=response.cost_usd,
            seed=self.seed,
        )


# ── The swarm orchestrator ────────────────────────────────────────────────────


class SwarmRunner:
    """Runs SwarmAgents in parallel across a list of TargetValidationCases."""

    def __init__(
        self,
        llm: LLMClient,
        layer_router: LayerRouter,
        *,
        customer_tag: Optional[str] = None,
        max_workers: int = 8,
    ):
        self.llm = llm
        self.router = layer_router
        self.customer_tag = customer_tag
        self.max_workers = max_workers

    def run(
        self,
        cases: list[TargetValidationCase],
        *,
        campaign_id: str,
        seed: int = 0,
    ) -> SwarmReport:
        memos: list[TargetMemo] = []
        total_cost = 0.0

        if not cases:
            return SwarmReport(campaign_id=campaign_id, memos=[], ranking=[])

        def _run_one(case: TargetValidationCase) -> TargetMemo:
            agent = SwarmAgent(
                case,
                self.llm,
                self.router,
                customer_tag=self.customer_tag,
                seed=seed,
            )
            return agent.run()

        if self.max_workers <= 1 or len(cases) == 1:
            for c in cases:
                memos.append(_run_one(c))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(_run_one, c): c for c in cases}
                for fut in as_completed(futures):
                    memos.append(fut.result())

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
