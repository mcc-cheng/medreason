"""End-to-end runner for the lead-op retro dry-run.

Orchestrates: for each seed in SEEDS, run the blind-replay harness
twice (memory OFF, memory ON), collect metrics, emit a JSON report.
Prints the +3pp / CI gate status. Does NOT render the prediction card
— that's `prediction_card.py` and is a deliberate second step so a
human gate decides whether to ship based on the gate outcome.

Supports both the deterministic heuristic agent (for offline dev) and
the Claude-backed LLM agent (for the real dry-run). Selects via
`--agent {det,llm}`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from medreason.llm.claude import ClaudeLLMClient
from medreason.runners.claude import resolve_claude_model

from .agent_llm import propose_llm
from .harness import (
    LeadOpContext,
    LeadOpDecision,
    _load_compounds,
    load_decision_points,
    propose_deterministic,
    run_blind_replay,
)
from .metrics import bootstrap_ci, cycle_waste, top1_direction_hit
from .schema import connect_campaign_db


GATE_LIFT_PP = 3.0
GATE_MIN_DPS = 4


@dataclass
class SeedRun:
    seed: int
    memory: bool
    top1_hits: int
    top1_total: int
    cycle_waste_correct: int
    cycle_waste_total: int
    decisions: list[dict]
    cost_usd: float


@dataclass
class DryRunReport:
    campaign_id: str
    n_decision_points: int
    seeds: list[int]
    agent: str
    top1_memoff_rates: list[float]
    top1_memon_rates: list[float]
    top1_lift_pp: float
    top1_lift_ci_90_lo: float
    top1_lift_ci_90_hi: float
    cw_memoff_rates: list[float]
    cw_memon_rates: list[float]
    cw_lift_pp: float
    gate_passed: bool
    gate_reason: str
    total_cost_usd: float
    runs: list[SeedRun]

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "runs"},
            "runs": [asdict(r) for r in self.runs],
        }


def _llm_propose_fn_factory(llm: ClaudeLLMClient, rag_corpus: str | None):
    # Wraps propose_llm into the `propose_fn(ctx, use_memory)` signature
    # expected by run_blind_replay, tracking cost as a side channel.
    cost_accumulator: list[float] = []

    def _fn(ctx, *, use_memory, _cost=cost_accumulator, _llm=llm, _rag=rag_corpus):
        dec, cost = propose_llm(
            ctx, use_memory=use_memory, llm=_llm, rag_corpus=_rag, seed=0
        )
        _cost.append(cost.cost_usd)
        return dec

    return _fn, cost_accumulator


def run_dry_run(
    db_path: str | Path,
    campaign_id: str,
    *,
    seeds: list[int],
    agent: str = "det",
    rag_corpus: str | None = None,
    model: str = "haiku",
) -> DryRunReport:
    con = connect_campaign_db(db_path)
    try:
        dps = load_decision_points(con, campaign_id)
        compounds = _load_compounds(con, campaign_id)
    finally:
        con.close()

    n_dps = len([dp for dp in dps if dp.team_direction_chosen])

    runs: list[SeedRun] = []
    total_cost = 0.0
    for seed in seeds:
        for memory in (False, True):
            if agent == "det":
                propose_fn: Callable = propose_deterministic
                decisions = run_blind_replay(
                    db_path, campaign_id, use_memory=memory, propose_fn=propose_fn
                )
                cost_usd = 0.0
            elif agent == "llm":
                llm = ClaudeLLMClient(model=resolve_claude_model(model))
                fn, cost_acc = _llm_propose_fn_factory(llm, rag_corpus)
                decisions = run_blind_replay(
                    db_path, campaign_id, use_memory=memory, propose_fn=fn
                )
                cost_usd = sum(cost_acc)
            else:
                raise ValueError(f"unknown agent: {agent!r}")

            top1 = top1_direction_hit(decisions, dps)
            cw = cycle_waste(decisions, dps, compounds, top_k=3)
            runs.append(
                SeedRun(
                    seed=seed,
                    memory=memory,
                    top1_hits=top1.hits,
                    top1_total=top1.total,
                    cycle_waste_correct=cw.correctly_deprioritized,
                    cycle_waste_total=cw.total_team_failures,
                    decisions=[asdict(d) for d in decisions],
                    cost_usd=cost_usd,
                )
            )
            total_cost += cost_usd

    memoff_rates = [
        r.top1_hits / r.top1_total if r.top1_total else 0.0
        for r in runs
        if not r.memory
    ]
    memon_rates = [
        r.top1_hits / r.top1_total if r.top1_total else 0.0
        for r in runs
        if r.memory
    ]
    cw_off = [
        r.cycle_waste_correct / r.cycle_waste_total if r.cycle_waste_total else 0.0
        for r in runs
        if not r.memory
    ]
    cw_on = [
        r.cycle_waste_correct / r.cycle_waste_total if r.cycle_waste_total else 0.0
        for r in runs
        if r.memory
    ]

    # Paired lift: memon_rate[i] - memoff_rate[i] per seed.
    lift_per_seed = [on - off for on, off in zip(memon_rates, memoff_rates)]
    lift_mean_pp = 100 * (statistics.mean(lift_per_seed) if lift_per_seed else 0)
    lo, hi = bootstrap_ci(lift_per_seed, iters=2000, alpha=0.10, seed=17)
    lo_pp, hi_pp = 100 * lo, 100 * hi

    cw_lift_pp = 100 * (
        statistics.mean([on - off for on, off in zip(cw_on, cw_off)])
        if cw_on and cw_off
        else 0
    )

    passed = (
        n_dps >= GATE_MIN_DPS
        and lift_mean_pp >= GATE_LIFT_PP
        and lo_pp > 0.0
    )
    if n_dps < GATE_MIN_DPS:
        reason = (
            f"Campaign has {n_dps} labeled DPs; gate requires >={GATE_MIN_DPS}. "
            "Pick another campaign or annotate more DPs."
        )
    elif lift_mean_pp < GATE_LIFT_PP:
        reason = (
            f"Mean top-1 lift {lift_mean_pp:.2f}pp < {GATE_LIFT_PP}pp threshold. "
            "Do NOT ship prediction card; investigate harness or reassess transfer."
        )
    elif lo_pp <= 0.0:
        reason = (
            f"90% CI lower bound {lo_pp:.2f}pp is not > 0. "
            "Lift is not significantly above zero. Do NOT ship."
        )
    else:
        reason = (
            f"Lift {lift_mean_pp:.2f}pp with 90% CI [{lo_pp:.2f}, {hi_pp:.2f}]pp "
            f"on {n_dps} DPs. GATE PASSED."
        )

    return DryRunReport(
        campaign_id=campaign_id,
        n_decision_points=n_dps,
        seeds=seeds,
        agent=agent,
        top1_memoff_rates=memoff_rates,
        top1_memon_rates=memon_rates,
        top1_lift_pp=lift_mean_pp,
        top1_lift_ci_90_lo=lo_pp,
        top1_lift_ci_90_hi=hi_pp,
        cw_memoff_rates=cw_off,
        cw_memon_rates=cw_on,
        cw_lift_pp=cw_lift_pp,
        gate_passed=passed,
        gate_reason=reason,
        total_cost_usd=total_cost,
        runs=runs,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lead-op retro dry-run")
    p.add_argument("--db", required=True, help="Path to campaign DuckDB file")
    p.add_argument("--campaign", required=True, help="Campaign ID")
    p.add_argument("--seeds", default="11,17,23", help="Comma-separated seed list")
    p.add_argument("--agent", choices=["det", "llm"], default="det")
    p.add_argument("--model", default="haiku")
    p.add_argument("--rag", default=None, help="Optional RAG corpus file")
    p.add_argument("--out", required=True, help="Path to write JSON report")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rag = Path(args.rag).read_text() if args.rag else None

    report = run_dry_run(
        args.db,
        args.campaign,
        seeds=seeds,
        agent=args.agent,
        rag_corpus=rag,
        model=args.model,
    )
    Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
    print(report.gate_reason)
    print(f"mean lift {report.top1_lift_pp:.2f}pp | CI [{report.top1_lift_ci_90_lo:.2f}, {report.top1_lift_ci_90_hi:.2f}]pp")
    print(f"cycle-waste lift {report.cw_lift_pp:.2f}pp")
    print(f"cost ${report.total_cost_usd:.4f}")
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
