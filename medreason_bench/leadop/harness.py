"""Blind re-run harness for the lead-op retrospective.

Given a campaign DB populated by `synthetic.write_toy_campaign` (or the
real ChEMBL ingest), the harness iterates decision points in order and,
at each DP, presents the agent with a temporally-gated context:

- Visible compounds: those in rounds < DP.round_index.
- Visible compound outcomes: those in rounds < DP.round_index (the team
  learns the prior round's outcomes by the time they plan the next one).
- Visible prior-DP rationales: DPs with list-index < current, annotated
  with a success/failure outcome derived from the compounds proposed
  after that DP.
- Candidate pool: the compounds with round == DP.round_index, stripped
  of outcome labels. Agent must produce a compound ranking over this
  pool for the cycle-waste metric.

The agent output is a `LeadOpDecision` = (direction_ranking,
compound_ranking, rationale). The deterministic heuristic agent
(`propose_deterministic`) is for smoke tests; LLM wiring lives in
`agent_llm.py` and is called by the real runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb

from .schema import (
    CompoundRow,
    DecisionPointRow,
    connect_campaign_db,
)


Direction = Literal["potency", "selectivity", "ADMET", "scaffold_hop"]
ALL_DIRECTIONS: tuple[Direction, ...] = (
    "potency",
    "selectivity",
    "ADMET",
    "scaffold_hop",
)


@dataclass
class DpRationale:
    decision_point_id: str
    round_index: int
    direction_chosen: Direction
    outcome: str  # "success" | "failure" | "unknown"
    note: str | None = None


@dataclass
class LeadOpContext:
    campaign_id: str
    decision_point_id: str
    round_index: int
    visible_compounds: list[CompoundRow]
    candidate_pool: list[CompoundRow]  # this round's compounds, outcome-stripped
    visible_rationales: list[DpRationale]
    memory_rules: list[str] = field(default_factory=list)


@dataclass
class LeadOpDecision:
    decision_point_id: str
    direction_ranking: list[Direction]
    compound_ranking: list[str]  # compound_ids, best -> worst
    rationale: str


def _rehydrate_compound(row: tuple) -> CompoundRow:
    (
        compound_id,
        campaign_id,
        timestamp,
        round_index,
        smiles,
        scaffold_key,
        mw,
        clogp,
        tpsa,
        hbd,
        hba,
        proposed_modification,
        agent_rationale,
        assay_json,
        decision_point_id,
        outcome_label,
    ) = row
    return CompoundRow(
        compound_id=compound_id,
        campaign_id=campaign_id,
        timestamp=str(timestamp),
        round_index=round_index,
        smiles=smiles,
        scaffold_key=scaffold_key,
        mw=mw,
        clogp=clogp,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        proposed_modification=proposed_modification,
        agent_rationale=agent_rationale,
        assay_readouts=json.loads(assay_json) if assay_json else {},
        decision_point_id=decision_point_id,
        outcome_label=outcome_label,
    )


def _rehydrate_dp(row: tuple) -> DecisionPointRow:
    (
        decision_point_id,
        campaign_id,
        timestamp,
        round_index,
        annotation_source,
        notes,
        team_direction_chosen,
    ) = row
    return DecisionPointRow(
        decision_point_id=decision_point_id,
        campaign_id=campaign_id,
        timestamp=str(timestamp),
        round_index=round_index,
        annotation_source=annotation_source,
        notes=notes,
        team_direction_chosen=team_direction_chosen,
    )


def load_decision_points(
    con: duckdb.DuckDBPyConnection, campaign_id: str
) -> list[DecisionPointRow]:
    rows = con.execute(
        "SELECT * FROM decision_points WHERE campaign_id=? ORDER BY round_index",
        [campaign_id],
    ).fetchall()
    return [_rehydrate_dp(r) for r in rows]


def _load_compounds(
    con: duckdb.DuckDBPyConnection, campaign_id: str
) -> list[CompoundRow]:
    rows = con.execute(
        "SELECT * FROM compounds WHERE campaign_id=? ORDER BY round_index, compound_id",
        [campaign_id],
    ).fetchall()
    return [_rehydrate_compound(r) for r in rows]


def _strip_outcome(c: CompoundRow) -> CompoundRow:
    """Return a copy with outcome_label blanked — used for candidate pools."""
    return CompoundRow(
        compound_id=c.compound_id,
        campaign_id=c.campaign_id,
        timestamp=c.timestamp,
        round_index=c.round_index,
        smiles=c.smiles,
        scaffold_key=c.scaffold_key,
        mw=c.mw,
        clogp=c.clogp,
        tpsa=c.tpsa,
        hbd=c.hbd,
        hba=c.hba,
        proposed_modification=c.proposed_modification,
        agent_rationale=c.agent_rationale,
        assay_readouts=dict(c.assay_readouts),
        decision_point_id=c.decision_point_id,
        outcome_label=None,
    )


def build_contexts(
    con: duckdb.DuckDBPyConnection,
    campaign_id: str,
) -> list[LeadOpContext]:
    """Precompute per-DP temporal contexts. Rationales are filled in during
    blind replay (they depend on the agent's own prior outputs) — here we
    populate only the team-outcome-derived visible_rationales shell, which
    the replay loop will override with agent-written rationales."""
    dps = load_decision_points(con, campaign_id)
    all_compounds = _load_compounds(con, campaign_id)

    contexts: list[LeadOpContext] = []
    for i, dp in enumerate(dps):
        visible = [c for c in all_compounds if c.round_index < dp.round_index]
        candidate_pool = [
            _strip_outcome(c) for c in all_compounds if c.round_index == dp.round_index
        ]
        visible_rationales: list[DpRationale] = []
        for prior in dps[:i]:
            visible_rationales.append(
                DpRationale(
                    decision_point_id=prior.decision_point_id,
                    round_index=prior.round_index,
                    direction_chosen=prior.team_direction_chosen or "potency",
                    outcome=_dp_outcome(prior, all_compounds),
                    note=prior.notes,
                )
            )
        contexts.append(
            LeadOpContext(
                campaign_id=campaign_id,
                decision_point_id=dp.decision_point_id,
                round_index=dp.round_index,
                visible_compounds=visible,
                candidate_pool=candidate_pool,
                visible_rationales=visible_rationales,
            )
        )
    return contexts


def _dp_outcome(dp: DecisionPointRow, all_compounds: list[CompoundRow]) -> str:
    """A prior DP's outcome = 'success' if any compound in that DP's round
    was labeled 'advanced', 'failure' if all were labeled failed_*, else
    'unknown'. This derives rationale-level outcomes from compound outcomes."""
    in_round = [c for c in all_compounds if c.round_index == dp.round_index]
    if not in_round:
        return "unknown"
    labels = [c.outcome_label for c in in_round]
    if any(lbl == "advanced" for lbl in labels):
        return "success"
    if all(lbl is not None and lbl.startswith("failed") for lbl in labels):
        return "failure"
    return "unknown"


# ── Deterministic heuristic agent (for smoke tests) ──────────────────────


def propose_deterministic(
    ctx: LeadOpContext,
    *,
    use_memory: bool = False,
) -> LeadOpDecision:
    """Simple rule-based agent. Without memory: prioritize potency always.
    With memory: look at visible_rationales — if last rationale failed and
    direction was potency, pivot to ADMET; otherwise same as baseline.

    Not a real model — this exists so smoke tests can drive the harness
    end-to-end without calling an LLM."""

    if use_memory and ctx.visible_rationales:
        last = ctx.visible_rationales[-1]
        if last.outcome == "failure" and last.direction_chosen == "potency":
            ranking: list[Direction] = ["ADMET", "selectivity", "scaffold_hop", "potency"]
            reason = (
                f"prior DP {last.decision_point_id} chose potency and failed; "
                "pivoting to ADMET per memory trace"
            )
        else:
            ranking = ["potency", "selectivity", "ADMET", "scaffold_hop"]
            reason = "baseline potency-first"
    else:
        ranking = ["potency", "selectivity", "ADMET", "scaffold_hop"]
        reason = "baseline potency-first"

    # Compound ranking: sort candidate pool by ic50_nm ascending (lower = better).
    # Compounds without ic50 go to the end.
    def _score(c: CompoundRow) -> float:
        ic = c.assay_readouts.get("ic50_nm")
        return float(ic) if ic is not None else float("inf")

    ranked = sorted(ctx.candidate_pool, key=_score)
    compound_ranking = [c.compound_id for c in ranked]

    return LeadOpDecision(
        decision_point_id=ctx.decision_point_id,
        direction_ranking=ranking,
        compound_ranking=compound_ranking,
        rationale=reason,
    )


def run_blind_replay(
    db_path: str | Path,
    campaign_id: str,
    *,
    use_memory: bool,
    propose_fn=propose_deterministic,
) -> list[LeadOpDecision]:
    """Iterate DPs in order, build temporal context, call agent, record
    decision. Each prior DP's visible rationale carries the agent's own
    reasoning text but the TEAM's actual round outcome (retrospective
    observation — the team's outcome is the ground truth the agent sees
    unfold, regardless of which direction the agent would have chosen)."""
    con = connect_campaign_db(db_path)
    try:
        contexts = build_contexts(con, campaign_id)
        dps = load_decision_points(con, campaign_id)
        all_compounds = _load_compounds(con, campaign_id)
    finally:
        con.close()

    decisions: list[LeadOpDecision] = []
    agent_rationales: list[DpRationale] = []
    for ctx in contexts:
        if agent_rationales:
            ctx.visible_rationales = agent_rationales[: len(ctx.visible_rationales)]
        dec = propose_fn(ctx, use_memory=use_memory)
        decisions.append(dec)
        matching_dp = next(
            dp for dp in dps if dp.decision_point_id == ctx.decision_point_id
        )
        team_outcome = _dp_outcome(matching_dp, all_compounds)
        agent_rationales.append(
            DpRationale(
                decision_point_id=ctx.decision_point_id,
                round_index=ctx.round_index,
                direction_chosen=dec.direction_ranking[0],
                outcome=team_outcome,
                note=dec.rationale,
            )
        )
    return decisions
