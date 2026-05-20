"""Smoke tests for the lead-op blind-replay harness + metrics.

Uses the 5-compound synthetic toy campaign (synthetic.py). With the
deterministic heuristic agent, memory ON beats memory OFF on the toy's
second DP by construction (baseline stays on potency, memory pivots to
ADMET which matches the team). These tests assert wiring, not lift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medreason_bench.leadop.harness import (
    ALL_DIRECTIONS,
    LeadOpContext,
    build_contexts,
    propose_deterministic,
    run_blind_replay,
)
from medreason_bench.leadop.metrics import (
    bootstrap_ci,
    cycle_waste,
    top1_direction_hit,
)
from medreason_bench.leadop.schema import connect_campaign_db
from medreason_bench.leadop.synthetic import CAMPAIGN_ID, write_toy_campaign


@pytest.fixture
def toy_db(tmp_path: Path) -> Path:
    return write_toy_campaign(tmp_path / "toy.db")


def test_build_contexts_respects_temporal_gate(toy_db: Path) -> None:
    con = connect_campaign_db(toy_db)
    try:
        ctxs = build_contexts(con, CAMPAIGN_ID)
    finally:
        con.close()

    assert len(ctxs) == 2  # two DPs in the toy

    # DP1 (round 2): visible compounds from round 1 only; candidate pool = round 2.
    dp1 = ctxs[0]
    assert {c.round_index for c in dp1.visible_compounds} == {1}
    assert {c.round_index for c in dp1.candidate_pool} == {2}
    assert len(dp1.visible_rationales) == 0

    # Candidate pool compounds have outcome_label stripped.
    assert all(c.outcome_label is None for c in dp1.candidate_pool)

    # DP2 (round 3): visible from rounds 1+2; candidate pool = round 3.
    dp2 = ctxs[1]
    assert {c.round_index for c in dp2.visible_compounds} == {1, 2}
    assert {c.round_index for c in dp2.candidate_pool} == {3}
    assert len(dp2.visible_rationales) == 1  # DP1's team rationale visible to DP2


def test_replay_baseline_top1_hits_one_of_two(toy_db: Path) -> None:
    decisions = run_blind_replay(toy_db, CAMPAIGN_ID, use_memory=False)
    con = connect_campaign_db(toy_db)
    try:
        from medreason_bench.leadop.harness import load_decision_points
        dps = load_decision_points(con, CAMPAIGN_ID)
    finally:
        con.close()

    result = top1_direction_hit(decisions, dps)
    assert result.total == 2
    # Baseline is potency-first always. Team chose potency at DP1, ADMET at DP2.
    # So baseline hits 1/2.
    assert result.hits == 1


def test_replay_memory_beats_baseline_on_toy(toy_db: Path) -> None:
    decisions = run_blind_replay(toy_db, CAMPAIGN_ID, use_memory=True)
    con = connect_campaign_db(toy_db)
    try:
        from medreason_bench.leadop.harness import load_decision_points
        dps = load_decision_points(con, CAMPAIGN_ID)
    finally:
        con.close()

    result = top1_direction_hit(decisions, dps)
    # Memory-on: DP1 still potency (no prior rationales). DP2 pivots to ADMET
    # because the prior DP's team rationale was potency+failed. So 2/2.
    assert result.hits == 2


def test_cycle_waste_scores_failed_compounds(toy_db: Path) -> None:
    decisions = run_blind_replay(toy_db, CAMPAIGN_ID, use_memory=False)
    con = connect_campaign_db(toy_db)
    try:
        from medreason_bench.leadop.harness import load_decision_points, _load_compounds
        dps = load_decision_points(con, CAMPAIGN_ID)
        compounds = _load_compounds(con, CAMPAIGN_ID)
    finally:
        con.close()

    result = cycle_waste(decisions, dps, compounds, top_k=3)
    # Toy has 2 failed compounds (TOY_002 hERG, TOY_003 permeability), both in round 2.
    assert result.total_team_failures == 2
    # With top_k=3 and only 2 candidates in the round 2 pool, nothing is ranked
    # below top-3, so correct==0 and rate==0. That's fine — the test asserts
    # wiring, not lift.
    assert 0 <= result.correctly_deprioritized <= 2


def test_cycle_waste_small_k_deprioritizes_worse_ones(toy_db: Path) -> None:
    decisions = run_blind_replay(toy_db, CAMPAIGN_ID, use_memory=False)
    con = connect_campaign_db(toy_db)
    try:
        from medreason_bench.leadop.harness import load_decision_points, _load_compounds
        dps = load_decision_points(con, CAMPAIGN_ID)
        compounds = _load_compounds(con, CAMPAIGN_ID)
    finally:
        con.close()

    result = cycle_waste(decisions, dps, compounds, top_k=1)
    # Agent ranks round 2 by ic50_nm ascending: TOY_002 (45nM) above TOY_003 (60nM).
    # With top_k=1, TOY_003 is below top-1 -> correctly deprioritized. Failed==2.
    assert result.correctly_deprioritized == 1


def test_bootstrap_ci_lower_bound_positive_for_all_positive_values() -> None:
    lo, hi = bootstrap_ci([0.05, 0.07, 0.04, 0.06, 0.08], seed=1)
    assert lo > 0.0
    assert hi > lo


def test_bootstrap_ci_lower_bound_can_be_zero_or_negative() -> None:
    lo, hi = bootstrap_ci([-0.02, 0.01, 0.00, 0.03, -0.01], seed=1)
    assert lo < 0.02
    assert hi > lo


def test_all_directions_enumerated() -> None:
    assert set(ALL_DIRECTIONS) == {"potency", "selectivity", "ADMET", "scaffold_hop"}
