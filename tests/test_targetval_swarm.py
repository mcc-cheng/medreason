"""Tests for SwarmRunner end-to-end with a FakeLLMClient.

The skeleton swarm uses a fixed-shape memo, so these tests assert the
*orchestration* is correct (every case gets a memo, ranking is total,
LLM gets called once per case) without asserting the content of the memo
(which depends on the parser, which is not yet implemented).
"""

from __future__ import annotations

from pathlib import Path

from medreason.llm.base import FakeLLMClient
from medreason.targetval.layers import LayerRouter, LayerStorePaths
from medreason.targetval.swarm import (
    SwarmAgent,
    SwarmRunner,
    aggregate_ranking,
    TargetMemo,
)

from medreason_bench.targetval.synthetic import (
    SYNTHETIC_CAMPAIGN_ID,
    build_synthetic_targets,
)


def _empty_router() -> LayerRouter:
    return LayerRouter(
        LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})
    )


def test_swarm_runs_one_agent_per_case():
    cases = build_synthetic_targets()
    llm = FakeLLMClient(default_text='{"placeholder": true}')
    runner = SwarmRunner(llm, _empty_router(), max_workers=1)
    report = runner.run(cases, campaign_id=SYNTHETIC_CAMPAIGN_ID, seed=11)
    assert report.n_targets == len(cases)
    assert len(report.memos) == len(cases)
    assert len(llm.calls) == len(cases)


def test_swarm_aggregate_ranking_total_and_ordered():
    cases = build_synthetic_targets()
    llm = FakeLLMClient(default_text="{}")
    runner = SwarmRunner(llm, _empty_router(), max_workers=1)
    report = runner.run(cases, campaign_id=SYNTHETIC_CAMPAIGN_ID, seed=11)
    # Every gene should appear exactly once in the ranking
    gene_set = {c.target.gene_symbol for c in cases}
    assert set(report.ranking) == gene_set
    assert len(report.ranking) == len(gene_set)


def test_swarm_handles_empty_case_list():
    llm = FakeLLMClient(default_text="{}")
    runner = SwarmRunner(llm, _empty_router(), max_workers=1)
    report = runner.run([], campaign_id="empty", seed=11)
    assert report.n_targets == 0
    assert report.ranking == []
    assert report.total_cost_usd == 0.0


def test_swarm_parallel_path_same_results_as_serial():
    cases = build_synthetic_targets()
    llm_serial = FakeLLMClient(default_text="{}")
    llm_parallel = FakeLLMClient(default_text="{}")
    serial = SwarmRunner(llm_serial, _empty_router(), max_workers=1)
    parallel = SwarmRunner(llm_parallel, _empty_router(), max_workers=4)
    a = serial.run(cases, campaign_id="x", seed=11)
    b = parallel.run(cases, campaign_id="x", seed=11)
    assert {m.case_id for m in a.memos} == {m.case_id for m in b.memos}
    assert set(a.ranking) == set(b.ranking)


def test_aggregate_ranking_prefers_high_priority_low_bypass():
    memos = [
        TargetMemo(
            case_id="c1", gene_symbol="HIGH",
            priority_score=0.9, bypass_risk_score=0.1,
        ),
        TargetMemo(
            case_id="c2", gene_symbol="LOW",
            priority_score=0.4, bypass_risk_score=0.7,
        ),
        TargetMemo(
            case_id="c3", gene_symbol="MID",
            priority_score=0.7, bypass_risk_score=0.3,
        ),
    ]
    ranking = aggregate_ranking(memos)
    assert ranking[0] == "HIGH"
    assert ranking[-1] == "LOW"
