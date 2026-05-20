"""Tests for SwarmRunner end-to-end with a FakeLLMClient.

Two sets of tests:

1. **Orchestration** (existing): every case gets a memo, ranking is total,
   the parallel path produces case-aligned results identical to the
   serial path. These were the v0.1 skeleton tests; they still pass
   with the v0.2 LLM-driven memo because they only assert structure.

2. **Parsing** (v0.2): the strict-JSON parser handles fenced blocks,
   clamps out-of-range scores, warns on unknown enum values, and
   raises ``MemoParseError`` only when no JSON object can be recovered.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from medreason.llm.base import FakeLLMClient
from medreason.targetval.case import BypassMechanism
from medreason.targetval.layers import LayerRouter, LayerStorePaths
from medreason.targetval.swarm import (
    SwarmAgent,
    SwarmRunner,
    aggregate_ranking,
    TargetMemo,
)
from medreason.targetval.swarm_parsing import (
    MemoParseError,
    extract_memo_json,
    parse_memo,
)
from medreason.targetval.topology import BypassSignal, PathwayCache

from medreason_bench.targetval.synthetic import (
    SYNTHETIC_CAMPAIGN_ID,
    build_synthetic_targets,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _empty_router() -> LayerRouter:
    return LayerRouter(
        LayerStorePaths(universal=Path("/dev/null"), disease={}, campaign={})
    )


# ── Existing orchestration tests (v0.1) ─────────────────────────────────────


def test_swarm_runs_one_agent_per_case():
    cases = build_synthetic_targets()
    llm = FakeLLMClient(default_text='{"priority_score": 0.5}')
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
    """The parallel path must collect futures in submission order, not
    via ``as_completed`` — otherwise memo ordering depends on thread
    scheduling and the cross-agent analyzer sees nondeterministic input.
    """
    cases = build_synthetic_targets()
    llm_serial = FakeLLMClient(default_text="{}")
    llm_parallel = FakeLLMClient(default_text="{}")
    serial = SwarmRunner(llm_serial, _empty_router(), max_workers=1)
    parallel = SwarmRunner(llm_parallel, _empty_router(), max_workers=4)
    a = serial.run(cases, campaign_id="x", seed=11)
    b = parallel.run(cases, campaign_id="x", seed=11)

    # Stronger than set-equality: case_ids must be IN ORDER.
    a_ids = [m.case_id for m in a.memos]
    b_ids = [m.case_id for m in b.memos]
    assert a_ids == b_ids
    # Same input → same ranking (deterministic FakeLLM, ordered futures)
    assert a.ranking == b.ranking


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


# ── New v0.2 parsing tests ──────────────────────────────────────────────────


_HAPPY_JSON = """\
{
  "priority_score": 0.82,
  "bypass_risk_score": 0.21,
  "predicted_bypass": "paralog_compensation",
  "supporting_evidence": ["BRAF V600E mutation", "vemurafenib precedent"],
  "weakening_evidence": ["paralog ARAF could compensate"],
  "proposed_experiments": ["paralog co-knockout"],
  "rationale": "Strong genetics + dependency. Paralog risk noted.",
  "applied_rule_ids": ["rule_abc", "rule_def"]
}
"""


def test_parse_memo_happy_path():
    memo = parse_memo(
        _HAPPY_JSON,
        case_id="TVS-001",
        gene_symbol="BRAF",
        retrieved_rule_ids=["rule_abc", "rule_def", "rule_ghi"],
        bypass_signals=[
            BypassSignal(
                mechanism="paralog_compensation",
                strength=0.45,
                evidence_summary="2 paralogs",
                contributing_features=("paralog_count=2",),
            )
        ],
        cost_usd=0.0007,
        seed=11,
    )
    assert memo.case_id == "TVS-001"
    assert memo.gene_symbol == "BRAF"
    assert memo.priority_score == pytest.approx(0.82)
    assert memo.bypass_risk_score == pytest.approx(0.21)
    assert memo.predicted_bypass is BypassMechanism.PARALOG_COMPENSATION
    assert "vemurafenib precedent" in memo.supporting_evidence
    assert memo.applied_rule_ids == ["rule_abc", "rule_def"]
    # bypass_signals_seen is the mechanism strings the agent was shown
    assert memo.bypass_signals_seen == ["paralog_compensation"]
    assert memo.parse_warnings == []
    assert memo.cost_usd == pytest.approx(0.0007)
    assert memo.seed == 11
    # retrieved_rule_ids passes through; applied_rule_ids is what LLM cited
    assert memo.retrieved_rule_ids == ["rule_abc", "rule_def", "rule_ghi"]


def test_parse_memo_fenced_json():
    fenced = "```json\n" + _HAPPY_JSON + "\n```"
    memo = parse_memo(
        fenced,
        case_id="TVS-001",
        gene_symbol="BRAF",
        retrieved_rule_ids=[],
        bypass_signals=[],
        cost_usd=0.0,
        seed=0,
    )
    assert memo.priority_score == pytest.approx(0.82)
    assert memo.predicted_bypass is BypassMechanism.PARALOG_COMPENSATION
    # Fence parse path should not introduce spurious warnings
    assert memo.parse_warnings == []


def test_parse_memo_extracts_json_with_surrounding_prose():
    """Stray prose around the JSON object should not break extraction."""
    text = (
        "Here is the memo you asked for:\n"
        + _HAPPY_JSON
        + "\nLet me know if you want me to revise."
    )
    obj = extract_memo_json(text)
    assert obj["priority_score"] == 0.82


def test_parse_memo_clamps_scores():
    """Out-of-range scores clamp to [0,1] and emit a warning each."""
    raw = (
        '{"priority_score": 1.5, "bypass_risk_score": -0.3, '
        '"predicted_bypass": "no_bypass_known"}'
    )
    memo = parse_memo(
        raw,
        case_id="x",
        gene_symbol="X",
        retrieved_rule_ids=[],
        bypass_signals=[],
        cost_usd=0.0,
        seed=0,
    )
    assert memo.priority_score == 1.0
    assert memo.bypass_risk_score == 0.0
    # Two warnings: one per clamp
    assert any("priority_score" in w for w in memo.parse_warnings)
    assert any("bypass_risk_score" in w for w in memo.parse_warnings)


def test_parse_memo_unknown_bypass_warns():
    raw = (
        '{"priority_score": 0.6, "bypass_risk_score": 0.4, '
        '"predicted_bypass": "fake_bypass_mechanism_xyz"}'
    )
    memo = parse_memo(
        raw,
        case_id="x",
        gene_symbol="X",
        retrieved_rule_ids=[],
        bypass_signals=[],
        cost_usd=0.0,
        seed=0,
    )
    assert memo.predicted_bypass is BypassMechanism.UNKNOWN
    assert any(
        "fake_bypass_mechanism_xyz" in w for w in memo.parse_warnings
    ), memo.parse_warnings


def test_parse_memo_garbage_raises_memo_parse_error():
    """No JSON object at all → raises. ``safe_parse_memo`` would
    swallow the exception; ``parse_memo`` re-raises it."""
    with pytest.raises(MemoParseError):
        parse_memo(
            "I refuse to answer. — the LLM",
            case_id="x",
            gene_symbol="X",
            retrieved_rule_ids=[],
            bypass_signals=[],
            cost_usd=0.0,
            seed=0,
        )


def test_parse_memo_missing_fields_default_quietly():
    """Missing optional fields should default without polluting warnings."""
    memo = parse_memo(
        "{}",
        case_id="x",
        gene_symbol="X",
        retrieved_rule_ids=[],
        bypass_signals=[],
        cost_usd=0.0,
        seed=0,
    )
    assert memo.priority_score == 0.0
    assert memo.bypass_risk_score == 0.0
    assert memo.predicted_bypass is BypassMechanism.UNKNOWN
    assert memo.supporting_evidence == []
    assert memo.rationale == ""
    assert memo.parse_warnings == []  # missing != malformed


def test_parse_memo_wrong_type_for_list_is_tolerant():
    """A bare string in a list field should be coerced into a list with a warning."""
    raw = '{"supporting_evidence": "single string"}'
    memo = parse_memo(
        raw,
        case_id="x",
        gene_symbol="X",
        retrieved_rule_ids=[],
        bypass_signals=[],
        cost_usd=0.0,
        seed=0,
    )
    assert memo.supporting_evidence == ["single string"]
    assert any(
        "supporting_evidence" in w for w in memo.parse_warnings
    )


# ── Swarm-level integration tests ───────────────────────────────────────────


def test_swarm_agent_uses_bypass_signals_in_prompt():
    """When a PathwayCache provides paralog data for BRAF, the user prompt
    must include the derived bypass signals so the LLM can act on them.
    The cross-agent analyzer relies on this: it asks 'was this signal
    shown to the swarm, and did it still mis-score?'."""
    cache = PathwayCache(
        paralogs_by_gene={"BRAF": ["ARAF", "RAF1"]},
        family_by_gene={"BRAF": "RAF_kinase"},
    )
    # Pick the BRAF case from the synthetic fixture.
    cases = [c for c in build_synthetic_targets() if c.target.gene_symbol == "BRAF"]
    assert len(cases) == 1
    llm = FakeLLMClient(default_text='{"priority_score": 0.5}')
    agent = SwarmAgent(
        cases[0],
        llm,
        _empty_router(),
        pathway_cache=cache,
    )
    memo = agent.run()
    # The user prompt was the second element of the recorded (system, user) tuple.
    assert len(llm.calls) == 1
    _, user_msg = llm.calls[0]
    assert "paralog_compensation" in user_msg
    assert "ARAF" in user_msg
    # And the memo records that the signal WAS seen
    assert "paralog_compensation" in memo.bypass_signals_seen


def test_swarm_agent_fallback_memo_on_unparseable_llm_output():
    """A misbehaving LLM should not crash the swarm — the agent must
    return a fallback memo with parse_warnings populated."""
    case = build_synthetic_targets()[0]
    llm = FakeLLMClient(default_text="this is a paragraph of prose, no JSON here")
    agent = SwarmAgent(case, llm, _empty_router())
    memo = agent.run()
    assert memo.case_id == case.case_id
    assert memo.priority_score == 0.0
    assert memo.bypass_risk_score == 0.0
    assert memo.parse_warnings  # non-empty
    assert any("memo_parse_error" in w for w in memo.parse_warnings)


def test_swarm_agent_propagates_real_scores_via_canned_json():
    """End-to-end: SwarmAgent uses FakeLLMClient.responses to drive a
    specific JSON payload and the resulting TargetMemo reflects it."""
    canned = (
        '{"priority_score": 0.77, "bypass_risk_score": 0.15, '
        '"predicted_bypass": "downstream_feedback", '
        '"supporting_evidence": ["genetics OT=0.92"], '
        '"rationale": "Strong genetics, MEK rebound documented."}'
    )
    case = build_synthetic_targets()[0]  # BRAF
    llm = FakeLLMClient(responses=[canned])
    agent = SwarmAgent(case, llm, _empty_router())
    memo = agent.run()
    assert memo.priority_score == pytest.approx(0.77)
    assert memo.bypass_risk_score == pytest.approx(0.15)
    assert memo.predicted_bypass is BypassMechanism.DOWNSTREAM_FEEDBACK
    assert memo.parse_warnings == []
    assert memo.rationale.startswith("Strong genetics")


def test_swarm_runner_threads_pathway_cache_into_every_agent():
    """The runner must share the cache across all agents (not allocate
    per-agent). Asserting via a side-effect: the prompt for each case
    references the modality-specific signals derived from the shared
    cache.
    """
    cache = PathwayCache(
        paralogs_by_gene={
            "BRAF": ["ARAF", "RAF1"],
            "KRAS": ["HRAS", "NRAS"],
            "EGFR": ["ERBB2", "ERBB3"],
        },
        family_by_gene={"BRAF": "RAF", "KRAS": "RAS", "EGFR": "HER"},
    )
    cases = build_synthetic_targets()
    llm = FakeLLMClient(default_text="{}")
    runner = SwarmRunner(
        llm,
        _empty_router(),
        max_workers=1,
        pathway_cache=cache,
    )
    report = runner.run(cases, campaign_id="x", seed=0)
    # Each agent saw paralog_compensation in its prompt.
    for (_sys, user_msg) in llm.calls:
        assert "paralog_compensation" in user_msg
    # And each memo records the signal it was shown.
    for memo in report.memos:
        assert "paralog_compensation" in memo.bypass_signals_seen


def test_swarm_parallel_path_with_scripted_responses_stays_ordered():
    """When FakeLLMClient.responses is a list, the call order under the
    parallel path determines which case gets which canned response.
    Because the runner submits futures in case-order and harvests them
    in case-order, the agent for the first case gets the first response.
    """
    cases = build_synthetic_targets()
    # Three distinct canned JSONs — easy to tell apart by priority_score.
    responses = [
        '{"priority_score": 0.1}',
        '{"priority_score": 0.5}',
        '{"priority_score": 0.9}',
    ]
    # NOTE: FakeLLMClient.responses is shared across threads and pops
    # FIFO. With max_workers > 1 the thread schedule decides who pops
    # first, so the priority↔case mapping can shuffle. The deterministic
    # contract is that *case order in memos* matches *case order in
    # input*. We assert that here, not the response↔case mapping.
    llm = FakeLLMClient(responses=list(responses))
    runner = SwarmRunner(llm, _empty_router(), max_workers=4)
    report = runner.run(cases, campaign_id="x", seed=0)
    # Memo order must match input case order.
    assert [m.case_id for m in report.memos] == [c.case_id for c in cases]
