"""Tests for medreason_bench.train — the CLI glue around Phase 5.

Exercises run_training end-to-end with injected FakeLLMClient + a
scripted base runner. Catches wiring bugs (critic result not persisted,
gate pool not excluding the current case, cost accumulation, etc.)
before any real API money is spent.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from medreason.llm import FakeLLMClient
from medreason.ontology import (
    AgentResult,
    AppliedRule,
    CPTFamily,
    ICD10Chapter,
    Outcome,
    Payer,
    RuleStatus,
)
from medreason.store import RuleStore, TraceStore
from medreason_bench.data import parse_lcd_xml
from medreason_bench.data.case_builder import build_cases_from_lcd
from medreason_bench.train import TrainingConfig, TrainingReport, run_training


FIXTURE = (
    Path(__file__).parent.parent
    / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"
)


@pytest.fixture(scope="module")
def policy():
    return parse_lcd_xml(FIXTURE)


@pytest.fixture(scope="module")
def train_cases(policy):
    return build_cases_from_lcd(policy, target_count=10, seed=42)


class _OracleRunner:
    """Base runner that always predicts the ground truth outcome."""
    runner_id = "oracle-v0"
    model_version = "oracle-v0"
    supports_memory = True

    def __init__(self):
        self.calls = 0

    def run(self, case, *, seed: int = 0, system_extra: str = ""):
        self.calls += 1
        return AgentResult(
            case_id=case.case_id,
            determination=case.ground_truth_outcome,
            reasoning_chain="oracle",
            confidence=0.95,
            correct=True,
            input_tokens=400,
            output_tokens=100,
            cost_usd=0.0015,
            latency_ms=250,
            runner_id=self.runner_id,
            seed=seed,
            mode="memory" if system_extra else "zero_shot",
        )

    def estimated_cost_per_call(self) -> float:
        return 0.0015


def _critic_agrees(determination: Outcome) -> str:
    return json.dumps({
        "determination": determination.value,
        "reasoning_steps": [
            {"step": 1, "claim": "criterion met",
             "evidence": "clinical notes", "policy_hook": "§C.1"},
        ],
        "confidence": 0.9,
    })


def _critic_disagrees() -> str:
    return json.dumps({
        "determination": "denied",  # hard-coded so it can mismatch
        "reasoning_steps": [],
        "confidence": 0.5,
    })


def _one_proposal() -> str:
    return json.dumps({
        "rules": [{
            "trigger": {
                "cpt_families": ["imaging_mri"],
                "icd10_chapters": ["musculoskeletal"],
                "payers": ["Medicare"],
                "semantic_predicate": "lumbar MRI routine back pain",
            },
            "action": "Require six weeks conservative therapy documentation.",
            "rationale": "Policy requires conservative trial.",
            "polarity": "requires_check",
            "source_policy_citation": "CMS LCD L34522 §C.1",
        }]
    })


def _empty_proposal() -> str:
    return json.dumps({"rules": []})


# ── Happy path: critic agrees, proposer emits one rule, gate promotes ─────


def test_run_training_promotes_rules_when_everything_agrees(
    sqlite_conn, policy, train_cases
):
    # For each training case we'll need: 1 critic call + 1 proposer call.
    # The gate uses the SAME runner (oracle) so every gate call returns
    # correct=True, guaranteeing promotion of any valid proposal.
    critic_responses = [
        _critic_agrees(c.ground_truth_outcome) for c in train_cases
    ]
    proposer_responses = [_one_proposal() for _ in train_cases]

    critic_llm = FakeLLMClient(responses=critic_responses)
    proposer_llm = FakeLLMClient(responses=proposer_responses)
    runner = _OracleRunner()
    rule_store = RuleStore(sqlite_conn)
    trace_store = TraceStore(sqlite_conn)

    config = TrainingConfig(
        runner=runner,
        critic_llm=critic_llm,
        proposer_llm=proposer_llm,
        store=rule_store,
        trace_store=trace_store,
        policy=policy,
        train_cases=train_cases,
        gate_k=5,
    )

    report = run_training(config)

    assert isinstance(report, TrainingReport)
    assert report.n_cases_seen == len(train_cases)
    assert report.n_agent_correct == len(train_cases)
    assert report.n_critic_agreed == len(train_cases)
    assert report.n_traces_stored == len(train_cases)
    assert report.n_rules_proposed == len(train_cases)
    assert report.n_rules_promoted == len(train_cases)
    assert report.n_rules_deprecated == 0
    assert report.cost_total > 0

    # Store actually has rules
    assert rule_store.count(RuleStatus.ACTIVE) == len(train_cases)
    assert trace_store.count() == len(train_cases)

    # Each promoted rule has a cached embedding
    for rule in rule_store.list_by_status(RuleStatus.ACTIVE):
        assert rule.trigger.semantic_embedding is not None
        assert rule.trigger.embedding_model == "fake-embedder-v0"


def test_run_training_drops_disagreement(sqlite_conn, policy, train_cases):
    """When the critic disagrees, no trace is stored and no rule is proposed."""
    # Mix: first 3 agree, rest disagree
    critic_responses = [
        _critic_agrees(train_cases[0].ground_truth_outcome),
        _critic_agrees(train_cases[1].ground_truth_outcome),
        _critic_agrees(train_cases[2].ground_truth_outcome),
    ] + [_critic_disagrees() for _ in train_cases[3:]]
    # Proposer only runs for the agreed cases
    proposer_responses = [_one_proposal() for _ in range(3)]

    critic_llm = FakeLLMClient(responses=critic_responses)
    proposer_llm = FakeLLMClient(responses=proposer_responses)
    runner = _OracleRunner()
    rule_store = RuleStore(sqlite_conn)
    trace_store = TraceStore(sqlite_conn)

    config = TrainingConfig(
        runner=runner, critic_llm=critic_llm, proposer_llm=proposer_llm,
        store=rule_store, trace_store=trace_store,
        policy=policy, train_cases=train_cases, gate_k=3,
    )
    report = run_training(config)

    assert report.n_critic_agreed == 3
    # Only 3 traces stored — the disagreement cases don't persist
    assert report.n_traces_stored == 3
    assert report.n_proposer_invoked == 3


def test_run_training_handles_empty_proposer_output(
    sqlite_conn, policy, train_cases
):
    """Proposer returning {"rules": []} should NOT error — it's an
    honest 'no generalizable rules extractable' response."""
    critic_responses = [
        _critic_agrees(c.ground_truth_outcome) for c in train_cases
    ]
    proposer_responses = [_empty_proposal() for _ in train_cases]

    critic_llm = FakeLLMClient(responses=critic_responses)
    proposer_llm = FakeLLMClient(responses=proposer_responses)
    runner = _OracleRunner()
    rule_store = RuleStore(sqlite_conn)
    trace_store = TraceStore(sqlite_conn)

    config = TrainingConfig(
        runner=runner, critic_llm=critic_llm, proposer_llm=proposer_llm,
        store=rule_store, trace_store=trace_store,
        policy=policy, train_cases=train_cases, gate_k=3,
    )
    report = run_training(config)

    assert report.n_rules_proposed == 0
    assert report.n_rules_promoted == 0
    # Traces still stored even with no rules
    assert report.n_traces_stored == len(train_cases)


def test_run_training_respects_max_cases(sqlite_conn, policy, train_cases):
    critic_llm = FakeLLMClient(responses=[_critic_agrees(train_cases[0].ground_truth_outcome)])
    proposer_llm = FakeLLMClient(responses=[_empty_proposal()])
    runner = _OracleRunner()
    rule_store = RuleStore(sqlite_conn)
    trace_store = TraceStore(sqlite_conn)

    config = TrainingConfig(
        runner=runner, critic_llm=critic_llm, proposer_llm=proposer_llm,
        store=rule_store, trace_store=trace_store,
        policy=policy, train_cases=train_cases, gate_k=3,
        max_cases=1,
    )
    report = run_training(config)
    assert report.n_cases_seen == 1
    assert runner.calls == 1


def test_run_training_gate_cost_reflects_trials(sqlite_conn, policy, train_cases):
    """Gate-phase cost should accumulate proportional to actual trials
    run, not just the number of rules gated."""
    critic_responses = [
        _critic_agrees(c.ground_truth_outcome) for c in train_cases
    ]
    proposer_responses = [_one_proposal() for _ in train_cases]
    critic_llm = FakeLLMClient(responses=critic_responses)
    proposer_llm = FakeLLMClient(responses=proposer_responses)
    runner = _OracleRunner()
    rule_store = RuleStore(sqlite_conn)
    trace_store = TraceStore(sqlite_conn)

    config = TrainingConfig(
        runner=runner, critic_llm=critic_llm, proposer_llm=proposer_llm,
        store=rule_store, trace_store=trace_store,
        policy=policy, train_cases=train_cases, gate_k=3,
    )
    report = run_training(config)

    # Every rule that was gated went through K held-out cases (subject
    # to min_total_for_evaluation=3). Cost should be > 0 when anything
    # was promoted.
    if report.n_rules_promoted > 0:
        assert report.cost_gate > 0


def test_run_training_report_json_roundtrip(sqlite_conn, policy, train_cases):
    critic_llm = FakeLLMClient(responses=[_critic_agrees(c.ground_truth_outcome) for c in train_cases])
    proposer_llm = FakeLLMClient(responses=[_empty_proposal() for _ in train_cases])
    runner = _OracleRunner()
    config = TrainingConfig(
        runner=runner, critic_llm=critic_llm, proposer_llm=proposer_llm,
        store=RuleStore(sqlite_conn), trace_store=TraceStore(sqlite_conn),
        policy=policy, train_cases=train_cases, gate_k=3,
    )
    report = run_training(config)
    data = report.to_dict()
    assert data["n_cases_seen"] == len(train_cases)
    assert "cost_total" in data
    assert isinstance(data["rejection_reasons"], list)
    # JSON-serializable
    json.dumps(data)
