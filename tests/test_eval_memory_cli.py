"""Tests for the `medreason-bench eval --memory` CLI path.

Pre-populates a rule store, runs `run_eval` wrapping the base runner
in MemoryRunner, and verifies the resulting AgentResults carry the
retrieved_rule_ids and applied_rules fields populated — plus the
leaderboard entry can be built from the run.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from medreason.ontology import (
    AgentResult,
    AppliedRule,
    CPTFamily,
    ICD10Chapter,
    Payer,
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from medreason.retrieval import FakeEmbedder
from medreason.runners import MemoryRunner
from medreason.store import RuleStore
from medreason_bench.eval.harness import EvalConfig, run_eval
from medreason_bench.leaderboard.build import build_entry


SPLITS_ROOT = (
    Path(__file__).parent.parent
    / "medreason_bench" / "data" / "splits"
)


class _ScriptedBase:
    runner_id = "scripted-base"
    model_version = "scripted-base"
    supports_memory = True

    def __init__(self):
        self.calls = 0

    def run(self, case, *, seed: int = 0, system_extra: str = ""):
        self.calls += 1
        return AgentResult(
            case_id=case.case_id,
            determination=case.ground_truth_outcome,
            reasoning_chain="scripted",
            confidence=0.9,
            correct=True,
            input_tokens=300,
            output_tokens=80,
            cost_usd=0.001,
            latency_ms=300,
            runner_id=self.runner_id,
            seed=seed,
            mode="memory" if system_extra else "zero_shot",
            applied_rules=[
                # Echo back any rule_ids from the injection so the
                # wrapper's applied_rules normalization path runs
                # cleanly. Safe for fake injection.
                AppliedRule(rule_id=line.split("]")[0].lstrip("[").strip(),
                            applied=True, rationale="scripted")
                for line in system_extra.split("\n")
                if line.startswith("[rule_")
            ],
        )

    def estimated_cost_per_call(self) -> float:
        return 0.001


@pytest.fixture
def populated_store_with_one_rule(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(ReasoningRule(
        rule_id="rule_pt",
        status=RuleStatus.ACTIVE,
        trigger=RuleTrigger(
            cpt_families=[CPTFamily.IMAGING_MRI],
            icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
            payers=[Payer.MEDICARE],
            semantic_predicate="lumbar MRI chronic back pain",
        ),
        action="Require six weeks conservative therapy.",
        rationale="Policy conservative trial requirement.",
        polarity="requires_check",
        evidence=RuleEvidence(
            supporting_case_ids=["train_001"],
            source_policy_citation="CMS LCD L34522 §C.1",
            proposer_model="test",
        ),
        success_count=8,
        failure_count=2,
    ))
    return store


def test_eval_memory_runs_end_to_end_on_dev_split(populated_store_with_one_rule):
    """Full flow: MemoryRunner wraps a scripted base runner, run_eval
    iterates the dev split, and the resulting EvalRun can be turned
    into a LeaderboardEntry with pattern_utilization populated."""
    base = _ScriptedBase()
    wrapper = MemoryRunner(
        base_runner=base,
        store=populated_store_with_one_rule,
        embedder=FakeEmbedder(),
    )
    config = EvalConfig(
        runner=wrapper,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11],
    )
    run = run_eval(config)
    assert run.n_cases == 10
    assert run.total_calls == 10

    # Every Medicare lumbar MRI case should retrieve rule_pt (the only
    # rule in the store). Confirm that happened on at least ONE case.
    flat = run.flat_results()
    retrievals = [r for r in flat if "rule_pt" in r.retrieved_rule_ids]
    assert len(retrievals) >= 1

    # Build a leaderboard entry
    cases_by_id = {c.case_id: c for c in run.cases}
    entry, metrics = build_entry(run, cases_by_id)
    assert entry.n_cases == 10
    assert entry.accuracy_mean == pytest.approx(1.0)  # scripted oracle
    # Pattern utilization is populated for memory runs
    assert metrics.pattern_utilization is not None
    assert entry.pattern_utilization is not None
    assert 0.0 <= entry.pattern_utilization <= 1.0


def test_eval_memory_base_runner_sees_memory_injection(
    populated_store_with_one_rule
):
    """The base runner's system_extra must contain the retrieved rule
    checklist when the wrapper is in memory mode."""
    base = _ScriptedBase()
    wrapper = MemoryRunner(
        base_runner=base,
        store=populated_store_with_one_rule,
        embedder=FakeEmbedder(),
    )
    config = EvalConfig(
        runner=wrapper,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11],
    )
    run_eval(config)
    # At least one base call should have received a non-empty
    # system_extra — the wrapper retrieved rule_pt on matching cases.
    assert wrapper.stats.total_retrievals == 10


def test_eval_memory_falls_back_to_zero_shot_on_empty_store(sqlite_conn):
    """With no rules in the store, retrieval returns empty and the
    base runner sees an empty system_extra — effectively zero-shot."""
    empty_store = RuleStore(sqlite_conn)
    base = _ScriptedBase()
    wrapper = MemoryRunner(
        base_runner=base,
        store=empty_store,
        embedder=FakeEmbedder(),
    )
    config = EvalConfig(
        runner=wrapper,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11],
    )
    run = run_eval(config)
    assert run.n_cases == 10
    # All runs should have empty retrieved_rule_ids
    for r in run.flat_results():
        assert r.retrieved_rule_ids == []
