"""Phase 5 end-to-end integration test.

Walks the full memory pipeline using FakeLLMClient + a scripted base
runner + FakeEmbedder — no network, no API keys. Proves the modules
from commits 1-3 wire together correctly:

    critic.run_critic        (commit 1)
    propose_rules            (commit 1)
    GeneralizationGate       (commit 2)
    RuleStore.put            (phase 1)
    retrieval.retrieve       (commit 2)
    build_rule_checklist     (commit 1)
    MemoryRunner.run         (commit 3)
    update_posteriors        (commit 3)
    apply_quarantine_policy  (commit 3)

This is NOT a performance test — it doesn't prove memory beats
zero-shot on real data. That evaluation happens in Phase 7 against
real LLMs. Phase 5's done-when is structural correctness: the pipeline
flows end-to-end, rules get promoted through the gate, retrieval
finds them, the memory wrapper injects them, the base runner reports
application, and posteriors update.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from medreason.extraction import (
    GeneralizationGate,
    propose_rules,
    run_critic,
)
from medreason.llm import FakeLLMClient
from medreason.ontology import (
    AgentResult,
    AppliedRule,
    BenchmarkCase,
    CPTFamily,
    Difficulty,
    FacilityType,
    ICD10Chapter,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
    ReasoningRule,
    RuleStatus,
)
from medreason.posterior import apply_quarantine_policy
from medreason.retrieval import FakeEmbedder
from medreason.runners import MemoryRunner
from medreason.store import RuleStore, TraceStore
from medreason_bench.data import parse_lcd_xml
from medreason_bench.data.case_builder import build_cases_from_lcd
from pathlib import Path


FIXTURE = (
    Path(__file__).parent.parent
    / "medreason_bench" / "data" / "fixtures" / "sample_lcd.xml"
)


class _ScriptedBaseRunner:
    """Agent runner whose determination + applied_rules are scripted
    per case_id, with a fallback oracle that predicts ground truth.

    Used for the end-to-end test: the 'agent' side is deterministic so
    we can check that the memory pipeline's retrieval, injection, and
    posterior updates happen in the right order.
    """
    runner_id = "scripted-e2e"
    model_version = "scripted-e2e"
    supports_memory = True

    def __init__(self, applied_rules_by_case: dict[str, list[AppliedRule]]):
        self._applied = applied_rules_by_case
        self.calls = 0

    def run(self, case, *, seed: int = 0, system_extra: str = ""):
        self.calls += 1
        applied = self._applied.get(case.case_id, [])
        return AgentResult(
            case_id=case.case_id,
            determination=case.ground_truth_outcome,  # oracle
            reasoning_chain="scripted",
            confidence=0.9,
            correct=True,
            input_tokens=200,
            output_tokens=60,
            cost_usd=0.001,
            latency_ms=300,
            runner_id=self.runner_id,
            seed=seed,
            mode="memory" if system_extra else "zero_shot",
            applied_rules=list(applied),
        )

    def estimated_cost_per_call(self) -> float:
        return 0.001


def _critic_agree_response(determination: Outcome) -> str:
    return json.dumps({
        "determination": determination.value,
        "reasoning_steps": [
            {"step": 1, "claim": "policy applies",
             "evidence": "notes show criterion met", "policy_hook": "§C.1"}
        ],
        "confidence": 0.9,
    })


def _proposer_response(n_rules: int = 1) -> str:
    rules = []
    for i in range(n_rules):
        rules.append({
            "trigger": {
                "cpt_families": ["imaging_mri"],
                "icd10_chapters": ["musculoskeletal"],
                "payers": ["Medicare"],
                "semantic_predicate": f"lumbar MRI rule {i}",
            },
            "action": f"Check rule number {i}.",
            "rationale": "Derived from verified trace.",
            "polarity": "requires_check",
            "source_policy_citation": "CMS LCD L34522 §C.1",
        })
    return json.dumps({"rules": rules})


@pytest.fixture(scope="module")
def lcd_policy():
    return parse_lcd_xml(FIXTURE)


@pytest.fixture
def training_cases(lcd_policy):
    """20 training cases from the v0.0 fixture — enough variety for
    the gate to find matching held-out cases per candidate rule."""
    return build_cases_from_lcd(lcd_policy, target_count=20, seed=42)


# ── Stage 1: critic → proposer → gate → store ──────────────────────────────


def test_critic_verified_traces_become_stored_rules(
    sqlite_conn, lcd_policy, training_cases
):
    """Run the critic on the first training case, propose a rule,
    validate it with the gate, and persist it. This is the full
    extraction pipeline end-to-end."""
    store = RuleStore(sqlite_conn)
    trace_store = TraceStore(sqlite_conn)

    train_case = training_cases[0]
    assert train_case.ground_truth_outcome == Outcome.APPROVED

    # Step 1: critic agrees
    critic_llm = FakeLLMClient(responses=[
        _critic_agree_response(train_case.ground_truth_outcome)
    ])
    critic_result = run_critic(
        train_case, train_case.ground_truth_outcome, critic_llm
    )
    assert critic_result.agrees is True
    assert critic_result.trace is not None
    trace_store.put(critic_result.trace)

    # Step 2: proposer suggests one rule
    proposer_llm = FakeLLMClient(responses=[_proposer_response(n_rules=1)])
    proposal = propose_rules(
        critic_result.trace,
        lcd_policy,
        proposer_llm,
        supporting_case_ids=[train_case.case_id],
    )
    assert proposal.n_candidates == 1
    candidate = proposal.candidates[0]
    assert candidate.status == RuleStatus.CANDIDATE

    # Step 3: gate validates on other training cases (scripted all-correct)
    held_out = training_cases[1:]
    scripted_runner = _ScriptedBaseRunner(applied_rules_by_case={})
    gate = GeneralizationGate(held_out, scripted_runner, k=10)
    gate_result = gate.validate(candidate)

    assert gate_result.promoted is True
    assert gate_result.new_status == RuleStatus.ACTIVE
    assert gate_result.total_trials >= 5

    # Step 4: persist the promoted rule
    candidate.status = RuleStatus.ACTIVE
    store.put(candidate)

    assert store.count(RuleStatus.ACTIVE) == 1
    assert trace_store.count() == 1
    reloaded = store.list_by_status(RuleStatus.ACTIVE)[0]
    assert reloaded.rule_id == candidate.rule_id


# ── Stage 2: MemoryRunner retrieves + injects + updates posteriors ────────


def test_memory_runner_uses_stored_rule_end_to_end(
    sqlite_conn, lcd_policy, training_cases
):
    """With an ACTIVE rule already in the store, run MemoryRunner on a
    dev case. Expected: retrieval finds the rule, the base runner
    gets an injection containing the rule_id, applied_rules lands
    in the AgentResult, and the posterior counts update."""
    store = RuleStore(sqlite_conn)

    # Pre-seed the store with a directly-matching rule
    from medreason.ontology import (
        ReasoningRule, RuleEvidence, RuleTrigger,
    )
    seed_rule = ReasoningRule(
        rule_id="seed_rule",
        status=RuleStatus.ACTIVE,
        trigger=RuleTrigger(
            cpt_families=[CPTFamily.IMAGING_MRI],
            icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
            payers=[Payer.MEDICARE],
            semantic_predicate="lumbar MRI chronic back pain",
        ),
        action="Require six weeks conservative therapy before approval.",
        rationale="Policy requires conservative trial.",
        polarity="requires_check",
        evidence=RuleEvidence(
            supporting_case_ids=["train_01"],
            source_policy_citation="CMS LCD L34522 §C.1",
            proposer_model="test",
        ),
        success_count=5,
        failure_count=1,
    )
    store.put(seed_rule)

    # Use a training case as the "dev" case — ensures trigger match
    dev_case = training_cases[0]
    assert dev_case.task_config.cpt_code in ("72148", "72149", "72158")
    assert dev_case.task_config.payer == Payer.MEDICARE

    # Scripted base runner: apply the rule correctly
    base = _ScriptedBaseRunner(applied_rules_by_case={
        dev_case.case_id: [
            AppliedRule(
                rule_id="seed_rule", applied=True,
                rationale="clinical notes document conservative therapy",
            ),
        ],
    })
    wrapper = MemoryRunner(
        base_runner=base, store=store, embedder=FakeEmbedder()
    )

    before = store.get("seed_rule")
    result = wrapper.run(dev_case, seed=11)
    after = store.get("seed_rule")

    assert result.correct is True
    assert result.mode == "memory"
    assert "seed_rule" in result.retrieved_rule_ids
    assert len(result.applied_rules) == 1
    assert result.applied_rules[0].rule_id == "seed_rule"
    assert result.applied_rules[0].applied is True

    # Posterior updated: applied=True + correct → success_count +1
    assert after.success_count == before.success_count + 1
    assert after.seen_count == before.seen_count + 1


# ── Stage 3: Full loop with multiple runs + quarantine policy ─────────────


def test_full_memory_loop_with_quarantine(
    sqlite_conn, training_cases
):
    """Exercise the feedback loop: run memory inference multiple times,
    let a rule accumulate failures, then verify the quarantine policy
    moves it."""
    store = RuleStore(sqlite_conn)

    from medreason.ontology import (
        ReasoningRule, RuleEvidence, RuleTrigger,
    )
    bad_rule = ReasoningRule(
        rule_id="bad_rule",
        status=RuleStatus.ACTIVE,
        trigger=RuleTrigger(
            cpt_families=[CPTFamily.IMAGING_MRI],
            icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
            payers=[Payer.MEDICARE],
        ),
        action="Always deny lumbar MRI.",
        evidence=RuleEvidence(
            supporting_case_ids=["train_01"],
            source_policy_citation="CMS LCD L34522 §L.2",
            proposer_model="test",
        ),
        success_count=0,
        failure_count=0,
    )
    store.put(bad_rule)

    # A base runner that ALWAYS applies the rule but the case is always correct
    # (ground truth approved → bad_rule leads to wrong denial answers if it
    # were enforced, but scripted runner returns ground truth regardless —
    # so the rule is "applied" on correct cases, and should see success_count
    # go UP. To make the test meaningful, we need a runner that returns
    # WRONG answers when applying the rule on approved cases).
    class _AlwaysDeniesButAppliesRule:
        runner_id = "denier"
        model_version = "denier"
        supports_memory = True

        def run(self, case, *, seed: int = 0, system_extra: str = ""):
            return AgentResult(
                case_id=case.case_id,
                determination=Outcome.DENIED,  # wrong on approved cases
                reasoning_chain="applied bad rule",
                confidence=0.7,
                correct=(case.ground_truth_outcome == Outcome.DENIED),
                input_tokens=200,
                output_tokens=50,
                cost_usd=0.001,
                latency_ms=300,
                runner_id=self.runner_id,
                seed=seed,
                mode="memory" if system_extra else "zero_shot",
                applied_rules=[
                    AppliedRule(rule_id="bad_rule", applied=True,
                                rationale="deny per rule"),
                ],
            )

        def estimated_cost_per_call(self) -> float:
            return 0.001

    wrapper = MemoryRunner(
        base_runner=_AlwaysDeniesButAppliesRule(),
        store=store,
        embedder=FakeEmbedder(),
    )

    # Run against 10 cases — all approved in ground truth, so every
    # "applied=True + DENIED result" → failure_count += 1
    approved_cases = [
        c for c in training_cases
        if c.ground_truth_outcome == Outcome.APPROVED
    ][:10]
    assert len(approved_cases) >= 8

    for case in approved_cases:
        wrapper.run(case, seed=11)

    after = store.get("bad_rule")
    assert after.failure_count >= 8  # rule failed on every approved case
    assert after.success_count == 0
    # posterior ~1/11 ≈ 0.09, well below quarantine tau

    # Apply quarantine policy
    moved = apply_quarantine_policy(store)
    assert "bad_rule" in moved
    quarantined = store.get("bad_rule")
    assert quarantined.status == RuleStatus.QUARANTINED
    assert quarantined.quarantine_count == 1

    # After quarantine, a new memory call on the same case should NOT
    # retrieve the rule (Tier 1 filters on ACTIVE by default)
    fresh_result = wrapper.run(approved_cases[0], seed=11)
    assert "bad_rule" not in fresh_result.retrieved_rule_ids
