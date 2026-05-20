"""Tests for medreason.runners.memory_wrapper — Phase 5 Commit 3.

MemoryRunner is the composition that wraps a base AgentRunner with the
Phase 5 memory pipeline: retrieve → inject → base.run() → parse
applied_rules → update posteriors. These tests exercise every branch
including the re-prompt path, the read-only store mode, and the
"retrieve-nothing-falls-through-to-zero-shot" path.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

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
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from medreason.retrieval import FakeEmbedder
from medreason.runners import AgentRunner, MemoryRunner
from medreason.store import LeakGuard, RuleStore


# ── Test doubles ────────────────────────────────────────────────────────────


class _RecordingBaseRunner:
    """AgentRunner that records every call and returns scripted AgentResults.

    Each entry in `responses` is an AgentResult template that will be
    patched with the case_id and seed on return. Supports a sequence of
    responses so we can simulate the re-prompt flow: first call returns
    an incomplete applied_rules list, second call returns the complete
    version.
    """
    runner_id = "recording-base"
    model_version = "recording-base"
    supports_memory = True

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []  # each is (case_id, system_extra, seed)

    def run(self, case, *, seed: int = 0, system_extra: str = ""):
        self.calls.append({
            "case_id": case.case_id,
            "system_extra": system_extra,
            "seed": seed,
        })
        template = self._responses.pop(0) if self._responses else {}
        det = template.get("determination", case.ground_truth_outcome)
        correct = (det == case.ground_truth_outcome)
        applied = template.get("applied_rules", [])
        return AgentResult(
            case_id=case.case_id,
            determination=det,
            reasoning_chain=template.get("reasoning_chain", "fake"),
            confidence=template.get("confidence", 0.8),
            correct=correct,
            input_tokens=template.get("input_tokens", 200),
            output_tokens=template.get("output_tokens", 50),
            cost_usd=template.get("cost_usd", 0.001),
            latency_ms=template.get("latency_ms", 300),
            runner_id=self.runner_id,
            seed=seed,
            mode="memory" if system_extra else "zero_shot",
            applied_rules=[
                AppliedRule(**a) if isinstance(a, dict) else a
                for a in applied
            ],
        )

    def estimated_cost_per_call(self) -> float:
        return 0.002


def _mk_rule(
    *,
    rule_id: str,
    cpt_families=(CPTFamily.IMAGING_MRI,),
    icd_chapters=(ICD10Chapter.MUSCULOSKELETAL,),
    payers=(Payer.AETNA,),
    action: str = "Require conservative therapy.",
    semantic: str = "lumbar MRI chronic back pain",
    success: int = 10,
    failure: int = 2,
    status: RuleStatus = RuleStatus.ACTIVE,
) -> ReasoningRule:
    return ReasoningRule(
        rule_id=rule_id,
        status=status,
        trigger=RuleTrigger(
            cpt_families=list(cpt_families),
            icd10_chapters=list(icd_chapters),
            payers=list(payers),
            semantic_predicate=semantic,
        ),
        action=action,
        evidence=RuleEvidence(
            supporting_case_ids=["train_01"],
            source_policy_citation="CMS LCD L1 §C.1",
            proposer_model="test",
        ),
        success_count=success,
        failure_count=failure,
    )


def _mk_case(
    *,
    case_id: str = "test_001",
    payer: Payer = Payer.AETNA,
    cpt: str = "72148",
    notes: str = "lumbar MRI chronic back pain physical therapy",
    ground_truth_outcome: Outcome = Outcome.APPROVED,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=payer,
            cpt_code=cpt,
            icd10_codes=["M54.5"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=notes,
        policy_excerpt="policy",
        ground_truth_outcome=ground_truth_outcome,
        difficulty=Difficulty.MEDIUM,
    )


@pytest.fixture
def populated_store(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_mk_rule(rule_id="rule_pt", action="Six weeks PT required."))
    store.put(_mk_rule(rule_id="rule_exam", action="Neurological exam required."))
    return store


# ── Protocol conformance ────────────────────────────────────────────────────


def test_memory_runner_is_protocol_conformant(populated_store):
    base = _RecordingBaseRunner(responses=[])
    wrapper = MemoryRunner(
        base_runner=base,
        store=populated_store,
        embedder=FakeEmbedder(),
    )
    assert isinstance(wrapper, AgentRunner)
    assert wrapper.runner_id == "recording-base:memory"
    assert wrapper.model_version == "recording-base"
    assert wrapper.supports_memory is True


def test_memory_runner_estimated_cost_includes_reranker_overhead(populated_store):
    base = _RecordingBaseRunner(responses=[])
    llm = FakeLLMClient()
    wrapper_no_rerank = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    wrapper_with_rerank = MemoryRunner(
        base_runner=base, store=populated_store,
        embedder=FakeEmbedder(), reranker_llm=llm,
    )
    assert wrapper_with_rerank.estimated_cost_per_call() > \
           wrapper_no_rerank.estimated_cost_per_call()


# ── Happy path: retrieve → inject → base.run → update posteriors ───────────


def test_memory_runner_retrieves_injects_and_updates_posteriors(populated_store):
    base = _RecordingBaseRunner(responses=[{
        "determination": Outcome.APPROVED,
        "applied_rules": [
            {"rule_id": "rule_pt", "applied": True, "rationale": "8 wks PT"},
            {"rule_id": "rule_exam", "applied": True, "rationale": "positive SLR"},
        ],
    }])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    case = _mk_case(ground_truth_outcome=Outcome.APPROVED)

    # Baseline counts
    before_pt = populated_store.get("rule_pt")
    before_exam = populated_store.get("rule_exam")
    assert before_pt.seen_count == 0

    result = wrapper.run(case, seed=11)

    assert result.correct is True
    assert result.runner_id == "recording-base:memory"
    assert result.mode == "memory"
    assert set(result.retrieved_rule_ids) == {"rule_pt", "rule_exam"}
    assert len(result.applied_rules) == 2
    assert all(a.applied for a in result.applied_rules)

    # Exactly one base call (no re-prompt needed)
    assert len(base.calls) == 1
    system_extra = base.calls[0]["system_extra"]
    assert "rule_pt" in system_extra
    assert "rule_exam" in system_extra
    # MemoryRunner defaults to compact_injection=True now; the compact
    # header omits "INSTITUTIONAL" but still has the REASONING MEMORY tag.
    assert "REASONING MEMORY" in system_extra

    # Posteriors updated: applied + correct → success+1, seen+1 on both
    after_pt = populated_store.get("rule_pt")
    after_exam = populated_store.get("rule_exam")
    assert after_pt.success_count == before_pt.success_count + 1
    assert after_pt.seen_count == before_pt.seen_count + 1
    assert after_exam.success_count == before_exam.success_count + 1


def test_memory_runner_applied_false_only_bumps_seen(populated_store):
    """Retrieved + not-applied rules should only increment seen_count."""
    base = _RecordingBaseRunner(responses=[{
        "determination": Outcome.APPROVED,
        "applied_rules": [
            {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            {"rule_id": "rule_exam", "applied": False, "rationale": "not relevant"},
        ],
    }])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    before_exam = populated_store.get("rule_exam")
    wrapper.run(_mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11)
    after_exam = populated_store.get("rule_exam")

    assert after_exam.success_count == before_exam.success_count  # unchanged
    assert after_exam.failure_count == before_exam.failure_count  # unchanged
    assert after_exam.seen_count == before_exam.seen_count + 1   # bumped


def test_memory_runner_wrong_case_failed_applied_rule_bumps_failure(populated_store):
    # Two rules retrieved (rule_pt, rule_exam) → both responses must
    # address both rule_ids so the wrapper doesn't re-prompt (or, if it
    # does, the retry must keep the wrong determination).
    wrong_det_response = {
        "determination": Outcome.DENIED,  # wrong for an approved case
        "applied_rules": [
            {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            {"rule_id": "rule_exam", "applied": False, "rationale": "n/a"},
        ],
    }
    base = _RecordingBaseRunner(responses=[wrong_det_response, wrong_det_response])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    before = populated_store.get("rule_pt")
    wrapper.run(_mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11)
    after = populated_store.get("rule_pt")

    assert after.failure_count == before.failure_count + 1
    assert after.success_count == before.success_count
    assert after.seen_count == before.seen_count + 1


# ── Re-prompt flow: agent misses rule_ids on first call ───────────────────


def test_memory_runner_reprompts_on_missing_rule_ids(populated_store):
    """First response missing rule_exam → wrapper re-prompts once."""
    base = _RecordingBaseRunner(responses=[
        # First call: only addresses rule_pt
        {
            "determination": Outcome.APPROVED,
            "applied_rules": [
                {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            ],
        },
        # Retry: addresses both
        {
            "determination": Outcome.APPROVED,
            "applied_rules": [
                {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
                {"rule_id": "rule_exam", "applied": False, "rationale": "n/a"},
            ],
            "input_tokens": 250,
            "output_tokens": 80,
            "cost_usd": 0.0015,
            "latency_ms": 400,
        },
    ])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    result = wrapper.run(
        _mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11
    )

    # Two base calls — first + retry
    assert len(base.calls) == 2
    assert wrapper.stats.total_re_prompts == 1

    # Retry system_extra contains the RETRY nudge
    assert "[RETRY]" in base.calls[1]["system_extra"]
    assert "rule_exam" in base.calls[1]["system_extra"]

    # Final applied_rules covers both rule_ids
    assert len(result.applied_rules) == 2
    by_id = {a.rule_id: a for a in result.applied_rules}
    assert by_id["rule_pt"].applied is True
    assert by_id["rule_exam"].applied is False

    # Usage accumulates across both calls
    # first: 200 + 50, retry: 250 + 80 → total 450 + 130
    assert result.input_tokens == 450
    assert result.output_tokens == 130
    assert result.cost_usd == pytest.approx(0.001 + 0.0015)


def test_memory_runner_still_normalizes_applied_rules_after_failed_reprompt(
    populated_store,
):
    """If the retry ALSO misses rule_ids, the wrapper normalizes the
    final applied_rules to have one entry per retrieved rule_id with
    the missing ones marked applied=False."""
    base = _RecordingBaseRunner(responses=[
        {
            "determination": Outcome.APPROVED,
            "applied_rules": [
                {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            ],
        },
        # Retry: STILL misses rule_exam
        {
            "determination": Outcome.APPROVED,
            "applied_rules": [
                {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            ],
        },
    ])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    result = wrapper.run(
        _mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11
    )

    # Two calls total
    assert len(base.calls) == 2
    # applied_rules normalized: two entries, rule_exam marked applied=False
    by_id = {a.rule_id: a for a in result.applied_rules}
    assert by_id["rule_pt"].applied is True
    assert by_id["rule_exam"].applied is False


def test_memory_runner_does_not_reprompt_when_all_applied(populated_store):
    base = _RecordingBaseRunner(responses=[{
        "determination": Outcome.APPROVED,
        "applied_rules": [
            {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            {"rule_id": "rule_exam", "applied": True, "rationale": "hit"},
        ],
    }])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    wrapper.run(_mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11)
    assert len(base.calls) == 1
    assert wrapper.stats.total_re_prompts == 0


# ── Empty retrieval cascades to zero-shot ──────────────────────────────────


def test_memory_runner_empty_retrieval_skips_reprompt_and_posterior_update(
    sqlite_conn,
):
    """No rules in the store → empty injection → base runner runs
    zero-shot. No re-prompt, no posterior update. The AgentResult
    should have mode='memory' (it went through the wrapper) but
    applied_rules + retrieved_rule_ids are empty."""
    store = RuleStore(sqlite_conn)  # empty store
    base = _RecordingBaseRunner(responses=[{
        "determination": Outcome.APPROVED,
        "applied_rules": [],
    }])
    wrapper = MemoryRunner(
        base_runner=base, store=store, embedder=FakeEmbedder()
    )
    result = wrapper.run(
        _mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11
    )
    assert result.retrieved_rule_ids == []
    assert result.applied_rules == []
    assert len(base.calls) == 1
    # system_extra should be empty since no rules to inject
    assert base.calls[0]["system_extra"] == ""
    assert wrapper.stats.total_re_prompts == 0


# ── Read-only store (test-phase): posterior updates skipped gracefully ─────


def test_memory_runner_posterior_update_skipped_in_test_phase(populated_store):
    """When the store's LeakGuard is in test-phase read-only mode, the
    wrapper catches TestSetLeakError on update_posteriors and counts
    it — it doesn't crash the eval run."""
    populated_store.leak_guard.enter_test_phase()

    base = _RecordingBaseRunner(responses=[{
        "determination": Outcome.APPROVED,
        "applied_rules": [
            {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
            {"rule_id": "rule_exam", "applied": True, "rationale": "hit"},
        ],
    }])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    before = populated_store.get("rule_pt")
    result = wrapper.run(
        _mk_case(ground_truth_outcome=Outcome.APPROVED), seed=11
    )
    after = populated_store.get("rule_pt")

    assert result.correct is True
    # Store was not updated
    assert after.success_count == before.success_count
    assert after.seen_count == before.seen_count
    # Wrapper recorded the read-only skip
    assert wrapper.stats.total_posterior_skips_readonly == 1
    assert wrapper.stats.total_posterior_updates == 0


# ── Stats accumulate across multiple calls ────────────────────────────────


def test_memory_runner_stats_accumulate(populated_store):
    base = _RecordingBaseRunner(responses=[
        {
            "determination": Outcome.APPROVED,
            "applied_rules": [
                {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
                {"rule_id": "rule_exam", "applied": True, "rationale": "hit"},
            ],
        },
        {
            "determination": Outcome.APPROVED,
            "applied_rules": [
                {"rule_id": "rule_pt", "applied": True, "rationale": "hit"},
                {"rule_id": "rule_exam", "applied": False, "rationale": "n/a"},
            ],
        },
    ])
    wrapper = MemoryRunner(
        base_runner=base, store=populated_store, embedder=FakeEmbedder()
    )
    wrapper.run(_mk_case(case_id="c1", ground_truth_outcome=Outcome.APPROVED))
    wrapper.run(_mk_case(case_id="c2", ground_truth_outcome=Outcome.APPROVED))

    assert wrapper.stats.total_calls == 2
    assert wrapper.stats.total_retrievals == 2
    assert wrapper.stats.total_posterior_updates == 2
    assert wrapper.stats.total_re_prompts == 0
