"""Tests for medreason.extraction.generalization_gate — Phase 5 Commit 2.

The gate is the load-bearing quality lever: without it the pipeline
devolves to 'store whatever the proposer suggested'. Every one of
these tests is a regression guardrail for the promote/deprecate/
candidate decision matrix.
"""

from __future__ import annotations

import pytest

from medreason.extraction import (
    DEFAULT_THRESHOLDS,
    GateResult,
    GateThresholds,
    GeneralizationGate,
)
from medreason.extraction.generalization_gate import _rule_content_hash
from medreason.ontology import (
    AgentResult,
    BenchmarkCase,
    CPTFamily,
    Difficulty,
    FacilityType,
    ICD10Chapter,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
    ReasoningRule,
    ReasoningTrace,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _mk_rule(**kwargs) -> ReasoningRule:
    defaults = dict(
        trigger=RuleTrigger(
            cpt_families=[CPTFamily.IMAGING_MRI],
            icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
        ),
        action="Require conservative therapy.",
        evidence=RuleEvidence(
            supporting_case_ids=["train_001"],
            source_policy_citation="CMS LCD L1 §C.1",
        ),
    )
    defaults.update(kwargs)
    return ReasoningRule(**defaults)


def _mk_case(
    case_id: str,
    *,
    outcome: Outcome = Outcome.APPROVED,
    cpt: str = "72148",
    icds: list[str] | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE,
            cpt_code=cpt,
            icd10_codes=icds or ["M54.5"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes="test",
        policy_excerpt="policy",
        ground_truth_outcome=outcome,
        difficulty=Difficulty.MEDIUM,
    )


class ScriptedRunner:
    """AgentRunner that returns a pre-scripted correctness flag per case_id.

    For gate tests we only care about `result.correct` and whether
    system_extra was passed — the exact determination doesn't matter,
    since the gate reads `result.correct`.
    """
    runner_id = "scripted-v0"
    model_version = "scripted-v0"
    supports_memory = True

    def __init__(
        self,
        correctness_by_case: dict[str, bool],
        *,
        assert_injection: bool = True,
    ):
        self._correctness = correctness_by_case
        self._assert_injection = assert_injection
        self.calls: list[tuple[str, str]] = []  # (case_id, system_extra)

    def run(self, case, *, seed: int = 0, system_extra: str = "") -> AgentResult:
        self.calls.append((case.case_id, system_extra))
        if self._assert_injection:
            assert system_extra, (
                "gate must inject a rule checklist into system_extra"
            )
        correct = self._correctness.get(case.case_id, False)
        return AgentResult(
            case_id=case.case_id,
            determination=(
                case.ground_truth_outcome if correct else Outcome.DENIED
            ),
            correct=correct,
            reasoning_chain="scripted",
            confidence=0.9,
            seed=seed,
            mode="memory" if system_extra else "zero_shot",
        )

    def estimated_cost_per_call(self) -> float:
        return 0.0


# ── Promotion: high score + sufficient trials → ACTIVE ────────────────────


def test_good_rule_gets_promoted_to_active():
    cases = [_mk_case(f"train_{i:03d}") for i in range(8)]
    # 7 of 8 correct — score 0.875
    correctness = {c.case_id: True for c in cases}
    correctness["train_005"] = False

    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=10)
    rule = _mk_rule()
    result = gate.validate(rule)

    assert isinstance(result, GateResult)
    assert result.new_status is RuleStatus.ACTIVE
    assert result.promoted is True
    assert result.total_trials == 8
    assert result.correct == 7
    assert result.score == pytest.approx(7 / 8)
    assert "promoted" in result.reason


def test_promotion_requires_both_score_and_min_trials():
    """score 1.0 on 4 trials should NOT promote (needs >= 5 by default)."""
    cases = [_mk_case(f"train_{i:03d}") for i in range(4)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=10)
    rule = _mk_rule()
    result = gate.validate(rule)
    assert result.new_status is RuleStatus.CANDIDATE  # deferred / candidate
    assert result.promoted is False


# ── Deprecation: very low score → DEPRECATED ─────────────────────────────


def test_bad_rule_gets_deprecated():
    cases = [_mk_case(f"train_{i:03d}") for i in range(10)]
    # 2 of 10 correct — score 0.2, below 0.4
    correctness = {c.case_id: False for c in cases}
    correctness["train_000"] = True
    correctness["train_001"] = True

    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=10)
    rule = _mk_rule()
    result = gate.validate(rule)

    assert result.new_status is RuleStatus.DEPRECATED
    assert result.deprecated is True
    assert result.score == pytest.approx(0.2)
    assert "deprecated" in result.reason


# ── Marginal: middle-score → stays CANDIDATE ─────────────────────────────


def test_marginal_rule_stays_candidate():
    cases = [_mk_case(f"train_{i:03d}") for i in range(10)]
    # 5 of 10 correct — score 0.5, between 0.4 and 0.7
    correctness = {c.case_id: i < 5 for i, c in enumerate(cases)}

    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=10)
    rule = _mk_rule()
    result = gate.validate(rule)

    assert result.new_status is RuleStatus.CANDIDATE
    assert result.deferred is True
    assert result.score == pytest.approx(0.5)
    assert "candidate" in result.reason


# ── Sparse held-out pool: defer without running ──────────────────────────


def test_too_few_matching_cases_defers_without_calling_runner():
    # Only 2 matching cases — below min_total_for_evaluation (3)
    cases = [_mk_case(f"train_{i:03d}") for i in range(2)]

    runner = ScriptedRunner({}, assert_injection=False)
    gate = GeneralizationGate(cases, runner, k=10)
    rule = _mk_rule()
    result = gate.validate(rule)

    assert result.new_status is RuleStatus.CANDIDATE
    assert result.total_trials == 2
    assert len(runner.calls) == 0  # runner never called — no cost incurred
    assert "deferred" in result.reason.lower()


def test_no_matching_cases_defers():
    cases = [_mk_case(f"train_{i:03d}") for i in range(5)]
    runner = ScriptedRunner({}, assert_injection=False)
    gate = GeneralizationGate(cases, runner, k=10)
    # Rule triggers on a different CPT family than the held-out cases
    rule = _mk_rule(trigger=RuleTrigger(
        cpt_families=[CPTFamily.PSYCHOTHERAPY],
    ))
    result = gate.validate(rule)
    assert result.new_status is RuleStatus.CANDIDATE
    assert result.total_trials == 0
    assert len(runner.calls) == 0


# ── K cap: never evaluate more than `k` cases ────────────────────────────


def test_k_caps_number_of_evaluated_cases():
    cases = [_mk_case(f"train_{i:03d}") for i in range(20)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=7)
    rule = _mk_rule()
    result = gate.validate(rule)
    assert result.total_trials == 7
    assert len(runner.calls) == 7


# ── Memoization: repeat calls hit the cache ──────────────────────────────


def test_memoization_avoids_redundant_runner_calls():
    cases = [_mk_case(f"train_{i:03d}") for i in range(6)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=6)
    rule = _mk_rule()

    gate.validate(rule)
    n_after_first = len(runner.calls)
    assert n_after_first == 6

    # Re-validate the same rule (or an identical one) — cache should hit
    gate.validate(rule)
    n_after_second = len(runner.calls)
    assert n_after_second == n_after_first  # no new calls


def test_memoization_keyed_by_content_not_rule_id():
    """Two rules with different rule_ids but identical action / trigger /
    citation should share the cache. A proposer re-run produces new
    rule_ids but the same semantic content — we don't want to
    re-evaluate every time."""
    cases = [_mk_case(f"train_{i:03d}") for i in range(6)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=6)

    rule_a = _mk_rule()
    rule_b = _mk_rule()  # new rule_id but identical content
    assert rule_a.rule_id != rule_b.rule_id
    assert _rule_content_hash(rule_a) == _rule_content_hash(rule_b)

    gate.validate(rule_a)
    assert len(runner.calls) == 6
    gate.validate(rule_b)
    assert len(runner.calls) == 6  # cache hit


def test_memoization_keyed_differently_for_semantically_different_rules():
    cases = [_mk_case(f"train_{i:03d}") for i in range(6)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=6)

    rule_a = _mk_rule(action="Require six weeks PT.")
    rule_b = _mk_rule(action="Require twelve weeks PT.")  # different action
    assert _rule_content_hash(rule_a) != _rule_content_hash(rule_b)

    gate.validate(rule_a)
    gate.validate(rule_b)
    assert len(runner.calls) == 12


# ── Thresholds are configurable ──────────────────────────────────────────


def test_custom_thresholds_shift_promotion_band():
    """A score of 0.65 promoted under a 0.6 threshold, held under 0.7."""
    cases = [_mk_case(f"train_{i:03d}") for i in range(20)]
    # 13 of 20 correct = 0.65
    correctness = {c.case_id: i < 13 for i, c in enumerate(cases)}
    runner = ScriptedRunner(correctness)

    strict_gate = GeneralizationGate(cases, runner, k=20)
    res_strict = strict_gate.validate(_mk_rule())
    assert res_strict.new_status is RuleStatus.CANDIDATE

    # Fresh runner so calls are counted independently
    loose_runner = ScriptedRunner(correctness)
    loose_gate = GeneralizationGate(
        cases, loose_runner, k=20,
        thresholds=GateThresholds(
            promote_score=0.6,
            deprecate_score=0.3,
        ),
    )
    res_loose = loose_gate.validate(_mk_rule())
    assert res_loose.new_status is RuleStatus.ACTIVE


# ── Injection isolation: gate injects ONLY the one rule ─────────────────


def test_gate_injects_only_the_candidate_rule():
    """The whole point of the gate is to measure a single rule's
    signal in isolation. The injected system_extra must contain the
    candidate's rule_id and NO others."""
    cases = [_mk_case(f"train_{i:03d}") for i in range(5)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=5)
    rule = _mk_rule()
    gate.validate(rule)

    for _, system_extra in runner.calls:
        assert rule.rule_id in system_extra
        # Compact mode header (default in build_rule_checklist)
        assert "REASONING MEMORY" in system_extra


# ── Gate does not mutate the rule.status ────────────────────────────────


def test_gate_does_not_mutate_rule_status():
    cases = [_mk_case(f"train_{i:03d}") for i in range(6)]
    correctness = {c.case_id: True for c in cases}
    runner = ScriptedRunner(correctness)
    gate = GeneralizationGate(cases, runner, k=6)

    rule = _mk_rule(status=RuleStatus.CANDIDATE)
    result = gate.validate(rule)
    assert result.new_status is RuleStatus.ACTIVE
    # The rule's own status stays CANDIDATE — the caller persists the
    # decision explicitly via store.set_status().
    assert rule.status is RuleStatus.CANDIDATE
