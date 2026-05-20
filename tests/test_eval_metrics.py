"""Tests for medreason_bench.eval.metrics — Phase 4."""

from __future__ import annotations

import pytest

from medreason.ontology import (
    AgentResult, BenchmarkCase, Difficulty, FacilityType, Outcome,
    Payer, PriorAuthTaskConfig,
)
from medreason_bench.eval.metrics import (
    EvalMetrics,
    accuracy,
    avg_token_counts,
    brier_score,
    compute_metrics,
    cost_per_case,
    ece,
    latency_percentiles,
    macro_f1,
    per_class_f1,
    per_stratum,
    total_cost_usd,
)


def _case(case_id: str, gold: Outcome, difficulty: Difficulty = Difficulty.EASY) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE,
            cpt_code="72148",
            icd10_codes=["M54.5"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes="x",
        policy_excerpt="y",
        ground_truth_outcome=gold,
        difficulty=difficulty,
    )


def _result(
    case_id: str,
    pred: Outcome,
    *,
    confidence: float = 0.8,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.001,
    latency_ms: float = 300,
    correct: bool | None = None,
) -> AgentResult:
    return AgentResult(
        case_id=case_id,
        determination=pred,
        confidence=confidence,
        correct=correct if correct is not None else True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


# ── accuracy ────────────────────────────────────────────────────────────────


def test_accuracy_empty_returns_zero():
    assert accuracy([]) == 0.0


def test_accuracy_all_correct():
    rs = [_result(f"c{i}", Outcome.APPROVED, correct=True) for i in range(5)]
    assert accuracy(rs) == 1.0


def test_accuracy_mixed():
    rs = [
        _result("c1", Outcome.APPROVED, correct=True),
        _result("c2", Outcome.APPROVED, correct=False),
        _result("c3", Outcome.APPROVED, correct=True),
    ]
    assert accuracy(rs) == pytest.approx(2 / 3)


# ── per_class_f1 ────────────────────────────────────────────────────────────


def test_per_class_f1_perfect():
    cases = {
        "c1": _case("c1", Outcome.APPROVED),
        "c2": _case("c2", Outcome.DENIED),
        "c3": _case("c3", Outcome.OVERTURNED_ON_APPEAL),
    }
    rs = [
        _result("c1", Outcome.APPROVED, correct=True),
        _result("c2", Outcome.DENIED, correct=True),
        _result("c3", Outcome.OVERTURNED_ON_APPEAL, correct=True),
    ]
    f1s = per_class_f1(rs, cases)
    assert f1s["approved"] == 1.0
    assert f1s["denied"] == 1.0
    assert f1s["overturned_on_appeal"] == 1.0
    assert macro_f1(rs, cases) == 1.0


def test_per_class_f1_handles_class_with_no_observations():
    """A class with no ground truth AND no predictions should return
    F1 = 0.0, not crash. This is the 'overturned never appears in
    eval' case at Phase 4 scale."""
    cases = {
        "c1": _case("c1", Outcome.APPROVED),
        "c2": _case("c2", Outcome.DENIED),
    }
    rs = [
        _result("c1", Outcome.APPROVED, correct=True),
        _result("c2", Outcome.DENIED, correct=True),
    ]
    f1s = per_class_f1(rs, cases)
    # Zero-observation class must be present in the output
    assert "overturned_on_appeal" in f1s
    assert f1s["overturned_on_appeal"] == 0.0


def test_per_class_f1_false_positive_penalizes_precision():
    cases = {
        "c1": _case("c1", Outcome.APPROVED),
        "c2": _case("c2", Outcome.DENIED),
    }
    # Predict APPROVED on both — TP for c1, FP for c2
    rs = [
        _result("c1", Outcome.APPROVED, correct=True),
        _result("c2", Outcome.APPROVED, correct=False),
    ]
    f1s = per_class_f1(rs, cases)
    # precision = 1/2, recall = 1/1, F1 = 2*0.5*1/(1.5) = 0.667
    assert f1s["approved"] == pytest.approx(2 / 3)
    # denied: no TP, no FP, one FN → F1 = 0
    assert f1s["denied"] == 0.0


def test_macro_f1_empty_results():
    assert macro_f1([], {}) == 0.0


# ── brier_score ─────────────────────────────────────────────────────────────


def test_brier_perfect_confident_prediction_is_zero():
    """conf=1.0 on correct class → Brier contribution is 0."""
    cases = {"c1": _case("c1", Outcome.APPROVED)}
    rs = [_result("c1", Outcome.APPROVED, confidence=1.0, correct=True)]
    assert brier_score(rs, cases) == pytest.approx(0.0)


def test_brier_worst_confident_wrong_prediction():
    """conf=1.0 on wrong class.
    Predicted distribution puts mass 1.0 on wrong class and 0 on others.
    Gold one-hot puts 1.0 on correct class. Per-class squared errors
    sum to 1 + 1 + 0 = 2.
    """
    cases = {"c1": _case("c1", Outcome.DENIED)}
    rs = [_result("c1", Outcome.APPROVED, confidence=1.0, correct=False)]
    assert brier_score(rs, cases) == pytest.approx(2.0)


def test_brier_empty():
    assert brier_score([], {}) == 0.0


# ── ECE ─────────────────────────────────────────────────────────────────────


def test_ece_perfect_calibration_is_zero():
    """If every case at confidence c is correct with rate c, ECE = 0."""
    cases = {}
    rs = []
    # 10 cases at conf 0.9, 9 correct, 1 wrong → bin 9 has acc=0.9
    for i in range(10):
        cases[f"c{i}"] = _case(f"c{i}", Outcome.APPROVED)
        rs.append(_result(
            f"c{i}", Outcome.APPROVED if i < 9 else Outcome.DENIED,
            confidence=0.9,
            correct=(i < 9),
        ))
    assert ece(rs, cases, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_max_miscalibration():
    """If everyone says conf 1.0 but everyone is wrong, ECE = 1.0."""
    cases = {}
    rs = []
    for i in range(5):
        cases[f"c{i}"] = _case(f"c{i}", Outcome.APPROVED)
        rs.append(_result(f"c{i}", Outcome.DENIED, confidence=1.0, correct=False))
    assert ece(rs, cases, n_bins=10) == pytest.approx(1.0)


def test_ece_empty():
    assert ece([], {}) == 0.0


# ── Token / latency / cost ──────────────────────────────────────────────────


def test_avg_token_counts():
    rs = [
        _result("c1", Outcome.APPROVED, input_tokens=400, output_tokens=100),
        _result("c2", Outcome.APPROVED, input_tokens=600, output_tokens=200),
    ]
    inp, out, total = avg_token_counts(rs)
    assert inp == 500
    assert out == 150
    assert total == 650


def test_avg_token_counts_empty():
    assert avg_token_counts([]) == (0.0, 0.0, 0.0)


def test_latency_percentiles():
    # 10 latencies spaced linearly from 100ms to 1000ms
    rs = [
        _result(f"c{i}", Outcome.APPROVED, latency_ms=100 + i * 100)
        for i in range(10)
    ]
    p50, p95 = latency_percentiles(rs)
    # p50 of [100..1000] linear interp → 550 (midpoint between 5th and 6th)
    assert 500 <= p50 <= 600
    assert p95 >= 900


def test_latency_percentiles_single_value():
    rs = [_result("c1", Outcome.APPROVED, latency_ms=123.0)]
    assert latency_percentiles(rs) == (123.0, 123.0)


def test_latency_percentiles_empty():
    assert latency_percentiles([]) == (0.0, 0.0)


def test_total_and_per_case_cost():
    rs = [
        _result("c1", Outcome.APPROVED, cost_usd=0.01),
        _result("c2", Outcome.APPROVED, cost_usd=0.03),
    ]
    assert total_cost_usd(rs) == pytest.approx(0.04)
    assert cost_per_case(rs) == pytest.approx(0.02)
    assert cost_per_case([]) == 0.0


# ── per_stratum ─────────────────────────────────────────────────────────────


def test_per_stratum_groups_by_difficulty():
    cases = {
        "c1": _case("c1", Outcome.APPROVED, Difficulty.EASY),
        "c2": _case("c2", Outcome.APPROVED, Difficulty.EASY),
        "c3": _case("c3", Outcome.APPROVED, Difficulty.HARD),
    }
    rs = [
        _result("c1", Outcome.APPROVED, correct=True),
        _result("c2", Outcome.APPROVED, correct=False),
        _result("c3", Outcome.APPROVED, correct=True),
    ]
    groups = per_stratum(rs, cases, key_fn=lambda c: c.difficulty.value)
    assert groups["easy"]["n"] == 2
    assert groups["easy"]["accuracy"] == 0.5
    assert groups["hard"]["n"] == 1
    assert groups["hard"]["accuracy"] == 1.0


# ── compute_metrics bundle ──────────────────────────────────────────────────


def test_compute_metrics_assembles_full_payload():
    cases = {
        "c1": _case("c1", Outcome.APPROVED, Difficulty.EASY),
        "c2": _case("c2", Outcome.DENIED, Difficulty.MEDIUM),
    }
    rs = [
        _result("c1", Outcome.APPROVED, confidence=0.9,
                correct=True, input_tokens=500, output_tokens=100,
                cost_usd=0.002, latency_ms=400),
        _result("c2", Outcome.DENIED, confidence=0.8,
                correct=True, input_tokens=450, output_tokens=80,
                cost_usd=0.0018, latency_ms=350),
    ]
    m = compute_metrics(rs, cases)
    assert isinstance(m, EvalMetrics)
    assert m.n_results == 2
    assert m.accuracy == 1.0
    assert m.macro_f1 > 0.0
    assert m.avg_total_tokens == 565
    assert m.cost_per_case_usd == pytest.approx((0.002 + 0.0018) / 2)
    assert m.pattern_utilization is None  # zero-shot, no rules retrieved
    d = m.to_dict()
    assert d["n_results"] == 2
    assert "per_difficulty" in d
    assert "per_outcome" in d
