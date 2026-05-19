"""Tests for medreason_bench.targetval.metrics."""

from __future__ import annotations

from medreason.targetval.case import BypassMechanism
from medreason_bench.targetval.metrics import (
    bootstrap_ci,
    bypass_precision_recall,
    top_k_target_hit,
)
from medreason_bench.targetval.synthetic import build_synthetic_targets


def test_top_k_hit_counts_successful_targets():
    cases = build_synthetic_targets()
    # BRAF and KRAS in synthetic have outcome APPROVED_LATER / PHASE2_EFFICACY_NO.
    # Only BRAF qualifies (APPROVED_LATER). KRAS NSCLC is in mapk_retro, not synthetic.
    ranking = ["BRAF", "KRAS", "EGFR"]
    hits, total = top_k_target_hit(ranking, cases, k=2)
    assert total == 1  # only BRAF is in the success set
    assert hits == 1


def test_top_k_misses_when_success_target_below_k():
    cases = build_synthetic_targets()
    ranking = ["EGFR", "KRAS", "BRAF"]
    hits, total = top_k_target_hit(ranking, cases, k=1)
    assert hits == 0
    assert total == 1


def test_bypass_precision_recall_perfect():
    cases = build_synthetic_targets()
    predicted = {
        "TVS-001": BypassMechanism.DOWNSTREAM_FEEDBACK,
        "TVS-002": BypassMechanism.DOWNSTREAM_FEEDBACK,
        "TVS-003": BypassMechanism.ALTERNATIVE_PATHWAY,
    }
    result = bypass_precision_recall(predicted, cases)
    # All 3 cases have positive ground truth, all 3 predicted positive.
    assert result.true_positive == 3
    assert result.false_negative == 0
    assert result.false_positive == 0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_bypass_precision_recall_missed():
    cases = build_synthetic_targets()
    predicted = {
        "TVS-001": BypassMechanism.NO_BYPASS_KNOWN,
        "TVS-002": BypassMechanism.NO_BYPASS_KNOWN,
        "TVS-003": BypassMechanism.NO_BYPASS_KNOWN,
    }
    result = bypass_precision_recall(predicted, cases)
    assert result.true_positive == 0
    assert result.false_negative == 3
    assert result.recall == 0.0


def test_bootstrap_ci_basic_shape():
    diffs = [0.05, 0.03, 0.10, 0.08, 0.06, 0.07, 0.04]
    lo, hi = bootstrap_ci(diffs, iters=500, alpha=0.10, seed=1)
    assert lo < hi
    # The CI should bracket the sample mean roughly
    mean = sum(diffs) / len(diffs)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty_returns_zero_zero():
    lo, hi = bootstrap_ci([], iters=100)
    assert (lo, hi) == (0.0, 0.0)
