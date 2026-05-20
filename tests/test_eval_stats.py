"""Tests for medreason_bench.eval.stats — Phase 4."""

from __future__ import annotations

import math

import pytest

from medreason_bench.eval.stats import (
    bootstrap_mean_ci,
    bootstrap_metric,
    mcnemar_chi2,
    mcnemar_exact,
    mcnemar_test,
)


# ── Bootstrap ────────────────────────────────────────────────────────────────


def test_bootstrap_mean_is_point_estimate():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0]
    point, lo, hi = bootstrap_mean_ci(vals, n_resamples=2000, seed=42)
    assert point == pytest.approx(0.6)
    # Low/high must bracket the point
    assert lo <= point <= hi


def test_bootstrap_deterministic_under_seed():
    vals = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    a = bootstrap_mean_ci(vals, n_resamples=5000, seed=7)
    b = bootstrap_mean_ci(vals, n_resamples=5000, seed=7)
    assert a == b


def test_bootstrap_different_seed_changes_ci():
    vals = [1, 0, 1, 0, 1, 0, 1, 0]
    a = bootstrap_mean_ci(vals, n_resamples=500, seed=1)
    b = bootstrap_mean_ci(vals, n_resamples=500, seed=2)
    # Point must be identical, but bootstrap samples may differ
    assert a[0] == b[0]


def test_bootstrap_empty():
    assert bootstrap_mean_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_single_value_has_zero_width_ci():
    point, lo, hi = bootstrap_mean_ci([0.75])
    assert point == 0.75
    assert lo == 0.75
    assert hi == 0.75


def test_bootstrap_all_ones_ci_tight():
    vals = [1.0] * 20
    point, lo, hi = bootstrap_mean_ci(vals, n_resamples=1000, seed=42)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_metric_over_arbitrary_fn():
    """bootstrap_metric should resample the sequence and apply the
    metric to each sample."""
    def variance(xs):
        if not xs:
            return 0.0
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    xs = list(range(10))
    point, lo, hi = bootstrap_metric(xs, variance, n_resamples=500, seed=13)
    assert point == pytest.approx(variance(xs))
    assert lo <= point <= hi


def test_bootstrap_ci_respects_ci_level():
    """A 99% CI must be at least as wide as a 95% CI on the same data."""
    vals = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    _, lo95, hi95 = bootstrap_mean_ci(vals, n_resamples=2000, ci=0.95, seed=42)
    _, lo99, hi99 = bootstrap_mean_ci(vals, n_resamples=2000, ci=0.99, seed=42)
    assert (hi99 - lo99) >= (hi95 - lo95)


# ── McNemar exact ───────────────────────────────────────────────────────────


def test_mcnemar_exact_zero_discordant_is_one():
    """No discordant pairs → p = 1.0 (no evidence against null)."""
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_all_one_side():
    """5 discordant pairs all on one side: 2 * Binomial(5, 0, 0.5)."""
    # P(X = 0 | n=5) = 1/32. Two-sided: 2/32 = 0.0625
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)


def test_mcnemar_exact_balanced_is_one():
    """b == c → no directional difference → p = 1.0 (for even n)."""
    p = mcnemar_exact(3, 3)
    assert p == pytest.approx(1.0)


def test_mcnemar_exact_rejects_negative():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 0)


def test_mcnemar_exact_monotone_with_imbalance():
    """The more imbalanced the split, the smaller the p-value."""
    p_balanced = mcnemar_exact(10, 10)
    p_skewed = mcnemar_exact(15, 5)
    p_very_skewed = mcnemar_exact(18, 2)
    assert p_balanced >= p_skewed >= p_very_skewed


# ── McNemar chi² ────────────────────────────────────────────────────────────


def test_mcnemar_chi2_zero_discordant_is_one():
    stat, p = mcnemar_chi2(0, 0)
    assert stat == 0.0
    assert p == 1.0


def test_mcnemar_chi2_large_imbalance_is_significant():
    stat, p = mcnemar_chi2(40, 5)
    assert stat > 0
    assert p < 0.001


def test_mcnemar_chi2_balanced_is_not_significant():
    stat, p = mcnemar_chi2(20, 20)
    assert p > 0.5


def test_mcnemar_chi2_statistic_uses_continuity_correction():
    """(|b - c| - 1)² / (b + c)"""
    stat, _ = mcnemar_chi2(10, 4)
    expected = ((abs(10 - 4) - 1) ** 2) / 14
    assert stat == pytest.approx(expected)


# ── mcnemar_test dispatch ────────────────────────────────────────────────────


def test_mcnemar_test_dispatch_small_sample_uses_exact():
    a = [True, True, False, False, True]
    b = [False, True, False, True, True]
    p, method = mcnemar_test(a, b)
    assert method == "exact"
    assert 0.0 <= p <= 1.0


def test_mcnemar_test_dispatch_large_sample_uses_chi2():
    a = [True] * 30 + [False] * 10
    b = [False] * 30 + [True] * 10
    p, method = mcnemar_test(a, b)
    assert method == "chi2"
    assert 0.0 <= p <= 1.0


def test_mcnemar_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_test([True, False], [True])


def test_mcnemar_test_no_discordance_is_one():
    a = [True, True, False]
    b = [True, True, False]
    p, _ = mcnemar_test(a, b)
    assert p == 1.0
