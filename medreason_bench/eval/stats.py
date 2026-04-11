"""Statistical routines for eval — bootstrap CIs and McNemar's paired test.

Implemented directly on top of stdlib so the install surface stays small.
If we later need extras (BCa bootstrap, stratified resampling, exact
Fisher's test), a scipy dependency can be added without changing the
public functions here.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence


# ── Bootstrap CIs ────────────────────────────────────────────────────────────


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`.

    Returns (point_estimate_mean, ci_low, ci_high).

    Empty input returns (0.0, 0.0, 0.0). Single-point input returns
    (x, x, x) — bootstrap has no signal there. Bootstrap is seeded for
    reproducibility so eval runs are bit-stable across re-runs.

    For a binary `correct` array, this is an accuracy CI. For per-case
    mean-correctness (averaged over seeds), this collapses case-level
    variance into a single point per case before bootstrapping, which
    is the "honest" CI for how the system would do on a fresh case
    draw from the same distribution.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = sum(values) / n
    if n == 1 or n_resamples <= 0:
        return (point, point, point)

    rng = random.Random(seed)
    means: list[float] = []
    vals = list(values)
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)

    means.sort()
    alpha = (1.0 - ci) / 2.0
    low_idx = max(0, int(math.floor(n_resamples * alpha)))
    high_idx = min(n_resamples - 1, int(math.ceil(n_resamples * (1.0 - alpha))) - 1)
    return (point, means[low_idx], means[high_idx])


def bootstrap_metric(
    results: Sequence,
    metric_fn: Callable[[Sequence], float],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for an arbitrary metric over a sequence.

    Resamples at the sequence-item level with replacement, computes
    `metric_fn` on each sample, and returns (point, ci_low, ci_high).

    Use this for metrics that aren't just a mean — e.g., macro F1,
    Brier, ECE — where resampling the underlying AgentResults and
    re-computing the full metric is the right thing.
    """
    n = len(results)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = metric_fn(list(results))
    if n == 1 or n_resamples <= 0:
        return (point, point, point)

    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(n_resamples):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        scores.append(metric_fn(sample))

    scores.sort()
    alpha = (1.0 - ci) / 2.0
    low_idx = max(0, int(math.floor(n_resamples * alpha)))
    high_idx = min(n_resamples - 1, int(math.ceil(n_resamples * (1.0 - alpha))) - 1)
    return (point, scores[low_idx], scores[high_idx])


# ── McNemar's paired test ────────────────────────────────────────────────────


def _count_discordant(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
) -> tuple[int, int]:
    """Count discordant cells for a paired McNemar table.

    b = # times A was correct and B was wrong.
    c = # times A was wrong and B was correct.

    Concordant cells (both correct / both wrong) don't carry any
    signal for a paired test and are discarded.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(
            f"Paired inputs must be equal length, got {len(a_correct)} vs "
            f"{len(b_correct)}"
        )
    b = 0
    c = 0
    for ai, bi in zip(a_correct, b_correct):
        if ai and not bi:
            b += 1
        elif not ai and bi:
            c += 1
    return b, c


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via the binomial distribution.

    Null: under no difference between methods, the discordant count `k`
    among `n = b + c` discordant pairs follows Binomial(n, 0.5). We
    compute the two-sided tail probability at min(b, c), capped at 1.0.

    Use this for small samples (n < 25). For larger samples,
    mcnemar_chi2 is both accurate and cheaper.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cum)


def mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    """McNemar's test with Yates continuity correction.

    chi² = (|b - c| - 1)² / (b + c),  df = 1

    P-value is the 1-df chi-squared survival function, computed directly
    from math.erfc (no scipy). Returns (statistic, p_value).

    For n = b + c < 25, prefer mcnemar_exact — the chi² approximation
    is sketchy at low counts and continuity correction over-corrects.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    stat = max(0.0, (abs(b - c) - 1) ** 2 / n)
    # P(X > x) for X ~ chi²(1) = erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return (stat, p)


def mcnemar_test(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    *,
    small_sample_threshold: int = 25,
) -> tuple[float, str]:
    """High-level McNemar dispatch.

    Returns (p_value, method) where method is "exact" or "chi2".
    Selects exact for small samples, chi² (with continuity correction)
    otherwise. The correctness arrays must be aligned pairs — a[i] and
    b[i] are two predictions on the same case.
    """
    b, c = _count_discordant(a_correct, b_correct)
    n = b + c
    if n < small_sample_threshold:
        return (mcnemar_exact(b, c), "exact")
    _, p = mcnemar_chi2(b, c)
    return (p, "chi2")
