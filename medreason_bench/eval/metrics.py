"""Eval metrics — pure functions over lists of AgentResult.

Every metric here takes `results: list[AgentResult]` (possibly narrowed
to one runner, one seed, one split) and produces a single number or a
nested dict. No metric mutates its inputs.

The per-class metrics (F1, Brier, ECE) need the full Outcome enum space
so a class with zero observations still shows up in the output — that's
what `_all_outcomes` is for. It prevents silent class-omission bugs
from hiding an over-predicted class.

The `compute_metrics()` top-level function assembles the full dashboard
payload the leaderboard builder consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional

from medreason.ontology import AgentResult, BenchmarkCase, Outcome


_all_outcomes: tuple[Outcome, ...] = (
    Outcome.APPROVED,
    Outcome.DENIED,
    Outcome.OVERTURNED_ON_APPEAL,
)


# ── Point estimates ─────────────────────────────────────────────────────────


def accuracy(results: list[AgentResult]) -> float:
    """Fraction of results where `correct` is True. 0.0 on empty input."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def per_class_f1(
    results: list[AgentResult],
    cases_by_id: dict[str, BenchmarkCase],
) -> dict[str, float]:
    """F1 for each Outcome class, keyed by the enum value string.

    Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 = 2pr/(p+r).
    When a class has no predictions AND no ground truth, F1 is defined
    as 0.0 (rather than NaN) so the leaderboard schema stays concrete.
    """
    out: dict[str, float] = {c.value: 0.0 for c in _all_outcomes}
    if not results:
        return out

    for cls in _all_outcomes:
        tp = fp = fn = 0
        for r in results:
            gold = cases_by_id[r.case_id].ground_truth_outcome
            pred = r.determination
            if pred == cls and gold == cls:
                tp += 1
            elif pred == cls and gold != cls:
                fp += 1
            elif pred != cls and gold == cls:
                fn += 1
        p_denom = tp + fp
        r_denom = tp + fn
        precision = tp / p_denom if p_denom > 0 else 0.0
        recall = tp / r_denom if r_denom > 0 else 0.0
        if precision + recall > 0:
            out[cls.value] = 2 * precision * recall / (precision + recall)
        else:
            out[cls.value] = 0.0
    return out


def macro_f1(
    results: list[AgentResult],
    cases_by_id: dict[str, BenchmarkCase],
) -> float:
    f1s = per_class_f1(results, cases_by_id)
    if not f1s:
        return 0.0
    return sum(f1s.values()) / len(f1s)


def brier_score(
    results: list[AgentResult],
    cases_by_id: dict[str, BenchmarkCase],
) -> float:
    """Multi-class Brier score averaged over results.

    For each result we treat the predicted determination as a one-hot
    distribution with mass `confidence` on the predicted class and
    `(1-confidence)/(K-1)` smeared across the other classes. This is
    the common reduction when the model doesn't emit a full class
    distribution — our agents currently don't.

    Lower is better. 0.0 is perfect; worst-case for 3 classes is ~2.
    """
    if not results:
        return 0.0
    k = len(_all_outcomes)
    total = 0.0
    for r in results:
        gold = cases_by_id[r.case_id].ground_truth_outcome
        pred_probs: dict[Outcome, float] = {}
        conf = max(0.0, min(1.0, r.confidence))
        residual = (1.0 - conf) / (k - 1) if k > 1 else 0.0
        for cls in _all_outcomes:
            pred_probs[cls] = conf if cls == r.determination else residual
        squared = 0.0
        for cls in _all_outcomes:
            target = 1.0 if cls == gold else 0.0
            squared += (pred_probs[cls] - target) ** 2
        total += squared
    return total / len(results)


def ece(
    results: list[AgentResult],
    cases_by_id: dict[str, BenchmarkCase],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Partitions [0, 1] confidence range into n_bins equal-width bins.
    For each bin, computes |empirical_accuracy - avg_confidence| weighted
    by bin population, then sums. Lower is better; 0 is perfectly
    calibrated.
    """
    if not results or n_bins < 1:
        return 0.0
    bins_correct: list[int] = [0] * n_bins
    bins_conf_sum: list[float] = [0.0] * n_bins
    bins_count: list[int] = [0] * n_bins

    for r in results:
        c = max(0.0, min(1.0, r.confidence))
        # Edge: confidence == 1.0 belongs in the top bin
        idx = min(int(c * n_bins), n_bins - 1)
        bins_count[idx] += 1
        bins_conf_sum[idx] += c
        if r.correct:
            bins_correct[idx] += 1

    n = len(results)
    err = 0.0
    for i in range(n_bins):
        if bins_count[i] == 0:
            continue
        emp_acc = bins_correct[i] / bins_count[i]
        avg_conf = bins_conf_sum[i] / bins_count[i]
        err += (bins_count[i] / n) * abs(emp_acc - avg_conf)
    return err


def avg_token_counts(results: list[AgentResult]) -> tuple[float, float, float]:
    """(avg_input, avg_output, avg_total) across results."""
    if not results:
        return (0.0, 0.0, 0.0)
    n = len(results)
    inp = sum(r.input_tokens for r in results) / n
    out = sum(r.output_tokens for r in results) / n
    return (inp, out, inp + out)


def latency_percentiles(results: list[AgentResult]) -> tuple[float, float]:
    """(p50, p95) latency in ms. Pure Python percentile, linear interpolation."""
    if not results:
        return (0.0, 0.0)
    sorted_l = sorted(r.latency_ms for r in results)
    return (_percentile(sorted_l, 50.0), _percentile(sorted_l, 95.0))


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def total_cost_usd(results: list[AgentResult]) -> float:
    return sum(r.cost_usd for r in results)


def cost_per_case(results: list[AgentResult]) -> float:
    if not results:
        return 0.0
    return total_cost_usd(results) / len(results)


# ── Pattern utilization (memory mode only) ──────────────────────────────────


def pattern_utilization(results: list[AgentResult]) -> Optional[float]:
    """Fraction of retrieved rules the agent actually applied.

    Returns None if no rule was retrieved for any result — this is the
    expected state during zero-shot eval in Phase 4. The memory wrapper
    (Phase 5i) will populate the applied_rules field and this metric
    will start returning concrete numbers.
    """
    total_retrieved = 0
    total_applied = 0
    for r in results:
        if not r.applied_rules:
            continue
        total_retrieved += len(r.applied_rules)
        total_applied += sum(1 for a in r.applied_rules if a.applied)
    if total_retrieved == 0:
        return None
    return total_applied / total_retrieved


# ── Per-stratum breakdown ────────────────────────────────────────────────────


def per_stratum(
    results: list[AgentResult],
    cases_by_id: dict[str, BenchmarkCase],
    key_fn: Callable[[BenchmarkCase], Hashable],
) -> dict[Hashable, dict[str, float]]:
    """Group results by a stratum key and compute a small metric set per group.

    The key_fn receives the full BenchmarkCase so callers can bucket by
    any field (payer, difficulty, cpt_family, outcome, ...).
    """
    groups: dict[Hashable, list[AgentResult]] = {}
    for r in results:
        case = cases_by_id.get(r.case_id)
        if case is None:
            continue
        key = key_fn(case)
        groups.setdefault(key, []).append(r)

    out: dict[Hashable, dict[str, float]] = {}
    for key, group in groups.items():
        out[key] = {
            "n": float(len(group)),
            "accuracy": accuracy(group),
            "avg_total_tokens": sum(r.input_tokens + r.output_tokens for r in group) / len(group),
            "cost_usd": total_cost_usd(group),
        }
    return out


# ── Top-level bundle ────────────────────────────────────────────────────────


@dataclass
class EvalMetrics:
    n_results: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    brier: float
    ece: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    cost_per_case_usd: float
    pattern_utilization: Optional[float]
    per_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    per_outcome: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "n_results": self.n_results,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": dict(self.per_class_f1),
            "brier": self.brier,
            "ece": self.ece,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_case_usd": self.cost_per_case_usd,
            "pattern_utilization": self.pattern_utilization,
            "per_difficulty": {str(k): v for k, v in self.per_difficulty.items()},
            "per_outcome": {str(k): v for k, v in self.per_outcome.items()},
        }
        return d


def compute_metrics(
    results: list[AgentResult],
    cases_by_id: dict[str, BenchmarkCase],
) -> EvalMetrics:
    """Assemble the full dashboard payload from flat AgentResult list."""
    inp, outp, tot = avg_token_counts(results)
    p50, p95 = latency_percentiles(results)
    return EvalMetrics(
        n_results=len(results),
        accuracy=accuracy(results),
        macro_f1=macro_f1(results, cases_by_id),
        per_class_f1=per_class_f1(results, cases_by_id),
        brier=brier_score(results, cases_by_id),
        ece=ece(results, cases_by_id),
        avg_input_tokens=inp,
        avg_output_tokens=outp,
        avg_total_tokens=tot,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_cost_usd=total_cost_usd(results),
        cost_per_case_usd=cost_per_case(results),
        pattern_utilization=pattern_utilization(results),
        per_difficulty=per_stratum(
            results, cases_by_id, lambda c: c.difficulty.value
        ),
        per_outcome=per_stratum(
            results, cases_by_id, lambda c: c.ground_truth_outcome.value
        ),
    )
