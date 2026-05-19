"""Target-validation benchmark metrics.

Three primary metrics:

- top_k_target_hit: fraction of historically successful targets the swarm
  places in its top-K ranking. The analog of leadop's top1_direction_hit.
- bypass_precision_recall: precision/recall of the swarm's bypass-risk
  flagging against ground-truth bypass labels.
- bootstrap_ci: 90% percentile CI on a paired-lift statistic, identical
  formula to medreason_bench.leadop.metrics.bootstrap_ci so the gate
  semantics are consistent across domains.

These metrics drive the prediction-card gate (prediction_card.py): the
swarm + universal-layer memory must beat the swarm-alone baseline by
≥ Δ_top1 percentage points with the 90% CI lower bound strictly above 0,
on ≥ N_min eligible cases.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from medreason.targetval.case import BypassMechanism, TargetValidationCase


# ── Top-K target hit ──────────────────────────────────────────────────────────


def top_k_target_hit(
    ranking: list[str],
    cases: list[TargetValidationCase],
    *,
    k: int = 5,
    success_label_set: tuple[str, ...] = (
        "phase2_efficacy_yes",
        "approved_later",
    ),
) -> tuple[int, int]:
    """Returns (hits, total_successful_cases).

    hits = number of historically successful targets in the top-K of the
    swarm's ranking. total_successful_cases is the denominator (the
    eligible set against which top-K is measured).
    """
    success_genes = {
        c.target.gene_symbol
        for c in cases
        if c.ground_truth_outcome.value in success_label_set
    }
    if not success_genes:
        return (0, 0)
    top_k = set(ranking[: max(0, k)])
    hits = len(success_genes & top_k)
    return (hits, len(success_genes))


# ── Bypass precision / recall ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BypassPrecRec:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    n_eligible: int


def bypass_precision_recall(
    predicted_bypass: dict[str, BypassMechanism],
    cases: list[TargetValidationCase],
    *,
    positive_set: tuple[BypassMechanism, ...] = (
        BypassMechanism.PARALOG_COMPENSATION,
        BypassMechanism.DOWNSTREAM_FEEDBACK,
        BypassMechanism.ALTERNATIVE_PATHWAY,
        BypassMechanism.MICROENVIRONMENT_RESCUE,
        BypassMechanism.RESISTANCE_MUTATION,
        BypassMechanism.OTHER_VALIDATED,
    ),
) -> BypassPrecRec:
    """Precision/recall of the swarm's predicted_bypass against ground
    truth, treating any value in `positive_set` as "bypass-predicted".

    Cases with ground_truth_bypass == UNKNOWN are excluded from the
    eligible set (we don't penalise the swarm for cases we can't grade).
    """
    pos = set(positive_set)
    tp = fp = fn = 0
    eligible = 0
    for c in cases:
        if c.ground_truth_bypass is BypassMechanism.UNKNOWN:
            continue
        eligible += 1
        gt_pos = c.ground_truth_bypass in pos
        pr = predicted_bypass.get(c.case_id, BypassMechanism.UNKNOWN)
        pr_pos = pr in pos
        if gt_pos and pr_pos:
            tp += 1
        elif pr_pos and not gt_pos:
            fp += 1
        elif gt_pos and not pr_pos:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return BypassPrecRec(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        n_eligible=eligible,
    )


# ── Bootstrap CI (paired-lift) ────────────────────────────────────────────────


def bootstrap_ci(
    paired_diffs: list[float],
    *,
    iters: int = 2000,
    alpha: float = 0.10,
    seed: int = 17,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of paired_diffs.

    Identical signature to medreason_bench.leadop.metrics.bootstrap_ci
    so the prediction-card gate is consistent across domains.
    """
    if not paired_diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(paired_diffs)
    means: list[float] = []
    for _ in range(iters):
        sample = [paired_diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int(iters * (alpha / 2))
    hi_idx = int(iters * (1 - alpha / 2)) - 1
    lo_idx = max(0, min(lo_idx, iters - 1))
    hi_idx = max(0, min(hi_idx, iters - 1))
    return (means[lo_idx], means[hi_idx])
