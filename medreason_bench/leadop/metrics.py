"""Metrics for the lead-op retrospective harness.

Two metrics per the approved plan:

- Top-1 direction-hit rate: fraction of decision points at which the
  agent's rank-1 direction matches the team's actual chosen direction.
- Cycle-waste delta: of compounds the team submitted in round T+1 that
  failed at the next assay gate, what fraction did the agent rank below
  its top-K at decision point T (i.e., correctly deprioritized).
  Reported as the difference between two configs (memory − baseline).

A small bootstrap-CI helper is included so the +3pp / CI-lower-bound-
above-zero gate can be checked without pulling in pandas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .harness import LeadOpDecision
from .schema import CompoundRow, DecisionPointRow


@dataclass
class DirectionHitResult:
    hits: int
    total: int

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def top1_direction_hit(
    decisions: list[LeadOpDecision],
    dps: list[DecisionPointRow],
) -> DirectionHitResult:
    by_id = {d.decision_point_id: d for d in decisions}
    hits = 0
    total = 0
    for dp in dps:
        if dp.team_direction_chosen is None:
            continue
        dec = by_id.get(dp.decision_point_id)
        if dec is None:
            continue
        total += 1
        if dec.direction_ranking[0] == dp.team_direction_chosen:
            hits += 1
    return DirectionHitResult(hits=hits, total=total)


@dataclass
class CycleWasteResult:
    correctly_deprioritized: int
    total_team_failures: int

    @property
    def rate(self) -> float:
        return (
            self.correctly_deprioritized / self.total_team_failures
            if self.total_team_failures
            else 0.0
        )


def cycle_waste(
    decisions: list[LeadOpDecision],
    dps: list[DecisionPointRow],
    compounds: list[CompoundRow],
    *,
    top_k: int = 3,
) -> CycleWasteResult:
    """A compound is 'correctly deprioritized' if it failed AND the agent
    ranked it below top-K at the DP whose round_index == its round."""
    by_dp = {d.decision_point_id: d for d in decisions}
    dp_by_round = {dp.round_index: dp.decision_point_id for dp in dps}

    correct = 0
    failed = 0
    for c in compounds:
        if c.outcome_label is None or not c.outcome_label.startswith("failed"):
            continue
        dp_id = dp_by_round.get(c.round_index)
        if dp_id is None:
            continue
        dec = by_dp.get(dp_id)
        if dec is None:
            continue
        failed += 1
        ranking = dec.compound_ranking
        # "Below top-K" = either rank >= top_k (0-indexed) or absent.
        if c.compound_id not in ranking:
            correct += 1
        else:
            idx = ranking.index(c.compound_id)
            if idx >= top_k:
                correct += 1
    return CycleWasteResult(correctly_deprioritized=correct, total_team_failures=failed)


def bootstrap_ci(
    values: list[float],
    *,
    iters: int = 2000,
    alpha: float = 0.10,  # 90% CI -> alpha=0.10, i.e., 5th and 95th percentiles
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * iters)
    hi_idx = int((1 - alpha / 2) * iters) - 1
    return (means[lo_idx], means[hi_idx])
