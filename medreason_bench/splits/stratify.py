"""Stratified train/dev/test split.

Design:
- Group cases by a stratification key (outcome, difficulty) by default.
  At Phase 6 scale we'll also stratify on payer × cpt_family, but for 50
  cases the (outcome, difficulty) cross is the highest-signal partition.
- Within each stratum, shuffle deterministically (seeded) then slice by
  ratio. Integer allocation uses largest-remainder to avoid losing cases.
- Strata too small to split cleanly (fewer cases than the number of
  splits) go entirely to train, because training is where we need
  diversity and test/dev under-population hurts eval stats more than
  training loses.
- The returned mapping has a STABLE ordering of cases within each split:
  sorted by case_id. The manifest writer depends on this to produce
  byte-identical outputs across re-runs.

A 60/20/20 ratio is the default. Eval harness bootstrap CIs on ~10
test cases (20% of 50) are very wide — this is expected at Phase 3
scale and is the primary reason Phase 6 must scale to 500+ before any
claim of "memory beats zero-shot".
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from medreason.ontology import BenchmarkCase


MANIFEST_SPLITS: tuple[str, ...] = ("train", "dev", "test")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.6
    dev: float = 0.2
    test: float = 0.2

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.train, self.dev, self.test)

    def __post_init__(self):
        total = self.train + self.dev + self.test
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"SplitRatios must sum to 1.0, got {total:.4f}")
        for name, v in (("train", self.train), ("dev", self.dev), ("test", self.test)):
            if v < 0:
                raise ValueError(f"SplitRatios.{name} must be >= 0, got {v}")


def _default_strat_key(case: BenchmarkCase) -> tuple[str, str]:
    return (case.ground_truth_outcome.value, case.difficulty.value)


def _largest_remainder(target_total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    """Integer allocation of `target_total` across bins by `ratios`.

    Uses the largest-remainder method so sum(bins) == target_total always.
    Priority for the remainder goes to bins with the largest fractional
    part, with a stable tiebreaker by bin index (earlier wins). This
    guarantees the same (total, ratios) always produces the same bin
    sizes, which the manifest writer depends on.
    """
    raw = [target_total * r for r in ratios]
    floors = [int(x) for x in raw]
    remainders = [(raw[i] - floors[i], i) for i in range(len(ratios))]
    leftover = target_total - sum(floors)
    # Sort by remainder desc, index asc (stable tiebreaker)
    remainders.sort(key=lambda pair: (-pair[0], pair[1]))
    for k in range(leftover):
        floors[remainders[k][1]] += 1
    return tuple(floors)


def stratify(
    cases: list[BenchmarkCase],
    *,
    ratios: SplitRatios | None = None,
    seed: int = 42,
    key_fn: Callable[[BenchmarkCase], tuple] | None = None,
) -> dict[str, list[BenchmarkCase]]:
    """Return a {"train": [...], "dev": [...], "test": [...]} mapping.

    Cases within each split are sorted by case_id for determinism. The
    seed governs intra-stratum shuffling only — the stratum assignment
    itself is deterministic by case.
    """
    ratios = ratios or SplitRatios()
    key_fn = key_fn or _default_strat_key

    # Group deterministically by key_fn. dict insertion order is stable
    # in Python 3.7+, and we further sort by stratum key to be explicit.
    strata: dict[tuple, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        strata[key_fn(case)].append(case)

    out: dict[str, list[BenchmarkCase]] = {s: [] for s in MANIFEST_SPLITS}

    for stratum_key in sorted(strata.keys()):
        bucket = sorted(strata[stratum_key], key=lambda c: c.case_id)
        n = len(bucket)
        if n < len(MANIFEST_SPLITS):
            # Too small to split — dump everything in train.
            out["train"].extend(bucket)
            continue

        # Per-stratum deterministic shuffle
        rng = random.Random(hash((seed, stratum_key)) & 0xFFFFFFFF)
        shuffled = list(bucket)
        rng.shuffle(shuffled)

        sizes = _largest_remainder(n, ratios.as_tuple())
        i_train, i_dev, _ = sizes
        out["train"].extend(shuffled[:i_train])
        out["dev"].extend(shuffled[i_train:i_train + i_dev])
        out["test"].extend(shuffled[i_train + i_dev:])

    # Stable per-split ordering for the manifest writer.
    for split in out:
        out[split].sort(key=lambda c: c.case_id)

    return out
