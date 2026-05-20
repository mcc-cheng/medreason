"""Generalization gate — promote candidate rules only if they actually
generalize on held-out cases.

This is the single load-bearing quality lever the old pipeline was
missing. Without it, the pipeline degenerates to "store whatever the
proposer suggested," which is exactly the state that gave us 86%
memory vs 88% zero-shot on the pre-rework benchmark.

Algorithm (per candidate rule):

1. Find K held-out cases from the train+dev pool whose task_config
   matches the rule's trigger. These must NEVER include test-split
   cases — the caller is responsible for passing only train+dev.
2. For each held-out case, run the base AgentRunner with ONLY this
   one rule injected (no other memory, no other tier output). This
   isolates the rule's signal.
3. Score = correct / total. Promotion thresholds:
     - score >= 0.7 AND total >= 5  → RuleStatus.ACTIVE
     - score <  0.4                 → RuleStatus.DEPRECATED
     - 0.4 <= score < 0.7           → RuleStatus.CANDIDATE (revisit)
     - total < 3                    → RuleStatus.CANDIDATE (defer,
                                       not enough evidence)

Cost control — the gate can easily blow the budget at scale:

- Memoized by (rule_content_hash, case_id) so repeated calls in the
  same training epoch don't re-run the same evaluation. The cache is
  stored on the GenerationGate instance.
- The plan risk #3 calls out that N rules × K=10 cases × $0.01/call
  ≈ $10/epoch at 100 rules. Memoization becomes critical at scale;
  without it, 1000 rules × 3 epochs × 10 cases = 30000 calls.

Design note: the gate uses the BASE runner (zero-shot), not the
memory-wrapped runner. Wrapping would give the candidate an unfair
boost from other active rules bleeding into its prediction via
retrieval. The isolation is the whole point.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from medreason.ontology import (
    AgentResult,
    BenchmarkCase,
    ReasoningRule,
    RuleStatus,
)
from medreason.runners.base import AgentRunner

from ..retrieval.injector import build_rule_checklist


# ── Thresholds ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateThresholds:
    """Phase 5 gate thresholds — tunable but anchored in the plan."""
    promote_score: float = 0.7          # >= → ACTIVE
    deprecate_score: float = 0.4         # < → DEPRECATED
    min_total_for_promote: int = 5       # need at least this many hits
    min_total_for_evaluation: int = 3    # below → defer


DEFAULT_THRESHOLDS = GateThresholds()


# ── Result dataclass ────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Outcome of one candidate rule's gate evaluation."""
    rule_id: str
    new_status: RuleStatus
    score: float
    total_trials: int
    correct: int
    reason: str  # human-readable
    evaluated_case_ids: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.new_status is RuleStatus.ACTIVE

    @property
    def deferred(self) -> bool:
        return self.new_status is RuleStatus.CANDIDATE

    @property
    def deprecated(self) -> bool:
        return self.new_status is RuleStatus.DEPRECATED


# ── Memoization helpers ─────────────────────────────────────────────────────


def _rule_content_hash(rule: ReasoningRule) -> str:
    """Stable hash over the rule's user-visible content.

    Excludes rule_id (different rule_ids can share content if the
    proposer runs twice) and excludes posterior counts (two rules with
    identical action text shouldn't be re-evaluated just because one
    has more trials). Including `action`, `trigger.semantic_predicate`,
    and `source_policy_citation` is enough to uniquely identify a
    rule's semantics.
    """
    parts = [
        rule.action,
        rule.trigger.semantic_predicate or "",
        rule.evidence.source_policy_citation,
        "|".join(sorted(f.value for f in rule.trigger.cpt_families)),
        "|".join(sorted(c.value for c in rule.trigger.icd10_chapters)),
        "|".join(sorted(p.value for p in rule.trigger.payers)),
    ]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


# ── The gate ────────────────────────────────────────────────────────────────


class GeneralizationGate:
    """Stateful gate that memoizes per-(rule_hash, case_id) evaluations.

    Construct once per training pass. Feed it a `held_out_cases` pool
    (train + dev, MUST NOT include test) and a base runner. Call
    `validate()` for each candidate rule.
    """

    def __init__(
        self,
        held_out_cases: list[BenchmarkCase],
        runner: AgentRunner,
        *,
        thresholds: GateThresholds = DEFAULT_THRESHOLDS,
        k: int = 10,
        seed: int = 0,
    ):
        self._cases = list(held_out_cases)
        self._runner = runner
        self._thresholds = thresholds
        self._k = k
        self._seed = seed
        self._cache: dict[tuple[str, str], bool] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def validate(self, rule: ReasoningRule) -> GateResult:
        """Evaluate `rule` against up to `k` matching held-out cases.

        Does NOT mutate `rule.status` — the caller is expected to read
        `GateResult.new_status` and call `store.set_status(rule_id,
        new_status)` if they want to persist the decision.
        """
        matching = [c for c in self._cases if rule.trigger.matches(c.task_config)]
        matching = matching[: self._k]

        if len(matching) < self._thresholds.min_total_for_evaluation:
            return GateResult(
                rule_id=rule.rule_id,
                new_status=RuleStatus.CANDIDATE,
                score=0.0,
                total_trials=len(matching),
                correct=0,
                reason=(
                    f"deferred: only {len(matching)} matching case(s), "
                    f"need {self._thresholds.min_total_for_evaluation}"
                ),
                evaluated_case_ids=[c.case_id for c in matching],
            )

        rule_hash = _rule_content_hash(rule)
        injection = build_rule_checklist([rule])

        correct = 0
        total = 0
        evaluated: list[str] = []
        for case in matching:
            key = (rule_hash, case.case_id)
            if key in self._cache:
                is_correct = self._cache[key]
            else:
                result = self._runner.run(
                    case,
                    seed=self._seed,
                    system_extra=injection,
                )
                is_correct = bool(result.correct)
                self._cache[key] = is_correct
            if is_correct:
                correct += 1
            total += 1
            evaluated.append(case.case_id)

        score = correct / total if total > 0 else 0.0

        new_status, reason = self._score_to_status(score, total)
        return GateResult(
            rule_id=rule.rule_id,
            new_status=new_status,
            score=score,
            total_trials=total,
            correct=correct,
            reason=reason,
            evaluated_case_ids=evaluated,
        )

    def _score_to_status(
        self, score: float, total: int
    ) -> tuple[RuleStatus, str]:
        t = self._thresholds
        if score < t.deprecate_score:
            return (
                RuleStatus.DEPRECATED,
                f"deprecated: score={score:.2f} < {t.deprecate_score}",
            )
        if score >= t.promote_score and total >= t.min_total_for_promote:
            return (
                RuleStatus.ACTIVE,
                f"promoted: score={score:.2f}, n={total}",
            )
        return (
            RuleStatus.CANDIDATE,
            f"candidate: score={score:.2f}, n={total} "
            f"(need >= {t.promote_score} on >= {t.min_total_for_promote})",
        )
