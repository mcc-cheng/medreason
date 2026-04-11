"""Quarantine + revival policy for ReasoningRule posteriors.

Scope: Phase 5 simple threshold-based quarantine. No hysteresis state
machine yet — the plan acknowledges hysteresis as a "subtle trap"
worth revisiting once we have real posterior dynamics to tune against.
Phase 5j ships a clean threshold check; Phase 6+ can extend with
N-consecutive-failure tracking if we see thrashing in real runs.

Thresholds are defaults, overridable per call:

- `quarantine_tau = 0.55`: an ACTIVE rule whose posterior_mean drops
  below this is moved to QUARANTINED.
- `min_trials_for_quarantine = 8`: we require at least this many trials
  before quarantining to avoid punishing new rules that happen to miss
  their first few cases.
- `revival_tau = 0.70`: a QUARANTINED rule whose posterior_mean climbs
  back above this (via update_posteriors from later cases that actually
  landed correctly) is moved to ACTIVE again.
- `min_trials_for_revival = 5`: minimum trials accumulated since
  quarantine — enforced via the quarantine_count being > 0 AND
  trials >= this bound.

Each policy function returns the list of rule_ids it moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ontology import RuleStatus
from .store import RuleStore


@dataclass(frozen=True)
class PosteriorThresholds:
    quarantine_tau: float = 0.55
    min_trials_for_quarantine: int = 8
    revival_tau: float = 0.70
    min_trials_for_revival: int = 5


DEFAULT_THRESHOLDS = PosteriorThresholds()


def apply_quarantine_policy(
    store: RuleStore,
    *,
    thresholds: Optional[PosteriorThresholds] = None,
    dry_run: bool = False,
) -> list[str]:
    """Scan ACTIVE rules and quarantine those whose posterior dropped.

    A rule is quarantined when:
      - its status is currently ACTIVE, AND
      - its posterior_mean < thresholds.quarantine_tau, AND
      - its trials >= thresholds.min_trials_for_quarantine.

    The third condition is critical: without a minimum-trials gate, a
    brand-new rule with zero trials has posterior_mean = 0.5 (uniform
    prior) and would trip the default tau of 0.55. We only quarantine
    when we have real evidence that the rule is underperforming.

    When a rule is quarantined:
      - status → RuleStatus.QUARANTINED
      - quarantine_count += 1 (persisted via full rule.put after
        mutation so the column is updated)

    Returns the list of rule_ids that were moved. Empty list if nothing
    met the criteria.

    `dry_run=True` returns the list of rule_ids that WOULD be
    quarantined but leaves the store untouched — useful for cost
    sanity checks.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    active_rules = store.list_by_status(RuleStatus.ACTIVE)

    moved: list[str] = []
    for rule in active_rules:
        if rule.trials < t.min_trials_for_quarantine:
            continue
        if rule.posterior_mean >= t.quarantine_tau:
            continue
        moved.append(rule.rule_id)
        if dry_run:
            continue
        # Persist the mutation. The RuleStore.put() path writes both
        # the JSON blob and the counter columns; we bump quarantine_count
        # on the loaded object and write it back as-is.
        rule.status = RuleStatus.QUARANTINED
        rule.quarantine_count += 1
        store.put(rule)
    return moved


def apply_revival_policy(
    store: RuleStore,
    *,
    thresholds: Optional[PosteriorThresholds] = None,
    dry_run: bool = False,
) -> list[str]:
    """Scan QUARANTINED rules and revive those whose posterior recovered.

    A rule is revived when:
      - its status is currently QUARANTINED, AND
      - its posterior_mean >= thresholds.revival_tau, AND
      - its trials >= thresholds.min_trials_for_revival.

    Revival is silent — we don't reset quarantine_count, so the
    lifecycle history is preserved for later analysis (a rule that
    has been quarantined multiple times has a different risk profile
    than a first-time quarantine).

    When a rule is revived:
      - status → RuleStatus.ACTIVE

    Returns the list of rule_ids that were moved.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    quarantined = store.list_by_status(RuleStatus.QUARANTINED)

    moved: list[str] = []
    for rule in quarantined:
        if rule.trials < t.min_trials_for_revival:
            continue
        if rule.posterior_mean < t.revival_tau:
            continue
        moved.append(rule.rule_id)
        if dry_run:
            continue
        rule.status = RuleStatus.ACTIVE
        store.put(rule)
    return moved
