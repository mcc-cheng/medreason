"""Tests for medreason.posterior — Phase 5 Commit 3."""

from __future__ import annotations

import sqlite3

import pytest

from medreason.ontology import (
    CPTFamily,
    ICD10Chapter,
    Payer,
    ReasoningRule,
    RuleEvidence,
    RuleStatus,
    RuleTrigger,
)
from medreason.posterior import (
    DEFAULT_THRESHOLDS,
    PosteriorThresholds,
    apply_quarantine_policy,
    apply_revival_policy,
)
from medreason.store import RuleStore


def _rule(
    *,
    rule_id: str,
    status: RuleStatus = RuleStatus.ACTIVE,
    success_count: int = 0,
    failure_count: int = 0,
    quarantine_count: int = 0,
) -> ReasoningRule:
    return ReasoningRule(
        rule_id=rule_id,
        status=status,
        trigger=RuleTrigger(
            cpt_families=[CPTFamily.IMAGING_MRI],
            icd10_chapters=[ICD10Chapter.MUSCULOSKELETAL],
            payers=[Payer.AETNA],
        ),
        action="Require PT.",
        evidence=RuleEvidence(
            supporting_case_ids=["train_01"],
            source_policy_citation="CMS LCD L1 §C.1",
            proposer_model="test",
        ),
        success_count=success_count,
        failure_count=failure_count,
        quarantine_count=quarantine_count,
    )


# ── Quarantine policy ───────────────────────────────────────────────────────


def test_quarantine_moves_underperforming_active_rules(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # posterior = (s+1)/(s+f+2) = 2/12 ≈ 0.17 — well below 0.55
    store.put(_rule(rule_id="r_bad", success_count=1, failure_count=9))
    # posterior = 9/12 = 0.75 — above tau
    store.put(_rule(rule_id="r_good", success_count=8, failure_count=2))

    moved = apply_quarantine_policy(store)
    assert moved == ["r_bad"]
    assert store.get("r_bad").status is RuleStatus.QUARANTINED
    assert store.get("r_good").status is RuleStatus.ACTIVE


def test_quarantine_requires_min_trials(sqlite_conn):
    """A brand-new rule with 0 trials has posterior_mean = 0.5 (uniform
    prior) — below the 0.55 threshold — but must NOT be quarantined
    because we don't have evidence yet."""
    store = RuleStore(sqlite_conn)
    store.put(_rule(rule_id="r_new", success_count=0, failure_count=0))
    moved = apply_quarantine_policy(store)
    assert moved == []
    assert store.get("r_new").status is RuleStatus.ACTIVE


def test_quarantine_at_exact_min_trials_with_low_posterior(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # 8 trials, 2 successes → posterior (3/11) ≈ 0.27
    store.put(_rule(rule_id="r_edge", success_count=2, failure_count=6))
    moved = apply_quarantine_policy(store)
    assert moved == ["r_edge"]


def test_quarantine_just_below_min_trials_defers(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # 7 trials, all failures → posterior (1/9) ≈ 0.11 but only 7 trials
    store.put(_rule(rule_id="r_young", success_count=0, failure_count=7))
    moved = apply_quarantine_policy(store)
    assert moved == []
    assert store.get("r_young").status is RuleStatus.ACTIVE


def test_quarantine_increments_quarantine_count(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_rule(
        rule_id="r_bad", success_count=1, failure_count=9, quarantine_count=2
    ))
    apply_quarantine_policy(store)
    assert store.get("r_bad").quarantine_count == 3


def test_quarantine_ignores_candidate_and_deprecated_rules(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_rule(
        rule_id="r_cand", status=RuleStatus.CANDIDATE,
        success_count=1, failure_count=9,
    ))
    store.put(_rule(
        rule_id="r_dep", status=RuleStatus.DEPRECATED,
        success_count=1, failure_count=9,
    ))
    moved = apply_quarantine_policy(store)
    assert moved == []
    assert store.get("r_cand").status is RuleStatus.CANDIDATE
    assert store.get("r_dep").status is RuleStatus.DEPRECATED


def test_quarantine_dry_run_does_not_mutate(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_rule(rule_id="r_bad", success_count=1, failure_count=9))
    moved = apply_quarantine_policy(store, dry_run=True)
    assert moved == ["r_bad"]
    assert store.get("r_bad").status is RuleStatus.ACTIVE  # not actually moved


def test_quarantine_custom_thresholds(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # posterior 0.58 — above default 0.55 but below loose 0.65
    store.put(_rule(rule_id="r_edge", success_count=6, failure_count=4))

    assert apply_quarantine_policy(store) == []  # default passes

    strict = PosteriorThresholds(quarantine_tau=0.65, min_trials_for_quarantine=8)
    # Not enough trials either (10 vs 8 OK), posterior 7/12 ≈ 0.58 < 0.65
    moved = apply_quarantine_policy(store, thresholds=strict)
    assert "r_edge" in moved


# ── Revival policy ──────────────────────────────────────────────────────────


def test_revive_moves_quarantined_rules_that_recovered(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # Quarantined rule that climbed back: 20 trials, 16 successes
    # posterior = 17/22 ≈ 0.77 — above revival tau 0.70
    store.put(_rule(
        rule_id="r_recovered", status=RuleStatus.QUARANTINED,
        success_count=16, failure_count=4, quarantine_count=1,
    ))
    moved = apply_revival_policy(store)
    assert moved == ["r_recovered"]
    assert store.get("r_recovered").status is RuleStatus.ACTIVE
    # quarantine_count preserved for history
    assert store.get("r_recovered").quarantine_count == 1


def test_revive_skips_quarantined_rules_still_below_tau(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # posterior 5/12 ≈ 0.42 — below revival tau
    store.put(_rule(
        rule_id="r_stuck", status=RuleStatus.QUARANTINED,
        success_count=4, failure_count=6,
    ))
    moved = apply_revival_policy(store)
    assert moved == []
    assert store.get("r_stuck").status is RuleStatus.QUARANTINED


def test_revive_requires_min_trials(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # 4 trials — below default 5 revival min
    store.put(_rule(
        rule_id="r_few", status=RuleStatus.QUARANTINED,
        success_count=4, failure_count=0,
    ))
    moved = apply_revival_policy(store)
    assert moved == []


def test_revive_ignores_active_and_deprecated(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_rule(
        rule_id="r_active", status=RuleStatus.ACTIVE,
        success_count=50, failure_count=1,
    ))
    store.put(_rule(
        rule_id="r_dep", status=RuleStatus.DEPRECATED,
        success_count=50, failure_count=1,
    ))
    moved = apply_revival_policy(store)
    assert moved == []


def test_revive_dry_run_does_not_mutate(sqlite_conn):
    store = RuleStore(sqlite_conn)
    store.put(_rule(
        rule_id="r_recovered", status=RuleStatus.QUARANTINED,
        success_count=16, failure_count=4,
    ))
    moved = apply_revival_policy(store, dry_run=True)
    assert moved == ["r_recovered"]
    assert store.get("r_recovered").status is RuleStatus.QUARANTINED


def test_revive_custom_thresholds(sqlite_conn):
    store = RuleStore(sqlite_conn)
    # posterior 0.68 — below default revival 0.70 but above loose 0.60
    store.put(_rule(
        rule_id="r_edge", status=RuleStatus.QUARANTINED,
        success_count=6, failure_count=3,
    ))
    assert apply_revival_policy(store) == []

    loose = PosteriorThresholds(revival_tau=0.60, min_trials_for_revival=5)
    moved = apply_revival_policy(store, thresholds=loose)
    assert "r_edge" in moved
