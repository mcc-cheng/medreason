"""Tests for medreason.store.rules.RuleStore — Phase 1."""

from __future__ import annotations

import pytest

from tests.conftest import make_rule


def test_rule_store_put_get_round_trip(sqlite_conn):
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    rule = make_rule(action="Check conservative therapy duration ≥6 weeks.")
    store.put(rule)
    revived = store.get(rule.rule_id)
    assert revived is not None
    assert revived.rule_id == rule.rule_id
    assert revived.action == rule.action
    assert revived.trigger.cpt_families == rule.trigger.cpt_families
    assert revived.trigger.icd10_chapters == rule.trigger.icd10_chapters
    assert revived.evidence.source_policy_citation == rule.evidence.source_policy_citation


def test_rule_store_get_missing_returns_none(sqlite_conn):
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    assert store.get("nonexistent") is None


def test_rule_store_put_is_upsert(sqlite_conn):
    """put() on an existing rule_id must replace, not duplicate."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    rule = make_rule(action="v1 action")
    store.put(rule)
    assert store.count() == 1

    rule.action = "v2 action"
    store.put(rule)
    assert store.count() == 1
    revived = store.get(rule.rule_id)
    assert revived.action == "v2 action"


def test_rule_store_list_by_status(sqlite_conn):
    from medreason.ontology import RuleStatus
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)

    active_a = make_rule(action="active A")
    active_a.status = RuleStatus.ACTIVE
    active_b = make_rule(action="active B")
    active_b.status = RuleStatus.ACTIVE
    active_b.success_count = 10  # should sort before active_a
    candidate = make_rule(action="candidate")  # default status = CANDIDATE
    deprecated = make_rule(action="deprecated")
    deprecated.status = RuleStatus.DEPRECATED

    for r in (active_a, active_b, candidate, deprecated):
        store.put(r)

    actives = store.list_by_status(RuleStatus.ACTIVE)
    assert len(actives) == 2
    # Sorted by success_count DESC
    assert actives[0].action == "active B"
    assert actives[1].action == "active A"

    assert len(store.list_by_status(RuleStatus.CANDIDATE)) == 1
    assert len(store.list_by_status(RuleStatus.DEPRECATED)) == 1
    assert store.count(RuleStatus.ACTIVE) == 2
    assert store.count() == 4


def test_rule_store_set_status_updates_both_column_and_json(sqlite_conn):
    from medreason.ontology import RuleStatus
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    rule = make_rule()
    store.put(rule)
    assert store.get(rule.rule_id).status == RuleStatus.CANDIDATE

    store.set_status(rule.rule_id, RuleStatus.ACTIVE)
    revived = store.get(rule.rule_id)
    assert revived.status == RuleStatus.ACTIVE
    # And list_by_status sees the change
    assert len(store.list_by_status(RuleStatus.ACTIVE)) == 1
    assert len(store.list_by_status(RuleStatus.CANDIDATE)) == 0


def test_rule_store_delete(sqlite_conn):
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    rule = make_rule()
    store.put(rule)
    assert store.delete(rule.rule_id) is True
    assert store.get(rule.rule_id) is None
    assert store.delete(rule.rule_id) is False  # idempotent


# ── update_posteriors semantics ───────────────────────────────────────────────


def test_update_posteriors_applied_and_correct(sqlite_conn):
    """Rule applied=True and case correct=True → success_count += 1, seen += 1."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    store.update_posteriors(
        rule_ids=[r.rule_id],
        correct=True,
        applied_map={r.rule_id: True},
    )
    fresh = store.get(r.rule_id)
    assert fresh.success_count == 1
    assert fresh.failure_count == 0
    assert fresh.seen_count == 1


def test_update_posteriors_applied_and_wrong(sqlite_conn):
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    store.update_posteriors([r.rule_id], correct=False, applied_map={r.rule_id: True})
    fresh = store.get(r.rule_id)
    assert fresh.success_count == 0
    assert fresh.failure_count == 1
    assert fresh.seen_count == 1


def test_update_posteriors_not_applied_only_increments_seen(sqlite_conn):
    """A rule that was retrieved but NOT applied must not be credited or
    punished — only seen_count bumps. This is the core of the new posterior
    contract (don't punish rules the agent ignored)."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    store.update_posteriors([r.rule_id], correct=True, applied_map={r.rule_id: False})
    fresh = store.get(r.rule_id)
    assert fresh.success_count == 0
    assert fresh.failure_count == 0
    assert fresh.seen_count == 1


def test_update_posteriors_mixed_applied_map(sqlite_conn):
    """Multiple rules in one call with heterogeneous applied flags."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r1 = make_rule(action="applied+correct")
    r2 = make_rule(action="applied+wrong")
    r3 = make_rule(action="not_applied")
    for r in (r1, r2, r3):
        store.put(r)

    store.update_posteriors(
        rule_ids=[r1.rule_id, r2.rule_id, r3.rule_id],
        correct=True,
        applied_map={r1.rule_id: True, r2.rule_id: True, r3.rule_id: False},
    )
    # Only r1 was applied+correct
    assert store.get(r1.rule_id).success_count == 1
    assert store.get(r1.rule_id).failure_count == 0
    # r2 was applied but we're passing correct=True — so it IS a success too.
    # (Test the contrapositive below.)
    assert store.get(r2.rule_id).success_count == 1
    # r3 retrieved but not applied: only seen
    assert store.get(r3.rule_id).success_count == 0
    assert store.get(r3.rule_id).failure_count == 0
    assert store.get(r3.rule_id).seen_count == 1


def test_update_posteriors_failure_contrapositive(sqlite_conn):
    """With correct=False, applied rules must increment failure_count."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r1 = make_rule(action="applied")
    r2 = make_rule(action="not applied")
    store.put(r1)
    store.put(r2)
    store.update_posteriors(
        rule_ids=[r1.rule_id, r2.rule_id],
        correct=False,
        applied_map={r1.rule_id: True, r2.rule_id: False},
    )
    assert store.get(r1.rule_id).failure_count == 1
    assert store.get(r1.rule_id).seen_count == 1
    assert store.get(r2.rule_id).failure_count == 0
    assert store.get(r2.rule_id).seen_count == 1


def test_update_posteriors_empty_is_noop(sqlite_conn):
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    store.update_posteriors([], correct=True, applied_map={})  # must not raise


def test_update_posteriors_default_applied_map_is_all_true(sqlite_conn):
    """Legacy callers without an applied_map get all-True semantics."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    store.update_posteriors([r.rule_id], correct=True)
    assert store.get(r.rule_id).success_count == 1


def test_posterior_counts_accumulate(sqlite_conn):
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    for _ in range(5):
        store.update_posteriors([r.rule_id], correct=True, applied_map={r.rule_id: True})
    for _ in range(3):
        store.update_posteriors([r.rule_id], correct=False, applied_map={r.rule_id: True})
    fresh = store.get(r.rule_id)
    assert fresh.success_count == 5
    assert fresh.failure_count == 3
    assert fresh.seen_count == 8
    assert fresh.posterior_mean == pytest.approx(6 / 10)  # Beta(6, 4)


def test_audit_log_records_writes(sqlite_conn):
    """Every put / update must leave an audit_log row."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    store.update_posteriors([r.rule_id], correct=True, applied_map={r.rule_id: True})

    rows = sqlite_conn.execute(
        "SELECT event_type, entity_id FROM audit_log ORDER BY id"
    ).fetchall()
    events = [(row["event_type"], row["entity_id"]) for row in rows]
    assert ("rule.put", r.rule_id) in events
    assert ("rule.update_posteriors", r.rule_id) in events


# ── Counter column vs JSON source-of-truth ───────────────────────────────────


def test_counter_columns_are_source_of_truth(sqlite_conn):
    """If the rule_json blob says success_count=0 but the column says 7,
    the hydrated rule must have success_count=7. Protects against
    inconsistency after update_posteriors."""
    from medreason.store import RuleStore
    store = RuleStore(sqlite_conn)
    r = make_rule()
    store.put(r)
    # Manually desync the JSON blob from the columns. In production this
    # should never happen, but if it does, the columns win.
    sqlite_conn.execute(
        "UPDATE rules SET success_count = 7, failure_count = 2, seen_count = 9 "
        "WHERE rule_id = ?",
        (r.rule_id,),
    )
    sqlite_conn.commit()
    fresh = store.get(r.rule_id)
    assert fresh.success_count == 7
    assert fresh.failure_count == 2
    assert fresh.seen_count == 9
    assert fresh.posterior_mean == pytest.approx(8 / 11)


# ── TraceStore smoke ──────────────────────────────────────────────────────────


def test_trace_store_round_trip(sqlite_conn):
    from medreason.ontology import Outcome
    from medreason.store import TraceStore
    from tests.conftest import make_trace
    store = TraceStore(sqlite_conn)
    trace = make_trace(source="critic", outcome=Outcome.DENIED)
    store.put(trace)
    revived = store.get(trace.trace_id)
    assert revived is not None
    assert revived.source == "critic"
    assert revived.outcome == Outcome.DENIED
    assert revived.task_config.cpt_code == "72148"
    assert len(revived.reasoning_steps) == 1


def test_trace_store_list_by_source(sqlite_conn):
    from medreason.store import TraceStore
    from tests.conftest import make_trace
    store = TraceStore(sqlite_conn)
    store.put(make_trace(source="agent"))
    store.put(make_trace(source="critic"))
    store.put(make_trace(source="critic"))
    assert len(store.list_by_source("agent")) == 1
    assert len(store.list_by_source("critic")) == 2
    assert store.count() == 3


def test_trace_store_in_test_phase_blocks(sqlite_conn):
    from medreason.store import LeakGuard, TestSetLeakError, TraceStore
    from tests.conftest import make_trace
    lg = LeakGuard()
    store = TraceStore(sqlite_conn, leak_guard=lg)
    lg.enter_test_phase()
    with pytest.raises(TestSetLeakError):
        store.put(make_trace())


# ── Shared connection: RuleStore + TraceStore co-exist ──────────────────────


def test_rule_and_trace_stores_share_connection(sqlite_conn):
    """A single SQLite connection hosting both stores must keep their
    data independent and neither should clobber the other's schema."""
    from medreason.store import LeakGuard, RuleStore, TraceStore
    from tests.conftest import make_trace
    lg = LeakGuard()
    rstore = RuleStore(sqlite_conn, leak_guard=lg)
    tstore = TraceStore(sqlite_conn, leak_guard=lg)
    rstore.put(make_rule())
    tstore.put(make_trace())
    assert rstore.count() == 1
    assert tstore.count() == 1
