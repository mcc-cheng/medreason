"""Tests for medreason.store.leak_guard — Phase 1.

The leak guard is the single protection between the memory store and test
set contamination. Every one of these tests is a regression guardrail.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from tests.conftest import make_rule


# ── Write-side case_id protection ─────────────────────────────────────────────


def test_leak_guard_case_id_exact_match_raises():
    from medreason.store import LeakGuard, TestSetLeakError
    lg = LeakGuard(test_case_ids={"test_001", "test_042"})
    bad = make_rule(supporting_case_ids=["train_005", "test_042"])
    with pytest.raises(TestSetLeakError) as exc:
        lg.check_rule_against_test_set(bad)
    assert "test_042" in str(exc.value)


def test_leak_guard_case_id_only_test_ids_trigger():
    """train_* and dev_* must not trigger the guard."""
    from medreason.store import LeakGuard
    lg = LeakGuard(
        test_case_ids={"test_001"},
        train_case_ids={"train_001", "train_002"},
        dev_case_ids={"dev_001"},
    )
    good = make_rule(supporting_case_ids=["train_001", "train_002", "dev_001"])
    # Should not raise.
    lg.check_rule_against_test_set(good)


def test_leak_guard_empty_test_set_is_passthrough():
    """Unit tests and local dev run with an empty guard — it must never
    raise unless explicitly configured."""
    from medreason.store import LeakGuard
    lg = LeakGuard()
    assert lg.test_case_ids == frozenset()
    lg.check_rule_against_test_set(make_rule(supporting_case_ids=["anything"]))


# ── Embedding-side similarity protection ─────────────────────────────────────


def test_leak_guard_embedding_near_duplicate_raises():
    """An exact embedding match must raise LikelyTestLeakError."""
    from medreason.store import LeakGuard, LikelyTestLeakError
    lg = LeakGuard(
        test_case_ids={"test_001"},
        test_case_embeddings={"test_001": [1.0, 0.0, 0.0]},
        embedding_similarity_threshold=0.9,
    )
    bad = make_rule(
        supporting_case_ids=["train_001"],
        semantic_embedding=[1.0, 0.0, 0.0],
    )
    with pytest.raises(LikelyTestLeakError) as exc:
        lg.check_rule_against_test_set(bad)
    assert "test_001" in str(exc.value)
    assert "cosine" in str(exc.value).lower()


def test_leak_guard_embedding_orthogonal_passes():
    """Orthogonal embeddings must not trip the guard."""
    from medreason.store import LeakGuard
    lg = LeakGuard(
        test_case_ids={"test_001"},
        test_case_embeddings={"test_001": [1.0, 0.0, 0.0]},
        embedding_similarity_threshold=0.9,
    )
    good = make_rule(
        supporting_case_ids=["train_001"],
        semantic_embedding=[0.0, 1.0, 0.0],
    )
    lg.check_rule_against_test_set(good)


def test_leak_guard_embedding_just_below_threshold_passes():
    """Cosine exactly at the threshold must pass (strict >)."""
    from medreason.store import LeakGuard
    import math
    lg = LeakGuard(
        test_case_ids={"test_001"},
        test_case_embeddings={"test_001": [1.0, 0.0]},
        embedding_similarity_threshold=0.92,
    )
    # Angle giving cosine ≈ 0.92.
    theta = math.acos(0.92)
    good = make_rule(
        supporting_case_ids=["train_001"],
        semantic_embedding=[math.cos(theta), math.sin(theta)],
    )
    lg.check_rule_against_test_set(good)  # must not raise


def test_leak_guard_embedding_ignored_without_test_embeddings():
    """If the guard has no test embeddings, the embedding check is skipped
    even when the rule carries one."""
    from medreason.store import LeakGuard
    lg = LeakGuard(test_case_ids={"test_001"})
    rule = make_rule(
        supporting_case_ids=["train_001"],
        semantic_embedding=[1.0, 0.0, 0.0],
    )
    lg.check_rule_against_test_set(rule)


def test_leak_guard_case_id_check_runs_before_embedding_check():
    """Case-id hit should raise TestSetLeakError (the base class) without
    falling through to the embedding check."""
    from medreason.store import LeakGuard, LikelyTestLeakError, TestSetLeakError
    lg = LeakGuard(
        test_case_ids={"test_001"},
        test_case_embeddings={"test_001": [1.0, 0.0, 0.0]},
    )
    bad = make_rule(
        supporting_case_ids=["test_001"],
        semantic_embedding=[0.0, 1.0, 0.0],  # orthogonal — embedding side OK
    )
    with pytest.raises(TestSetLeakError) as exc:
        lg.check_rule_against_test_set(bad)
    # Must be the base class, not LikelyTestLeakError
    assert not isinstance(exc.value, LikelyTestLeakError)


# ── Read-only test-phase mode ─────────────────────────────────────────────────


def test_leak_guard_test_phase_blocks_writes():
    from medreason.store import LeakGuard, TestSetLeakError
    lg = LeakGuard(test_case_ids={"test_001"})
    assert lg.read_only is False
    lg.enter_test_phase()
    assert lg.read_only is True
    with pytest.raises(TestSetLeakError) as exc:
        lg.check_write_allowed("rule.put")
    assert "read-only" in str(exc.value).lower()


def test_leak_guard_exit_test_phase_restores_writes():
    from medreason.store import LeakGuard
    lg = LeakGuard()
    lg.enter_test_phase()
    lg.exit_test_phase()
    lg.check_write_allowed("rule.put")  # must not raise


def test_leak_guard_test_phase_does_not_affect_pure_checks():
    """enter_test_phase blocks writes but does not change the semantics
    of check_rule_against_test_set (which is a read-only check)."""
    from medreason.store import LeakGuard
    lg = LeakGuard(test_case_ids={"test_001"})
    lg.enter_test_phase()
    good = make_rule(supporting_case_ids=["train_001"])
    lg.check_rule_against_test_set(good)  # must not raise on check alone


# ── Fingerprint file loader ───────────────────────────────────────────────────


def test_from_fingerprint_file(tmp_path):
    from medreason.store import LeakGuard
    fp = tmp_path / "fingerprints.json"
    fp.write_text(json.dumps({
        "train": {"train_001": "h1", "train_002": "h2"},
        "dev":   {"dev_001": "h3"},
        "test":  {"test_001": "h4", "test_002": "h5"},
    }))
    lg = LeakGuard.from_fingerprint_file(fp)
    assert lg.test_case_ids == frozenset({"test_001", "test_002"})
    assert lg.train_case_ids == frozenset({"train_001", "train_002"})
    assert lg.dev_case_ids == frozenset({"dev_001"})


def test_from_fingerprint_file_tolerates_missing_splits(tmp_path):
    from medreason.store import LeakGuard
    fp = tmp_path / "fingerprints.json"
    fp.write_text(json.dumps({"test": {"test_001": "h"}}))
    lg = LeakGuard.from_fingerprint_file(fp)
    assert lg.test_case_ids == frozenset({"test_001"})
    assert lg.train_case_ids == frozenset()
    assert lg.dev_case_ids == frozenset()


# ── RuleStore integration with LeakGuard ─────────────────────────────────────


def test_rule_store_put_integrates_leak_guard(sqlite_conn):
    from medreason.store import LeakGuard, RuleStore, TestSetLeakError
    lg = LeakGuard(test_case_ids={"test_001"})
    store = RuleStore(sqlite_conn, leak_guard=lg)
    bad = make_rule(supporting_case_ids=["test_001"])
    with pytest.raises(TestSetLeakError):
        store.put(bad)
    # The rule must NOT be persisted.
    assert store.count() == 0


def test_rule_store_put_in_test_phase_blocks(sqlite_conn):
    from medreason.store import LeakGuard, RuleStore, TestSetLeakError
    store = RuleStore(sqlite_conn, leak_guard=LeakGuard())
    store.leak_guard.enter_test_phase()
    with pytest.raises(TestSetLeakError):
        store.put(make_rule())
    assert store.count() == 0


def test_rule_store_update_posteriors_in_test_phase_blocks(sqlite_conn):
    from medreason.store import LeakGuard, RuleStore, TestSetLeakError
    store = RuleStore(sqlite_conn, leak_guard=LeakGuard())
    rule = make_rule()
    store.put(rule)
    store.leak_guard.enter_test_phase()
    with pytest.raises(TestSetLeakError):
        store.update_posteriors([rule.rule_id], correct=True,
                                applied_map={rule.rule_id: True})
    # Counts unchanged.
    fresh = store.get(rule.rule_id)
    assert fresh.success_count == 0
    assert fresh.seen_count == 0


def test_leak_violation_is_persisted(sqlite_conn):
    """The store's violation logger must record every guard trip in the
    leak_violations table, BEFORE the exception propagates."""
    from medreason.store import LeakGuard, RuleStore, TestSetLeakError
    lg = LeakGuard(test_case_ids={"test_777"})
    store = RuleStore(sqlite_conn, leak_guard=lg)
    bad = make_rule(supporting_case_ids=["test_777"])
    with pytest.raises(TestSetLeakError):
        store.put(bad)
    rows = sqlite_conn.execute(
        "SELECT violation_type, case_ids_json, details_json FROM leak_violations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["violation_type"] == "case_id"
    case_ids = json.loads(rows[0]["case_ids_json"])
    assert case_ids == ["test_777"]


def test_leak_violation_embedding_logged(sqlite_conn):
    from medreason.store import (
        LeakGuard, LikelyTestLeakError, RuleStore,
    )
    lg = LeakGuard(
        test_case_ids={"test_001"},
        test_case_embeddings={"test_001": [1.0, 0.0, 0.0]},
        embedding_similarity_threshold=0.9,
    )
    store = RuleStore(sqlite_conn, leak_guard=lg)
    bad = make_rule(
        supporting_case_ids=["train_001"],
        semantic_embedding=[1.0, 0.0, 0.0],
    )
    with pytest.raises(LikelyTestLeakError):
        store.put(bad)
    rows = sqlite_conn.execute(
        "SELECT violation_type, details_json FROM leak_violations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["violation_type"] == "embedding"
    details = json.loads(rows[0]["details_json"])
    assert details["max_cosine"] == pytest.approx(1.0)
    assert details["nearest_test_case_id"] == "test_001"


def test_leak_violation_read_only_logged(sqlite_conn):
    from medreason.store import LeakGuard, RuleStore, TestSetLeakError
    store = RuleStore(sqlite_conn, leak_guard=LeakGuard())
    store.leak_guard.enter_test_phase()
    with pytest.raises(TestSetLeakError):
        store.put(make_rule())
    rows = sqlite_conn.execute(
        "SELECT violation_type, details_json FROM leak_violations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["violation_type"] == "read_only"
    assert json.loads(rows[0]["details_json"])["operation"] == "rule.put"
