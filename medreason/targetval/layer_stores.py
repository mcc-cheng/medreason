"""Per-layer rule store wrapping for the LayerRouter.

The LayerRouter needs a small, well-defined surface to write/read
ReasoningRules at each of the three layers (Universal / Disease /
Campaign). The canonical persistent implementation already exists at
`medreason.store.RuleStore` (SQLite-backed, leak-guarded, audit-logged).

This module:

1. Defines `TargetvalRuleStore` — the narrow Protocol the router relies
   on (put / get / list_active / set_status).
2. Provides `open_targetval_rule_store(path)` — the factory the router
   calls per (layer, scope). Reuses the canonical `RuleStore`; does NOT
   reimplement persistence.
3. Treats `Path("/dev/null")` as a request for an in-memory ephemeral
   store. Existing tests (`test_router_ingest_rule_accepts_public_universal`)
   pass `/dev/null` to avoid touching the filesystem; that contract is
   preserved.

Design notes:

- One sqlite3 connection per store, lifetime-bound to the store object.
  The store closes it via `close()` if the caller cares; the targetval
  router currently keeps them open for the LayerRouter's lifetime.
- The wrapper exposes `list_active` only — UNIVERSAL/DISEASE/CAMPAIGN
  retrieval reads only ACTIVE rules. CANDIDATE rules are inert until
  the generalization gate promotes them.
- No back-door writes: the only mutating entry points are `put` and
  `set_status`, both of which go through `medreason.store.RuleStore`,
  which itself runs the leak guard on every write.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ..ontology.rule import ReasoningRule, RuleStatus
from ..store.rules import RuleStore


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class TargetvalRuleStore(Protocol):
    """Subset of the canonical RuleStore API the LayerRouter relies on."""

    def put(self, rule: ReasoningRule) -> None: ...
    def get(self, rule_id: str) -> Optional[ReasoningRule]: ...
    def list_active(self) -> list[ReasoningRule]: ...
    def set_status(self, rule_id: str, status: RuleStatus) -> None: ...


# ── Wrappers ─────────────────────────────────────────────────────────────────


class _RuleStoreAdapter:
    """Adapter that exposes a `RuleStore` through the `TargetvalRuleStore`
    Protocol shape.

    The canonical `RuleStore` has `list_by_status(status)` and not
    `list_active()`; this thin shim renames the call so callers don't
    have to know about it.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._store = RuleStore(conn)

    def put(self, rule: ReasoningRule) -> None:
        self._store.put(rule)

    def get(self, rule_id: str) -> Optional[ReasoningRule]:
        return self._store.get(rule_id)

    def list_active(self) -> list[ReasoningRule]:
        return self._store.list_by_status(RuleStatus.ACTIVE)

    def set_status(self, rule_id: str, status: RuleStatus) -> None:
        self._store.set_status(rule_id, status)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class _InMemoryRuleStore:
    """Ephemeral in-memory rule store.

    Used when callers pass `Path("/dev/null")` — they signaled they don't
    want persistence (typical for unit tests that only exercise the
    policy validator). Put/get/list/set_status all work; nothing touches
    the filesystem. Mirrors `_RuleStoreAdapter` for the Protocol surface.
    """

    def __init__(self) -> None:
        self._rules: dict[str, ReasoningRule] = {}

    def put(self, rule: ReasoningRule) -> None:
        # Pydantic model is mutable; copy by re-validating from dump so
        # later mutations by the caller don't leak into the store.
        self._rules[rule.rule_id] = ReasoningRule.model_validate(
            rule.model_dump()
        )

    def get(self, rule_id: str) -> Optional[ReasoningRule]:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        return ReasoningRule.model_validate(rule.model_dump())

    def list_active(self) -> list[ReasoningRule]:
        return [
            ReasoningRule.model_validate(r.model_dump())
            for r in self._rules.values()
            if r.status is RuleStatus.ACTIVE
        ]

    def set_status(self, rule_id: str, status: RuleStatus) -> None:
        rule = self._rules.get(rule_id)
        if rule is None:
            return
        rule.status = status

    def close(self) -> None:  # parity with _RuleStoreAdapter
        return None


# ── Factory ──────────────────────────────────────────────────────────────────


_DEV_NULL = Path("/dev/null")


def open_targetval_rule_store(path: Path) -> TargetvalRuleStore:
    """Open a sqlite-backed rule store at `path`, or an in-memory store
    if `path == Path("/dev/null")`.

    Lifetime contract: the returned store owns its connection. Callers
    that want to drop the store and free the FD should call `.close()`
    (both the sqlite adapter and the in-memory wrapper expose it).
    """
    if path == _DEV_NULL:
        return _InMemoryRuleStore()

    # Ensure parent dirs exist; sqlite3.connect won't mkdir for us.
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    return _RuleStoreAdapter(conn)
