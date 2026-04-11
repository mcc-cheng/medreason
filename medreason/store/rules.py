"""Rule store — SQLite-backed CRUD for ReasoningRule.

Phase 1 scope:
- put / get / delete / list_by_status
- update_posteriors with the "only credit applied rules" semantic
- LeakGuard hook on every write
- audit_log entries for every mutation
- leak_violations entries for every guard trip

Phase 5 will add: ontology-lookup indexes, hybrid retrieval (rules.py stays
CRUD-only; retrieval lives in medreason/retrieval/), rule quarantine /
revival hysteresis, and a PG backend.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..ontology.rule import ReasoningRule, RuleStatus
from .leak_guard import LeakGuard


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Idempotent schema creation. Safe to call on every connect."""
    sql = _SCHEMA_PATH.read_text()
    conn.executescript(sql)
    conn.commit()


class RuleStore:
    """SQLite-backed CRUD for ReasoningRule.

    The caller provides the connection (makes tests easy: pass
    sqlite3.connect(":memory:")). The store runs its schema idempotently
    on construction.

    Every write consults `leak_guard`. Pass a no-arg LeakGuard() to disable
    (useful for unit tests of the store itself).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        leak_guard: Optional[LeakGuard] = None,
    ):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.leak_guard = leak_guard or LeakGuard()
        # Wire violation logging into the same DB so every guard trip is
        # persisted without the caller needing to know it happened.
        self.leak_guard.set_violation_logger(self._log_leak_violation)
        _apply_schema(self.conn)

    # ── Write path ───────────────────────────────────────────────────────────

    def put(self, rule: ReasoningRule) -> None:
        """Insert or replace a rule. All guard checks run first.

        On conflict (same rule_id), the existing row is replaced. Counter
        columns are written from the rule object — if you want to preserve
        existing counts across an upsert, load first, copy counts, then put.
        """
        self.leak_guard.check_write_allowed(operation="rule.put")
        self.leak_guard.check_rule_against_test_set(rule)

        rule_json = rule.model_dump_json()
        now = _utcnow_iso()
        # Preserve the stored created_at if we're replacing; the passed
        # rule's created_at is authoritative for new rows.
        cur = self.conn.execute(
            "SELECT created_at FROM rules WHERE rule_id = ?",
            (rule.rule_id,),
        )
        row = cur.fetchone()
        created_at = row["created_at"] if row else rule.created_at.isoformat()

        self.conn.execute(
            """
            INSERT INTO rules
                (rule_id, status, rule_json, created_at, last_used,
                 success_count, failure_count, seen_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rule_id) DO UPDATE SET
                status = excluded.status,
                rule_json = excluded.rule_json,
                last_used = excluded.last_used,
                success_count = excluded.success_count,
                failure_count = excluded.failure_count,
                seen_count = excluded.seen_count
            """,
            (
                rule.rule_id,
                rule.status.value,
                rule_json,
                created_at,
                rule.last_used.isoformat() if rule.last_used else None,
                rule.success_count,
                rule.failure_count,
                rule.seen_count,
            ),
        )
        self._audit("rule.put", entity_id=rule.rule_id,
                    details={"status": rule.status.value})
        self.conn.commit()

    def delete(self, rule_id: str) -> bool:
        """Hard-delete a rule. Returns True if a row was removed.

        Prefer `set_status(rule_id, DEPRECATED)` for soft-delete in normal
        operation. Hard delete is for tests and manual cleanup.
        """
        self.leak_guard.check_write_allowed(operation="rule.delete")
        cur = self.conn.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
        self.conn.commit()
        removed = cur.rowcount > 0
        if removed:
            self._audit("rule.delete", entity_id=rule_id)
        return removed

    def set_status(self, rule_id: str, status: RuleStatus) -> None:
        """Change a rule's status without rewriting its JSON body."""
        self.leak_guard.check_write_allowed(operation="rule.set_status")
        self.conn.execute(
            "UPDATE rules SET status = ? WHERE rule_id = ?",
            (status.value, rule_id),
        )
        # Also update the embedded JSON so get() returns the correct status.
        row = self.conn.execute(
            "SELECT rule_json FROM rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        if row is not None:
            data = json.loads(row["rule_json"])
            data["status"] = status.value
            self.conn.execute(
                "UPDATE rules SET rule_json = ? WHERE rule_id = ?",
                (json.dumps(data), rule_id),
            )
        self.conn.commit()
        self._audit("rule.set_status", entity_id=rule_id,
                    details={"status": status.value})

    def update_posteriors(
        self,
        rule_ids: list[str],
        correct: bool,
        applied_map: Optional[dict[str, bool]] = None,
        case_id: Optional[str] = None,
    ) -> None:
        """Update success/failure/seen counts for retrieved rules.

        Semantics (matches Phase 5 posterior spec):
        - seen_count increments for every retrieved rule, regardless of
          whether the agent applied it.
        - success_count / failure_count increment ONLY when
          applied_map[rule_id] is True. Rules the agent marked applied=False
          are NOT credited or punished — ignoring a rule is a signal about
          the reranker, not the rule itself.
        - `correct` is the case-level outcome (AgentResult.correct).

        `applied_map` defaults to all-True if omitted, which matches the
        legacy semantics. New code should always pass the parsed
        AgentResult.applied_rules map.
        """
        self.leak_guard.check_write_allowed(operation="rule.update_posteriors")
        if not rule_ids:
            return

        applied_map = applied_map or {rid: True for rid in rule_ids}
        now = _utcnow_iso()

        for rid in rule_ids:
            applied = applied_map.get(rid, False)
            if applied:
                col = "success_count" if correct else "failure_count"
                self.conn.execute(
                    f"""
                    UPDATE rules
                       SET {col} = {col} + 1,
                           seen_count = seen_count + 1,
                           last_used = ?
                     WHERE rule_id = ?
                    """,
                    (now, rid),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE rules
                       SET seen_count = seen_count + 1,
                           last_used = ?
                     WHERE rule_id = ?
                    """,
                    (now, rid),
                )
            self._audit(
                "rule.update_posteriors",
                entity_id=rid,
                case_id=case_id,
                details={"correct": correct, "applied": applied},
            )
        self.conn.commit()

    # ── Read path ────────────────────────────────────────────────────────────

    def get(self, rule_id: str) -> Optional[ReasoningRule]:
        """Load a rule. Counter columns (success/failure/seen_count) are
        the source of truth — the JSON blob's copies of those fields are
        overwritten before the Pydantic model is rehydrated.
        """
        row = self.conn.execute(
            """
            SELECT rule_json, success_count, failure_count, seen_count, last_used
              FROM rules
             WHERE rule_id = ?
            """,
            (rule_id,),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row["rule_json"])
        data["success_count"] = row["success_count"]
        data["failure_count"] = row["failure_count"]
        data["seen_count"] = row["seen_count"]
        if row["last_used"] is not None:
            data["last_used"] = row["last_used"]
        return ReasoningRule.model_validate(data)

    def list_by_status(self, status: RuleStatus) -> list[ReasoningRule]:
        rows = self.conn.execute(
            """
            SELECT rule_json, success_count, failure_count, seen_count, last_used
              FROM rules
             WHERE status = ?
             ORDER BY success_count DESC
            """,
            (status.value,),
        ).fetchall()
        out: list[ReasoningRule] = []
        for row in rows:
            data = json.loads(row["rule_json"])
            data["success_count"] = row["success_count"]
            data["failure_count"] = row["failure_count"]
            data["seen_count"] = row["seen_count"]
            if row["last_used"] is not None:
                data["last_used"] = row["last_used"]
            out.append(ReasoningRule.model_validate(data))
        return out

    def count(self, status: Optional[RuleStatus] = None) -> int:
        if status is None:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM rules").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM rules WHERE status = ?",
                (status.value,),
            ).fetchone()
        return int(row["n"])

    # ── Internal: audit + violation logging ─────────────────────────────────

    def _audit(
        self,
        event_type: str,
        entity_id: Optional[str] = None,
        case_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO audit_log (timestamp, event_type, entity_id, case_id, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_utcnow_iso(), event_type, entity_id, case_id,
             json.dumps(details or {})),
        )

    def _log_leak_violation(self, violation_type: str, details: dict) -> None:
        """Callback handed to the LeakGuard so every guard trip is
        persisted. Invoked BEFORE the exception propagates — if this
        raises, the guard swallows the error to avoid masking the
        primary TestSetLeakError."""
        self.conn.execute(
            """
            INSERT INTO leak_violations
                (timestamp, violation_type, rule_id, trace_id, case_ids_json, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _utcnow_iso(),
                violation_type,
                details.get("rule_id"),
                details.get("trace_id"),
                json.dumps(details.get("offending_case_ids", [])),
                json.dumps(details),
            ),
        )
        self.conn.commit()
