"""Trace store — SQLite-backed CRUD for ReasoningTrace.

Traces are the audit unit of the new pipeline: one per critic-verified
resolution. They're stored here for lineage (rule_proposer reads them) and
for --explain mode in the eval harness, but they are NEVER injected raw
into agent prompts.

Phase 1 scope: put / get / list_by_source. The critic-verified-vs-not
distinction is carried by the `source` field on ReasoningTrace.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..ontology.trace import ReasoningTrace
from .leak_guard import LeakGuard


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_schema(conn: sqlite3.Connection) -> None:
    sql = _SCHEMA_PATH.read_text()
    conn.executescript(sql)
    conn.commit()


class TraceStore:
    def __init__(
        self,
        conn: sqlite3.Connection,
        leak_guard: Optional[LeakGuard] = None,
    ):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.leak_guard = leak_guard or LeakGuard()
        # Note: if the same connection is shared with RuleStore, the
        # RuleStore has already installed a violation logger on this
        # LeakGuard instance. We don't clobber it here.
        _apply_schema(self.conn)

    def put(self, trace: ReasoningTrace) -> None:
        self.leak_guard.check_write_allowed(operation="trace.put")
        self.leak_guard.check_trace_against_test_set(trace)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO traces
                (trace_id, trace_json, source, outcome, resolved_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.model_dump_json(),
                trace.source,
                trace.outcome.value,
                trace.resolved_at.isoformat(),
            ),
        )
        self._audit("trace.put", entity_id=trace.trace_id,
                    details={"source": trace.source,
                             "outcome": trace.outcome.value})
        self.conn.commit()

    def get(self, trace_id: str) -> Optional[ReasoningTrace]:
        row = self.conn.execute(
            "SELECT trace_json FROM traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        return ReasoningTrace.model_validate_json(row["trace_json"])

    def list_by_source(self, source: str) -> list[ReasoningTrace]:
        rows = self.conn.execute(
            "SELECT trace_json FROM traces WHERE source = ? ORDER BY resolved_at DESC",
            (source,),
        ).fetchall()
        return [ReasoningTrace.model_validate_json(r["trace_json"]) for r in rows]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM traces").fetchone()
        return int(row["n"])

    def _audit(
        self,
        event_type: str,
        entity_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO audit_log (timestamp, event_type, entity_id, details_json)
            VALUES (?, ?, ?, ?)
            """,
            (_utcnow_iso(), event_type, entity_id, json.dumps(details or {})),
        )
