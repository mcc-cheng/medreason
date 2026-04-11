"""Pattern store — PostgreSQL+pgvector with SQLite+numpy fallback."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import DATABASE_URL, EMBEDDING_DIM, SQLITE_PATH
from .ontology import PriorAuthTaskConfig, ReasoningPattern


def _deterministic_embedding(text: str) -> list[float]:
    """Hash-based deterministic embedding for local dev (no OpenAI key)."""
    h = hashlib.sha512(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _embed(task_config: PriorAuthTaskConfig) -> list[float]:
    """Embed a task config. Uses OpenAI if available, else deterministic hash."""
    text = (
        f"{task_config.payer.value} {task_config.cpt_code} "
        f"{' '.join(task_config.icd10_codes)} {task_config.facility_type.value} "
        f"{task_config.denial_reason.value if task_config.denial_reason else 'none'}"
    )
    try:
        from openai import OpenAI
        from .config import OPENAI_API_KEY
        if OPENAI_API_KEY:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return resp.data[0].embedding
    except Exception:
        pass
    return _deterministic_embedding(text)


class PatternStore:
    """Hybrid pattern store with PostgreSQL primary and SQLite fallback."""

    def __init__(self):
        self._pg_conn = None
        self._sqlite_conn = None
        self._use_pg = False
        self._connect()

    def _connect(self):
        try:
            import psycopg2
            self._pg_conn = psycopg2.connect(DATABASE_URL)
            self._pg_conn.autocommit = True
            self._init_pg()
            self._use_pg = True
        except Exception:
            self._sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
            self._sqlite_conn.row_factory = sqlite3.Row
            self._init_sqlite()
            self._use_pg = False

    def _init_pg(self):
        cur = self._pg_conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                task_config JSONB NOT NULL,
                argument_structure JSONB NOT NULL,
                key_reasoning_steps JSONB NOT NULL,
                success_count INT DEFAULT 0,
                failure_count INT DEFAULT 0,
                valid_from TEXT DEFAULT '2026-Q1',
                valid_until TEXT,
                last_used TIMESTAMPTZ,
                embedding vector(1536)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_config_hash ON patterns(config_hash)"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                event_type TEXT NOT NULL,
                pattern_id TEXT,
                case_id TEXT,
                injected_pattern_ids TEXT[],
                details JSONB
            )
        """)
        cur.close()

    def _init_sqlite(self):
        cur = self._sqlite_conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                task_config TEXT NOT NULL,
                argument_structure TEXT NOT NULL,
                key_reasoning_steps TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                valid_from TEXT DEFAULT '2026-Q1',
                valid_until TEXT,
                last_used TEXT,
                embedding TEXT
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_config_hash ON patterns(config_hash)"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                pattern_id TEXT,
                case_id TEXT,
                injected_pattern_ids TEXT,
                details TEXT
            )
        """)
        self._sqlite_conn.commit()

    def _log_event(self, event_type: str, pattern_id: str = None,
                   case_id: str = None, injected_ids: list[str] = None,
                   details: dict = None):
        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute(
                "INSERT INTO audit_log (event_type, pattern_id, case_id, injected_pattern_ids, details) "
                "VALUES (%s, %s, %s, %s, %s)",
                (event_type, pattern_id, case_id, injected_ids,
                 json.dumps(details) if details else None),
            )
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                "INSERT INTO audit_log (event_type, pattern_id, case_id, injected_pattern_ids, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_type, pattern_id, case_id,
                 json.dumps(injected_ids) if injected_ids else None,
                 json.dumps(details) if details else None),
            )
            self._sqlite_conn.commit()

    def store_pattern(self, pattern: ReasoningPattern):
        emb = _embed(pattern.task_config)
        self._log_event("store_pattern", pattern_id=pattern.pattern_id)

        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute(
                """INSERT INTO patterns
                   (pattern_id, config_hash, task_config, argument_structure,
                    key_reasoning_steps, success_count, failure_count,
                    valid_from, valid_until, last_used, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (pattern_id) DO UPDATE SET
                    success_count = EXCLUDED.success_count,
                    failure_count = EXCLUDED.failure_count,
                    last_used = EXCLUDED.last_used""",
                (pattern.pattern_id, pattern.config_hash,
                 pattern.task_config.model_dump_json(),
                 pattern.argument_structure.model_dump_json(),
                 json.dumps(pattern.key_reasoning_steps),
                 pattern.success_count, pattern.failure_count,
                 pattern.valid_from, pattern.valid_until,
                 datetime.now(timezone.utc).isoformat(),
                 str(emb)),
            )
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO patterns
                   (pattern_id, config_hash, task_config, argument_structure,
                    key_reasoning_steps, success_count, failure_count,
                    valid_from, valid_until, last_used, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (pattern.pattern_id, pattern.config_hash,
                 pattern.task_config.model_dump_json(),
                 pattern.argument_structure.model_dump_json(),
                 json.dumps(pattern.key_reasoning_steps),
                 pattern.success_count, pattern.failure_count,
                 pattern.valid_from, pattern.valid_until,
                 datetime.now(timezone.utc).isoformat(),
                 json.dumps(emb)),
            )
            self._sqlite_conn.commit()

    def retrieve_patterns(
        self, task_config: PriorAuthTaskConfig, top_k: int = 3
    ) -> list[ReasoningPattern]:
        config_hash = task_config.config_hash
        patterns: list[ReasoningPattern] = []

        # Exact match first
        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute(
                "SELECT * FROM patterns WHERE config_hash = %s AND valid_until IS NULL "
                "ORDER BY success_count DESC LIMIT %s",
                (config_hash, top_k),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            for row in rows:
                patterns.append(self._row_to_pattern(dict(zip(cols, row))))
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                "SELECT * FROM patterns WHERE config_hash = ? AND valid_until IS NULL "
                "ORDER BY success_count DESC LIMIT ?",
                (config_hash, top_k),
            )
            for row in cur.fetchall():
                patterns.append(self._row_to_pattern(dict(row)))

        if len(patterns) >= top_k:
            return patterns[:top_k]

        # Vector similarity fallback
        remaining = top_k - len(patterns)
        existing_ids = {p.pattern_id for p in patterns}
        query_emb = _embed(task_config)

        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute(
                "SELECT *, embedding <=> %s::vector AS dist FROM patterns "
                "WHERE valid_until IS NULL ORDER BY dist LIMIT %s",
                (str(query_emb), remaining + len(existing_ids)),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            for row in rows:
                d = dict(zip(cols, row))
                if d["pattern_id"] not in existing_ids:
                    patterns.append(self._row_to_pattern(d))
                    if len(patterns) >= top_k:
                        break
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                "SELECT * FROM patterns WHERE valid_until IS NULL"
            )
            all_rows = cur.fetchall()
            query_vec = np.array(query_emb, dtype=np.float32)
            scored = []
            for row in all_rows:
                d = dict(row)
                if d["pattern_id"] in existing_ids:
                    continue
                stored_emb = np.array(json.loads(d["embedding"]), dtype=np.float32)
                sim = float(np.dot(query_vec, stored_emb))
                scored.append((sim, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, d in scored[:remaining]:
                patterns.append(self._row_to_pattern(d))

        return patterns[:top_k]

    def reinforce(self, pattern_id: str, success: bool):
        col = "success_count" if success else "failure_count"
        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute(
                f"UPDATE patterns SET {col} = {col} + 1, last_used = NOW() "
                "WHERE pattern_id = %s",
                (pattern_id,),
            )
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                f"UPDATE patterns SET {col} = {col} + 1, last_used = datetime('now') "
                "WHERE pattern_id = ?",
                (pattern_id,),
            )
            self._sqlite_conn.commit()
        self._log_event("reinforce", pattern_id=pattern_id,
                        details={"success": success})

    def deprecate(self, pattern_id: str, reason: str):
        now = datetime.now(timezone.utc).isoformat()
        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute(
                "UPDATE patterns SET valid_until = %s WHERE pattern_id = %s",
                (now, pattern_id),
            )
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute(
                "UPDATE patterns SET valid_until = ? WHERE pattern_id = ?",
                (now, pattern_id),
            )
            self._sqlite_conn.commit()
        self._log_event("deprecate", pattern_id=pattern_id,
                        details={"reason": reason})

    def get_stats(self) -> dict:
        if self._use_pg:
            cur = self._pg_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM patterns WHERE valid_until IS NULL")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT AVG(success_count::float / NULLIF(success_count + failure_count, 0)) "
                "FROM patterns WHERE valid_until IS NULL"
            )
            avg_rate = cur.fetchone()[0] or 0.0
            cur.execute(
                "SELECT config_hash, COUNT(*) FROM patterns "
                "WHERE valid_until IS NULL GROUP BY config_hash"
            )
            coverage = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
        else:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM patterns WHERE valid_until IS NULL")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT AVG(CAST(success_count AS REAL) / MAX(success_count + failure_count, 1)) "
                "FROM patterns WHERE valid_until IS NULL"
            )
            avg_rate = cur.fetchone()[0] or 0.0
            cur.execute(
                "SELECT config_hash, COUNT(*) FROM patterns "
                "WHERE valid_until IS NULL GROUP BY config_hash"
            )
            coverage = {r[0]: r[1] for r in cur.fetchall()}
        return {
            "total_patterns": total,
            "avg_success_rate": round(float(avg_rate), 3),
            "coverage_configs": len(coverage),
        }

    def _row_to_pattern(self, d: dict) -> ReasoningPattern:
        tc_raw = d["task_config"]
        if isinstance(tc_raw, str):
            tc_raw = json.loads(tc_raw)
        arg_raw = d["argument_structure"]
        if isinstance(arg_raw, str):
            arg_raw = json.loads(arg_raw)
        steps_raw = d["key_reasoning_steps"]
        if isinstance(steps_raw, str):
            steps_raw = json.loads(steps_raw)
        return ReasoningPattern(
            pattern_id=d["pattern_id"],
            config_hash=d["config_hash"],
            task_config=PriorAuthTaskConfig(**tc_raw),
            argument_structure=arg_raw,
            key_reasoning_steps=steps_raw,
            success_count=d["success_count"],
            failure_count=d["failure_count"],
            valid_from=d["valid_from"],
            valid_until=d.get("valid_until"),
        )

    def close(self):
        if self._pg_conn:
            self._pg_conn.close()
        if self._sqlite_conn:
            self._sqlite_conn.close()
