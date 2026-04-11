-- medreason/store schema — Phase 1
--
-- Stores atomic ReasoningRules, ReasoningTraces, an audit log, and
-- a record of every leak-guard violation. Dialect: SQLite. Postgres
-- equivalents are analogous (TEXT → TEXT, INTEGER → INT, BLOB → BYTEA,
-- AUTOINCREMENT → SERIAL).
--
-- Design notes:
-- - rule_json is the canonical serialized ReasoningRule. The denormalized
--   counter columns (success_count, failure_count, seen_count) are the
--   source of truth for posterior math; on read we overwrite those fields
--   in the deserialized JSON before returning the Pydantic model. This
--   avoids rewriting the full blob on every update_posteriors() call.
-- - status is denormalized so we can list active rules without parsing JSON.
-- - Indexes kept minimal for Phase 1. Retrieval Tier 1 indexes land in
--   Phase 5d once ontology lookup is built.

CREATE TABLE IF NOT EXISTS rules (
    rule_id           TEXT PRIMARY KEY,
    status            TEXT NOT NULL,
    rule_json         TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    last_used         TEXT,
    success_count     INTEGER NOT NULL DEFAULT 0,
    failure_count     INTEGER NOT NULL DEFAULT 0,
    seen_count        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status);

CREATE TABLE IF NOT EXISTS traces (
    trace_id          TEXT PRIMARY KEY,
    trace_json        TEXT NOT NULL,
    source            TEXT NOT NULL,        -- 'agent' | 'critic'
    outcome           TEXT NOT NULL,
    resolved_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_source ON traces(source);

CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    event_type        TEXT NOT NULL,        -- 'rule.put' | 'rule.update_posteriors' | 'trace.put' | ...
    entity_id         TEXT,                 -- rule_id or trace_id
    case_id           TEXT,
    details_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);

CREATE TABLE IF NOT EXISTS leak_violations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    violation_type    TEXT NOT NULL,        -- 'case_id' | 'embedding' | 'read_only'
    rule_id           TEXT,
    trace_id          TEXT,
    case_ids_json     TEXT NOT NULL DEFAULT '[]',
    details_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_leak_violations_type ON leak_violations(violation_type);
