"""DuckDB schema for target-validation campaigns.

Two tables:

- `targets`: one row per (target, disease_context) pair in the
  campaign. Mirrors the role `compounds` plays in the leadop schema.
- `bypass_outcomes`: ground-truth Phase 2 outcomes and bypass-mechanism
  labels, populated for retrospective benchmark cases. Empty in
  inference-time customer engagements (those are predictions, not
  retrospectives).

The schema is intentionally flat — gene/disease/modality and the most
common evidence summary fields live as columns; the rest of the evidence
bundle is JSON in `evidence_json`.

Imports duckdb lazily — the dependency is added when this module is
first used, not at package import time, so the broader medreason_bench
import graph stays free of the duckdb requirement until needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


TARGETVAL_CAMPAIGN_SCHEMA_VERSION = 1


_TARGETS_DDL = """
CREATE TABLE IF NOT EXISTS targets (
    case_id                  VARCHAR PRIMARY KEY,
    campaign_id              VARCHAR NOT NULL,
    gene_symbol              VARCHAR NOT NULL,
    uniprot_accession        VARCHAR,
    family                   VARCHAR,
    disease_label            VARCHAR NOT NULL,
    ontology_code            VARCHAR,
    biomarker_context        VARCHAR,
    therapeutic_area         VARCHAR,
    modality                 VARCHAR,
    evidence_json            VARCHAR,
    source                   VARCHAR
);
"""

_BYPASS_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS bypass_outcomes (
    case_id                  VARCHAR PRIMARY KEY,
    campaign_id              VARCHAR NOT NULL,
    ground_truth_outcome     VARCHAR NOT NULL,
    ground_truth_bypass      VARCHAR NOT NULL,
    ground_truth_notes       VARCHAR,
    label_source             VARCHAR
);
"""

_META_DDL = """
CREATE TABLE IF NOT EXISTS targetval_meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR
);
"""


@dataclass
class TargetRow:
    case_id: str
    campaign_id: str
    gene_symbol: str
    disease_label: str
    uniprot_accession: Optional[str] = None
    family: Optional[str] = None
    ontology_code: Optional[str] = None
    biomarker_context: Optional[str] = None
    therapeutic_area: Optional[str] = None
    modality: Optional[str] = None
    evidence_json: str = "{}"
    source: Optional[str] = None


@dataclass
class BypassOutcomeRow:
    case_id: str
    campaign_id: str
    ground_truth_outcome: str
    ground_truth_bypass: str
    ground_truth_notes: Optional[str] = None
    label_source: Optional[str] = None


def connect_targetval_db(db_path: str | Path):
    """Open (or create) the targetval DuckDB campaign DB.

    Lazy import of duckdb so the broader medreason_bench package import
    is free of the dependency until someone touches this module.
    """
    import duckdb  # type: ignore

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_TARGETS_DDL)
    con.execute(_BYPASS_OUTCOMES_DDL)
    con.execute(_META_DDL)
    con.execute(
        "INSERT OR REPLACE INTO targetval_meta VALUES (?, ?)",
        ["schema_version", str(TARGETVAL_CAMPAIGN_SCHEMA_VERSION)],
    )
    return con


def insert_targets(con, rows: list[TargetRow]) -> None:
    payload = [
        (
            r.case_id,
            r.campaign_id,
            r.gene_symbol,
            r.uniprot_accession,
            r.family,
            r.disease_label,
            r.ontology_code,
            r.biomarker_context,
            r.therapeutic_area,
            r.modality,
            r.evidence_json,
            r.source,
        )
        for r in rows
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO targets VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def insert_bypass_outcomes(con, rows: list[BypassOutcomeRow]) -> None:
    payload = [
        (
            r.case_id,
            r.campaign_id,
            r.ground_truth_outcome,
            r.ground_truth_bypass,
            r.ground_truth_notes,
            r.label_source,
        )
        for r in rows
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO bypass_outcomes VALUES
        (?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
