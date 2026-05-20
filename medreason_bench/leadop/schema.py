"""DuckDB campaign schema for the lead-op retrospective harness.

Two tables, per the approved plan (krish-master-design-20260414):

- compounds: one row per compound in a campaign.
- decision_points: one row per decision point (SAR round boundary).

The schema is intentionally flat: MW/logP/TPSA/HBD/HBA live as columns
on the compounds table (not inside the descriptors JSON), because every
retrieval and scaffold-hop rule needs cheap column access to them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb

CAMPAIGN_SCHEMA_VERSION = 1

_COMPOUNDS_DDL = """
CREATE TABLE IF NOT EXISTS compounds (
    compound_id              VARCHAR PRIMARY KEY,
    campaign_id              VARCHAR NOT NULL,
    timestamp                TIMESTAMP NOT NULL,
    round_index              INTEGER NOT NULL,
    smiles                   VARCHAR NOT NULL,
    scaffold_key             VARCHAR,
    mw                       DOUBLE,
    clogp                    DOUBLE,
    tpsa                     DOUBLE,
    hbd                      INTEGER,
    hba                      INTEGER,
    proposed_modification    VARCHAR,
    agent_rationale          VARCHAR,
    assay_readouts_json      VARCHAR,
    decision_point_id        VARCHAR,
    outcome_label            VARCHAR
);
"""

_DECISION_POINTS_DDL = """
CREATE TABLE IF NOT EXISTS decision_points (
    decision_point_id        VARCHAR PRIMARY KEY,
    campaign_id              VARCHAR NOT NULL,
    timestamp                TIMESTAMP NOT NULL,
    round_index              INTEGER NOT NULL,
    annotation_source        VARCHAR NOT NULL,
    notes                    VARCHAR,
    team_direction_chosen    VARCHAR
);
"""

_META_DDL = """
CREATE TABLE IF NOT EXISTS leadop_meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR
);
"""


_VALID_ANNOTATION_SOURCES = {
    "hand-annotated-from-paper",
    "scientist-marked",
    "extractor",
}

_VALID_DIRECTIONS = {"potency", "selectivity", "ADMET", "scaffold_hop"}


@dataclass
class CompoundRow:
    compound_id: str
    campaign_id: str
    timestamp: str
    round_index: int
    smiles: str
    scaffold_key: str | None = None
    mw: float | None = None
    clogp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    proposed_modification: str | None = None
    agent_rationale: str | None = None
    assay_readouts: dict[str, Any] = field(default_factory=dict)
    decision_point_id: str | None = None
    outcome_label: str | None = None

    def to_insert_row(self) -> tuple:
        return (
            self.compound_id,
            self.campaign_id,
            self.timestamp,
            self.round_index,
            self.smiles,
            self.scaffold_key,
            self.mw,
            self.clogp,
            self.tpsa,
            self.hbd,
            self.hba,
            self.proposed_modification,
            self.agent_rationale,
            json.dumps(self.assay_readouts, sort_keys=True),
            self.decision_point_id,
            self.outcome_label,
        )


@dataclass
class DecisionPointRow:
    decision_point_id: str
    campaign_id: str
    timestamp: str
    round_index: int
    annotation_source: str
    notes: str | None = None
    team_direction_chosen: str | None = None

    def __post_init__(self) -> None:
        if self.annotation_source not in _VALID_ANNOTATION_SOURCES:
            raise ValueError(
                f"annotation_source must be one of {_VALID_ANNOTATION_SOURCES}, "
                f"got {self.annotation_source!r}"
            )
        if (
            self.team_direction_chosen is not None
            and self.team_direction_chosen not in _VALID_DIRECTIONS
        ):
            raise ValueError(
                f"team_direction_chosen must be one of {_VALID_DIRECTIONS} or None, "
                f"got {self.team_direction_chosen!r}"
            )

    def to_insert_row(self) -> tuple:
        return (
            self.decision_point_id,
            self.campaign_id,
            self.timestamp,
            self.round_index,
            self.annotation_source,
            self.notes,
            self.team_direction_chosen,
        )


def connect_campaign_db(path: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path))


def create_campaign_db(
    path: str | Path,
    *,
    overwrite: bool = False,
) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if overwrite and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_COMPOUNDS_DDL)
    con.execute(_DECISION_POINTS_DDL)
    con.execute(_META_DDL)
    con.execute(
        "INSERT OR REPLACE INTO leadop_meta(key, value) VALUES (?, ?)",
        ["schema_version", str(CAMPAIGN_SCHEMA_VERSION)],
    )
    return con


def insert_compounds(
    con: duckdb.DuckDBPyConnection, rows: list[CompoundRow]
) -> None:
    con.executemany(
        "INSERT INTO compounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [r.to_insert_row() for r in rows],
    )


def insert_decision_points(
    con: duckdb.DuckDBPyConnection, rows: list[DecisionPointRow]
) -> None:
    con.executemany(
        "INSERT INTO decision_points VALUES (?,?,?,?,?,?,?)",
        [r.to_insert_row() for r in rows],
    )
