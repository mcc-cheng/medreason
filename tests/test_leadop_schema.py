"""Smoke tests for the lead-op DuckDB schema + Murcko pipeline.

Covers: schema DDL applies cleanly, inserts roundtrip, Murcko scaffold
canonicalization is deterministic, descriptors parse on the 5-compound
toy, invalid annotation_source / direction are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medreason_bench.leadop.schema import (
    CAMPAIGN_SCHEMA_VERSION,
    CompoundRow,
    DecisionPointRow,
    connect_campaign_db,
    create_campaign_db,
    insert_compounds,
    insert_decision_points,
)
from medreason_bench.leadop.scaffolds import (
    compute_descriptors,
    murcko_scaffold_smiles,
    scaffold_key_and_descriptors,
)
from medreason_bench.leadop.synthetic import CAMPAIGN_ID, write_toy_campaign


def test_create_campaign_db_writes_meta(tmp_path: Path) -> None:
    con = create_campaign_db(tmp_path / "c.db")
    try:
        version = con.execute(
            "SELECT value FROM leadop_meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == str(CAMPAIGN_SCHEMA_VERSION)
    finally:
        con.close()


def test_decision_point_rejects_bad_source() -> None:
    with pytest.raises(ValueError):
        DecisionPointRow(
            decision_point_id="dp1",
            campaign_id="c",
            timestamp="2024-01-01T00:00:00",
            round_index=1,
            annotation_source="made-up-source",
        )


def test_decision_point_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        DecisionPointRow(
            decision_point_id="dp1",
            campaign_id="c",
            timestamp="2024-01-01T00:00:00",
            round_index=1,
            annotation_source="scientist-marked",
            team_direction_chosen="wrong",
        )


def test_compound_roundtrip_via_insert(tmp_path: Path) -> None:
    con = create_campaign_db(tmp_path / "r.db")
    try:
        row = CompoundRow(
            compound_id="C1",
            campaign_id="c",
            timestamp="2024-01-01T00:00:00",
            round_index=1,
            smiles="c1ccncc1",
            scaffold_key="c1ccncc1",
            mw=79.1,
            clogp=0.5,
            tpsa=12.9,
            hbd=0,
            hba=1,
            assay_readouts={"ic50_nm": 10.0},
        )
        insert_compounds(con, [row])
        got = con.execute(
            "SELECT compound_id, assay_readouts_json FROM compounds"
        ).fetchone()
        assert got[0] == "C1"
        assert json.loads(got[1]) == {"ic50_nm": 10.0}
    finally:
        con.close()


def test_murcko_is_canonical_and_deterministic() -> None:
    smi1 = "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1"
    smi2 = "c1cc(C)ccc1Nc1ncnc2[nH]ccc12"  # same molecule, different written form
    assert murcko_scaffold_smiles(smi1) == murcko_scaffold_smiles(smi2)


def test_descriptors_are_reasonable() -> None:
    d = compute_descriptors("CCO")  # ethanol
    assert 44 < d.mw < 48
    assert d.hbd == 1
    assert d.hba == 1


def test_toy_campaign_writes_and_reads(tmp_path: Path) -> None:
    db_path = write_toy_campaign(tmp_path / "toy.db")
    con = connect_campaign_db(db_path)
    try:
        n_compounds = con.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        n_dps = con.execute("SELECT COUNT(*) FROM decision_points").fetchone()[0]
        assert n_compounds == 5
        assert n_dps == 2

        rounds = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT round_index FROM compounds ORDER BY round_index"
            ).fetchall()
        ]
        assert rounds == [1, 2, 3]

        directions = {
            r[0]
            for r in con.execute(
                "SELECT team_direction_chosen FROM decision_points"
            ).fetchall()
        }
        assert directions == {"potency", "ADMET"}

        scaffolds_nonempty = con.execute(
            "SELECT COUNT(*) FROM compounds WHERE scaffold_key IS NULL"
        ).fetchone()[0]
        assert scaffolds_nonempty == 0
    finally:
        con.close()


def test_scaffold_key_and_descriptors_together() -> None:
    scaffold, d = scaffold_key_and_descriptors("Cc1ccc(Nc2ncnc3[nH]ccc23)cc1")
    assert scaffold
    assert d.mw > 0
    assert d.hba >= 3  # pyrrolopyrimidine Ns
