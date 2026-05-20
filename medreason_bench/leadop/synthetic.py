"""5-compound synthetic toy campaign for smoke-testing the leadop harness.

The plan requires a <=5-compound synthetic campaign that exercises every
new component (schema, Murcko pipeline, blind re-run with temporal gate,
metrics) before the real ChEMBL dry-run runs. This module is the fixture.

Story: a tiny 3-round, 5-compound campaign on a kinase target. Round 1
hit is submicromolar but hERG-liable; Round 2 team chose POTENCY first,
which failed at the gate; Round 3 after the agent gets a memory of
"potency-first with basic amine + clogp>3.5 = hERG fail", should pick
ADMET. So: top-1 hit goes 0/2 without memory -> 1/2 with memory on the
(trivial) toy. Lift exists by construction; smoke tests don't assert it,
they just assert schema wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .schema import (
    CompoundRow,
    DecisionPointRow,
    connect_campaign_db,
    create_campaign_db,
    insert_compounds,
    insert_decision_points,
)
from .scaffolds import scaffold_key_and_descriptors


CAMPAIGN_ID = "toy_kinase_v1"


_SEED_COMPOUNDS: list[tuple[str, int, str, str, str | None, dict]] = [
    # (compound_id, round, smiles, proposed_modification, outcome_label, assay_readouts)
    (
        "TOY_001",
        1,
        "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1",  # simple pyrrolopyrimidine aminoaryl hit
        "initial_hit",
        "advanced",
        {"ic50_nm": 450.0, "selectivity_ratio": 12.0, "herg_ic50_um": 30.0},
    ),
    (
        "TOY_002",
        2,
        "CCN(CC)CCc1ccc(Nc2ncnc3[nH]ccc23)cc1",  # basic amine tail, clogp up
        "add_basic_amine_tail_for_potency",
        "failed_herg",
        {"ic50_nm": 45.0, "selectivity_ratio": 14.0, "herg_ic50_um": 1.2},
    ),
    (
        "TOY_003",
        2,
        "COc1ccc(Nc2ncnc3[nH]ccc23)cc1OC",  # methoxy decorated, potency but poor perm
        "electron_donating_groups_for_potency",
        "failed_permeability",
        {"ic50_nm": 60.0, "selectivity_ratio": 18.0, "herg_ic50_um": 28.0, "caco2_papp": 0.4},
    ),
    (
        "TOY_004",
        3,
        "OCC1CCN(c2ccc(Nc3ncnc4[nH]ccc34)cc2)CC1",  # polar pendant; ADMET-first pivot
        "reduce_clogp_and_polar_pendant",
        "advanced",
        {"ic50_nm": 120.0, "selectivity_ratio": 25.0, "herg_ic50_um": 40.0, "caco2_papp": 8.2},
    ),
    (
        "TOY_005",
        3,
        "O=C(N)C1CCN(c2ccc(Nc3ncnc4[nH]ccc34)cc2)CC1",  # amide pendant, further ADMET
        "amide_pendant_metabolic_shielding",
        "advanced",
        {"ic50_nm": 95.0, "selectivity_ratio": 30.0, "herg_ic50_um": 50.0, "caco2_papp": 12.0},
    ),
]


_DECISION_POINTS: list[tuple[str, int, str, str]] = [
    # (decision_point_id, round, team_direction_chosen, notes)
    ("TOY_DP_1", 2, "potency", "team chose potency-first after Round 1 submicromolar hit"),
    ("TOY_DP_2", 3, "ADMET", "team pivoted to ADMET after Round 2 hERG failure"),
]


def write_toy_campaign(db_path: str | Path) -> Path:
    db_path = Path(db_path)
    con = create_campaign_db(db_path, overwrite=True)
    try:
        base_ts = datetime(2024, 1, 1, 12, 0, 0)

        dp_rows = []
        for dp_id, rnd, direction, note in _DECISION_POINTS:
            dp_rows.append(
                DecisionPointRow(
                    decision_point_id=dp_id,
                    campaign_id=CAMPAIGN_ID,
                    timestamp=(base_ts + timedelta(days=30 * (rnd - 1))).isoformat(),
                    round_index=rnd,
                    annotation_source="hand-annotated-from-paper",
                    notes=note,
                    team_direction_chosen=direction,
                )
            )
        insert_decision_points(con, dp_rows)

        c_rows = []
        for cid, rnd, smi, mod, outcome, assays in _SEED_COMPOUNDS:
            scaffold, d5 = scaffold_key_and_descriptors(smi)
            dp_id = None
            for dp_r in dp_rows:
                if dp_r.round_index == rnd:
                    dp_id = dp_r.decision_point_id
                    break
            c_rows.append(
                CompoundRow(
                    compound_id=cid,
                    campaign_id=CAMPAIGN_ID,
                    timestamp=(
                        base_ts + timedelta(days=30 * (rnd - 1) + int(cid[-1]))
                    ).isoformat(),
                    round_index=rnd,
                    smiles=smi,
                    scaffold_key=scaffold,
                    mw=d5.mw,
                    clogp=d5.clogp,
                    tpsa=d5.tpsa,
                    hbd=d5.hbd,
                    hba=d5.hba,
                    proposed_modification=mod,
                    agent_rationale=None,
                    assay_readouts=assays,
                    decision_point_id=dp_id,
                    outcome_label=outcome,
                )
            )
        insert_compounds(con, c_rows)
    finally:
        con.close()
    return db_path
