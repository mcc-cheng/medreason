"""Build the Crizotinib-shaped ChEMBL dry-run campaign.

Per Option B of the sprint design decision: we pull real ChEMBL c-MET
IC50 activities and arrange 20-25 compounds into a 5-round / 4-DP
SAR arc that mimics the Cui et al 2011 narrative
(HTS → potency → selectivity → scaffold_hop → ADMET).

The ChEMBL IC50 values are real. The selectivity_ratio, herg_ic50_um,
and caco2_papp values are fabricated from descriptor-driven heuristics
so the agent has discriminating signals that track plausible medchem
priors (e.g., higher cLogP + basic amine -> lower hERG IC50). The
outcome_label per compound is chosen so the team's narrative direction
works out on the dry-run (potency round mostly advances, selectivity
round hits hERG failures, ADMET round succeeds).

This is scaffolding for the +3pp gate check, not a customer deliverable.
The prediction card commits to Recursion numbers, not the ChEMBL number
(plan line 244).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Lipinski

from .chembl_ingest import ChEMBLActivity, fetch_activities
from .scaffolds import murcko_scaffold_smiles, compute_descriptors
from .schema import (
    CompoundRow,
    DecisionPointRow,
    create_campaign_db,
    insert_compounds,
    insert_decision_points,
)


CAMPAIGN_ID = "crizotinib_chembl_dryrun"
C_MET_TARGET = "CHEMBL3717"
CRIZOTINIB_CHEMBL_ID = "CHEMBL601719"


# --------------------------------------------------------------------------
# Round structure — 5 rounds, 4 decision points, Cui 2011 arc.
# --------------------------------------------------------------------------

ROUND_PLAN: tuple[tuple[int, str, str | None], ...] = (
    (1, "HTS hits + baseline SAR", None),                 # no prior DP
    (2, "first potency push",   "potency"),
    (3, "second potency push",  "potency"),
    (4, "selectivity push",     "selectivity"),
    (5, "ADMET polish to crizotinib", "ADMET"),
)

# Per-round IC50 windows (nM): lower → later round. Narrative: potency
# progresses across R2-R3, saturates by R4, ADMET becomes the bottleneck
# by R5.
ROUND_IC50_WINDOWS: dict[int, tuple[float, float]] = {
    1: (1000.0, 10000.0),
    2: (300.0, 1000.0),
    3: (30.0, 300.0),
    4: (10.0, 100.0),
    5: (1.0, 30.0),
}

COMPOUNDS_PER_ROUND = 5


# --------------------------------------------------------------------------
# Fabricated assay heuristics — descriptor-driven, reproducible per compound.
# --------------------------------------------------------------------------


def _has_basic_amine(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    # Very rough heuristic: any aliphatic nitrogen with H-count >= 0 and
    # formal charge 0 — stand-in for "basic amine" signal.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and not atom.GetIsAromatic():
            return True
    return False


def _synthesize_assays(
    smi: str,
    ic50_nm: float,
    round_index: int,
    *,
    rng: random.Random,
) -> dict:
    """Build plausible assay readouts from IC50 + descriptor proxies.
    Signal is staged by round to reflect a natural optimization cascade:

    - Potency: declines monotonically with round (real ChEMBL IC50 band).
    - Selectivity: low in R1-R3, improves only when team explicitly pushes
      selectivity in R4.
    - ADMET (hERG/CLint/TDI): MW/cLogP-driven baseline; stays bad through
      R4 (potency+selectivity work raised cLogP/basicity); only fixed in
      R5 where the team does ADMET polish.
    """
    d = compute_descriptors(smi)
    basic = _has_basic_amine(smi)

    # --- Selectivity ---
    # Until R4, selectivity ratios are modest (team hasn't pushed yet).
    # In R4, the team explicitly screens for selectivity so ratios jump.
    if round_index <= 3:
        selectivity_ratio = round(rng.uniform(2.0, 8.0), 1)
    elif round_index == 4:
        selectivity_ratio = round(rng.uniform(15.0, 35.0), 1)
    else:
        selectivity_ratio = round(rng.uniform(25.0, 60.0), 1)

    # --- hERG ---
    # Tightens with cLogP + basic amine. Rounds 2-4 inherit liabilities
    # from the potency push; R5 polish relieves them.
    herg_base = 30.0
    if d.clogp > 3.5:
        herg_base *= 0.5
    if basic:
        herg_base *= 0.6
    if round_index == 5:
        herg_base *= 2.0  # team's ADMET polish
    herg_ic50_um = round(max(0.3, herg_base * rng.uniform(0.8, 1.3)), 2)

    # --- Caco-2 Papp ---
    caco2 = 15.0
    if d.tpsa > 90:
        caco2 *= 0.5
    if d.tpsa > 120:
        caco2 *= 0.5
    caco2_papp = round(max(0.2, caco2 * rng.uniform(0.7, 1.3)), 2)

    # --- Microsomal intrinsic clearance (CLint, uL/min/mg) ---
    # High is bad. Rounds 2-4 see rising CLint as MW/cLogP grow with
    # potency; R5 polish lowers it.
    if round_index <= 2:
        clint_base = 60.0
    elif round_index <= 4:
        clint_base = 160.0  # the ADMET-bottleneck signal
    else:
        clint_base = 25.0
    clint_ul_min_mg = round(clint_base * rng.uniform(0.8, 1.3), 1)

    # --- CYP3A4 time-dependent inhibition (hard-stop flag) ---
    # Triggered by round 4 build-up (aromatic + basic amine + elevated
    # cLogP); cleaned up in R5.
    cyp3a4_tdi = False
    if round_index == 4 and d.clogp > 3.0 and basic and rng.random() < 0.6:
        cyp3a4_tdi = True

    return {
        "ic50_nm": round(float(ic50_nm), 2),
        "selectivity_ratio": selectivity_ratio,
        "herg_ic50_um": herg_ic50_um,
        "caco2_papp": caco2_papp,
        "clint_ul_min_mg": clint_ul_min_mg,
        "cyp3a4_tdi": cyp3a4_tdi,
    }


def _outcome_for_round(
    round_index: int, assays: dict, *, has_basic_amine: bool
) -> str:
    """Per-round outcome narrative (team's observed result):
    - R1: all advance (seeds).
    - R2: 1-2 fail hERG from the potency push (compounds with cLogP/
      basic-amine liabilities).
    - R3: 1-2 fail hERG (continuing potency push carries risk).
    - R4: 1-2 fail ADMET — selectivity is earned but CLint high or
      CYP3A4 TDI triggered.
    - R5: all advance (ADMET polish to crizotinib).
    """
    if round_index == 1:
        return "advanced"
    if round_index == 2:
        return "failed_herg" if assays["herg_ic50_um"] < 5.0 else "advanced"
    if round_index == 3:
        return "failed_herg" if assays["herg_ic50_um"] < 4.0 else "advanced"
    if round_index == 4:
        if assays.get("cyp3a4_tdi") or assays.get("clint_ul_min_mg", 0) > 180:
            return "failed_admet"
        return "advanced"
    return "advanced"


# --------------------------------------------------------------------------
# Campaign assembly
# --------------------------------------------------------------------------


def _eligible(a: ChEMBLActivity) -> bool:
    if not a.canonical_smiles or a.standard_type != "IC50":
        return False
    if a.standard_units != "nM" or a.standard_value is None:
        return False
    if a.standard_value <= 0:
        return False
    mol = Chem.MolFromSmiles(a.canonical_smiles)
    if mol is None:
        return False
    mw = Chem.Descriptors.MolWt(mol)
    return 200.0 <= mw <= 700.0


def _select_for_round(
    pool: list[ChEMBLActivity],
    round_index: int,
    used_ids: set[str],
    used_scaffolds: set[str],
) -> list[ChEMBLActivity]:
    lo, hi = ROUND_IC50_WINDOWS[round_index]
    in_window = [
        a for a in pool
        if a.molecule_chembl_id not in used_ids
        and a.standard_value is not None
        and lo <= a.standard_value <= hi
    ]
    in_window.sort(key=lambda a: a.standard_value or float("inf"))
    picked: list[ChEMBLActivity] = []
    for a in in_window:
        scaf = murcko_scaffold_smiles(a.canonical_smiles)  # type: ignore[arg-type]
        # Loosely diversify scaffolds so the harness sees variety.
        if len(picked) >= COMPOUNDS_PER_ROUND:
            break
        if scaf in used_scaffolds and len(picked) >= COMPOUNDS_PER_ROUND // 2:
            continue
        picked.append(a)
        used_ids.add(a.molecule_chembl_id)
        used_scaffolds.add(scaf)
    return picked


def build_crizotinib_campaign(
    db_path: str | Path,
    *,
    target_chembl_id: str = C_MET_TARGET,
    fetch_limit: int = 2000,
    seed: int = 42,
) -> Path:
    db_path = Path(db_path)
    activities = fetch_activities(
        target_chembl_id, limit=fetch_limit, use_cache=True
    )
    pool = [a for a in activities if _eligible(a)]
    if len(pool) < COMPOUNDS_PER_ROUND * 5:
        raise RuntimeError(
            f"Only {len(pool)} eligible c-MET actives found "
            f"(need >= {COMPOUNDS_PER_ROUND * 5}). "
            "Increase fetch_limit or adjust IC50 windows."
        )

    rng = random.Random(seed)

    # Ensure crizotinib is reserved for round 5.
    crizo = next(
        (a for a in pool if a.molecule_chembl_id == CRIZOTINIB_CHEMBL_ID), None
    )

    used_ids: set[str] = set()
    used_scaffolds: set[str] = set()
    round_compounds: dict[int, list[ChEMBLActivity]] = {}
    for rnd_idx, _label, _direction in ROUND_PLAN:
        selected = _select_for_round(pool, rnd_idx, used_ids, used_scaffolds)
        round_compounds[rnd_idx] = selected

    if crizo is not None and crizo.molecule_chembl_id not in used_ids:
        # Force crizotinib into round 5 (swap out last compound there).
        r5 = round_compounds[5]
        if r5:
            swapped = r5.pop()
            used_ids.discard(swapped.molecule_chembl_id)
        r5.append(crizo)
        used_ids.add(crizo.molecule_chembl_id)
        round_compounds[5] = r5

    # Assemble DuckDB rows.
    con = create_campaign_db(db_path, overwrite=True)
    try:
        base_ts = datetime(2009, 1, 1, 9, 0, 0)

        dp_rows: list[DecisionPointRow] = []
        for rnd_idx, label, direction in ROUND_PLAN:
            if direction is None:
                continue
            dp_rows.append(
                DecisionPointRow(
                    decision_point_id=f"CRIZO_DP_{rnd_idx}",
                    campaign_id=CAMPAIGN_ID,
                    timestamp=(
                        base_ts + timedelta(days=90 * (rnd_idx - 1))
                    ).isoformat(),
                    round_index=rnd_idx,
                    annotation_source="hand-annotated-from-paper",
                    notes=(
                        f"{label} — direction per Cui 2011 SAR arc "
                        "(scaffolding, not per-compound verified)"
                    ),
                    team_direction_chosen=direction,
                )
            )
        insert_decision_points(con, dp_rows)

        compound_rows: list[CompoundRow] = []
        for rnd_idx, _label, _direction in ROUND_PLAN:
            dp_id = f"CRIZO_DP_{rnd_idx}" if rnd_idx > 1 else None
            for i, act in enumerate(round_compounds[rnd_idx]):
                smi = act.canonical_smiles or ""
                scaffold = murcko_scaffold_smiles(smi)
                d = compute_descriptors(smi)
                assays = _synthesize_assays(
                    smi, act.standard_value or 1000.0, rnd_idx, rng=rng
                )
                outcome = _outcome_for_round(
                    rnd_idx, assays, has_basic_amine=_has_basic_amine(smi)
                )
                compound_rows.append(
                    CompoundRow(
                        compound_id=act.molecule_chembl_id,
                        campaign_id=CAMPAIGN_ID,
                        timestamp=(
                            base_ts
                            + timedelta(days=90 * (rnd_idx - 1) + i + 1)
                        ).isoformat(),
                        round_index=rnd_idx,
                        smiles=smi,
                        scaffold_key=scaffold,
                        mw=d.mw,
                        clogp=d.clogp,
                        tpsa=d.tpsa,
                        hbd=d.hbd,
                        hba=d.hba,
                        proposed_modification=None,
                        agent_rationale=None,
                        assay_readouts=assays,
                        decision_point_id=dp_id,
                        outcome_label=outcome,
                    )
                )
        insert_compounds(con, compound_rows)
    finally:
        con.close()

    return db_path
