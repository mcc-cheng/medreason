"""Synthetic drug discovery benchmark — 200 pressure-test cases.

Generated from templates with seeded RNG. Each case has a deterministic
ground truth derived from the policy rules. ~40% are "trap" cases where
default/naive reasoning gives the wrong answer.

Case IDs: dd_syn_001 .. dd_syn_200
Compound IDs: VDC-S001 .. VDC-S200
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from medreason.ontology import (
    BenchmarkCase,
    DenialReason,
    Difficulty,
    FacilityType,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
)
from medreason_bench.data.drugdisc_cases import DRUG_DESIGN_GUIDELINES


# ── Scaffolds and targets for variety ──────────────────────────────────────

KINASE_SCAFFOLDS = [
    "pyrimidine-based", "pyrazolopyrimidine", "aminopyridine",
    "quinazoline", "imidazopyridine", "thiazolopyrimidine",
    "indazole", "pyrrolopyridine", "benzimidazole", "triazolopyridine",
]
KINASE_TARGETS = [
    ("CDK4/6", "CDK"), ("EGFR", "EGFR"), ("ALK", "ALK"),
    ("JAK1", "JAK"), ("BTK", "BTK"), ("FGFR2", "FGFR"),
    ("PI3Kα", "PI3K"), ("VEGFR2", "VEGFR"), ("ABL1", "ABL"),
    ("MET", "MET"), ("RET", "RET"), ("SRC", "SRC"),
]
GPCR_SCAFFOLDS = [
    "piperazine-based", "piperidine-based", "pyrrolidine",
    "dihydroisoquinoline", "benzodiazepine", "morpholine",
    "tetrahydropyridine", "spiro-indoline",
]
GPCR_TARGETS = [
    ("5-HT2A", "serotonin"), ("D2", "dopamine"), ("CB1", "cannabinoid"),
    ("µ-opioid", "opioid"), ("OX2R", "orexin"), ("CCR5", "chemokine"),
    ("H3", "histamine"), ("M1", "muscarinic"),
]
ENZYME_SCAFFOLDS = [
    "hydroxamic acid", "boronic acid", "thiazolidinedione",
    "nitrile-warhead", "sulfonamide", "phosphonate",
    "ketone-based", "spirocyclic",
]
ENZYME_TARGETS = [
    ("HDAC6", "HDAC"), ("PDE4", "PDE"), ("IDO1", "IDO"),
    ("MAO-B", "MAO"), ("AChE", "cholinesterase"), ("PARP1", "PARP"),
    ("NAMPT", "NAMPT"), ("DPP-4", "DPP"),
]
ION_CHANNEL_SCAFFOLDS = [
    "dihydropyridine", "benzothiazepine", "pyrrole-based",
    "quinoline", "isoquinoline", "chromene",
]
ION_CHANNEL_TARGETS = [
    ("Nav1.7", "sodium channel"), ("Kv1.3", "potassium channel"),
    ("TRPV1", "TRP channel"), ("Cav2.2", "calcium channel"),
    ("ASIC1a", "acid-sensing"), ("GABAA", "GABA"),
]
CNS_SCAFFOLDS = [
    "piperazine-linked", "tropane", "morpholine-fused",
    "dihydrobenzofuran", "pyridylpiperidine", "azaindole",
]
CNS_TARGETS = [
    ("5-HT2A", "serotonin"), ("D2/D3", "dopamine"),
    ("GluN2B", "NMDA"), ("mGlu5", "glutamate"),
    ("OX2R", "orexin"), ("H3", "histamine"),
]
ONCOLOGY_TARGETS = [
    ("KRAS G12C", "RAS"), ("BRD4", "bromodomain"), ("EZH2", "methyltransferase"),
    ("MDM2", "p53"), ("MCL-1", "apoptosis"), ("WEE1", "checkpoint"),
]
MACROCYCLE_TYPES = [
    "macrocyclic peptide", "stapled peptide", "cyclic depsipeptide",
    "macrolactam", "macrolide-derived",
]
PROTAC_TARGETS = [
    "BRD4", "AR", "ERα", "IRAK4", "CDK9", "STAT3",
]


@dataclass
class _Template:
    category: str
    difficulty: Difficulty
    outcome: Outcome
    is_trap: bool
    count: int  # how many cases to generate from this template
    therapeutic_areas: list[str]
    make_notes: object  # callable(rng, idx) -> (notes_str, reasoning_list)


def _case(
    *,
    case_id: str,
    compound_id: str,
    therapeutic_area: list[str],
    notes: str,
    outcome: Outcome,
    reasoning: list[str],
    difficulty: Difficulty,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA,
            cpt_code=compound_id,
            icd10_codes=therapeutic_area,
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=notes,
        policy_excerpt=DRUG_DESIGN_GUIDELINES,
        ground_truth_outcome=outcome,
        ground_truth_reasoning=reasoning,
        difficulty=difficulty,
    )


# ── Property generation helpers ────────────────────────────────────────────

def _clean_props(rng):
    """Generate a clean small-molecule profile (all pass)."""
    return {
        "MW": rng.randint(300, 480),
        "cLogP": round(rng.uniform(1.5, 3.4), 1),
        "HBD": rng.randint(1, 3),
        "HBA": rng.randint(3, 8),
        "TPSA": rng.randint(50, 120),
        "rotbonds": rng.randint(3, 8),
        "ic50_nM": rng.randint(5, 500),
        "selectivity_x": rng.choice([100, 150, 200, 300, 500]),
        "herg_uM": rng.randint(15, 50),
        "clint": rng.randint(10, 80),
        "solubility": rng.randint(20, 150),
        "caco2_papp": round(rng.uniform(10, 30), 1),
        "efflux_ratio": round(rng.uniform(0.8, 1.8), 1),
        "sa_score": round(rng.uniform(2.0, 4.5), 1),
    }


# ── EASY templates ─────────────────────────────────────────────────────────

def _clean_kinase(rng, idx):
    p = _clean_props(rng)
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt, fam = rng.choice(KINASE_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} kinase inhibitor. "
        f"Target: {tgt}. MW {p['MW']}, cLogP {p['cLogP']}, HBD {p['HBD']}, HBA {p['HBA']}, "
        f"TPSA {p['TPSA']}, rotatable bonds {p['rotbonds']}. "
        f"IC50 on {tgt} = {p['ic50_nM']} nM. "
        f"Selectivity: >{p['selectivity_x']}x vs broad kinase panel. "
        f"hERG IC50 = {p['herg_uM']} µM (clean). "
        f"CYP panel: no inhibition < 10 µM. "
        f"Caco-2 Papp = {p['caco2_papp']} x10⁻⁶ cm/s, efflux ratio {p['efflux_ratio']}. "
        f"HLM CLint = {p['clint']} µL/min/mg. "
        f"Solubility {p['solubility']} µg/mL. SA Score {p['sa_score']}. "
        f"No PAINS alerts. No structural alerts."
    )
    reasoning = [
        "§1 RO5: all parameters within limits.",
        f"§3.1 hERG clean ({p['herg_uM']} µM >> 10 µM cutoff).",
        "§3 ADMET: all thresholds met.",
        f"§6 selectivity: >{p['selectivity_x']}x on kinase panel.",
        "Clean candidate → SYNTHESIZE.",
    ]
    return notes, reasoning

def _clean_gpcr(rng, idx):
    p = _clean_props(rng)
    scaf = rng.choice(GPCR_SCAFFOLDS)
    tgt, fam = rng.choice(GPCR_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} {'inverse agonist' if rng.random() > 0.5 else 'antagonist'}. "
        f"MW {p['MW']}, cLogP {p['cLogP']}, HBD {p['HBD']}, HBA {p['HBA']}, "
        f"TPSA {p['TPSA']}, rotatable bonds {p['rotbonds']}. "
        f"IC50 {tgt} = {p['ic50_nM']} nM. "
        f"Selectivity: >{p['selectivity_x']}x vs GPCR panel. "
        f"hERG IC50 = {p['herg_uM']} µM. CYP: clean. "
        f"HLM CLint = {p['clint']}. Solubility {p['solubility']} µg/mL."
    )
    reasoning = ["All properties within limits.", "GPCR selectivity adequate.", "SYNTHESIZE."]
    return notes, reasoning

def _clean_enzyme(rng, idx):
    p = _clean_props(rng)
    scaf = rng.choice(ENZYME_SCAFFOLDS)
    tgt, fam = rng.choice(ENZYME_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {p['MW']}, cLogP {p['cLogP']}, HBD {p['HBD']}, HBA {p['HBA']}. "
        f"IC50 {tgt} = {p['ic50_nM']} nM. Selectivity >{p['selectivity_x']}x. "
        f"hERG = {p['herg_uM']} µM. CYP: no TDI, no reversible inhibition < 10 µM. "
        f"HLM CLint = {p['clint']}. Solubility {p['solubility']} µg/mL. "
        f"No PAINS. No structural alerts."
    )
    reasoning = ["All ADMET within limits.", f"Selectivity >{p['selectivity_x']}x.", "SYNTHESIZE."]
    return notes, reasoning

def _multi_ro5_violation(rng, idx):
    mw = rng.randint(550, 700)
    clogp = round(rng.uniform(5.5, 7.0), 1)
    hbd = rng.randint(6, 8)
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["CDK4/6", "EGFR", "PDE4", "HDAC6"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor (oral small molecule). "
        f"MW {mw}, cLogP {clogp}, HBD {hbd}, HBA {rng.randint(11, 14)}. "
        f"IC50 = {rng.randint(5, 50)} nM. hERG = {rng.randint(15, 40)} µM. "
        f"CYP: clean. HLM CLint = {rng.randint(20, 60)}."
    )
    reasoning = [
        f"§1 RO5: MW {mw} > 500, cLogP {clogp} > 5, HBD {hbd} > 5 → 3 violations.",
        "Oral small molecule with ≥2 RO5 violations → REJECT.",
    ]
    return notes, reasoning

def _obvious_herg(rng, idx):
    herg = round(rng.uniform(1.0, 8.0), 1)
    scaf = rng.choice(KINASE_SCAFFOLDS + GPCR_SCAFFOLDS)
    tgt = rng.choice(["CDK4", "BTK", "5-HT2A", "D2"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 480)}, cLogP {round(rng.uniform(2.0, 4.5), 1)}, "
        f"HBD {rng.randint(1, 3)}, HBA {rng.randint(4, 8)}. "
        f"IC50 = {rng.randint(5, 50)} nM. hERG IC50 = {herg} µM (FLAG). "
        f"CYP: clean. HLM CLint = {rng.randint(15, 60)}."
    )
    reasoning = [f"§3.1 hERG IC50 = {herg} µM < 10 µM threshold → cardiac liability.", "REJECT."]
    return notes, reasoning

def _obvious_cyp_tdi(rng, idx):
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["FGFR2", "PI3Kα", "HDAC6", "PARP1"])
    ki = round(rng.uniform(1.0, 5.0), 1)
    kinact = round(rng.uniform(0.02, 0.08), 3)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 480)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(3, 30)} nM. Excellent selectivity. "
        f"hERG = {rng.randint(15, 40)} µM. "
        f"CYP3A4 TIME-DEPENDENT INHIBITION detected (KI = {ki} µM, kinact = {kinact} min⁻¹). "
        f"HLM CLint = {rng.randint(15, 50)}."
    )
    reasoning = ["§3.2 CYP3A4 TDI is a HARD STOP.", "REJECT regardless of potency."]
    return notes, reasoning

def _high_clint(rng, idx):
    clint = rng.randint(120, 300)
    scaf = rng.choice(KINASE_SCAFFOLDS + GPCR_SCAFFOLDS)
    tgt = rng.choice(["JAK1", "BTK", "5-HT2A", "CB1"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 480)}, cLogP {round(rng.uniform(2.0, 4.0), 1)}. "
        f"IC50 = {rng.randint(5, 100)} nM. Selectivity >200x. "
        f"hERG = {rng.randint(15, 40)} µM. CYP: clean. "
        f"HLM CLint = {clint} µL/min/mg (VERY HIGH). "
        f"Solubility {rng.randint(30, 100)} µg/mL."
    )
    reasoning = [f"§3.3 CLint = {clint} >> 100 µL/min/mg threshold.", "Metabolic instability → REJECT."]
    return notes, reasoning

def _poor_solubility(rng, idx):
    sol = round(rng.uniform(1.0, 8.0), 1)
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt = rng.choice(["CDK4/6", "EGFR", "ALK"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(400, 500)}, cLogP {round(rng.uniform(4.0, 5.0), 1)}. "
        f"IC50 = {rng.randint(5, 50)} nM. hERG = {rng.randint(15, 40)} µM. CYP: clean. "
        f"HLM CLint = {rng.randint(15, 60)}. "
        f"Aqueous solubility = {sol} µg/mL at pH 6.8 (FLAG — below 10 µg/mL threshold). "
        f"No salt form or co-crystal identified."
    )
    reasoning = [f"§3.4 Solubility {sol} µg/mL < 10 µg/mL.", "Development risk, no formulation fix identified → REJECT."]
    return notes, reasoning

def _weak_potency(rng, idx):
    ic50 = rng.randint(5000, 20000)
    sel = rng.randint(10, 50)
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["CDK4", "PDE4", "HDAC6"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 480)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 on {tgt} = {ic50} nM (weak). Selectivity: only {sel}x vs panel. "
        f"hERG = {rng.randint(15, 40)} µM. CYP: clean."
    )
    reasoning = [f"IC50 = {ic50} nM (very weak).", f"Selectivity only {sel}x.", "Not viable → REJECT."]
    return notes, reasoning


# ── MEDIUM templates ───────────────────────────────────────────────────────

def _herg_risk_factor_trap(rng, idx):
    """hERG > 10 uM but cLogP > 3.5 + basic nitrogen → still a risk."""
    herg = rng.randint(11, 18)
    clogp = round(rng.uniform(3.8, 4.8), 1)
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt, _ = rng.choice(KINASE_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(380, 480)}, cLogP {clogp}, HBD {rng.randint(1, 3)}, HBA {rng.randint(4, 7)}. "
        f"Contains basic piperidine nitrogen (pKa = {round(rng.uniform(7.5, 9.0), 1)}). "
        f"IC50 = {rng.randint(5, 50)} nM. Selectivity >200x. "
        f"hERG IC50 = {herg} µM. "
        f"CYP: clean. HLM CLint = {rng.randint(15, 60)}."
    )
    reasoning = [
        f"§3.1 hERG = {herg} µM (above 10 µM cutoff) but cLogP {clogp} > 3.5 PLUS basic nitrogen present.",
        "Risk factor combination → hERG liability probability > 70%.",
        "REJECT — check hERG before advancing.",
    ]
    return notes, reasoning

def _herg_borderline_clean(rng, idx):
    """hERG 10-15 uM with NO risk factors → acceptable."""
    herg = rng.randint(10, 15)
    clogp = round(rng.uniform(1.5, 3.2), 1)
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt, _ = rng.choice(KINASE_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 450)}, cLogP {clogp}, HBD {rng.randint(1, 2)}, HBA {rng.randint(4, 7)}. "
        f"No basic nitrogen centers. No planar aromatic systems > 3 fused rings. "
        f"IC50 = {rng.randint(10, 200)} nM. Selectivity >{rng.choice([100, 150, 200])}x. "
        f"hERG IC50 = {herg} µM. "
        f"CYP: clean. HLM CLint = {rng.randint(15, 60)}. Solubility {rng.randint(30, 100)} µg/mL."
    )
    reasoning = [
        f"§3.1 hERG = {herg} µM (at/above 10 µM), no risk factors (cLogP {clogp} < 3.5, no basic N).",
        "hERG acceptable → SYNTHESIZE.",
    ]
    return notes, reasoning

def _pgp_efflux_mw_trap(rng, idx):
    """Poor bioavailability from P-gp efflux, not MW. Deny."""
    er = round(rng.uniform(2.5, 5.0), 1)
    scaf = rng.choice(GPCR_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["5-HT2A", "D2", "PDE4", "DPP-4"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 420)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}, "
        f"HBD {rng.randint(2, 4)}, HBA {rng.randint(4, 7)}. "
        f"IC50 = {rng.randint(10, 100)} nM. hERG = {rng.randint(20, 40)} µM. "
        f"Oral bioavailability in rat: < 5%. "
        f"Caco-2: Papp A→B = {round(rng.uniform(2.0, 8.0), 1)} x10⁻⁶ cm/s, "
        f"efflux ratio (BA/AB) = {er} (FLAG — P-gp substrate). "
        f"HLM CLint = {rng.randint(15, 50)}."
    )
    reasoning = [
        f"§2.1 + §3.5: Poor bioavailability is due to P-gp efflux (ER = {er}), not passive permeability.",
        "MW reduction will NOT fix this — P-gp is an active transport problem.",
        "Requires HBD reduction or P-gp recognition motif blocking → REJECT current form.",
    ]
    return notes, reasoning

def _ro5_macrocycle_exception(rng, idx):
    """Macrocyclic peptide with RO5 violations → approve (§1.1a)."""
    mtype = rng.choice(MACROCYCLE_TYPES)
    tgt = rng.choice(["PPI-target", "intracellular p53-MDM2", "β-catenin/TCF", "BRD4-NUT"])
    mw = rng.randint(600, 900)
    hbd = rng.randint(6, 10)
    hba = rng.randint(10, 16)
    notes = (
        f"Candidate VDC-S{idx:03d}: {mtype} targeting {tgt} (protein-protein interaction). "
        f"MW {mw}, cLogP {round(rng.uniform(2.0, 4.0), 1)}, HBD {hbd}, HBA {hba}. "
        f"LIPINSKI VIOLATIONS: MW > 500, HBD > 5, HBA > 10 (3 violations). "
        f"Cell permeability via passive membrane diffusion: Papp = {round(rng.uniform(1.0, 5.0), 1)} x10⁻⁶ cm/s. "
        f"IC50 = {rng.randint(20, 200)} nM in cell-based assay. "
        f"Oral bioavailability {rng.randint(15, 35)}% in rat (demonstrated)."
    )
    reasoning = [
        "§1.1(a): Macrocyclic peptide targeting intracellular PPI — RO5 does NOT apply.",
        "Cell permeability and oral bioavailability demonstrated.",
        "SYNTHESIZE despite RO5 violations.",
    ]
    return notes, reasoning

def _ro5_protac_exception(rng, idx):
    """PROTAC degrader with RO5 violations → approve (§1.1b)."""
    tgt = rng.choice(PROTAC_TARGETS)
    mw = rng.randint(800, 1100)
    notes = (
        f"Candidate VDC-S{idx:03d}: PROTAC degrader targeting {tgt}. "
        f"Bifunctional molecule: {tgt}-binding warhead + VHL E3 ligase recruiter + PEG linker. "
        f"MW {mw}, cLogP {round(rng.uniform(3.0, 5.5), 1)}, HBD {rng.randint(4, 8)}, HBA {rng.randint(12, 20)}. "
        f"LIPINSKI VIOLATIONS: MW >> 500, HBA >> 10 (≥2 violations). "
        f"DC50 ({tgt} degradation) = {rng.randint(5, 50)} nM. "
        f"Dmax = {rng.randint(85, 99)}%. "
        f"Oral bioavailability {rng.randint(10, 25)}% in rat."
    )
    reasoning = [
        "§1.1(b): PROTAC degrader molecule — RO5 does NOT apply.",
        f"{tgt} degradation confirmed.",
        "SYNTHESIZE despite RO5 violations.",
    ]
    return notes, reasoning

def _cyp_tdi_despite_potency(rng, idx):
    """Exceptional potency but CYP3A4 TDI → hard stop."""
    ic50 = round(rng.uniform(0.1, 2.0), 1)
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["EGFR", "ALK", "PARP1", "HDAC6"])
    ki = round(rng.uniform(1.0, 5.0), 1)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {ic50} nM (EXCEPTIONAL potency). "
        f"Selectivity: >500x. hERG = {rng.randint(20, 50)} µM. "
        f"CYP3A4 TIME-DEPENDENT INHIBITION detected (KI = {ki} µM, kinact = {round(rng.uniform(0.02, 0.06), 3)} min⁻¹). "
        f"All other properties excellent."
    )
    reasoning = [
        "§3.2 CYP3A4 TDI is a HARD STOP regardless of potency.",
        f"IC50 = {ic50} nM is exceptional but irrelevant — TDI cannot be fixed without scaffold change.",
        "REJECT.",
    ]
    return notes, reasoning

def _cns_tpsa_violation(rng, idx):
    """CNS compound with TPSA > 90 → deny."""
    tpsa = rng.randint(92, 120)
    scaf = rng.choice(CNS_SCAFFOLDS)
    tgt, fam = rng.choice(CNS_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} {'antagonist' if rng.random() > 0.5 else 'inverse agonist'} "
        f"for {'schizophrenia' if 'HT' in tgt or 'D2' in tgt else 'insomnia' if 'OX' in tgt else 'neurodegeneration'} (CNS target). "
        f"MW {rng.randint(350, 440)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}, "
        f"HBD {rng.randint(1, 3)}, TPSA {tpsa} (FLAG — exceeds CNS threshold of 90). "
        f"IC50 = {rng.randint(5, 50)} nM. hERG = {rng.randint(15, 40)} µM."
    )
    reasoning = [
        f"§4 CNS criteria: TPSA {tpsa} > 90 threshold.",
        "Poor BBB penetration predicted → REJECT for CNS indication.",
    ]
    return notes, reasoning

def _cns_pgp_hard_exclusion(rng, idx):
    """CNS compound that is a P-gp substrate → hard exclusion."""
    er = round(rng.uniform(2.5, 5.0), 1)
    scaf = rng.choice(CNS_SCAFFOLDS)
    tgt, fam = rng.choice(CNS_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} modulator for "
        f"{'schizophrenia' if 'HT' in tgt or 'D2' in tgt else 'insomnia' if 'OX' in tgt else 'epilepsy'} (CNS target). "
        f"MW {rng.randint(320, 440)}, cLogP {round(rng.uniform(2.0, 3.0), 1)}, "
        f"HBD {rng.randint(1, 2)}, TPSA {rng.randint(60, 85)}. "
        f"IC50 = {rng.randint(5, 50)} nM. Good selectivity. "
        f"P-gp substrate: efflux ratio = {er} in MDR1-MDCK (FLAG). "
        f"hERG = {rng.randint(20, 40)} µM. HLM CLint = {rng.randint(15, 50)}."
    )
    reasoning = [
        f"§4 P-gp substrate (ER = {er}) is a HARD EXCLUSION for CNS targets.",
        "Efflux at BBB prevents brain exposure regardless of passive permeability.",
        "REJECT.",
    ]
    return notes, reasoning

def _prodrug_form_trap(rng, idx):
    """Prodrug with poor in vitro props — should evaluate parent. Approve."""
    parent_ic50 = rng.randint(3, 30)
    parent_f = rng.randint(40, 75)
    tgt = rng.choice(["JAK1", "JAK2", "PDE4", "DPP-4", "SGLT2"])
    ester_type = rng.choice(["ester", "phosphate", "carbonate", "amino acid ester"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {ester_type} prodrug of a {tgt} inhibitor. "
        f"The PRODRUG form shows poor in vitro properties: IC50 on {tgt} > 10 µM (inactive), "
        f"HLM CLint = {rng.randint(200, 400)} µL/min/mg (very high clearance), "
        f"Caco-2 Papp low. HOWEVER: the prodrug is designed to be cleaved by "
        f"{'intestinal esterases' if 'ester' in ester_type else 'alkaline phosphatase' if 'phosphate' in ester_type else 'plasma esterases'} "
        f"to release the active parent compound, which has IC50 {tgt} = {parent_ic50} nM, "
        f"excellent ADMET, and oral bioavailability of {parent_f}% in rat (as prodrug formulation)."
    )
    reasoning = [
        "Prodrug in vitro properties are EXPECTED to look poor.",
        f"Parent: IC50 = {parent_ic50} nM, clean ADMET.",
        f"Prodrug formulation: {parent_f}% oral F in rat.",
        "SYNTHESIZE — evaluate as prodrug, not as parent.",
    ]
    return notes, reasoning

def _pains_unvalidated(rng, idx):
    """PAINS flag with no orthogonal validation → deny."""
    pains_type = rng.choice(["rhodanine", "catechol", "quinone methide", "hydroxamic acid", "enone"])
    scaf = rng.choice(ENZYME_SCAFFOLDS)
    tgt = rng.choice(["HDAC6", "PDE4", "IDO1", "NAMPT"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(50, 500)} nM (biochemical). "
        f"PAINS alert: {pains_type} substructure flagged. "
        f"No orthogonal binding validation available (no SPR, ITC, or co-crystal data). "
        f"Cellular potency: IC50 = {rng.randint(2000, 10000)} nM "
        f"(large biochemical-to-cellular shift, {rng.randint(10, 50)}-fold)."
    )
    reasoning = [
        f"§5.1 PAINS alert ({pains_type}) with no orthogonal validation.",
        "Large biochemical-to-cellular shift suggests non-specific binding.",
        "REJECT until binding mechanism validated.",
    ]
    return notes, reasoning

def _fluorine_cyp2d6(rng, idx):
    """Fluorine substitution to fix CLint but introduces CYP2D6 → deny."""
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt = rng.choice(["EGFR", "ALK", "FGFR2"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor, fluorinated analog. "
        f"MW {rng.randint(380, 460)}, cLogP {round(rng.uniform(2.5, 3.8), 1)}. "
        f"IC50 = {rng.randint(5, 50)} nM. "
        f"Parent compound had HLM CLint = {rng.randint(120, 200)} (metabolically unstable). "
        f"Fluorine substitution at C-{rng.randint(3, 7)} reduced CLint to {rng.randint(25, 55)} µL/min/mg. "
        f"HOWEVER: fluorinated analog now shows CYP2D6 inhibition IC50 = {round(rng.uniform(1.0, 5.0), 1)} µM "
        f"(FLAG — narrow therapeutic index co-medications affected). "
        f"hERG = {rng.randint(15, 35)} µM."
    )
    reasoning = [
        "§3.3 Fluorination fixed CLint but introduced CYP2D6 inhibition.",
        "CYP2D6 at low µM is a liability for NTI co-medications.",
        "REJECT — need alternative metabolic stability strategy.",
    ]
    return notes, reasoning

def _moderate_potency_clean(rng, idx):
    """100-500 nM potency but all ADMET excellent → approve."""
    ic50 = rng.randint(100, 500)
    scaf = rng.choice(KINASE_SCAFFOLDS + GPCR_SCAFFOLDS)
    tgt = rng.choice(["BTK", "JAK1", "5-HT2A", "D2"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(330, 420)}, cLogP {round(rng.uniform(1.5, 2.8), 1)}, "
        f"HBD {rng.randint(1, 2)}, HBA {rng.randint(3, 6)}. "
        f"IC50 = {ic50} nM (moderate — not sub-nM). "
        f"Selectivity >{rng.choice([150, 200, 300])}x. "
        f"hERG = {rng.randint(20, 50)} µM. CYP: clean. "
        f"HLM CLint = {rng.randint(10, 40)}. Solubility {rng.randint(50, 150)} µg/mL. "
        f"Excellent oral F ({rng.randint(50, 80)}% in rat)."
    )
    reasoning = [
        f"IC50 = {ic50} nM is moderate but all ADMET properties excellent.",
        "§8.2: Do not chase sub-nM potency when drug properties are already good.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning


# ── HARD templates ─────────────────────────────────────────────────────────

def _pains_validated_binding(rng, idx):
    """PAINS flag BUT orthogonal validation confirms binding → approve."""
    pains_type = rng.choice(["catechol", "hydroxamic acid", "metal-chelating motif"])
    validation = rng.choice(["SPR (KD = 25 nM)", "ITC (KD = 40 nM, stoichiometry 1:1)", "co-crystal structure at 1.8 Å resolution"])
    scaf = rng.choice(ENZYME_SCAFFOLDS)
    tgt = rng.choice(["HDAC6", "MMP-9", "LpxC", "ADAM17"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 450)}, cLogP {round(rng.uniform(1.5, 3.0), 1)}. "
        f"IC50 = {rng.randint(10, 100)} nM. "
        f"PAINS alert: {pains_type} substructure. "
        f"Orthogonal binding validation: {validation}. "
        f"Cellular potency: IC50 = {rng.randint(30, 300)} nM (consistent with biochemical). "
        f"hERG = {rng.randint(20, 40)} µM. CYP: clean."
    )
    reasoning = [
        f"§5.1 PAINS alert ({pains_type}) BUT validated by {validation.split('(')[0].strip()}.",
        "Cellular potency consistent with biochemical → specific binding confirmed.",
        "SYNTHESIZE despite PAINS flag.",
    ]
    return notes, reasoning

def _allosteric_potency(rng, idx):
    """Allosteric modulator with 100-1000 nM potency → approve (§6.2)."""
    ic50 = rng.randint(100, 800)
    tgt = rng.choice(["mGlu5", "GABAB", "GluN2B", "CaSR", "P2Y1"])
    mechanism = rng.choice(["negative allosteric modulator", "positive allosteric modulator"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {mechanism} of {tgt}. "
        f"MW {rng.randint(320, 440)}, cLogP {round(rng.uniform(1.5, 3.0), 1)}. "
        f"Functional EC50/IC50 = {ic50} nM. "
        f"NOTE: this is an ALLOSTERIC mechanism — potency is evaluated on functional efficacy, "
        f"not raw binding affinity. "
        f"Selectivity >{rng.choice([100, 200])}x vs related family members. "
        f"hERG = {rng.randint(20, 40)} µM. CYP: clean. "
        f"HLM CLint = {rng.randint(15, 60)}. Oral F = {rng.randint(30, 60)}%."
    )
    reasoning = [
        f"§6.2 Allosteric modulator: {ic50} nM is within expected 100-1000 nM range.",
        "Evaluate on functional efficacy, not raw IC50.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning

def _multi_target_designed(rng, idx):
    """Designed polypharmacology — moderate activity on 2-3 targets is intentional."""
    t1 = rng.choice(["CDK4", "CDK6", "EGFR", "VEGFR2"])
    t2 = rng.choice(["CDK2", "HER2", "PDGFR"])
    notes = (
        f"Candidate VDC-S{idx:03d}: designed dual {t1}/{t2} inhibitor. "
        f"MW {rng.randint(380, 480)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 {t1} = {rng.randint(5, 30)} nM, IC50 {t2} = {rng.randint(15, 80)} nM. "
        f"SELECTIVITY NOTE: {t1}/{t2} dual inhibition is BY DESIGN (polypharmacology). "
        f"Selectivity vs UNINTENDED off-targets: >200x (hERG, Nav1.5, Cav1.2 all clean). "
        f"hERG = {rng.randint(25, 50)} µM. CYP: clean. "
        f"HLM CLint = {rng.randint(15, 55)}."
    )
    reasoning = [
        f"§6.1: Dual {t1}/{t2} is designed polypharmacology, not poor selectivity.",
        "Evaluate selectivity against UNINTENDED off-targets only.",
        "Safety panel clean → SYNTHESIZE.",
    ]
    return notes, reasoning

def _oncology_genotox_exception(rng, idx):
    """Oncology compound with genotoxicity/reactive met alert → approve if unmet need."""
    alert = rng.choice(["aromatic amine", "nitroaromatic", "alkylating motif"])
    cancer = rng.choice([
        "pancreatic adenocarcinoma (no standard of care)",
        "triple-negative breast cancer (limited options)",
        "glioblastoma multiforme (poor prognosis)",
        "cholangiocarcinoma (high unmet need)",
    ])
    notes = (
        f"Candidate VDC-S{idx:03d}: cytotoxic agent for {cancer}. "
        f"MW {rng.randint(350, 480)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(5, 50)} nM against tumor cell line. "
        f"Structural alert: {alert} (genotoxicity flag). "
        f"Ames test: positive in TA98 strain. "
        f"HOWEVER: compound is cytotoxic by design for oncology with high unmet need. "
        f"Anti-tumor efficacy: {rng.randint(60, 90)}% TGI in xenograft model."
    )
    reasoning = [
        f"§5.3 + §9.1: Genotoxicity alert ({alert}) in oncology context.",
        f"Indication: {cancer.split('(')[0].strip()} — high unmet need.",
        "Oncology benefit-risk calculus accepts these alerts.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning

def _oncology_reactive_met_gst(rng, idx):
    """Reactive metabolite but rapid GST clearance in oncology → approve."""
    notes = (
        f"Candidate VDC-S{idx:03d}: covalent kinase inhibitor for "
        f"{rng.choice(['NSCLC', 'CRC', 'AML'])} (oncology). "
        f"MW {rng.randint(380, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(5, 30)} nM. "
        f"Structural alert: predicted reactive epoxide metabolite. "
        f"HOWEVER: GST conjugation assay shows rapid clearance "
        f"(t½ = {round(rng.uniform(1.0, 5.0), 1)} min, GST CLint = {rng.randint(200, 500)} µL/min/mg). "
        f"In vivo: no glutathione adducts detected in bile duct-cannulated rat. "
        f"hERG = {rng.randint(20, 40)} µM. CYP: clean."
    )
    reasoning = [
        "§5.2: Reactive metabolite alert BUT rapid GST conjugation mitigates risk.",
        "§9.1: Oncology benefit-risk further supports advancement.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning

def _species_dependent_metab(rng, idx):
    """Good rat PK but human-relevant metabolism pathway present → deny."""
    enzyme = rng.choice(["aldehyde oxidase (AO)", "UGT1A4", "FMO3"])
    species_note = {
        "aldehyde oxidase (AO)": "AO activity is HIGH in humans but minimal in rats/dogs",
        "UGT1A4": "UGT1A4 glucuronidation is extensive in humans but minimal in rodents",
        "FMO3": "FMO3 N-oxidation is significant in humans but has different isoform distribution in rats",
    }[enzyme]
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["JAK1", "BTK", "PDE4", "HDAC6"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(5, 50)} nM. Selectivity >200x. "
        f"Rat PK: oral F = {rng.randint(55, 80)}%, CL = {rng.randint(5, 15)} mL/min/kg (excellent). "
        f"Dog PK: oral F = {rng.randint(50, 70)}%. "
        f"SPECIES NOTE: compound contains {enzyme} substrate motif. "
        f"{species_note}. "
        f"Predicted human CLint (using recombinant {enzyme.split('(')[0].strip()}) = "
        f"{rng.randint(150, 350)} µL/min/mg."
    )
    reasoning = [
        f"§9.2: Good rat/dog PK but {enzyme} substrate.",
        f"{species_note}.",
        "Cannot advance based on rodent data alone.",
        "REJECT.",
    ]
    return notes, reasoning

def _flat_sar(rng, idx):
    """10+ analogs with no SAR trend → deprioritize."""
    n_analogs = rng.randint(10, 18)
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["CDK4", "PI3Kα", "PDE4", "HDAC6"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor, member of lead series LS-{rng.randint(100, 999)}. "
        f"MW {rng.randint(380, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(200, 800)} nM. "
        f"SAR NOTE: {n_analogs} structural analogs have been synthesized in this series. "
        f"IC50 range across all analogs: {rng.randint(150, 400)}-{rng.randint(600, 1200)} nM. "
        f"No measurable SAR trend — modifications at R1, R2, R3 positions all give "
        f"flat activity with no clear optimization handle. "
        f"No new binding hypothesis proposed."
    )
    reasoning = [
        f"§8.1: {n_analogs} analogs with flat SAR, no optimization handle.",
        "No new hypothesis proposed.",
        "Deprioritize this series → REJECT.",
    ]
    return notes, reasoning

def _potency_over_optimization(rng, idx):
    """Sub-nM potency but poor ADMET → reject (potency trap)."""
    ic50 = round(rng.uniform(0.05, 0.8), 2)
    clint = rng.randint(120, 250)
    clogp = round(rng.uniform(4.5, 6.0), 1)
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt = rng.choice(["EGFR", "ALK", "ABL1", "MET"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(450, 520)}, cLogP {clogp}. "
        f"IC50 = {ic50} nM (EXCEPTIONAL — sub-nanomolar). "
        f"Selectivity: >100x vs kinase panel. "
        f"HOWEVER: hERG = {rng.randint(8, 14)} µM (borderline), "
        f"HLM CLint = {clint} µL/min/mg (high), "
        f"Solubility = {rng.randint(5, 15)} µg/mL (poor). "
        f"Oral F = {rng.randint(5, 15)}% in rat."
    )
    reasoning = [
        f"§8.2: IC50 = {ic50} nM is sub-nM but ADMET is poor (CLint {clint}, cLogP {clogp}).",
        "Do not prioritize further potency optimization when drug properties are unacceptable.",
        "REJECT — fix ADMET before advancing.",
    ]
    return notes, reasoning

def _selectivity_below_100x(rng, idx):
    """Potent but < 100x selectivity on safety-relevant off-target → deny."""
    sel = rng.randint(20, 80)
    off_tgt = rng.choice(["hERG (ion channel)", "Nav1.5 (cardiac)", "Cav1.2 (cardiac)"])
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt = rng.choice(["CDK4", "EGFR", "JAK1", "BTK"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(370, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 {tgt} = {rng.randint(3, 20)} nM. "
        f"SELECTIVITY FLAG: only {sel}x selectivity vs {off_tgt}. "
        f"CYP: clean. HLM CLint = {rng.randint(15, 50)}."
    )
    reasoning = [
        f"§6: Selectivity {sel}x < 100x vs safety-relevant off-target {off_tgt}.",
        "Hard stop for safety off-targets.",
        "REJECT.",
    ]
    return notes, reasoning

def _ro5_exception_misapplied(rng, idx):
    """Oral small molecule claims modality exception but isn't one → deny."""
    mw = rng.randint(550, 650)
    hbd = rng.randint(6, 8)
    scaf = rng.choice(KINASE_SCAFFOLDS)
    tgt = rng.choice(["CDK4", "EGFR", "PI3Kα"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor (oral small molecule). "
        f"MW {mw}, cLogP {round(rng.uniform(5.2, 6.5), 1)}, HBD {hbd}, HBA {rng.randint(11, 14)}. "
        f"LIPINSKI VIOLATIONS: MW > 500, cLogP > 5, HBD > 5 (3 violations). "
        f"Team claims RO5 exception under §1.1, citing 'peptide-like character.' "
        f"HOWEVER: this is a standard oral small molecule — NOT a macrocyclic peptide, "
        f"PROTAC, natural product, or ADC payload. "
        f"IC50 = {rng.randint(10, 100)} nM."
    )
    reasoning = [
        "§1.1 exception does NOT apply — this is a standard oral small molecule.",
        "'Peptide-like character' does not qualify for any exemption category.",
        f"3 RO5 violations (MW {mw}, cLogP > 5, HBD {hbd}) → REJECT.",
    ]
    return notes, reasoning

def _covalent_warhead_gst(rng, idx):
    """Covalent inhibitor with reactive warhead but rapid GST clearance → approve."""
    warhead = rng.choice(["acrylamide", "chloroacetamide", "vinyl sulfone"])
    tgt = rng.choice(["EGFR C797", "BTK C481", "KRAS G12C"])
    notes = (
        f"Candidate VDC-S{idx:03d}: covalent {tgt} inhibitor with {warhead} warhead. "
        f"MW {rng.randint(370, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(3, 30)} nM (kinact/KI = {round(rng.uniform(5000, 20000), 0)} M⁻¹s⁻¹). "
        f"Structural alert: {warhead} reactive warhead. "
        f"GST reactivity: t½ = {round(rng.uniform(30, 120), 0)} min (moderate). "
        f"Selectivity of covalent modification: >100x vs other reactive cysteines in proteome. "
        f"hERG = {rng.randint(20, 40)} µM. CYP: clean."
    )
    reasoning = [
        f"§5.2: {warhead} warhead is a deliberate covalent mechanism, not an off-target alert.",
        "Selectivity of covalent modification confirmed.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning

def _prodrug_with_real_liability(rng, idx):
    """Prodrug form AND parent both have issues → deny."""
    tgt = rng.choice(["JAK1", "PDE4", "DPP-4"])
    parent_herg = round(rng.uniform(3.0, 8.0), 1)
    notes = (
        f"Candidate VDC-S{idx:03d}: ester prodrug of a {tgt} inhibitor. "
        f"The PRODRUG form shows poor in vitro properties (expected for a prodrug). "
        f"The ACTIVE PARENT compound has IC50 {tgt} = {rng.randint(5, 30)} nM. "
        f"HOWEVER: parent compound also shows hERG IC50 = {parent_herg} µM (FLAG — below 10 µM) "
        f"and CYP3A4 TIME-DEPENDENT INHIBITION (KI = {round(rng.uniform(2.0, 5.0), 1)} µM). "
        f"The prodrug strategy fixes bioavailability but NOT the parent's safety liabilities."
    )
    reasoning = [
        "Prodrug strategy noted — evaluate parent compound properties.",
        f"§3.1: Parent hERG = {parent_herg} µM < 10 µM.",
        "§3.2: Parent CYP3A4 TDI is a HARD STOP.",
        "Prodrug fixes PK but not safety → REJECT.",
    ]
    return notes, reasoning

def _reversible_cyp_acceptable(rng, idx):
    """Reversible CYP inhibition at > 10 µM → acceptable."""
    cyp_ic50 = rng.randint(12, 30)
    cyp = rng.choice(["CYP3A4", "CYP2C9", "CYP2C19"])
    scaf = rng.choice(KINASE_SCAFFOLDS + ENZYME_SCAFFOLDS)
    tgt = rng.choice(["CDK4", "BTK", "HDAC6", "PARP1"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {scaf} {tgt} inhibitor. "
        f"MW {rng.randint(350, 460)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}. "
        f"IC50 = {rng.randint(5, 50)} nM. Selectivity >200x. "
        f"hERG = {rng.randint(20, 40)} µM. "
        f"CYP panel: {cyp} REVERSIBLE inhibition IC50 = {cyp_ic50} µM. "
        f"No time-dependent inhibition detected. "
        f"HLM CLint = {rng.randint(15, 50)}. Solubility {rng.randint(30, 100)} µg/mL."
    )
    reasoning = [
        f"§3.2: {cyp} reversible inhibition at {cyp_ic50} µM (> 10 µM threshold) is acceptable.",
        "No TDI detected.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning

def _cns_double_failure(rng, idx):
    """CNS with both TPSA > 90 AND P-gp substrate → deny."""
    tpsa = rng.randint(95, 115)
    er = round(rng.uniform(3.0, 6.0), 1)
    tgt, _ = rng.choice(CNS_TARGETS)
    notes = (
        f"Candidate VDC-S{idx:03d}: {rng.choice(CNS_SCAFFOLDS)} {tgt} modulator (CNS target). "
        f"MW {rng.randint(380, 450)}, cLogP {round(rng.uniform(2.0, 3.5), 1)}, "
        f"HBD {rng.randint(2, 4)}, TPSA {tpsa} (FLAG — exceeds CNS 90 threshold). "
        f"P-gp substrate: efflux ratio = {er} (FLAG). "
        f"IC50 = {rng.randint(5, 30)} nM. hERG = {rng.randint(20, 40)} µM."
    )
    reasoning = [
        f"§4: TPSA {tpsa} > 90 (passive barrier) AND P-gp ER {er} (active barrier).",
        "Both passive and active transport failures for CNS.",
        "REJECT.",
    ]
    return notes, reasoning

def _molecular_obesity(rng, idx):
    """All RO5 params individually borderline pass but predicted poor oral F."""
    mw = rng.randint(490, 499)
    clogp = round(rng.uniform(4.5, 4.9), 1)
    hbd = 5
    hba = rng.randint(9, 10)
    tpsa = rng.randint(135, 139)
    rotbonds = rng.randint(9, 10)
    notes = (
        f"Candidate VDC-S{idx:03d}: {rng.choice(KINASE_SCAFFOLDS)} {rng.choice(['EGFR', 'CDK4', 'JAK1'])} inhibitor. "
        f"MW {mw}, cLogP {clogp}, HBD {hbd}, HBA {hba}, "
        f"TPSA {tpsa}, rotatable bonds {rotbonds}. "
        f"LIPINSKI: technically 0 violations (all parameters at upper boundary). "
        f"VEBER: TPSA {tpsa} ≤ 140 AND rotbonds {rotbonds} ≤ 10 (borderline pass). "
        f"HOWEVER: predicted oral bioavailability < 5% from combined property burden. "
        f"Rat oral F = {rng.randint(2, 8)}%. "
        f"IC50 = {rng.randint(10, 50)} nM."
    )
    reasoning = [
        f"All RO5/Veber parameters borderline pass but combined property burden is too high.",
        "Predicted and observed oral F < 10%.",
        "'Molecular obesity' — REJECT.",
    ]
    return notes, reasoning

def _ro5_natural_product(rng, idx):
    """Natural product with oral absorption demonstrated → approve despite RO5."""
    mw = rng.randint(520, 650)
    source = rng.choice(["marine sponge-derived", "fungal metabolite", "plant alkaloid-derived", "soil actinomycete"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {source} natural product scaffold. "
        f"MW {mw}, cLogP {round(rng.uniform(2.0, 4.0), 1)}, "
        f"HBD {rng.randint(3, 6)}, HBA {rng.randint(8, 14)}. "
        f"LIPINSKI VIOLATIONS: MW > 500, HBA > 10 (2 violations). "
        f"HOWEVER: natural product with demonstrated oral absorption — "
        f"oral F = {rng.randint(20, 45)}% in rat, {rng.randint(15, 35)}% in dog. "
        f"IC50 = {rng.randint(10, 100)} nM. hERG = {rng.randint(20, 40)} µM. CYP: clean."
    )
    reasoning = [
        "§1.1(c): Natural product scaffold with demonstrated oral absorption.",
        "RO5 does not apply — oral bioavailability confirmed in two species.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning

def _ro5_adc_payload(rng, idx):
    """ADC payload evaluated for parenteral delivery → approve despite RO5."""
    mw = rng.randint(700, 1000)
    tgt = rng.choice(["HER2", "TROP2", "Nectin-4", "FOLR1"])
    payload = rng.choice(["auristatin derivative", "maytansinoid", "camptothecin analog", "PBD dimer"])
    notes = (
        f"Candidate VDC-S{idx:03d}: {payload} payload for {tgt}-directed ADC. "
        f"Intended delivery: antibody-drug conjugate (parenteral, IV infusion). "
        f"MW {mw}, cLogP {round(rng.uniform(2.0, 5.0), 1)}, HBD {rng.randint(4, 8)}, HBA {rng.randint(10, 18)}. "
        f"LIPINSKI VIOLATIONS: MW >> 500, HBA > 10 (≥2 violations). "
        f"Cytotoxicity IC50 = {round(rng.uniform(0.01, 1.0), 2)} nM (free payload). "
        f"DAR = {rng.choice([2, 4, 8])}. Linker: {rng.choice(['cleavable protease', 'non-cleavable thioether'])}."
    )
    reasoning = [
        "§1.1(d): ADC payload evaluated for parenteral delivery — RO5 does NOT apply.",
        "Oral bioavailability is irrelevant for IV-administered ADC.",
        "SYNTHESIZE.",
    ]
    return notes, reasoning


# ── Template registry ──────────────────────────────────────────────────────

TEMPLATES: list[_Template] = [
    # EASY — clear pass/fail (48 cases)
    _Template("clean_kinase", Difficulty.EASY, Outcome.APPROVED, False, 5, ["ONCOLOGY", "KINASE"], _clean_kinase),
    _Template("clean_gpcr", Difficulty.EASY, Outcome.APPROVED, False, 5, ["CNS", "GPCR"], _clean_gpcr),
    _Template("clean_enzyme", Difficulty.EASY, Outcome.APPROVED, False, 4, ["METABOLIC", "ENZYME"], _clean_enzyme),
    _Template("multi_ro5", Difficulty.EASY, Outcome.DENIED, False, 5, ["ONCOLOGY", "KINASE"], _multi_ro5_violation),
    _Template("obvious_herg", Difficulty.EASY, Outcome.DENIED, False, 5, ["ONCOLOGY", "KINASE"], _obvious_herg),
    _Template("obvious_cyp_tdi", Difficulty.EASY, Outcome.DENIED, False, 5, ["ONCOLOGY", "KINASE"], _obvious_cyp_tdi),
    _Template("high_clint", Difficulty.EASY, Outcome.DENIED, False, 5, ["INFLAMMATION", "KINASE"], _high_clint),
    _Template("poor_solubility", Difficulty.EASY, Outcome.DENIED, False, 4, ["ONCOLOGY", "KINASE"], _poor_solubility),
    _Template("weak_potency", Difficulty.EASY, Outcome.DENIED, False, 4, ["ONCOLOGY", "ENZYME"], _weak_potency),

    # MEDIUM — single-trap or single-exception (72 cases)
    _Template("herg_risk_trap", Difficulty.MEDIUM, Outcome.DENIED, True, 6, ["ONCOLOGY", "KINASE"], _herg_risk_factor_trap),
    _Template("herg_borderline_clean", Difficulty.MEDIUM, Outcome.APPROVED, True, 6, ["ONCOLOGY", "KINASE"], _herg_borderline_clean),
    _Template("pgp_efflux_trap", Difficulty.MEDIUM, Outcome.DENIED, True, 6, ["CNS", "GPCR"], _pgp_efflux_mw_trap),
    _Template("ro5_macrocycle", Difficulty.MEDIUM, Outcome.APPROVED, True, 6, ["ONCOLOGY", "PPI"], _ro5_macrocycle_exception),
    _Template("ro5_protac", Difficulty.MEDIUM, Outcome.APPROVED, True, 5, ["ONCOLOGY", "PROTAC"], _ro5_protac_exception),
    _Template("cyp_tdi_potency", Difficulty.MEDIUM, Outcome.DENIED, False, 6, ["ONCOLOGY", "KINASE"], _cyp_tdi_despite_potency),
    _Template("cns_tpsa", Difficulty.MEDIUM, Outcome.DENIED, False, 5, ["CNS", "GPCR"], _cns_tpsa_violation),
    _Template("cns_pgp", Difficulty.MEDIUM, Outcome.DENIED, False, 5, ["CNS", "GPCR"], _cns_pgp_hard_exclusion),
    _Template("prodrug_trap", Difficulty.MEDIUM, Outcome.APPROVED, True, 6, ["INFLAMMATION", "KINASE"], _prodrug_form_trap),
    _Template("pains_unvalidated", Difficulty.MEDIUM, Outcome.DENIED, False, 5, ["ONCOLOGY", "ENZYME"], _pains_unvalidated),
    _Template("fluorine_cyp2d6", Difficulty.MEDIUM, Outcome.DENIED, False, 5, ["ONCOLOGY", "KINASE"], _fluorine_cyp2d6),
    _Template("moderate_potency", Difficulty.MEDIUM, Outcome.APPROVED, False, 5, ["INFLAMMATION", "KINASE"], _moderate_potency_clean),
    _Template("ro5_natural_product", Difficulty.MEDIUM, Outcome.APPROVED, True, 3, ["ONCOLOGY", "NATURAL"], _ro5_natural_product),
    _Template("ro5_adc_payload", Difficulty.MEDIUM, Outcome.APPROVED, True, 3, ["ONCOLOGY", "ADC"], _ro5_adc_payload),

    # HARD — multi-conflict, rare exceptions (80 cases)
    _Template("pains_validated", Difficulty.HARD, Outcome.APPROVED, True, 6, ["ONCOLOGY", "ENZYME"], _pains_validated_binding),
    _Template("allosteric_potency", Difficulty.HARD, Outcome.APPROVED, True, 6, ["CNS", "GPCR"], _allosteric_potency),
    _Template("multi_target", Difficulty.HARD, Outcome.APPROVED, True, 6, ["ONCOLOGY", "KINASE"], _multi_target_designed),
    _Template("oncology_genotox", Difficulty.HARD, Outcome.APPROVED, True, 6, ["ONCOLOGY", "CYTOTOXIC"], _oncology_genotox_exception),
    _Template("oncology_reactive_gst", Difficulty.HARD, Outcome.APPROVED, True, 6, ["ONCOLOGY", "KINASE"], _oncology_reactive_met_gst),
    _Template("species_metab", Difficulty.HARD, Outcome.DENIED, True, 6, ["INFLAMMATION", "KINASE"], _species_dependent_metab),
    _Template("flat_sar", Difficulty.HARD, Outcome.DENIED, True, 6, ["ONCOLOGY", "KINASE"], _flat_sar),
    _Template("potency_over_opt", Difficulty.HARD, Outcome.DENIED, True, 6, ["ONCOLOGY", "KINASE"], _potency_over_optimization),
    _Template("selectivity_below_100x", Difficulty.HARD, Outcome.DENIED, False, 6, ["ONCOLOGY", "KINASE"], _selectivity_below_100x),
    _Template("ro5_misapplied", Difficulty.HARD, Outcome.DENIED, True, 5, ["ONCOLOGY", "KINASE"], _ro5_exception_misapplied),
    _Template("covalent_gst", Difficulty.HARD, Outcome.APPROVED, True, 6, ["ONCOLOGY", "KINASE"], _covalent_warhead_gst),
    _Template("prodrug_real_liability", Difficulty.HARD, Outcome.DENIED, True, 5, ["INFLAMMATION", "KINASE"], _prodrug_with_real_liability),
    _Template("reversible_cyp", Difficulty.HARD, Outcome.APPROVED, False, 6, ["ONCOLOGY", "KINASE"], _reversible_cyp_acceptable),
    _Template("cns_double", Difficulty.HARD, Outcome.DENIED, False, 5, ["CNS", "GPCR"], _cns_double_failure),
    _Template("molecular_obesity", Difficulty.HARD, Outcome.DENIED, True, 5, ["ONCOLOGY", "KINASE"], _molecular_obesity),
]


def build_synthetic_drugdisc_cases(seed: int = 42) -> list[BenchmarkCase]:
    """Generate 200 synthetic drug discovery benchmark cases."""
    rng = random.Random(seed)
    cases: list[BenchmarkCase] = []
    idx = 1

    for tmpl in TEMPLATES:
        for _ in range(tmpl.count):
            notes, reasoning = tmpl.make_notes(rng, idx)
            cases.append(_case(
                case_id=f"dd_syn_{idx:03d}",
                compound_id=f"VDC-S{idx:03d}",
                therapeutic_area=tmpl.therapeutic_areas,
                notes=notes,
                outcome=tmpl.outcome,
                reasoning=reasoning,
                difficulty=tmpl.difficulty,
            ))
            idx += 1

    # Shuffle deterministically so templates aren't in blocks
    rng.shuffle(cases)

    # Validation
    n = len(cases)
    n_trap = sum(1 for t in TEMPLATES for _ in range(t.count) if t.is_trap)
    n_approved = sum(1 for c in cases if c.ground_truth_outcome == Outcome.APPROVED)
    n_denied = n - n_approved

    assert n == 200, f"Expected 200 cases, got {n}"
    assert 60 <= n_trap <= 120, f"Expected ~40% traps, got {n_trap} ({n_trap/n*100:.0f}%)"
    assert 80 <= n_approved <= 120, f"Expected ~100 approved, got {n_approved}"

    return cases


if __name__ == "__main__":
    cases = build_synthetic_drugdisc_cases()
    n_trap = sum(1 for t in TEMPLATES for _ in range(t.count) if t.is_trap)
    n_easy = sum(1 for c in cases if c.difficulty == Difficulty.EASY)
    n_med = sum(1 for c in cases if c.difficulty == Difficulty.MEDIUM)
    n_hard = sum(1 for c in cases if c.difficulty == Difficulty.HARD)
    n_app = sum(1 for c in cases if c.ground_truth_outcome == Outcome.APPROVED)
    print(f"Generated {len(cases)} cases")
    print(f"  EASY: {n_easy}, MEDIUM: {n_med}, HARD: {n_hard}")
    print(f"  APPROVED: {n_app}, DENIED: {len(cases) - n_app}")
    print(f"  Traps: {n_trap} ({n_trap/len(cases)*100:.0f}%)")
    print(f"  Categories: {len(TEMPLATES)}")
