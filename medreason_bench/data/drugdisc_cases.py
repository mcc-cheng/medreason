"""v0.1 drug discovery benchmark — 25 lead optimization triage cases.

Each case represents a decision point in a lead optimization campaign:
given a candidate molecule's property profile + design guidelines,
should the molecule be SYNTHESIZED (approved for synthesis and further
testing) or REJECTED (archived due to predicted liabilities)?

The cases are organized in three tiers:
  Tier 1 (7 cases, EASY):   Straightforward decisions. No traps.
  Tier 2 (10 cases, MEDIUM): Single-parameter optimization traps.
  Tier 3 (8 cases, HARD):    Multi-parameter tradeoffs, adversarial.

Outcome distribution: 13 synthesize (approved) + 12 reject (denied).

Field mapping to prior-auth BenchmarkCase:
  payer         → Payer.AETNA (placeholder, ignored by retrieval)
  cpt_code      → compound identifier string (e.g., "VDC-001")
  icd10_codes   → therapeutic area tags (e.g., ["ONCOLOGY", "KINASE"])
  facility_type → FacilityType.OUTPATIENT (placeholder)
  clinical_notes → molecule property profile (MW, cLogP, IC50, ADMET)
  policy_excerpt → drug design guidelines (the "policy" the agent
                   checks against)
  ground_truth_outcome → APPROVED = synthesize, DENIED = reject
"""

from __future__ import annotations

from medreason.ontology import (
    BenchmarkCase,
    DenialReason,
    Difficulty,
    FacilityType,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
)


# ── Drug design guidelines ("policy") ──────────────────────────────────────

DRUG_DESIGN_GUIDELINES = """Lead Optimization Triage Guidelines — Computational Chemistry Review Standard v1.0

§1 LIPINSKI RULE OF FIVE (RO5)
Reject oral small-molecule candidates that violate ≥2 of: MW ≤ 500, cLogP ≤ 5, HBD ≤ 5, HBA ≤ 10.
§1.1 EXCEPTIONS: RO5 was derived from orally-absorbed small molecules and does NOT apply to:
  (a) Macrocyclic peptides targeting intracellular protein-protein interactions
  (b) PROTAC (proteolysis-targeting chimera) degrader molecules
  (c) Natural product-derived scaffolds with demonstrated oral absorption
  (d) Antibody-drug conjugate (ADC) payloads evaluated for parenteral delivery
  Check the intended modality and delivery route BEFORE applying RO5 filters.

§2 VEBER RULES FOR ORAL BIOAVAILABILITY
For oral candidates: TPSA ≤ 140 Å² AND rotatable bonds ≤ 10. Violations predict poor oral bioavailability.
§2.1 NOTE: Poor oral bioavailability can arise from EITHER passive permeability failure (TPSA/MW-driven) OR active efflux (P-glycoprotein substrate). These have DIFFERENT solutions — reducing MW fixes passive permeability but does NOT fix P-gp efflux. Always check the efflux ratio (BA/AB in Caco-2) BEFORE optimizing MW.

§3 ADMET THRESHOLDS
§3.1 hERG CARDIOTOXICITY: hERG IC50 < 10 µM is a liability flag. Risk factors: cLogP > 3.5 combined with basic nitrogen (pKa > 7), planar aromatic systems, and certain pharmacophore patterns. hERG IC50 ≥ 10 µM with no risk factors = acceptable.
§3.2 CYP INHIBITION: Time-dependent inhibition (TDI) of CYP3A4 is a hard stop — deprioritize regardless of potency. Reversible CYP inhibition at > 10 µM is acceptable. CYP2D6 inhibition requires careful evaluation especially for drugs with narrow therapeutic index co-medications.
§3.3 METABOLIC STABILITY: Intrinsic clearance (CLint) > 100 µL/min/mg in human liver microsomes is a liability. Fluorine substitution at metabolic soft spots can improve stability but introduces risk of CYP2D6 inhibition or defluorination — check both before and after fluorination.
§3.4 SOLUBILITY: Aqueous solubility < 10 µg/mL at pH 6.8 is a development risk for oral formulation. Consider salt forms, co-crystals, or amorphous solid dispersions before rejecting.
§3.5 P-GLYCOPROTEIN EFFLUX: Efflux ratio (BA/AB) > 2 in Caco-2 indicates P-gp substrate liability. This is an ACTIVE TRANSPORT problem, NOT a passive permeability problem. Reducing molecular weight will NOT fix P-gp efflux. Strategies: reduce HBD count, increase lipophilicity modestly, or add methyl groups to block P-gp recognition motifs.

§4 CNS-SPECIFIC CRITERIA (applies only to CNS targets)
For blood-brain barrier penetration: TPSA ≤ 90 Å², MW ≤ 450, HBD ≤ 3, cLogP 1-3 optimal range. P-gp substrate status is a hard exclusion for CNS — efflux at the BBB prevents brain exposure regardless of passive permeability. CNS MPO score ≥ 4 is the composite threshold.

§5 STRUCTURAL ALERTS
§5.1 PAINS FILTERS: Compounds matching PAINS (Pan-Assay Interference Compounds) substructures should be flagged. HOWEVER: not all PAINS alerts represent genuine interference. Some PAINS scaffolds (e.g., certain catechols, hydroxamic acids) are legitimate pharmacophores when the binding mechanism is validated by orthogonal assays (SPR, ITC, crystallography). Do NOT auto-reject PAINS-flagged compounds with validated binding.
§5.2 REACTIVE METABOLITE ALERTS: Structural features predicted to form reactive metabolites (epoxides, quinones, acyl glucuronides, nitroso intermediates) are liabilities. EXCEPTION: If the reactive intermediate is rapidly conjugated (e.g., by glutathione S-transferase with demonstrated high GST clearance), the risk is mitigated. Context matters: oncology programs accept higher reactive-met risk than chronic-disease programs due to different benefit-risk calculus.
§5.3 GENOTOXICITY ALERTS: Aromatic amines, nitroaromatics, and alkylating motifs require Ames test data before advancement. EXCEPTION: In oncology indications where the therapeutic benefit outweighs genotoxicity risk AND the compound is cytotoxic by design (e.g., DNA-damaging agents), these alerts are evaluated differently.

§6 SELECTIVITY REQUIREMENTS
Primary target selectivity ≥ 100x vs critical off-target panel (kinase selectivity panel for kinase inhibitors, GPCR panel for GPCR ligands, ion channel panel for channel modulators). Selectivity < 100x on safety-relevant off-targets (hERG, Nav1.5, Cav1.2) is a hard stop.
§6.1 EXCEPTION — MULTI-TARGET PHARMACOLOGY: Deliberately designed polypharmacology agents (e.g., dual kinase inhibitors, multi-target CNS agents) may have intentional moderate activity on 2-3 targets. Evaluate selectivity against UNINTENDED off-targets, not designed targets.
§6.2 EXCEPTION — ALLOSTERIC MECHANISMS: Allosteric modulators may show lower absolute potency (IC50 100-1000 nM) than orthosteric inhibitors. This does NOT indicate weak binding — allosteric mechanisms have different potency scaling. Evaluate allosteric candidates on functional efficacy and selectivity, not raw IC50.

§7 SYNTHETIC ACCESSIBILITY
SA Score > 6 (RDKit scale 1-10) flags synthetic difficulty. Candidates requiring > 12 linear synthetic steps, hazardous reagents (organolithium at scale, heavy metals, azides), or key steps with < 30% yield should be deprioritized unless the compound addresses an unmet structural gap.

§8 RESOURCE ALLOCATION
§8.1 SAR TRACTABILITY: If ≥ 10 structural analogs have been synthesized with no measurable SAR trend (flat SAR), the chemical series may lack an optimization handle. Deprioritize unless a new hypothesis (e.g., binding mode change, scaffold hop) is proposed.
§8.2 POTENCY-DRIVEN OVER-OPTIMIZATION: Do not prioritize further potency optimization below IC50 < 1 nM unless selectivity, ADMET, and PK are already acceptable. Sub-nanomolar potency with poor drug-like properties is not a viable candidate.

§9 CONTEXT-DEPENDENT EVALUATION
§9.1 ONCOLOGY BENEFIT-RISK: Oncology programs accept higher toxicity risk, including structural alerts for genotoxicity and reactive metabolites, IF the compound demonstrates meaningful anti-tumor efficacy and the indication has high unmet need. Apply §5.2 and §5.3 with oncology-adjusted risk tolerance.
§9.2 SPECIES-DEPENDENT METABOLISM: Rat/mouse PK data may not predict human PK. Compounds showing favorable rat PK but predicted to have different human metabolic pathways (e.g., aldehyde oxidase metabolism absent in rat) should not be advanced based on rodent data alone without human-relevant metabolic assessment."""


# ── Case builder ────────────────────────────────────────────────────────────


def _case(
    *,
    case_id: str,
    compound_id: str,
    therapeutic_area: list[str],
    notes: str,
    policy: str = DRUG_DESIGN_GUIDELINES,
    outcome: Outcome,
    reasoning: list[str],
    difficulty: Difficulty,
    denial_reason: DenialReason | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA,  # placeholder
            cpt_code=compound_id,
            icd10_codes=therapeutic_area,
            facility_type=FacilityType.OUTPATIENT,  # placeholder
            denial_reason=denial_reason,
        ),
        clinical_notes=notes,
        policy_excerpt=policy,
        ground_truth_outcome=outcome,
        ground_truth_reasoning=reasoning,
        difficulty=difficulty,
    )


def build_drugdisc_cases() -> list[BenchmarkCase]:
    """Return the v0.1 drug discovery benchmark (25 cases)."""
    cases: list[BenchmarkCase] = []

    # ═══════════════════════════════════════════════════════════════════
    # TIER 1 — STRAIGHTFORWARD (7 cases: 4 synthesize + 3 reject)
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="dd_001",
        compound_id="VDC-001",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-001: pyrimidine-based kinase inhibitor. "
            "Target: CDK4/6. MW 425, cLogP 2.8, HBD 2, HBA 6, "
            "TPSA 78, rotatable bonds 5. IC50 on CDK4 = 12 nM, "
            "CDK6 = 18 nM. Selectivity: >200x vs CDK1, CDK2, CDK5. "
            "hERG IC50 = 32 µM (clean). CYP panel: no inhibition "
            "< 10 µM. Caco-2 Papp A→B = 18 x10⁻⁶ cm/s, efflux ratio "
            "1.1. HLM CLint = 22 µL/min/mg. Aqueous solubility 85 "
            "µg/mL at pH 6.8. SA Score 3.2. No PAINS alerts. No "
            "structural alerts."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§1 RO5: all parameters within limits.",
            "§3.1 hERG clean (32 µM >> 10 µM cutoff).",
            "§3 ADMET: all thresholds met.",
            "§6 selectivity: >200x on kinase panel.",
            "Clean candidate, no flags → SYNTHESIZE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="dd_002",
        compound_id="VDC-002",
        therapeutic_area=["METABOLIC", "GPCR"],
        notes=(
            "Candidate VDC-002: biphenyl GPCR modulator. Target: GLP-1R. "
            "MW 680, cLogP 7.2, HBD 3, HBA 4, TPSA 52, rotatable "
            "bonds 12. IC50 on GLP-1R = 45 nM. hERG IC50 = 4.8 µM "
            "(FLAG). Aqueous solubility 0.8 µg/mL (very poor). HLM "
            "CLint = 180 µL/min/mg (high clearance). Intended route: "
            "oral small molecule."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§1 RO5: 3 violations (MW 680, cLogP 7.2, RotBonds 12).",
            "§1.1: oral small molecule, no modality exception applies.",
            "§3.1 hERG liability (4.8 µM < 10 µM).",
            "§3.3 metabolic instability (CLint 180).",
            "§3.4 solubility failure (0.8 µg/mL).",
            "Multiple independent liabilities → REJECT.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="dd_003",
        compound_id="VDC-003",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-003: indazole-based VEGFR2 inhibitor. "
            "MW 410, cLogP 3.1, HBD 2, HBA 5, TPSA 72. IC50 on "
            "VEGFR2 = 8 nM. Selectivity: >150x vs VEGFR1, PDGFR, "
            "FGFR panel. hERG IC50 = 28 µM. CYP clean. Caco-2 good "
            "(Papp 22, ER 0.9). HLM CLint = 18. Solubility 120 µg/mL. "
            "SA Score 2.8."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "All RO5, ADMET, selectivity, SA criteria met.",
            "Clean lead — no optimization concerns.",
            "SYNTHESIZE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="dd_004",
        compound_id="VDC-004",
        therapeutic_area=["INFLAMMATION", "PROTEASE"],
        notes=(
            "Candidate VDC-004: small molecule MMP-13 inhibitor. "
            "MW 380, cLogP 2.2, HBD 3, HBA 7, TPSA 95. IC50 on "
            "MMP-13 = 18 µM (weak). Selectivity vs MMP-1: 2x. "
            "All ADMET parameters acceptable. SA Score 2.1."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "IC50 = 18 µM is too weak for a lead candidate.",
            "§6 selectivity: only 2x vs MMP-1 (safety-relevant).",
            "Weak potency + poor selectivity = no optimization path.",
            "REJECT.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="dd_005",
        compound_id="VDC-005",
        therapeutic_area=["ONCOLOGY", "EPIGENETIC"],
        notes=(
            "Candidate VDC-005: quinazoline HDAC6 inhibitor with "
            "hydroxamic acid zinc-binding group. MW 385, cLogP 1.9, "
            "HBD 3, HBA 6, TPSA 88. IC50 HDAC6 = 5 nM. Selectivity: "
            ">500x vs HDAC1, HDAC2, HDAC3. hERG = 45 µM. CYP clean. "
            "Solubility 60 µg/mL. HLM CLint = 35. SA Score 3.5."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "All property criteria met.",
            "Excellent isoform selectivity.",
            "Hydroxamic acid is a known pharmacophore for HDAC, "
            "not a nonspecific chelator in this context.",
            "SYNTHESIZE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="dd_006",
        compound_id="VDC-006",
        therapeutic_area=["INFECTIOUS", "ENZYME"],
        notes=(
            "Candidate VDC-006: thiazolidinone derivative, target: "
            "bacterial FabI. MW 320, cLogP 2.5. PAINS filter: "
            "FLAGGED — rhodanine substructure detected. IC50 on "
            "FabI = 2 µM in biochemical assay. No orthogonal binding "
            "validation (no SPR, no ITC, no co-crystal). "
            "Cell-based activity IC50 = 25 µM (13x shift)."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§5.1 PAINS: rhodanine flagged.",
            "No orthogonal binding validation available.",
            "Large biochemical-to-cell shift (13x) consistent with "
            "nonspecific interference.",
            "REJECT until binding validated by SPR or crystallography.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="dd_007",
        compound_id="VDC-007",
        therapeutic_area=["CARDIOVASCULAR", "ION_CHANNEL"],
        notes=(
            "Candidate VDC-007: dihydropyridine calcium channel "
            "blocker. MW 390, cLogP 2.6, HBD 1, HBA 5, TPSA 65. "
            "IC50 on Cav1.2 = 15 nM. Selectivity: >300x vs Nav1.5, "
            "hERG. CYP clean, Caco-2 good (Papp 25, ER 0.8), "
            "HLM CLint = 28, solubility 95 µg/mL. SA Score 2.5."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "All criteria met. Excellent selectivity vs cardiac "
            "ion channels. Clean ADMET. Good drug-likeness.",
            "SYNTHESIZE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2 — SINGLE-PARAMETER TRAPS (10 cases: 5 synth + 5 reject)
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="dd_008",
        compound_id="VDC-008",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-008: aminopyrimidine JAK2 inhibitor. "
            "MW 445, cLogP 4.5, HBD 2, HBA 5, TPSA 68. IC50 JAK2 = "
            "0.8 nM (sub-nanomolar potency). Contains basic "
            "dimethylamine (pKa 8.2). hERG IC50 = 3.2 µM (FLAG). "
            "Selectivity: >100x vs JAK1, JAK3, TYK2. CYP clean. "
            "Solubility 45 µg/mL. HLM CLint = 25."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§3.1 hERG liability: 3.2 µM < 10 µM cutoff.",
            "Risk factors present: cLogP 4.5 + basic nitrogen "
            "(pKa 8.2) = high hERG probability.",
            "Sub-nanomolar potency is meaningless if the compound "
            "has cardiac toxicity risk.",
            "§8.2: potency over-optimization without ADMET balance.",
            "REJECT — fix hERG before further advancement.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_009",
        compound_id="VDC-009",
        therapeutic_area=["METABOLIC", "GPCR"],
        notes=(
            "Candidate VDC-009: oral GLP-1R agonist. MW 470, "
            "cLogP 3.0, HBD 4, HBA 8, TPSA 110. IC50 = 35 nM. "
            "hERG = 22 µM (clean). CYP clean. Caco-2: Papp A→B = "
            "3.2 x10⁻⁶ cm/s (low), efflux ratio = 8.3 (FLAG — "
            "P-gp substrate). Oral bioavailability in rat: 2%. "
            "HLM CLint = 15 (stable). Solubility 55 µg/mL (OK). "
            "SA Score 3.8."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§2.1 + §3.5: oral bioavailability 2% is driven by "
            "P-gp efflux (ER 8.3), NOT passive permeability.",
            "Reducing MW will NOT fix P-gp efflux — this is an "
            "active transport problem.",
            "Must address P-gp recognition motifs (reduce HBD, "
            "modify hydrogen bond topology).",
            "REJECT as-is — redesign needed, not MW optimization.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_010",
        compound_id="VDC-010",
        therapeutic_area=["ONCOLOGY", "PPI"],
        notes=(
            "Candidate VDC-010: macrocyclic peptide. Target: MDM2-p53 "
            "protein-protein interaction (intracellular). MW 780, "
            "cLogP 3.8, HBD 7, HBA 14, TPSA 185, rotatable bonds "
            "15. IC50 on MDM2-p53 disruption = 18 nM. Cell-based "
            "p53 reporter EC50 = 45 nM (2.5x shift — excellent "
            "for a macrocycle). Demonstrated cell permeability via "
            "passive diffusion (chloroalkane penetration assay). "
            "hERG = 50 µM (clean). SA Score 5.8."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§1 RO5: multiple violations. BUT §1.1(a): macrocyclic "
            "peptide targeting intracellular PPI — RO5 does NOT "
            "apply to this modality.",
            "Demonstrated cell permeability despite high MW.",
            "Small biochemical-to-cell shift (2.5x) confirms "
            "cellular target engagement.",
            "SYNTHESIZE — modality exception applies.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_011",
        compound_id="VDC-011",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-011: pyrazolopyrimidine ALK inhibitor. "
            "MW 430, cLogP 3.2, HBD 2, HBA 6. IC50 ALK = 3 nM. "
            "Selectivity: >500x vs broad kinase panel. hERG = 18 µM. "
            "CYP panel: CYP3A4 TIME-DEPENDENT INHIBITION detected "
            "(KI = 2.5 µM, kinact = 0.04 min⁻¹). CYP2D6 clean. "
            "HLM CLint = 20. Solubility 75 µg/mL."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§3.2 CYP3A4 TDI is a HARD STOP.",
            "Time-dependent inhibition creates risk of DDI with "
            "co-medications metabolized by CYP3A4.",
            "Cannot be fixed without scaffold modification.",
            "REJECT regardless of potency and other properties.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_012",
        compound_id="VDC-012",
        therapeutic_area=["CNS", "GPCR"],
        notes=(
            "Candidate VDC-012: piperazine-based 5-HT2A inverse "
            "agonist for schizophrenia (CNS target). MW 420, "
            "cLogP 3.8, HBD 1, HBA 4, TPSA 95 (FLAG — exceeds "
            "CNS threshold of 90). IC50 5-HT2A = 8 nM. "
            "Selectivity: >200x vs D2, 5-HT2C. P-gp substrate "
            "status: efflux ratio 3.5 in MDR1-MDCK (FLAG). "
            "hERG = 25 µM (clean). HLM CLint = 30."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§4 CNS criteria: TPSA 95 exceeds 90 threshold.",
            "§4 P-gp substrate (ER 3.5) is a HARD EXCLUSION for "
            "CNS — prevents brain penetration at BBB.",
            "Both passive and active transport barriers present.",
            "REJECT — needs structural redesign for CNS exposure.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_013",
        compound_id="VDC-013",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-013: imidazopyridine BRAF V600E inhibitor. "
            "MW 460, cLogP 3.5, HBD 2, HBA 7, TPSA 85. IC50 BRAF "
            "V600E = 200 nM (moderate). Selectivity: >500x vs "
            "CRAF, >200x vs broad kinase panel (442/450 kinases "
            "<50% inhibition at 1 µM). hERG = 40 µM. CYP clean. "
            "Caco-2 good. HLM CLint = 22. Solubility 90 µg/mL. "
            "SA Score 3.0. No structural alerts."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "IC50 200 nM is moderate — not sub-nanomolar — BUT "
            "all other properties are excellent.",
            "§8.2: do not over-optimize potency when ADMET and "
            "selectivity are already in range.",
            "Exceptional kinase selectivity (442/450 clean).",
            "SYNTHESIZE — potency is adequate for a clean lead.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_014",
        compound_id="VDC-014",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-014: fluorinated pyrrolopyrimidine EGFR "
            "inhibitor. MW 415, cLogP 3.0. Fluorine added at the "
            "C-3 position of the phenyl ring to block CYP-mediated "
            "oxidation (metabolic soft spot). Result: HLM CLint "
            "improved from 85 → 18 µL/min/mg. HOWEVER: CYP2D6 "
            "inhibition IC50 = 1.8 µM (new liability introduced "
            "by fluorine). IC50 EGFR = 5 nM. hERG = 20 µM."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§3.3: fluorine substitution improved metabolic stability "
            "BUT introduced CYP2D6 inhibition (1.8 µM).",
            "CYP2D6 at 1.8 µM is a significant DDI risk for "
            "narrow-therapeutic-index co-medications.",
            "Net effect of fluorination is negative.",
            "REJECT — explore alternative soft-spot blocking strategy.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_015",
        compound_id="VDC-015",
        therapeutic_area=["ONCOLOGY", "E3_LIGASE"],
        notes=(
            "Candidate VDC-015: PROTAC BET degrader. MW 950, "
            "cLogP 4.8, HBD 5, HBA 16, TPSA 220, rotatable bonds "
            "22. DC50 (degradation) BRD4 = 1.2 nM. Selectivity: "
            "degrades BRD4 > BRD2 > BRD3 with >50x selectivity "
            "vs non-BET bromodomains. Cellular degradation confirmed "
            "by Western blot. hERG = 35 µM. Intended route: "
            "intravenous infusion."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§1 RO5: massive violations. BUT §1.1(b): PROTAC "
            "modality — RO5 does NOT apply.",
            "§1.1(d): IV delivery — oral bioavailability irrelevant.",
            "Potent degradation (DC50 1.2 nM) with BET selectivity.",
            "Cellular activity confirmed.",
            "SYNTHESIZE — modality exception applies.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_016",
        compound_id="VDC-016",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-016: aminoquinoline FLT3 inhibitor. "
            "MW 395, cLogP 3.6, HBD 2, HBA 5. Contains basic "
            "piperidine nitrogen (pKa 7.8). IC50 FLT3 = 15 nM. "
            "hERG IC50 = 12 µM. Selectivity: >100x vs KIT, PDGFR. "
            "CYP clean. HLM CLint = 32. Solubility 55 µg/mL."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§3.1 hERG: IC50 12 µM > 10 µM cutoff = acceptable.",
            "Basic nitrogen present (pKa 7.8) but hERG IC50 is "
            "above the threshold — do NOT over-flag.",
            "All other criteria met.",
            "SYNTHESIZE — hERG is borderline-safe, not a liability.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="dd_017",
        compound_id="VDC-017",
        therapeutic_area=["INFLAMMATION", "KINASE"],
        notes=(
            "Candidate VDC-017: ester prodrug of a JAK1 inhibitor. "
            "The PRODRUG form shows poor in vitro properties: IC50 "
            "on JAK1 > 10 µM (inactive), HLM CLint = 250 µL/min/mg "
            "(very high clearance), Caco-2 Papp low. HOWEVER: the "
            "prodrug is designed to be cleaved by intestinal "
            "esterases to release the active parent compound, which "
            "has IC50 JAK1 = 8 nM, excellent ADMET, and oral "
            "bioavailability of 65% in rat (as prodrug formulation)."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Prodrug in vitro properties are EXPECTED to look poor.",
            "The relevant data is the active parent's properties "
            "AND the prodrug formulation's in vivo PK.",
            "Parent: IC50 8 nM, clean ADMET.",
            "Prodrug formulation: 65% oral F in rat.",
            "SYNTHESIZE — evaluate as prodrug, not as parent.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 3 — ADVERSARIAL / MULTI-PARAMETER (8 cases: 4 synth + 4 rej)
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="dd_018",
        compound_id="VDC-018",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-018: pyridopyrimidine multi-kinase "
            "inhibitor. IC50 ABL1 = 5 nM (potent). HOWEVER: "
            "selectivity panel shows IC50 SRC = 22 nM (4.4x), "
            "IC50 LCK = 45 nM (9x), IC50 FYN = 80 nM (16x). "
            "Overall kinase selectivity: 15x against closest "
            "off-target (SRC). All ADMET properties acceptable. "
            "MW 435, cLogP 3.1. SA Score 3.2."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§6 selectivity: 15x vs SRC is far below the 100x "
            "threshold for kinase inhibitors.",
            "SRC, LCK, FYN are all safety-relevant — inhibition "
            "causes immunosuppression.",
            "Potency is meaningless without adequate selectivity.",
            "REJECT — requires scaffold redesign for selectivity.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_019",
        compound_id="VDC-019",
        therapeutic_area=["ONCOLOGY", "DNA_DAMAGE"],
        notes=(
            "Candidate VDC-019: nitrogen mustard alkylating agent "
            "with aromatic amine moiety. MW 310, cLogP 1.5. "
            "Structural alert: aromatic amine (§5.3 genotoxicity). "
            "Structural alert: alkylating motif (§5.2 reactive met). "
            "IC50 on tumor cell panel (A549, HCT116, MCF7) = "
            "50-120 nM. Selectivity index vs normal fibroblasts: "
            "8-12x. Indication: refractory AML with no standard "
            "of care remaining. Unmet medical need: HIGH."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§5.2 + §5.3: structural alerts present for both "
            "reactive metabolite and genotoxicity.",
            "HOWEVER: §9.1 oncology benefit-risk applies.",
            "Compound is cytotoxic BY DESIGN (DNA alkylator).",
            "Refractory AML with high unmet need adjusts the "
            "risk threshold.",
            "SYNTHESIZE under §9.1 oncology exception.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_020",
        compound_id="VDC-020",
        therapeutic_area=["METABOLIC", "GPCR"],
        notes=(
            "Candidate VDC-020: oral small molecule. MW 498, "
            "cLogP 4.9, HBD 4, HBA 9, TPSA 130, rotatable bonds "
            "9. Technically passes RO5 on every individual parameter "
            "(MW < 500, cLogP < 5, HBD < 5, HBA < 10). IC50 = 25 nM. "
            "Caco-2 Papp = 2.1 x10⁻⁶ cm/s (borderline). Efflux "
            "ratio = 1.8 (borderline). Predicted oral F = 8%. "
            "HLM CLint = 55 (moderate)."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§1 RO5: technically passes each individual parameter.",
            "BUT: the compound has 'molecular obesity' — TPSA 130 "
            "violates §2 Veber (TPSA ≤ 140 borderline), predicted "
            "oral F is only 8%.",
            "RO5 compliance does not guarantee drug-likeness when "
            "all parameters are at their limits simultaneously.",
            "Multiple borderline properties + poor predicted F → "
            "REJECT.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_021",
        compound_id="VDC-021",
        therapeutic_area=["METABOLIC", "ENZYME"],
        notes=(
            "Candidate VDC-021: aldehyde-containing DPP-4 inhibitor "
            "(oral, chronic use). MW 360, cLogP 1.8. Rat PK: "
            "oral F = 52%, t1/2 = 4.2 hours (excellent). Dog PK: "
            "oral F = 45%, t1/2 = 3.8 hours. HOWEVER: compound "
            "contains an aldehyde oxidase (AO) substrate motif. "
            "AO activity is high in humans but absent in rats and "
            "dogs. Predicted human AO-mediated clearance: CLint "
            "= 320 µL/min/mg (using human cytosol)."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§9.2 species-dependent metabolism: rat and dog PK "
            "are not predictive of human PK for this compound.",
            "AO activity is absent in rat/dog but high in humans.",
            "Human cytosol CLint 320 µL/min/mg predicts rapid "
            "first-pass metabolism and poor human oral F.",
            "REJECT — do not advance based on rodent data alone.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_022",
        compound_id="VDC-022",
        therapeutic_area=["CNS", "GPCR"],
        notes=(
            "Candidate VDC-022: allosteric positive modulator (PAM) "
            "of mGluR5 for treatment-resistant depression. IC50 in "
            "orthosteric binding assay = not measurable (no "
            "displacement). Functional EC50 in calcium flux assay = "
            "650 nM (allosteric potentiation of glutamate response). "
            "Selectivity: >100x vs mGluR1, no activity on mGluR2/3. "
            "MW 355, cLogP 2.2, TPSA 62, HBD 1. P-gp ER = 1.1 "
            "(not a substrate). hERG = 30 µM. CNS MPO = 5.2."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§6.2 allosteric exception: lower absolute potency is "
            "expected for allosteric mechanisms.",
            "EC50 650 nM is within acceptable range for a PAM.",
            "§4 CNS criteria: all met (TPSA 62, MW 355, HBD 1, "
            "P-gp negative, MPO 5.2).",
            "SYNTHESIZE — evaluate on functional efficacy and "
            "selectivity, not raw potency.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_023",
        compound_id="VDC-023",
        therapeutic_area=["ONCOLOGY", "KINASE"],
        notes=(
            "Candidate VDC-023: latest in a series of benzimidazole "
            "PLK1 inhibitors. This is analog #14 in the series. "
            "IC50 PLK1 = 22 nM (same as analogs #3, #7, #11). "
            "SAR analysis across all 14 analogs: IC50 range 15-45 "
            "nM regardless of substitution pattern at R1, R2, R3 "
            "positions. No clear SAR trend. cLogP ranges 2.5-4.5 "
            "across series. ADMET acceptable for this analog."
        ),
        outcome=Outcome.DENIED,
        reasoning=[
            "§8.1 SAR tractability: 14 analogs with flat SAR.",
            "No optimization handle — substitution doesn't "
            "modulate potency, selectivity, or ADMET.",
            "Cannot design toward improved properties without "
            "a new scaffold hypothesis.",
            "REJECT — deprioritize series until scaffold hop or "
            "new binding mode hypothesis.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_024",
        compound_id="VDC-024",
        therapeutic_area=["CNS", "MULTI_TARGET"],
        notes=(
            "Candidate VDC-024: deliberately designed multi-target "
            "agent for Alzheimer's disease. Targets: AChE IC50 = "
            "120 nM, MAO-B IC50 = 85 nM, 5-HT4 EC50 = 250 nM. "
            "Selectivity vs unintended off-targets: >200x vs "
            "hERG, Nav1.5, Cav1.2, D2. MW 410, cLogP 2.5, "
            "TPSA 72, HBD 2. P-gp ER = 1.2. CNS MPO = 4.8."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§6.1 multi-target pharmacology exception: AChE + "
            "MAO-B + 5-HT4 are INTENTIONAL targets.",
            "Evaluate selectivity against UNINTENDED off-targets "
            "only — >200x achieved.",
            "§4 CNS criteria: all met.",
            "SYNTHESIZE — designed polypharmacology with clean "
            "off-target profile.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="dd_025",
        compound_id="VDC-025",
        therapeutic_area=["ONCOLOGY", "PROTEASE"],
        notes=(
            "Candidate VDC-025: covalent cysteine-reactive inhibitor "
            "of KRAS G12C. MW 480, cLogP 3.2, HBD 2, HBA 7. "
            "Contains acrylamide warhead — structural alert for "
            "reactive metabolite (§5.2). IC50 KRAS G12C = 8 nM "
            "(covalent). Selectivity: no activity vs KRAS WT, "
            "KRAS G12D, KRAS G12V (>1000x). GST conjugation "
            "half-life of the acrylamide = 12 minutes (rapid "
            "clearance). Daily glutathione turnover sufficient "
            "to handle therapeutic doses."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "§5.2 reactive metabolite alert: acrylamide warhead.",
            "HOWEVER: §5.2 exception — GST conjugation with "
            "rapid clearance (t1/2 12 min) mitigates reactive "
            "metabolite risk.",
            "§9.1 oncology context: KRAS G12C is a validated "
            "oncology target with high unmet need.",
            "Exquisite mutation selectivity (>1000x vs WT).",
            "SYNTHESIZE — reactive warhead risk mitigated by "
            "rapid GST clearance.",
        ],
        difficulty=Difficulty.HARD,
    ))

    return cases
