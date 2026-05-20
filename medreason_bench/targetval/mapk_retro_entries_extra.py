"""MAPK retro v0.2 — RTK + non-MAPK adjunct entries.

Twelve entries covering targets that intersect or sit adjacent to the
MAPK axis but are not RAF/MEK/ERK/RAS themselves:

- EGFR, MET, ALK, FGFR2: RTKs upstream of MAPK
- PTPN11 (SHP2): RTK-to-RAS scaffold
- CDK4, PIK3CA, WEE1, AXL, MDM2: pathways that interact with MAPK
  via parallel signalling or cell-cycle handoff
- RAF1: pan-RAF dimer-breaker case, distinct from BRAF-V600E

The MAPK-011..MAPK-022 id range belongs here. Core MAPK-axis entries
(MAPK-001..MAPK-010) live in ``mapk_retro_entries.py``.

Every entry here is Universal-safe: no internal data, every claim
sourceable via a real NCT ID or a ``per literature:`` description.
"""

from __future__ import annotations

from medreason.targetval.case import BypassMechanism, GroundTruthOutcome

from .mapk_retro_data import MapkRetroEntry


MAPK_RETRO_ENTRIES_EXTRA: tuple[MapkRetroEntry, ...] = (
    # ── EGFR / MET / ALK ────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-011",
        gene="EGFR",
        family="RTK",
        disease="metastatic_NSCLC_EGFR_exon19del_L858R",
        paralogs=("ERBB2", "ERBB3", "ERBB4"),
        paralog_count=3,
        redundancy=0.55,
        feedback_loops=(
            "MET amplification bypass",
            "AXL upregulation",
            "ERK reactivation",
        ),
        n_trials=20,
        n_p2_fail=3,
        n_approvals=4,
        trial_summary=(
            "erlotinib, gefitinib, afatinib, osimertinib all approved "
            "for EGFR-mutant NSCLC; osimertinib (FLAURA, NCT02296125) is "
            "current first-line. Resistance via T790M then C797S."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.RESISTANCE_MUTATION,
        notes=(
            "NCT02296125 (FLAURA). per literature: Soria et al. (NEJM "
            "2018) on osimertinib first-line; T790M/C797S resistance "
            "well-catalogued. Paralogs (HER2/3/4) drive a secondary "
            "bypass route documented by Engelman 2007 on MET amplification."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-012",
        gene="MET",
        family="RTK",
        disease="MET_exon14_skipping_NSCLC",
        paralogs=("MST1R",),
        paralog_count=1,
        redundancy=0.4,
        feedback_loops=(
            "EGFR cross-activation",
            "KRAS amplification on-treatment",
        ),
        n_trials=8,
        n_p2_fail=1,
        n_approvals=2,
        trial_summary=(
            "capmatinib (GEOMETRY mono-1, NCT02414139) and tepotinib "
            "(VISION, NCT02864992) approved for MET exon 14 NSCLC; "
            "secondary D1228 and Y1230 mutations drive resistance."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.RESISTANCE_MUTATION,
        notes=(
            "NCT02414139 (GEOMETRY mono-1), NCT02864992 (VISION). per "
            "literature: Wolf et al. (NEJM 2020) on capmatinib; "
            "secondary kinase-domain mutations and EGFR-bypass documented "
            "in Recondo et al. (Clin Cancer Res 2020)."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-013",
        gene="MET",
        family="RTK",
        disease="EGFR_TKI_resistant_NSCLC_MET_amplified",
        paralogs=("MST1R",),
        paralog_count=1,
        redundancy=0.5,
        feedback_loops=(
            "EGFR pathway reactivation",
            "HER3 dimerisation",
        ),
        n_trials=10,
        n_p2_fail=5,
        n_approvals=0,
        trial_summary=(
            "MET amplification as EGFR-TKI bypass: tivantinib + erlotinib "
            "(MARQUEE, NCT01244191) failed Phase 3; savolitinib + "
            "osimertinib (SAVANNAH, NCT03778229) reached only conditional "
            "signal in MET-high biomarker-selected patients."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.ALTERNATIVE_PATHWAY,
        notes=(
            "NCT01244191 (MARQUEE). per literature: Engelman et al. "
            "(Science 2007) on MET amplification as EGFR-TKI bypass; "
            "MARQUEE negative due to tivantinib's actual mechanism "
            "(tubulin) being non-MET — the bypass biology is real but "
            "this trial design didn't test it cleanly."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-014",
        gene="ALK",
        family="RTK",
        disease="ALK_rearranged_NSCLC",
        paralogs=("LTK",),
        paralog_count=1,
        redundancy=0.35,
        feedback_loops=(
            "bypass via EGFR / KIT activation",
            "MAPK reactivation post-ALK-i",
        ),
        n_trials=15,
        n_p2_fail=2,
        n_approvals=5,
        trial_summary=(
            "crizotinib, alectinib, brigatinib, ceritinib, lorlatinib all "
            "approved for ALK-rearranged NSCLC; alectinib (ALEX, "
            "NCT02075840) overtook crizotinib first-line; lorlatinib "
            "covers G1202R resistance."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.RESISTANCE_MUTATION,
        notes=(
            "NCT02075840 (ALEX, alectinib). per literature: Peters et al. "
            "(NEJM 2017) on ALEX; Gainor et al. (Cancer Discovery 2016) "
            "catalogued ALK-i resistance mutations (G1202R, L1196M, etc.)."
        ),
    ),
    # ── SHP2 (PTPN11) ───────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-015",
        gene="PTPN11",
        family="phosphatase",
        disease="KRAS_G12C_NSCLC_combo_with_SHP2_inhibitor",
        paralogs=("PTPN6",),
        paralog_count=1,
        redundancy=0.35,
        feedback_loops=(
            "RAS-GTP feedback after KRAS-G12C inhibition",
            "RTK upstream rebound",
        ),
        n_trials=5,
        n_p2_fail=2,
        n_approvals=0,
        trial_summary=(
            "TNO155, RMC-4630, JAB-3068 in Phase 1/2 combos with KRAS-G12C "
            "inhibitors (NCT04000529, NCT03634982); rationale is to block "
            "RAS-GTP reload, but durable responses have not yet read out."
        ),
        outcome=GroundTruthOutcome.ACTIVE_UNKNOWN,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "NCT04000529 (TNO155), NCT03634982 (RMC-4630). per literature: "
            "Nichols et al. (Nat Cell Biol 2018) and Fedele et al. "
            "(Cancer Discovery 2018) on SHP2-i blocking RTK-driven "
            "RAS-GTP reload after KRAS-G12C-i. [best-effort outcome] "
            "still maturing; ground-truth bypass mechanism is the "
            "documented rationale, outcome is held as ACTIVE_UNKNOWN."
        ),
    ),
    # ── FGFR / cholangiocarcinoma ──────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-016",
        gene="FGFR2",
        family="RTK",
        disease="FGFR2_fusion_cholangiocarcinoma",
        paralogs=("FGFR1", "FGFR3", "FGFR4"),
        paralog_count=3,
        redundancy=0.5,
        feedback_loops=(
            "gatekeeper V564F resistance",
            "ERK reactivation",
        ),
        n_trials=6,
        n_p2_fail=1,
        n_approvals=2,
        trial_summary=(
            "pemigatinib (FIGHT-202, NCT02924376) and futibatinib "
            "(FOENIX-CCA2, NCT02052778) approved for FGFR2-fusion+ "
            "cholangiocarcinoma; gatekeeper mutations drive resistance."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.RESISTANCE_MUTATION,
        notes=(
            "NCT02924376 (FIGHT-202). per literature: Abou-Alfa et al. "
            "(Lancet Oncol 2020) on pemigatinib; Goyal et al. (Cancer "
            "Discovery 2017) catalogued FGFR2 kinase-domain resistance."
        ),
    ),
    # ── CDK4/6 ──────────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-017",
        gene="CDK4",
        family="CDK",
        disease="HR_positive_HER2_negative_breast_cancer",
        paralogs=("CDK6",),
        paralog_count=1,
        redundancy=0.4,
        feedback_loops=(
            "RB1 loss",
            "cyclin E1 amplification (CCNE1) bypass",
            "PI3K-AKT reactivation",
        ),
        n_trials=12,
        n_p2_fail=1,
        n_approvals=3,
        trial_summary=(
            "palbociclib (PALOMA, NCT01740427), ribociclib (MONALEESA, "
            "NCT01958021), abemaciclib (MONARCH, NCT02107703) approved "
            "with endocrine therapy for HR+/HER2- breast; resistance via "
            "RB1 loss and CCNE1 amplification."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.ALTERNATIVE_PATHWAY,
        notes=(
            "NCT01740427 (PALOMA-2). per literature: Finn et al. (NEJM "
            "2016) on palbociclib + letrozole; Wander et al. (Cancer "
            "Discovery 2020) catalogued CCNE1 / RB1 resistance routes."
        ),
    ),
    # ── PI3K-alpha ─────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-018",
        gene="PIK3CA",
        family="PI3K",
        disease="HR_positive_PIK3CA_mutant_breast_cancer",
        paralogs=("PIK3CB", "PIK3CD", "PIK3CG"),
        paralog_count=3,
        redundancy=0.6,
        feedback_loops=(
            "AKT-mTOR rebound",
            "ERK pathway compensation",
        ),
        n_trials=10,
        n_p2_fail=2,
        n_approvals=1,
        trial_summary=(
            "alpelisib + fulvestrant approved (SOLAR-1, NCT02437318) "
            "for PIK3CA-mutant HR+ breast; toxicity (hyperglycemia) "
            "limits durability."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.ALTERNATIVE_PATHWAY,
        notes=(
            "NCT02437318 (SOLAR-1). per literature: Andre et al. (NEJM "
            "2019) on SOLAR-1; ERK rebound and PIK3CB paralog driving "
            "resistance described by Costa et al. (Cancer Cell 2015)."
        ),
    ),
    # ── WEE1 ────────────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-019",
        gene="WEE1",
        family="checkpoint_kinase",
        disease="TP53_mutant_solid_tumors",
        paralogs=("PKMYT1",),
        paralog_count=1,
        redundancy=0.35,
        feedback_loops=(
            "ATR-CHK1 backup checkpoint",
            "CDK2 hyperactivation",
        ),
        n_trials=14,
        n_p2_fail=8,
        n_approvals=0,
        trial_summary=(
            "adavosertib (AZD1775) Phase 2 in p53-mutant ovarian and "
            "uterine serous (NCT03330847) showed activity signals but "
            "toxicity-limited PFS; development paused 2021."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.ALTERNATIVE_PATHWAY,
        notes=(
            "NCT03330847 (adavosertib uterine serous Phase 2). per "
            "literature: Liu et al. (J Clin Oncol 2021) reported "
            "promising ORR but durability limited by ATR-CHK1 backup "
            "and PKMYT1 redundancy."
        ),
    ),
    # ── AXL ────────────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-020",
        gene="AXL",
        family="RTK",
        disease="triple_negative_breast_cancer",
        paralogs=("MERTK", "TYRO3"),
        paralog_count=2,
        redundancy=0.55,
        feedback_loops=(
            "TAM family compensation",
            "EMT-related plasticity",
        ),
        n_trials=6,
        n_p2_fail=5,
        n_approvals=0,
        trial_summary=(
            "bemcentinib (BGB324) Phase 2 in TNBC + paclitaxel "
            "(NCT03184558) showed limited signal; AXL-i monotherapy "
            "broadly disappointing in solid tumors."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.PARALOG_COMPENSATION,
        notes=(
            "NCT03184558 (BGB324 TNBC Phase 2). per literature: Gjerdrum "
            "et al. (PNAS 2010) on AXL/TAM redundancy in breast cancer; "
            "MERTK/TYRO3 paralog compensation is the canonical "
            "bemcentinib-resistance story."
        ),
    ),
    # ── MDM2 ───────────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-021",
        gene="MDM2",
        family="E3_ligase",
        disease="TP53_wildtype_liposarcoma",
        paralogs=("MDM4",),
        paralog_count=1,
        redundancy=0.45,
        feedback_loops=(
            "MDM2 transcriptional auto-induction by p53",
            "MDM4 compensatory binding",
        ),
        n_trials=8,
        n_p2_fail=3,
        n_approvals=0,
        trial_summary=(
            "milademetan (DS-3032) and idasanutlin Phase 2 (NCT03362723, "
            "NCT02545283) showed partial responses in MDM2-amplified "
            "liposarcoma but thrombocytopenia and MDM4 bypass limit "
            "durability."
        ),
        outcome=GroundTruthOutcome.PHASE2_SAFETY_FAILURE,
        bypass=BypassMechanism.PARALOG_COMPENSATION,
        notes=(
            "NCT03362723 (milademetan liposarcoma). per literature: "
            "Konopleva et al. (Leukemia 2020) and Gounder et al. on "
            "MDM2-i hematologic toxicity; MDM4 escape documented by "
            "Pant et al. (Genes Dev 2013)."
        ),
    ),
    # ── pan-RAF / RAF-dimer inhibitor ───────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-022",
        gene="RAF1",
        family="RAF_kinase",
        disease="RAS_mutant_solid_tumors_pan_RAF",
        paralogs=("BRAF", "ARAF"),
        paralog_count=2,
        redundancy=0.55,
        feedback_loops=(
            "RAF dimer paradox in WT-RAF cells",
            "MEK-ERK feedback",
        ),
        n_trials=7,
        n_p2_fail=5,
        n_approvals=0,
        trial_summary=(
            "belvarafenib (HM95573, NCT03284502) and naporafenib + "
            "trametinib in NRAS-mutant melanoma showed partial responses "
            "but no Phase 3 success; paradoxical activation in WT-RAF "
            "constrains therapeutic window."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "NCT03284502 (belvarafenib). per literature: Yen et al. "
            "(Nature 2021) on belvarafenib's RAF-dimer mechanism; "
            "Karoulia et al. (Nat Rev Cancer 2017) on the paradox "
            "constraining pan-RAF therapeutic window."
        ),
    ),
)
