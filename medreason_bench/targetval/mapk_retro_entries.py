"""MAPK retro v0.2 — core MAPK-axis entries (BRAF / MEK / ERK / RAS).

Ten entries covering the canonical MAPK signalling axis:

- BRAF in melanoma / CRC / thyroid (3 cases — same target, three diseases)
- MEK1 in NF1 vs KRAS-mut NSCLC (2 cases — tissue-specific tractability)
- ERK1/2 across MAPK-altered solid tumors (1 case)
- KRAS G12C in NSCLC vs CRC (2 cases)
- NRAS / HRAS in their canonical diseases (2 cases)

The MAPK-001..MAPK-010 id range belongs here. RTK + non-MAPK adjuncts
(MAPK-011..MAPK-022) live in ``mapk_retro_entries_extra.py``.

Every entry here is Universal-safe: no internal data, every claim
sourceable via a real NCT ID or a ``per literature:`` description.
"""

from __future__ import annotations

from medreason.targetval.case import BypassMechanism, GroundTruthOutcome

from .mapk_retro_data import MapkRetroEntry


MAPK_RETRO_ENTRIES_CORE: tuple[MapkRetroEntry, ...] = (
    # ── BRAF axis ───────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-001",
        gene="BRAF",
        family="RAF_kinase",
        disease="metastatic_melanoma_BRAF_V600E",
        paralogs=("ARAF", "RAF1"),
        paralog_count=2,
        redundancy=0.4,
        feedback_loops=(
            "MEK-ERK rebound after BRAF-i monotherapy",
            "RAS-GTP reactivation",
        ),
        n_trials=12,
        n_p2_fail=2,
        n_approvals=2,
        trial_summary=(
            "vemurafenib/dabrafenib approved monotherapy; MEK combos "
            "(dabrafenib+trametinib) extend PFS by addressing rebound."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "NCT00949702 (BRIM-3, vemurafenib) and NCT01584648 "
            "(COMBI-d, dabrafenib+trametinib). per literature: Nazarian "
            "et al. described RAS-pathway reactivation in resistant "
            "melanomas (Nature 2010); combo benefit reported in COMBI-d."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-002",
        gene="BRAF",
        family="RAF_kinase",
        disease="metastatic_colorectal_BRAF_V600E",
        paralogs=("ARAF", "RAF1"),
        paralog_count=2,
        redundancy=0.65,
        feedback_loops=(
            "EGFR feedback amplification post-BRAF-i",
            "PI3K-AKT rebound",
        ),
        n_trials=8,
        n_p2_fail=5,
        n_approvals=1,
        trial_summary=(
            "BRAF-i monotherapy failed in CRC (Kopetz NEJM 2015); "
            "encorafenib+cetuximab (BEACON, NCT02928224) eventually "
            "approved for BRAF V600E CRC."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "per literature: Prahallad et al. (Nature 2012) showed EGFR "
            "feedback bypass in CRC distinguishes it from melanoma; "
            "monotherapy CRC trials (e.g. NCT00405587) failed efficacy."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-003",
        gene="BRAF",
        family="RAF_kinase",
        disease="papillary_thyroid_cancer_BRAF_V600E",
        paralogs=("ARAF", "RAF1"),
        paralog_count=2,
        redundancy=0.5,
        feedback_loops=("HER3 reactivation post-BRAF-i",),
        n_trials=5,
        n_p2_fail=1,
        n_approvals=1,
        trial_summary=(
            "dabrafenib+trametinib approved for anaplastic thyroid BRAF "
            "V600E (ROAR basket); vemurafenib monotherapy showed partial "
            "responses in PTC but resistance via HER3 (Montero-Conde 2013)."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "NCT02034110 (ROAR basket). per literature: Montero-Conde "
            "et al. (Cancer Discovery 2013) on HER3 reactivation in "
            "BRAF-mutant thyroid."
        ),
    ),
    # ── MEK / ERK ───────────────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-004",
        gene="MAP2K1",
        family="MEK",
        disease="NF1_plexiform_neurofibroma_pediatric",
        paralogs=("MAP2K2",),
        paralog_count=1,
        redundancy=0.3,
        feedback_loops=("ERK negative feedback to RAF",),
        n_trials=6,
        n_p2_fail=1,
        n_approvals=1,
        trial_summary=(
            "selumetinib approved (NCT01362803, SPRINT trial) for "
            "inoperable NF1 plexiform neurofibromas; first MEK inhibitor "
            "approved outside oncology indications."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.NO_BYPASS_KNOWN,
        notes=(
            "NCT01362803 (SPRINT). per literature: Dombi et al. "
            "(NEJM 2016) reported sustained tumor shrinkage; bypass "
            "low because NF1 loss removes the only physiologic brake "
            "and the paralog (MAP2K2) is also inhibited by selumetinib."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-005",
        gene="MAP2K1",
        family="MEK",
        disease="metastatic_KRAS_mutant_NSCLC",
        paralogs=("MAP2K2",),
        paralog_count=1,
        redundancy=0.5,
        feedback_loops=(
            "ERK negative feedback to RAF",
            "RTK reactivation",
        ),
        n_trials=9,
        n_p2_fail=6,
        n_approvals=0,
        trial_summary=(
            "selumetinib + docetaxel failed Phase 3 SELECT-1 "
            "(NCT01933932); trametinib monotherapy Phase 2 in KRAS-mut "
            "NSCLC showed limited durable responses."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "NCT01933932 (SELECT-1). per literature: Janne et al. "
            "(JAMA 2017) reported SELECT-1 negative result; feedback "
            "via FGFR / IGF1R drives MEK-i resistance in KRAS-mut lung."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-006",
        gene="MAPK1",
        family="ERK",
        disease="MAPK_pathway_altered_solid_tumors",
        paralogs=("MAPK3",),
        paralog_count=1,
        redundancy=0.45,
        feedback_loops=(
            "ERK auto-feedback to RAF",
            "DUSP6 suppression",
        ),
        n_trials=7,
        n_p2_fail=5,
        n_approvals=0,
        trial_summary=(
            "ulixertinib (BVD-523) and LY3214996 Phase 1/2 showed "
            "limited single-agent activity; therapeutic window "
            "constrained by on-target rebound (Sullivan 2018)."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "NCT01781429 (ulixertinib FIH). per literature: Sullivan "
            "et al. (Cancer Discovery 2018) documented ERK-i rebound "
            "via DUSP loss and MEK reactivation."
        ),
    ),
    # ── KRAS / NRAS / HRAS ──────────────────────────────────────────────────
    MapkRetroEntry(
        case_id="MAPK-007",
        gene="KRAS",
        family="RAS_GTPase",
        disease="metastatic_NSCLC_KRAS_G12C",
        paralogs=("HRAS", "NRAS"),
        paralog_count=2,
        redundancy=0.45,
        feedback_loops=(
            "RTK reactivation (EGFR, AXL, MET)",
            "wild-type RAS amplification",
        ),
        n_trials=10,
        n_p2_fail=2,
        n_approvals=2,
        trial_summary=(
            "sotorasib approved (CodeBreaK 100, NCT03600883); adagrasib "
            "approved (KRYSTAL-1, NCT03785249). Y96D and RTK rewiring "
            "drive acquired resistance."
        ),
        outcome=GroundTruthOutcome.APPROVED_LATER,
        bypass=BypassMechanism.RESISTANCE_MUTATION,
        notes=(
            "NCT03600883 (CodeBreaK 100, sotorasib), NCT03785249 "
            "(KRYSTAL-1, adagrasib). per literature: Awad et al. (NEJM "
            "2021) catalogued Y96D + bypass RTK as adagrasib resistance."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-008",
        gene="KRAS",
        family="RAS_GTPase",
        disease="metastatic_colorectal_KRAS_G12C",
        paralogs=("HRAS", "NRAS"),
        paralog_count=2,
        redundancy=0.6,
        feedback_loops=(
            "EGFR feedback reactivation",
            "wild-type RAS amplification",
        ),
        n_trials=6,
        n_p2_fail=4,
        n_approvals=0,
        trial_summary=(
            "sotorasib monotherapy in CRC: ORR ~9% (CodeBreaK 100 CRC "
            "cohort); adagrasib+cetuximab (KRYSTAL-1 expansion) improved "
            "ORR but no monotherapy approval. EGFR feedback is the "
            "tissue-specific bypass."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.DOWNSTREAM_FEEDBACK,
        notes=(
            "per literature: Fakih et al. (Lancet Oncol 2022) on CodeBreaK "
            "100 CRC; Yaeger et al. (NEJM 2023) on adagrasib+cetuximab "
            "in CRC. Same target as MAPK-007, different disease, "
            "different bypass mechanism — the tissue-specificity case."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-009",
        gene="NRAS",
        family="RAS_GTPase",
        disease="metastatic_melanoma_NRAS_mutant",
        paralogs=("KRAS", "HRAS"),
        paralog_count=2,
        redundancy=0.55,
        feedback_loops=(
            "MEK-ERK negative feedback",
            "CDK4 cell-cycle escape",
        ),
        n_trials=8,
        n_p2_fail=6,
        n_approvals=0,
        trial_summary=(
            "binimetinib monotherapy (NEMO Phase 3, NCT01763164) showed "
            "modest PFS but no OS benefit; pimasertib+SAR405838 and "
            "binimetinib+ribociclib (NCT01781572) explored CDK4/6 combos "
            "without approval."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.ALTERNATIVE_PATHWAY,
        notes=(
            "NCT01763164 (NEMO). per literature: Dummer et al. (Lancet "
            "Oncol 2017) reported NEMO; failure attributed to CDK4 and "
            "PI3K rescue rather than on-target resistance."
        ),
    ),
    MapkRetroEntry(
        case_id="MAPK-010",
        gene="HRAS",
        family="RAS_GTPase",
        disease="recurrent_HRAS_mutant_HNSCC",
        paralogs=("KRAS", "NRAS"),
        paralog_count=2,
        redundancy=0.5,
        feedback_loops=("PI3K-AKT compensation",),
        n_trials=4,
        n_p2_fail=2,
        n_approvals=0,
        trial_summary=(
            "tipifarnib (farnesyltransferase inhibitor targeting HRAS) "
            "Phase 2 (NCT02383927) showed ORR ~55% in HRAS-mutant HNSCC "
            "but durable responses limited; no approval."
        ),
        outcome=GroundTruthOutcome.PHASE2_EFFICACY_NO,
        bypass=BypassMechanism.PARALOG_COMPENSATION,
        notes=(
            "NCT02383927 (tipifarnib HNSCC Phase 2). per literature: "
            "Ho et al. (J Clin Oncol 2021) reported promising ORR but "
            "PFS limited by KRAS/NRAS-driven escape; FTI specificity for "
            "HRAS leaves paralogs untouched."
        ),
    ),
)
