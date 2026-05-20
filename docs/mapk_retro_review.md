# MAPK retro v0.2 — biology review

Curated by `mapk-curator` for Phase 2 of medreason target-validation.
This file is the human-review surface for the 22-entry retrospective
fixture in `medreason_bench/targetval/mapk_retro_data.py`. One
paragraph per case: the biology, the ground-truth outcome, and a
`per literature:` citation pattern (NCT IDs are real; primary-literature
references are described in prose rather than via DOI so a biologist
can spot-check the claim without trusting a fabricated link).

All cases here are Universal-safe: `InternalEvidence` is empty on every
entry. No customer data, no per-tenant readouts. This is the surface the
swarm + cross-agent analyzer is graded against.

---

## MAPK-001 — BRAF / metastatic melanoma BRAF V600E (APPROVED_LATER, downstream feedback)

V600E-mutant BRAF in melanoma is the archetypal MAPK precision-oncology
win. Vemurafenib (NCT00949702, BRIM-3) and dabrafenib produced rapid
deep responses as monotherapy, but resistance emerged within ~6 months
via RAS-GTP–driven CRAF/ARAF dimerisation and downstream MEK-ERK rebound.
The fix was combination MEK inhibition: dabrafenib + trametinib
(NCT01584648, COMBI-d) extended PFS by addressing the rebound mechanism
the swarm should now be able to predict from the paralog count of 2 plus
the documented feedback loop. **per literature**: Nazarian et al.
described RAS-pathway reactivation in BRIM-3 resistance specimens
(Nature 2010); the COMBI-d combo benefit was reported in Long et al.
(Lancet 2015).

## MAPK-002 — BRAF / metastatic colorectal BRAF V600E (PHASE2_EFFICACY_NO, downstream feedback)

The textbook "same target, different disease, different outcome" case.
BRAF-V600E monotherapy that works in melanoma fails in CRC because
colon epithelia retain robust EGFR signalling — inhibiting BRAF relieves
ERK-mediated negative feedback on EGFR, which then re-activates the MAPK
pathway from above. The fix is encorafenib + cetuximab (BEACON,
NCT02928224), which clamps both ends. This case forces the cross-agent
analyzer to learn that tissue context matters even when the driver
mutation is identical. **per literature**: Prahallad et al. (Nature
2012) characterised the EGFR feedback bypass; Kopetz et al. (NEJM 2015)
reported the negative monotherapy CRC result; Kopetz et al. (NEJM 2019)
reported BEACON.

## MAPK-003 — BRAF / papillary and anaplastic thyroid BRAF V600E (APPROVED_LATER, downstream feedback)

A third BRAF V600E disease, with yet another feedback partner. In
thyroid, HER3 (ERBB3) reactivation is the documented bypass. Vemurafenib
monotherapy showed only partial responses in PTC, but dabrafenib +
trametinib was approved for anaplastic thyroid via the ROAR basket
trial. The swarm should flag bypass risk even though approval was
eventually achieved — the combo regimen is the actionable signal.
**per literature**: Montero-Conde et al. (Cancer Discovery 2013) on HER3
reactivation in BRAF-mutant thyroid; ROAR is NCT02034110.

## MAPK-004 — MAP2K1 (MEK1) / NF1 plexiform neurofibroma (APPROVED_LATER, no bypass known)

The MEK inhibitor that escaped the MEK curse. NF1 loss removes the
GTPase-activating brake on RAS, leading to constitutive MAPK activation.
In pediatric plexiform neurofibroma (a benign-but-debilitating tumor),
selumetinib produced sustained tumor shrinkage in the SPRINT trial
(NCT01362803). Bypass risk is low here because selumetinib hits both
MEK1 and MEK2, the only paralog pair, and because there is no
"downstream of MEK" oncogenic node that could compensate without ERK.
This is the case the swarm should NOT flag as bypass-risky. **per
literature**: Dombi et al. (NEJM 2016) reported SPRINT.

## MAPK-005 — MAP2K1 / KRAS-mutant NSCLC (PHASE2_EFFICACY_NO, downstream feedback)

Same target, opposite outcome. In KRAS-mutant NSCLC, MEK inhibitors
relieve ERK-mediated negative feedback on RTKs (FGFR, IGF1R), which
re-feed the pathway. SELECT-1 (selumetinib + docetaxel, NCT01933932)
failed Phase 3, and trametinib monotherapy in KRAS-mutant lung never
showed durable signal. The swarm's bypass-risk score should differ
sharply from MAPK-004 even though the molecule is the same family. **per
literature**: Jänne et al. (JAMA 2017) reported SELECT-1; the RTK
feedback mechanism is reviewed in Sun et al. (Cell Reports 2014).

## MAPK-006 — MAPK1 (ERK2) / MAPK-pathway-altered solid tumors (PHASE2_EFFICACY_NO, downstream feedback)

ERK is the most-downstream MAPK kinase, so inhibiting it should — in
principle — close off most feedback routes. In practice, ERK inhibitors
(ulixertinib/BVD-523, LY3214996) trigger DUSP6 phosphatase loss and
upstream RAF/MEK reactivation, narrowing the therapeutic window. Phase 2
ORR has been modest and durability poor. **per literature**: Sullivan
et al. (Cancer Discovery 2018) documented ERK-i rebound via DUSP loss
and MEK reactivation; ulixertinib FIH is NCT01781429.

## MAPK-007 — KRAS G12C / metastatic NSCLC (APPROVED_LATER, resistance mutation)

The first KRAS direct inhibitor wins. Sotorasib (CodeBreaK 100,
NCT03600883) and adagrasib (KRYSTAL-1, NCT03785249) covalently engage
the G12C cysteine in the switch-II pocket. ORR in NSCLC is durable
enough for approval, but acquired resistance is well-mapped: secondary
mutations at Y96D and the switch-II region, plus RTK upregulation
(EGFR, AXL, MET) that reloads wild-type RAS. The swarm should learn
that resistance mutations are the dominant bypass mode here, distinct
from the feedback story in CRC (MAPK-008). **per literature**: Awad
et al. (NEJM 2021) catalogued adagrasib resistance; Skoulidis et al.
(NEJM 2021) reported CodeBreaK 100.

## MAPK-008 — KRAS G12C / metastatic colorectal (PHASE2_EFFICACY_NO, downstream feedback)

KRAS G12C inhibitors collapsed in CRC monotherapy: ORR ~9% in the
CodeBreaK 100 CRC cohort, vs ~37% in NSCLC. Why? EGFR feedback —
same tissue-specific bypass as BRAF in CRC (MAPK-002). The fix is the
same: KRAS-i + cetuximab. Adagrasib + cetuximab in the KRYSTAL-1
expansion cohort improved ORR to ~46%, supporting the combo strategy
but not enabling monotherapy approval. The swarm should learn the
"colon epithelia + EGFR feedback" pattern is generalisable. **per
literature**: Fakih et al. (Lancet Oncology 2022) on CodeBreaK 100 CRC;
Yaeger et al. (NEJM 2023) on adagrasib+cetuximab in CRC.

## MAPK-009 — NRAS / metastatic melanoma (PHASE2_EFFICACY_NO, alternative pathway)

NRAS-mutant melanoma is the "MAPK case that has nothing approved." The
NEMO Phase 3 of binimetinib (NCT01763164) showed modest PFS benefit
but no OS — the FDA declined to approve. The dominant bypass mechanism
isn't on-target resistance but parallel-pathway escape: CDK4/cyclin D1
cell-cycle activation and PI3K-AKT rescue. Combo trials with CDK4/6
inhibitors (e.g., binimetinib + ribociclib, NCT01781572) showed signal
but no approval. **per literature**: Dummer et al. (Lancet Oncology
2017) reported NEMO; the CDK4 + PI3K rescue rationale is reviewed in
Posch et al. (PNAS 2013).

## MAPK-010 — HRAS / recurrent HRAS-mutant HNSCC (PHASE2_EFFICACY_NO, paralog compensation)

HRAS is the only RAS paralog amenable to farnesyltransferase inhibition
because it lacks the alternative geranylgeranyl-transferase escape
route that KRAS and NRAS use. Tipifarnib (NCT02383927) showed promising
ORR (~55%) in HRAS-mutant HNSCC but durability was limited — the
hypothesis is that KRAS and NRAS paralogs, present and signalling at
baseline, fill in over time. This is the canonical paralog-compensation
case the swarm should learn from. **per literature**: Ho et al. (J Clin
Oncol 2021) reported the tipifarnib HNSCC Phase 2.

## MAPK-011 — EGFR / metastatic NSCLC EGFR exon 19 del / L858R (APPROVED_LATER, resistance mutation)

The MAPK upstream-RTK exemplar. Four generations of EGFR TKIs have
been approved (gefitinib, erlotinib, afatinib, osimertinib);
osimertinib (FLAURA, NCT02296125) is the current first-line standard.
Resistance is dominated by on-target gatekeeper mutations: T790M
emerged on first-/second-gen, C797S emerges on osimertinib. MET
amplification (the same biology that drives MAPK-013) is the secondary
bypass route. **per literature**: Soria et al. (NEJM 2018) on FLAURA;
Thress et al. (Nature Medicine 2015) catalogued C797S; Engelman et al.
(Science 2007) on MET amplification as the prototype RTK-bypass.

## MAPK-012 — MET / MET exon 14 skipping NSCLC (APPROVED_LATER, resistance mutation)

MET exon 14 skipping creates a constitutively active receptor by losing
the Cbl ubiquitin-ligase binding site. Capmatinib (GEOMETRY mono-1,
NCT02414139) and tepotinib (VISION, NCT02864992) are both approved.
Acquired resistance maps to second-site kinase-domain mutations (D1228,
Y1230), and EGFR cross-activation can rescue some cells. The swarm
should learn that exon-14 MET is its own driver class, distinct from
MET amplification (MAPK-013). **per literature**: Wolf et al. (NEJM 2020)
reported GEOMETRY mono-1; Recondo et al. (Clin Cancer Res 2020)
catalogued resistance.

## MAPK-013 — MET / EGFR-TKI-resistant NSCLC, MET amplified (PHASE2_EFFICACY_NO, alternative pathway)

MET amplification is the textbook RTK-bypass mechanism — it can rescue
the MAPK pathway after EGFR inhibition by dimerising with HER3 and
re-feeding ERK. The biology is real (Engelman et al., Science 2007),
but most clinical attempts to drug it have stumbled. MARQUEE
(tivantinib + erlotinib, NCT01244191) failed Phase 3, partly because
tivantinib's actual mechanism turned out to be tubulin binding rather
than MET inhibition — a "trial design didn't test the biology" failure
rather than a biology failure. SAVANNAH (savolitinib + osimertinib,
NCT03778229) reached only conditional signal in MET-high biomarker-
selected subgroups. **per literature**: Engelman et al. (Science 2007);
Scagliotti et al. (J Clin Oncol 2015) on MARQUEE; Hartmaier et al.
(Cancer Discovery 2023) on SAVANNAH.

## MAPK-014 — ALK / ALK-rearranged NSCLC (APPROVED_LATER, resistance mutation)

The most successful "kinase-translocation driver" story to date. Five
ALK inhibitors are approved (crizotinib, ceritinib, alectinib,
brigatinib, lorlatinib), each addressing the resistance landscape of
the previous. Alectinib (ALEX, NCT02075840) replaced crizotinib as
first-line; lorlatinib covers the difficult G1202R compound-resistance
mutation. Bypass via EGFR/KIT activation is documented but secondary to
on-target resistance. The swarm should learn that ALK is a low-paralog
(only LTK is a true paralog), high-druggability target — different
risk profile from KRAS or BRAF. **per literature**: Peters et al. (NEJM
2017) on ALEX; Gainor et al. (Cancer Discovery 2016) catalogued ALK-i
resistance mutations (G1202R, L1196M, etc.).

## MAPK-015 — PTPN11 (SHP2) / KRAS-G12C combo, NSCLC (ACTIVE_UNKNOWN, downstream feedback)

SHP2 inhibitors are the canonical "block RAS-GTP reload" play.
Mechanism: SHP2 dephosphorylates Ras-GAP-recruiting scaffold proteins
downstream of RTKs; inhibiting SHP2 starves RAS of GTP loading. The
rationale for combining SHP2-i with KRAS-G12C-i is that the G12C
inhibitor only binds RAS in the inactive GDP state, so depleting GTP
loading should keep RAS in the drug-accessible state. TNO155
(NCT04000529), RMC-4630 (NCT03634982), and JAB-3068 are all in
Phase 1/2 combos. Outcomes haven't fully matured — this case is held
at `ACTIVE_UNKNOWN` deliberately, marked as **[best-effort outcome]**
in `notes` because the biology is well-characterised but trial outcomes
are still in flight. The ground-truth bypass mechanism is the
documented rationale (downstream feedback via RAS-GTP reload). **per
literature**: Nichols et al. (Nature Cell Biology 2018) and Fedele
et al. (Cancer Discovery 2018) on SHP2-i blocking RTK-driven RAS-GTP
reload.

## MAPK-016 — FGFR2 / FGFR2-fusion cholangiocarcinoma (APPROVED_LATER, resistance mutation)

Cholangiocarcinoma was one of the first FGFR-targeted approvals.
Pemigatinib (FIGHT-202, NCT02924376) and futibatinib (FOENIX-CCA2,
NCT02052778) are both approved for FGFR2-fusion-positive disease.
Acquired resistance maps to gatekeeper mutations (V564F most commonly,
plus N549K and others in the kinase domain). Paralog count is high
(FGFR1/3/4 are all related) but the fusion event is FGFR2-specific.
**per literature**: Abou-Alfa et al. (Lancet Oncology 2020) reported
FIGHT-202; Goyal et al. (Cancer Discovery 2017) catalogued kinase-
domain resistance.

## MAPK-017 — CDK4 / HR+ HER2- breast cancer (APPROVED_LATER, alternative pathway)

Not strictly MAPK, but the case the swarm should reach for when an
"upstream" pathway hands off to cell-cycle control. Three CDK4/6
inhibitors are approved with endocrine partners (palbociclib + letrozole
PALOMA-2 NCT01740427; ribociclib MONALEESA; abemaciclib MONARCH).
Resistance mechanisms: RB1 loss (which removes the inhibitor's
downstream target entirely), CCNE1 (cyclin E1) amplification (which
bypasses CDK4 via CDK2 activation), and PI3K-AKT reactivation. The
paralog is just CDK6 (which is already co-inhibited), so the dominant
bypass mechanism is alternative pathway, not paralog. **per literature**:
Finn et al. (NEJM 2016) on PALOMA-2; Wander et al. (Cancer Discovery
2020) catalogued resistance routes.

## MAPK-018 — PIK3CA / HR+ PIK3CA-mutant breast cancer (APPROVED_LATER, alternative pathway)

Alpelisib + fulvestrant (SOLAR-1, NCT02437318) is approved for
PIK3CA-mutant HR+ breast. The biology is clean — PI3K-alpha drives
AKT-mTOR signalling, alpelisib selectively inhibits the p110-alpha
isoform — but durability is constrained by ERK-pathway rebound and
PIK3CB paralog upregulation. Toxicity (hyperglycemia) compounds the
durability problem. The swarm should learn that PI3K's three other
paralogs create a paralog-compensation route, but the dominant
resistance mode reported clinically is alternative-pathway ERK rebound.
**per literature**: André et al. (NEJM 2019) on SOLAR-1; Costa et al.
(Cancer Cell 2015) on ERK rebound and PIK3CB paralog escape.

## MAPK-019 — WEE1 / TP53-mutant solid tumors (PHASE2_EFFICACY_NO, alternative pathway)

WEE1 is the G2/M checkpoint kinase; inhibiting it forces cells with
already-compromised G1 checkpoints (TP53-mutant) into mitotic
catastrophe. Adavosertib (AZD1775) Phase 2 in p53-mutant uterine
serous and ovarian carcinoma (NCT03330847) showed promising ORR but
the development was paused in 2021 due to durability and toxicity.
The bypass biology is real: ATR-CHK1 acts as a backup checkpoint
arm, and PKMYT1 is a paralog kinase that phosphorylates the same
CDK1 substrate as WEE1. **per literature**: Liu et al. (J Clin Oncol
2021) reported the uterine serous Phase 2; PKMYT1 redundancy is
described in Gallo et al. (Nature 2022).

## MAPK-020 — AXL / triple-negative breast cancer (PHASE2_EFFICACY_NO, paralog compensation)

AXL is part of the TAM (Tyro3, AXL, MerTK) receptor family. AXL drives
EMT and chemoresistance in TNBC, which made it a popular target — but
bemcentinib (BGB324) Phase 2 in TNBC + paclitaxel (NCT03184558) and
related AXL monotherapy trials have all disappointed. The dominant
hypothesis is that MerTK and Tyro3 compensate. **per literature**:
Gjerdrum et al. (PNAS 2010) on AXL/TAM redundancy in breast cancer;
the bemcentinib Phase 2 result was reported in conference proceedings
(SABCS 2020) and remains the canonical AXL-monotherapy negative.

## MAPK-021 — MDM2 / TP53-wildtype liposarcoma (PHASE2_SAFETY_FAILURE, paralog compensation)

The first MDM2 inhibitor wave (nutlin descendants — milademetan
DS-3032, idasanutlin) reached Phase 2 in MDM2-amplified TP53-wildtype
liposarcoma (NCT03362723, NCT02545283). Efficacy signals were real but
thrombocytopenia consistently capped tolerable doses. Bypass biology:
MDM4 (the MDM2 paralog) binds and inhibits p53 redundantly, and p53
auto-induces MDM2 expression — both effects narrow the therapeutic
window. The outcome is recorded as `PHASE2_SAFETY_FAILURE` because the
limit was hematologic toxicity rather than tumor escape. **per
literature**: Gounder et al. and Konopleva et al. (Leukemia 2020) on
MDM2-i hematologic toxicity; MDM4 paralog escape characterised by
Pant et al. (Genes Dev 2013).

## MAPK-022 — RAF1 / RAS-mutant solid tumors, pan-RAF (PHASE2_EFFICACY_NO, downstream feedback)

The "pan-RAF dimer-breaker" story. Conventional BRAF-V600E inhibitors
(vemurafenib, dabrafenib) cause paradoxical activation in RAS-mutant
cells because they bind one RAF protomer and trans-activate the dimer
partner. Belvarafenib (HM95573, NCT03284502) and naporafenib are
designed to break the dimer entirely. Phase 1/2 in NRAS-mutant melanoma
and RAS-mutant solid tumors showed partial responses but no Phase 3
success; the therapeutic window is constrained because RAF dimers in
WT-RAF cells are also engaged. **per literature**: Yen et al. (Nature
2021) on belvarafenib's RAF-dimer mechanism; Karoulia et al. (Nat Rev
Cancer 2017) reviewed the paradox.

---

## Source-quality caveats

- All NCT IDs are real clinical-trial identifiers; the user can spot-
  check any of them at clinicaltrials.gov.
- Primary-literature references are described in prose so a biologist
  can verify the claim by searching author + journal + year. No DOIs
  are fabricated.
- One entry (MAPK-015, SHP2 + KRAS-G12C combo) carries a
  **[best-effort outcome]** marker in its `notes` field. The biology
  is well-characterised but trial outcomes are still maturing, so the
  outcome is held at `ACTIVE_UNKNOWN` and the bypass mechanism reflects
  the documented rationale rather than a closed-out clinical result.
- The 22-entry set covers all five non-`UNKNOWN` `BypassMechanism`
  values that the swarm cross-agent analyzer scores against
  (PARALOG_COMPENSATION, DOWNSTREAM_FEEDBACK, ALTERNATIVE_PATHWAY,
  RESISTANCE_MUTATION, NO_BYPASS_KNOWN) plus four `GroundTruthOutcome`
  values (APPROVED_LATER, PHASE2_EFFICACY_NO, PHASE2_SAFETY_FAILURE,
  ACTIVE_UNKNOWN). The mix is intentional: it gives the cross-agent
  analyzer something to discriminate on.

## Universal-safety attestation

Every case in this fixture is built from `MapkRetroEntry` rows in
`mapk_retro_data.py`. The `_entry_from` builder in `mapk_retro.py`
does NOT touch `EvidenceBundle.internal`, so every emitted case has
`evidence.has_internal_data() == False`. This is asserted in
`test_targetval_end_to_end.py::test_mapk_retro_v0_2_size`. Any rule
the swarm + cross-agent analyzer learns from this fixture is therefore
legally writeable to the Universal layer — the moat holds.
