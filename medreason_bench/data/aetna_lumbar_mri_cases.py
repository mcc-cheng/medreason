"""v0.3 fixture — 30 within-domain Aetna lumbar MRI cases.

Source: hand-authored by the user (2026-04-11). All 30 cases are prior
authorization requests for lumbar MRI (CPT 72148) under Aetna
CPB-0236 and CPB-0093 (the Aetna commercial coverage policies for
lumbar imaging). Two cases (25, 30) deliberately involve plan-type
edge conditions that swap the governing policy to CMS LCD L33408
(Medicare Advantage) or to Aetna Medicaid managed care — these are
realistic operational traps that hit PA companies in production.

Outcome distribution:
    12 approved (cases 1, 4, 6, 8, 11, 14, 16, 19, 21, 22, 23, 24)
    18 denied   (cases 2, 3, 5, 7, 9, 10, 12, 13, 15, 17, 18, 20,
                 25, 26, 27, 28, 29, 30)

PEND mapping: the user's original labels used "PEND" for 9 cases
(3, 7, 12, 15, 17, 18, 28, 29, 30). All 9 are mapped to `denied`
because under payer rules "PEND" means the PA as submitted fails at
least one criterion — that's operationally a denial. The agent's
task is to decide the PA verdict, not predict appeal outcomes.

Cases 13 and 14 are a deliberate documentation-completeness pair on
the same patient (claustrophobia / open MRI). Case 13 fails for
missing closed-bore-attempt documentation (DENY). Case 14 approves
with the documentation present. This pair tests whether the system
detects the doc-delta that flips the outcome.

This is the within-domain generalization fixture referenced in
SESSION_BRIEF.md §9. It's the first fixture in the project where all
cases share the same payer + condition family, which lets us run a
legitimate "do rules from case A help case B within the same domain"
test.
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


# ── Shared policy excerpts ──────────────────────────────────────────────────
#
# Every case references a real Aetna policy. CPB-0236 governs lumbar MRI
# medical necessity; CPB-0093 governs MRI modality choices (open vs
# closed-bore, positional, weight-bearing). These strings approximate the
# operational content reviewers actually apply — they include the
# independent-criterion list, the conservative-therapy prerequisite, the
# frequency cap, and the experimental-modality clauses that come up in
# this fixture.

AETNA_CPB_0236 = """Aetna Clinical Policy Bulletin CPB-0236 — Lumbar Spine Imaging.
§A.1 STANDARD COVERAGE: Lumbar MRI is covered after documented failure of >=6 weeks conservative therapy (physical therapy for the lumbar spine AND NSAIDs or equivalent) in a patient with lumbar radiculopathy evidenced by both pain AND objective motor or reflex changes on exam.
§B.1 INDEPENDENT CRITERIA (no conservative therapy wait required):
  (a) Suspected cauda equina syndrome (saddle anesthesia, bowel/bladder dysfunction, bilateral motor weakness)
  (b) Rapidly progressing neurological deficit documented on serial exams
  (c) Suspected vertebral, paraspinal, or intraspinal metastatic disease with supporting evidence (known primary, bone scan, labs)
  (d) Suspected vertebral fracture / differentiation of benign vs pathological fracture in osteoporotic patient with inconclusive plain films
  (e) Suspected lumbar epidural lipomatosis in patient on chronic systemic steroids
  (f) Suspected lumbar arachnoiditis (prior intrathecal procedures + appropriate clinical picture)
  (g) Suspected myelopathy with lumbar segment clinically implicated
  (h) Failed back surgery syndrome with new objective neurological deficit post-operatively
  (i) Congenital spinal anomaly / scoliosis pre-operative planning (including rule-out syrinx)
§C.1 FREQUENCY LIMIT: Repeat lumbar MRI within 12 months of a prior study is NOT covered unless one of: new neurological deficit since prior imaging, significant clinical change, post-operative evaluation, or interval radiation therapy. Routine surveillance for chronic pain is excluded.
§D.1 CONSERVATIVE THERAPY DEFINITION: Physical therapy must target the lumbar spine specifically. PT records for a different anatomic region (e.g., shoulder rehab) do NOT satisfy the conservative therapy prerequisite. Out-of-network PT requires documentation provided by the OON facility; patient self-report is insufficient.
§E.1 SUBJECTIVE RADICULOPATHY: Radiculopathy must be evidenced by both pain AND objective motor, sensory, or reflex changes on exam. Pain alone without objective findings does not satisfy the radiculopathy criterion.
§F.1 AHCPR GUIDANCE: Routine imaging for acute low back pain without red-flag symptoms is not covered (AHCPR recommends against imaging in acute LBP without red flags).
§G.1 BONE-MRI EXPERIMENTAL CLAUSE: BoneMRI (MRI-based synthetic CT) is classified as experimental/investigational for spinal pre-operative assessment and surgical planning. Not covered regardless of clinical rationale.
§H.1 APPEAL PRECEDENT: Conservative therapy shortfalls (e.g., 5 weeks when 6 are required) may be overridden on appeal if the physician documents progressing motor weakness in parallel, invoking the rapidly-progressing-deficit clause."""

AETNA_CPB_0093 = """Aetna Clinical Policy Bulletin CPB-0093 — Alternative MRI Modalities.
§P.1 OPEN / LOW-FIELD MRI: Covered only when a documented contraindication to closed-bore MRI exists. "Patient prefers" or "patient requests" is NOT a contraindication. Acceptable documentation includes: failed closed-bore attempt with sedation support, psychiatry confirmation of severe claustrophobia, or body habitus exceeding closed-bore weight/diameter limits.
§P.2 POSITIONAL / WEIGHT-BEARING MRI: Classified as experimental/investigational for Ehlers-Danlos Syndrome (EDS) and hypermobility disorders. Hard designation — no exception pathway. Covered only for specific upright imaging indications enumerated in CPB-0093 Section Q (none of which apply to routine lumbar radiculopathy)."""

CMS_LCD_L33408 = """CMS LCD L33408 — Magnetic Resonance Imaging (MRI) (applicable to Medicare Advantage plans).
This Local Coverage Determination governs lumbar MRI coverage for Medicare and Medicare Advantage beneficiaries and supersedes Aetna commercial CPB-0236 for those plan types.
§M.1 PLAN-TYPE REQUIREMENT: Prior authorization requests submitted under Aetna commercial policy CPB-0236 for a Medicare Advantage beneficiary will be denied as a plan-type error regardless of clinical merit. The request must be resubmitted under the CMS LCD L33408 pathway with the additional documentation requirements enumerated therein (specifically §M.2 physician attestation of failed conservative therapy and §M.3 objective exam findings documented within the past 90 days).
§M.2 Conservative therapy attestation: physician must sign an attestation that conservative therapy was trialed and failed; PT discharge summary alone is insufficient for Medicare Advantage.
§M.3 Objective findings currency: exam findings must be documented within the past 90 days of the PA request date, not historical."""

AETNA_MEDICAID_MANAGED = """Aetna Medicaid Managed Care — Lumbar MRI Prior Authorization Pathway.
§X.1 PLAN-TYPE REQUIREMENT: Members enrolled in Aetna Medicaid managed care plans use a separate PA pathway distinct from Aetna commercial CPB-0236. Requests submitted under CPB-0236 for Medicaid members are invalid regardless of clinical merit and must be restarted under the Medicaid pathway.
§X.2 Eligibility verification: PA verdicts are bound to the member's plan type on the date of service determination, not the plan type on the date of PA submission. Mid-month plan reclassification requires PA reset.
§X.3 Medicaid clinical criteria for lumbar MRI are more restrictive than commercial and include additional step-therapy requirements (manipulation therapy trial, targeted injection trial) before advanced imaging."""


# ── Case builder ────────────────────────────────────────────────────────────


def _case(
    *,
    case_id: str,
    cpt: str = "72148",
    icds: list[str],
    notes: str,
    policy: str,
    outcome: Outcome,
    reasoning: list[str],
    difficulty: Difficulty,
    denial_reason: DenialReason | None = None,
    facility: FacilityType = FacilityType.OUTPATIENT,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA,
            cpt_code=cpt,
            icd10_codes=icds,
            facility_type=facility,
            denial_reason=denial_reason,
        ),
        clinical_notes=notes,
        policy_excerpt=policy,
        ground_truth_outcome=outcome,
        ground_truth_reasoning=reasoning,
        difficulty=difficulty,
    )


def build_aetna_lumbar_mri_cases() -> list[BenchmarkCase]:
    """Return the full v0.3 Aetna lumbar MRI fixture (30 cases)."""
    cases: list[BenchmarkCase] = []

    # ═══════════════════════════════════════════════════════════════════
    # CLEAN APPROVALS (12)
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="aetna_001",
        icds=["M51.16"],
        notes=(
            "58 y/o F with right L4-L5 radiculopathy and foot drop. "
            "Completed 7 weeks of supervised PT targeting the lumbar "
            "spine plus trial of NSAIDs without relief. Exam: "
            "dermatomal L5 pain distribution, dorsiflexion weakness "
            "4/5, diminished right patellar reflex. PT discharge "
            "summary and NSAID Rx history in chart."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.1 standard coverage path.",
            "Conservative therapy >= 6 weeks documented.",
            "Radiculopathy evidenced by both pain (dermatomal) and "
            "objective findings (motor + reflex).",
            "All criteria satisfied -> APPROVE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_004",
        icds=["G95.89", "M54.5"],
        notes=(
            "71 y/o F with bilateral lower extremity weakness "
            "progressing to gait difficulty over 10 days. Serial ED "
            "neuro exams show strength decline: right 4+/5 to 3/5 and "
            "left 4/5 to 3+/5 over 4 days. No prior conservative "
            "therapy attempted."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(b) rapidly progressing neurological deficit.",
            "Serial documentation of strength decline on exam.",
            "Independent criterion bypasses §A.1 conservative therapy "
            "wait.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_006",
        icds=["G83.4", "M54.5"],
        notes=(
            "55 y/o M presenting to ED with acute saddle anesthesia, "
            "urinary retention, and bilateral lower extremity motor "
            "weakness. ED note documents all three cardinal signs of "
            "cauda equina syndrome. Requesting emergent lumbar MRI."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(a) suspected cauda equina syndrome.",
            "All three cardinal signs documented.",
            "Independent standalone criterion; no conservative "
            "therapy required. Emergent.",
            "APPROVE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_008",
        icds=["C79.51", "C50.912"],
        notes=(
            "68 y/o F with Stage IV breast cancer, new mid-back pain "
            "x 3 weeks. Bone scan shows focal uptake at L2. Oncology "
            "note documents ongoing metastatic workup and raises "
            "concern for vertebral metastasis."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(c) suspected vertebral metastatic disease.",
            "Known primary (breast), supporting evidence (bone scan "
            "uptake), oncology concern.",
            "Independent criterion; no wait period.",
            "APPROVE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_011",
        icds=["M96.1", "M51.17"],
        notes=(
            "61 y/o M, 8 months post L4-L5 discectomy, now presenting "
            "with new left S1 dermatomal numbness, diminished left "
            "Achilles reflex, and left calf weakness 4/5. No "
            "radiculopathy or deficit in this distribution "
            "pre-operatively. Physician documents new onset."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(h) failed back surgery syndrome with new objective "
            "neurological deficit post-operatively.",
            "New S1 findings not present pre-op.",
            "Independent criterion.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_014",
        icds=["M51.16", "F40.240"],
        notes=(
            "74 y/o F with right L5 radiculopathy meeting §A.1 "
            "criteria (8 weeks PT + NSAIDs). History of severe "
            "claustrophobia. Patient attempted closed-bore MRI with "
            "oral lorazepam sedation last week; attempt failed with "
            "patient unable to tolerate. Psychiatry note confirms "
            "severe claustrophobia diagnosis. Physician requesting "
            "open / low-field MRI."
        ),
        policy=AETNA_CPB_0236 + "\n\n" + AETNA_CPB_0093,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.1 clinical criteria met.",
            "§P.1 open MRI: documented failed closed-bore attempt "
            "with sedation + psychiatry confirmation.",
            "Contraindication standard satisfied.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_016",
        icds=["Q67.5", "M41.0"],
        notes=(
            "17 y/o F with congenital scoliosis. Cobb angle 52 degrees "
            "on recent standing X-ray, up from 44 degrees one year "
            "prior. Surgical team planning instrumented fusion and "
            "requesting pre-operative MRI to rule out syrinx or other "
            "intraspinal anomaly."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(i) congenital spinal anomaly with pre-operative "
            "planning (rule out syrinx).",
            "Standalone criterion; no conservative therapy "
            "prerequisite.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_019",
        icds=["E66.1", "K50.90"],  # hand-mapped; Crohn's + steroid-related
        notes=(
            "61 y/o M with Crohn's disease on chronic prednisone "
            "(5-10 mg/day for 4+ years). New bilateral lower extremity "
            "weakness x 6 weeks. Neurology note explicitly raises "
            "clinical suspicion of lumbar epidural lipomatosis given "
            "the steroid exposure history and presentation."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(e) suspected lumbar epidural lipomatosis with "
            "chronic systemic steroid use.",
            "Explicitly enumerated independent criterion.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_021",
        icds=["G95.9", "M54.5"],
        notes=(
            "67 y/o M with progressive lower extremity spasticity, "
            "hyperreflexia, and bilateral Babinski signs. Neurology "
            "suspects cord compression at either cervical or lumbar "
            "level and requests full spine MRI including lumbar as "
            "part of the myelopathy workup."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(g) suspected myelopathy with lumbar segment "
            "clinically implicated.",
            "Upper motor neuron signs (Babinski, hyperreflexia, "
            "spasticity).",
            "Lumbar segment approvable as part of myelopathy workup.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_022",
        icds=["M80.08XA", "M48.50XA"],
        notes=(
            "78 y/o F with severe osteoporosis (DEXA T-score -3.4). "
            "Sudden onset severe mid-lumbar pain after minor "
            "positional change at home. X-ray shows possible L2 "
            "compression deformity but appearance is inconclusive for "
            "osteoporotic vs pathological etiology. No known "
            "malignancy but differential includes occult lesion."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(d) suspected vertebral fracture with inconclusive "
            "plain films, osteoporotic patient.",
            "Differentiation of benign vs pathological is a "
            "standalone criterion.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_023",
        icds=["M96.1", "M51.17"],
        notes=(
            "59 y/o M with 3 prior lumbar surgeries (discectomy, "
            "revision discectomy, posterior fusion L4-S1). Now "
            "presenting 14 months after most recent surgery with "
            "NEW right L3 dermatomal numbness and weakness (3/5 hip "
            "flexion) not present on the prior post-op exam. "
            "Differential includes recurrent disc, fibrosis, hardware "
            "failure, adjacent-segment disease."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(h) failed back surgery syndrome with new objective "
            "neurological deficit post-operatively.",
            "New L3 findings distinct from prior presentation.",
            "Multiple surgical history STRENGTHENS medical necessity "
            "given active differentials.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_024",
        icds=["G03.9", "R32"],  # arachnoiditis + bladder dysfunction
        notes=(
            "52 y/o F with prior history of multiple intrathecal "
            "steroid injections for chronic lumbar pain. Now "
            "presenting with bilateral burning leg pain and new "
            "bladder dysfunction. Clinician suspects lumbar "
            "arachnoiditis."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(f) suspected lumbar arachnoiditis.",
            "Prior intrathecal procedures + bilateral burning pain + "
            "bladder dysfunction constitutes complete clinical "
            "picture.",
            "Explicitly enumerated independent criterion.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    # ═══════════════════════════════════════════════════════════════════
    # CLEAN DENIALS (6 clean + 6 pend mapped to denied from first 20 +
    # 3 clean + 3 pend from last 10 = 18 total)
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="aetna_002",
        icds=["M54.5"],
        notes=(
            "44 y/o M with acute low back pain x 3 days after lifting "
            "a heavy box at home. No radiculopathy, no motor or "
            "sensory deficits, no red-flag symptoms. Zero prior "
            "conservative treatment. Requesting lumbar MRI for "
            "reassurance."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§F.1 AHCPR guidance: no imaging for acute LBP without "
            "red flags.",
            "No §A.1 conservative therapy attempted.",
            "No §B.1 independent criterion met.",
            "DENY.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_003",
        icds=["M51.16"],
        notes=(
            "62 y/o M with LBP and bilateral L5 radiculopathy "
            "(dermatomal pain, reduced patellar reflex bilaterally). "
            "Currently at week 5 of ongoing PT program. No documented "
            "progressive motor weakness. Physician requesting MRI "
            "now, one week short of 6-week threshold."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§A.1 conservative therapy requirement: >= 6 weeks "
            "needed, only 5 documented.",
            "§H.1 appeal clause available IF physician documents "
            "progressing motor weakness — NOT documented here.",
            "As submitted, criterion unmet.",
            "DENY (appeal path exists if deficit documented).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_005",
        icds=["M54.5"],
        notes=(
            "49 y/o F with 8 weeks of LBP and patient-reported "
            "radiating pain down the right leg. Neurological exam "
            "completely normal: intact reflexes bilaterally, full "
            "strength (5/5), no sensory deficits. PT trial and "
            "NSAIDs both documented."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§E.1 subjective radiculopathy clause: pain alone "
            "without objective motor, sensory, or reflex findings "
            "does not satisfy radiculopathy.",
            "Conjunctive criterion (pain AND objective findings) "
            "not met.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_007",
        icds=["M54.5", "M51.16"],
        notes=(
            "67 y/o M reports 8 weeks of PT at an out-of-network "
            "provider. No PT records available; patient self-report "
            "only. Has documented L5 radiculopathy on exam. "
            "Physician requesting lumbar MRI."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MISSING_INFO,
        reasoning=[
            "§D.1 OON conservative therapy requires documentation "
            "from the OON facility; patient self-report insufficient.",
            "Conservative therapy prerequisite unverified.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_009",
        icds=["Q79.60", "M54.5"],  # EDS + LBP
        notes=(
            "38 y/o F with Ehlers-Danlos Syndrome and chronic LBP "
            "aggravated in upright positions. Provider requesting "
            "upright weight-bearing MRI to evaluate positional "
            "mechanics."
        ),
        policy=AETNA_CPB_0236 + "\n\n" + AETNA_CPB_0093,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.EXPERIMENTAL,
        reasoning=[
            "§P.2 CPB-0093: positional / weight-bearing MRI is "
            "experimental for EDS and hypermobility disorders.",
            "Hard designation, no exception pathway.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_010",
        icds=["M96.1"],
        notes=(
            "61 y/o M, 6 months post L4-L5 discectomy. "
            "Asymptomatic per chart. Surgeon orders routine "
            "follow-up lumbar MRI. No new symptoms, no new "
            "neurological deficits, no clinical change documented."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.NOT_COVERED,
        reasoning=[
            "§C.1 frequency limit (implied here via routine "
            "surveillance): no new deficit, no change, no "
            "qualifying indication.",
            "Asymptomatic routine follow-up not a listed criterion.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_012",
        icds=["M54.5"],
        notes=(
            "52 y/o F admitted for 10/10 severe LBP requiring IV "
            "opioids. Still inpatient. Lumbar MRI ordered by "
            "admitting team. PA request form does NOT explicitly "
            "cite hospitalization or severe-pain-requiring-admission "
            "as the triggering criterion — it cites "
            "'chronic LBP workup'."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MISSING_INFO,
        reasoning=[
            "Clinical scenario would qualify under a "
            "hospitalization / severe-pain exception, but the PA as "
            "submitted does not cite that criterion.",
            "Reviewer evaluates the PA form as submitted, not the "
            "clinical gestalt.",
            "DENY as submitted (documentation gap).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_013",
        icds=["M51.16", "F40.240"],
        notes=(
            "74 y/o F with right L5 radiculopathy meeting §A.1 "
            "clinical criteria. History of claustrophobia. Physician "
            "requesting open MRI. No documentation of any closed-"
            "bore attempt. PA rationale: 'patient prefers open due to "
            "anxiety.'"
        ),
        policy=AETNA_CPB_0236 + "\n\n" + AETNA_CPB_0093,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.NOT_COVERED,
        reasoning=[
            "§P.1 CPB-0093: open MRI requires documented "
            "contraindication, not patient preference.",
            "No failed closed-bore attempt documented.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_015",
        icds=["M48.06"],
        notes=(
            "66 y/o M with LBP. PA form has 'clinical evidence of "
            "spinal stenosis' checked as the indication, but the "
            "chart contains no formal exam findings of neurogenic "
            "claudication, no ABI, no tandem gait documentation, no "
            "pulse exam. Checkbox assertion only."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MISSING_INFO,
        reasoning=[
            "Checked indication not substantiated by documented "
            "clinical findings.",
            "Reviewer expects exam findings supporting the stenosis "
            "assertion (neurogenic claudication, tandem gait).",
            "DENY (insufficient documentation).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_017",
        icds=["M51.16", "S33.5XXA"],  # disc w/ radic + lumbar sprain
        notes=(
            "39 y/o M with lumbar disc herniation from a workplace "
            "lifting injury 3 months ago. Both active Workers' Comp "
            "coverage and commercial Aetna coverage. PT records "
            "submitted are from the Workers' Comp insurer network. "
            "Medical necessity is clinically met for lumbar MRI. "
            "COB (coordination of benefits) indicates WC is the "
            "primary payer for the work-related injury."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.NOT_COVERED,
        reasoning=[
            "COB conflict: Workers' Comp is primary payer for "
            "work-related injury.",
            "Aetna will not adjudicate this PA until primary payer "
            "determination is resolved.",
            "DENY (COB review required; resubmit through WC).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_018",
        icds=["M54.5"],
        notes=(
            "53 y/o F with 10 weeks of documented PT targeting the "
            "lumbar spine. Chart note from physician reads "
            "'per patient, symptoms worsening over past month' but "
            "no serial objective exam findings, no VAS pain scores "
            "tracked, no documented change in motor/sensory exam."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MISSING_INFO,
        reasoning=[
            "Progressive symptoms requires clinical substantiation "
            "(serial exams, VAS scores, objective change).",
            "Single note quoting the patient does not constitute "
            "documented progression.",
            "DENY (insufficient objective documentation).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_020",
        icds=["M51.36"],  # lumbar degenerative
        notes=(
            "69 y/o M scheduled for L3-L5 fusion. Surgeon requests "
            "'BoneMRI' (MRI-based synthetic CT) for surgical planning, "
            "citing lower radiation dose vs standard CT. Clinical "
            "presentation otherwise reasonable."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.EXPERIMENTAL,
        reasoning=[
            "§G.1 BoneMRI is experimental/investigational for "
            "spinal pre-operative assessment and surgical planning.",
            "Hard designation; no exception pathway absent national "
            "coverage change.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_025",
        icds=["M51.16"],
        notes=(
            "71 y/o M enrolled in an Aetna MEDICARE ADVANTAGE plan. "
            "PA submitted by physician citing Aetna CPB-0236 "
            "criteria. Patient clinically meets commercial criteria "
            "(documented 6 weeks lumbar PT, right L5 radiculopathy "
            "with objective reflex and motor findings). However, "
            "submission is under CPB-0236, NOT under CMS LCD L33408 "
            "which is the governing coverage determination for "
            "Medicare Advantage beneficiaries."
        ),
        policy=AETNA_CPB_0236 + "\n\n" + CMS_LCD_L33408,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.CODING_ERROR,
        reasoning=[
            "§M.1 plan-type requirement: Medicare Advantage "
            "beneficiaries are governed by CMS LCD L33408, not "
            "commercial CPB-0236.",
            "PA submitted under wrong policy framework.",
            "Denied as plan-type error regardless of clinical merit.",
            "Provider must resubmit under LCD L33408 with §M.2 "
            "attestation and §M.3 current exam findings.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_026",
        icds=["M51.26"],
        notes=(
            "60 y/o F had lumbar MRI 8 months ago showing L4-L5 "
            "disc herniation. Physician now orders repeat MRI. Chart "
            "note states 'follow-up imaging' with no documented new "
            "symptoms, no new neurological findings, and no change "
            "in clinical status since prior scan."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.FREQUENCY_LIMIT,
        reasoning=[
            "§C.1 frequency limit: repeat MRI within 12 months not "
            "covered absent new deficit, clinical change, post-op, "
            "or radiation.",
            "'Follow-up' without clinical justification = routine "
            "surveillance = excluded.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_027",
        icds=["M79.7", "M54.5"],  # fibromyalgia + LBP
        notes=(
            "45 y/o F with diagnosed fibromyalgia and chronic diffuse "
            "LBP. No focal neurological findings, no radiculopathy, "
            "no red flags. Physician rationale: 'evaluate pain source.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "Fibromyalgia-related diffuse pain without specific "
            "spinal indication (radiculopathy, red flag, stenosis) "
            "meets no §A.1 or §B.1 criterion.",
            "MRI will not change management in this context.",
            "§F.1 AHCPR guidance applies.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_028",
        icds=["M51.16"],
        notes=(
            "55 y/o M with documented bilateral L5-S1 radiculopathy "
            "and objective reflex changes. Chart contains 8 weeks of "
            "PT records — but the PT was for SHOULDER rehabilitation "
            "following a rotator cuff repair, not for the lumbar "
            "spine. PT discharge summary explicitly states shoulder "
            "focus."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MISSING_INFO,
        reasoning=[
            "§D.1 conservative therapy definition: PT must target "
            "the lumbar spine specifically.",
            "Shoulder PT does not satisfy lumbar conservative "
            "therapy prerequisite.",
            "§A.1 criterion unmet.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_029",
        icds=["M43.06"],  # spondylolysis lumbar
        notes=(
            "16 y/o M competitive gymnast with acute low back pain "
            "after hyperextension. Physician orders lumbar MRI to "
            "evaluate for spondylolysis. Reviewer notes that per ACR "
            "appropriateness guidelines, CT is the gold standard for "
            "spondylolysis diagnosis; MRI may be acceptable if "
            "radiation concerns are documented. PA submission does "
            "NOT document radiation rationale."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "Modality appropriateness conflict: CT is preferred "
            "study per ACR for spondylolysis.",
            "MRI may be approved if physician documents radiation "
            "concern given patient age — NOT documented.",
            "As submitted, MRI request denied; provider should "
            "resubmit CT OR amend with radiation rationale.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_030",
        icds=["M51.16"],
        notes=(
            "41 y/o F submitted PA under Aetna Commercial PPO. "
            "During review, discovered that the member was "
            "reclassified to Aetna MEDICAID managed care mid-month "
            "due to income eligibility redetermination. Commercial "
            "CPB-0236 clinical criteria were met. Medicaid managed "
            "care has a separate PA pathway with additional "
            "step-therapy prerequisites not satisfied by this "
            "submission."
        ),
        policy=AETNA_CPB_0236 + "\n\n" + AETNA_MEDICAID_MANAGED,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.CODING_ERROR,
        reasoning=[
            "§X.1 plan-type requirement: Medicaid members use a "
            "separate PA pathway, NOT commercial CPB-0236.",
            "§X.2 plan type is bound to date-of-service, not "
            "date-of-submission. Mid-month reclassification "
            "requires PA reset.",
            "Regardless of commercial clinical merit, PA is invalid "
            "under the member's current plan type.",
            "DENY (must restart under Medicaid pathway).",
        ],
        difficulty=Difficulty.HARD,
    ))

    return cases
