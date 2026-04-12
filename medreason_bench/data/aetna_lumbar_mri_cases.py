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
§A.1 STANDARD COVERAGE: Lumbar MRI is covered after documented failure of >=6 weeks conservative therapy (physical therapy for the lumbar spine AND NSAIDs or equivalent) in a patient with lumbar radiculopathy evidenced by both pain AND objective motor or reflex changes on exam. The PT clock runs from treatment initiation date, not symptom onset date. PT course must be contiguous and substantially consistent — non-contiguous blocks with gaps or PRN-only NSAID use do not satisfy the requirement.
§A.2 ADDITIONAL STANDALONE APPROVAL PATHS (no conservative therapy wait required):
  - Pre-epidural injection evaluation: MRI to rule out tumor or infection and to delineate the optimal anatomical injection site
  - Post-surgical recurrent symptoms evaluation: MRI with and without gadolinium (CPT 72158) is the preferred modality to distinguish post-operative scar from recurrent disc
  - Severe back pain requiring hospitalization OR ED presentation requiring IV opioid analgesia (ED with IV opioids is the clinical equivalent of hospitalization-level severity for this criterion)
  - Progressively severe symptoms despite conservative management — requires substantiation by serial quantified data (VAS score escalation, documented functional decline, emerging objective motor or reflex changes over multiple visits)
§A.3 SPONDYLOLISTHESIS-SPECIFIC THRESHOLD: For radiographically-confirmed spondylolisthesis, the conservative therapy threshold is 4 WEEKS (not the 6-week radiculopathy threshold). PT must be contiguous without multi-week gaps; PRN-only NSAID use does not satisfy consistency. Two distinct timeline thresholds exist within §A.1 — the 6-week rule applies to radiculopathy, the 4-week rule applies to spondylolisthesis.
§A.4 SUSPECTED INFECTIOUS PROCESS: Suspected osteomyelitis, epidural abscess, or spondylodiscitis is a standalone approval path with no conservative therapy wait. Typical trigger pattern: elevated inflammatory markers (CRP, ESR, WBC) + focal spinal percussion tenderness + high-risk host (IV drug use, immunocompromised status, recent bacteremia, post-procedural).
§B.1 INDEPENDENT CRITERIA (no conservative therapy wait required):
  (a) Clinical suspicion of cauda equina syndrome or spinal cord compression. Policy requires SUSPICION, not the confirmed classic triad — any combination of saddle anesthesia, bowel/bladder dysfunction, bilateral lower extremity motor weakness, or diminished perianal sensation in context of severe LBP is sufficient. Urinary retention + reduced perianal sensation alone meets suspicion threshold.
  (b) Rapidly progressing neurological deficit, OR major motor weakness. Both prongs are independently sufficient. Rapid progression requires serial objective measurements showing deterioration — this is satisfied by documented strength decline across visits regardless of whether the words "progressive" or "rapidly" appear in the physician's language. Major motor weakness at a single time point (e.g., 0/5 dorsiflexion) is also sufficient even without temporal progression.
  (c) Suspected vertebral, paraspinal, or intraspinal metastatic disease with supporting evidence. Evidence pattern: known primary malignancy + new back pain, OR rising tumor marker + new LBP, OR unexplained weight loss + LBP (occult primary — no prior cancer diagnosis required; "suspected" covers occult presentations).
  (d) Suspected vertebral fracture / differentiation of benign vs pathological fracture in osteoporotic patient with inconclusive plain films
  (e) Suspected lumbar epidural lipomatosis in patient on chronic systemic steroids
  (f) Suspected lumbar arachnoiditis (prior intrathecal procedures + appropriate clinical picture)
  (g) Suspected myelopathy with lumbar segment clinically implicated
  (h) Failed back surgery syndrome with NEW objective neurological deficit post-operatively. "New" means not present on the prior post-op exam. Absence of documented deficit (asymptomatic / "no new neurological findings") does NOT trigger this criterion.
  (i) Congenital spinal anomaly / scoliosis pre-operative planning (including rule-out syrinx)
§C.1 FREQUENCY LIMIT: Repeat lumbar MRI within 12 months of a prior study is NOT covered unless one of: new neurological deficit since prior imaging, significant clinical change, post-operative evaluation, or interval radiation therapy. Routine surveillance for chronic pain is excluded. Improved or unchanged symptoms do NOT constitute clinical change. A new contralateral deficit (e.g., new right-sided findings in a patient with documented left-sided prior imaging) IS a new clinical indication.
§D.1 CONSERVATIVE THERAPY DEFINITION (FOOTNOTE 1): The accepted forms of conservative therapy under this policy are: moderate activity modification, analgesics, NSAIDs, muscle relaxants, and supervised physical therapy targeting the lumbar spine. Acupuncture, chiropractic manipulation, yoga, and massage are NOT accepted as conservative therapy under CPB-0236 even though they may be accepted under CMS LCD or other payer policies. Physical therapy must target the lumbar spine specifically (shoulder or other regional PT does not satisfy the prerequisite). Out-of-network PT requires documentation provided by the OON facility; patient self-report is insufficient.
§E.1 SUBJECTIVE RADICULOPATHY: Radiculopathy must be evidenced by both pain AND objective motor, sensory, or reflex changes on exam. Pain alone without objective findings does not satisfy the radiculopathy criterion.
§F.1 AHCPR GUIDANCE: Routine imaging for acute low back pain without red-flag symptoms is not covered (AHCPR recommends against imaging in acute LBP without red flags). Patient preference for imaging is not a red flag and does not create medical necessity. Physician notes citing "patient insists" do not substitute for objective indication.
§G.1 BONE-MRI EXPERIMENTAL CLAUSE: BoneMRI (MRI-based synthetic CT) is classified as experimental/investigational for spinal pre-operative assessment and surgical planning. Not covered regardless of clinical rationale.
§H.1 APPEAL PRECEDENT: Conservative therapy shortfalls (e.g., 5 weeks when 6 are required) may be overridden on appeal if the physician documents progressing motor weakness in parallel, invoking the rapidly-progressing-deficit clause. This override applies only to MARGINAL shortfalls WITH parallel progressive-deficit documentation. Total absence of therapy or absent deficit documentation does not qualify.
§I.1 COORDINATION OF BENEFITS: Aetna CPB-0236 governs as the authorization gate only when Aetna is the primary payer. When Medicare is primary per Medicare Secondary Payer rules (retirement employer with <100 employees, Medicare Advantage, etc.), CPB-0236 does not bind the authorization — Aetna follows Medicare's LCD determination as secondary. When Workers' Compensation is primary for a work-related injury, Aetna will not adjudicate authorization until the primary payer determination resolves.
§J.1 PLAN-TYPE AND REFERRING PROVIDER: Under Aetna HMO and EPO plans, prior authorization submissions must come from an in-network referring physician — OON-physician NPI on HMO/EPO cases invalidates the PA regardless of clinical merit. Under PPO plans, out-of-network referrals are permitted. Plan-type verification must precede clinical-criteria review.
§K.1 DOCUMENTATION AUTHENTICITY AND TEMPLATE LANGUAGE: §B.1(b) rapidly-progressing-deficit, §A.1 progressively-severe-symptoms, and red-flag checkbox criteria must be substantiated by the documented clinical data — not by physician template language. "Neurological deterioration", "red flags present", and "progressively worsening" phrasings that are not supported by serial objective measurements, quantified strength findings, or documented personal clinical red-flag history do not satisfy the criteria. Chronic stable findings re-labeled as "progressive", family history of malignancy without personal red flags, and subjective "may be progressing" speculation directly contradicted by normal exam do not qualify."""

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

    # ═══════════════════════════════════════════════════════════════════
    # TIER 1 — STRAIGHTFORWARD (cases 31-40)
    # Clean standard pathways. Regressions here signal retrieval
    # precision problems. Outcome split: 6 approve + 4 deny.
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="aetna_031",
        icds=["M51.16"],
        notes=(
            "54 y/o M with right L3-L4 radiculopathy radiating to "
            "anterior thigh. Right quad 4/5, patellar reflex 1+ "
            "right vs 2+ left, dermatomal sensory loss along right "
            "L3-L4. PT discharge summary documents 16 sessions over "
            "8 weeks targeting the lumbar spine. Naproxen 500mg BID "
            "continuously for 8 weeks, no meaningful improvement."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.1 standard path — all three elements present.",
            "Radiculopathy + objective motor/reflex findings + "
            "8 weeks contiguous conservative therapy.",
            "§D.1 Footnote 1 satisfied by PT + NSAIDs.",
            "APPROVE.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_032",
        icds=["M54.9"],
        notes=(
            "33 y/o F with 2-week LBP after moving furniture. Normal "
            "neuro exam (5/5 bilateral, 2+ symmetric reflexes, "
            "negative SLR). No PT, no medications. Physician note: "
            "'Patient requests MRI to rule out a slipped disc.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§F.1 AHCPR guidance: no imaging for acute LBP without "
            "red flags.",
            "Zero conservative therapy, normal exam, no objective "
            "findings, no red flags.",
            "Patient preference does not create medical necessity.",
            "DENY.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_033",
        icds=["M51.17"],
        notes=(
            "61 y/o F with left S1 radiculopathy x12 weeks. Plantar "
            "flexion 3/5 left (unable to perform single heel raise). "
            "Achilles reflex absent left, 2+ right. Lateral foot "
            "sensory loss. 12 weeks PT (3x/week), ibuprofen 800mg "
            "TID continuously, AND 18 chiropractic sessions — all "
            "failed. PA form lists: PT, NSAIDs, chiropractic."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.1 standard path — criteria met by PT + NSAIDs alone.",
            "§D.1: chiropractic is NOT in Footnote 1 but its presence "
            "does NOT disqualify otherwise-sufficient PT + NSAIDs.",
            "Radiculopathy + objective motor/reflex findings + "
            "12 weeks PT + NSAIDs.",
            "APPROVE (chiropractic is incidental, not disqualifying).",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_034",
        icds=["M54.9"],
        notes=(
            "41 y/o M with 4-month intermittent LBP 3/10 after desk "
            "work. Full ROM, normal neuro exam. 3 sessions of yoga "
            "(patient-initiated). Physician note: 'Patient counseled "
            "but insists on MRI.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§D.1: yoga is NOT in Footnote 1 conservative therapy list.",
            "§A.1 conservative therapy prerequisite unmet.",
            "Normal exam, no objective findings, no red flags.",
            "§F.1 patient insistence is not medical necessity.",
            "DENY.",
        ],
        difficulty=Difficulty.EASY,
    ))

    cases.append(_case(
        case_id="aetna_035",
        icds=["M43.16"],
        notes=(
            "48 y/o M with Grade I L4-L5 spondylolisthesis confirmed "
            "on X-ray. PT for 3 weeks, then a 2-week gap, then "
            "resumed for 2 more weeks. NSAIDs used PRN only ('when "
            "it gets bad'). No radiculopathy. Surgical planning "
            "requested."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§A.3 spondylolisthesis threshold is 4 weeks CONTIGUOUS.",
            "PT here is non-contiguous (3 + gap + 2), not consistent.",
            "NSAIDs are PRN-only, not consistent use.",
            "§A.3 consistency requirement unmet.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_036",
        icds=["M43.16"],
        notes=(
            "52 y/o F with X-ray confirmed Grade II L5-S1 "
            "spondylolisthesis. PT 3x/week for 4 consecutive weeks. "
            "Naproxen 500mg BID daily for 4 weeks. No improvement, "
            "ADLs limited. No radiculopathy."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.3 spondylolisthesis-specific 4-week threshold "
            "(NOT the 6-week radiculopathy threshold).",
            "PT contiguous 4 weeks + daily NSAIDs satisfies §A.3 "
            "consistency requirement.",
            "Two distinct thresholds exist within §A.1 — applying "
            "the 6-week rule here is incorrect.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_037",
        icds=["M51.16"],
        notes=(
            "67 y/o M with known L4-L5 disc herniation, scheduled "
            "for lumbar epidural steroid injection next week. Pain "
            "management physician requests MRI to rule out tumor or "
            "infection and to delineate optimal injection site. No "
            "prior conservative therapy documented for this visit."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.2 pre-epidural injection evaluation is a standalone "
            "path — no conservative therapy wait applies.",
            "Rule out tumor/infection + injection site planning.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_038",
        cpt="72158",
        icds=["M96.1"],
        notes=(
            "58 y/o M, 9 months post-L5-S1 microdiscectomy. Initially "
            "improved, now with recurrent left S1 radiculopathy. "
            "Diminished Achilles reflex left, plantar flexion 4/5 "
            "left, SLR positive left 50 degrees. Surgeon requests "
            "MRI with and without gadolinium to distinguish "
            "post-operative scar from recurrent disc."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.2 post-surgical recurrent-symptoms evaluation — "
            "CPT 72158 (with and without contrast) is the "
            "preferred modality.",
            "No conservative therapy re-required for post-surgical "
            "evaluation.",
            "§B.1(h) new objective deficit also supports approval.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_039",
        icds=["M46.26"],
        notes=(
            "62 y/o F, IV drug user, 10-day progressive LBP, fever "
            "38.9°C, CRP 184 mg/L, ESR 112 mm/hr, midline percussion "
            "tenderness at L3-L4. No conservative therapy. Neuro "
            "exam intact."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.4 suspected infectious process — standalone approval.",
            "Pattern: elevated inflammatory markers + focal "
            "tenderness + high-risk host (IVDU).",
            "No conservative therapy wait applies to red flag "
            "infectious presentations.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_040",
        icds=["M51.16"],
        notes=(
            "55 y/o M, MRI 7 months ago showed L4-L5 herniation. "
            "PT completed post-MRI, symptoms improved significantly, "
            "pain now 1-2/10. SLR negative, motor 5/5, reflexes "
            "symmetric. Physician orders 'routine follow-up MRI.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.FREQUENCY_LIMIT,
        reasoning=[
            "§C.1 frequency limit: repeat MRI within 12 months.",
            "Improved symptoms and normal exam is NOT clinical "
            "change for §C.1 purposes — routine surveillance.",
            "No active criterion met.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2 — EXCEPTION CASES (cases 41-52)
    # §B.1 override pattern generalization across sub-criteria.
    # Outcome split: 10 approve + 2 deny.
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="aetna_041",
        icds=["M51.16"],
        notes=(
            "59 y/o M with right foot drop developing rapidly. Ankle "
            "dorsiflexion declined from partial weakness to 0/5 "
            "(complete foot drop) over 7 days. Right EHL 1/5. "
            "Steppage gait. PT started 2 weeks ago for LBP."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(b) — 0/5 dorsiflexion = major motor weakness.",
            "AND actively progressing over 7 days = rapid progression.",
            "Both prongs independently present.",
            "§A.1 6-week wait overridden; only 2 weeks PT irrelevant.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_042",
        icds=["G83.4"],
        notes=(
            "47 y/o F with 4-day progressive LBP with new urinary "
            "hesitancy and two episodes of urinary retention "
            "requiring self-catheterization. Perianal sensation "
            "diminished on exam (not spontaneously reported by "
            "patient). Bilateral LE strength 5/5. No saddle "
            "anesthesia volunteered."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(a) requires SUSPICION, not the confirmed classic "
            "triad.",
            "Urinary retention + reduced perianal sensation + "
            "severe LBP = sufficient clinical suspicion.",
            "Policy language is broader than 'saddle anesthesia'.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_043",
        icds=["G83.4"],
        notes=(
            "55 y/o M with severe LBP x5 days. Bilateral hip/knee/"
            "ankle weakness developing over 48 hours (bilateral "
            "3-4/5). Bilateral Achilles reflexes diminished from "
            "prior documented baseline (2+ → 1+). No bladder "
            "symptoms reported yet. SLR positive bilaterally."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "Dual trigger: §B.1(a) suspicion of cauda equina "
            "(bladder involvement may not yet be apparent).",
            "AND §B.1(b) rapidly progressing bilateral deficit.",
            "Waiting for explicit bladder symptoms = harmful delay.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_044",
        icds=["M51.16"],
        notes=(
            "63 y/o F with 1-week right leg weakness. Physician "
            "writes: 'Rapidly progressive right lower extremity "
            "weakness — strength declined from 4/5 to 2/5 over past "
            "5 days per serial exams across 3 visits this week.' "
            "Serial exam data: hip flexion 4/5 → 2/5, knee extension "
            "4/5 → 3/5, ankle dorsiflexion 4/5 → 2/5 over 5 days."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(b) rapidly progressing neurological deficit.",
            "Explicit physician attestation + quantified serial "
            "strength decline.",
            "§K.1 substantiation: serial data supports the claim.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_045",
        icds=["M51.16"],
        notes=(
            "57 y/o M. Visit 1 (14 days ago): tibialis anterior 5/5, "
            "EHL 5/5, SLR at 60°, Achilles 2+. Visit 2 (today): "
            "tibialis anterior 1/5, EHL 2/5, foot drop emerging, "
            "Achilles 1+. Physician writes: 'Given worsening, "
            "imaging indicated.' Physician does not use the words "
            "'progressing' or 'rapidly'."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(b) rapid progression — serial measurements show "
            "5/5 → 1/5 tibialis anterior over 14 days.",
            "§K.1: rule applies to CLINICAL DATA, not keywords. "
            "Absence of the word 'progressing' is irrelevant.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_046",
        icds=["C79.51"],
        notes=(
            "66 y/o M, 'routine follow-up' visit. Chief complaint: "
            "'back hurts more lately, maybe from sleeping wrong.' "
            "PMH buried in EMR: prostate cancer x3 years on "
            "enzalutamide. PSA elevated from 0.3 → 8.4 over 6 months "
            "per oncology note. New-onset LBP x3 weeks. No "
            "radiculopathy. Physician assessment: 'LBP, possible "
            "musculoskeletal. MRI to evaluate.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(c) suspected vertebral metastases.",
            "Known primary + rising tumor marker + new back pain.",
            "Physician's understated framing doesn't negate facts.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_047",
        icds=["M46.26"],
        notes=(
            "44 y/o M, 'routine LBP visit'. PMH: kidney transplant "
            "on tacrolimus + mycophenolate. Temperature 37.9°C. "
            "Point tenderness over L3 spinous process. CRP 42, "
            "WBC 11.2. Physician's plan: 'Naproxen + PT.' MRI "
            "requested after NP flags immunosuppressed status + "
            "fever in chart review."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.4 suspected infectious spondylodiscitis.",
            "Immunocompromised + fever + focal tenderness + "
            "elevated markers = mandatory imaging.",
            "Red flag in PMH, not surfaced in assessment — but "
            "clinically decisive.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_048",
        icds=["C79.51"],
        notes=(
            "71 y/o F with 6-week mid/low back pain. Unintentional "
            "14-lb weight loss over 3 months (chart confirmed: "
            "132 lbs → 118 lbs). No known malignancy. No other "
            "constitutional symptoms. Normal neuro exam."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§B.1(c) suspected vertebral metastases (occult primary).",
            "Known cancer is NOT required — 'suspected' covers "
            "occult presentations with weight loss red flag.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_049",
        icds=["M54.16"],
        notes=(
            "49 y/o M with 5-week LBP and intermittent right leg "
            "numbness ('comes and goes, a few minutes daily'). "
            "4 weeks PT, ibuprofen PRN. Neuro exam fully normal: "
            "5/5 bilaterally, 2+ symmetric reflexes, SLR negative. "
            "Physician writes: 'Patient has intermittent "
            "neurological symptoms that may be progressing.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§B.1(b) NOT triggered — exam is fully normal.",
            "§K.1: 'may be progressing' is speculative, directly "
            "contradicted by normal exam.",
            "§A.1 not met (4 weeks PT, PRN NSAIDs).",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_050",
        icds=["M51.16"],
        notes=(
            "64 y/o F with chronic L4-L5 radiculopathy. Right quad "
            "weakness at 4/5 STABLE for 8 months across 3 documented "
            "visits. Today: right quad 4/5 (unchanged). Patellar "
            "reflex 1+ right (chronic, present since first visit). "
            "5 weeks PT + NSAIDs. PA form checkbox: 'Major motor "
            "weakness.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§B.1(b) requires RAPID progression.",
            "§K.1: 8-month stable weakness doesn't meet 'rapidly "
            "progressing' regardless of checkbox label.",
            "§A.1 at 5 weeks — one more week completes conservative "
            "therapy. Peer-to-peer recommended.",
            "DENY (as submitted).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_051",
        icds=["M54.59"],
        notes=(
            "58 y/o M in ED with acute LBP 10/10, unable to "
            "ambulate. IV morphine 8mg administered. ED physician: "
            "'Severe acute LBP requiring IV opioid analgesia and "
            "observation.' No radiculopathy. No prior conservative "
            "therapy."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.2 severe back pain requiring hospitalization OR ED "
            "with IV opioid analgesia.",
            "ED + IV opioids is the clinical equivalent of "
            "hospitalization-level severity.",
            "No conservative therapy wait applies.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_052",
        icds=["M54.16"],
        notes=(
            "53 y/o F with 10 weeks LBP + right L5 radiculopathy. "
            "8 weeks PT + naproxen BID. Serial VAS documented: "
            "Week 1: 5/10 → Week 3: 6/10 → Week 5: 7/10 → Week 8: "
            "8/10. Functional decline documented at each visit — "
            "now unable to dress independently. Right EHL 4/5 at "
            "Week 8 (was 5/5 at Week 1). Right Achilles 1+ "
            "developing (was 2+ at Week 1)."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§A.2 progressively severe symptoms despite conservative "
            "management.",
            "§K.1 substantiation: quantified VAS escalation + "
            "functional decline + emerging motor/reflex changes.",
            "APPROVE.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    # ═══════════════════════════════════════════════════════════════════
    # TIER 3 — ADVERSARIAL (cases 53-60)
    # Operational footnotes that zero-shot and sparse-RAG miss.
    # Outcome split: 2 approve + 6 deny.
    # ═══════════════════════════════════════════════════════════════════

    cases.append(_case(
        case_id="aetna_053",
        icds=["M51.16"],
        notes=(
            "46 y/o F with 8-week LBP and right L4 radiculopathy. "
            "Patellar reflex 1+ right vs 2+ left, right quad 4/5. "
            "Completed 16 sessions of acupuncture (8 weeks, 2x/week). "
            "No PT, no NSAIDs (patient declines due to GI history), "
            "no muscle relaxants. PA form states: '8 weeks "
            "conservative therapy — acupuncture.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§D.1 Footnote 1: acupuncture is NOT in the accepted "
            "conservative therapy list (PT, NSAIDs, analgesics, "
            "muscle relaxants, activity modification).",
            "Aetna-specific footnote overrides general clinical "
            "convention regardless of other payer policies.",
            "§A.1 conservative therapy prerequisite unmet.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_054",
        icds=["M51.16"],
        notes=(
            "50 y/o M with LBP + left L5 radiculopathy. Symptoms "
            "began 10 weeks ago. Patient did not seek treatment "
            "until 4 weeks ago. PT started 4 weeks ago (8 sessions "
            "completed). Naproxen not mentioned. Physician note: "
            "'Patient has had symptoms for 10 weeks, which exceeds "
            "the 6-week conservative therapy threshold.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§A.1 PT clock runs from treatment initiation, not "
            "symptom onset.",
            "Active PT = 4 weeks only (not 10).",
            "NSAIDs undocumented.",
            "Physician's conflation of symptom duration with "
            "therapy duration is incorrect.",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_055",
        icds=["M51.16"],
        notes=(
            "57 y/o M with prior lumbar MRI 9 months ago (correctly "
            "approved) showing L4-L5 herniation with left L5 "
            "radiculopathy. Today: same symptoms, same severity, "
            "same distribution. Neuro exam identical to 9-month-ago "
            "visit: left EHL 4/5, left patellar reflex 1+, SLR "
            "positive left 45° — all unchanged. Physician writes: "
            "'Repeat MRI to reassess disc herniation.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.FREQUENCY_LIMIT,
        reasoning=[
            "§C.1 repeat MRI within 12 months requires documented "
            "clinical change.",
            "Exam findings documented as identical to prior visit — "
            "no clinical change.",
            "'Reassessment' without new indication is not covered.",
            "DENY.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="aetna_056",
        icds=["M51.16"],
        notes=(
            "57 y/o M, 11 months after prior MRI. New symptom: "
            "right leg weakness developed 2 weeks ago — "
            "contralateral to prior left L5 findings. Right "
            "tibialis anterior 3/5 (previously 5/5 on all prior "
            "visits). Right SLR positive 40° (new). Left findings "
            "unchanged. Physician: 'New right-sided radiculopathy "
            "with motor deficit — clinically distinct from prior "
            "presentation.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§C.1 new contralateral objective finding = documented "
            "clinical change = new independent indication.",
            "Paired with aetna_055 to isolate the clinical-change "
            "variable.",
            "APPROVE.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_057",
        icds=["M51.16"],
        notes=(
            "52 y/o F. All medical necessity criteria met per "
            "CPB-0236 (7 weeks PT, NSAIDs, right L5 radiculopathy "
            "with objective findings). Referring physician is "
            "out-of-network. Imaging facility is in-network. PA "
            "submitted under OON physician's NPI. Member is on "
            "Aetna HMO plan."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.CODING_ERROR,
        reasoning=[
            "§J.1 under Aetna HMO/EPO, OON referring physician "
            "invalidates the PA regardless of clinical merit.",
            "Plan-type verification precedes clinical-criteria "
            "review.",
            "Medical necessity is unambiguously satisfied — "
            "irrelevant under the structural issue.",
            "DENY (must resubmit via in-network physician).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_058",
        icds=["M51.16"],
        notes=(
            "68 y/o M, Medicare-eligible (Part B), Aetna commercial "
            "supplemental through retirement employer (fewer than "
            "100 employees — Medicare is primary per MSP rules). "
            "Classic L4-L5 radiculopathy, 7 weeks PT, NSAIDs, "
            "objective findings. Meets both Medicare LCD and Aetna "
            "CPB-0236. PA submitted to Aetna."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.APPROVED,
        reasoning=[
            "§I.1 COB: Medicare is primary per MSP rules.",
            "Aetna as secondary follows Medicare's LCD.",
            "Both Medicare LCD and CPB-0236 criteria clinically met.",
            "APPROVE (Aetna does not gate as secondary; if Aetna "
            "must respond it cannot deny when clinical criteria "
            "are met and Medicare approves).",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_059",
        icds=["M51.16"],
        notes=(
            "61 y/o F with chronic L5-S1 disc herniation and left "
            "S1 radiculopathy x3 years. Left Achilles reflex absent "
            "for 3+ years (confirmed in all prior notes — "
            "unchanged). Left plantar flexion 4/5 — identical to "
            "all prior visits. Today's physician note: 'Patient "
            "continues to demonstrate neurological deterioration "
            "consistent with chronic L5-S1 radiculopathy.' 4 weeks "
            "PT this episode. PA form cites 'neurological "
            "deterioration' as the §B.1(b) trigger."
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§K.1 template language vs clinical data: 'neurological "
            "deterioration' phrasing directly contradicted by "
            "3-year-stable findings.",
            "§B.1(b) requires rapid progression. Stable chronic "
            "deficit does not qualify.",
            "§A.1 unmet (4 weeks PT).",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="aetna_060",
        icds=["M54.16"],
        notes=(
            "48 y/o F with 5-week LBP + left L4 radiculopathy. "
            "Left patellar reflex 1+ right 2+, left quad 4/5. "
            "4 weeks PT + naproxen. PA form: 'Red flags present' "
            "checkbox CHECKED. Physician note: 'Red flags: "
            "positive family history of colon cancer (mother, age "
            "72). Patient has no personal history of malignancy. "
            "No weight loss, no fever, no night sweats, no "
            "constitutional symptoms.'"
        ),
        policy=AETNA_CPB_0236,
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "§K.1 checkbox requires substantiating personal "
            "clinical red-flag findings.",
            "Family history of cancer is not a personal red flag.",
            "No personal weight loss, fever, immunosuppression, "
            "IVDU, or known malignancy.",
            "§A.1 also unmet (4 weeks PT).",
            "DENY.",
        ],
        difficulty=Difficulty.HARD,
    ))

    return cases
