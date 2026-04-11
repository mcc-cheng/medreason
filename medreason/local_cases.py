"""Pre-written realistic benchmark cases for offline/local mode."""

from __future__ import annotations

from .ontology import (
    BenchmarkCase, DenialReason, Difficulty, FacilityType, Outcome, Payer,
    PriorAuthTaskConfig,
)

LOCAL_CASES: list[BenchmarkCase] = [
    # ── EASY — APPROVED ────────────────────────────────────────────────────────
    BenchmarkCase(
        case_id="local-001",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="99213",
            icd10_codes=["M54.5", "M79.3"], facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "45yo F presents with chronic low back pain for 6 months. "
            "Failed NSAIDs (ibuprofen 800mg TID x 8 weeks) and structured home exercise program. "
            "ROM limited to 40 degrees flexion. Positive SLR bilateral. "
            "Reports difficulty with ADLs including dressing and prolonged sitting. "
            "Requesting office visit for pain management evaluation."
        ),
        policy_excerpt=(
            "Office visits for chronic pain management require documented failure of conservative "
            "treatment >6 weeks, objective functional impairment findings, and impact on ADLs. "
            "Must include ROM measurements or validated pain scale scores."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Patient has 6 months of documented chronic LBP meeting chronicity threshold",
            "Conservative treatment failure documented: NSAIDs x 8 weeks + home exercise program",
            "Objective findings present: ROM limitation to 40 degrees, positive bilateral SLR",
            "Functional impairment documented: difficulty with ADLs",
            "All policy criteria met — approve office visit",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-002",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="43239",
            icd10_codes=["K21.0", "K25.9"], facility_type=FacilityType.ASC,
        ),
        clinical_notes=(
            "62yo M with refractory GERD for 4 months despite twice-daily PPI therapy "
            "(omeprazole 40mg BID). Presents with progressive dysphagia to solids and "
            "unintentional weight loss of 12 lbs. Family history significant for gastric cancer "
            "in father. Requesting upper GI endoscopy with biopsy."
        ),
        policy_excerpt=(
            "Upper GI endoscopy with biopsy is covered when patient has failed PPI trial >8 weeks "
            "OR presents with alarm symptoms including dysphagia, unintentional weight loss >10 lbs, "
            "or family history of upper GI malignancy."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Patient has refractory GERD despite adequate PPI trial (omeprazole 40mg BID x 4 months)",
            "Multiple alarm symptoms present: dysphagia and 12 lb weight loss",
            "Family history of gastric cancer is additional alarm indicator",
            "Policy covers EGD with alarm symptoms OR failed PPI >8 weeks — patient meets both",
            "Approve upper GI endoscopy with biopsy",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-003",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="90837",
            icd10_codes=["F33.1", "F41.1"], facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "34yo F with recurrent major depressive disorder and generalized anxiety disorder. "
            "PHQ-9 score of 18 (moderately severe). GAD-7 score of 15 (severe anxiety). "
            "Currently on sertraline 100mg with partial response. Reports missing 8 workdays "
            "in past month due to symptoms. Requesting 60-minute psychotherapy sessions."
        ),
        policy_excerpt=(
            "Individual psychotherapy (60 min) is covered when PHQ-9 >= 15 or GAD-7 >= 10, "
            "with documented functional impairment (occupational, social, or daily living), "
            "and concurrent pharmacotherapy or documented contraindication to medication."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "PHQ-9 of 18 exceeds threshold of 15 for moderate-severe depression",
            "GAD-7 of 15 exceeds threshold of 10 for anxiety criteria",
            "Functional impairment documented: 8 missed workdays in past month",
            "Concurrent pharmacotherapy present: sertraline 100mg",
            "All criteria met — approve 60-minute psychotherapy",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-004",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE, cpt_code="20610",
            icd10_codes=["M17.11", "M19.011"], facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "72yo M with bilateral knee osteoarthritis, right worse than left. "
            "Kellgren-Lawrence grade 3 on recent X-ray. Failed acetaminophen and topical "
            "diclofenac over 3 months. Reports pain 7/10 with weight bearing, difficulty "
            "climbing stairs. Requesting corticosteroid injection right knee."
        ),
        policy_excerpt=(
            "Intra-articular corticosteroid injection of a major joint is covered for documented "
            "osteoarthritis with radiographic evidence (KL grade >= 2), failed oral analgesics "
            ">4 weeks, and functional limitation. Limited to 4 injections per joint per year."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Radiographic evidence confirms KL grade 3 OA, exceeding minimum KL >= 2",
            "Failed conservative therapy: acetaminophen and topical diclofenac for 3 months",
            "Functional limitation documented: difficulty with stairs, weight-bearing pain 7/10",
            "No frequency limit concern — this appears to be initial injection request",
            "Approve corticosteroid injection right knee",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-005",
        task_config=PriorAuthTaskConfig(
            payer=Payer.HUMANA, cpt_code="97110",
            icd10_codes=["M54.5", "S83.511A"], facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "55yo F s/p left knee meniscal tear repair 3 weeks ago. Post-surgical ROM limited "
            "to 60 degrees flexion. Moderate effusion present. Unable to bear full weight. "
            "Requesting therapeutic exercises for post-operative rehabilitation per surgeon protocol."
        ),
        policy_excerpt=(
            "Therapeutic exercises are covered post-surgically when there is documented functional "
            "deficit, measurable ROM limitation, and physician order. Initial authorization for "
            "up to 12 visits over 6 weeks."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Post-surgical status confirmed: meniscal tear repair 3 weeks prior",
            "Functional deficit documented: unable to bear full weight",
            "ROM limitation measurable: flexion limited to 60 degrees",
            "Effusion indicates ongoing inflammatory process requiring rehabilitation",
            "Approve therapeutic exercises per post-surgical protocol",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-006",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="77386",
            icd10_codes=["C61"], facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "68yo M with intermediate-risk prostate cancer, Gleason 3+4=7, PSA 12.3, T2b. "
            "Multidisciplinary tumor board recommends definitive IMRT. CT simulation completed. "
            "Treatment plan: 44 fractions to 79.2 Gy. ECOG performance status 0."
        ),
        policy_excerpt=(
            "IMRT is covered for prostate cancer when clinically indicated per NCCN guidelines, "
            "with documented staging, tumor board recommendation or radiation oncology evaluation, "
            "and treatment planning with CT simulation."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Intermediate-risk prostate cancer confirmed: Gleason 7, PSA 12.3, T2b",
            "Treatment per NCCN guidelines for intermediate-risk disease",
            "Tumor board recommendation documented",
            "CT simulation completed with appropriate dose fractionation plan",
            "Approve IMRT delivery",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-007",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="70553",
            icd10_codes=["R51.9", "G43.909"], facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "29yo F with new-onset severe headaches x 6 weeks, different from prior migraines. "
            "Focal neurological finding: right-sided visual field deficit on confrontation testing. "
            "No papilledema. Neuro exam otherwise intact. CT head without contrast was normal. "
            "Requesting MRI brain with and without contrast to evaluate for intracranial pathology."
        ),
        policy_excerpt=(
            "MRI brain with/without contrast is covered for new or changed headache pattern with "
            "focal neurological deficit, failed prior imaging (CT), or suspicion for intracranial "
            "pathology. Not covered for routine migraine evaluation without red flags."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "New headache pattern different from prior migraines — not routine migraine evaluation",
            "Focal neurological deficit present: right visual field deficit",
            "Prior imaging (CT) was non-diagnostic — further workup indicated",
            "Red flags present justify advanced imaging per policy",
            "Approve MRI brain with and without contrast",
        ],
        difficulty=Difficulty.EASY,
    ),
    # ── MEDIUM — DENIED ────────────────────────────────────────────────────────
    BenchmarkCase(
        case_id="local-008",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="72148",
            icd10_codes=["M54.5", "M51.16"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "38yo M presents with low back pain for 3 weeks. No radicular symptoms. "
            "Neurological exam within normal limits. No bowel or bladder changes. "
            "Has not tried physical therapy or prescription medications. "
            "Requesting lumbar MRI to evaluate pain."
        ),
        policy_excerpt=(
            "Lumbar MRI without contrast requires 6 weeks of conservative management "
            "(physical therapy, NSAIDs, or muscle relaxants) unless red flags present: "
            "progressive neurological deficit, cauda equina syndrome, suspected malignancy, "
            "or infection. Red flag exceptions must be clearly documented."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Pain duration is only 3 weeks, below the 6-week conservative management threshold",
            "No conservative treatment has been attempted (no PT, no medications)",
            "No red flags: normal neuro exam, no radiculopathy, no bowel/bladder changes",
            "Does not meet medical necessity criteria per policy",
            "Deny — recommend 6 weeks conservative management before imaging",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-009",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="64483",
            icd10_codes=["M54.5", "M54.16"],
            facility_type=FacilityType.ASC,
            denial_reason=DenialReason.FREQUENCY_LIMIT,
        ),
        clinical_notes=(
            "55yo F with chronic lumbar radiculopathy requesting 4th epidural steroid injection "
            "in 12 months. Prior 3 ESIs (Jan, Apr, Jul) provided 2-3 weeks of relief each. "
            "VAS pain scores improved from 8 to 5 transiently but returned to baseline. "
            "No sustained functional improvement documented."
        ),
        policy_excerpt=(
            "Epidural steroid injections are limited to 3 per spinal region per 12-month period. "
            "Extension beyond 3 requires documentation of >50% pain reduction sustained for "
            ">3 months from prior injection. Transient relief <4 weeks does not qualify."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Patient has already received 3 ESIs in 12-month period — at frequency limit",
            "Prior injections provided only 2-3 weeks transient relief, below 3-month threshold",
            "Pain reduction was approximately 37% (8 to 5), below 50% threshold for extension",
            "No sustained functional improvement documented",
            "Deny — frequency limit reached without qualifying sustained response",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-010",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="99214",
            icd10_codes=["M54.5", "G89.29"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.CODING_ERROR,
        ),
        clinical_notes=(
            "52yo M presents for 15-minute follow-up visit. Chronic pain stable on current "
            "medication regimen (gabapentin 300mg TID). No medication changes needed. Brief "
            "focused exam of lumbar spine. Single problem addressed with straightforward "
            "management. Coded as 99214."
        ),
        policy_excerpt=(
            "CPT 99214 requires moderate level medical decision making: moderate number of "
            "diagnoses, moderate data review, or moderate risk. A brief follow-up with stable "
            "condition, no medication changes, and single straightforward problem supports "
            "99213 (low complexity) at most."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Visit documentation shows single stable problem with straightforward management",
            "No medication changes, no new workup ordered — low complexity data review",
            "Brief 15-minute encounter with focused exam — consistent with low MDM",
            "99214 requires moderate MDM — visit supports 99213 (low MDM) at most",
            "Deny 99214 — coding error, recommend rebilling as 99213",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-011",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="27447",
            icd10_codes=["M17.11", "M17.12"],
            facility_type=FacilityType.INPATIENT,
        ),
        clinical_notes=(
            "67yo F with bilateral knee OA, Kellgren-Lawrence grade 4 bilateral. "
            "Failed 6 months of PT (3x/week x 12 weeks then 2x/week x 12 weeks), "
            "NSAIDs (meloxicam 15mg daily), 2 corticosteroid injections each knee, "
            "and viscosupplementation (Synvisc-One bilateral). Cannot walk >1 block. "
            "Uses walker for ambulation. WOMAC score 72/96. Requesting total knee arthroplasty."
        ),
        policy_excerpt=(
            "Total knee arthroplasty requires: KL grade >= 3, minimum 3 months documented "
            "conservative treatment failure (PT, medications, injections), functional impairment "
            "documented with validated outcome measure (WOMAC, KOOS), and radiographic evidence "
            "of bone-on-bone or near bone-on-bone changes."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Radiographic severity meets criteria: KL grade 4 bilateral (exceeds minimum KL >= 3)",
            "Extensive conservative treatment failure: 6 months PT, NSAIDs, cortisone x2, viscosupplementation",
            "Validated outcome measure documented: WOMAC 72/96 indicating severe impairment",
            "Functional impairment severe: cannot walk >1 block, requires walker",
            "All criteria met — approve total knee arthroplasty",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-012",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="72148",
            icd10_codes=["M54.5"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MISSING_INFO,
        ),
        clinical_notes=(
            "41yo F with chronic LBP. Notes indicate 'has tried various treatments.' "
            "Requesting lumbar MRI. No specific treatment dates, durations, or outcomes "
            "documented. Physical exam section is blank. No functional assessment provided."
        ),
        policy_excerpt=(
            "Lumbar MRI requires documentation of specific conservative treatments tried "
            "with dates, duration, and outcomes. Physical examination findings must be "
            "included. Vague references to prior treatment are insufficient."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Clinical notes lack specific treatment details — 'various treatments' is insufficient",
            "No dates, durations, or outcomes for any conservative therapy documented",
            "Physical exam section is blank — objective findings not documented",
            "No functional assessment provided to support medical necessity",
            "Deny — missing information required for medical necessity determination",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-013",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="29881",
            icd10_codes=["M23.21", "S83.511A"],
            facility_type=FacilityType.ASC,
        ),
        clinical_notes=(
            "42yo M amateur soccer player with acute left knee injury 6 weeks ago. "
            "MRI shows medial meniscus tear with mechanical symptoms: locking, catching, "
            "and giving way. Failed 6 weeks of conservative management including PT 2x/week, "
            "ice, NSAIDs, and activity modification. Unable to return to daily activities. "
            "Requesting knee arthroscopy with meniscectomy."
        ),
        policy_excerpt=(
            "Knee arthroscopy with meniscectomy is covered for MRI-confirmed meniscal tear "
            "with mechanical symptoms and failure of conservative treatment >= 6 weeks. "
            "Must document specific mechanical symptoms (locking, catching, or giving way)."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "MRI-confirmed medial meniscus tear — diagnostic imaging requirement met",
            "Mechanical symptoms documented: locking, catching, and giving way",
            "Conservative treatment failure: 6 weeks PT, NSAIDs, ice, activity modification",
            "Functional impairment: unable to return to daily activities",
            "Approve knee arthroscopy with meniscectomy",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-014",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICAID, cpt_code="90837",
            icd10_codes=["F33.1", "F41.1"],
            facility_type=FacilityType.TELEHEALTH,
            denial_reason=DenialReason.NOT_COVERED,
        ),
        clinical_notes=(
            "28yo M requesting continued 60-minute psychotherapy via telehealth. "
            "PHQ-9 improved from 22 to 8 over 6 months. GAD-7 improved from 16 to 5. "
            "Reports returning to full-time work, sleeping well, and good social functioning. "
            "Currently on escitalopram 10mg with good response."
        ),
        policy_excerpt=(
            "Continued psychotherapy sessions require ongoing clinical need demonstrated by "
            "PHQ-9 >= 10 or GAD-7 >= 7 or documented functional impairment. When scores are "
            "in remission range (PHQ-9 < 10 AND GAD-7 < 7), step down to maintenance frequency "
            "(monthly) rather than weekly 60-minute sessions."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "PHQ-9 of 8 is below the >= 10 threshold, indicating remission from depression",
            "GAD-7 of 5 is below the >= 7 threshold, indicating remission from anxiety",
            "Patient reports functional improvement: working full-time, sleeping well, good social life",
            "Clinical indicators support step-down to maintenance frequency, not weekly 60-min sessions",
            "Deny weekly 60-minute sessions — recommend monthly maintenance per step-down protocol",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-015",
        task_config=PriorAuthTaskConfig(
            payer=Payer.HUMANA, cpt_code="70553",
            icd10_codes=["R51.9", "G43.909"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "35yo M with chronic migraines, unchanged pattern for 5 years. Typical aura followed "
            "by unilateral throbbing headache with nausea. No new neurological symptoms. Normal "
            "neuro exam. Requesting MRI brain with/without contrast for 'peace of mind.'"
        ),
        policy_excerpt=(
            "MRI brain is not covered for routine migraine evaluation with stable, unchanged "
            "headache pattern and normal neurological examination. Requires new or changed "
            "pattern, focal neurological deficit, or clinical red flags to establish medical necessity."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Headache pattern stable and unchanged for 5 years — no new features",
            "Neurological exam is normal — no focal deficits or red flags",
            "Indication is 'peace of mind' — not a covered medical necessity indication",
            "Policy specifically excludes stable migraine without new symptoms or deficits",
            "Deny — does not meet medical necessity for MRI brain",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    # ── HARD — OVERTURNED ON APPEAL ────────────────────────────────────────────
    BenchmarkCase(
        case_id="local-016",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="72148",
            icd10_codes=["M54.5", "M51.16"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "44yo construction worker with 4-week progressive low back pain with NEW onset "
            "left leg radiculopathy. Decreased left ankle reflex (1+ vs 2+ right). Positive "
            "straight leg raise at 30 degrees on left. Reports left foot drop noticed 2 days ago. "
            "Unable to work due to weakness. Initially denied for <6 weeks conservative treatment."
        ),
        policy_excerpt=(
            "Lumbar MRI requires 6 weeks conservative management. EXCEPTION: progressive "
            "neurological deficit (new weakness, reflex changes, bowel/bladder dysfunction) "
            "bypasses the conservative treatment requirement. Progressive deficit must be "
            "clearly documented with comparison to prior exam."
        ),
        ground_truth_outcome=Outcome.OVERTURNED_ON_APPEAL,
        ground_truth_reasoning=[
            "Initial denial was based on <6 weeks duration, but failed to identify red flag exceptions",
            "Progressive neurological deficit present: NEW left leg radiculopathy with foot drop",
            "Objective findings: decreased ankle reflex (asymmetric 1+ vs 2+), positive SLR at 30 degrees",
            "Foot drop is an acute progressive deficit requiring urgent imaging per policy exception",
            "Overturn on appeal — progressive neurological deficit meets red flag exception criteria",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-017",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="27447",
            icd10_codes=["M17.11"],
            facility_type=FacilityType.INPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "58yo F with right knee OA, KL grade 4. BMI 42. Failed PT x 6 months, NSAIDs, "
            "3 corticosteroid injections, and viscosupplementation over 14 months. WOMAC 78/96. "
            "Wheelchair-dependent for distances >50 ft. Endocrinologist documents metabolic syndrome "
            "(insulin resistance, PCOS history) prevents safe weight loss. Bariatric surgery "
            "contraindicated due to prior gastric perforation. Initially denied for BMI >40."
        ),
        policy_excerpt=(
            "TKA requires BMI < 40 due to increased complication risk. Exceptions considered "
            "with multi-specialty documentation that: (a) weight loss is medically contraindicated "
            "or has failed supervised attempt, (b) delay in surgery would cause irreversible harm, "
            "(c) benefits outweigh elevated surgical risk per orthopedic attestation."
        ),
        ground_truth_outcome=Outcome.OVERTURNED_ON_APPEAL,
        ground_truth_reasoning=[
            "Initial denial based on BMI 42 exceeding BMI < 40 threshold",
            "Endocrinology documents metabolic syndrome preventing safe weight loss",
            "Bariatric surgery contraindicated due to prior gastric perforation — weight loss options exhausted",
            "Severe functional impairment: WOMAC 78/96, wheelchair-dependent for >50 ft — delay risks irreversible deconditioning",
            "Multi-specialty exception criteria met — overturn on appeal",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-018",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="64483",
            icd10_codes=["M54.12", "M50.121"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.EXPERIMENTAL,
        ),
        clinical_notes=(
            "48yo M requesting epidural steroid injection for cervical radiculopathy C6-7. "
            "Prior 2 ESIs at C5-6 gave 6-week relief each. MRI shows multi-level disc disease "
            "C4-7. Surgeon notes C6-7 is a new symptomatic level distinct from prior C5-6 treatment. "
            "Requesting transforaminal approach at C6-7. Denied as multi-level ESI considered experimental."
        ),
        policy_excerpt=(
            "Cervical ESIs are covered for single-level radiculopathy with documented disc pathology. "
            "Multi-level ESIs in the same session are considered experimental. Sequential treatment "
            "of distinct levels in separate sessions may be considered with documentation of level-specific "
            "pathology and symptoms."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Request is for single-level C6-7 injection, NOT multi-level same-session",
            "However, prior C5-6 injections suggest multi-level disease progression",
            "Policy allows sequential single-level treatment but requires level-specific documentation",
            "While C6-7 pathology is documented on MRI, the pattern of multi-level treatment raises experimental concern",
            "Deny — the sequential multi-level pattern is appropriately flagged, though a stronger appeal with EMG nerve-specific evidence could succeed",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-019",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="29881",
            icd10_codes=["M23.21"],
            facility_type=FacilityType.ASC,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "58yo F with degenerative medial meniscus tear on MRI. Mild OA changes (KL grade 2). "
            "Reports knee pain for 8 months. Completed 12 weeks PT with partial improvement. "
            "No mechanical symptoms (no locking, catching, or giving way). Pain is primarily "
            "weight-bearing and activity-related. Requesting arthroscopic meniscectomy."
        ),
        policy_excerpt=(
            "Arthroscopic meniscectomy for degenerative meniscal tears requires documented "
            "mechanical symptoms (locking, catching, giving way). Weight-bearing pain alone "
            "in the setting of degenerative tear with underlying OA does not meet criteria. "
            "Evidence shows arthroscopy for degenerative tears without mechanical symptoms "
            "has no benefit over PT alone."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Degenerative meniscal tear (not traumatic) in setting of underlying OA",
            "No mechanical symptoms documented — pain is weight-bearing only",
            "PT completed with partial improvement — conservative management had some benefit",
            "Evidence-based policy: arthroscopy for degenerative tears without mechanical symptoms shows no benefit over PT",
            "Deny — does not meet mechanical symptom requirement for degenerative tear",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-020",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="77386",
            icd10_codes=["C34.11"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "71yo M with stage IIIA NSCLC. ECOG 3. Recent hospitalization for pneumonia. "
            "Significant weight loss (20 lbs in 2 months). FEV1 0.9L (35% predicted). "
            "Oncology requesting definitive IMRT 60 Gy/30 fractions. Pulmonology notes "
            "marginal respiratory reserve. Radiation oncology concerned about radiation "
            "pneumonitis risk given low FEV1."
        ),
        policy_excerpt=(
            "Definitive thoracic radiation therapy requires ECOG performance status 0-2 "
            "and adequate pulmonary function (FEV1 > 1.0L or > 40% predicted). ECOG 3-4 "
            "patients may be considered for palliative doses only. Exceptions require "
            "multi-disciplinary tumor board attestation of favorable risk-benefit ratio."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "ECOG performance status 3 exceeds the 0-2 requirement for definitive radiation",
            "FEV1 0.9L (35% predicted) below the >1.0L or >40% predicted threshold",
            "Recent pneumonia and significant weight loss indicate poor treatment tolerance",
            "Radiation oncology expresses concern about pneumonitis risk — unfavorable risk profile",
            "Deny definitive IMRT — patient may qualify for palliative dose regimen instead",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-021",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE, cpt_code="27447",
            icd10_codes=["M17.12"],
            facility_type=FacilityType.INPATIENT,
        ),
        clinical_notes=(
            "74yo M with left knee OA KL grade 4, bone-on-bone medial compartment. "
            "History of DVT 6 months ago, completed warfarin course. Cardiology clears for "
            "surgery with perioperative anticoagulation bridge plan. Failed 9 months conservative "
            "treatment including PT, NSAIDs, cortisone injections x3, and PRP. VAS pain 9/10 at rest."
        ),
        policy_excerpt=(
            "TKA covered with KL >= 3, conservative failure >= 3 months, and surgical clearance. "
            "History of VTE within 12 months requires hematology or cardiology clearance with "
            "documented perioperative anticoagulation plan."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "KL grade 4 with bone-on-bone changes exceeds radiographic threshold",
            "Extensive conservative failure over 9 months: PT, NSAIDs, cortisone x3, PRP",
            "VTE history within 12 months — cardiology clearance obtained with bridge plan",
            "Perioperative anticoagulation plan satisfies VTE-related safety requirement",
            "Approve TKA with perioperative anticoagulation protocol",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-022",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="97110",
            icd10_codes=["S83.511A", "M23.21"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.FREQUENCY_LIMIT,
        ),
        clinical_notes=(
            "36yo F requesting additional 12 PT visits after completing initial 24-visit "
            "authorization for ACL reconstruction rehab. Currently at 16 weeks post-op. "
            "ROM improved from 30 to 110 degrees flexion. Quad strength 3+/5 (was 2/5). "
            "Still unable to descend stairs reciprocally or jog. Surgeon requests continued PT "
            "for return-to-sport protocol."
        ),
        policy_excerpt=(
            "Post-ACL reconstruction PT authorized for up to 24 visits. Extension requires "
            "documented objective progress AND residual functional deficit preventing return "
            "to prior activity level. Must show continued measurable gains between evaluations. "
            "Maximum lifetime benefit: 60 visits per surgical event."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Significant objective progress documented: ROM 30° → 110°, quad strength 2/5 → 3+/5",
            "Residual functional deficit present: cannot descend stairs reciprocally or jog",
            "16 weeks post-ACL reconstruction is within expected rehabilitation timeline",
            "Within lifetime benefit limit: 24 + 12 = 36 visits < 60 maximum",
            "Approve extension — documented progress with residual deficit at appropriate rehab phase",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-023",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="99214",
            icd10_codes=["E11.65", "I10", "E78.5"],
            facility_type=FacilityType.TELEHEALTH,
        ),
        clinical_notes=(
            "63yo F with poorly controlled T2DM (A1c 9.2%), HTN, and hyperlipidemia via "
            "telehealth. Adjusting metformin to 1000mg BID, adding semaglutide, changing "
            "lisinopril to losartan due to cough, and increasing atorvastatin. Reviewed "
            "home glucose logs, recent labs, and medication interactions. 3 chronic conditions "
            "addressed with prescription drug management requiring moderate risk assessment."
        ),
        policy_excerpt=(
            "99214 telehealth visits require moderate MDM: moderate number of problems (2+ chronic), "
            "moderate data review (review of labs, external records, or drug interaction checking), "
            "OR moderate risk (prescription drug management). Telehealth modifier -95 applies. "
            "Audio-only visits require documented reason for no video."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Three chronic conditions addressed (T2DM, HTN, hyperlipidemia) — moderate number of problems",
            "Multiple medication changes with interaction review — moderate data and risk",
            "Prescription drug management with new medications — moderate risk level",
            "Documentation supports moderate MDM on all three elements",
            "Approve 99214 with telehealth modifier",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-024",
        task_config=PriorAuthTaskConfig(
            payer=Payer.HUMANA, cpt_code="43239",
            icd10_codes=["K21.0"],
            facility_type=FacilityType.ASC,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "38yo F with GERD for 3 months. Currently on omeprazole 20mg daily with moderate "
            "improvement. No alarm symptoms: no dysphagia, no weight loss, no GI bleeding, "
            "no anemia. No family history of GI malignancy. Age <45 without alarm symptoms. "
            "Requesting EGD for persistent symptoms."
        ),
        policy_excerpt=(
            "EGD for GERD is covered after failed PPI trial >8 weeks at adequate dose (>=40mg/day), "
            "OR presence of alarm symptoms (dysphagia, weight loss, GI bleeding, anemia), "
            "OR age >50 with new-onset GERD, OR family history of upper GI malignancy. "
            "PPI dose 20mg daily is subtherapeutic for treatment failure determination."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "PPI trial is subtherapeutic: omeprazole 20mg daily, not the 40mg minimum for treatment failure",
            "Duration is only 3 months with moderate improvement — not true treatment failure",
            "No alarm symptoms present (no dysphagia, weight loss, bleeding, anemia)",
            "Age 38, below the >50 new-onset threshold; no family history of GI malignancy",
            "Deny — recommend optimizing PPI to adequate dose before pursuing endoscopy",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-025",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="27447",
            icd10_codes=["M17.11"],
            facility_type=FacilityType.INPATIENT,
            denial_reason=DenialReason.MISSING_INFO,
        ),
        clinical_notes=(
            "60yo M requesting TKA for right knee pain. X-ray shows OA changes. "
            "Notes mention 'some PT in the past' and 'tried medications.' "
            "No KL grade documented. No WOMAC or functional outcome measure. "
            "No specific PT dates, duration, frequency, or response documented."
        ),
        policy_excerpt=(
            "TKA authorization requires: documented KL grade >= 3 on imaging, specific "
            "conservative treatment history with dates and outcomes (PT frequency/duration, "
            "medication names/doses/duration, injection dates and response), and validated "
            "functional outcome measure (WOMAC, KOOS, or KSS)."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "No KL grade documented — radiographic severity cannot be assessed",
            "PT history is vague: 'some PT in the past' lacks dates, frequency, and duration",
            "Medication history non-specific: 'tried medications' lacks drug names, doses, duration",
            "No validated functional outcome measure (WOMAC, KOOS, or KSS) documented",
            "Deny — missing multiple required documentation elements for TKA authorization",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-026",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICAID, cpt_code="64483",
            icd10_codes=["M54.5", "M54.16"],
            facility_type=FacilityType.ASC,
        ),
        clinical_notes=(
            "50yo M with chronic lumbar radiculopathy L5-S1. MRI confirms L5-S1 disc herniation "
            "with left S1 nerve root compression. Failed 8 weeks of PT (2x/week) and oral "
            "medications (gabapentin 600mg TID, meloxicam 15mg daily). EMG confirms left S1 "
            "radiculopathy. VAS pain 8/10 with radiation to left posterior calf. "
            "Requesting first lumbar epidural steroid injection."
        ),
        policy_excerpt=(
            "Lumbar ESI is covered for radiculopathy with correlating MRI findings and failure "
            "of conservative management >= 6 weeks. EMG confirmation preferred but not required. "
            "First injection in a series does not require prior injection failure."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "MRI-confirmed L5-S1 disc herniation with nerve root compression — imaging criteria met",
            "Conservative treatment failure documented: 8 weeks PT and medications",
            "EMG confirms radiculopathy, providing objective electrodiagnostic correlation",
            "First ESI request — no prior injection frequency concerns",
            "Approve first lumbar epidural steroid injection",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-027",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="90837",
            icd10_codes=["F33.1", "F43.10"],
            facility_type=FacilityType.TELEHEALTH,
            denial_reason=DenialReason.OUT_OF_NETWORK,
        ),
        clinical_notes=(
            "31yo M with MDD and PTSD requesting 60-minute psychotherapy with out-of-network "
            "EMDR-certified therapist. PHQ-9=21 (severe). PCL-5=62 (probable PTSD). Currently "
            "on venlafaxine 225mg. Reports 3 in-network therapists contacted, all have >8 week "
            "waitlists. Has been seen by this OON therapist for 4 sessions with documented improvement."
        ),
        policy_excerpt=(
            "Out-of-network providers require network adequacy exception. Must document: "
            "(1) minimum 3 in-network providers contacted with unavailability demonstrated, "
            "(2) clinical urgency (PHQ-9 >= 20, active SI, or acute trauma), "
            "(3) specialty not available in-network (e.g., EMDR certification). "
            "OON exceptions authorized for 12 sessions, then re-evaluate."
        ),
        ground_truth_outcome=Outcome.OVERTURNED_ON_APPEAL,
        ground_truth_reasoning=[
            "3 in-network providers contacted, all with >8 week waitlists — network inadequacy documented",
            "Clinical urgency: PHQ-9 of 21 (severe, >=20 threshold) with comorbid PTSD",
            "Specialty need: EMDR certification for PTSD treatment not readily available in-network",
            "Established therapeutic relationship with documented improvement — disruption contraindicated",
            "Overturn — network adequacy exception criteria met on all three elements",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-028",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="99213",
            icd10_codes=["J06.9", "J02.9"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "22yo M presents with sore throat and nasal congestion x 4 days. Temp 99.1F. "
            "Pharynx mildly erythematous without exudate. No cervical lymphadenopathy. "
            "Rapid strep negative. Assessment: acute viral URI. Plan: supportive care, "
            "OTC decongestants, return if worsening."
        ),
        policy_excerpt=(
            "Office visits for acute complaints are covered at the appropriate E/M level. "
            "99213 requires low-complexity MDM with 2+ self-limited problems or 1 acute "
            "uncomplicated illness, limited data review, and low risk management."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Acute uncomplicated illness (viral URI) appropriately coded",
            "Low MDM: single self-limited problem with OTC management",
            "Limited data review: rapid strep performed in office",
            "Low risk: no prescription medications, supportive care only",
            "Approve 99213 — appropriate level for acute uncomplicated office visit",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-029",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="20610",
            icd10_codes=["M19.011"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.DUPLICATE,
        ),
        clinical_notes=(
            "65yo F requesting right shoulder corticosteroid injection. Records show same "
            "injection was administered 5 weeks ago at an in-network urgent care. Patient "
            "reports injection 'didn't work' but records from urgent care show injection "
            "was administered with standard technique and dose (40mg triamcinolone)."
        ),
        policy_excerpt=(
            "Repeat intra-articular corticosteroid injection to the same joint requires "
            "minimum 12-week interval from prior injection. Shorter intervals increase risk "
            "of cartilage damage and are not covered. Applies regardless of treating provider."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Prior injection to same joint was 5 weeks ago — below 12-week minimum interval",
            "Policy applies regardless of treating provider — urgent care injection counts",
            "Repeat injection at <12 weeks carries increased cartilage damage risk",
            "5-week interval insufficient for clinical reassessment of prior injection efficacy",
            "Deny — duplicate injection within minimum interval period",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-030",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="72148",
            icd10_codes=["M54.5", "M51.16", "G83.4"],
            facility_type=FacilityType.ED,
        ),
        clinical_notes=(
            "52yo F presents to ED with acute onset urinary retention and bilateral leg "
            "weakness starting 6 hours ago. Saddle anesthesia on exam. Known lumbar disc "
            "disease. Unable to void, post-void residual >500mL by bladder scan. "
            "Bilateral lower extremity strength 3/5. Requesting emergent lumbar MRI."
        ),
        policy_excerpt=(
            "Emergent lumbar MRI is covered in ED setting for suspected cauda equina syndrome: "
            "new bladder dysfunction, saddle anesthesia, bilateral lower extremity weakness, "
            "or rapidly progressive neurological deficit. No prior conservative treatment "
            "requirement for emergent indications."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Classic cauda equina syndrome presentation: urinary retention, saddle anesthesia, bilateral weakness",
            "Acute onset (<6 hours) constitutes emergent indication",
            "Known disc disease provides anatomical substrate for cauda equina compression",
            "Emergent indication bypasses all conservative treatment requirements per policy",
            "Approve emergent lumbar MRI — suspected cauda equina syndrome requires immediate imaging",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-031",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="29881",
            icd10_codes=["M23.21", "M17.11"],
            facility_type=FacilityType.ASC,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "63yo M with right knee pain and MRI showing complex degenerative medial meniscus "
            "tear. Also has moderate OA (KL grade 3). Arthroscopy requested. No mechanical "
            "symptoms documented. Pain is primarily aching with prolonged standing. "
            "Patient reports knee 'feels loose' but exam shows no instability or locking."
        ),
        policy_excerpt=(
            "Arthroscopic meniscectomy in patients age >60 with concomitant OA (KL >= 3) "
            "requires documented true mechanical symptoms on physical exam (locking, catching). "
            "Subjective complaints of instability without exam findings are insufficient. "
            "AAOS guidelines recommend against arthroscopy for degenerative tears with OA."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Age >60 with concomitant OA KL 3 triggers stricter criteria",
            "No true mechanical symptoms on exam — 'feels loose' is subjective without clinical confirmation",
            "Pain pattern (aching with standing) is consistent with OA, not mechanical derangement",
            "AAOS guidelines recommend against arthroscopy for degenerative tears with significant OA",
            "Deny — does not meet criteria for arthroscopy in setting of age >60 with OA",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="local-032",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE, cpt_code="77386",
            icd10_codes=["C50.911"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "56yo F with stage IIA left breast cancer (T2N0M0), ER+/PR+/HER2-. "
            "S/p lumpectomy with negative margins. Pathology: 2.3cm IDC, grade 2, "
            "Oncotype DX RS 28. Tumor board recommends adjuvant whole breast IMRT "
            "followed by boost. ECOG 0. No significant comorbidities."
        ),
        policy_excerpt=(
            "IMRT for breast cancer is covered per NCCN guidelines for post-lumpectomy "
            "adjuvant radiation. Must include surgical pathology, staging, and treatment "
            "plan from radiation oncology. Standard whole breast radiation with boost "
            "is standard of care for early-stage breast cancer after BCS."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Post-lumpectomy adjuvant radiation is standard of care per NCCN guidelines",
            "Complete surgical pathology documented: IDC, 2.3cm, grade 2, margins negative",
            "Appropriate staging: T2N0M0, stage IIA with molecular profiling (Oncotype DX)",
            "Tumor board recommendation documented; ECOG 0 indicates good treatment tolerance",
            "Approve IMRT with boost — standard adjuvant therapy for early-stage breast cancer after BCS",
        ],
        difficulty=Difficulty.EASY,
    ),
    BenchmarkCase(
        case_id="local-033",
        task_config=PriorAuthTaskConfig(
            payer=Payer.HUMANA, cpt_code="99214",
            icd10_codes=["E11.65", "E11.40"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "58yo M with T2DM complicated by diabetic nephropathy (eGFR 42). A1c 8.9% on "
            "metformin 1000mg BID + glipizide 10mg BID. Adding empagliflozin for cardiorenal "
            "benefit per ADA guidelines. Discussed hypoglycemia risk with glipizide, reducing "
            "dose. Ordered comprehensive metabolic panel and urine albumin-to-creatinine ratio. "
            "Referral to nephrology. 35-minute visit."
        ),
        policy_excerpt=(
            "99214 requires moderate MDM. Multiple chronic conditions with medication management "
            "qualifies as moderate number/complexity of problems. Ordering new labs and making "
            "specialist referral constitutes moderate data review. Prescription drug management "
            "represents moderate treatment risk."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Multiple chronic conditions managed: T2DM with nephropathy — moderate problem complexity",
            "Medication changes: adding empagliflozin, reducing glipizide — prescription drug management",
            "Moderate data review: ordering labs (CMP, UACR) and specialist referral",
            "Moderate risk: new prescription with drug interaction consideration (empagliflozin + glipizide)",
            "Approve 99214 — moderate MDM criteria met on all elements",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-034",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="70553",
            icd10_codes=["G43.909", "R55"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "45yo M with known migraines now presenting with 2 episodes of syncope in past month. "
            "Headache character has changed: now bilateral, constant, worse in morning. "
            "New finding: bilateral papilledema on fundoscopic exam. Cardiology workup for "
            "syncope negative (ECG, echo, Holter normal). Requesting MRI brain with/without contrast."
        ),
        policy_excerpt=(
            "MRI brain with/without contrast is covered for new or changed headache pattern "
            "with red flag features: papilledema, new syncope, positional component (worse in "
            "morning suggests elevated ICP), or change from unilateral to bilateral pattern. "
            "Urgent/emergent processing for suspected elevated intracranial pressure."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Changed headache pattern: previously unilateral migraine, now bilateral and constant",
            "Red flag: papilledema on exam suggesting elevated intracranial pressure",
            "Red flag: new syncope episodes with negative cardiac workup raises intracranial concern",
            "Positional component (worse in morning) is classic ICP elevation pattern",
            "Approve with urgency — multiple red flags for elevated ICP requiring urgent brain MRI",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="local-035",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="97110",
            icd10_codes=["M75.11"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.FREQUENCY_LIMIT,
        ),
        clinical_notes=(
            "48yo F with right rotator cuff tendinopathy. Completed 24 PT visits over 12 weeks. "
            "ROM improved: forward flexion 90° to 140° (normal 180°). Strength 3/5 abduction "
            "(was 2+/5). Still unable to reach overhead or lift >5 lbs. Requesting 12 additional visits. "
            "However, progress has plateaued over last 4 visits with no measurable change."
        ),
        policy_excerpt=(
            "PT extension beyond initial authorization requires documented continued objective "
            "progress between evaluations. When objective measures plateau (no measurable improvement "
            "over 4+ consecutive visits), further PT is not considered medically necessary. "
            "Recommend transition to home exercise program."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Initial progress well-documented: ROM and strength improvements over 24 visits",
            "However, progress has plateaued over last 4 visits — no measurable change",
            "Policy requires continued objective progress for extension — plateau is disqualifying",
            "Residual deficits remain but are not improving with current PT approach",
            "Deny — plateau documented, recommend transition to HEP or alternative intervention",
        ],
        difficulty=Difficulty.HARD,
    ),
    # ── ADVERSARIAL — designed to trick zero-shot agents ───────────────────────
    BenchmarkCase(
        case_id="adv-001",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="72148",
            icd10_codes=["M54.5", "M51.16"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "51yo M with chronic LBP and left L5 radiculopathy. MRI 14 months ago showed "
            "L4-5 disc protrusion. Completed 12 weeks PT, tried gabapentin, NSAIDs, and one ESI "
            "with 4-month relief. Pain has returned to baseline 7/10. Neurological exam stable "
            "from prior visit — no new deficits. Requesting repeat lumbar MRI."
        ),
        policy_excerpt=(
            "Repeat lumbar MRI within 24 months of prior MRI requires documentation of NEW "
            "or CHANGED neurological findings, new mechanism of injury, or failure to respond "
            "to treatment directed by prior imaging. Stable symptoms with stable exam do not "
            "meet criteria for repeat imaging. Clinical deterioration alone without objective "
            "change is insufficient."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Prior MRI was only 14 months ago — within the 24-month repeat imaging window",
            "Neurological exam is explicitly stable with no new deficits — no objective change",
            "Pain returning to baseline is clinical deterioration but WITHOUT objective neurological change",
            "Policy specifically states clinical deterioration alone without objective change is insufficient",
            "Deny — stable exam despite symptom recurrence does not meet repeat imaging criteria",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-002",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="27447",
            icd10_codes=["M17.11"],
            facility_type=FacilityType.INPATIENT,
        ),
        clinical_notes=(
            "69yo M with right knee OA KL grade 4. Failed 4 months of PT, NSAIDs, 2 cortisone "
            "injections. WOMAC 65/96. Can walk 2 blocks with cane. A1c is 9.8% (uncontrolled diabetes). "
            "Endocrinology note states diabetes optimization will take 3-6 months. Orthopedic surgeon "
            "states patient is 'surgical candidate' and requests authorization."
        ),
        policy_excerpt=(
            "TKA requires KL >= 3, conservative treatment failure >= 3 months, and documented "
            "functional impairment. Perioperative risk factors must be optimized prior to elective "
            "surgery. A1c > 8.0% is associated with significantly increased surgical site infection "
            "risk and must be < 8.0% prior to elective joint replacement authorization."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "KL grade 4, conservative failure 4 months, WOMAC 65/96 — orthopedic criteria appear met",
            "However, A1c 9.8% significantly exceeds the < 8.0% perioperative threshold",
            "Policy requires A1c < 8.0% for elective TKA — this is a hard requirement, not a suggestion",
            "Endocrinology confirms optimization will take 3-6 months — patient is not currently optimized",
            "Deny — perioperative A1c requirement not met despite meeting orthopedic criteria",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-003",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="90837",
            icd10_codes=["F33.1", "F10.20"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.NOT_COVERED,
        ),
        clinical_notes=(
            "45yo M with recurrent MDD (PHQ-9=19) and active alcohol use disorder (moderate, "
            "drinking 6-8 drinks daily). Psychiatrist prescribing naltrexone and sertraline. "
            "Requesting 60-minute individual psychotherapy. Patient is currently drinking and "
            "has not entered a formal substance abuse treatment program. Last drink was yesterday."
        ),
        policy_excerpt=(
            "Individual psychotherapy for MDD is covered with documented severity (PHQ-9 >= 15) "
            "and functional impairment. For patients with co-occurring active substance use "
            "disorder, psychotherapy for MDD is covered ONLY when patient is concurrently "
            "enrolled in or has completed a substance abuse treatment program. Active untreated "
            "substance use disorder without concurrent addiction treatment is excluded."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "PHQ-9 of 19 meets severity threshold for MDD psychotherapy (>= 15)",
            "However, patient has active moderate alcohol use disorder — drinking 6-8 drinks daily",
            "Policy requires concurrent enrollment in substance abuse treatment for co-occurring AUD",
            "Patient is NOT in a substance abuse treatment program — only receiving naltrexone from psychiatrist",
            "Deny — active untreated AUD without formal substance abuse treatment program enrollment",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-004",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="64483",
            icd10_codes=["M54.5", "M54.16"],
            facility_type=FacilityType.ASC,
        ),
        clinical_notes=(
            "47yo F with lumbar radiculopathy L4-5. MRI shows moderate L4-5 disc herniation "
            "with left L5 nerve root impingement. Completed 8 weeks PT, failed gabapentin (side effects) "
            "and meloxicam. EMG confirms L5 radiculopathy. VAS 7/10. This would be her first ESI. "
            "However, she is currently 14 weeks pregnant. OB/GYN states pregnancy is uncomplicated."
        ),
        policy_excerpt=(
            "Lumbar ESI is covered for radiculopathy with correlating MRI and failed conservative "
            "treatment >= 6 weeks. Fluoroscopic guidance is required for ESI procedures. "
            "Fluoroscopically-guided procedures involving ionizing radiation are contraindicated "
            "during pregnancy regardless of trimester. Non-fluoroscopic (ultrasound-guided) ESI "
            "is considered investigational and not covered."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Clinical criteria for ESI are fully met: MRI-confirmed herniation, EMG confirmation, 8 weeks conservative failure",
            "However, patient is 14 weeks pregnant — a critical contraindication factor",
            "Policy requires fluoroscopic guidance for ESI — fluoroscopy uses ionizing radiation",
            "Fluoroscopy is contraindicated in pregnancy regardless of trimester",
            "Alternative ultrasound-guided approach is considered investigational and not covered",
            "Deny — pregnancy contraindication for required fluoroscopic guidance, no covered alternative",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-005",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="99214",
            icd10_codes=["Z00.00"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.CODING_ERROR,
        ),
        clinical_notes=(
            "35yo F presents for annual wellness visit. Reviews immunizations, cancer screening "
            "schedule, discusses diet and exercise. During visit, mentions occasional heartburn. "
            "Provider briefly discusses OTC antacids, no workup ordered, no prescription given. "
            "Provider bills 99214 with Z00.00 (routine exam) as primary diagnosis."
        ),
        policy_excerpt=(
            "Annual wellness visits should be coded as preventive medicine E/M codes (99385-99397) "
            "with appropriate preventive diagnosis (Z00.00). Billing a problem-oriented E/M code "
            "(99201-99215) with a preventive diagnosis is a coding mismatch. If a separately "
            "identifiable problem is addressed during a preventive visit, modifier 25 must be "
            "appended and a problem-oriented diagnosis must be linked to the problem-oriented code."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Visit is clearly a preventive/wellness visit — immunizations, screening, lifestyle counseling",
            "Provider coded 99214 (problem-oriented E/M) with Z00.00 (preventive diagnosis) — coding mismatch",
            "Brief mention of heartburn with OTC recommendation is not a separately identifiable problem-oriented service",
            "Correct coding would be 99395 or 99396 (established patient preventive) with Z00.00",
            "Deny 99214 — coding error, should be preventive medicine code, not problem-oriented E/M",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-006",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="43239",
            icd10_codes=["K21.0", "F50.9"],
            facility_type=FacilityType.ASC,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "24yo F with GERD symptoms for 3 months and bulimia nervosa (active, purging type). "
            "Reports daily retrosternal burning and occasional dysphagia. Currently in outpatient "
            "eating disorder program. Purging frequency reduced from daily to 2-3x/week. "
            "GI requests EGD to evaluate esophageal damage. PPI trial: omeprazole 40mg x 6 weeks "
            "with partial symptom improvement."
        ),
        policy_excerpt=(
            "EGD for GERD requires failed PPI trial >= 8 weeks at adequate dose OR alarm symptoms. "
            "In patients with active eating disorders involving purging behavior, GERD symptoms "
            "may be secondary to purging rather than primary GERD. EGD for purging-related "
            "symptoms requires eating disorder to be in sustained remission (>= 3 months abstinence "
            "from purging) before EGD is considered medically necessary for GERD evaluation."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "PPI trial is only 6 weeks — below the 8-week minimum for treatment failure",
            "Active bulimia with ongoing purging (2-3x/week) — NOT in sustained remission",
            "Policy requires >= 3 months purging abstinence before EGD for GERD in active eating disorders",
            "GERD symptoms may be secondary to active purging behavior, not primary GERD",
            "Deny — active eating disorder with purging not in remission, and PPI trial inadequate duration",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-007",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="29881",
            icd10_codes=["M23.21", "S83.511A"],
            facility_type=FacilityType.ASC,
        ),
        clinical_notes=(
            "19yo M college soccer player with acute right knee injury 2 weeks ago during game. "
            "MRI shows bucket-handle medial meniscus tear with displaced fragment causing locked knee. "
            "Unable to fully extend knee. Mechanical block on exam at 20 degrees. Effusion present. "
            "Requesting urgent arthroscopic meniscectomy. Has NOT completed 6 weeks of conservative treatment."
        ),
        policy_excerpt=(
            "Knee arthroscopy with meniscectomy requires MRI-confirmed tear with mechanical "
            "symptoms AND failure of conservative treatment >= 6 weeks. EXCEPTION: displaced "
            "bucket-handle tears causing true mechanical block (inability to achieve full extension "
            "on exam) are considered urgent and exempt from conservative treatment requirement."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "MRI confirms bucket-handle medial meniscus tear with displaced fragment",
            "True mechanical block documented: unable to fully extend, blocked at 20 degrees",
            "This is NOT a standard degenerative tear — it's an acute displaced bucket-handle tear",
            "Policy has explicit exception: displaced bucket-handle tears with mechanical block are urgent",
            "Approve — urgent exception for displaced bucket-handle tear with true mechanical block, conservative treatment requirement waived",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-008",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICARE, cpt_code="97110",
            icd10_codes=["G20", "R26.89"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "71yo M with Parkinson's disease (Hoehn & Yahr stage 3) and progressive gait instability. "
            "3 falls in past month. Berg Balance Score 32/56 (fall risk). Completed 12 PT visits — "
            "Berg improved from 28 to 32. Therapist documents continued slow progress. Requesting "
            "12 additional visits. Patient will NEVER return to baseline — Parkinson's is progressive. "
            "Goal is fall prevention and maintaining current function."
        ),
        policy_excerpt=(
            "Therapeutic exercises require skilled therapy services and expectation of meaningful "
            "improvement. Maintenance therapy (maintaining current function without expected improvement) "
            "is covered under Medicare ONLY when skilled services are required to maintain function "
            "or prevent decline, AND a maintenance program requires the skills of a therapist "
            "(not achievable through a caregiver-directed home program). The Jimmo v. Sebelius "
            "settlement clarifies that improvement is NOT required — maintenance of function or "
            "prevention of decline IS a covered goal when skilled services are needed."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Patient has progressive Parkinson's — improvement to normal is not expected or required",
            "However, Berg Balance Score DID improve (28 to 32) showing skilled therapy is effective",
            "Jimmo v. Sebelius: Medicare cannot deny PT solely because patient won't improve to baseline",
            "Maintenance of function and fall prevention ARE covered goals requiring skilled services",
            "3 falls/month with fall-risk Berg score demonstrates continued skilled therapy need",
            "Approve — maintenance therapy for progressive neurological condition with documented skilled need",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-009",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="70553",
            icd10_codes=["R51.9", "G43.909", "Z87.39"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "42yo F with chronic migraines requesting MRI brain. Headache pattern unchanged for 3 years. "
            "Normal neurological exam. She had a normal brain MRI 18 months ago. Her concern is a "
            "coworker was recently diagnosed with a brain tumor. She also has personal history of "
            "breast cancer (in remission 5 years, BRCA1 negative). No new symptoms. Oncologist "
            "states no indication for surveillance brain imaging."
        ),
        policy_excerpt=(
            "MRI brain requires new or changed headache pattern, focal neurological deficit, or "
            "clinical red flags. History of non-CNS malignancy does not constitute indication for "
            "brain MRI surveillance unless specific cancer type with known high CNS metastasis risk "
            "(melanoma, lung, renal). Breast cancer CNS surveillance is indicated only for HER2+ or "
            "triple-negative subtypes with active disease. Anxiety about a coworker's diagnosis is "
            "not a clinical indication."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Headache pattern unchanged for 3 years — no new or changed features",
            "Normal neurological exam — no focal deficits or red flags",
            "Prior normal MRI 18 months ago — no interval change in symptoms to justify repeat",
            "Breast cancer in 5-year remission, BRCA1 negative — oncologist confirms no brain surveillance indication",
            "Coworker's diagnosis creating anxiety is NOT a clinical indication for imaging",
            "Deny — no medical necessity, stable migraines with recent normal MRI, non-clinical motivation",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-010",
        task_config=PriorAuthTaskConfig(
            payer=Payer.HUMANA, cpt_code="27447",
            icd10_codes=["M17.11"],
            facility_type=FacilityType.INPATIENT,
            denial_reason=DenialReason.MEDICAL_NECESSITY,
        ),
        clinical_notes=(
            "72yo F with severe right knee OA, KL grade 4. Failed extensive conservative treatment "
            "over 18 months: PT, NSAIDs, 3 cortisone injections, viscosupplementation, PRP. "
            "WOMAC 82/96. Uses wheelchair for all mobility. However, she also has severe aortic "
            "stenosis (valve area 0.7 cm2, mean gradient 48 mmHg) and is scheduled for TAVR in "
            "6 weeks. Cardiologist states she is NOT cleared for elective non-cardiac surgery "
            "until post-TAVR recovery (minimum 3 months after TAVR)."
        ),
        policy_excerpt=(
            "TKA requires KL >= 3, conservative failure, and functional impairment (all met here). "
            "Elective surgery requires medical clearance from all relevant specialists. "
            "Severe aortic stenosis (valve area < 1.0 cm2) is an absolute contraindication for "
            "elective non-cardiac surgery until treated. Proceeding without cardiac clearance "
            "carries unacceptable perioperative mortality risk (estimated 5-10%)."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "All orthopedic criteria for TKA are met: KL4, 18 months conservative failure, WOMAC 82/96",
            "However, severe aortic stenosis (AVA 0.7 cm2) is an absolute surgical contraindication",
            "Cardiologist explicitly denies surgical clearance until post-TAVR recovery",
            "TAVR scheduled in 6 weeks with minimum 3-month recovery — cannot authorize TKA now",
            "Deny — absolute cardiac contraindication for elective surgery, authorize after TAVR recovery and clearance",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-011",
        task_config=PriorAuthTaskConfig(
            payer=Payer.AETNA, cpt_code="99213",
            icd10_codes=["Z76.89"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.NOT_COVERED,
        ),
        clinical_notes=(
            "28yo M presents requesting office visit for completion of employment physical form. "
            "No active medical complaints. No chronic conditions. Needs blood pressure check, "
            "vision screening, and physician signature on employer form. Vital signs normal. "
            "Brief exam performed. Coded as 99213 with Z76.89 (encounter for administrative purpose)."
        ),
        policy_excerpt=(
            "Office visits coded with administrative purpose diagnosis codes (Z02.x, Z76.89) "
            "for employment physicals, insurance exams, or disability forms are not covered "
            "medical benefits. These are considered administrative services and are the "
            "responsibility of the requesting employer or entity. Even when a brief exam "
            "is performed, the primary purpose determines coverage."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Primary purpose of visit is employment physical form completion — administrative",
            "Z76.89 (administrative purpose) is explicitly excluded from medical benefits coverage",
            "No active medical complaint or condition being addressed",
            "Brief exam was performed but purpose is employer requirement, not medical necessity",
            "Deny — administrative/employment physical is not a covered medical benefit",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-012",
        task_config=PriorAuthTaskConfig(
            payer=Payer.UNITEDHEALTHCARE, cpt_code="72148",
            icd10_codes=["M54.5", "M51.16", "R29.6"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "56yo F with 5-week history of progressive bilateral leg weakness and LBP. "
            "Started as LBP, now has bilateral hamstring weakness (4/5) and difficulty rising "
            "from chair. Reports new urinary urgency but no incontinence yet. "
            "No prior imaging. Gait is wide-based and unsteady. Deep tendon reflexes brisk "
            "bilaterally in lower extremities. Has NOT completed 6 weeks of conservative treatment."
        ),
        policy_excerpt=(
            "Lumbar MRI requires 6 weeks conservative management for routine LBP. "
            "URGENT exception: suspected cauda equina syndrome or progressive myelopathy — "
            "bilateral weakness, gait disturbance, bladder symptoms, or upper motor neuron signs "
            "(hyperreflexia, clonus) bypass conservative treatment requirement. "
            "Progressive bilateral symptoms with upper motor neuron signs may indicate "
            "cord compression requiring emergent imaging."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Only 5 weeks of symptoms — below 6-week threshold for routine LBP imaging",
            "However, this is NOT routine LBP — progressive bilateral weakness with gait disturbance",
            "Upper motor neuron signs present: brisk deep tendon reflexes bilaterally — suggests cord compression",
            "New urinary urgency with bilateral weakness raises concern for myelopathy/cord compression",
            "Progressive bilateral symptoms with UMN signs meet urgent exception criteria",
            "Approve — urgent exception for suspected myelopathy/cord compression, conservative treatment requirement waived",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-013",
        task_config=PriorAuthTaskConfig(
            payer=Payer.BCBS, cpt_code="97110",
            icd10_codes=["M54.5", "Z96.611"],
            facility_type=FacilityType.OUTPATIENT,
            denial_reason=DenialReason.FREQUENCY_LIMIT,
        ),
        clinical_notes=(
            "68yo F with chronic LBP, s/p right total hip replacement 3 years ago. Completed "
            "36 PT visits in past 12 months for LBP (separate from hip rehab which was years ago). "
            "Pain stable at 5/10. ROM unchanged over last 8 visits. Reports PT 'helps her feel "
            "better' but all objective measures are at plateau. Therapist recommends ongoing "
            "maintenance PT 1x/week indefinitely. Requesting 24 more visits."
        ),
        policy_excerpt=(
            "Therapeutic exercises are covered when skilled therapy services are needed and "
            "objective progress is documented. Maximum 36 visits per condition per calendar year. "
            "Beyond 36 visits requires exceptional circumstances with documented continued objective "
            "improvement. Subjective benefit ('feels better') without measurable objective change "
            "does not justify extension beyond annual visit limit."
        ),
        ground_truth_outcome=Outcome.DENIED,
        ground_truth_reasoning=[
            "Already at 36-visit annual maximum — any extension requires exceptional documentation",
            "Objective measures at plateau for last 8 visits — no continued measurable improvement",
            "ROM unchanged, pain stable — no objective data supporting skilled therapy need",
            "Subjective benefit without objective change explicitly does not qualify per policy",
            "Therapist recommendation for indefinite 1x/week is maintenance without skilled need justification",
            "Deny — annual visit limit reached with objective plateau, subjective benefit insufficient",
        ],
        difficulty=Difficulty.HARD,
    ),
    BenchmarkCase(
        case_id="adv-014",
        task_config=PriorAuthTaskConfig(
            payer=Payer.CIGNA, cpt_code="43239",
            icd10_codes=["K21.0", "K44.9"],
            facility_type=FacilityType.ASC,
        ),
        clinical_notes=(
            "57yo M with GERD refractory to omeprazole 40mg BID for 12 weeks. Also has known "
            "large (5cm) paraesophageal hernia on prior CT. New onset dysphagia to solids x 3 weeks. "
            "Weight stable. No GI bleeding. GI requests EGD to evaluate dysphagia and hernia "
            "prior to planned surgical repair."
        ),
        policy_excerpt=(
            "EGD is covered for refractory GERD with failed adequate PPI trial >= 8 weeks, "
            "OR alarm symptoms (dysphagia, weight loss, bleeding, anemia). Pre-operative EGD "
            "for planned paraesophageal hernia repair is considered medically necessary for "
            "surgical planning regardless of GERD treatment status."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "Multiple independent indications: refractory GERD after 12 weeks adequate PPI (exceeds 8-week minimum)",
            "Alarm symptom present: new dysphagia to solids x 3 weeks",
            "Pre-operative indication: EGD for surgical planning of paraesophageal hernia repair",
            "Any ONE of these three indications is sufficient — patient has all three",
            "Approve — meets criteria on multiple independent grounds",
        ],
        difficulty=Difficulty.MEDIUM,
    ),
    BenchmarkCase(
        case_id="adv-015",
        task_config=PriorAuthTaskConfig(
            payer=Payer.MEDICAID, cpt_code="90837",
            icd10_codes=["F33.2", "F60.3"],
            facility_type=FacilityType.OUTPATIENT,
        ),
        clinical_notes=(
            "32yo F with severe recurrent MDD (PHQ-9=23) and borderline personality disorder. "
            "History of 4 psychiatric hospitalizations in 2 years. Currently on lithium + quetiapine. "
            "Not currently suicidal but has chronic passive SI. Requesting 60-minute individual "
            "psychotherapy (DBT-informed). Has been in weekly therapy for 14 months with current "
            "provider — hospitalizations decreased from 4 to 0 in past 8 months."
        ),
        policy_excerpt=(
            "Extended psychotherapy (>12 months) requires re-authorization with documentation of: "
            "(1) continued clinical need (PHQ-9 >= 15 or active safety concerns), "
            "(2) measurable treatment progress using validated outcomes (not just session attendance), "
            "(3) updated treatment plan with time-limited goals. Chronic passive suicidal ideation "
            "without acute plan/intent requires safety plan documentation but does not independently "
            "justify ongoing weekly frequency."
        ),
        ground_truth_outcome=Outcome.APPROVED,
        ground_truth_reasoning=[
            "PHQ-9 of 23 (severe) meets continued clinical need threshold (>= 15)",
            "Measurable outcome: psychiatric hospitalizations decreased from 4 to 0 in 8 months",
            "BPD with hospitalization history demonstrates high acuity requiring specialized ongoing therapy",
            "DBT is evidence-based for BPD — treatment modality is appropriate",
            "Treatment progress clearly documented with validated outcome (hospitalization rate)",
            "Approve — severe MDD with measurable progress in high-acuity BPD patient",
        ],
        difficulty=Difficulty.HARD,
    ),
]
