"""v0.1 adversarial case fixture — hand-authored cases that defeat
zero-shot LLMs that rely on general clinical knowledge.

Each case is a synthetic prior-auth scenario with one of these patterns:

(A) **Surface-approve, actually deny.** The clinical picture looks
    approvable on a quick read. A buried clause in the policy excerpt
    or a missing detail in the notes flips the answer to deny.

(B) **Surface-deny, actually approve.** The clinical picture looks
    deny-worthy. A red-flag symptom, an exception clause, or an emergent
    indication overrides the conservative-therapy / step-therapy /
    frequency-limit rule.

(C) **Overturned on appeal.** Initial reading says deny per the literal
    policy. An appeal-precedent clause buried in the policy excerpt
    flips it to overturned on appeal.

The trap in each case is something a generic LLM cannot know from
training — it requires reading the specific policy excerpt carefully.
This is the population the memory pipeline should help with: Haiku
zero-shot won't pick up the trick clause from general knowledge, but
a memory store seeded with rules extracted from prior verified
resolutions of the same policy will surface the trick rule and inject
it.

This module is NOT a real-data ingestion. It's a hand-crafted
adversarial fixture for the v0.1 dev iteration. Phase 6+ will replace
it with real CMS LCD/NCD coverage determinations once network access
arrives.
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


def _case(
    *,
    case_id: str,
    payer: Payer,
    cpt: str,
    icds: list[str],
    facility: FacilityType,
    notes: str,
    policy: str,
    outcome: Outcome,
    reasoning: list[str],
    difficulty: Difficulty,
    denial_reason=None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        task_config=PriorAuthTaskConfig(
            payer=payer,
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


def build_adversarial_cases() -> list[BenchmarkCase]:
    """Return the full v0.1 adversarial case set.

    Cases are intentionally diverse across CPT family, payer, and
    outcome class so the stratifier has cells to work with.
    """
    cases: list[BenchmarkCase] = []

    # ── Pattern A: surface-approve, actually deny ────────────────────────

    cases.append(_case(
        case_id="adv_001",
        payer=Payer.MEDICARE,
        cpt="72148",
        icds=["M51.16", "M54.5"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "55 y/o with chronic low back pain x 4 months. Completed 8 weeks "
            "of supervised PT and trial of NSAIDs without meaningful relief. "
            "Exam: positive straight-leg raise on the right at 40 degrees, "
            "intact strength and reflexes. Provider requesting lumbar MRI to "
            "evaluate for disc herniation. Prior lumbar MRI 9 months ago "
            "showed mild multilevel degenerative changes without focal "
            "herniation. No interval surgery. No new neurological deficit "
            "since prior imaging."
        ),
        policy=(
            "CMS LCD L34522 §C.1: Lumbar MRI is covered after documented "
            "failure of >=6 weeks conservative therapy.\n"
            "§L.1 (FREQUENCY LIMIT): Repeat lumbar MRI within 12 months of a "
            "prior study is NOT covered unless one of the following is "
            "documented: (a) new neurologic deficit since prior imaging, "
            "(b) significant clinical change, (c) post-operative evaluation, "
            "or (d) interval radiation therapy. Routine surveillance imaging "
            "for chronic pain is excluded."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.FREQUENCY_LIMIT,
        reasoning=[
            "C.1 met: 8 weeks documented PT + NSAIDs.",
            "L.1 triggered: prior MRI 9 months ago (within 12-month window).",
            "L.1 exception NOT met: no new deficit, no significant change, "
            "no post-op, no radiation.",
            "Frequency limit excludes coverage → DENIED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_002",
        payer=Payer.AETNA,
        cpt="43239",
        icds=["K21.9"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "38 y/o with reflux symptoms responsive to PPI for 8 weeks. "
            "Provider requesting EGD with biopsy to evaluate for "
            "esophagitis. Patient is otherwise healthy, no alarm features "
            "(no dysphagia, no weight loss, no GI bleeding, no anemia). "
            "Family history negative for upper GI cancer. Patient requests "
            "the procedure for reassurance."
        ),
        policy=(
            "Aetna Commercial Coverage 0103 — Upper GI Endoscopy.\n"
            "Indication 1: Persistent reflux symptoms despite >=8 weeks "
            "PPI therapy AND age >=50, OR\n"
            "Indication 2: Any age with documented alarm features "
            "(dysphagia, odynophagia, unintentional weight loss >5% body "
            "weight, hematemesis, melena, iron-deficiency anemia, or "
            "first-degree family history of esophageal/gastric cancer).\n"
            "Routine screening EGD without alarm features in patients "
            "<50 is not covered regardless of symptom duration."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.NOT_COVERED,
        reasoning=[
            "Indication 1 NOT met: patient is 38, below age >=50 threshold.",
            "Indication 2 NOT met: no alarm features, no family history.",
            "Routine EGD <50 without alarms is excluded → DENIED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_003",
        payer=Payer.UNITEDHEALTHCARE,
        cpt="20610",
        icds=["M17.11"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "62 y/o with bilateral knee osteoarthritis, right worse than "
            "left. Failed acetaminophen and PT for 8 weeks. Provider "
            "requesting intra-articular hyaluronic acid injection. Patient "
            "had a prior hyaluronic acid injection series in the same knee "
            "4 months ago with documented partial relief. Provider notes "
            "the prior injection 'helped some' and is requesting a second "
            "series."
        ),
        policy=(
            "UnitedHealthcare Commercial Medical Policy: Intra-Articular "
            "Hyaluronic Acid Injections.\n"
            "Coverage: Repeat injection series may be authorized once per "
            "6-month interval per knee, contingent on documented "
            ">=50% pain reduction or >=20-point WOMAC improvement from the "
            "prior series. Subjective reports of 'partial relief' or "
            "'helped some' without quantified functional improvement do "
            "not satisfy the response criterion."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "Repeat series eligibility: 6-month interval — only 4 months "
            "have passed since prior series.",
            "Even if interval were met, response criterion requires "
            "quantified >=50% pain reduction or >=20pt WOMAC improvement.",
            "Notes document only subjective 'helped some' — does not satisfy "
            "the quantitative criterion.",
            "Both criteria fail → DENIED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_004",
        payer=Payer.BCBS,
        cpt="29881",
        icds=["M23.205"],
        facility=FacilityType.ASC,
        notes=(
            "44 y/o male with right knee medial meniscal tear confirmed on "
            "MRI 6 weeks ago. Pain and mechanical symptoms (catching). "
            "Completed 4 weeks of PT without improvement. Provider "
            "requesting arthroscopic partial meniscectomy. Patient also has "
            "BMI 38, A1C 9.2%, smokes 1 PPD."
        ),
        policy=(
            "BCBS Surgical Policy: Knee Arthroscopy.\n"
            "Coverage requires:\n"
            "(1) Documented mechanical symptoms or persistent pain >=6 weeks "
            "despite conservative care.\n"
            "(2) MRI confirmation of meniscal pathology.\n"
            "PERIOPERATIVE PREREQUISITES (all required):\n"
            "(a) HbA1c <8.0 within 30 days of surgery for diabetic patients;\n"
            "(b) BMI <40 for elective ASC procedures;\n"
            "(c) Smoking cessation counseling documented within 60 days.\n"
            "Failure of any perioperative prerequisite results in denial; "
            "patient may resubmit after meeting criteria."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "Indication criteria (1) and (2) met: mechanical symptoms, "
            "MRI-confirmed meniscal tear.",
            "Perioperative prerequisite (a) FAILED: A1C 9.2% > 8.0.",
            "Smoking cessation counseling not documented either.",
            "Any failed prerequisite → DENIED. Patient may resubmit after "
            "glycemic optimization.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_005",
        payer=Payer.CIGNA,
        cpt="64483",
        icds=["M54.16"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "51 y/o with right L5 radiculopathy confirmed by EMG. Failed "
            "8 weeks PT, NSAIDs, and a Medrol dose pack. Provider "
            "requesting transforaminal epidural steroid injection at L5. "
            "Patient is on apixaban 5mg BID for atrial fibrillation. "
            "Provider notes he will hold apixaban for 24 hours pre-procedure."
        ),
        policy=(
            "Cigna Coverage Policy 0163: Spinal Injections.\n"
            "Indications: Radicular pain refractory to >=6 weeks "
            "conservative therapy, with EMG or imaging correlate.\n"
            "ANTICOAGULATION REQUIREMENT: Patients on direct oral "
            "anticoagulants (DOACs including apixaban, rivaroxaban, "
            "dabigatran, edoxaban) must hold the medication for the "
            "duration specified by the most recent ASRA Pain Medicine "
            "guidelines: apixaban requires 3 days off prior to neuraxial "
            "or transforaminal injection (NOT 24 hours). Insufficient "
            "anticoagulant interruption is a hard contraindication."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "Indication criteria met: EMG-confirmed radiculopathy, failed "
            "conservative trial.",
            "Anticoagulation contraindication: provider plans 24-hour hold "
            "but ASRA guidelines (cited in policy) require 3 days off "
            "apixaban for transforaminal injection.",
            "Insufficient interruption is a hard contraindication → DENIED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_006",
        payer=Payer.MEDICARE,
        cpt="77386",
        icds=["C61"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "68 y/o with newly diagnosed prostate adenocarcinoma. Gleason "
            "3+3=6, PSA 6.2, clinical stage T1c. Patient and oncologist "
            "have elected definitive radiation therapy. Provider requesting "
            "IMRT (intensity-modulated radiation therapy)."
        ),
        policy=(
            "Medicare NCD 220.5: External Beam Radiation Therapy for "
            "Prostate Cancer.\n"
            "IMRT covered for prostate cancer when ANY of: (a) high-risk "
            "disease (PSA >20, Gleason >=8, or stage T2c+), (b) intermediate-"
            "risk with documented anatomic complexity precluding 3D-CRT "
            "delivery, (c) post-prostatectomy salvage setting.\n"
            "For low-risk disease (PSA <=10 AND Gleason <=6 AND <=T2a) IMRT "
            "is not medically necessary; conventional 3D conformal therapy "
            "(3D-CRT, CPT 77385) is the appropriate technique."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        reasoning=[
            "Patient is low-risk: PSA 6.2 (<=10), Gleason 6 (<=6), T1c "
            "(<=T2a) — meets all three low-risk criteria.",
            "Policy specifies 3D-CRT (77385) for low-risk disease, not IMRT "
            "(77386).",
            "IMRT not medically necessary for this risk stratum → DENIED. "
            "Resubmission with CPT 77385 likely covered.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="adv_007",
        payer=Payer.AETNA,
        cpt="97110",
        icds=["M25.511"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "47 y/o with right shoulder pain x 6 months. Provider requesting "
            "12 PT visits over 6 weeks. Patient has previously completed "
            "30 PT visits over the prior 6 months for the same shoulder pain "
            "without improvement. Notes do not document a re-evaluation or "
            "change in treatment plan from the prior course."
        ),
        policy=(
            "Aetna Outpatient Rehabilitation Policy.\n"
            "Initial PT: up to 12 visits per body region per episode.\n"
            "Continued PT: requires re-evaluation documenting (a) measurable "
            "objective improvement in functional status from baseline, "
            "(b) updated treatment plan reflecting current deficits, and "
            "(c) reasonable expectation of further improvement. PT for "
            "maintenance of chronic stable conditions is not covered. "
            "Patients who have not improved after 30 visits for a single "
            "condition require independent specialist re-evaluation before "
            "additional PT will be authorized."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.MISSING_INFO,
        reasoning=[
            "Patient has already received 30 PT visits for this condition.",
            "Policy requires independent specialist re-evaluation after "
            "30 visits before additional PT.",
            "Notes do not document the re-evaluation or any objective "
            "functional improvement from prior course.",
            "Continued PT criteria unmet → DENIED for missing info.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="adv_008",
        payer=Payer.HUMANA,
        cpt="72148",
        icds=["M54.5"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "29 y/o with low back pain x 3 months following a CrossFit "
            "workout. No radiation, no neurological symptoms. Completed "
            "8 weeks of PT and ibuprofen. Pain has improved from 7/10 to "
            "4/10. Exam is normal — full strength, intact sensation, "
            "negative SLR. Patient is requesting an MRI 'to make sure "
            "nothing serious is going on' before returning to heavy lifting."
        ),
        policy=(
            "Humana Coverage Policy: Imaging for Low Back Pain.\n"
            "Lumbar MRI is covered when at least ONE red-flag indication "
            "is documented:\n"
            "(a) Neurological deficit (motor weakness, dermatomal sensory "
            "loss, reflex changes, bowel/bladder dysfunction);\n"
            "(b) History of malignancy with new back pain;\n"
            "(c) Suspected infection (fever, IV drug use, immunocompromise);\n"
            "(d) Suspected fracture (significant trauma, age >70, "
            "osteoporosis);\n"
            "(e) Progressive neurological symptoms over weeks;\n"
            "(f) Cauda equina symptoms.\n"
            "Imaging requested for reassurance, return-to-sport clearance, "
            "or improving symptoms is not covered regardless of symptom "
            "duration or conservative therapy completion."
        ),
        outcome=Outcome.DENIED,
        denial_reason=DenialReason.NOT_COVERED,
        reasoning=[
            "No red-flag indication: exam normal, no malignancy, no "
            "infection signs, no fracture risk, no progression, no cauda "
            "equina.",
            "Patient symptoms are improving (7/10 → 4/10).",
            "Reassurance/return-to-sport clearance is explicitly excluded.",
            "→ DENIED.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    # ── Pattern B: surface-deny, actually approve ────────────────────────

    cases.append(_case(
        case_id="adv_009",
        payer=Payer.MEDICARE,
        cpt="72148",
        icds=["M54.16", "G83.4"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "62 y/o with 4 weeks of low back pain that became progressively "
            "worse. Started PT 3 weeks ago, completed 6 sessions. Now "
            "reporting new right foot drop over the past week, with "
            "documented 3/5 tibialis anterior weakness on exam, 4/5 "
            "extensor hallucis longus weakness, and dermatomal L5 sensory "
            "loss. Provider requesting urgent lumbar MRI."
        ),
        policy=(
            "CMS LCD L34522 §C.1: Conservative therapy >=6 weeks required "
            "before lumbar MRI for nonspecific low back pain.\n"
            "§C.2 (NEUROLOGICAL OVERRIDE): The conservative therapy "
            "requirement is WAIVED when the clinical presentation includes "
            "any of the following: (a) progressive motor deficit documented "
            "over consecutive visits, (b) cauda equina symptoms, "
            "(c) suspected epidural abscess or hematoma, "
            "(d) acute foot drop, (e) post-traumatic radiculopathy. "
            "These indications represent surgical emergencies or evolving "
            "deficits where delayed imaging risks permanent injury."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Conservative therapy <6 weeks (only 3 weeks PT) — would "
            "normally fail C.1.",
            "C.2 NEUROLOGICAL OVERRIDE applies: documented progressive "
            "motor deficit AND acute foot drop — TWO independent override "
            "criteria.",
            "C.2 explicitly waives the conservative therapy requirement.",
            "→ APPROVED under the override clause.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_010",
        payer=Payer.AETNA,
        cpt="70553",
        icds=["G44.1", "R51.9"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "34 y/o with new headache x 2 days. No prior headache history. "
            "Reports the worst headache of her life, sudden onset while "
            "exercising. No fever, no rash. Exam: photophobia, mild neck "
            "stiffness, no focal deficit. CT head without contrast read as "
            "normal at the outside ED 24 hours ago. Provider requesting "
            "brain MRI with and without contrast to evaluate for vascular "
            "abnormality."
        ),
        policy=(
            "Aetna Commercial Coverage Policy: Neuroimaging for Headache.\n"
            "Routine imaging for primary headache disorders is not covered.\n"
            "EXCEPTIONS — coverage applies for new headache when ANY of:\n"
            "(a) Thunderclap onset (worst-headache-of-life with sudden onset);\n"
            "(b) New focal neurologic deficit;\n"
            "(c) Onset after age 50 with new pattern;\n"
            "(d) Immunocompromised host;\n"
            "(e) History of malignancy;\n"
            "(f) Trauma within prior 30 days;\n"
            "(g) Meningismus or papilledema on exam.\n"
            "Negative non-contrast CT does NOT rule out subarachnoid "
            "hemorrhage or vascular abnormality beyond 6 hours post-onset; "
            "MRI/MRA is the appropriate next study when thunderclap "
            "presentation persists."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Surface read: routine imaging for headache is excluded.",
            "BUT: thunderclap onset (exception (a)) AND meningismus on exam "
            "(exception (g)) — TWO independent exception criteria met.",
            "Negative CT >6 hours after onset does not rule out vascular "
            "etiology — MRI/MRA explicitly indicated by policy.",
            "→ APPROVED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_011",
        payer=Payer.UNITEDHEALTHCARE,
        cpt="93458",
        icds=["I20.0"],
        facility=FacilityType.INPATIENT,
        notes=(
            "58 y/o male presented to ED with crushing substernal chest "
            "pain at rest, troponin elevated to 1.8 (normal <0.04), ECG "
            "with new T-wave inversions in V3-V6. Hemodynamically stable. "
            "Cardiology consulted, recommends urgent left heart "
            "catheterization. No prior stress testing has been performed."
        ),
        policy=(
            "UnitedHealthcare Cardiac Catheterization Policy.\n"
            "ELECTIVE catheterization requires prior non-invasive risk "
            "stratification (stress test, CCTA, or myocardial perfusion "
            "imaging) demonstrating ischemia or high-risk anatomy.\n"
            "URGENT/EMERGENT catheterization without prior stress testing "
            "is COVERED for any of: (a) acute coronary syndrome with "
            "biomarker elevation, (b) acute ST-elevation MI, "
            "(c) hemodynamic instability suggesting cardiogenic shock, "
            "(d) post-arrest with shockable rhythm, (e) refractory chest "
            "pain despite maximal medical therapy. The requirement for "
            "prior stress testing applies only to elective evaluation of "
            "stable chest pain."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Surface read: no prior stress test → fail elective criterion.",
            "BUT: NSTEMI (troponin 1.8, new T-wave changes) = ACS with "
            "biomarker elevation → exception (a) applies.",
            "Stress-test prerequisite is for ELECTIVE evaluation only; "
            "urgent ACS bypass is explicitly covered.",
            "→ APPROVED.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="adv_012",
        payer=Payer.MEDICARE,
        cpt="J9035",
        icds=["C18.7", "C78.7"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "65 y/o with metastatic colorectal cancer, KRAS wild-type, "
            "BRAF wild-type, microsatellite stable. Liver and lung "
            "metastases. ECOG 1. Oncologist requesting bevacizumab in "
            "combination with FOLFOX as first-line therapy. Plan submitted "
            "without separate prior authorization for the bevacizumab "
            "component because oncologist believes it's part of the "
            "standard regimen."
        ),
        policy=(
            "Medicare Part B Drug Coverage: Bevacizumab (J9035).\n"
            "Coverage criteria: FDA-approved indication AND adherence to "
            "NCCN Compendium category 1 or 2A recommendations. Metastatic "
            "colorectal cancer first-line is FDA-approved.\n"
            "DOCUMENTATION REQUIREMENTS: blood pressure baseline within "
            "30 days, urinalysis for proteinuria, no recent surgery within "
            "28 days, no active uncontrolled hypertension, no recent GI "
            "perforation history. Notes referencing 'standard regimen' "
            "without explicit documentation of these contraindication "
            "screens are processed administratively.\n"
            "ADMINISTRATIVE PROCESSING NOTE: For first-line metastatic CRC "
            "with documented FDA indication, claims with incomplete "
            "documentation are auto-approved under the NCCN Compendium "
            "category 1 standing order, contingent on retrospective "
            "documentation review."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Surface read: documentation requirements not explicitly met.",
            "BUT: FDA-approved indication confirmed (mCRC first-line).",
            "NCCN category 1 standing order provides administrative "
            "auto-approval for this exact scenario.",
            "Retrospective documentation review covers the contraindication "
            "screen requirement.",
            "→ APPROVED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_013",
        payer=Payer.BCBS,
        cpt="99285",
        icds=["R07.9"],
        facility=FacilityType.ED,
        notes=(
            "72 y/o female with chest pain, presented to an OUT-OF-NETWORK "
            "emergency department after collapsing in a grocery store. "
            "Initial ECG showed STEMI; treated with door-to-balloon PCI "
            "within 60 minutes. Patient was discharged 3 days later in "
            "stable condition. The hospital is filing for level-5 ED "
            "evaluation; the BCBS plan requires in-network for all "
            "non-emergent care."
        ),
        policy=(
            "BCBS Standard Plan: Out-of-Network Emergency Services.\n"
            "Emergency services are covered at in-network rates regardless "
            "of facility network status when the prudent layperson standard "
            "is met: a reasonable layperson with average knowledge of "
            "health and medicine would believe immediate medical attention "
            "is required to avoid serious harm to health, serious "
            "impairment of bodily function, or serious dysfunction of any "
            "bodily organ. EMTALA additionally requires acceptance of "
            "patients with emergency medical conditions regardless of "
            "ability to pay or insurance status. Out-of-network exclusions "
            "apply ONLY to non-emergent services."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Surface read: out-of-network → not covered.",
            "BUT: STEMI with collapse meets the prudent layperson standard "
            "trivially (immediate threat to life).",
            "EMTALA + prudent layperson clause covers OON ED at in-network "
            "rates for emergency conditions.",
            "Out-of-network exclusion explicitly only applies to "
            "non-emergent services.",
            "→ APPROVED.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="adv_014",
        payer=Payer.CIGNA,
        cpt="J2350",
        icds=["G35"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "42 y/o with relapsing-remitting multiple sclerosis, EDSS 2.5. "
            "Failed glatiramer acetate (Copaxone) due to injection site "
            "reactions and subsequent breakthrough relapse on therapy. "
            "Provider requesting natalizumab (Tysabri) infusions. Patient "
            "is JCV antibody negative (recent test, index 0.2). MRI shows "
            "no new lesions. Provider notes the patient prefers monthly "
            "infusion to daily injection."
        ),
        policy=(
            "Cigna Specialty Drug Policy: Natalizumab (J2350) for MS.\n"
            "Natalizumab is a second-line therapy. Coverage requires:\n"
            "(1) Documented relapsing-remitting MS;\n"
            "(2) JCV antibody status known (negative or low-positive <0.9);\n"
            "(3) Failed or intolerant of at least ONE first-line "
            "disease-modifying therapy (interferon beta, glatiramer "
            "acetate, dimethyl fumarate, teriflunomide, or fingolimod) "
            "with EITHER documented intolerance OR breakthrough disease "
            "activity (clinical relapse or new MRI lesion);\n"
            "(4) Provider attestation of PML risk discussion."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "(1) RRMS confirmed.",
            "(2) JCV antibody negative (index 0.2 is negative).",
            "(3) Failed glatiramer acetate (first-line) with BOTH "
            "intolerance (injection reactions) AND breakthrough relapse.",
            "(4) Provider attestation expected at administration.",
            "All criteria met → APPROVED. Patient preference is "
            "irrelevant to coverage but doesn't disqualify.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    cases.append(_case(
        case_id="adv_015",
        payer=Payer.MEDICAID,
        cpt="J0490",
        icds=["M32.10"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "31 y/o female with systemic lupus erythematosus, severe "
            "lupus nephritis biopsy class IV. Failed mycophenolate mofetil "
            "with progression to nephrotic-range proteinuria. Rheumatology "
            "and nephrology jointly recommend belimumab as add-on therapy. "
            "Patient has not yet tried cyclophosphamide. State Medicaid "
            "step therapy normally requires cyclophosphamide trial before "
            "belimumab."
        ),
        policy=(
            "State Medicaid Specialty Drug Policy: Belimumab.\n"
            "Step therapy requires sequential trial of (1) hydroxychloroquine, "
            "(2) mycophenolate mofetil OR azathioprine, (3) cyclophosphamide, "
            "before belimumab is authorized.\n"
            "STEP THERAPY OVERRIDE — coverage proceeds without exhausting "
            "step therapy when ANY of:\n"
            "(a) Documented absolute contraindication to a required step "
            "(prior allergy, severe organ dysfunction);\n"
            "(b) Pregnancy or pregnancy planning within 12 months "
            "(cyclophosphamide is teratogenic and gonadotoxic);\n"
            "(c) Class V/VI lupus nephritis with active disease;\n"
            "(d) Joint specialty recommendation from rheumatology AND "
            "nephrology documenting that the next step therapy poses "
            "greater risk than direct progression."
        ),
        outcome=Outcome.APPROVED,
        reasoning=[
            "Patient has not tried cyclophosphamide — would normally fail "
            "step therapy.",
            "Override (d) applies: joint rheum + nephro recommendation is "
            "documented.",
            "Override (a) potentially applies if there's any prior reaction "
            "but not documented in notes.",
            "Override (d) alone is sufficient.",
            "→ APPROVED.",
        ],
        difficulty=Difficulty.HARD,
    ))

    # ── Pattern C: overturned on appeal ──────────────────────────────────

    cases.append(_case(
        case_id="adv_016",
        payer=Payer.MEDICARE,
        cpt="72148",
        icds=["M54.16"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "59 y/o with right L5 radiculopathy. Completed 4 weeks of "
            "formal billed PT plus an additional 4 weeks of supervised "
            "home exercise program directed by the PT (total 8 weeks). "
            "Initial submission included only the 4 weeks of formal billed "
            "PT and was denied for insufficient conservative therapy "
            "duration. Provider appealed, submitting the home exercise "
            "documentation and a letter from the PT verifying the "
            "supervision and progression. EMG confirms L5 radiculopathy."
        ),
        policy=(
            "CMS LCD L34522 §C.1: Lumbar MRI requires >=6 weeks documented "
            "conservative therapy.\n"
            "§A.7 (APPEAL CRITERIA): On appeal, supervised home exercise "
            "programs documented by a licensed PT may count toward the "
            "conservative therapy duration requirement at the "
            "approximate ratio of 1 week home program = 1 week formal PT, "
            "PROVIDED the home program is documented in the PT records "
            "with progression notes and the patient demonstrated "
            "compliance. Initial denials based on formal PT visit count "
            "alone may be overturned when the home exercise component is "
            "subsequently documented."
        ),
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        reasoning=[
            "Initial denial: 4 weeks formal PT < 6-week C.1 requirement.",
            "Appeal submission: 4 additional weeks of supervised home "
            "exercise documented by PT with progression notes.",
            "§A.7 explicitly allows home exercise to count toward "
            "duration on appeal at 1:1 ratio.",
            "Total documented: 8 weeks (4 formal + 4 home) >= 6.",
            "EMG confirms radiculopathy → C.2 also met.",
            "Appeal granted → OVERTURNED ON APPEAL.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_017",
        payer=Payer.AETNA,
        cpt="78815",
        icds=["C34.91"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "67 y/o newly diagnosed non-small cell lung cancer, stage IIIA "
            "by CT and bronchoscopic biopsy. Provider requesting whole-body "
            "PET-CT for staging prior to definitive chemoradiation. Initial "
            "submission denied because policy requires staging PET only for "
            "stages I-II being considered for surgical resection. Provider "
            "appealed, citing the patient's stage IIIA classification and "
            "the role of PET in identifying occult metastases that would "
            "change the treatment intent from curative chemoradiation to "
            "palliative."
        ),
        policy=(
            "Aetna Imaging Coverage: PET-CT for Lung Cancer.\n"
            "Coverage Indication 1: Initial staging in clinical stage I-II "
            "NSCLC being considered for surgical resection.\n"
            "Coverage Indication 2 (APPEAL-ELIGIBLE): Stage III NSCLC when "
            "PET would alter the planned treatment intent (curative vs "
            "palliative). Initial denials under Indication 1 based on "
            "stage classification may be overturned on appeal when "
            "Indication 2 is invoked with documentation that the imaging "
            "would distinguish between locoregional disease appropriate "
            "for chemoradiation and metastatic disease appropriate for "
            "systemic therapy alone.\n"
            "Coverage Indication 3: Treatment response assessment 12 weeks "
            "after definitive therapy."
        ),
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        reasoning=[
            "Initial denial: Indication 1 only covers stages I-II, patient "
            "is IIIA.",
            "Appeal cites Indication 2: Stage III NSCLC with treatment-"
            "intent decision (curative chemoradiation vs palliative).",
            "Indication 2 is explicitly appeal-eligible.",
            "Provider documents the curative-vs-palliative distinction.",
            "→ OVERTURNED ON APPEAL.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_018",
        payer=Payer.UNITEDHEALTHCARE,
        cpt="0001U",
        icds=["C50.911"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "54 y/o with newly diagnosed invasive ductal carcinoma of the "
            "right breast, ER+/PR+/HER2-, T2N1M0. Provider requesting "
            "MammaPrint 70-gene assay to guide adjuvant chemotherapy "
            "decision. Initial submission was denied for lack of medical "
            "necessity because Oncotype DX is the plan's preferred assay. "
            "Provider appealed, noting that the patient is node-positive "
            "(N1) with one positive sentinel node and that MammaPrint is "
            "specifically validated and prospectively trialed (MINDACT) in "
            "node-positive disease while Oncotype DX RxPONDER trial "
            "showed limited benefit prediction in postmenopausal "
            "node-positive women, leaving MammaPrint as the more clinically "
            "informative test for this specific patient."
        ),
        policy=(
            "UnitedHealthcare Genomic Profiling for Breast Cancer.\n"
            "Preferred assay: Oncotype DX (CPT 81519) for node-negative ER+ "
            "HER2- early breast cancer.\n"
            "ALTERNATIVE ASSAY APPEAL: MammaPrint may be substituted on "
            "appeal when documented that (a) the patient is node-positive "
            "(N1-N2) with 1-3 positive nodes, (b) MammaPrint validation "
            "in this subgroup (MINDACT trial) is more clinically applicable "
            "than Oncotype DX, AND (c) the result will inform a specific "
            "adjuvant chemotherapy decision. Plan's standard preference for "
            "Oncotype DX does not preclude alternative assay coverage when "
            "trial-specific applicability is documented."
        ),
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        reasoning=[
            "Initial denial: Oncotype DX is the plan-preferred assay.",
            "Appeal documents patient is N1 (1 positive node), within the "
            "MINDACT-validated subgroup.",
            "Appeal documents the clinical decision (adjuvant chemo) the "
            "result would guide.",
            "Alternative assay appeal criteria (a), (b), (c) all met.",
            "→ OVERTURNED ON APPEAL.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_019",
        payer=Payer.HUMANA,
        cpt="J9355",
        icds=["C61"],
        facility=FacilityType.OUTPATIENT,
        notes=(
            "76 y/o with metastatic castration-resistant prostate cancer, "
            "previously treated with docetaxel and abiraterone. Now with "
            "rising PSA and new bone pain. Provider requesting lutetium-177 "
            "PSMA-617 (Pluvicto). Initial submission denied because policy "
            "requires PSMA-PET imaging confirmation, which the patient has "
            "not yet undergone. Provider appealed, noting the patient has a "
            "documented contrast allergy to gadolinium-based agents and is "
            "unable to tolerate the PET scanning protocol; additionally, "
            "the patient has prior bone scan and CT documenting PSMA-"
            "expressing metastatic disease pattern characteristic of "
            "PSMA+ tumor biology."
        ),
        policy=(
            "Humana Specialty Drug Policy: Lu-177 PSMA-617 (Pluvicto).\n"
            "Standard prerequisite: PSMA-PET confirmation of PSMA-positive "
            "metastatic castration-resistant prostate cancer.\n"
            "PSMA-PET WAIVER (APPEAL-ELIGIBLE): The PSMA-PET requirement "
            "may be waived on appeal when the patient has a documented "
            "contraindication to PET imaging (severe contrast allergy, "
            "inability to lie still due to pain or claustrophobia "
            "refractory to sedation, dialysis-dependent renal failure) "
            "AND alternative imaging (bone scan, CT, MRI) demonstrates "
            "metastatic burden consistent with PSMA-expressing prostate "
            "adenocarcinoma."
        ),
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        reasoning=[
            "Initial denial: no PSMA-PET confirmation.",
            "Appeal: documented contrast allergy = PET contraindication.",
            "Alternative imaging (bone scan, CT) shows metastatic pattern.",
            "Both waiver conditions (contraindication AND alt imaging) met.",
            "→ OVERTURNED ON APPEAL.",
        ],
        difficulty=Difficulty.HARD,
    ))

    cases.append(_case(
        case_id="adv_020",
        payer=Payer.MEDICARE,
        cpt="33533",
        icds=["I25.10"],
        facility=FacilityType.INPATIENT,
        notes=(
            "71 y/o male with three-vessel coronary disease, EF 35%, "
            "diabetic, prior stroke. Cardiology evaluated for revascularization, "
            "concluded surgical CABG superior to PCI per SYNTAX score. "
            "Initial CABG authorization denied as 'PCI is the standard "
            "first-line approach.' Provider appealed citing the patient's "
            "diabetes, three-vessel disease, and reduced EF — all factors "
            "where guideline-directed therapy favors CABG over PCI."
        ),
        policy=(
            "Medicare NCD: Coronary Revascularization Procedures.\n"
            "Both PCI and CABG are covered for appropriate clinical "
            "indications. The choice between modalities follows "
            "ACC/AHA/SCAI guideline-directed appropriate use criteria.\n"
            "APPEAL-ELIGIBLE OVERRIDE for CABG over PCI (Class I "
            "recommendations): (a) Three-vessel disease with diabetes, "
            "(b) Left main disease with SYNTAX score >32, "
            "(c) Reduced EF <=35% with multivessel disease, "
            "(d) Failed prior PCI of the same lesion, "
            "(e) Anatomy unsuitable for percutaneous approach.\n"
            "Initial denials based on a generic PCI-first preference may "
            "be overturned when guideline-directed Class I CABG "
            "indications are documented."
        ),
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        reasoning=[
            "Initial denial: PCI-first generic preference.",
            "Appeal: patient meets MULTIPLE Class I CABG indications: "
            "(a) three-vessel + diabetes, (c) EF 35% with multivessel.",
            "Override criteria explicitly appeal-eligible.",
            "→ OVERTURNED ON APPEAL.",
        ],
        difficulty=Difficulty.MEDIUM,
    ))

    return cases
