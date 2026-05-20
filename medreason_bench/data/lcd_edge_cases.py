"""v0.2 fixture — 30 LCD-derived prior auth edge cases.

Source: hand-authored by the user from real Local Coverage Determination
(LCD) edge-case patterns. Each case is a tricky scenario where general
clinical knowledge would lead to the wrong determination unless the
agent reads a specific operational nuance buried in the policy text.

Loading flow:
1. Read mvp_dashboard/edge_cases_raw.json (the user's xlsx exported as JSON).
2. For each row, map (category → CPT/ICD-10/payer) and build a synthetic
   LCD-style policy_excerpt that contains the trick clause.
3. Map the "Expected Correct Determination" to our 3-class Outcome:
     APPROVE      → Outcome.APPROVED
     DENY (any)   → Outcome.DENIED
     CONDITIONAL  → Outcome.DENIED (conservative — agent should not auto-approve)
     FLAG/DENY    → Outcome.DENIED
     REQUIRES     → Outcome.DENIED
     PARTIAL      → Outcome.DENIED (request as-stated has a deny component)
   Notes:
   - 1/30 case is APPROVE (case 1: split-night PSG after HSAT failure)
   - 29/30 cases collapse to DENIED
   - Class imbalance is REAL: real prior auth review is mostly catching
     tricky denials. The headline metric should be DENY-recall + the
     single APPROVE case as a sanity check that the agent isn't just
     blindly denying everything.
4. ground_truth_reasoning is derived from the "Tricky Because" column
   so it carries the operational nuance for human auditors.
5. difficulty is HARD on all 30 by construction.

This module is loaded by `medreason_bench data build --source lcd_edge`.
"""

from __future__ import annotations

import json
from pathlib import Path
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


_RAW_PATH = Path(__file__).parent.parent.parent / "mvp_dashboard" / "edge_cases_raw.json"


# ── Per-case overrides ──────────────────────────────────────────────────────
#
# Each case authored from the xlsx row gets a CPT, ICD-10, and a synthesized
# LCD-style policy excerpt that contains the trick clause. Indexed by the
# row number from the xlsx (1-based).

_CASE_OVERRIDES: dict[int, dict] = {
    1: dict(
        cpt="95810", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L37044 — Polysomnography and Related Sleep Studies\n\n"
            "Indications for facility-based polysomnography (PSG, CPT 95810):\n"
            "1. AHI >=15 documented on prior sleep study, OR\n"
            "2. AHI 5-14 with documented daytime sleepiness, comorbidities, "
            "or other clinical concern, OR\n"
            "3. Failed home sleep apnea test (HSAT) due to technical "
            "inadequacy (probe dislodgement, signal loss, insufficient "
            "study duration). Technical HSAT failure overrides any "
            "pre-test AHI threshold — facility PSG is covered regardless "
            "of mild AHI when HSAT was technically inadequate.\n\n"
            "Split-night PSG: covered when initial diagnostic study "
            "confirms moderate-to-severe OSA within the same session "
            "and titration is initiated."
        ),
    ),
    2: dict(
        cpt="95800", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L37044 — Home Sleep Apnea Testing (HSAT, CPT 95800/95806)\n\n"
            "Coverage criteria:\n"
            "1. Adult patients (age >=18) with high pre-test probability "
            "of moderate-to-severe OSA;\n"
            "2. No significant comorbidities that would contraindicate "
            "ambulatory testing.\n\n"
            "EXCLUSIONS — HSAT is NOT covered for any of:\n"
            "- Pediatric patients (age <18). HSAT is not validated for "
            "children; facility-based in-lab PSG is required.\n"
            "- Patients with significant cardiopulmonary disease "
            "(CHF NYHA III/IV, severe COPD, neuromuscular disease).\n"
            "- Patients with suspected non-obstructive sleep disorders.\n\n"
            "When an exclusion applies, in-lab PSG (CPT 95810) is the "
            "appropriate test."
        ),
    ),
    3: dict(
        cpt="95800", icd=["G47.33", "I50.32"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L37044 — Home Sleep Apnea Testing (HSAT)\n\n"
            "HSAT is excluded for patients with significant cardiopulmonary "
            "disease, defined as:\n"
            "- Congestive heart failure NYHA class III or IV;\n"
            "- Severe COPD (FEV1 <50% predicted) or chronic respiratory "
            "failure;\n"
            "- Neuromuscular disease affecting respiratory muscles;\n"
            "- Hypoventilation syndromes.\n\n"
            "These patients require attended in-lab polysomnography "
            "(CPT 95810) due to the higher likelihood of central apnea, "
            "complex sleep-disordered breathing, and the need for direct "
            "physician/technologist supervision during the study."
        ),
    ),
    4: dict(
        cpt="95805", icd=["G47.419"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L34527 — Multiple Sleep Latency Test (MSLT, CPT 95805)\n\n"
            "MSLT is covered for the diagnosis of narcolepsy or idiopathic "
            "hypersomnia ONLY when ALL of the following are met:\n"
            "1. The MSLT is preceded by an overnight nocturnal "
            "polysomnogram (PSG) within the prior 24-48 hours, performed "
            "to rule out other sleep disorders that could explain "
            "excessive daytime sleepiness;\n"
            "2. The PSG demonstrates adequate sleep duration (>=6 hours);\n"
            "3. The patient is not on REM-suppressing medications during "
            "the study (washout period documented).\n\n"
            "MSLT performed in isolation, without a preceding overnight "
            "PSG, is not covered. Clinical history of narcolepsy alone "
            "does not establish coverage."
        ),
    ),
    5: dict(
        cpt="95810", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L37044 — Repeat Sleep Studies\n\n"
            "Repeat polysomnography is covered only when ONE of the "
            "following medical necessity criteria is documented:\n"
            "(a) Failure of prior treatment with documented clinical "
            "change (worsening daytime symptoms, weight gain >10%, "
            "new comorbidity);\n"
            "(b) Need to retitrate CPAP/BiPAP after >12 months on therapy;\n"
            "(c) New onset of central sleep apnea on prior treatment;\n"
            "(d) Treatment failure with documented inadequate adherence "
            "or persistent symptoms despite >=4 hours nightly use.\n\n"
            "Repeat studies for patient request, second opinion, or "
            "without intervening treatment trial are NOT covered."
        ),
    ),
    6: dict(
        cpt="E0486", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33611 — Oral Appliances for Obstructive Sleep Apnea\n\n"
            "Oral appliances (E0486) are covered for OSA when:\n"
            "1. AHI 5-29 (mild to moderate OSA), OR\n"
            "2. AHI >=30 (severe OSA) AND documented CPAP failure, "
            "intolerance, or refusal after adequate trial.\n\n"
            "STEP THERAPY: Severe OSA (AHI >=30) requires documented "
            "CPAP trial failure before oral appliance is covered as "
            "first-line therapy. The CPAP trial must be at least 30 days "
            "with documented compliance attempt or contraindication.\n"
            "Patient preference alone does not satisfy step therapy."
        ),
    ),
    7: dict(
        cpt="E0486", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33611 — Oral Appliances for OSA: Replacement\n\n"
            "Replacement of an existing oral appliance is covered when:\n"
            "(a) The original device is documented as broken, lost, or "
            "no longer functional;\n"
            "(b) Significant change in dental or skeletal anatomy "
            "(orthodontic treatment, extraction, restoration);\n"
            "(c) Documented clinical re-evaluation showing the device "
            "is no longer effective at controlling AHI;\n"
            "(d) Material wear documented by the fitting dentist.\n\n"
            "Patient comfort or preference alone does NOT establish "
            "medical necessity for replacement. The original device "
            "must be functionally inadequate."
        ),
    ),
    8: dict(
        cpt="E0486", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33611 — Oral Appliances: Ordering Provider Requirements\n\n"
            "The diagnosis of obstructive sleep apnea and the order for "
            "an oral appliance must be made by a physician (MD or DO). "
            "A dentist may fabricate, fit, and adjust the appliance, "
            "but cannot independently diagnose OSA or write the "
            "Medicare DME order.\n\n"
            "Required documentation:\n"
            "- Sleep study interpreted by a physician credentialed in "
            "sleep medicine;\n"
            "- Written physician order for the oral appliance;\n"
            "- Physician evaluation of the patient distinct from the "
            "dentist's fitting visit.\n\n"
            "Claims with sleep studies interpreted by the dentist's "
            "affiliated lab without independent physician involvement "
            "do not meet the ordering provider requirement."
        ),
    ),
    9: dict(
        cpt="E0486", icd=["G47.31"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33611 — Oral Appliances: Covered Diagnoses\n\n"
            "Oral mandibular advancement appliances (E0486) are covered "
            "exclusively for OBSTRUCTIVE sleep apnea (ICD-10 G47.33).\n\n"
            "NOT COVERED for any of:\n"
            "- Central sleep apnea (G47.31). Mandibular advancement "
            "addresses upper airway obstruction; central apnea is a "
            "different pathophysiology and oral appliances are not "
            "indicated.\n"
            "- Mixed sleep apnea where central events predominate.\n"
            "- Snoring without documented OSA.\n"
            "- Insomnia or other non-respiratory sleep disorders."
        ),
    ),
    10: dict(
        cpt="E0485", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "HCPCS Coding for Oral Appliances:\n\n"
            "E0485 (Oral Device, Prefabricated): Covered for "
            "off-the-shelf, mass-produced devices that do not require "
            "dental impressions or laboratory fabrication.\n\n"
            "E0486 (Oral Device, Custom Fabricated): Required when the "
            "device is constructed via dental impressions, bite "
            "registration, and laboratory fabrication. Custom devices "
            "documented as such must be billed under E0486, not E0485.\n\n"
            "Code-documentation mismatch (billing E0485 when "
            "documentation describes custom impressions, bite "
            "registration, and lab fabrication) constitutes a coding "
            "error. The claim should be denied or recoded to E0486."
        ),
    ),
    11: dict(
        cpt="E0470", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33718 — Positive Airway Pressure (PAP) Devices\n\n"
            "CPAP (E0601) is the first-line therapy for obstructive "
            "sleep apnea. BiPAP (E0470) is covered ONLY when one of:\n"
            "(a) Documented CPAP failure or intolerance after adequate "
            "trial of >=30 days with appropriate mask fitting and "
            "pressure adjustment;\n"
            "(b) Documented central sleep apnea component requiring "
            "bilevel support;\n"
            "(c) Hypoventilation syndromes (obesity hypoventilation, "
            "neuromuscular disease).\n\n"
            "Patient preference for BiPAP does NOT establish medical "
            "necessity. CPAP must be tried first for uncomplicated OSA."
        ),
    ),
    12: dict(
        cpt="E0601", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33718 — CPAP Compliance Requirement\n\n"
            "Continued coverage of CPAP rental beyond the initial 90-day "
            "trial period requires documentation of compliance, defined as:\n"
            "- CPAP usage of >=4 hours per night on >=70% of nights "
            "during a 30-day period within the first 90 days of therapy.\n\n"
            "Compliance is determined by the device's data download "
            "(usage hours per night). Average usage <4 hours per night "
            "fails the compliance threshold regardless of clinical "
            "improvement reported by the patient. Partial compliance "
            "and 'patient is trying' do not satisfy the criterion."
        ),
    ),
    13: dict(
        cpt="E0466", icd=["G47.31", "I50.20"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33718 — Adaptive Servo-Ventilation (ASV, E0466)\n\n"
            "ASV is a specialized PAP modality covered for treatment of "
            "complex sleep-disordered breathing including treatment-"
            "emergent central apnea on CPAP.\n\n"
            "SAFETY CONTRAINDICATION: ASV is contraindicated and NOT "
            "covered in patients with predominant central sleep apnea "
            "AND symptomatic chronic heart failure (NYHA II-IV) with a "
            "left ventricular ejection fraction (LVEF) <=45%. The "
            "SERVE-HF trial demonstrated increased cardiovascular "
            "mortality in this population.\n\n"
            "Required documentation: recent echocardiogram with LVEF, "
            "characterization of central vs obstructive event "
            "predominance, and provider attestation that the SERVE-HF "
            "contraindication does not apply."
        ),
    ),
    14: dict(
        cpt="97605", icd=["L89.30"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L37166 — Negative Pressure Wound Therapy (NPWT)\n\n"
            "NPWT is covered for full-thickness wounds where:\n"
            "1. Stage III or Stage IV pressure ulcer documented, OR\n"
            "2. Complex diabetic foot ulcer Wagner grade 3-5, OR\n"
            "3. Surgical wound dehiscence requiring secondary closure, OR\n"
            "4. Traumatic wound with significant tissue loss.\n\n"
            "Coverage requires documented failure of prior conservative "
            "wound care (debridement, moisture-balanced dressings) for "
            ">=30 days unless the wound is acutely deteriorating.\n\n"
            "Stage I and Stage II pressure ulcers are NOT covered for "
            "NPWT — these are partial-thickness wounds where standard "
            "moist wound healing is appropriate. Wound presence alone "
            "does not establish NPWT necessity."
        ),
    ),
    15: dict(
        cpt="15275", icd=["L97.529"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L36690 — Skin Substitute Grafts\n\n"
            "Skin substitute application is covered when documentation "
            "supports the indication and product selection. ONLY ONE "
            "skin substitute product may be applied to a single wound "
            "on a single date of service. Stacking multiple skin "
            "substitutes on the same wound on the same DOS is "
            "prohibited regardless of total wound area or product type.\n\n"
            "When two products are billed for the same wound on the "
            "same date, one of the two must be denied. Provider should "
            "select the more appropriate product based on wound "
            "characteristics; the secondary product is non-covered. "
            "Sequential applications on different dates are acceptable "
            "when prior application has been evaluated and shown "
            "incomplete coverage or ongoing healing need."
        ),
    ),
    16: dict(
        cpt="81479", icd=["Z79.899"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L38394 — Pharmacogenomic Testing\n\n"
            "Pharmacogenomic panel testing is covered when ALL of:\n"
            "1. The patient has a current active medication management "
            "problem (treatment failure, intolerance, ADR, or planned "
            "initiation of a drug with known PGx-relevant interactions);\n"
            "2. Specific genes tested are clinically actionable per "
            "CPIC or FDA guidance for the patient's medications;\n"
            "3. Results will inform a documented treatment decision.\n\n"
            "PROPHYLACTIC or screening pharmacogenomic testing without "
            "an active medication management problem is NOT covered. "
            "Testing 'in case' the patient ever needs PGx-relevant "
            "drugs is excluded regardless of panel sophistication or "
            "number of genes."
        ),
    ),
    17: dict(
        cpt="81415", icd=["Z13.79"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L38419 — Whole Exome and Whole Genome Sequencing\n\n"
            "Whole exome sequencing (WES) is covered for patients with "
            "suspected genetic disease when ALL prerequisites are met:\n"
            "1. Prior single-gene or targeted panel testing has been "
            "performed and is non-diagnostic, OR the suspected condition "
            "is too genetically heterogeneous for targeted testing;\n"
            "2. Genetic counseling has been provided and documented "
            "before testing, including informed consent for incidental "
            "findings;\n"
            "3. Results will inform clinical management or reproductive "
            "decision-making.\n\n"
            "Single negative gene test alone does not satisfy the "
            "tiered-testing requirement; documented targeted panel "
            "testing or appropriate justification for skipping is "
            "required before WES is covered."
        ),
    ),
    18: dict(
        cpt="72148", icd=["M54.5"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L34522 / NCD 220.2 — Lumbar MRI for Low Back Pain\n\n"
            "Lumbar MRI is covered for low back pain when ALL of:\n"
            "1. Documented failure of >=6 weeks of conservative therapy "
            "(physical therapy, NSAIDs, activity modification);\n"
            "2. Persistent symptoms despite conservative care;\n"
            "3. Plan to alter management based on imaging results.\n\n"
            "EXCEPTIONS — the 6-week conservative trial is waived when "
            "ANY red-flag indication is present:\n"
            "(a) New or progressive neurological deficit;\n"
            "(b) Cauda equina symptoms;\n"
            "(c) Known or suspected malignancy;\n"
            "(d) Suspected infection;\n"
            "(e) Significant trauma.\n\n"
            "Acute uncomplicated low back pain (<6 weeks duration) "
            "without red flags or completed conservative trial is NOT "
            "covered for advanced imaging. Patient pain complaints alone "
            "do not establish necessity."
        ),
    ),
    19: dict(
        cpt="78815", icd=["C64.9"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "NCD 220.6 — PET Scan Coverage by Cancer Type\n\n"
            "PET scan coverage is cancer-specific. PET is NOT a covered "
            "modality for INITIAL STAGING of:\n"
            "- Renal cell carcinoma (any stage);\n"
            "- Prostate cancer (use bone scan / PSMA-PET only when "
            "specific criteria met);\n"
            "- Localized thyroid cancer (use ultrasound / I-131 scan).\n\n"
            "For renal cell carcinoma specifically: initial staging is "
            "performed with cross-sectional imaging (CT abdomen/pelvis "
            "with contrast). PET-CT does not have sufficient incremental "
            "value over CT for initial RCC staging and is not covered. "
            "PET may be considered for restaging after recurrence or "
            "for indeterminate findings on conventional imaging."
        ),
    ),
    20: dict(
        cpt="75561", icd=["I25.10"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33559 — Cardiac MRI\n\n"
            "Cardiac MRI is covered when it provides incremental "
            "diagnostic value beyond prior workup. For evaluation of "
            "stable chest pain, the appropriate workup hierarchy is:\n"
            "1. Stress testing (exercise or pharmacologic);\n"
            "2. Cardiac CTA or coronary calcium scoring;\n"
            "3. Echocardiography;\n"
            "4. Cardiac MRI as a problem-solving modality when prior "
            "tests are inconclusive or contradictory.\n\n"
            "When prior comprehensive workup (stress test, echo, "
            "coronary CTA) is normal and recent (<6 months), cardiac "
            "MRI is unlikely to add diagnostic value and is NOT "
            "covered absent new symptoms, abnormal findings on the "
            "prior workup, or a specific indication (suspected "
            "myocarditis, infiltrative disease, ARVD)."
        ),
    ),
    21: dict(
        cpt="43644", icd=["E66.01", "E11.9"], facility=FacilityType.INPATIENT,
        policy_excerpt=(
            "NCD 100.1 — Bariatric Surgery\n\n"
            "Roux-en-Y gastric bypass is covered when ALL of:\n"
            "1. BMI >=40, OR BMI 35-39.9 with at least one obesity-"
            "related comorbidity (diabetes, hypertension, sleep apnea, "
            "cardiovascular disease);\n"
            "2. BMI 30-34.9 may qualify ONLY in select MAC jurisdictions "
            "with type 2 diabetes AND extensive supporting documentation;\n"
            "3. Documented participation in a medically supervised "
            "weight-management program for >=6 consecutive months "
            "within the prior 24 months. The 6-month requirement is "
            "absolute — partial documentation (4-5 months) does not "
            "satisfy the criterion;\n"
            "4. Psychological evaluation clearing for surgery;\n"
            "5. Multidisciplinary evaluation (surgeon, dietitian, "
            "behavioral health)."
        ),
    ),
    22: dict(
        cpt="22612", icd=["M51.36"], facility=FacilityType.INPATIENT,
        policy_excerpt=(
            "LCD L34960 — Lumbar Fusion Surgery\n\n"
            "Lumbar fusion is covered for degenerative disc disease "
            "ONLY when ALL of:\n"
            "1. Documented mechanical instability (translation >4mm, "
            "angulation >10 degrees on flexion-extension films), OR "
            "high-grade spondylolisthesis, OR confirmed pseudarthrosis "
            "from prior fusion;\n"
            "2. Severe radiculopathy or neurogenic claudication "
            "correlating with imaging findings;\n"
            "3. Documented failure of >=6 months of comprehensive "
            "conservative care including physical therapy, medications, "
            "and at least one interventional procedure (epidural "
            "injection, facet injection).\n\n"
            "Disc desiccation or degenerative changes on imaging "
            "WITHOUT instability or radiculopathy do NOT meet fusion "
            "criteria. 3 months of PT is insufficient conservative "
            "care for elective fusion."
        ),
    ),
    23: dict(
        cpt="J0517", icd=["L20.9"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "Specialty Drug Coverage: Dupilumab (Dupixent) for Atopic Dermatitis\n\n"
            "Dupilumab is covered for moderate-to-severe atopic dermatitis "
            "when documentation establishes:\n"
            "1. Diagnosis of moderate-to-severe atopic dermatitis "
            "by a dermatologist;\n"
            "2. Adequate trial and failure of high-potency topical "
            "corticosteroids — 'adequate trial' means continuous use "
            "for >=3 months OR documented intolerance/contraindication;\n"
            "3. Adequate trial and failure of systemic immunosuppressant "
            "therapy (cyclosporine, methotrexate, azathioprine, or "
            "phototherapy) — again >=3 months continuous use or "
            "documented contraindication;\n"
            "4. Body surface area involvement >=10% OR head/face/genital "
            "involvement.\n\n"
            "Trial durations of <2-3 weeks do NOT constitute an adequate "
            "trial of any prerequisite therapy."
        ),
    ),
    24: dict(
        cpt="J3490", icd=["E66.9"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "Specialty Drug Coverage: Semaglutide (Ozempic / Wegovy)\n\n"
            "OZEMPIC (semaglutide for type 2 diabetes management):\n"
            "Coverage requires documented type 2 diabetes mellitus "
            "(HbA1c >=6.5% on prior testing OR established diagnosis "
            "with current pharmacotherapy). Patients with HbA1c <6.5% "
            "and no prior diabetes diagnosis do not meet the diabetes "
            "indication.\n\n"
            "WEGOVY (semaglutide for chronic weight management):\n"
            "Coverage requires BMI >=30, OR BMI >=27 with at least one "
            "weight-related comorbidity (hypertension, dyslipidemia, "
            "cardiovascular disease, type 2 diabetes).\n\n"
            "Diagnosis coding integrity: claims billing T2DM (E11.x) "
            "with current HbA1c <6.5% and no prior diabetes "
            "documentation may indicate diagnosis miscoding. Plans "
            "review such claims for fraud, waste, and abuse."
        ),
    ),
    25: dict(
        cpt="E0486", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "Medicare DME MAC Jurisdiction Routing\n\n"
            "Durable medical equipment claims are routed to the "
            "appropriate DME MAC based on the BENEFICIARY'S permanent "
            "residence address, not the supplier's location.\n\n"
            "Jurisdictional assignments:\n"
            "- Jurisdiction A (Noridian, J-A): Northeast US states;\n"
            "- Jurisdiction B (CGS Administrators, J-B): Midwest;\n"
            "- Jurisdiction C (CGS Administrators, J-C): Southeast US "
            "states AND Puerto Rico, US Virgin Islands per LCD L33611;\n"
            "- Jurisdiction D (Noridian, J-D): Western US.\n\n"
            "Claims for Puerto Rico beneficiaries must be submitted to "
            "CGS (J-C). Claims submitted to the wrong MAC jurisdiction "
            "are returned for resubmission. Beneficiary jurisdiction "
            "must be validated before applying LCD criteria."
        ),
    ),
    26: dict(
        cpt="E0486", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "Prior Authorization Timing Requirements for DME\n\n"
            "Prior authorization for durable medical equipment must be "
            "submitted and APPROVED before the date of service. "
            "Retroactive PA is generally NOT permitted for elective "
            "DME items.\n\n"
            "Limited exceptions for retroactive consideration:\n"
            "(a) Documented emergency / urgent medical necessity that "
            "precluded prior submission (must be supported by clinical "
            "documentation of the emergent presentation);\n"
            "(b) Beneficiary eligibility was retroactively established "
            "after service;\n"
            "(c) System or processing failure documented by the "
            "submitting provider.\n\n"
            "Routine elective DME items delivered before the PA "
            "submission date are NOT eligible for retroactive PA. "
            "Convenience and provider scheduling do not constitute "
            "urgent medical necessity."
        ),
    ),
    27: dict(
        cpt="95810", icd=["G47.33", "F51.01"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "Mixed-Coverage Claims Processing\n\n"
            "When a single prior authorization request includes "
            "multiple items or services covered under different LCDs, "
            "each item must be evaluated independently against its "
            "applicable coverage policy. Blanket approval or denial "
            "of the full claim is inappropriate.\n\n"
            "For combined sleep study + behavioral health device claims:\n"
            "- Polysomnography (CPT 95810) is covered under LCD L33405 "
            "for OSA evaluation when criteria are met;\n"
            "- Cognitive behavioral therapy for insomnia (CBT-I) "
            "delivered via dedicated device or app is NOT a covered "
            "Medicare benefit and must be denied separately;\n\n"
            "The reviewer should approve the PSG component if it meets "
            "L33405 criteria and deny the CBT-I device component as "
            "non-covered. The combined claim cannot be approved as a "
            "single item."
        ),
    ),
    28: dict(
        cpt="95810", icd=["G47.33", "G25.81"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33405 — Sleep Studies (Revision History R9, eff. 07/01/2020)\n\n"
            "Covered services:\n"
            "1. Polysomnography (95810) — covered for OSA evaluation;\n"
            "2. Multiple Sleep Latency Test (95805) — covered ONLY when "
            "the indication is suspected narcolepsy or idiopathic "
            "hypersomnia AND preceded by overnight PSG;\n"
            "3. Maintenance of Wakefulness Test (95805) — covered for "
            "occupational fitness-for-duty in safety-sensitive roles.\n\n"
            "REMOVED FROM COVERAGE in R9 revision (effective 07/01/2020):\n"
            "- Actigraphy (95803) is no longer a covered diagnostic "
            "modality for sleep disorders. Prior to 07/01/2020 it was "
            "covered as an adjunct study; the R9 revision removed this "
            "coverage based on insufficient evidence of incremental "
            "diagnostic value.\n\n"
            "Restless leg syndrome alone does not establish PSG or "
            "MSLT necessity."
        ),
    ),
    29: dict(
        cpt="E0601", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "LCD L33718 — Sleep Study Quality Requirements\n\n"
            "Sleep studies submitted to support PAP device coverage "
            "must meet quality and personnel standards:\n"
            "1. The testing facility must be accredited by AASM, ACHC, "
            "or The Joint Commission, OR\n"
            "2. The study must be performed under the direct supervision "
            "of a board-certified sleep medicine physician AND "
            "interpreted by a credentialed sleep specialist;\n"
            "3. Polysomnographic technologists must hold RPSGT "
            "credential or equivalent;\n"
            "4. Documentation of facility accreditation OR personnel "
            "credentials must be included with the PA submission.\n\n"
            "Sleep studies from non-accredited facilities without "
            "documented credentialed personnel do not meet the quality "
            "requirement. Clinical results alone (e.g., AHI value) "
            "are insufficient if the facility/personnel "
            "qualifications are not documented."
        ),
    ),
    30: dict(
        cpt="E0486", icd=["G47.33"], facility=FacilityType.OUTPATIENT,
        policy_excerpt=(
            "Temporal Validity of Diagnostic Studies\n\n"
            "Sleep studies supporting PA submissions for ongoing "
            "therapy must reflect current clinical status. Accepted "
            "study age:\n"
            "- Sleep study within the prior 24 months: generally "
            "accepted without re-evaluation;\n"
            "- Sleep study 24-36 months old: requires brief clinical "
            "re-evaluation confirming continued need;\n"
            "- Sleep study >36 months old: REQUIRES updated sleep "
            "study OR documented clinical re-evaluation establishing "
            "that no material change has occurred.\n\n"
            "Material clinical changes (significant weight gain or "
            "loss >10% body weight, new comorbid conditions, change in "
            "symptoms) invalidate prior studies regardless of age. A "
            "+40 lb weight change is material and requires repeat "
            "diagnostic evaluation before continued therapy is "
            "authorized."
        ),
    ),
}


def _map_outcome(expected: str) -> tuple[Outcome, Optional[DenialReason]]:
    """Map the xlsx 'Expected Correct Determination' string to our 3-class
    Outcome enum + an optional DenialReason for denied cases.

    The xlsx uses freeform strings like "DENY (HSAT not validated for
    pediatrics)" or "FLAG/DENY". This function collapses everything to
    APPROVED / DENIED. The single APPROVE case is the test that the
    agent isn't blindly denying.
    """
    if not expected:
        return Outcome.DENIED, DenialReason.MEDICAL_NECESSITY
    s = expected.strip().upper()
    # Mixed-decision claim: contains both APPROVE and DENY → DENIED.
    # The request as-stated cannot be approved in full because at least
    # one component is denied. The agent's job is to recognize the
    # deny component.
    if "APPROVE" in s and "DENY" in s:
        return Outcome.DENIED, DenialReason.NOT_COVERED
    if s.startswith("APPROVE"):
        if "PARTIAL" in s:
            return Outcome.DENIED, DenialReason.NOT_COVERED
        return Outcome.APPROVED, None
    if "PARTIAL" in s:
        return Outcome.DENIED, DenialReason.NOT_COVERED
    # Everything else (DENY, FLAG, REQUIRES, CONDITIONAL) → DENIED
    if "FREQUENCY" in s or "REPEAT" in s:
        return Outcome.DENIED, DenialReason.FREQUENCY_LIMIT
    if "EXPERIMENTAL" in s or "INVESTIGATIONAL" in s:
        return Outcome.DENIED, DenialReason.EXPERIMENTAL
    if "MISSING" in s or "REQUEST MORE" in s or "DOCUMENTATION" in s:
        return Outcome.DENIED, DenialReason.MISSING_INFO
    if "NOT COVERED" in s or "NOT VALIDATED" in s or "PROHIBITED" in s:
        return Outcome.DENIED, DenialReason.NOT_COVERED
    if "FLAG" in s or "CODE" in s or "MISMATCH" in s or "RECODE" in s:
        return Outcome.DENIED, DenialReason.CODING_ERROR
    return Outcome.DENIED, DenialReason.MEDICAL_NECESSITY


def build_lcd_edge_cases(raw_path: Path | None = None) -> list[BenchmarkCase]:
    """Read the user's xlsx-derived JSON and emit BenchmarkCase objects.

    Args:
        raw_path: Path to edge_cases_raw.json. Defaults to the path
            shipped with the repo (mvp_dashboard/edge_cases_raw.json).
    """
    p = raw_path or _RAW_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"LCD edge cases JSON not found at {p}. Did you run the xlsx "
            f"conversion step?"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))

    cases: list[BenchmarkCase] = []
    for row in raw:
        num = int(row["num"])
        override = _CASE_OVERRIDES.get(num)
        if override is None:
            # Skip cases without an override — every case in the v0.2
            # fixture must be hand-mapped to a CPT/ICD/policy excerpt
            # that the agent can actually reason about.
            continue

        outcome, denial = _map_outcome(row["expected"])
        case = BenchmarkCase(
            case_id=f"lcd_{num:03d}",
            task_config=PriorAuthTaskConfig(
                payer=Payer.MEDICARE,
                cpt_code=override["cpt"],
                icd10_codes=list(override["icd"]),
                facility_type=override["facility"],
                denial_reason=denial,
            ),
            clinical_notes=row["scenario"],
            policy_excerpt=override["policy_excerpt"],
            ground_truth_outcome=outcome,
            ground_truth_reasoning=[
                f"Category: {row['category']}",
                f"Key complexity: {row['complexity']}",
                f"Trick clause: {row['tricky']}",
                f"Correct determination: {row['expected']}",
            ],
            difficulty=Difficulty.HARD,
        )
        cases.append(case)
    return cases
