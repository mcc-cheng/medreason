# Aetna Lumbar MRI — 20 Prior Auth Test Cases

User-provided fixture for the within-domain generalization test (Section 9
option C of SESSION_BRIEF.md). All 20 cases are lumbar MRI prior auths
under Aetna CPB-0236 / CPB-0093, designed to test whether rules learned
from one Aetna lumbar MRI case generalize to another within the same
payer + condition.

Authored by user, dropped 2026-04-11.

Verdict distribution: 8 APPROVE, 6 DENY, 6 PEND/EDGE.

Cases 13/14 are an intentional documentation-completeness pair on the
same patient — useful for testing whether the system detects the delta
in documentation that flips an outcome.

---

## Clean Approvals (8)

### Case 1 — Textbook Radiculopathy
58F, right L4-L5 radiculopathy, foot drop, 6+ weeks of failed PT and
NSAIDs. Complete documentation: dermatomal pain map, reflex changes, PT
discharge summary, NSAID Rx history.
→ **APPROVE.** Meets the gold-standard conjunctive criteria verbatim.

### Case 4 — Rapidly Progressing Neurological Deficit
71F, bilateral leg weakness progressing to gait difficulty over 10 days.
Serial ED neuro exams showing strength decline.
→ **APPROVE.** "Rapidly progressing neurological deficit" is an exception
that explicitly bypasses the 6-week conservative therapy requirement.

### Case 6 — Suspected Cauda Equina Syndrome
55M, acute saddle anesthesia, urinary retention, bilateral motor weakness.
ED note documenting all three cardinal signs.
→ **APPROVE (emergent).** Cauda equina is an independent standalone
criterion — no conservative therapy required, should be expedited.

### Case 8 — Vertebral Metastases
68F, Stage IV breast cancer, new mid-back pain, bone scan showing L2
uptake. Oncology note present.
→ **APPROVE.** Suspected vertebral/paraspinal/intraspinal metastases is
an independent criterion, no wait period.

### Case 11 — Post-Surgical New Radiculopathy
61M, 8 months post-discectomy, new left S1 radiculopathy with diminished
Achilles reflex. Physician note with objective findings.
→ **APPROVE.** New clinical presentation creates a fresh medically
necessary indication; prior surgical history strengthens the recurrent
disc suspicion.

### Case 14 — Open MRI: With Documented Failed Closed-Bore Attempt
74F, claustrophobia, documented failed closed-bore MRI attempt with
sedation and psychiatry confirmation.
→ **APPROVE.** CPB-0093 requires documented contraindication, not just
patient preference. (Paired with Case 13.)

### Case 16 — Congenital Scoliosis Pre-Op
17F, congenital scoliosis, Cobb angle 52° and progressing on serial
X-rays, MRI requested to rule out syrinx before fusion.
→ **APPROVE.** Congenital anomalies/deformities are a standalone
criterion with no conservative therapy prerequisite.

### Case 19 — Epidural Lipomatosis
61M on chronic systemic steroids for Crohn's, new bilateral leg weakness.
Neurology note explicitly citing clinical suspicion of lumbar epidural
lipomatosis.
→ **APPROVE.** "Diagnosis and evaluation of lumbar epidural lipomatosis"
is a rare but explicitly enumerated standalone criterion in CPB-0236.

---

## Clean Denials (6)

### Case 2 — Acute LBP, No Red Flags
44M, 3-day acute LBP after lifting. No radiculopathy, no neurological
deficits, zero prior treatment.
→ **DENY.** Aetna explicitly cites AHCPR guidelines recommending against
routine imaging for acute LBP without red flags.

### Case 5 — Subjective Radiculopathy Only
49F, 8 weeks LBP, patient-reported radiating pain but neurological exam
fully normal — intact reflexes, full strength.
→ **DENY.** Policy requires radiculopathy evidenced by pain plus objective
motor or reflex changes — both prongs are conjunctive.

### Case 9 — Positional/Weight-Bearing MRI (EDS)
38F with Ehlers-Danlos Syndrome requesting upright weight-bearing MRI.
→ **DENY.** CPB-0093 explicitly names EDS in its experimental clause for
positional MRI; this is a hard designation with no exception pathway.

### Case 10 — Asymptomatic Post-Op Surveillance
61M, 6 months post-L4-L5 discectomy, asymptomatic. Surgeon orders
"routine follow-up" MRI.
→ **DENY.** Routine asymptomatic surveillance meets no listed criteria
in CPB-0236; new symptoms would reopen eligibility.

### Case 13 — Open MRI Without Documented Failed Closed-Bore Attempt
74F, claustrophobia. Open MRI ordered without documentation of failed
closed-bore attempt.
→ **DENY.** CPB-0093 requires documented contraindication, not just
patient preference. (Paired with Case 14.)

### Case 20 — BoneMRI for Surgical Planning
69M scheduled for L3-L5 fusion, surgeon requests BoneMRI (MRI-based
synthetic CT) citing lower radiation vs. standard CT.
→ **DENY.** CPB-0236 explicitly classifies BoneMRI as
experimental/investigational for spinal pre-operative assessment and
surgical planning — appeals are unlikely to succeed absent a national
coverage change.

---

## Edge Cases & Pends (6)

### Case 3 — One Week Short of 6-Week Threshold
62M, LBP with bilateral L5 radiculopathy, PT at week 5 still ongoing.
→ **DENY (appeal likely).** Technically one week short. If physician
documents progressing motor weakness in parallel, the "rapidly progressing
deficit" clause could override — but requires explicit attestation.

### Case 7 — OON Conservative Therapy, No Records
67M claims 8 weeks of PT at out-of-network provider; no records available.
→ **PEND/DENY.** Patient self-report without clinical documentation is
insufficient. Appeals require actual PT records from the OON provider.

### Case 12 — Hospitalization Criterion, Documentation Gap
52F admitted for 10/10 severe LBP requiring IV opioids. MRI ordered but
the authorization request doesn't explicitly cite "hospitalization" as
the triggering criterion.
→ **PEND.** The clinical scenario qualifies under "severe back pain
requiring hospitalization," but the PA form must explicitly link the MRI
order to that criterion.

### Case 15 — Stenosis Asserted, Not Evidenced
66M, "clinical evidence of spinal stenosis" checked on PA form, but no
formal exam findings (no neurogenic claudication documented, no ABI, no
tandem gait).
→ **PEND/DENY.** Checking a box is not sufficient — reviewers expect
documented clinical findings substantiating the assertion.

### Case 17 — Workers' Comp COB Conflict
39M, lumbar disc herniation from workplace injury, both active WC and
commercial Aetna coverage. PT records are from the WC insurer. Medical
necessity is met, but WC is primary payer.
→ **PEND (COB review).** Aetna will not adjudicate authorization
liability until the primary payer designation is resolved.

### Case 18 — Subjectively Progressive Symptoms
53F, physician notes "per patient, symptoms worsening" after 10 weeks of
PT — but no serial objective exam findings or tracked VAS scores.
→ **PEND/DENY.** "Progressively severe symptoms" requires clinical
substantiation; a single chart note quoting the patient fails;
peer-to-peer review is the recommended path.

---

## Verdict Distribution

| Verdict | Cases |
|---|---|
| APPROVE | 1, 4, 6, 8, 11, 14, 16, 19 (8 cases) |
| DENY | 2, 5, 9, 10, 13, 20 (6 cases) |
| PEND / Edge | 3, 7, 12, 15, 17, 18 (6 cases) |

Cases 13/14 are intentionally paired as a documentation-completeness
contrast for the same patient — useful for testing whether the system
can detect the delta in documentation that flips an outcome.
