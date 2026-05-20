"""Case builder — turns an LCDPolicy into a deterministic set of
BenchmarkCases with policy-grounded ground truth.

Design constraints:
1. Deterministic. Same (policy, target_count, seed) → byte-identical output.
   The manifest SHA256 must be stable across re-runs so the leak guard can
   enforce frozen test fingerprints.
2. Ground-truth outcome is DERIVED from which LCD criteria the case's
   generated clinical notes satisfy, NOT injected by the author. Each
   template explicitly declares the satisfaction pattern.
3. Clinical notes are synthetic but policy-faithful — they cite the exact
   criterion the template is testing. Phase 6 will replace with MIMIC-IV
   joins where available; the interface (returns list[BenchmarkCase]) does
   not change.
4. `ground_truth_reasoning` on each case is a list of short strings
   referencing LCD criterion ids. The pre-rework `_extract_local` extractor
   used this field as an oracle leak; the new pipeline never reads it.
   It is kept purely as a human-audit signal on the benchmark dataset card.

This module MUST NOT import anything from medreason.agent / injector /
extractor — those are the pre-rework modules and will be archived in
Phase 5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
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

from .schemas import LCDPolicy


# ── Case templates ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CaseTemplate:
    """One abstract case shape.

    The `notes_template` format placeholders are filled from a deterministic
    per-case parameter dict (age, sex, icd, cpt_code, duration). The
    ground_truth_reasoning list is emitted verbatim, with LCD citations
    inserted via str.format() from the policy's document_id.
    """
    name: str
    outcome: Outcome
    difficulty: Difficulty
    denial_reason: Optional[DenialReason]
    applied_criterion_ids: tuple[str, ...]   # criteria the case turns on
    notes_template: str
    reasoning_template: tuple[str, ...]


_TEMPLATES: tuple[_CaseTemplate, ...] = (
    # ── APPROVED (4) ────────────────────────────────────────────────────
    _CaseTemplate(
        name="APPROVED_meets_all",
        outcome=Outcome.APPROVED,
        difficulty=Difficulty.EASY,
        denial_reason=None,
        applied_criterion_ids=("C.1", "C.2", "C.3", "C.4"),
        notes_template=(
            "{age} y/o {sex} with chronic low back pain radiating down the "
            "{leg} leg. Completed {weeks} weeks of supervised physical "
            "therapy including core strengthening and lumbar stabilization, "
            "plus a course of NSAIDs, without meaningful improvement. Exam "
            "reveals positive straight-leg raise at 30° on the {leg}, 4/5 "
            "weakness of the {toe}-toe extensor, diminished Achilles reflex, "
            "and {dermatome} dermatomal sensory loss. Ordered by "
            "neurosurgery for pre-operative planning of an L4-L5 "
            "decompression."
        ),
        reasoning_template=(
            "C.1 met: {weeks} weeks documented PT + NSAIDs with no improvement.",
            "C.2 met: motor deficit and reflex change on exam.",
            "C.3 met: radicular pattern correlates with L4-L5 distribution.",
            "C.4 met: ordered for surgical planning.",
            "All coverage criteria satisfied → APPROVED.",
        ),
    ),
    _CaseTemplate(
        name="APPROVED_post_surgical",
        outcome=Outcome.APPROVED,
        difficulty=Difficulty.MEDIUM,
        denial_reason=None,
        applied_criterion_ids=("C.2", "C.3", "C.4", "L.1"),
        notes_template=(
            "{age} y/o {sex} status-post L4-L5 laminectomy 3 months ago with "
            "initial improvement, now reporting new onset of {leg}-sided "
            "radicular pain and progressive foot drop over 2 weeks. Exam "
            "shows 3/5 tibialis anterior weakness on the {leg} and new "
            "{dermatome} sensory loss. MRI ordered by operating neurosurgeon "
            "to evaluate for residual disc, recurrent herniation, or "
            "post-operative hematoma."
        ),
        reasoning_template=(
            "L.1 exception applies: post-operative evaluation with new "
            "neurologic deficit.",
            "C.2 met: progressive 3/5 weakness and new sensory loss.",
            "C.3 met: findings localize to operated level.",
            "C.4 met: ordered to guide surgical decision.",
            "L.1 exception satisfied → APPROVED.",
        ),
    ),
    _CaseTemplate(
        name="APPROVED_progressive_deficit",
        outcome=Outcome.APPROVED,
        difficulty=Difficulty.HARD,
        denial_reason=None,
        applied_criterion_ids=("C.2", "C.3", "C.4"),
        notes_template=(
            "{age} y/o {sex} with {weeks}-week history of low back pain now "
            "reporting rapidly progressive {leg}-sided weakness. Had been "
            "attempting home exercise but did not complete formal PT because "
            "symptoms escalated. Exam: 2/5 extensor hallucis longus weakness, "
            "foot slap gait, hyperreflexia at the knee, clonus absent. "
            "Progressive deficit documented on two visits separated by 4 "
            "days. Ordered for urgent surgical consultation."
        ),
        reasoning_template=(
            "C.1 technically unmet: formal 6-week conservative trial not "
            "completed.",
            "Progressive motor deficit provides clinical override of C.1 "
            "per standard medical necessity.",
            "C.2 strongly met: rapidly worsening motor findings.",
            "C.3 met: level localizes clinically.",
            "C.4 met: ordered for urgent surgical decision.",
            "Progressive deficit overrides conservative trial → APPROVED.",
        ),
    ),
    _CaseTemplate(
        name="APPROVED_cauda_warning",
        outcome=Outcome.APPROVED,
        difficulty=Difficulty.MEDIUM,
        denial_reason=None,
        applied_criterion_ids=("C.2", "C.3", "C.4"),
        notes_template=(
            "{age} y/o {sex} presenting with sudden severe low back pain 48 "
            "hours ago and new saddle-distribution numbness. No urinary "
            "retention yet. No bowel incontinence. Exam: reduced perianal "
            "sensation, preserved anal tone, bilateral {leg} L5-S1 sensory "
            "changes. PT not attempted given emergent presentation. Ordered "
            "to rule out cauda equina syndrome."
        ),
        reasoning_template=(
            "C.1 not required: emergent indication bypasses conservative "
            "trial.",
            "C.2 met: perianal sensory deficit raises cauda equina concern.",
            "C.3 met: findings correlate with lumbosacral distribution.",
            "C.4 met: rule-out surgical emergency.",
            "Emergent indication → APPROVED.",
        ),
    ),
    # ── DENIED (4) ──────────────────────────────────────────────────────
    _CaseTemplate(
        name="DENIED_insufficient_pt",
        outcome=Outcome.DENIED,
        difficulty=Difficulty.EASY,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        applied_criterion_ids=("C.1",),
        notes_template=(
            "{age} y/o {sex} with {weeks} weeks of low back pain. Has tried "
            "over-the-counter ibuprofen and a single visit to physical "
            "therapy. No formal PT course completed. Exam is nonfocal: full "
            "strength, intact reflexes, negative straight-leg raise. Patient "
            "requests MRI to see 'what's causing the pain.'"
        ),
        reasoning_template=(
            "C.1 unmet: only {weeks} weeks of informal conservative therapy, "
            "no completed PT course.",
            "C.2 unmet: exam is nonfocal, no neurological deficit.",
            "Criteria not satisfied → DENIED (medical necessity).",
        ),
    ),
    _CaseTemplate(
        name="DENIED_no_neuro",
        outcome=Outcome.DENIED,
        difficulty=Difficulty.EASY,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        applied_criterion_ids=("C.2", "C.3"),
        notes_template=(
            "{age} y/o {sex} with chronic axial low back pain for "
            "{weeks} months. Completed 8 weeks of formal PT and a course of "
            "meloxicam with minimal change. Pain is axial only, no radiation "
            "below the knee. Exam: full motor strength throughout, symmetric "
            "reflexes, negative straight-leg raise, no dermatomal sensory "
            "loss. No red-flag symptoms."
        ),
        reasoning_template=(
            "C.1 met: completed formal 8-week PT and NSAIDs.",
            "C.2 unmet: no motor, reflex, or sensory deficit on exam.",
            "C.3 unmet: pain pattern does not correlate with radiculopathy.",
            "Neurological correlation absent → DENIED (medical necessity).",
        ),
    ),
    _CaseTemplate(
        name="DENIED_repeat_mri",
        outcome=Outcome.DENIED,
        difficulty=Difficulty.MEDIUM,
        denial_reason=DenialReason.FREQUENCY_LIMIT,
        applied_criterion_ids=("L.1",),
        notes_template=(
            "{age} y/o {sex} with persistent low back pain. Had lumbar MRI "
            "6 months ago showing multilevel degenerative changes without "
            "focal herniation or stenosis. Currently on chronic PT. No new "
            "neurological deficit since prior imaging. No interval surgery. "
            "Provider is requesting repeat MRI because 'the pain hasn't "
            "gotten better.'"
        ),
        reasoning_template=(
            "L.1 triggered: repeat MRI within 12 months without new "
            "neurologic deficit, clinical change, or post-operative context.",
            "No new deficit documented.",
            "Routine surveillance excluded → DENIED (frequency limit).",
        ),
    ),
    _CaseTemplate(
        name="DENIED_screening",
        outcome=Outcome.DENIED,
        difficulty=Difficulty.EASY,
        denial_reason=DenialReason.NOT_COVERED,
        applied_criterion_ids=("L.2",),
        notes_template=(
            "{age} y/o {sex} seen for annual physical. Mentions occasional "
            "lower back stiffness that resolves with rest. No functional "
            "limitation. Exam is normal. Requesting lumbar MRI 'to check "
            "for problems before they start.' No objective findings. No "
            "conservative therapy attempted or indicated."
        ),
        reasoning_template=(
            "L.2 triggered: routine screening for nonspecific back pain "
            "without neurological findings is not a covered indication.",
            "No neurological deficit.",
            "Screening exclusion applies → DENIED (not covered).",
        ),
    ),
    # ── OVERTURNED ON APPEAL (2) ────────────────────────────────────────
    _CaseTemplate(
        name="OVERTURNED_borderline_conservative",
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        difficulty=Difficulty.HARD,
        denial_reason=DenialReason.MEDICAL_NECESSITY,
        applied_criterion_ids=("C.1", "C.2", "C.3"),
        notes_template=(
            "{age} y/o {sex} with 4 weeks of formal PT plus 3 additional "
            "weeks of home exercise program supervised by a PT (total ~7 "
            "weeks conservative but only 4 weeks formally documented as "
            "billed visits). Developed progressive L5 radiculopathy with "
            "new 4/5 EHL weakness and {dermatome} sensory loss during PT "
            "course. Initial request denied citing insufficient formal PT "
            "duration (C.1). Appeal submitted documenting the home exercise "
            "component and the progressive nature of the deficit."
        ),
        reasoning_template=(
            "C.1 borderline: formal PT 4 weeks; total conservative trial "
            "~7 weeks with supervised home exercise component.",
            "Initial denial cited C.1 strictly.",
            "On appeal: progressive neurological deficit during the trial "
            "justifies overriding the strict 6-week counter.",
            "C.2 and C.3 clearly met on appeal review.",
            "Appeal panel reversed the denial → OVERTURNED ON APPEAL.",
        ),
    ),
    _CaseTemplate(
        name="OVERTURNED_missed_red_flag",
        outcome=Outcome.OVERTURNED_ON_APPEAL,
        difficulty=Difficulty.HARD,
        denial_reason=DenialReason.MISSING_INFO,
        applied_criterion_ids=("C.2", "C.3"),
        notes_template=(
            "{age} y/o {sex} with 5-day history of severe low back pain and "
            "new urinary hesitancy. Initial submission from primary care "
            "noted only 'low back pain, requests MRI.' Payer denied for "
            "missing clinical information. On resubmission with full exam "
            "documentation — reduced perianal sensation, post-void residual "
            "300 mL, reduced anal tone — the appeal was granted citing "
            "concern for cauda equina syndrome."
        ),
        reasoning_template=(
            "Initial denial: MISSING_INFO (inadequate exam documentation).",
            "On appeal: full documentation shows perianal sensory loss, "
            "urinary retention, reduced anal tone.",
            "C.2 clearly met on appeal: red-flag cauda equina signs.",
            "C.3 met: findings localize to sacral roots.",
            "Appeal panel reversed the denial → OVERTURNED ON APPEAL.",
        ),
    ),
)


# ── Parameter pools (for deterministic variant generation) ──────────────────

_AGES: tuple[int, ...] = (38, 44, 51, 56, 58, 62, 65, 68, 72, 77)
_SEXES: tuple[str, ...] = ("male", "female")
_LEGS: tuple[str, ...] = ("left", "right")
_TOES: tuple[str, ...] = ("great", "second", "third")
_DERMATOMES: tuple[str, ...] = ("L4", "L5", "S1")
_WEEKS_APPROVED: tuple[int, ...] = (6, 8, 10, 12)
_WEEKS_DENIED_PT: tuple[int, ...] = (1, 2, 3)
_WEEKS_DENIED_NO_NEURO: tuple[int, ...] = (3, 4, 6, 9)
_WEEKS_PROGRESSIVE: tuple[int, ...] = (2, 3, 4)


def _pick(rng: random.Random, pool: tuple):
    """Deterministic pick that actually consumes RNG state.

    Callers must instantiate a fresh `random.Random` per case with a
    seed derived from both the user-provided seed and the case index,
    so (a) changing `seed` changes parameter choices, and (b) the
    overall output remains deterministic for any fixed `seed`.
    """
    return rng.choice(pool)


def _default_icd_for_template(template_name: str, covered_icd10: list[str]) -> str:
    """Choose a semantically sensible ICD-10 for a given template.

    We prefer radiculopathy codes (M54.16/M54.17) for templates with
    radicular findings, and M54.5 for nonspecific low back pain.
    Falls back to the first covered ICD if nothing matches.
    """
    has_rad = any(icd in covered_icd10 for icd in ("M54.16", "M54.17"))
    has_disc = any(icd in covered_icd10 for icd in ("M51.16", "M51.17"))
    has_nonspecific = "M54.5" in covered_icd10

    if template_name in (
        "APPROVED_meets_all",
        "APPROVED_post_surgical",
        "APPROVED_progressive_deficit",
        "OVERTURNED_borderline_conservative",
    ):
        if has_disc:
            return "M51.16"
        if has_rad:
            return "M54.16"
    if template_name == "APPROVED_cauda_warning":
        if has_rad:
            return "M54.17"
    if template_name in ("DENIED_no_neuro", "DENIED_screening"):
        if has_nonspecific:
            return "M54.5"
    if template_name == "DENIED_insufficient_pt":
        if has_nonspecific:
            return "M54.5"
    if template_name == "DENIED_repeat_mri":
        if has_disc:
            return "M51.16"
    if template_name == "OVERTURNED_missed_red_flag":
        if has_rad:
            return "M54.17"
    return covered_icd10[0] if covered_icd10 else "M54.5"


# ── Public API ───────────────────────────────────────────────────────────────


def build_cases_from_lcd(
    policy: LCDPolicy,
    *,
    target_count: int = 50,
    seed: int = 42,
    payer: Payer = Payer.MEDICARE,
    facility_type: FacilityType = FacilityType.OUTPATIENT,
) -> list[BenchmarkCase]:
    """Generate a deterministic set of BenchmarkCases grounded in `policy`.

    The output is sorted by case_id and is byte-identical across re-runs
    with the same (policy, target_count, seed). This is enforced by:
    - Templates in a fixed tuple (_TEMPLATES).
    - CPTs read from policy.cpt_codes in the order they appear in the XML.
    - Slot ordering = (template_index, cpt_index, variant_index).
    - Parameters picked via `_pick(rng, pool, index)` on an index derived
      from the slot number.

    Args:
        policy: Parsed LCDPolicy from cms_lcd_ncd.parse_lcd_xml().
        target_count: Exact number of cases to produce.
        seed: Reproducibility seed. Changing it changes parameter choices
            (age, leg, dermatome) but not the (template, cpt) slot order.
        payer: Payer tag for every generated case. Defaults to Medicare
            because LCDs are Medicare coverage documents. Phase 6 will
            distribute across multiple payers as more sources come online.
        facility_type: Facility tag for every case.
    """
    if target_count <= 0:
        return []
    if not policy.cpt_codes:
        raise ValueError(
            f"LCDPolicy {policy.document_id} has no cpt_codes; cannot build cases"
        )

    cpts = list(policy.cpt_codes)
    templates = list(_TEMPLATES)

    # Deterministic policy excerpt shared across cases — the full set of
    # criteria + limitations, each prefixed by their section id.
    excerpt_parts = [f"LCD {policy.document_id} — {policy.title}"]
    for crit in policy.indications:
        excerpt_parts.append(f"§{crit.criterion_id}: {crit.text}")
    for lim in policy.limitations:
        excerpt_parts.append(f"§{lim.limitation_id}: {lim.text}")
    full_policy_excerpt = "\n".join(excerpt_parts)

    cases: list[BenchmarkCase] = []
    base_slot_count = len(templates) * len(cpts)

    for i in range(target_count):
        slot_idx = i % base_slot_count
        template_idx = slot_idx // len(cpts)
        cpt_idx = slot_idx % len(cpts)
        template = templates[template_idx]
        cpt_code = cpts[cpt_idx]

        # Per-case deterministic RNG: seed + (i * 2654435761) mixes bits
        # so adjacent cases don't share their first pick. The fresh
        # Random() per iteration means changing the outer `seed` changes
        # every case's parameters, while the (template, cpt) slot order
        # remains intact so the stratifier's behavior is stable.
        rng = random.Random(seed ^ (i * 2654435761))

        # Per-case parameters. Order of _pick() calls is load-bearing:
        # it consumes RNG state. Don't reorder without updating the
        # deterministic fingerprints in the v0.0 manifest.
        age = _pick(rng, _AGES)
        sex = _pick(rng, _SEXES)
        leg = _pick(rng, _LEGS)
        toe = _pick(rng, _TOES)
        dermatome = _pick(rng, _DERMATOMES)

        if template.name == "APPROVED_meets_all":
            weeks = _pick(rng, _WEEKS_APPROVED)
        elif template.name in (
            "APPROVED_progressive_deficit",
            "OVERTURNED_borderline_conservative",
        ):
            weeks = _pick(rng, _WEEKS_PROGRESSIVE)
        elif template.name == "DENIED_insufficient_pt":
            weeks = _pick(rng, _WEEKS_DENIED_PT)
        elif template.name == "DENIED_no_neuro":
            weeks = _pick(rng, _WEEKS_DENIED_NO_NEURO)
        else:
            weeks = 0  # unused by template

        params = {
            "age": age,
            "sex": sex,
            "leg": leg,
            "toe": toe,
            "dermatome": dermatome,
            "weeks": weeks,
        }
        clinical_notes = template.notes_template.format(**params)
        reasoning = [step.format(**params) for step in template.reasoning_template]

        icd = _default_icd_for_template(template.name, list(policy.covered_icd10))
        task_config = PriorAuthTaskConfig(
            payer=payer,
            cpt_code=cpt_code,
            icd10_codes=[icd],
            modifiers=[],
            facility_type=facility_type,
            denial_reason=template.denial_reason,
        )

        case_id = f"case_{i + 1:04d}"
        cases.append(BenchmarkCase(
            case_id=case_id,
            task_config=task_config,
            clinical_notes=clinical_notes,
            prior_eobs=[],
            policy_excerpt=full_policy_excerpt,
            ground_truth_outcome=template.outcome,
            ground_truth_reasoning=reasoning,
            difficulty=template.difficulty,
        ))

    return cases
