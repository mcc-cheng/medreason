"""CPT and ICD-10 taxonomy used by retrieval Tier 1 (ontology lookup).

These coarse families are deliberately not exhaustive — they are the buckets
the hybrid retriever filters on before dense embedding + reranking. Adding
more families is fine; renaming is a breaking change because rule triggers
store these enum values.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


# ── CPT families ──────────────────────────────────────────────────────────────


class CPTFamily(str, Enum):
    EM_OUTPATIENT      = "em_outpatient"
    EM_INPATIENT       = "em_inpatient"
    IMAGING_MRI        = "imaging_mri"
    IMAGING_CT         = "imaging_ct"
    IMAGING_OTHER      = "imaging_other"
    SURGERY_ORTHO      = "surgery_ortho"
    SURGERY_GENERAL    = "surgery_general"
    GI_ENDOSCOPY       = "gi_endoscopy"
    PAIN_INJECTION     = "pain_injection"
    PSYCHOTHERAPY      = "psychotherapy"
    PT_OT              = "pt_ot"
    ONCOLOGY_RADIATION = "oncology_radiation"
    LAB_PATHOLOGY      = "lab_pathology"
    OTHER              = "other"


# Numeric ranges (inclusive). First match wins, so order matters: more specific
# ranges should come before broader ones in the same numeric neighborhood.
_CPT_RANGES: list[tuple[int, int, CPTFamily]] = [
    # E/M
    (99201, 99215, CPTFamily.EM_OUTPATIENT),
    (99221, 99239, CPTFamily.EM_INPATIENT),
    (99241, 99255, CPTFamily.EM_OUTPATIENT),  # consults

    # Pain / joint injections (must come before SURGERY_ORTHO 20000-29999)
    (20600, 20611, CPTFamily.PAIN_INJECTION),
    (64400, 64999, CPTFamily.PAIN_INJECTION),

    # Orthopedic surgery
    (23000, 25999, CPTFamily.SURGERY_ORTHO),
    (27000, 27899, CPTFamily.SURGERY_ORTHO),
    (29800, 29999, CPTFamily.SURGERY_ORTHO),

    # General surgery (very rough — covers gen surg + integumentary excisions)
    (10000, 19999, CPTFamily.SURGERY_GENERAL),

    # Imaging — MRI
    (70551, 70559, CPTFamily.IMAGING_MRI),
    (71550, 71552, CPTFamily.IMAGING_MRI),
    (72141, 72158, CPTFamily.IMAGING_MRI),
    (73218, 73225, CPTFamily.IMAGING_MRI),
    (73718, 73725, CPTFamily.IMAGING_MRI),
    (74181, 74183, CPTFamily.IMAGING_MRI),
    (75557, 75565, CPTFamily.IMAGING_MRI),
    (77046, 77049, CPTFamily.IMAGING_MRI),

    # Imaging — CT
    (70450, 70498, CPTFamily.IMAGING_CT),
    (71250, 71275, CPTFamily.IMAGING_CT),
    (72125, 72133, CPTFamily.IMAGING_CT),
    (73200, 73206, CPTFamily.IMAGING_CT),
    (73700, 73706, CPTFamily.IMAGING_CT),
    (74150, 74178, CPTFamily.IMAGING_CT),

    # Imaging — other (X-ray, US, nuclear)
    (70010, 76499, CPTFamily.IMAGING_OTHER),
    (76500, 76999, CPTFamily.IMAGING_OTHER),
    (78000, 79999, CPTFamily.IMAGING_OTHER),

    # GI endoscopy
    (43200, 43289, CPTFamily.GI_ENDOSCOPY),
    (44360, 44408, CPTFamily.GI_ENDOSCOPY),
    (45300, 45398, CPTFamily.GI_ENDOSCOPY),

    # Oncology radiation
    (77261, 77799, CPTFamily.ONCOLOGY_RADIATION),

    # PT / OT
    (97001, 97799, CPTFamily.PT_OT),

    # Psychiatry
    (90785, 90899, CPTFamily.PSYCHOTHERAPY),

    # Lab / pathology
    (80047, 89398, CPTFamily.LAB_PATHOLOGY),
]


def cpt_family(cpt: str) -> CPTFamily:
    """Map a CPT code string to a family.

    Returns CPTFamily.OTHER for unrecognized codes (Category II/III codes,
    HCPCS letter-prefixed codes, malformed input). Never raises.
    """
    if not cpt:
        return CPTFamily.OTHER
    digits = "".join(c for c in cpt if c.isdigit())
    if not digits or len(digits) < 5:
        return CPTFamily.OTHER
    try:
        n = int(digits[:5])
    except ValueError:
        return CPTFamily.OTHER
    for lo, hi, fam in _CPT_RANGES:
        if lo <= n <= hi:
            return fam
    return CPTFamily.OTHER


def cpt_families(cpts: Iterable[str]) -> list[CPTFamily]:
    """Map a list of CPTs to deduped families, preserving first-seen order."""
    out: list[CPTFamily] = []
    seen: set[CPTFamily] = set()
    for c in cpts:
        f = cpt_family(c)
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ── ICD-10 chapters ───────────────────────────────────────────────────────────


class ICD10Chapter(str, Enum):
    INFECTIOUS         = "A00-B99"
    NEOPLASMS          = "C00-D49"
    BLOOD              = "D50-D89"
    ENDOCRINE          = "E00-E89"
    MENTAL             = "F01-F99"
    NERVOUS            = "G00-G99"
    EYE                = "H00-H59"
    EAR                = "H60-H95"
    CIRCULATORY        = "I00-I99"
    RESPIRATORY        = "J00-J99"
    DIGESTIVE          = "K00-K95"
    SKIN               = "L00-L99"
    MUSCULOSKELETAL    = "M00-M99"
    GENITOURINARY      = "N00-N99"
    PREGNANCY          = "O00-O9A"
    PERINATAL          = "P00-P96"
    CONGENITAL         = "Q00-Q99"
    SYMPTOMS           = "R00-R99"
    INJURY             = "S00-T88"
    EXTERNAL           = "V00-Y99"
    HEALTH_STATUS      = "Z00-Z99"
    UNKNOWN            = "??"


def icd10_chapter(icd: str) -> ICD10Chapter:
    """Map an ICD-10-CM code to its chapter.

    Handles the letter+number boundaries that don't fall on whole letters:
    - C/D split: C00-D49 = neoplasms, D50-D89 = blood
    - H split:   H00-H59 = eye,        H60-H95 = ear
    - S/T:       S00-T88 = injury
    - O0-O9A:    pregnancy block ends mid-letter

    Unknown / malformed input returns ICD10Chapter.UNKNOWN. Never raises.
    """
    if not icd:
        return ICD10Chapter.UNKNOWN
    code = icd.strip().upper()
    if len(code) < 3:
        return ICD10Chapter.UNKNOWN
    letter = code[0]
    # Numeric portion is the first 2 digits after the letter
    try:
        num = int(code[1:3])
    except ValueError:
        return ICD10Chapter.UNKNOWN

    if letter in ("A", "B"):
        return ICD10Chapter.INFECTIOUS
    if letter == "C":
        return ICD10Chapter.NEOPLASMS
    if letter == "D":
        return ICD10Chapter.NEOPLASMS if num <= 49 else ICD10Chapter.BLOOD
    if letter == "E":
        return ICD10Chapter.ENDOCRINE
    if letter == "F":
        return ICD10Chapter.MENTAL
    if letter == "G":
        return ICD10Chapter.NERVOUS
    if letter == "H":
        return ICD10Chapter.EYE if num <= 59 else ICD10Chapter.EAR
    if letter == "I":
        return ICD10Chapter.CIRCULATORY
    if letter == "J":
        return ICD10Chapter.RESPIRATORY
    if letter == "K":
        return ICD10Chapter.DIGESTIVE
    if letter == "L":
        return ICD10Chapter.SKIN
    if letter == "M":
        return ICD10Chapter.MUSCULOSKELETAL
    if letter == "N":
        return ICD10Chapter.GENITOURINARY
    if letter == "O":
        return ICD10Chapter.PREGNANCY
    if letter == "P":
        return ICD10Chapter.PERINATAL
    if letter == "Q":
        return ICD10Chapter.CONGENITAL
    if letter == "R":
        return ICD10Chapter.SYMPTOMS
    if letter in ("S", "T"):
        return ICD10Chapter.INJURY
    if letter in ("V", "W", "X", "Y"):
        return ICD10Chapter.EXTERNAL
    if letter == "Z":
        return ICD10Chapter.HEALTH_STATUS
    return ICD10Chapter.UNKNOWN


def icd10_chapters(icds: Iterable[str]) -> list[ICD10Chapter]:
    """Map a list of ICD-10 codes to deduped chapters, preserving order."""
    out: list[ICD10Chapter] = []
    seen: set[ICD10Chapter] = set()
    for c in icds:
        ch = icd10_chapter(c)
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out
