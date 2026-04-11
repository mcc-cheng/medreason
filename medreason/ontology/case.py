"""Case + task config models. Moved from the legacy single-file ontology.

These models are unchanged from the pre-rework state so existing modules
(extractor, generator, local_cases, agent) keep working until later phases
replace them.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ── Enums ─────────────────────────────────────────────────────────────────────


class Payer(str, Enum):
    UNITEDHEALTHCARE = "UnitedHealthcare"
    AETNA = "Aetna"
    BCBS = "BCBS"
    CIGNA = "Cigna"
    HUMANA = "Humana"
    MEDICARE = "Medicare"
    MEDICAID = "Medicaid"


class FacilityType(str, Enum):
    OUTPATIENT = "outpatient"
    INPATIENT = "inpatient"
    ASC = "ASC"
    ED = "ED"
    TELEHEALTH = "telehealth"


class DenialReason(str, Enum):
    MEDICAL_NECESSITY = "medical_necessity"
    EXPERIMENTAL = "experimental"
    OUT_OF_NETWORK = "out_of_network"
    MISSING_INFO = "missing_info"
    DUPLICATE = "duplicate"
    NOT_COVERED = "not_covered"
    CODING_ERROR = "coding_error"
    FREQUENCY_LIMIT = "frequency_limit"


class Outcome(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    OVERTURNED_ON_APPEAL = "overturned_on_appeal"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── Task config + benchmark case ──────────────────────────────────────────────


class PriorAuthTaskConfig(BaseModel):
    payer: Payer
    cpt_code: str
    icd10_codes: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    facility_type: FacilityType
    denial_reason: Optional[DenialReason] = None
    policy_version: str = "2026-Q1"

    @computed_field
    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "payer": self.payer.value,
                "cpt": self.cpt_code,
                "icd10": sorted(self.icd10_codes),
                "mods": sorted(self.modifiers),
                "facility": self.facility_type.value,
                "denial": self.denial_reason.value if self.denial_reason else None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class BenchmarkCase(BaseModel):
    case_id: str
    task_config: PriorAuthTaskConfig
    clinical_notes: str
    prior_eobs: list[str] = Field(default_factory=list)
    policy_excerpt: str
    ground_truth_outcome: Outcome
    ground_truth_reasoning: list[str] = Field(default_factory=list)
    difficulty: Difficulty
