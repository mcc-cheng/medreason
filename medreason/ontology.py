"""Domain ontology — Pydantic v2 models for Veridicus."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


# ── Enums ──────────────────────────────────────────────────────────────────────

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


class ArgumentFraming(str, Enum):
    FUNCTIONAL_IMPAIRMENT = "functional_impairment"
    CLINICAL_SEVERITY = "clinical_severity"
    GUIDELINE_ADHERENCE = "guideline_adherence"
    COST_EFFECTIVENESS = "cost_effectiveness"


# ── Core Models ────────────────────────────────────────────────────────────────

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


class ReasoningStep(BaseModel):
    step_number: int
    action: str
    evidence_cited: str
    decision_branch: str


class ArgumentStructure(BaseModel):
    framing: ArgumentFraming
    clinical_evidence_elements: list[str] = Field(default_factory=list)
    payer_policy_hooks: list[str] = Field(default_factory=list)


class ReasoningTrace(BaseModel):
    task_config: PriorAuthTaskConfig
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    argument_structure: ArgumentStructure
    uncertainty_flags: list[str] = Field(default_factory=list)
    raw_chain_of_thought: str = ""
    outcome: Outcome
    resolved_at: datetime = Field(default_factory=datetime.utcnow)


class ReasoningPattern(BaseModel):
    pattern_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    config_hash: str
    task_config: PriorAuthTaskConfig
    argument_structure: ArgumentStructure
    key_reasoning_steps: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    valid_from: str = "2026-Q1"
    valid_until: Optional[str] = None

    @computed_field
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class BenchmarkCase(BaseModel):
    case_id: str
    task_config: PriorAuthTaskConfig
    clinical_notes: str
    prior_eobs: list[str] = Field(default_factory=list)
    policy_excerpt: str
    ground_truth_outcome: Outcome
    ground_truth_reasoning: list[str] = Field(default_factory=list)
    difficulty: Difficulty


class AgentResult(BaseModel):
    case_id: str
    mode: str  # "zero_shot" or "memory"
    determination: Outcome
    reasoning_chain: str
    confidence: float
    key_factors: list[str] = Field(default_factory=list)
    correct: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    injected_pattern_ids: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_cases: int = 0
    zero_shot_results: list[AgentResult] = Field(default_factory=list)
    memory_results: list[AgentResult] = Field(default_factory=list)
    patterns_created: int = 0
    patterns_reinforced: int = 0
