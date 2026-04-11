"""Reasoning trace models — what the agent (and the critic) actually did.

Traces are audit-only in the new pipeline: they are stored alongside rules
but never injected raw into agent prompts. Retrieval may surface them in
--explain mode.

`ReasoningPattern` is a legacy shim retained so the pre-rework
extractor / store / injector keep importing during Phase 0. It will be
removed when those modules are replaced in Phase 1+.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

from .case import Outcome, PriorAuthTaskConfig


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArgumentFraming(str, Enum):
    FUNCTIONAL_IMPAIRMENT = "functional_impairment"
    CLINICAL_SEVERITY = "clinical_severity"
    GUIDELINE_ADHERENCE = "guideline_adherence"
    COST_EFFECTIVENESS = "cost_effectiveness"


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
    """An audit record of one resolution attempt.

    In the new pipeline this is the unit the rule_proposer reads to extract
    candidate rules. The critic produces an independent ReasoningTrace from
    the case inputs alone (no agent reasoning, no ground truth) — agreement
    between the agent's trace and the critic's trace is the gating condition
    before the trace is stored.
    """
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid4().hex[:10]}")
    task_config: PriorAuthTaskConfig
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    argument_structure: ArgumentStructure
    uncertainty_flags: list[str] = Field(default_factory=list)
    raw_chain_of_thought: str = ""
    outcome: Outcome
    source: str = "agent"  # "agent" | "critic"
    resolved_at: datetime = Field(default_factory=_utcnow)


# ── Legacy shim ───────────────────────────────────────────────────────────────


class ReasoningPattern(BaseModel):
    """Legacy pattern type. DO NOT use in new code.

    Retained so the pre-rework extractor / store / injector / benchmark keep
    importing during Phase 0. Replaced by ReasoningRule (in rule.py) once
    Phase 5 lands the new memory pipeline.
    """
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
