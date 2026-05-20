"""Atomic IF-THEN reasoning rules.

A ReasoningRule is the unit injected into the agent prompt and updated by
the posterior tracker. Each rule has:

- A structural trigger (CPT family, ICD chapter, payer, facility) that the
  Tier 1 ontology lookup uses for cheap SQL prefiltering.
- A semantic predicate (NL string + cached embedding) that Tier 2/3 use.
- An action (the THEN clause): a single atomic check or criterion.
- Provenance pointing only at train+dev case_ids (leak guard enforces this).
- A Beta posterior over success/failure counts.

The generalization gate must validate a candidate against ≥5 held-out cases
matching its trigger before promotion to ACTIVE. See the plan file for the
critic → proposer → gate flow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import uuid4


def _utcnow() -> datetime:
    """Timezone-aware UTC now. `datetime.utcnow()` is deprecated in 3.12+."""
    return datetime.now(timezone.utc)

from pydantic import BaseModel, Field, computed_field

from .case import FacilityType, Payer, PriorAuthTaskConfig
from .codes import CPTFamily, ICD10Chapter, cpt_families, icd10_chapters


class RuleStatus(str, Enum):
    CANDIDATE   = "candidate"      # proposed, not yet generalization-validated
    ACTIVE      = "active"         # passed gate, eligible for injection
    QUARANTINED = "quarantined"    # posterior dropped below threshold
    DEPRECATED  = "deprecated"     # superseded or policy expired


class RuleTrigger(BaseModel):
    """Structural + semantic conditions for when a rule applies.

    Empty lists mean "any" — a trigger with `payers=[]` matches every payer.
    The `matches()` prefilter is intentionally cheap; semantic match is the
    reranker's job.
    """
    cpt_families: list[CPTFamily] = Field(default_factory=list)
    cpt_codes: list[str] = Field(default_factory=list)            # specific overrides
    icd10_chapters: list[ICD10Chapter] = Field(default_factory=list)
    icd10_prefixes: list[str] = Field(default_factory=list)       # e.g. "M17", "M54.5"
    payers: list[Payer] = Field(default_factory=list)
    facility_types: list[FacilityType] = Field(default_factory=list)
    required_modifiers: list[str] = Field(default_factory=list)
    semantic_predicate: str = ""
    semantic_embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None  # pinned id of embedding model used

    def matches(self, task_cfg: PriorAuthTaskConfig) -> bool:
        """Cheap structural prefilter — empty list = wildcard."""
        # Payer
        if self.payers and task_cfg.payer not in self.payers:
            return False
        # Facility
        if self.facility_types and task_cfg.facility_type not in self.facility_types:
            return False
        # Required modifiers — every required modifier must be present
        if self.required_modifiers:
            cfg_mods = set(task_cfg.modifiers)
            if not set(self.required_modifiers).issubset(cfg_mods):
                return False
        # CPT — either an explicit code match or a family match
        if self.cpt_codes or self.cpt_families:
            code_ok = task_cfg.cpt_code in self.cpt_codes if self.cpt_codes else False
            fam_ok = False
            if self.cpt_families:
                fam_ok = any(f in self.cpt_families
                             for f in cpt_families([task_cfg.cpt_code]))
            if not (code_ok or fam_ok):
                return False
        # ICD — either prefix match or chapter match
        if self.icd10_prefixes or self.icd10_chapters:
            prefix_ok = False
            if self.icd10_prefixes:
                prefix_ok = any(
                    icd.upper().startswith(p.upper())
                    for icd in task_cfg.icd10_codes
                    for p in self.icd10_prefixes
                )
            chapter_ok = False
            if self.icd10_chapters:
                case_chapters = set(icd10_chapters(task_cfg.icd10_codes))
                chapter_ok = bool(case_chapters & set(self.icd10_chapters))
            if not (prefix_ok or chapter_ok):
                return False
        return True


class RuleEvidence(BaseModel):
    """Provenance for a rule. NEVER references test-set case_ids — the
    leak guard enforces this on every store.put().
    """
    supporting_case_ids: list[str] = Field(default_factory=list)
    source_policy_citation: str = ""             # e.g. "CMS LCD L34522 §C.1"
    source_policy_url: Optional[str] = None
    extracted_from_trace_ids: list[str] = Field(default_factory=list)
    proposer_model: str = ""                     # version-pinned proposer LLM id
    proposer_run_id: str = ""


class ReasoningRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: f"rule_{uuid4().hex[:10]}")
    version: int = 1
    status: RuleStatus = RuleStatus.CANDIDATE

    # The IF-THEN body
    trigger: RuleTrigger
    action: str                  # the THEN clause: ≤25 words, one atomic check
    rationale: str = ""          # 1-2 sentences for human auditors
    polarity: Literal["supports_approval", "supports_denial", "requires_check"] = "requires_check"

    # Provenance
    evidence: RuleEvidence

    # Posterior tracking — Beta(α=successes+1, β=failures+1)
    success_count: int = 0
    failure_count: int = 0
    seen_count: int = 0          # times retrieved (whether or not applied)
    quarantine_count: int = 0    # how many times quarantined historically

    # Counter-rules / known exceptions
    counter_rule_ids: list[str] = Field(default_factory=list)
    known_exception_notes: list[str] = Field(default_factory=list)

    # Lifecycle
    created_at: datetime = Field(default_factory=_utcnow)
    last_used: Optional[datetime] = None
    valid_from: str = "2026-Q1"
    valid_until: Optional[str] = None

    @computed_field
    @property
    def posterior_mean(self) -> float:
        a = self.success_count + 1
        b = self.failure_count + 1
        return a / (a + b)

    @computed_field
    @property
    def posterior_variance(self) -> float:
        a = self.success_count + 1
        b = self.failure_count + 1
        denom = (a + b) ** 2 * (a + b + 1)
        return (a * b) / denom

    @computed_field
    @property
    def trials(self) -> int:
        return self.success_count + self.failure_count
