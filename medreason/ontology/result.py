"""Agent run result + applied-rule contract.

AgentResult is the unit returned by every AgentRunner. The new fields
(applied_rules, retrieved_rule_ids, runner_id, cost_usd, seed) are additive:
the legacy fields (mode, injected_pattern_ids) are kept with defaults so
the pre-rework agent.py / benchmark.py keep validating successfully until
they are replaced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from .case import Outcome


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppliedRule(BaseModel):
    """One entry in the agent's applied_rules contract.

    The agent must emit exactly one entry per retrieved rule_id; the
    memory_wrapper validates and re-prompts once if any are missing.
    """
    rule_id: str
    applied: bool
    rationale: str = ""


class AgentResult(BaseModel):
    case_id: str
    determination: Outcome
    reasoning_chain: str = ""
    confidence: float = 0.5
    key_factors: list[str] = Field(default_factory=list)
    correct: bool = False

    # Token / cost / latency
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    # Legacy fields — pre-rework agent.py still sets these
    mode: str = "zero_shot"  # "zero_shot" | "memory" (legacy)
    injected_pattern_ids: list[str] = Field(default_factory=list)

    # New fields — Phase 5 memory pipeline
    runner_id: str = ""
    seed: int = 0
    retrieved_rule_ids: list[str] = Field(default_factory=list)
    retrieved_trace_ids: list[str] = Field(default_factory=list)
    applied_rules: list[AppliedRule] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    timestamp: datetime = Field(default_factory=_utcnow)
    total_cases: int = 0
    zero_shot_results: list[AgentResult] = Field(default_factory=list)
    memory_results: list[AgentResult] = Field(default_factory=list)
    patterns_created: int = 0
    patterns_reinforced: int = 0
