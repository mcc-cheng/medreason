"""LeaderboardEntry — the public JSON schema a submitted run must produce.

One entry = one (runner, split, memory_pipeline_version, dataset_version)
tuple. Multiple entries per dataset version are fine (Claude + GPT + Gemini,
zero-shot + memory); the leaderboard is the aggregation of all of them.

Fields with `Optional[...]` are expected to be None during Phase 4 (when
memory pipeline isn't wired yet): `pattern_utilization`,
`mcnemar_p_vs_zero_shot`, `delta_accuracy_pp`. Phase 5+ populates them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LeaderboardEntry(BaseModel):
    # Identity
    runner_id: str
    base_model: str
    memory_pipeline_version: str = "none"   # "none" during Phase 4 zero-shot
    dataset_version: str
    split: str
    seed_set: list[int]
    n_cases: int

    # Point estimates
    accuracy_mean: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    macro_f1: float
    brier: float
    ece: float

    # Compute + cost
    avg_total_tokens: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_case_usd: float
    total_cost_usd: float

    # Memory-specific (Phase 5+)
    pattern_utilization: Optional[float] = None
    mcnemar_p_vs_zero_shot: Optional[float] = None
    mcnemar_method: Optional[str] = None
    delta_accuracy_pp: Optional[float] = None

    # Provenance
    submitted_at: datetime = Field(default_factory=_utcnow)
    submitter: str = "local"
    code_revision: str = ""
    prompts_lock_sha: str = ""
