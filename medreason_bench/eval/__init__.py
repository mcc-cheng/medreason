"""Eval harness — runs AgentRunners across cases, computes metrics, and
emits leaderboard entries.

Phase 4 ships zero-shot eval only. Phase 5 will add the memory wrapper
layer and the cross-seed McNemar paired test against zero-shot baselines.
"""

from .metrics import (
    EvalMetrics,
    accuracy,
    avg_token_counts,
    brier_score,
    compute_metrics,
    cost_per_case,
    ece,
    latency_percentiles,
    macro_f1,
    per_class_f1,
    per_stratum,
    total_cost_usd,
)
from .stats import (
    bootstrap_mean_ci,
    mcnemar_chi2,
    mcnemar_exact,
    mcnemar_test,
)
from .harness import EvalConfig, EvalRun, run_eval

__all__ = [
    # Metrics
    "EvalMetrics",
    "accuracy",
    "per_class_f1",
    "macro_f1",
    "brier_score",
    "ece",
    "avg_token_counts",
    "latency_percentiles",
    "total_cost_usd",
    "cost_per_case",
    "per_stratum",
    "compute_metrics",
    # Stats
    "bootstrap_mean_ci",
    "mcnemar_exact",
    "mcnemar_chi2",
    "mcnemar_test",
    # Harness
    "EvalConfig",
    "EvalRun",
    "run_eval",
]
