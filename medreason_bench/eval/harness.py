"""Eval harness — runs an AgentRunner across a split for N seeds.

Phase 4 scope: zero-shot only. The memory wrapper is Phase 5; when that
lands, it will be another AgentRunner that plugs in via `config.runner`,
so the harness itself won't need to change.

Tripwires enforced at run time:

1. **Prompts lock verification.** When `split == "test"`, the harness
   calls `medreason.prompts.verify_lock(allow_bypass=False)` before any
   runner call. The `allow_bypass=False` argument makes the local-dev
   bypass env var literally unreachable at scored-test time.

2. **LeakGuard read-only mode.** When `split == "test"` AND a LeakGuard
   was wired into the config, the harness calls `leak_guard.enter_test_phase()`
   so any accidental store write from a runner (Phase 5 memory wrapper
   will have a store) would raise TestSetLeakError. On exit, we call
   `exit_test_phase()` even if the run raised — this is belt-and-braces,
   since the guard is usually a fresh object per run anyway.

3. **Quick mode** (`--quick`) samples a deterministic 10-case stratified
   subset of the target split. Used for cheap dev iteration. Never valid
   for a scored test run — the harness refuses `quick=True` AND
   `split == "test"` simultaneously.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from medreason.ontology import AgentResult, BenchmarkCase
from medreason.prompts import verify_lock
from medreason.runners import AgentRunner
from medreason.store import LeakGuard

from ..splits import load_split, verify_manifest


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Config + result dataclasses ──────────────────────────────────────────────


@dataclass
class EvalConfig:
    """One eval run's full configuration.

    The default seeds [11, 17, 23, 29, 31] are chosen to be prime and
    spread out so cross-seed variance isn't dominated by any one
    PRNG-cycle alignment.
    """
    runner: AgentRunner
    splits_root: Path
    version: str = "v0.0"
    split: str = "dev"
    seeds: list[int] = field(default_factory=lambda: [11, 17, 23, 29, 31])
    quick: bool = False
    quick_sample_size: int = 10
    enforce_prompts_lock: bool = True
    leak_guard: Optional[LeakGuard] = None
    progress_hook: Optional[Callable[[str], None]] = None
    verify_manifest_first: bool = True

    def split_dir(self) -> Path:
        return self.splits_root / self.version


@dataclass
class EvalRun:
    config: EvalConfig
    results_by_seed: dict[int, list[AgentResult]] = field(default_factory=dict)
    cases: list[BenchmarkCase] = field(default_factory=list)
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None

    @property
    def total_calls(self) -> int:
        return sum(len(v) for v in self.results_by_seed.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for rs in self.results_by_seed.values() for r in rs)

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def flat_results(self) -> list[AgentResult]:
        out: list[AgentResult] = []
        for seed in sorted(self.results_by_seed.keys()):
            out.extend(self.results_by_seed[seed])
        return out


# ── Quick-mode stratified sampler ────────────────────────────────────────────


def _quick_sample(
    cases: list[BenchmarkCase],
    sample_size: int,
) -> list[BenchmarkCase]:
    """Stratified sample by (outcome, difficulty), deterministic by
    case_id ordering within each stratum.

    Grabs up to ceil(sample_size / n_strata) from each stratum, then
    trims to exactly `sample_size`. If the requested size exceeds the
    input, returns the full input."""
    if sample_size >= len(cases) or sample_size <= 0:
        return list(cases)

    buckets: dict[tuple, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        key = (case.ground_truth_outcome.value, case.difficulty.value)
        buckets[key].append(case)

    n_strata = len(buckets)
    per_stratum = max(1, -(-sample_size // n_strata))  # ceil div

    picked: list[BenchmarkCase] = []
    # Sort strata keys for determinism
    for key in sorted(buckets.keys()):
        bucket = sorted(buckets[key], key=lambda c: c.case_id)
        picked.extend(bucket[:per_stratum])

    picked.sort(key=lambda c: c.case_id)
    return picked[:sample_size]


# ── The runner ──────────────────────────────────────────────────────────────


def _log(progress_hook: Optional[Callable[[str], None]], msg: str) -> None:
    if progress_hook is not None:
        progress_hook(msg)


def run_eval(config: EvalConfig) -> EvalRun:
    """Execute one eval run and return a populated EvalRun.

    The runner is called once per (seed, case). Results are kept in
    per-seed buckets so downstream metric/stats code can do both pooled
    and per-seed analyses.

    Errors inside a single runner call are NOT caught here — an
    unhandled exception aborts the run. This is intentional: the Phase 2
    ClaudeRunner is specifically designed to return a concrete
    AgentResult tagged `[parse_error]` on model misbehavior, so the
    only things that bubble up are real infrastructure failures (missing
    API key, network death, schema drift) that should not be silently
    averaged into metrics.
    """
    # Early refusal: quick + test is a benchmark-integrity red flag.
    if config.quick and config.split == "test":
        raise RuntimeError(
            "EvalConfig(quick=True, split='test') is not allowed. Quick mode "
            "takes a 10-case stratified sample, which is insufficient for a "
            "scored test run and would also drift the test fingerprint set."
        )

    # Tripwire 1: prompts lock on the test split.
    if config.split == "test" and config.enforce_prompts_lock:
        _log(config.progress_hook, "[harness] verify_lock(allow_bypass=False) for test split")
        verify_lock(allow_bypass=False)

    # Tripwire 2: manifest verification (drift detection).
    if config.verify_manifest_first:
        _log(config.progress_hook, f"[harness] verify_manifest({config.split_dir()})")
        verify_manifest(config.split_dir())

    # Load the split.
    cases = load_split(config.split_dir(), config.split)
    if config.quick:
        cases = _quick_sample(cases, config.quick_sample_size)
    _log(config.progress_hook,
         f"[harness] loaded {len(cases)} cases from {config.split}"
         + (f" (quick mode, {config.quick_sample_size} target)" if config.quick else ""))

    # Tripwire 3: read-only test-phase mode on the store.
    flipped_store = False
    if config.split == "test" and config.leak_guard is not None:
        config.leak_guard.enter_test_phase()
        flipped_store = True
        _log(config.progress_hook, "[harness] LeakGuard.enter_test_phase()")

    run = EvalRun(config=config, cases=cases)

    try:
        for seed in config.seeds:
            _log(config.progress_hook, f"[harness] seed={seed}")
            bucket: list[AgentResult] = []
            for case in cases:
                result = config.runner.run(case, seed=seed)
                bucket.append(result)
            run.results_by_seed[seed] = bucket

        run.finished_at = _utcnow()
        _log(
            config.progress_hook,
            f"[harness] complete: {run.total_calls} calls, "
            f"${run.total_cost_usd:.4f}, "
            f"{(run.finished_at - run.started_at).total_seconds():.1f}s",
        )
        return run
    finally:
        if flipped_store and config.leak_guard is not None:
            config.leak_guard.exit_test_phase()


# ── Helper used by metric aggregation ────────────────────────────────────────


def per_case_mean_correctness(run: EvalRun) -> list[float]:
    """For each case, return its mean correctness across all seeds.

    Used as the input to bootstrap_mean_ci for the honest case-level CI
    on accuracy. Returned list is ordered by case_id for determinism.
    """
    per_case: dict[str, list[int]] = defaultdict(list)
    for results in run.results_by_seed.values():
        for r in results:
            per_case[r.case_id].append(1 if r.correct else 0)

    out: list[float] = []
    for case_id in sorted(per_case.keys()):
        correct = per_case[case_id]
        out.append(sum(correct) / len(correct))
    return out
