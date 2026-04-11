"""Build and save LeaderboardEntry objects from completed EvalRuns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from medreason.prompts import LOCK_PATH

from ..eval.harness import EvalRun, per_case_mean_correctness
from ..eval.metrics import EvalMetrics, compute_metrics
from ..eval.stats import bootstrap_mean_ci
from .schema import LeaderboardEntry


def _prompts_lock_sha() -> str:
    """SHA256 of the PROMPTS_LOCK.json file itself.

    This is the single identifier stamped on every leaderboard entry to
    indicate which frozen prompt set produced it. If the lock file has
    drifted since the entry was submitted, the stamp lets reviewers
    detect it without re-running.
    """
    if not LOCK_PATH.exists():
        return ""
    return hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()


def build_entry(
    run: EvalRun,
    cases_by_id: dict,
    *,
    submitter: str = "local",
    code_revision: str = "",
    zero_shot_correct_by_case: Optional[dict[str, bool]] = None,
) -> tuple[LeaderboardEntry, EvalMetrics]:
    """Assemble a LeaderboardEntry from an EvalRun.

    Returns the entry AND the full EvalMetrics so callers can log both
    the public schema row and the detailed per-stratum breakdown.

    Args:
        run: Completed EvalRun with results_by_seed populated.
        cases_by_id: All BenchmarkCases keyed by case_id (needed for
            ground-truth lookups inside metric functions).
        submitter: Free-form submitter id stamped on the entry.
        code_revision: Git sha or equivalent. Recommended but optional.
        zero_shot_correct_by_case: Paired zero-shot baseline correctness,
            used to compute McNemar + delta_accuracy_pp when this run
            is a memory-augmented comparison. Phase 4 callers pass None.
    """
    flat = run.flat_results()
    metrics = compute_metrics(flat, cases_by_id)

    # Honest accuracy CI: per-case mean across seeds, then bootstrap
    per_case = per_case_mean_correctness(run)
    acc_point, acc_low, acc_high = bootstrap_mean_ci(per_case)

    # McNemar paired test vs an optional zero-shot baseline.
    mcnemar_p: Optional[float] = None
    mcnemar_method: Optional[str] = None
    delta_pp: Optional[float] = None
    if zero_shot_correct_by_case is not None and flat:
        from ..eval.stats import mcnemar_test  # local import avoids cycles
        a_list: list[bool] = []
        b_list: list[bool] = []
        for r in flat:
            zs = zero_shot_correct_by_case.get(r.case_id)
            if zs is None:
                continue
            a_list.append(r.correct)
            b_list.append(zs)
        if a_list:
            mcnemar_p, mcnemar_method = mcnemar_test(a_list, b_list)
            delta_pp = 100.0 * (
                sum(1 for x in a_list if x) / len(a_list)
                - sum(1 for x in b_list if x) / len(b_list)
            )

    entry = LeaderboardEntry(
        runner_id=run.config.runner.runner_id,
        base_model=run.config.runner.model_version,
        memory_pipeline_version="none",  # Phase 5 will set this
        dataset_version=run.config.version,
        split=run.config.split,
        seed_set=list(run.config.seeds),
        n_cases=run.n_cases,
        accuracy_mean=acc_point,
        accuracy_ci_low=acc_low,
        accuracy_ci_high=acc_high,
        macro_f1=metrics.macro_f1,
        brier=metrics.brier,
        ece=metrics.ece,
        avg_total_tokens=metrics.avg_total_tokens,
        p50_latency_ms=metrics.p50_latency_ms,
        p95_latency_ms=metrics.p95_latency_ms,
        cost_per_case_usd=metrics.cost_per_case_usd,
        total_cost_usd=metrics.total_cost_usd,
        pattern_utilization=metrics.pattern_utilization,
        mcnemar_p_vs_zero_shot=mcnemar_p,
        mcnemar_method=mcnemar_method,
        delta_accuracy_pp=delta_pp,
        submitter=submitter,
        code_revision=code_revision,
        prompts_lock_sha=_prompts_lock_sha(),
    )
    return entry, metrics


def save_entry(entry: LeaderboardEntry, out_dir: Path | str) -> Path:
    """Persist a LeaderboardEntry as JSON under a stable filename.

    Filename: <runner_id>__<dataset_version>__<split>.json with any
    non-filesystem-safe characters replaced by underscores.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_runner = entry.runner_id.replace(":", "_").replace("/", "_")
    fname = f"{safe_runner}__{entry.dataset_version}__{entry.split}.json"
    path = out / fname
    path.write_text(
        json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
