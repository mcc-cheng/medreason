"""Tests for medreason_bench.leaderboard — Phase 4."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medreason_bench.eval.harness import EvalConfig, run_eval
from medreason_bench.leaderboard import (
    LeaderboardEntry,
    build_entry,
    save_entry,
)

from tests.conftest import FakeRunner


SPLITS_ROOT = (
    Path(__file__).parent.parent / "medreason_bench" / "data" / "splits"
)


def test_leaderboard_entry_round_trip():
    entry = LeaderboardEntry(
        runner_id="fake-runner-v0",
        base_model="fake",
        dataset_version="v0.0",
        split="dev",
        seed_set=[11, 17],
        n_cases=10,
        accuracy_mean=0.8,
        accuracy_ci_low=0.6,
        accuracy_ci_high=0.95,
        macro_f1=0.78,
        brier=0.3,
        ece=0.1,
        avg_total_tokens=600,
        p50_latency_ms=250,
        p95_latency_ms=400,
        cost_per_case_usd=0.002,
        total_cost_usd=0.02,
    )
    data = entry.model_dump(mode="json")
    assert data["runner_id"] == "fake-runner-v0"
    assert data["pattern_utilization"] is None
    assert data["mcnemar_p_vs_zero_shot"] is None
    revived = LeaderboardEntry.model_validate(data)
    assert revived.runner_id == entry.runner_id
    assert revived.accuracy_mean == pytest.approx(entry.accuracy_mean)


def test_build_entry_from_oracle_run_is_perfect_accuracy():
    """Oracle FakeRunner → every case correct → accuracy_mean == 1.0
    and both CI bounds pinned at 1.0 (bootstrap of all ones)."""
    runner = FakeRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11, 17],
    )
    run = run_eval(config)
    cases_by_id = {c.case_id: c for c in run.cases}
    entry, metrics = build_entry(run, cases_by_id)

    assert isinstance(entry, LeaderboardEntry)
    assert entry.runner_id == "fake-runner-v0"
    assert entry.dataset_version == "v0.0"
    assert entry.split == "dev"
    assert entry.n_cases == run.n_cases
    assert entry.seed_set == [11, 17]
    assert entry.accuracy_mean == pytest.approx(1.0)
    assert entry.accuracy_ci_low == pytest.approx(1.0)
    assert entry.accuracy_ci_high == pytest.approx(1.0)
    assert entry.macro_f1 > 0
    # No zero-shot baseline passed → mcnemar fields are None
    assert entry.mcnemar_p_vs_zero_shot is None
    assert entry.delta_accuracy_pp is None
    # Pattern utilization is None (zero-shot)
    assert entry.pattern_utilization is None


def test_build_entry_with_zero_shot_baseline_populates_mcnemar():
    """Pass a mocked zero-shot correctness map and confirm the entry
    carries a McNemar p-value + delta_accuracy_pp."""
    runner = FakeRunner()  # oracle → perfect
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11],
    )
    run = run_eval(config)
    cases_by_id = {c.case_id: c for c in run.cases}

    # Simulate a zero-shot baseline that got HALF the cases wrong.
    zs = {}
    ordered = sorted(cases_by_id.keys())
    for i, cid in enumerate(ordered):
        zs[cid] = (i % 2 == 0)

    entry, _ = build_entry(run, cases_by_id, zero_shot_correct_by_case=zs)

    assert entry.mcnemar_p_vs_zero_shot is not None
    assert 0.0 <= entry.mcnemar_p_vs_zero_shot <= 1.0
    assert entry.mcnemar_method in ("exact", "chi2")
    # Memory run is 100% correct, zs is 50% correct → delta should be +50pp
    assert entry.delta_accuracy_pp == pytest.approx(50.0, abs=15.0)


def test_save_entry_writes_json(tmp_path):
    entry = LeaderboardEntry(
        runner_id="claude-sonnet-4-20250514",
        base_model="claude-sonnet-4-20250514",
        dataset_version="v0.0",
        split="dev",
        seed_set=[11, 17, 23, 29, 31],
        n_cases=10,
        accuracy_mean=0.85,
        accuracy_ci_low=0.7,
        accuracy_ci_high=0.95,
        macro_f1=0.8,
        brier=0.25,
        ece=0.08,
        avg_total_tokens=820,
        p50_latency_ms=300,
        p95_latency_ms=500,
        cost_per_case_usd=0.003,
        total_cost_usd=0.03,
    )
    path = save_entry(entry, tmp_path)
    assert path.exists()
    # Filename encodes runner_id / dataset_version / split
    assert "claude-sonnet-4-20250514" in path.name
    assert "v0.0" in path.name
    assert "dev" in path.name
    data = json.loads(path.read_text())
    assert data["runner_id"] == "claude-sonnet-4-20250514"


def test_save_entry_safe_filename_replaces_colons(tmp_path):
    """Runner id suffixes use ':' which is illegal on some filesystems."""
    entry = LeaderboardEntry(
        runner_id="claude-sonnet-4:memory",
        base_model="claude-sonnet-4",
        dataset_version="v0.0",
        split="dev",
        seed_set=[11],
        n_cases=5,
        accuracy_mean=0.6,
        accuracy_ci_low=0.4,
        accuracy_ci_high=0.8,
        macro_f1=0.5,
        brier=0.4,
        ece=0.1,
        avg_total_tokens=500,
        p50_latency_ms=200,
        p95_latency_ms=400,
        cost_per_case_usd=0.001,
        total_cost_usd=0.005,
    )
    path = save_entry(entry, tmp_path)
    assert ":" not in path.name
    assert "claude-sonnet-4_memory" in path.name


def test_build_entry_stamps_prompts_lock_sha():
    """Every entry carries the hash of the prompts lock file so a
    reviewer can detect prompt drift after submission."""
    runner = FakeRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11],
    )
    run = run_eval(config)
    cases_by_id = {c.case_id: c for c in run.cases}
    entry, _ = build_entry(run, cases_by_id)
    assert entry.prompts_lock_sha
    assert len(entry.prompts_lock_sha) == 64
    int(entry.prompts_lock_sha, 16)
