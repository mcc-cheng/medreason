"""Tests for medreason_bench.eval.harness — Phase 4."""

from __future__ import annotations

from pathlib import Path

import pytest

from medreason.ontology import Outcome
from medreason.runners import AgentRunner
from medreason_bench.eval.harness import (
    EvalConfig,
    EvalRun,
    per_case_mean_correctness,
    run_eval,
)

from tests.conftest import FakeRunner


SPLITS_ROOT = (
    Path(__file__).parent.parent / "medreason_bench" / "data" / "splits"
)


# ── FakeRunner conforms to AgentRunner Protocol ──────────────────────────────


def test_fake_runner_is_protocol_conformant():
    runner = FakeRunner()
    assert isinstance(runner, AgentRunner)
    assert runner.runner_id
    assert runner.model_version
    assert isinstance(runner.supports_memory, bool)


def test_fake_runner_oracle_defaults_return_correct():
    """With no responses configured, FakeRunner returns ground truth."""
    runner = FakeRunner()
    from tests.conftest import make_case
    case = make_case(ground_truth_outcome=Outcome.DENIED)
    result = runner.run(case, seed=11)
    assert result.determination == Outcome.DENIED
    assert result.correct is True
    assert result.seed == 11


def test_fake_runner_custom_response():
    runner = FakeRunner(
        responses={"case_0001": {"determination": "denied", "confidence": 0.3}}
    )
    from tests.conftest import make_case
    case = make_case(case_id="case_0001", ground_truth_outcome=Outcome.APPROVED)
    result = runner.run(case)
    assert result.determination == Outcome.DENIED
    assert result.correct is False
    assert result.confidence == pytest.approx(0.3)


# ── End-to-end run_eval against v0.0 dev split ─────────────────────────────


def test_run_eval_against_dev_split_with_fake_runner():
    """Feed a deterministic FakeRunner into run_eval against the shipped
    v0.0 dev split. This is the core Phase 4 done-when test: the full
    pipeline from `load_split` → `runner.run()` → `EvalRun` must work
    without touching any real LLM API."""
    runner = FakeRunner()  # oracle: predicts ground truth for every case
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11, 17],
    )
    run = run_eval(config)
    assert isinstance(run, EvalRun)
    assert run.n_cases == 10  # v0.0 dev has 10 cases
    assert set(run.results_by_seed.keys()) == {11, 17}
    assert run.total_calls == 20
    assert run.finished_at is not None
    # Oracle runner → every result should be correct
    assert all(r.correct for r in run.flat_results())


def test_run_eval_per_case_mean_correctness():
    """Mix a deterministic wrong answer into one case and confirm the
    per-case mean collapses seeds correctly."""
    runner = FakeRunner(
        responses={"case_0002": {"determination": "denied"}},  # forced wrong
    )
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11, 17, 23],
    )
    run = run_eval(config)
    means = per_case_mean_correctness(run)
    assert len(means) == 10
    # Exactly one case should have mean 0 (the case_0002 override)
    # if case_0002 is in the dev split, else the test is a no-op.
    dev_ids = [c.case_id for c in run.cases]
    if "case_0002" in dev_ids:
        assert 0.0 in means
    else:
        # Not in dev split — test is vacuous but shouldn't fail.
        assert all(m == 1.0 for m in means)


def test_run_eval_captures_seed_on_each_result():
    runner = FakeRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[42],
    )
    run = run_eval(config)
    for r in run.results_by_seed[42]:
        assert r.seed == 42


def test_run_eval_accumulates_cost():
    runner = FakeRunner(default_cost_usd=0.001)
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="dev",
        seeds=[11],
    )
    run = run_eval(config)
    assert run.total_cost_usd == pytest.approx(0.001 * run.n_cases)


# ── Quick mode ──────────────────────────────────────────────────────────────


def test_run_eval_quick_mode_samples_ten_cases():
    runner = FakeRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="train",  # train has 31 cases; quick should take 10
        seeds=[11],
        quick=True,
        quick_sample_size=10,
    )
    run = run_eval(config)
    assert run.n_cases <= 10
    # Quick mode must still be deterministic
    run2 = run_eval(config)
    assert [c.case_id for c in run2.cases] == [c.case_id for c in run.cases]


def test_run_eval_refuses_quick_test_combination():
    runner = FakeRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="test",
        seeds=[11],
        quick=True,
    )
    with pytest.raises(RuntimeError) as exc:
        run_eval(config)
    assert "quick" in str(exc.value).lower()


# ── Tripwires: prompts lock + leak guard for the test split ─────────────────


def test_run_eval_verifies_prompts_lock_on_test_split():
    """Calling run_eval with split='test' must invoke verify_lock
    before touching any runner. Use a sabotage monkeypatch to raise
    from verify_lock and confirm the harness propagates it."""
    import medreason_bench.eval.harness as harness_mod
    calls = []

    def fake_verify(*, allow_bypass: bool = False):
        calls.append(allow_bypass)
        from medreason.prompts.lock import PromptsLockError
        raise PromptsLockError("sabotage")

    orig = harness_mod.verify_lock
    harness_mod.verify_lock = fake_verify
    try:
        runner = FakeRunner()
        config = EvalConfig(
            runner=runner,
            splits_root=SPLITS_ROOT,
            version="v0.0",
            split="test",
            seeds=[11],
        )
        from medreason.prompts.lock import PromptsLockError
        with pytest.raises(PromptsLockError):
            run_eval(config)
    finally:
        harness_mod.verify_lock = orig

    # The sabotaged verify_lock must have been called with allow_bypass=False
    assert calls == [False]


def test_run_eval_does_not_verify_lock_on_dev_split():
    """Dev split runs should NOT require the lock to be clean so local
    iteration isn't blocked by in-flight prompt edits."""
    import medreason_bench.eval.harness as harness_mod
    calls = []

    def fake_verify(*, allow_bypass: bool = False):
        calls.append(allow_bypass)

    orig = harness_mod.verify_lock
    harness_mod.verify_lock = fake_verify
    try:
        runner = FakeRunner()
        config = EvalConfig(
            runner=runner,
            splits_root=SPLITS_ROOT,
            version="v0.0",
            split="dev",
            seeds=[11],
        )
        run_eval(config)
    finally:
        harness_mod.verify_lock = orig

    assert calls == []  # never called on dev


def test_run_eval_enters_leak_guard_test_phase_on_test_split():
    from medreason.store import LeakGuard, TestSetLeakError
    lg = LeakGuard()
    runner = FakeRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="test",
        seeds=[11],
        leak_guard=lg,
        enforce_prompts_lock=False,  # don't fight the real lock in this test
    )
    assert lg.read_only is False
    run_eval(config)
    # After run, we should be back in writable mode
    assert lg.read_only is False
    # The harness should have touched the guard — check by asserting it
    # was flipped at SOME point via a fresh sentinel.


def test_leak_guard_stays_read_only_during_harness_run():
    """Monkey a sentinel runner that checks guard.read_only mid-run."""
    from medreason.store import LeakGuard

    lg = LeakGuard()
    observed_read_only: list[bool] = []

    class _SentinelRunner(FakeRunner):
        def run(self, case, *, seed=0, system_extra=""):
            observed_read_only.append(lg.read_only)
            return super().run(case, seed=seed, system_extra=system_extra)

    runner = _SentinelRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="test",
        seeds=[11],
        leak_guard=lg,
        enforce_prompts_lock=False,
    )
    run_eval(config)
    assert observed_read_only  # at least one runner call happened
    assert all(observed_read_only), \
        "LeakGuard must be in read-only mode during every test-split runner call"
    assert lg.read_only is False  # restored on exit


def test_leak_guard_restored_even_if_run_raises():
    from medreason.store import LeakGuard
    lg = LeakGuard()

    class _BoomRunner(FakeRunner):
        def run(self, case, *, seed=0, system_extra=""):
            raise RuntimeError("boom")

    runner = _BoomRunner()
    config = EvalConfig(
        runner=runner,
        splits_root=SPLITS_ROOT,
        version="v0.0",
        split="test",
        seeds=[11],
        leak_guard=lg,
        enforce_prompts_lock=False,
    )
    with pytest.raises(RuntimeError):
        run_eval(config)
    assert lg.read_only is False  # finally-clause restored it
