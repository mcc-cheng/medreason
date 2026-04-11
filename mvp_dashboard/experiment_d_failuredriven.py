"""Experiment D — failure-driven rule extraction (Phase 51).

Background: in every prior experiment (v0.0, v0.1, v0.2), the training
loop only kept traces from cases the agent got *right*. The 4 cases
Haiku always missed in v0.2 (adv_004, adv_006, adv_014, adv_015)
therefore never produced a rule — the chicken-and-egg. Memory had zero
rules to retrieve for the exact cases it was most needed for.

Phase 51 is the architectural fix. When the agent reaches the wrong
determination, instead of discarding the case we route it to
`analyze_failure(case, agent_determination, ground_truth, ...)` which
asks an LLM: "what atomic policy rule would have led a reviewer to the
correct determination?" The rules go through the same
citation/PII/action-length validators as the normal proposer and land
in the store with status=ACTIVE (under --skip-gate for sparse
fixtures).

This experiment:
  1. Train a NEW memory store on v0.2 train with --include-failures.
     Store path: v0.2_failuredriven.db (leaves the baseline v0.2.db
     untouched).
  2. Eval sparse-RAG + memory on v0.2 train using the new store.
  3. Compare against the baseline from v0.2 results (leaderboard):
       zero-shot Haiku          26/30
       sparse-RAG               27/30
       sparse-RAG + memory      28/30   ← baseline memory store
       full-RAG                 29/30
     The question: does failure-driven memory beat 28/30? If yes, the
     chicken-and-egg was the binding constraint. If no, memory at 28/30
     is already as good as the architecture can deliver on this
     fixture.

Cost estimate (all Haiku):
  Training (30 cases including failures ~4 wrong): ~$0.22
  Eval sparse-RAG + memory (30 cases):             ~$0.18
  Total:                                           ~$0.40
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medreason.llm import ClaudeLLMClient
from medreason.retrieval.embedder import FakeEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.splits import load_split
from medreason_bench.train import TrainingConfig, run_training


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
OUT_PATH = Path(__file__).parent / "experiment_d_failuredriven.json"

POLICY_MAX_CHARS = 200
SEED = 11
TRAIN_SEED = 42

STORE_PATH = STORES_ROOT / "v0.2_failuredriven.db"


def _build_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


def _train_failure_driven(train_cases: list) -> dict:
    """Train a fresh store with include_failures=True."""
    print(f"\n[D] training failure-driven store on {len(train_cases)} cases")
    print(f"[D] store path: {STORE_PATH}")
    if STORE_PATH.exists():
        STORE_PATH.unlink()
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(include_policy=False, policy_max_chars=None)
    critic_llm = _build_llm()
    proposer_llm = _build_llm()
    failure_llm = _build_llm()

    conn = sqlite3.connect(str(STORE_PATH))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    config = TrainingConfig(
        runner=runner,
        critic_llm=critic_llm,
        proposer_llm=proposer_llm,
        store=rule_store,
        trace_store=trace_store,
        policy=None,  # multi-policy mode
        train_cases=train_cases,
        version="v0.2_failuredriven",
        split="train",
        gate_k=5,
        gate_seed=TRAIN_SEED,
        skip_gate=True,
        embedder=FakeEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=TRAIN_SEED,
        include_failures=True,
        failure_analyzer_llm=failure_llm,
    )
    report = run_training(config)
    conn.commit()
    conn.close()

    print()
    print(f"[D] training done in {report.elapsed_seconds:.1f}s")
    print(f"[D]   normal-path rules promoted:  {report.n_rules_promoted}")
    print(f"[D]   failure-path rules promoted: {report.n_failure_rules_promoted}")
    print(f"[D]   agent correct / wrong:       "
          f"{report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"[D]   failure analyzer invoked:    {report.n_failure_analyzer_invoked}")
    print(f"[D]   failure rules proposed:      {report.n_failure_rules_proposed}")
    print(f"[D]   failure rules rejected:      {report.n_failure_rules_rejected}")
    print(f"[D]   total cost:                  ${report.cost_total:.4f}")

    return {
        "n_train_cases": len(train_cases),
        "n_agent_correct": report.n_agent_correct,
        "n_agent_wrong": report.n_agent_wrong,
        "n_rules_promoted_normal": report.n_rules_promoted,
        "n_failure_analyzer_invoked": report.n_failure_analyzer_invoked,
        "n_failure_rules_proposed": report.n_failure_rules_proposed,
        "n_failure_rules_rejected": report.n_failure_rules_rejected,
        "n_failure_rules_promoted": report.n_failure_rules_promoted,
        "cost_agent": round(report.cost_agent, 5),
        "cost_critic": round(report.cost_critic, 5),
        "cost_proposer": round(report.cost_proposer, 5),
        "cost_failure_analyzer": round(report.cost_failure_analyzer, 5),
        "cost_total": round(report.cost_total, 5),
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }


def _eval_sparse_rag_memory(eval_cases: list) -> dict:
    """Run sparse-RAG + failure-driven memory on the given cases."""
    print(f"\n[D] eval sparse-RAG + failure-driven memory on {len(eval_cases)} cases")
    base_runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    conn = sqlite3.connect(str(STORE_PATH))
    rule_store = RuleStore(conn)
    runner = MemoryRunner(
        base_runner=base_runner,
        store=rule_store,
        embedder=FakeEmbedder(),
        reranker_llm=None,
        tier3_top_k=3,
    )

    results = []
    total_cost = 0.0
    for i, case in enumerate(eval_cases, 1):
        result = runner.run(case, seed=SEED)
        ok = "OK" if result.correct else "WRONG"
        n_applied = sum(1 for a in result.applied_rules if a.applied)
        print(
            f"  [{i:2d}/{len(eval_cases)}] {case.case_id:12s} "
            f"gt={case.ground_truth_outcome.value:22s} "
            f"pred={result.determination.value:22s} {ok}  "
            f"applied={n_applied}/{len(result.applied_rules)}  "
            f"${result.cost_usd:.5f}"
        )
        total_cost += result.cost_usd
        results.append({
            "case_id": result.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "determination": result.determination.value,
            "correct": result.correct,
            "confidence": result.confidence,
            "cost_usd": round(result.cost_usd, 5),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "retrieved_rule_ids": list(result.retrieved_rule_ids),
            "applied_rules": [
                {"rule_id": a.rule_id, "applied": a.applied,
                 "rationale": a.rationale}
                for a in result.applied_rules
            ],
        })
    conn.close()

    n_correct = sum(1 for r in results if r["correct"])
    print(
        f"[D] accuracy: {n_correct}/{len(results)}  cost ${total_cost:.4f}"
    )
    return {
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else 0.0,
        "cost_usd": round(total_cost, 5),
        "results": results,
    }


def main():
    started = time.time()
    print("=" * 70)
    print("Experiment D — failure-driven rule extraction on v0.2")
    print("=" * 70)

    cases = load_split(SPLITS_ROOT / "v0.2", "train")
    print(f"loaded {len(cases)} v0.2 train cases")

    train_summary = _train_failure_driven(cases)
    eval_summary = _eval_sparse_rag_memory(cases)

    elapsed = time.time() - started
    total_cost = train_summary["cost_total"] + eval_summary["cost_usd"]

    # Identify which cases newly flipped from wrong→right or right→wrong
    # compared to the baseline v0.2 sparse-RAG + memory (28/30).
    # Baseline wrong cases: adv_014, adv_015 (per SESSION_BRIEF §4).
    baseline_wrong = {"adv_014", "adv_015"}
    d_wrong = {r["case_id"] for r in eval_summary["results"] if not r["correct"]}
    newly_right = sorted(baseline_wrong - d_wrong)
    newly_wrong = sorted(d_wrong - baseline_wrong)

    payload = {
        "experiment": "D_failure_driven_extraction",
        "version": "v0.2",
        "split": "train",
        "policy_max_chars": POLICY_MAX_CHARS,
        "eval_seed": SEED,
        "train_seed": TRAIN_SEED,
        "store_path": str(STORE_PATH),
        "train": train_summary,
        "sparse_rag_plus_memory": eval_summary,
        "comparison": {
            "baseline_v0_2_sparse_rag_plus_memory_accuracy": "28/30",
            "baseline_wrong_cases": sorted(baseline_wrong),
            "experiment_d_wrong_cases": sorted(d_wrong),
            "newly_right_cases": newly_right,
            "newly_wrong_cases": newly_wrong,
            "net_case_delta": len(newly_right) - len(newly_wrong),
        },
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print()
    print("=" * 70)
    print("Experiment D summary")
    print("=" * 70)
    print(f"  n_agent_wrong (chicken-and-egg targets): "
          f"{train_summary['n_agent_wrong']}")
    print(f"  failure rules promoted:                  "
          f"{train_summary['n_failure_rules_promoted']}")
    print(f"  sparse-RAG + memory accuracy:            "
          f"{eval_summary['n_correct']}/{eval_summary['n_cases']}")
    print(f"  vs baseline (28/30):                     "
          f"delta={payload['comparison']['net_case_delta']:+d} case(s)")
    print(f"  newly right: {newly_right or '[]'}")
    print(f"  newly wrong: {newly_wrong or '[]'}")
    print(f"  total cost:  ${total_cost:.4f}")
    print(f"  elapsed:     {elapsed:.1f}s")
    print(f"  saved to:    {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
