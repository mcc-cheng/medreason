"""Experiments D2 + D3 — follow-ups on the Phase 51 failure-driven result.

Experiment D ran the failure analyzer but the flat 28/30 result had two
identifiable causes:

  1. adv_014 never reached the failure analyzer because training used
     zero-shot Haiku as the base runner, and zero-shot Haiku gets
     adv_014 right. adv_014 only fails under sparse-RAG at eval time
     (when the policy excerpt is truncated and the agent over-anchors
     on "patient prefers" language).

  2. adv_015's failure-derived rule WAS retrieved at eval time but was
     marked applied=False by the agent. The rule's predicate is narrow
     (Belimumab + joint specialty recommendation) and the agent chose
     overturned_on_appeal instead of approved — a polarity-to-outcome
     confusion.

This file runs both follow-ups.

D2 — symmetric training configuration
  Train v0.2 with --include-failures AND sparse-RAG base runner
  (include_policy=True, policy_max_chars=200) AND the same seed as
  eval (11). That way cases wrong at eval time are exactly the cases
  the failure analyzer sees during training. adv_014 should now enter
  the failure path and produce rules.

D3 — single-case diagnostic on adv_015
  Re-run adv_015 in isolation with sparse-RAG + memory using the
  Exp-D store, and capture the FULL reasoning chain from the agent.
  This tells us whether adv_015's apply-failure is about rule
  specificity, prompt framing, or polarity-vs-outcome confusion. Cheap
  (one call, ~$0.008).

Cost estimate:
  D2 training (30 cases, sparse-RAG runner + failure analyzer): ~$0.25
  D2 eval (30 cases sparse-RAG + memory):                       ~$0.18
  D3 single-case diagnostic:                                    ~$0.008
  Total:                                                        ~$0.44
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
OUT_PATH = Path(__file__).parent / "experiment_d2_d3.json"

POLICY_MAX_CHARS = 200
# D2: use the SAME seed for training and eval so the cases wrong at
# training time are exactly the cases wrong at eval time. Seed 11 is
# the v0.2 eval seed.
SEED = 11

D2_STORE_PATH = STORES_ROOT / "v0.2_fd_sparse.db"
D_STORE_PATH = STORES_ROOT / "v0.2_failuredriven.db"  # Exp-D store, for D3


def _build_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


# ─────────────────────────────────────────────────────────────────────────
# D2 — Train with sparse-RAG base runner + same seed as eval
# ─────────────────────────────────────────────────────────────────────────


def _d2_train(train_cases: list) -> dict:
    print(f"\n[D2] training failure-driven store on {len(train_cases)} cases")
    print(f"[D2] base runner: sparse-RAG (include_policy=True, "
          f"policy_max_chars={POLICY_MAX_CHARS})")
    print(f"[D2] seed: {SEED} (matches eval seed for determinism)")
    print(f"[D2] store path: {D2_STORE_PATH}")

    if D2_STORE_PATH.exists():
        D2_STORE_PATH.unlink()
    D2_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # KEY CHANGE vs Experiment D: the training base runner now matches
    # the eval configuration. Zero-shot correct cases that become wrong
    # under sparse-RAG will now trigger the failure path.
    runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    critic_llm = _build_llm()
    proposer_llm = _build_llm()
    failure_llm = _build_llm()

    conn = sqlite3.connect(str(D2_STORE_PATH))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    config = TrainingConfig(
        runner=runner,
        critic_llm=critic_llm,
        proposer_llm=proposer_llm,
        store=rule_store,
        trace_store=trace_store,
        policy=None,
        train_cases=train_cases,
        version="v0.2_fd_sparse",
        split="train",
        gate_k=5,
        gate_seed=SEED,
        skip_gate=True,
        embedder=FakeEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=SEED,
        include_failures=True,
        failure_analyzer_llm=failure_llm,
    )
    report = run_training(config)
    conn.commit()
    conn.close()

    print()
    print(f"[D2] training done in {report.elapsed_seconds:.1f}s")
    print(f"[D2]   agent correct / wrong:       "
          f"{report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"[D2]   normal-path rules promoted:  {report.n_rules_promoted}")
    print(f"[D2]   failure analyzer invoked:    {report.n_failure_analyzer_invoked}")
    print(f"[D2]   failure rules proposed:      {report.n_failure_rules_proposed}")
    print(f"[D2]   failure rules rejected:      {report.n_failure_rules_rejected}")
    print(f"[D2]   failure rules promoted:      {report.n_failure_rules_promoted}")
    print(f"[D2]   total cost:                  ${report.cost_total:.4f}")

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
        "cost_proposer": round(report.cost_proposer, 5),
        "cost_failure_analyzer": round(report.cost_failure_analyzer, 5),
        "cost_total": round(report.cost_total, 5),
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }


def _d2_eval(eval_cases: list) -> dict:
    print(f"\n[D2] eval sparse-RAG + memory on {len(eval_cases)} cases")
    print(f"[D2] memory store: {D2_STORE_PATH}")

    base_runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    conn = sqlite3.connect(str(D2_STORE_PATH))
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
            "retrieved_rule_ids": list(result.retrieved_rule_ids),
            "applied_rules": [
                {"rule_id": a.rule_id, "applied": a.applied,
                 "rationale": a.rationale}
                for a in result.applied_rules
            ],
        })
    conn.close()

    n_correct = sum(1 for r in results if r["correct"])
    print(f"[D2] accuracy: {n_correct}/{len(results)}  cost ${total_cost:.4f}")
    return {
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else 0.0,
        "cost_usd": round(total_cost, 5),
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────
# D3 — Single-case diagnostic on adv_015 using the Exp-D store
# ─────────────────────────────────────────────────────────────────────────


def _d3_adv_015_diagnostic(all_cases: list) -> dict:
    print("\n[D3] single-case diagnostic on adv_015")
    print(f"[D3] store: {D_STORE_PATH}")

    target = next((c for c in all_cases if c.case_id == "adv_015"), None)
    if target is None:
        print("[D3] adv_015 not found — aborting")
        return {"error": "adv_015 not found"}

    base_runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    conn = sqlite3.connect(str(D_STORE_PATH))
    rule_store = RuleStore(conn)
    runner = MemoryRunner(
        base_runner=base_runner,
        store=rule_store,
        embedder=FakeEmbedder(),
        reranker_llm=None,
        tier3_top_k=3,
    )

    result = runner.run(target, seed=SEED)
    conn.close()

    print(f"[D3] determination: {result.determination.value} "
          f"(gt={target.ground_truth_outcome.value})")
    print(f"[D3] correct: {result.correct}  confidence: {result.confidence}")
    print(f"[D3] retrieved rules: {result.retrieved_rule_ids}")
    print(f"[D3] applied rules:")
    for a in result.applied_rules:
        print(f"      applied={a.applied}  rule_id={a.rule_id}")
        if a.rationale:
            print(f"        rationale: {a.rationale[:200]}")

    print()
    print("[D3] full reasoning chain:")
    print("-" * 70)
    print(result.reasoning_chain)
    print("-" * 70)

    return {
        "case_id": target.case_id,
        "ground_truth": target.ground_truth_outcome.value,
        "determination": result.determination.value,
        "correct": result.correct,
        "confidence": result.confidence,
        "reasoning_chain_full": result.reasoning_chain,
        "retrieved_rule_ids": list(result.retrieved_rule_ids),
        "applied_rules": [
            {"rule_id": a.rule_id, "applied": a.applied,
             "rationale": a.rationale}
            for a in result.applied_rules
        ],
        "cost_usd": round(result.cost_usd, 5),
    }


def main():
    started = time.time()
    print("=" * 70)
    print("Experiments D2 + D3 — Phase 51 follow-ups")
    print("=" * 70)

    cases = load_split(SPLITS_ROOT / "v0.2", "train")
    print(f"loaded {len(cases)} v0.2 train cases")

    # ── D2 ───────────────────────────────────────────────────────────────
    d2_train_summary = _d2_train(cases)
    d2_eval_summary = _d2_eval(cases)

    # Baseline comparison
    baseline_wrong = {"adv_014", "adv_015"}  # v0.2 sparse-RAG+memory baseline
    d2_wrong = {r["case_id"] for r in d2_eval_summary["results"]
                if not r["correct"]}
    newly_right = sorted(baseline_wrong - d2_wrong)
    newly_wrong = sorted(d2_wrong - baseline_wrong)

    # ── D3 ───────────────────────────────────────────────────────────────
    d3 = _d3_adv_015_diagnostic(cases)

    # ── Save ─────────────────────────────────────────────────────────────
    elapsed = time.time() - started
    total_cost = (
        d2_train_summary["cost_total"]
        + d2_eval_summary["cost_usd"]
        + d3.get("cost_usd", 0.0)
    )

    payload = {
        "experiment": "D2_D3_followups",
        "version": "v0.2",
        "policy_max_chars": POLICY_MAX_CHARS,
        "seed": SEED,
        "D2_symmetric_training": {
            "store_path": str(D2_STORE_PATH),
            "train": d2_train_summary,
            "sparse_rag_plus_memory": d2_eval_summary,
            "comparison": {
                "baseline_wrong_cases": sorted(baseline_wrong),
                "D2_wrong_cases": sorted(d2_wrong),
                "newly_right_cases": newly_right,
                "newly_wrong_cases": newly_wrong,
                "net_case_delta": len(newly_right) - len(newly_wrong),
            },
        },
        "D3_adv_015_diagnostic": d3,
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    print("=" * 70)
    print("D2 + D3 summary")
    print("=" * 70)
    print(f"  D2 train: agent_wrong={d2_train_summary['n_agent_wrong']}, "
          f"failure_rules_promoted={d2_train_summary['n_failure_rules_promoted']}")
    print(f"  D2 eval:  "
          f"{d2_eval_summary['n_correct']}/{d2_eval_summary['n_cases']} "
          f"(baseline 28/30, delta {len(newly_right) - len(newly_wrong):+d})")
    print(f"  D2 wrong cases: {sorted(d2_wrong) or '[]'}")
    print(f"  D2 newly right: {newly_right or '[]'}")
    print(f"  D2 newly wrong: {newly_wrong or '[]'}")
    print(f"  D3 adv_015 determination: {d3.get('determination')} "
          f"(correct={d3.get('correct')})")
    print(f"  total cost: ${total_cost:.4f}")
    print(f"  elapsed:    {elapsed:.1f}s")
    print(f"  saved to:   {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
