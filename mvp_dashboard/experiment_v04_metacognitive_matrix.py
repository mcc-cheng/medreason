"""Experiment v0.4 — metacognitive matrix on the expanded Aetna fixture.

The replication experiment the user requested after the v0.3
metacognitive hypothesis hit (8/9 sparse-RAG + memory, +22pp vs
additive). Uses the expanded 60-case within-domain Aetna lumbar MRI
fixture (30 original + 30 new hand-authored cases across Tier 1
straightforward / Tier 2 exception / Tier 3 adversarial).

Split: stratified 41 train / 19 test, seed 42 (one extra case landed
in train from largest-remainder rounding on an odd stratum).

Method: identical to v0.3 metacognitive experiment — sparse-RAG base
runner for training, seed 11 for training, include_failures=True,
use_metacognitive_proposer=True, skip_gate=True, multi-policy.
Store: v0.4_storeD.db.

Matrix: all 6 configurations × 3 seeds on the 19 held-out test cases.
  1. Zero-shot                      (no policy, no memory)
  2. Zero-shot + memory             (no policy, Store D memory)
  3. Sparse-RAG                     (200-char truncation, no memory)
  4. Sparse-RAG + memory            (200-char truncation, Store D)
  5. Full-RAG                       (complete policy, no memory)
  6. Full-RAG + memory              (complete policy, Store D memory)

The hypothesis under test: metacognitive memory recovers what sparse-
RAG truncation loses, on held-out within-domain cases, reproducibly
across 3 seeds, on a 19-case test set (wider CI than v0.3's n=9).

Cost estimate: ~$2.30
  Training Store D (41 cases, metacognitive + failure-driven): ~$0.26
  Eval 6 configs × 3 seeds × 19 cases = 342 calls:              ~$2.05
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medreason.llm import ClaudeLLMClient
from medreason.ontology import Outcome
from medreason.retrieval.embedder import FakeEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.eval.metrics import macro_f1
from medreason_bench.splits import load_split
from medreason_bench.train import TrainingConfig, run_training


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
MANIFEST_DIR = SPLITS_ROOT / "v0.4_holdout"
OUT_PATH = Path(__file__).parent / "experiment_v04_metacognitive_matrix.json"

POLICY_MAX_CHARS = 200
TRAIN_SEED = 11
EVAL_SEEDS = [11, 17, 23]
STORE_D_PATH = STORES_ROOT / "v0.4_storeD_metacognitive.db"

CONFIGS = [
    # (label, include_policy, policy_max_chars, use_memory)
    ("zero_shot",         False, None,              False),
    ("zero_shot_memory",  False, None,              True),
    ("sparse_rag",        True,  POLICY_MAX_CHARS,  False),
    ("sparse_rag_memory", True,  POLICY_MAX_CHARS,  True),
    ("full_rag",          True,  None,              False),
    ("full_rag_memory",   True,  None,              True),
]


def _build_base_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


def _train_store_d(train_cases: list) -> dict:
    print(f"\n[v0.4] training Store D (metacognitive + failure-driven) on "
          f"{len(train_cases)} cases")
    print(f"[v0.4]   proposer prompt: metacognitive_rule_proposer.txt")
    print(f"[v0.4]   base runner: sparse-RAG ({POLICY_MAX_CHARS} chars)")
    print(f"[v0.4]   seed: {TRAIN_SEED}")
    print(f"[v0.4]   store path: {STORE_D_PATH}")

    if STORE_D_PATH.exists():
        STORE_D_PATH.unlink()
    STORE_D_PATH.parent.mkdir(parents=True, exist_ok=True)

    runner = _build_base_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    critic_llm = _build_llm()
    proposer_llm = _build_llm()
    failure_llm = _build_llm()

    conn = sqlite3.connect(str(STORE_D_PATH))
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
        version="v0.4_metacognitive",
        split="train",
        gate_k=5,
        gate_seed=TRAIN_SEED,
        skip_gate=True,
        embedder=FakeEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=TRAIN_SEED,
        include_failures=True,
        failure_analyzer_llm=failure_llm,
        use_metacognitive_proposer=True,
    )
    report = run_training(config)
    conn.commit()
    conn.close()

    print()
    print(f"[v0.4]   training done in {report.elapsed_seconds:.1f}s")
    print(f"[v0.4]   agent correct / wrong:        "
          f"{report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"[v0.4]   metacognitive rules promoted: {report.n_rules_promoted}")
    print(f"[v0.4]   failure rules promoted:       {report.n_failure_rules_promoted}")
    print(f"[v0.4]   total rules in Store D:       "
          f"{report.n_rules_promoted + report.n_failure_rules_promoted}")
    print(f"[v0.4]   training cost: ${report.cost_total:.4f}")

    return {
        "n_train_cases": len(train_cases),
        "n_agent_correct": report.n_agent_correct,
        "n_agent_wrong": report.n_agent_wrong,
        "n_rules_promoted_normal": report.n_rules_promoted,
        "n_rules_promoted_failure": report.n_failure_rules_promoted,
        "n_rules_total": report.n_rules_promoted + report.n_failure_rules_promoted,
        "cost_total": round(report.cost_total, 5),
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }


def _eval_one(label: str, eval_cases: list, *,
              include_policy: bool, policy_max_chars: int | None,
              use_memory: bool, seed: int) -> dict:
    base_runner = _build_base_runner(
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )

    if use_memory:
        conn = sqlite3.connect(str(STORE_D_PATH))
        rule_store = RuleStore(conn)
        runner = MemoryRunner(
            base_runner=base_runner,
            store=rule_store,
            embedder=FakeEmbedder(),
            reranker_llm=None,
            tier3_top_k=3,
        )
    else:
        conn = None
        runner = base_runner

    results = []
    total_cost = 0.0
    total_in = 0
    total_out = 0

    for case in eval_cases:
        r = runner.run(case, seed=seed)
        ok = "OK" if r.correct else "WR"
        applied_str = ""
        if use_memory:
            n_applied = sum(1 for a in r.applied_rules if a.applied)
            applied_str = f"  app={n_applied}/{len(r.applied_rules)}"
        print(f"    {case.case_id:12s} gt={case.ground_truth_outcome.value:8s} "
              f"pred={r.determination.value:8s} {ok}{applied_str}  ${r.cost_usd:.5f}")
        total_cost += r.cost_usd
        total_in += r.input_tokens
        total_out += r.output_tokens

        pc = {
            "case_id": r.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "determination": r.determination.value,
            "correct": r.correct,
            "confidence": r.confidence,
            "cost_usd": round(r.cost_usd, 5),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "seed": seed,
        }
        if use_memory:
            pc["retrieved_rule_ids"] = list(r.retrieved_rule_ids)
            pc["applied_rules"] = [
                {"rule_id": a.rule_id, "applied": a.applied,
                 "rationale": a.rationale}
                for a in r.applied_rules
            ]
        results.append((r, case, pc))

    if conn is not None:
        conn.close()

    agent_results = [t[0] for t in results]
    cases_by_id = {t[1].case_id: t[1] for t in results}
    n_correct = sum(1 for r in agent_results if r.correct)
    accuracy = n_correct / len(agent_results) if agent_results else 0.0
    f1 = macro_f1(agent_results, cases_by_id)

    return {
        "label": label,
        "seed": seed,
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "macro_f1": f1,
        "cost_usd": round(total_cost, 5),
        "avg_total_tokens": (total_in + total_out) / len(results) if results else 0,
        "cost_per_case": round(total_cost / len(results), 5) if results else 0,
        "per_case_results": [t[2] for t in results],
    }


def _run_matrix(eval_cases: list) -> dict:
    print(f"\n[v0.4] running 6-config x 3-seed matrix on "
          f"{len(eval_cases)} held-out cases")
    matrix = {}
    total_cost = 0.0

    for label, include_policy, policy_max_chars, use_memory in CONFIGS:
        print(f"\n  === config: {label} ===")
        per_seed = []
        for seed in EVAL_SEEDS:
            print(f"    -- seed {seed} --")
            run = _eval_one(label, eval_cases,
                            include_policy=include_policy,
                            policy_max_chars=policy_max_chars,
                            use_memory=use_memory, seed=seed)
            per_seed.append(run)
            total_cost += run["cost_usd"]
            print(f"    seed {seed}: acc={run['n_correct']}/{run['n_cases']}  "
                  f"f1={run['macro_f1']:.3f}  cost ${run['cost_usd']:.4f}")

        accs = [r["accuracy"] for r in per_seed]
        f1s = [r["macro_f1"] for r in per_seed]
        toks = [r["avg_total_tokens"] for r in per_seed]
        costs = [r["cost_per_case"] for r in per_seed]

        matrix[label] = {
            "accuracy_mean": statistics.mean(accs),
            "accuracy_stdev": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "accuracy_per_seed": accs,
            "macro_f1_mean": statistics.mean(f1s),
            "avg_total_tokens_mean": statistics.mean(toks),
            "cost_per_case_mean": statistics.mean(costs),
            "n_cases": per_seed[0]["n_cases"],
            "per_seed": per_seed,
        }
        n = per_seed[0]["n_cases"]
        print(f"  >>> {label}: "
              f"acc={statistics.mean(accs)*n:.1f}/{n} "
              f"(±{statistics.stdev(accs)*100 if len(accs) > 1 else 0:.1f}%)  "
              f"f1={statistics.mean(f1s):.3f}  "
              f"cost/case=${statistics.mean(costs):.5f}")

    return {"configs": matrix, "total_cost_usd": round(total_cost, 5)}


def _case_matrix(matrix: dict, eval_cases: list) -> dict:
    out = {}
    for case in eval_cases:
        out[case.case_id] = {"ground_truth": case.ground_truth_outcome.value}
        for label, _, _, _ in CONFIGS:
            runs = matrix["configs"][label]["per_seed"]
            corrects = [
                next((pc["correct"] for pc in r["per_case_results"]
                      if pc["case_id"] == case.case_id), False)
                for r in runs
            ]
            n_right = sum(1 for c in corrects if c)
            out[case.case_id][label] = {
                "n_right": n_right,
                "n_total": len(corrects),
                "majority_correct": n_right >= 2,
            }
    return out


def main():
    started = time.time()
    print("=" * 72)
    print("Experiment v0.4 — metacognitive matrix on expanded Aetna fixture")
    print("=" * 72)

    train_cases = load_split(MANIFEST_DIR, "train")
    test_cases = load_split(MANIFEST_DIR, "test")
    print(f"manifest: {MANIFEST_DIR}")
    print(f"train: {len(train_cases)}  /  test: {len(test_cases)} held-out")

    train_summary = _train_store_d(train_cases)
    matrix = _run_matrix(test_cases)
    case_mat = _case_matrix(matrix, test_cases)

    elapsed = time.time() - started
    total_cost = train_summary["cost_total"] + matrix["total_cost_usd"]

    payload = {
        "experiment": "v0.4_metacognitive_matrix",
        "manifest": str(MANIFEST_DIR),
        "store": str(STORE_D_PATH),
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "policy_max_chars": POLICY_MAX_CHARS,
        "train_seed": TRAIN_SEED,
        "eval_seeds": EVAL_SEEDS,
        "train": train_summary,
        "matrix": matrix,
        "case_level_matrix": case_mat,
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    print("=" * 72)
    print("v0.4 results (19-case held-out, 3 seeds)")
    print("=" * 72)
    n = test_cases.__len__()
    print(f"  {'config':<24s} {'accuracy':<16s} {'macro F1':<10s} "
          f"{'tokens':<10s} {'cost/case':<12s}")
    print("  " + "-"*72)
    for label, _, _, _ in CONFIGS:
        m = matrix["configs"][label]
        acc_n = m["accuracy_mean"] * m["n_cases"]
        acc_str = f"{acc_n:.1f}/{m['n_cases']} ({m['accuracy_mean']*100:.1f}%)"
        print(f"  {label:<24s} {acc_str:<16s} {m['macro_f1_mean']:<10.3f} "
              f"{m['avg_total_tokens_mean']:<10.0f} ${m['cost_per_case_mean']:.5f}")

    print()
    print(f"training cost: ${train_summary['cost_total']:.4f}")
    print(f"eval cost:     ${matrix['total_cost_usd']:.4f}")
    print(f"TOTAL COST:    ${total_cost:.4f}")
    print(f"elapsed:       {elapsed:.1f}s")
    print(f"saved to:      {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
