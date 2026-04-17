"""Drug discovery v0.2 — 200 synthetic cases, generalization test.

Pipeline:
  1. Generate 200 synthetic drug discovery cases (seeded RNG)
  2. Stratify 50 train / 150 test
  3. Seed metacognitive rules + train on 50 cases
  4. Eval 4 configs × 3 seeds on 150 held-out test cases
  5. The core question: does memory trained on 50 cases
     generalize to 150 unseen cases?

Configs (trimmed to the ones that matter):
  1. sparse_rag           (200-char guidelines, no memory)
  2. sparse_rag_memory    (200-char guidelines, memory)
  3. full_rag             (full guidelines, no memory)
  4. full_rag_memory      (full guidelines, memory)

Cost estimate: ~$8-12 (training) + ~$10-15 (eval 150 × 4 × 3)
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medreason.llm import ClaudeLLMClient
from medreason.ontology import RuleStatus
from medreason.retrieval.embedder import OpenAIEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.data.drugdisc_synthetic import build_synthetic_drugdisc_cases
from medreason_bench.data.drugdisc_rules import build_drugdisc_seed_rules
from medreason_bench.eval.metrics import macro_f1
from medreason_bench.splits import SplitRatios, stratify, write_manifest, verify_manifest
from medreason_bench.train import TrainingConfig, run_training


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
MANIFEST_DIR = SPLITS_ROOT / "dd_v02_synthetic"
STORE_PATH = STORES_ROOT / "dd_v02_synthetic.db"
OUT_PATH = Path(__file__).parent / "experiment_drugdisc_v02_synthetic.json"
LOG_PATH = Path(__file__).parent / "experiment_drugdisc_v02_synthetic.log"

SYSTEM_PROMPT = "system_drugdisc.txt"
POLICY_MAX_CHARS = 200
TRAIN_SEED = 11
EVAL_SEEDS = [11, 17, 23]

CONFIGS = [
    # (label,              include_policy, policy_max_chars, use_memory)
    ("sparse_rag",         True,  POLICY_MAX_CHARS,  False),
    ("sparse_rag_memory",  True,  POLICY_MAX_CHARS,  True),
    # ("full_rag",           True,  None,              False),
    # ("full_rag_memory",    True,  None,              True),
]


def _build_runner(*, include_policy, policy_max_chars):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
        system_prompt_file=SYSTEM_PROMPT,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


# ── Phase 1: Generate + split ─────────────────────────────────────────────

def _build_manifest():
    print("[dd v0.2] generating 200 synthetic cases")
    cases = build_synthetic_drugdisc_cases(seed=42)
    print(f"  generated {len(cases)} cases")

    # 25/75 train/test (50 train, 150 test)
    ratios = SplitRatios(train=0.25, dev=0.0, test=0.75)
    splits = stratify(cases, ratios=ratios, seed=42)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(splits, MANIFEST_DIR)
    verify_manifest(MANIFEST_DIR)

    for name in ("train", "test"):
        oc = Counter(c.ground_truth_outcome.value for c in splits[name])
        print(f"  {name}: {len(splits[name])} cases  {dict(oc)}")

    return splits["train"], splits["test"]


# ── Phase 2: Train ────────────────────────────────────────────────────────

def _train_store(train_cases):
    print(f"\n[dd v0.2] training store on {len(train_cases)} cases")
    print(f"  store: {STORE_PATH}")

    if STORE_PATH.exists():
        STORE_PATH.unlink()
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(STORE_PATH))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    # Seed rules
    seed_rules = build_drugdisc_seed_rules()
    for r in seed_rules:
        rule_store.put(r)
    print(f"  seeded {len(seed_rules)} hand-crafted rules")

    runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    config = TrainingConfig(
        runner=runner,
        critic_llm=_build_llm(),
        proposer_llm=_build_llm(),
        store=rule_store,
        trace_store=trace_store,
        policy=None,
        train_cases=train_cases,
        version="dd_v02",
        split="train",
        gate_k=5,
        gate_seed=TRAIN_SEED,
        skip_gate=True,
        embedder=OpenAIEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=TRAIN_SEED,
        include_failures=True,
        failure_analyzer_llm=_build_llm(),
        use_metacognitive_proposer=True,
    )
    report = run_training(config)
    conn.commit()

    total_rules = rule_store.count(RuleStatus.ACTIVE)
    conn.close()

    print(f"\n  training done in {report.elapsed_seconds:.1f}s")
    print(f"  agent correct / wrong: {report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"  extracted rules: {report.n_rules_promoted} normal + {report.n_failure_rules_promoted} failure")
    print(f"  total rules in store: {total_rules}")
    print(f"  training cost: ${report.cost_total:.4f}")

    return {
        "n_train": len(train_cases),
        "n_seed_rules": len(seed_rules),
        "n_rules_total": total_rules,
        "n_rules_normal": report.n_rules_promoted,
        "n_rules_failure": report.n_failure_rules_promoted,
        "n_agent_correct": report.n_agent_correct,
        "n_agent_wrong": report.n_agent_wrong,
        "cost": round(report.cost_total, 5),
        "elapsed_s": round(report.elapsed_seconds, 1),
    }


# ── Phase 3: Eval matrix ─────────────────────────────────────────────────

def _eval_one(label, cases, *, include_policy, policy_max_chars, use_memory, seed):
    base = _build_runner(include_policy=include_policy, policy_max_chars=policy_max_chars)
    conn = None
    if use_memory:
        conn = sqlite3.connect(str(STORE_PATH))
        reranker = ClaudeLLMClient(model=resolve_claude_model("haiku"))
        runner = MemoryRunner(
            base_runner=base, store=RuleStore(conn),
            embedder=OpenAIEmbedder(), reranker_llm=reranker, tier3_top_k=5)
    else:
        runner = base

    results = []
    cost = 0.0
    tin = tout = 0
    for c in cases:
        r = runner.run(c, seed=seed)
        ok = "OK" if r.correct else "WR"
        app = ""
        if use_memory:
            na = sum(1 for a in r.applied_rules if a.applied)
            app = f" app={na}/{len(r.applied_rules)}"
        print(f"    {c.case_id:12s} gt={c.ground_truth_outcome.value:8s} "
              f"pred={r.determination.value:8s} {ok}{app} ${r.cost_usd:.5f}")
        cost += r.cost_usd
        tin += r.input_tokens
        tout += r.output_tokens
        pc = {
            "case_id": r.case_id,
            "ground_truth": c.ground_truth_outcome.value,
            "determination": r.determination.value,
            "correct": r.correct,
            "confidence": r.confidence,
            "cost_usd": round(r.cost_usd, 5),
            "seed": seed,
            "difficulty": c.difficulty.value,
        }
        if use_memory:
            pc["retrieved_rule_ids"] = list(r.retrieved_rule_ids)
            pc["applied_rules"] = [
                {"rule_id": a.rule_id, "applied": a.applied, "rationale": a.rationale}
                for a in r.applied_rules]
        results.append((r, c, pc))
    if conn:
        conn.close()

    ars = [t[0] for t in results]
    cbi = {t[1].case_id: t[1] for t in results}
    nc = sum(1 for r in ars if r.correct)
    return {
        "label": label, "seed": seed, "n_cases": len(results),
        "n_correct": nc,
        "accuracy": nc / len(results) if results else 0,
        "macro_f1": macro_f1(ars, cbi),
        "cost_usd": round(cost, 5),
        "avg_total_tokens": (tin + tout) / len(results) if results else 0,
        "cost_per_case": round(cost / len(results), 5) if results else 0,
        "per_case_results": [t[2] for t in results],
    }


def _run_matrix(test_cases):
    n = len(test_cases)
    print(f"\n[dd v0.2] running {len(CONFIGS)}-config x {len(EVAL_SEEDS)}-seed matrix on {n} held-out")
    matrix = {}
    total_cost = 0.0
    for label, inc_pol, pmc, use_mem in CONFIGS:
        print(f"\n  === {label} ===")
        per_seed = []
        for seed in EVAL_SEEDS:
            print(f"    -- seed {seed} --")
            run = _eval_one(label, test_cases,
                            include_policy=inc_pol, policy_max_chars=pmc,
                            use_memory=use_mem, seed=seed)
            per_seed.append(run)
            total_cost += run["cost_usd"]
            print(f"    seed {seed}: {run['n_correct']}/{run['n_cases']}  "
                  f"f1={run['macro_f1']:.3f}  ${run['cost_usd']:.4f}")
        accs = [r["accuracy"] for r in per_seed]
        matrix[label] = {
            "accuracy_mean": statistics.mean(accs),
            "accuracy_stdev": statistics.stdev(accs) if len(accs) > 1 else 0,
            "accuracy_per_seed": accs,
            "macro_f1_mean": statistics.mean(r["macro_f1"] for r in per_seed),
            "avg_total_tokens_mean": statistics.mean(r["avg_total_tokens"] for r in per_seed),
            "cost_per_case_mean": statistics.mean(r["cost_per_case"] for r in per_seed),
            "n_cases": per_seed[0]["n_cases"],
            "per_seed": per_seed,
        }
        n = per_seed[0]["n_cases"]
        print(f"  >>> {label}: {statistics.mean(accs)*n:.1f}/{n}  "
              f"f1={matrix[label]['macro_f1_mean']:.3f}")
    return {"configs": matrix, "total_cost_usd": round(total_cost, 5)}


def main():
    started = time.time()
    print("=" * 72)
    print("Drug Discovery v0.2 — 200 synthetic cases, generalization test")
    print("=" * 72)

    # Phase 1
    train_cases, test_cases = _build_manifest()

    # Phase 2
    train_report = _train_store(train_cases)

    # Phase 3
    matrix = _run_matrix(test_cases)

    # Summary
    elapsed = time.time() - started
    print()
    print("=" * 72)
    print("Drug Discovery v0.2 Results (150 held-out cases)")
    print("=" * 72)
    print(f"  {'config':<24s} {'accuracy':<16s} {'F1':<8s} {'cost/case':<10s}")
    print("  " + "-" * 58)
    for label, _, _, _ in CONFIGS:
        m = matrix["configs"][label]
        n = m["n_cases"]
        acc_n = m["accuracy_mean"] * n
        std = m["accuracy_stdev"] * 100
        print(f"  {label:<24s} {acc_n:5.1f}/{n} ({m['accuracy_mean']*100:5.1f}%) +/-{std:4.1f}%  "
              f"{m['macro_f1_mean']:5.3f}  ${m['cost_per_case_mean']:.5f}")
    print()
    print(f"  training: {train_report['n_rules_total']} rules, ${train_report['cost']:.4f}")
    print(f"  total cost: ${matrix['total_cost_usd'] + train_report['cost']:.4f}")
    print(f"  elapsed:    {elapsed:.1f}s")

    # Save
    payload = {
        "experiment": "drugdisc_v02_synthetic_generalization",
        "n_total_cases": 200,
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "training": train_report,
        "matrix": matrix,
        "total_cost_usd": round(matrix["total_cost_usd"] + train_report["cost"], 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  saved:      {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
