"""Drug discovery v0.1 — EVAL ONLY re-run (training already done).

The full run crashed due to network error after training completed.
This script skips training and re-runs just the 6-config eval matrix
using the existing store at dd_v01_metacognitive.db.
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
from medreason.ontology import RuleStatus
from medreason.retrieval.embedder import OpenAIEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore
from medreason_bench.eval.metrics import macro_f1
from medreason_bench.splits import load_split


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
MANIFEST_DIR = SPLITS_ROOT / "dd_v01"
OUT_PATH = Path(__file__).parent / "experiment_drugdisc_v01_matrix.json"

SYSTEM_PROMPT = "system_drugdisc.txt"
POLICY_MAX_CHARS = 200
EVAL_SEEDS = [11, 17, 23]
STORE_PATH = STORES_ROOT / "dd_v01_metacognitive.db"

CONFIGS = [
    ("zero_shot",         False, None,              False),
    ("zero_shot_memory",  False, None,              True),
    ("sparse_rag",        True,  POLICY_MAX_CHARS,  False),
    ("sparse_rag_memory", True,  POLICY_MAX_CHARS,  True),
    ("full_rag",          True,  None,              False),
    ("full_rag_memory",   True,  None,              True),
]


def _build_runner(*, include_policy, policy_max_chars):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
        system_prompt_file=SYSTEM_PROMPT,
    )


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
        print(f"    {c.case_id:8s} gt={c.ground_truth_outcome.value:8s} "
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


def main():
    started = time.time()
    print("=" * 72)
    print("Drug Discovery v0.1 — EVAL ONLY (store already trained)")
    print("=" * 72)

    test_cases = load_split(MANIFEST_DIR, "test")
    print(f"manifest: {MANIFEST_DIR}")
    print(f"test: {len(test_cases)} held-out cases")
    print(f"store: {STORE_PATH}")

    # Check store
    conn = sqlite3.connect(str(STORE_PATH))
    rs = RuleStore(conn)
    n_rules = rs.count(RuleStatus.ACTIVE)
    conn.close()
    print(f"store has {n_rules} active rules")

    # Run matrix
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

    # Case matrix
    cm = {}
    for c in test_cases:
        cm[c.case_id] = {"ground_truth": c.ground_truth_outcome.value}
        for label, _, _, _ in CONFIGS:
            runs = matrix[label]["per_seed"]
            corrects = [
                next((pc["correct"] for pc in r["per_case_results"]
                      if pc["case_id"] == c.case_id), False)
                for r in runs]
            nr = sum(1 for x in corrects if x)
            cm[c.case_id][label] = {"n_right": nr, "n_total": len(corrects),
                                    "majority_correct": nr >= 2}

    elapsed = time.time() - started
    payload = {
        "experiment": "drugdisc_v01_metacognitive_matrix",
        "manifest": str(MANIFEST_DIR),
        "store": str(STORE_PATH),
        "n_rules_in_store": n_rules,
        "n_test": len(test_cases),
        "matrix": {"configs": matrix, "total_cost_usd": round(total_cost, 5)},
        "case_level_matrix": cm,
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    print("=" * 72)
    print("Drug Discovery v0.1 Results")
    print("=" * 72)
    print(f"  {'config':<24s} {'accuracy':<16s} {'F1':<8s} {'cost/case':<10s}")
    print("  " + "-" * 58)
    for label, _, _, _ in CONFIGS:
        m = matrix[label]
        n = m["n_cases"]
        acc_n = m["accuracy_mean"] * n
        std = m["accuracy_stdev"] * 100
        print(f"  {label:<24s} {acc_n:5.1f}/{n} ({m['accuracy_mean']*100:5.1f}%) +/-{std:4.1f}%  "
              f"{m['macro_f1_mean']:5.3f}  ${m['cost_per_case_mean']:.5f}")
    print()
    print(f"  total cost: ${total_cost:.4f}")
    print(f"  elapsed:    {elapsed:.1f}s")
    print(f"  saved:      {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
