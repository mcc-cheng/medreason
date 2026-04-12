"""Experiment v0.3 metacognitive — test the metacognitive rule proposer.

Hypothesis (user's framing): the additive rule proposer produces rules
that are too specific to their source case to transfer ("verify X ≥Y
weeks") and too authoritative to be safely retrieved on
structurally-different cases (the aetna_008 BoneMRI false application
is the canonical failure mode). The fix is to extract METACOGNITIVE
rules that describe the AI's default reasoning on a case type and
the specific override/protection that prevents failure.

This experiment trains a new Store C on the v0.3 train cases using
the metacognitive rule proposer prompt (same pipeline otherwise:
sparse-RAG base runner, seed 11, include_failures=True, skip_gate)
and re-runs the 3 memory configs on the 9 held-out test cases for
3 seeds each.

The comparison point is the baseline v0.3 Store B matrix results
(experiment_v03_aetna_matrix.json):
    zero-shot                 8/9  (88.9%)
    zero-shot + memory        7/9  (77.8%)  <- memory HURT
    sparse-RAG                6/9  (66.7%)
    sparse-RAG + memory       6/9  (66.7%)
    full-RAG                  9/9  (100%)
    full-RAG + memory         9/9  (100%)

Success criteria:
- zero-shot + metacognitive memory ≥ 8/9 (at minimum, don't hurt)
- sparse-RAG + metacognitive memory > 6/9 (the strongest test —
  memory should recover what sparse truncation lost)
- Per-case: aetna_008 should NOT flip right->wrong under memory
  anymore (the core hypothesis — metacognitive rules protect the
  correct default instead of overriding it with irrelevant retrievals)

Cost estimate:
  Training Store C:                             ~$0.15
  3 memory configs × 3 seeds × 9 cases = 81 calls × ~$0.008 ≈ $0.65
  Total:                                         ~$0.80
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
from medreason.retrieval.embedder import FakeEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.eval.metrics import macro_f1
from medreason_bench.splits import load_split
from medreason_bench.train import TrainingConfig, run_training


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
MANIFEST_DIR = SPLITS_ROOT / "v0.3_holdout"
OUT_PATH = Path(__file__).parent / "experiment_v03_metacognitive.json"

POLICY_MAX_CHARS = 200
TRAIN_SEED = 11
EVAL_SEEDS = [11, 17, 23]
STORE_C_PATH = STORES_ROOT / "v0.3_storeC_metacognitive.db"

# Baseline (Store B) results for direct comparison
BASELINE_MATRIX = {
    "zero_shot":         {"acc": 8/9, "n_correct": 8},
    "zero_shot_memory":  {"acc": 7/9, "n_correct": 7},
    "sparse_rag":        {"acc": 6/9, "n_correct": 6},
    "sparse_rag_memory": {"acc": 6/9, "n_correct": 6},
    "full_rag":          {"acc": 9/9, "n_correct": 9},
    "full_rag_memory":   {"acc": 9/9, "n_correct": 9},
}

MEMORY_CONFIGS = [
    # label, include_policy, policy_max_chars
    ("zero_shot_memory",  False, None),
    ("sparse_rag_memory", True,  POLICY_MAX_CHARS),
    ("full_rag_memory",   True,  None),
]


def _build_base_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


def _train_store_c(train_cases: list) -> dict:
    print(f"\n[v0.3 meta] training Store C (metacognitive proposer) on "
          f"{len(train_cases)} cases")
    print(f"[v0.3 meta]   proposer prompt: metacognitive_rule_proposer.txt")
    print(f"[v0.3 meta]   base runner: sparse-RAG ({POLICY_MAX_CHARS} chars)")
    print(f"[v0.3 meta]   seed: {TRAIN_SEED}")
    print(f"[v0.3 meta]   store path: {STORE_C_PATH}")

    if STORE_C_PATH.exists():
        STORE_C_PATH.unlink()
    STORE_C_PATH.parent.mkdir(parents=True, exist_ok=True)

    runner = _build_base_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    critic_llm = _build_llm()
    proposer_llm = _build_llm()
    failure_llm = _build_llm()

    conn = sqlite3.connect(str(STORE_C_PATH))
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
        version="v0.3_metacognitive",
        split="train",
        gate_k=5,
        gate_seed=TRAIN_SEED,
        skip_gate=True,
        embedder=FakeEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=TRAIN_SEED,
        include_failures=True,
        failure_analyzer_llm=failure_llm,
        use_metacognitive_proposer=True,  # <-- THE KEY CHANGE
    )
    report = run_training(config)
    conn.commit()
    conn.close()

    print()
    print(f"[v0.3 meta]   training done in {report.elapsed_seconds:.1f}s")
    print(f"[v0.3 meta]   agent correct / wrong:        "
          f"{report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"[v0.3 meta]   metacognitive rules promoted:  "
          f"{report.n_rules_promoted}")
    print(f"[v0.3 meta]   failure-driven rules promoted: "
          f"{report.n_failure_rules_promoted}")
    print(f"[v0.3 meta]   total rules in Store C:        "
          f"{report.n_rules_promoted + report.n_failure_rules_promoted}")
    print(f"[v0.3 meta]   rejection reasons (first 5):")
    for reason in report.rejection_reasons[:5]:
        print(f"     - {reason[:100]}")
    print(f"[v0.3 meta]   total rejection count:         "
          f"{report.n_rules_rejected + report.n_failure_rules_rejected}")
    print(f"[v0.3 meta]   training cost: ${report.cost_total:.4f}")

    return {
        "n_train_cases": len(train_cases),
        "n_agent_correct": report.n_agent_correct,
        "n_agent_wrong": report.n_agent_wrong,
        "n_rules_promoted_normal": report.n_rules_promoted,
        "n_rules_rejected_normal": report.n_rules_rejected,
        "n_rules_promoted_failure": report.n_failure_rules_promoted,
        "n_rules_rejected_failure": report.n_failure_rules_rejected,
        "n_rules_total": report.n_rules_promoted + report.n_failure_rules_promoted,
        "rejection_reasons": report.rejection_reasons[:10],
        "cost_total": round(report.cost_total, 5),
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }


def _eval_one_memory_config(
    label: str,
    eval_cases: list,
    *,
    include_policy: bool,
    policy_max_chars: int | None,
    seed: int,
) -> dict:
    base_runner = _build_base_runner(
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )
    conn = sqlite3.connect(str(STORE_C_PATH))
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
    total_in = 0
    total_out = 0

    for case in eval_cases:
        r = runner.run(case, seed=seed)
        ok = "OK" if r.correct else "WRONG"
        n_applied = sum(1 for a in r.applied_rules if a.applied)
        print(
            f"    {case.case_id:12s} gt={case.ground_truth_outcome.value:10s} "
            f"pred={r.determination.value:22s} {ok}  "
            f"applied={n_applied}/{len(r.applied_rules)}  ${r.cost_usd:.5f}"
        )
        total_cost += r.cost_usd
        total_in += r.input_tokens
        total_out += r.output_tokens

        results.append({
            "case_id": r.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "determination": r.determination.value,
            "correct": r.correct,
            "confidence": r.confidence,
            "cost_usd": round(r.cost_usd, 5),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "seed": seed,
            "retrieved_rule_ids": list(r.retrieved_rule_ids),
            "applied_rules": [
                {"rule_id": a.rule_id, "applied": a.applied,
                 "rationale": a.rationale}
                for a in r.applied_rules
            ],
        })
    conn.close()

    agent_results = [_AgentResultShim(pc) for pc in results]
    cases_by_id = {c.case_id: c for c in eval_cases}
    n_correct = sum(1 for r in results if r["correct"])
    f1 = macro_f1(agent_results, cases_by_id)

    return {
        "label": label,
        "seed": seed,
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else 0.0,
        "macro_f1": f1,
        "cost_usd": round(total_cost, 5),
        "avg_total_tokens": (total_in + total_out) / len(results) if results else 0,
        "cost_per_case": round(total_cost / len(results), 5) if results else 0,
        "per_case_results": results,
    }


class _AgentResultShim:
    """Thin shim to adapt our per_case dict back to the AgentResult
    fields macro_f1 needs."""
    def __init__(self, pc: dict):
        from medreason.ontology import Outcome
        self.case_id = pc["case_id"]
        self.determination = Outcome(pc["determination"])
        self.correct = pc["correct"]


def _run_matrix(eval_cases: list) -> dict:
    print(f"\n[v0.3 meta] running memory configs on "
          f"{len(eval_cases)} held-out cases")
    matrix = {}
    total_cost = 0.0

    for label, include_policy, policy_max_chars in MEMORY_CONFIGS:
        print(f"\n  === config: {label} ===")
        per_seed = []
        for seed in EVAL_SEEDS:
            print(f"    -- seed {seed} --")
            run_result = _eval_one_memory_config(
                label, eval_cases,
                include_policy=include_policy,
                policy_max_chars=policy_max_chars,
                seed=seed,
            )
            per_seed.append(run_result)
            total_cost += run_result["cost_usd"]
            print(f"    seed {seed}: acc={run_result['n_correct']}/{run_result['n_cases']} "
                  f"f1={run_result['macro_f1']:.3f}  cost ${run_result['cost_usd']:.4f}")

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
        print(f"  >>> {label}: "
              f"acc={statistics.mean(accs)*per_seed[0]['n_cases']:.1f}/{per_seed[0]['n_cases']} "
              f"±{statistics.stdev(accs)*100 if len(accs) > 1 else 0:.1f}%  "
              f"f1={statistics.mean(f1s):.3f}  "
              f"cost/case=${statistics.mean(costs):.5f}")

    return {"configs": matrix, "total_cost_usd": round(total_cost, 5)}


def _case_level_matrix(matrix: dict, eval_cases: list) -> dict:
    out = {}
    for case in eval_cases:
        out[case.case_id] = {"ground_truth": case.ground_truth_outcome.value}
        for label, _, _ in MEMORY_CONFIGS:
            per_seed_runs = matrix["configs"][label]["per_seed"]
            corrects = [
                next((pc["correct"] for pc in run["per_case_results"]
                      if pc["case_id"] == case.case_id), False)
                for run in per_seed_runs
            ]
            n_right = sum(1 for c in corrects if c)
            out[case.case_id][label] = {
                "per_seed_correct": corrects,
                "majority_correct": n_right >= 2,
                "n_right": n_right,
                "n_total": len(corrects),
            }
    return out


def main():
    started = time.time()
    print("=" * 72)
    print("Experiment v0.3 metacognitive — prompt-swap hypothesis test")
    print("=" * 72)

    train_cases = load_split(MANIFEST_DIR, "train")
    test_cases = load_split(MANIFEST_DIR, "test")
    print(f"manifest: {MANIFEST_DIR}")
    print(f"train: {len(train_cases)} cases  /  test: {len(test_cases)} held-out")

    train_summary = _train_store_c(train_cases)
    matrix = _run_matrix(test_cases)
    case_matrix = _case_level_matrix(matrix, test_cases)

    elapsed = time.time() - started
    total_cost = train_summary["cost_total"] + matrix["total_cost_usd"]

    payload = {
        "experiment": "v0.3_metacognitive",
        "hypothesis": (
            "Metacognitive rules describing AI failure modes transfer "
            "better than additive policy checks on held-out within-"
            "domain cases. The prompt changes; the architecture stays."
        ),
        "manifest": str(MANIFEST_DIR),
        "store": str(STORE_C_PATH),
        "train_seed": TRAIN_SEED,
        "eval_seeds": EVAL_SEEDS,
        "train": train_summary,
        "matrix": matrix,
        "case_level_matrix": case_matrix,
        "baseline_matrix": BASELINE_MATRIX,
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    print("=" * 72)
    print("Experiment v0.3 metacognitive results")
    print("=" * 72)
    print()
    print(f"  {'config':<24s} {'v0.3 baseline':<18s} {'metacognitive':<18s} {'delta':<10s}")
    print(f"  {'-'*24} {'-'*18} {'-'*18} {'-'*10}")
    for label, _, _ in MEMORY_CONFIGS:
        base_n = BASELINE_MATRIX[label]["n_correct"]
        base_acc = BASELINE_MATRIX[label]["acc"]
        meta = matrix["configs"][label]
        n = meta["n_cases"]
        meta_n = round(meta["accuracy_mean"] * n)
        meta_acc = meta["accuracy_mean"]
        delta = meta_n - base_n
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {label:<24s} {base_n}/{n} ({base_acc*100:.1f}%)      "
              f"  {meta_n}/{n} ({meta_acc*100:.1f}%)      "
              f"  {arrow}{abs(delta)}")
    print()
    print("Per-case correctness (majority across 3 seeds):")
    print(f"  {'case':<12s} {'gt':<10s}", end="")
    for label, _, _ in MEMORY_CONFIGS:
        short = label.replace("_memory", "+m")[:14]
        print(f" {short:<16s}", end="")
    print()
    for case_id in sorted(case_matrix.keys()):
        row = case_matrix[case_id]
        print(f"  {case_id:<12s} {row['ground_truth']:<10s}", end="")
        for label, _, _ in MEMORY_CONFIGS:
            cell = row[label]
            mark = "OK" if cell["majority_correct"] else "WR"
            print(f" {mark} {cell['n_right']}/{cell['n_total']}          ", end="")
        print()
    print()
    print(f"training cost: ${train_summary['cost_total']:.4f}")
    print(f"eval cost:     ${matrix['total_cost_usd']:.4f}")
    print(f"TOTAL COST:    ${total_cost:.4f}")
    print(f"elapsed:       {elapsed:.1f}s")
    print(f"saved to:      {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
