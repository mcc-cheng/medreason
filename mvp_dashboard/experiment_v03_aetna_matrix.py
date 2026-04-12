"""Experiment v0.3 — Aetna within-domain held-out matrix.

The definitive YC-application experiment. Tests whether memory adds
value on a TRUE held-out within-domain generalization test:
  - All 30 cases are Aetna lumbar MRI (same payer + same condition)
  - Train memory on 21 stratified cases
  - Eval on a disjoint 9-case held-out set
  - 6 configurations × 3 seeds each
  - Store B = normal proposer + failure-driven (Phase 51)
  - Training base runner = sparse-RAG, seed=11 (config symmetric with
    the primary eval cell)

Configurations:
  1. Zero-shot           (no policy, no memory)
  2. Zero-shot + memory  (no policy, Store B memory)
  3. Sparse-RAG          (200-char truncated policy, no memory)
  4. Sparse-RAG + memory (200-char truncated policy, Store B memory)
  5. Full-RAG            (complete policy, no memory)
  6. Full-RAG + memory   (complete policy, Store B memory)

This is the experiment that decides the YC slide framing. If the
memory configs show held-out improvement on within-domain cases
across multiple seeds, the generalization thesis is validated. If
they don't, we need an architectural change before pitching.

Cost estimate: ~$1.15 (train ~$0.18, eval ~$0.97 for 162 calls)
Runtime: ~25-35 min
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
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
OUT_PATH = Path(__file__).parent / "experiment_v03_aetna_matrix.json"

POLICY_MAX_CHARS = 200
TRAIN_SEED = 11  # matches primary eval seed for config symmetry
EVAL_SEEDS = [11, 17, 23]
STORE_B_PATH = STORES_ROOT / "v0.3_storeB.db"

CONFIGS = [
    # (label, include_policy, policy_max_chars, use_memory)
    ("zero_shot",           False, None,              False),
    ("zero_shot_memory",    False, None,              True),
    ("sparse_rag",          True,  POLICY_MAX_CHARS,  False),
    ("sparse_rag_memory",   True,  POLICY_MAX_CHARS,  True),
    ("full_rag",            True,  None,              False),
    ("full_rag_memory",     True,  None,              True),
]


def _build_base_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


# ─────────────────────────────────────────────────────────────────────
# Phase 1 — Train Store B on the 21 train cases
# ─────────────────────────────────────────────────────────────────────


def _train_store_b(train_cases: list) -> dict:
    print(f"\n[v0.3] training Store B (normal + failure-driven) on "
          f"{len(train_cases)} train cases")
    print(f"[v0.3]   base runner: sparse-RAG (include_policy=True, "
          f"policy_max_chars={POLICY_MAX_CHARS})")
    print(f"[v0.3]   seed: {TRAIN_SEED}")
    print(f"[v0.3]   store path: {STORE_B_PATH}")

    if STORE_B_PATH.exists():
        STORE_B_PATH.unlink()
    STORE_B_PATH.parent.mkdir(parents=True, exist_ok=True)

    runner = _build_base_runner(
        include_policy=True, policy_max_chars=POLICY_MAX_CHARS
    )
    critic_llm = _build_llm()
    proposer_llm = _build_llm()
    failure_llm = _build_llm()

    conn = sqlite3.connect(str(STORE_B_PATH))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    config = TrainingConfig(
        runner=runner,
        critic_llm=critic_llm,
        proposer_llm=proposer_llm,
        store=rule_store,
        trace_store=trace_store,
        policy=None,  # multi-policy (each case has its own excerpt)
        train_cases=train_cases,
        version="v0.3",
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
    print(f"[v0.3]   training done in {report.elapsed_seconds:.1f}s")
    print(f"[v0.3]   agent correct / wrong:        "
          f"{report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"[v0.3]   normal rules promoted:         "
          f"{report.n_rules_promoted}")
    print(f"[v0.3]   failure-driven rules promoted: "
          f"{report.n_failure_rules_promoted}")
    print(f"[v0.3]   total rules in Store B:        "
          f"{report.n_rules_promoted + report.n_failure_rules_promoted}")
    print(f"[v0.3]   training cost: ${report.cost_total:.4f}")

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


# ─────────────────────────────────────────────────────────────────────
# Phase 2 — Run 6 configs × 3 seeds on the 9 held-out test cases
# ─────────────────────────────────────────────────────────────────────


def _eval_one_config(
    label: str,
    eval_cases: list,
    *,
    include_policy: bool,
    policy_max_chars: int | None,
    use_memory: bool,
    seed: int,
) -> dict:
    """Run a single (config, seed) over all eval cases. Returns
    per-case AgentResults + aggregated metrics."""
    base_runner = _build_base_runner(
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )

    if use_memory:
        conn = sqlite3.connect(str(STORE_B_PATH))
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
    total_input_tokens = 0
    total_output_tokens = 0

    for case in eval_cases:
        r = runner.run(case, seed=seed)
        ok = "OK" if r.correct else "WRONG"
        n_applied = sum(1 for a in r.applied_rules if a.applied) if r.applied_rules else 0
        applied_str = f"  applied={n_applied}/{len(r.applied_rules)}" if use_memory else ""
        print(
            f"    {case.case_id:12s} gt={case.ground_truth_outcome.value:10s} "
            f"pred={r.determination.value:22s} {ok}{applied_str}  "
            f"${r.cost_usd:.5f}"
        )
        total_cost += r.cost_usd
        total_input_tokens += r.input_tokens
        total_output_tokens += r.output_tokens

        result_dict = {
            "case_id": r.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "determination": r.determination.value,
            "correct": r.correct,
            "confidence": r.confidence,
            "cost_usd": round(r.cost_usd, 5),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_ms": r.latency_ms,
            "seed": seed,
        }
        if use_memory:
            result_dict["retrieved_rule_ids"] = list(r.retrieved_rule_ids)
            result_dict["applied_rules"] = [
                {"rule_id": a.rule_id, "applied": a.applied,
                 "rationale": a.rationale}
                for a in r.applied_rules
            ]
        results.append((r, case, result_dict))

    if conn is not None:
        conn.close()

    # Metrics
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
        "avg_input_tokens": total_input_tokens / len(results) if results else 0,
        "avg_output_tokens": total_output_tokens / len(results) if results else 0,
        "avg_total_tokens": (total_input_tokens + total_output_tokens) / len(results) if results else 0,
        "cost_per_case": round(total_cost / len(results), 5) if results else 0,
        "per_case_results": [t[2] for t in results],
    }


def _run_matrix(eval_cases: list) -> dict:
    print(f"\n[v0.3] running 6-config × 3-seed matrix on "
          f"{len(eval_cases)} held-out test cases")
    matrix = {}
    total_cost = 0.0

    for label, include_policy, policy_max_chars, use_memory in CONFIGS:
        print(f"\n  === config: {label} ===")
        per_seed = []
        for seed in EVAL_SEEDS:
            print(f"    -- seed {seed} --")
            run_result = _eval_one_config(
                label, eval_cases,
                include_policy=include_policy,
                policy_max_chars=policy_max_chars,
                use_memory=use_memory,
                seed=seed,
            )
            per_seed.append(run_result)
            total_cost += run_result["cost_usd"]
            print(f"    seed {seed}: acc={run_result['n_correct']}/{run_result['n_cases']} "
                  f"f1={run_result['macro_f1']:.3f}  cost ${run_result['cost_usd']:.4f}")

        # Aggregate across seeds
        accs = [r["accuracy"] for r in per_seed]
        f1s = [r["macro_f1"] for r in per_seed]
        tokens = [r["avg_total_tokens"] for r in per_seed]
        costs = [r["cost_per_case"] for r in per_seed]

        matrix[label] = {
            "accuracy_mean": statistics.mean(accs),
            "accuracy_stdev": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "accuracy_per_seed": accs,
            "macro_f1_mean": statistics.mean(f1s),
            "macro_f1_stdev": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "macro_f1_per_seed": f1s,
            "avg_total_tokens_mean": statistics.mean(tokens),
            "cost_per_case_mean": statistics.mean(costs),
            "n_cases": per_seed[0]["n_cases"],
            "per_seed": per_seed,
        }
        print(f"  >>> {label}: "
              f"acc={statistics.mean(accs):.3f}±{statistics.stdev(accs) if len(accs) > 1 else 0.0:.3f}  "
              f"f1={statistics.mean(f1s):.3f}  "
              f"tokens={statistics.mean(tokens):.0f}  "
              f"cost/case=${statistics.mean(costs):.5f}")

    return {"configs": matrix, "total_cost_usd": round(total_cost, 5)}


# ─────────────────────────────────────────────────────────────────────
# Phase 3 — Per-case correctness matrix (which cases flip where)
# ─────────────────────────────────────────────────────────────────────


def _case_level_matrix(matrix: dict, eval_cases: list) -> dict:
    """For each (case, config) pair, return the majority-vote correct
    flag across seeds (2/3 or 3/3 right = right for this table)."""
    out = {}
    for case in eval_cases:
        out[case.case_id] = {"ground_truth": case.ground_truth_outcome.value}
        for label in [c[0] for c in CONFIGS]:
            config_runs = matrix["configs"][label]["per_seed"]
            corrects = [
                next((pc["correct"] for pc in run["per_case_results"]
                      if pc["case_id"] == case.case_id), False)
                for run in config_runs
            ]
            n_right = sum(1 for c in corrects if c)
            out[case.case_id][label] = {
                "per_seed_correct": corrects,
                "majority_correct": n_right >= 2,
                "n_right": n_right,
                "n_total": len(corrects),
            }
    return out


# ─────────────────────────────────────────────────────────────────────
# Phase 4 — Format comparison table
# ─────────────────────────────────────────────────────────────────────


def _format_table(matrix: dict) -> str:
    lines = []
    lines.append("┌" + "─" * 26 + "┬" + "─" * 14 + "┬" + "─" * 12 + "┬"
                 + "─" * 10 + "┬" + "─" * 13 + "┐")
    lines.append(
        f"│ {'Configuration':<24} │ {'Accuracy':<12} │ "
        f"{'Macro F1':<10} │ {'Tokens':<8} │ {'Cost/case':<11} │"
    )
    lines.append("├" + "─" * 26 + "┼" + "─" * 14 + "┼" + "─" * 12 + "┼"
                 + "─" * 10 + "┼" + "─" * 13 + "┤")
    for label, _, _, _ in CONFIGS:
        m = matrix["configs"][label]
        n = m["n_cases"]
        acc_str = f"{m['accuracy_mean']*n:.1f}/{n} ±{m['accuracy_stdev']:.2f}"
        f1_str = f"{m['macro_f1_mean']:.3f}"
        tok_str = f"{m['avg_total_tokens_mean']:.0f}"
        cost_str = f"${m['cost_per_case_mean']:.5f}"
        lines.append(
            f"│ {label:<24} │ {acc_str:<12} │ {f1_str:<10} │ "
            f"{tok_str:<8} │ {cost_str:<11} │"
        )
    lines.append("└" + "─" * 26 + "┴" + "─" * 14 + "┴" + "─" * 12 + "┴"
                 + "─" * 10 + "┴" + "─" * 13 + "┘")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    started = time.time()
    print("=" * 72)
    print("Experiment v0.3 — Aetna within-domain held-out matrix")
    print("=" * 72)

    train_cases = load_split(MANIFEST_DIR, "train")
    test_cases = load_split(MANIFEST_DIR, "test")
    print(f"manifest: {MANIFEST_DIR}")
    print(f"train: {len(train_cases)} cases")
    print(f"test:  {len(test_cases)} held-out cases")

    # Phase 1: train
    train_summary = _train_store_b(train_cases)

    # Phase 2: eval 6 configs × 3 seeds
    matrix = _run_matrix(test_cases)

    # Phase 3: per-case correctness
    case_matrix = _case_level_matrix(matrix, test_cases)

    # Phase 4: save + print table
    elapsed = time.time() - started
    total_cost = train_summary["cost_total"] + matrix["total_cost_usd"]

    payload = {
        "experiment": "v0.3_aetna_within_domain_matrix",
        "manifest": str(MANIFEST_DIR),
        "store": str(STORE_B_PATH),
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "policy_max_chars": POLICY_MAX_CHARS,
        "train_seed": TRAIN_SEED,
        "eval_seeds": EVAL_SEEDS,
        "train": train_summary,
        "matrix": matrix,
        "case_level_matrix": case_matrix,
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print()
    print("=" * 72)
    print("Experiment v0.3 results")
    print("=" * 72)
    print(_format_table(matrix))
    print()
    print("Per-case correctness (majority vote across 3 seeds):")
    print(f"  {'case':<12s} {'gt':<8s} ", end="")
    for label, _, _, _ in CONFIGS:
        print(f"{label[:14]:<15s}", end="")
    print()
    for case_id in sorted(case_matrix.keys()):
        row = case_matrix[case_id]
        print(f"  {case_id:<12s} {row['ground_truth']:<8s} ", end="")
        for label, _, _, _ in CONFIGS:
            n_right = row[label]["n_right"]
            n_total = row[label]["n_total"]
            mark = "OK" if row[label]["majority_correct"] else "WR"
            print(f"{mark} {n_right}/{n_total}          ", end="")
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
