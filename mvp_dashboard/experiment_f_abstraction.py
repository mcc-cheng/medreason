"""Experiment F: Rule Abstraction for Cross-Case Transfer

Tests whether abstracting case-specific rules into concept-level rules
enables generalization that raw extraction cannot.

Two experiments:
  F1: Prior-auth LOO (the Exp E cases that were MEMORIZED)
      Train on 29 cases with --include-failures --abstract-rules,
      then test whether adv_010/adv_014/adv_015 recover under LOO.

  F2: Drug-discovery dd_v02 with abstraction
      Train on 52 cases with --abstract-rules, eval on 148 held-out.
      Then decompose: did rules transfer across archetypes?

Cost estimate: ~$0.50 total (F1 ~$0.15, F2 ~$0.35)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
SPLITS = PROJ / "medreason_bench" / "data" / "splits"
STORES = PROJ / "medreason_bench" / "data" / "stores"
OUT = PROJ / "mvp_dashboard"

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(PROJ))

from medreason.llm import ClaudeLLMClient
from medreason.ontology import BenchmarkCase
from medreason.retrieval.embedder import FakeEmbedder
from medreason.runners import ClaudeRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.eval.harness import EvalConfig, run_eval
from medreason_bench.splits import load_split
from medreason_bench.train import TrainingConfig, run_training


MODEL = resolve_claude_model("haiku")


def _fresh_store(name: str) -> tuple[sqlite3.Connection, RuleStore, TraceStore]:
    STORES.mkdir(parents=True, exist_ok=True)
    db = STORES / f"{name}.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(str(db))
    return conn, RuleStore(conn), TraceStore(conn)


def _train(
    cases: list[BenchmarkCase],
    store: RuleStore,
    trace_store: TraceStore,
    *,
    include_failures: bool = True,
    abstract_rules: bool = False,
    include_policy: bool = True,
    policy_max_chars: int | None = 200,
    seed: int = 11,
    version: str = "exp_f",
):
    runner = ClaudeRunner(
        model=MODEL,
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )
    llm = ClaudeLLMClient(model=MODEL)

    config = TrainingConfig(
        runner=runner,
        critic_llm=llm,
        proposer_llm=llm,
        store=store,
        trace_store=trace_store,
        policy=None,
        train_cases=cases,
        version=version,
        split="train",
        skip_gate=True,
        embedder=FakeEmbedder(),
        progress_hook=lambda msg: print(msg.encode('ascii', 'replace').decode()),
        seed=seed,
        include_failures=include_failures,
        abstract_rules=abstract_rules,
    )
    return run_training(config)


def _eval_cases(
    cases: list[BenchmarkCase],
    store_path: Path,
    *,
    include_policy: bool = True,
    policy_max_chars: int | None = 200,
    seed: int = 11,
):
    """Eval specific cases against a store, return per-case results."""
    from medreason.runners import MemoryRunner

    base = ClaudeRunner(
        model=MODEL,
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )
    conn = sqlite3.connect(str(store_path))
    rs = RuleStore(conn)

    mem_runner = MemoryRunner(
        base_runner=base,
        store=rs,
        embedder=FakeEmbedder(),
        reranker_llm=None,
        tier3_top_k=3,
    )

    results = []
    for case in cases:
        # With memory
        mem_result = mem_runner.run(case, seed=seed)
        # Without memory (baseline)
        base_result = base.run(case, seed=seed)
        results.append({
            "case_id": case.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "sparse_rag": {
                "determination": base_result.determination.value,
                "correct": base_result.correct,
                "confidence": base_result.confidence,
                "cost_usd": base_result.cost_usd,
            },
            "sparse_rag_memory": {
                "determination": mem_result.determination.value,
                "correct": mem_result.correct,
                "confidence": mem_result.confidence,
                "cost_usd": mem_result.cost_usd,
                "retrieved_rule_ids": list(mem_result.retrieved_rule_ids),
                "applied_rules": [
                    {"rule_id": a.rule_id, "applied": a.applied, "rationale": a.rationale}
                    for a in mem_result.applied_rules
                ],
            },
        })
    conn.close()
    return results


def run_f1_prior_auth_loo():
    """F1: LOO on adv_010, adv_014, adv_015 with abstracted rules."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F1: Prior-Auth LOO with Rule Abstraction")
    print("=" * 70)

    all_train = load_split(SPLITS / "v0.2", "train")
    targets = ["adv_010", "adv_014", "adv_015"]

    loo_results = {}
    total_cost = 0.0

    for target_id in targets:
        print(f"\n--- LOO: holding out {target_id} ---")
        train_cases = [c for c in all_train if c.case_id != target_id]
        target_cases = [c for c in all_train if c.case_id == target_id]

        store_name = f"exp_f1_loo_{target_id}"
        conn, rs, ts = _fresh_store(store_name)

        report = _train(
            train_cases, rs, ts,
            include_failures=True,
            abstract_rules=True,
            seed=11,
            version=store_name,
        )
        conn.commit()
        total_cost += report.cost_total

        print(f"  Training: {report.n_rules_promoted} normal + "
              f"{report.n_failure_rules_promoted} failure rules promoted")
        print(f"  Cost: ${report.cost_total:.4f}")

        # Dump the abstracted failure rules for inspection
        for row in conn.execute(
            "SELECT rule_id, rule_json FROM rules WHERE rule_id NOT LIKE 'dd_seed%'"
        ):
            rj = json.loads(row[1])
            ev = rj["evidence"]
            if "failure" in ev.get("proposer_run_id", ""):
                print(f"  Failure rule {rj['rule_id']}:")
                print(f"    action: {rj['action']}")
                print(f"    predicate: {rj['trigger']['semantic_predicate']}")
                print(f"    source: {ev['supporting_case_ids']}")

        # Eval the held-out case
        store_path = STORES / f"{store_name}.db"
        eval_results = _eval_cases(
            target_cases, store_path, seed=11,
        )
        conn.close()

        for r in eval_results:
            sr = r["sparse_rag"]
            srm = r["sparse_rag_memory"]
            total_cost += sr["cost_usd"] + srm["cost_usd"]
            recovered = not sr["correct"] and srm["correct"]
            print(f"  {r['case_id']}: SR={sr['correct']}  SRM={srm['correct']}  "
                  f"recovered={recovered}")
            if srm.get("applied_rules"):
                for ar in srm["applied_rules"]:
                    if ar["applied"]:
                        print(f"    Applied: {ar['rule_id']}")

            loo_results[target_id] = {
                "sparse_rag": sr,
                "sparse_rag_memory": srm,
                "recovered": recovered,
                "generalization_verdict": "GENERALIZED" if recovered else "MEMORIZED",
            }

    n_gen = sum(1 for v in loo_results.values() if v["recovered"])
    print(f"\nF1 RESULT: {n_gen}/3 generalized (vs 0/3 in Exp E without abstraction)")
    print(f"F1 cost: ${total_cost:.3f}")
    return loo_results, total_cost


def run_f2_drugdisc_abstract():
    """F2: dd_v02 with rule abstraction, then transfer analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F2: Drug Discovery dd_v02 with Rule Abstraction")
    print("=" * 70)

    train_cases = load_split(SPLITS / "dd_v02_synthetic", "train")
    test_cases = load_split(SPLITS / "dd_v02_synthetic", "test")

    store_name = "dd_v02_abstracted"
    conn, rs, ts = _fresh_store(store_name)

    report = _train(
        train_cases, rs, ts,
        include_failures=True,
        abstract_rules=True,
        seed=11,
        version=store_name,
    )
    conn.commit()
    total_cost = report.cost_total

    print(f"Training: {report.n_rules_promoted} normal + "
          f"{report.n_failure_rules_promoted} failure rules promoted")
    print(f"Cost: ${report.cost_total:.4f}")

    # Inspect a sample of abstracted rules
    print("\n--- Sample abstracted rules ---")
    count = 0
    for row in conn.execute(
        "SELECT rule_id, rule_json FROM rules WHERE rule_id NOT LIKE 'dd_seed%' LIMIT 10"
    ):
        rj = json.loads(row[1])
        ev = rj["evidence"]
        is_failure = "failure" in ev.get("proposer_run_id", "")
        print(f"  {rj['rule_id']} ({'failure' if is_failure else 'normal'}):")
        print(f"    action: {rj['action'][:100]}")
        print(f"    predicate: {rj['trigger']['semantic_predicate'][:80]}")
        print(f"    source: {ev['supporting_case_ids']}")
        count += 1

    # Run eval on held-out test set (3 seeds)
    store_path = STORES / f"{store_name}.db"
    conn.close()

    from medreason.runners import MemoryRunner

    base = ClaudeRunner(
        model=MODEL,
        include_policy=True,
        policy_max_chars=200,
    )

    per_seed_results = {}
    for seed in [11, 17, 23]:
        conn2 = sqlite3.connect(str(store_path))
        rs2 = RuleStore(conn2)
        mem_runner = MemoryRunner(
            base_runner=base,
            store=rs2,
            embedder=FakeEmbedder(),
            reranker_llm=None,
            tier3_top_k=5,
        )

        seed_results = []
        n_correct_sr = 0
        n_correct_srm = 0

        for case in test_cases:
            sr_result = base.run(case, seed=seed)
            srm_result = mem_runner.run(case, seed=seed)
            total_cost += sr_result.cost_usd + srm_result.cost_usd

            n_correct_sr += int(sr_result.correct)
            n_correct_srm += int(srm_result.correct)

            seed_results.append({
                "case_id": case.case_id,
                "ground_truth": case.ground_truth_outcome.value,
                "seed": seed,
                "sparse_rag_correct": sr_result.correct,
                "sparse_rag_memory_correct": srm_result.correct,
                "applied_rules": [
                    {"rule_id": a.rule_id, "applied": a.applied}
                    for a in srm_result.applied_rules
                ],
                "retrieved_rule_ids": list(srm_result.retrieved_rule_ids),
            })

        per_seed_results[seed] = {
            "n_cases": len(test_cases),
            "sr_correct": n_correct_sr,
            "srm_correct": n_correct_srm,
            "sr_accuracy": n_correct_sr / len(test_cases),
            "srm_accuracy": n_correct_srm / len(test_cases),
            "per_case": seed_results,
        }
        print(f"  Seed {seed}: SR={n_correct_sr}/{len(test_cases)} "
              f"({n_correct_sr/len(test_cases):.1%})  "
              f"SRM={n_correct_srm}/{len(test_cases)} "
              f"({n_correct_srm/len(test_cases):.1%})")
        conn2.close()

    # Summary
    sr_mean = sum(s["sr_accuracy"] for s in per_seed_results.values()) / 3
    srm_mean = sum(s["srm_accuracy"] for s in per_seed_results.values()) / 3
    delta = srm_mean - sr_mean
    print(f"\nF2 RESULT: SR={sr_mean:.1%}  SRM={srm_mean:.1%}  "
          f"delta={delta:+.1%}")
    print(f"F2 cost: ${total_cost:.3f}")

    return per_seed_results, total_cost


def main():
    t0 = time.time()

    f1_results, f1_cost = run_f1_prior_auth_loo()
    f2_results, f2_cost = run_f2_drugdisc_abstract()

    elapsed = time.time() - t0
    total_cost = f1_cost + f2_cost

    output = {
        "experiment": "F_rule_abstraction",
        "elapsed_seconds": round(elapsed, 1),
        "total_cost_usd": round(total_cost, 5),
        "f1_prior_auth_loo": {
            "description": "LOO on adv_010/014/015 with abstracted failure-driven rules",
            "baseline_comparison": "Exp E: 0/3 generalized without abstraction",
            "per_case": f1_results,
            "n_generalized": sum(1 for v in f1_results.values() if v["recovered"]),
            "n_targets": len(f1_results),
        },
        "f2_drugdisc_abstract": {
            "description": "dd_v02 with rule abstraction, 148 held-out test",
            "baseline_comparison": "Without abstraction: SR=93.5%, SRM=98.4%",
            "per_seed": f2_results,
            "sr_mean": sum(s["sr_accuracy"] for s in f2_results.values()) / 3,
            "srm_mean": sum(s["srm_accuracy"] for s in f2_results.values()) / 3,
        },
    }

    out_path = OUT / "experiment_f_abstraction.json"
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"Total cost: ${total_cost:.3f}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
