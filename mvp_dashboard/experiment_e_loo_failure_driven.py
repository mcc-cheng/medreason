"""Experiment E — LOO generalization test on the failure-driven path.

Question: Experiment D2 hit 30/30 on v0.2 train with
sparse-RAG + failure-driven memory. But D2 trained and evaluated on
the same 30 cases, so the rules that fixed adv_010, adv_014, adv_015
were extracted from those exact cases. Does the failure-driven path
generalize, or does it just memorize its own training errors?

Method — rule-masking LOO. Under skip_gate=True (our sparse multi-
policy config) rule extraction is *independent per case*:
  - Each rule is extracted from exactly one source case
    (supporting_case_ids has exactly 1 entry)
  - There's no cross-case interaction in the training pipeline
  - No generalization gate that depends on other cases

Under these conditions, "train on 29 cases" and "train on 30 cases
then drop the target's rules" produce IDENTICAL stores. So we can
simulate LOO by filtering the D2 store at eval time:

  For each target in {adv_010, adv_014, adv_015}:
    1. Eval sparse-RAG alone on target → sanity check (wrong)
    2. Eval sparse-RAG + full D2 memory on target → reproduce D2 30/30
    3. Eval sparse-RAG + LOO-filtered memory on target → THE TEST
       (filter = drop any rule whose supporting_case_ids contains target)

If step 3 also recovers the target, the failure-driven rules
generalized from OTHER cases. If step 3 matches step 1, the D2 fix
for that case was memorization of its own training error.

Cost estimate: ~$0.05 total (9 single-case evals).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medreason.ontology import ReasoningRule
from medreason.retrieval.embedder import FakeEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore
from medreason_bench.splits import load_split


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
OUT_PATH = Path(__file__).parent / "experiment_e_loo_failure_driven.json"

SOURCE_STORE = STORES_ROOT / "v0.2_fd_sparse.db"  # the D2 store
POLICY_MAX_CHARS = 200
SEED = 11
TARGETS = ["adv_010", "adv_014", "adv_015"]


def _build_base_runner():
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=True,
        policy_max_chars=POLICY_MAX_CHARS,
    )


def _build_filtered_store_conn(exclude_case_id: str | None) -> sqlite3.Connection:
    """Copy the D2 store into an in-memory DB, optionally skipping
    rules whose supporting_case_ids contains `exclude_case_id`.

    Returns a connection with a populated `rules` table that a
    RuleStore can wrap. If exclude_case_id is None, returns a faithful
    copy (no filtering — used for the control eval).
    """
    dst = sqlite3.connect(":memory:")
    # Initialize schema by creating a fresh RuleStore on dst
    RuleStore(dst)

    src = sqlite3.connect(str(SOURCE_STORE))
    src.row_factory = sqlite3.Row
    dst_store = RuleStore(dst)

    n_total = 0
    n_kept = 0
    n_dropped = 0
    for row in src.execute("SELECT rule_json FROM rules"):
        n_total += 1
        rule = ReasoningRule.model_validate_json(row["rule_json"])
        if exclude_case_id is not None and exclude_case_id in rule.evidence.supporting_case_ids:
            n_dropped += 1
            continue
        dst_store.put(rule)
        n_kept += 1
    src.close()
    print(f"    store copy: {n_total} total, {n_kept} kept, {n_dropped} dropped")
    return dst


def _eval_sparse_rag(target) -> dict:
    runner = _build_base_runner()
    r = runner.run(target, seed=SEED)
    ok = "OK" if r.correct else "WRONG"
    print(f"    sparse-RAG:     pred={r.determination.value:22s} {ok}  conf={r.confidence:.2f}")
    return {
        "mode": "sparse_rag_no_memory",
        "determination": r.determination.value,
        "correct": r.correct,
        "confidence": r.confidence,
        "cost_usd": round(r.cost_usd, 5),
    }


def _eval_sparse_rag_memory(target, conn, label: str) -> dict:
    base_runner = _build_base_runner()
    rule_store = RuleStore(conn)
    runner = MemoryRunner(
        base_runner=base_runner,
        store=rule_store,
        embedder=FakeEmbedder(),
        reranker_llm=None,
        tier3_top_k=3,
    )
    r = runner.run(target, seed=SEED)
    ok = "OK" if r.correct else "WRONG"
    n_applied = sum(1 for a in r.applied_rules if a.applied)
    print(f"    {label}: pred={r.determination.value:22s} {ok}  "
          f"conf={r.confidence:.2f}  applied={n_applied}/{len(r.applied_rules)}")
    return {
        "mode": f"sparse_rag_plus_memory_{label}",
        "determination": r.determination.value,
        "correct": r.correct,
        "confidence": r.confidence,
        "cost_usd": round(r.cost_usd, 5),
        "retrieved_rule_ids": list(r.retrieved_rule_ids),
        "applied_rules": [
            {"rule_id": a.rule_id, "applied": a.applied,
             "rationale": a.rationale}
            for a in r.applied_rules
        ],
    }


def main():
    started = time.time()
    print("=" * 70)
    print("Experiment E — LOO generalization on failure-driven path")
    print("=" * 70)
    print(f"source store: {SOURCE_STORE}")
    print(f"targets: {TARGETS}")
    print()

    cases_by_id = {c.case_id: c for c in load_split(SPLITS_ROOT / "v0.2", "train")}

    per_case_results = {}
    total_cost = 0.0

    for target_id in TARGETS:
        print(f"--- {target_id} (gt={cases_by_id[target_id].ground_truth_outcome.value}) ---")
        target = cases_by_id[target_id]

        # 1. Control 1 — plain sparse-RAG (should be wrong by design)
        print("  [1] sparse-RAG (no memory)")
        sparse_only = _eval_sparse_rag(target)
        total_cost += sparse_only["cost_usd"]

        # 2. Control 2 — full D2 store (should reproduce D2 30/30)
        print("  [2] sparse-RAG + full D2 memory (control)")
        full_conn = _build_filtered_store_conn(exclude_case_id=None)
        full_eval = _eval_sparse_rag_memory(target, full_conn, "full")
        total_cost += full_eval["cost_usd"]
        full_conn.close()

        # 3. LOO — store with target's own rules filtered out
        print(f"  [3] sparse-RAG + LOO memory (dropped rules from {target_id})")
        loo_conn = _build_filtered_store_conn(exclude_case_id=target_id)
        loo_eval = _eval_sparse_rag_memory(target, loo_conn, "loo")
        total_cost += loo_eval["cost_usd"]
        loo_conn.close()

        per_case_results[target_id] = {
            "ground_truth": cases_by_id[target_id].ground_truth_outcome.value,
            "sparse_rag_no_memory": sparse_only,
            "sparse_rag_plus_full_memory": full_eval,
            "sparse_rag_plus_loo_memory": loo_eval,
            "loo_recovered": loo_eval["correct"],
            "generalization_verdict": (
                "GENERALIZES" if loo_eval["correct"] and not sparse_only["correct"]
                else "MEMORIZED" if full_eval["correct"] and not loo_eval["correct"]
                else "SPARSE_ALREADY_RIGHT" if sparse_only["correct"]
                else "BOTH_WRONG"
            ),
        }
        print(f"    → verdict: {per_case_results[target_id]['generalization_verdict']}")
        print()

    # Summary
    n_generalized = sum(
        1 for r in per_case_results.values()
        if r["generalization_verdict"] == "GENERALIZES"
    )
    n_memorized = sum(
        1 for r in per_case_results.values()
        if r["generalization_verdict"] == "MEMORIZED"
    )

    elapsed = time.time() - started
    payload = {
        "experiment": "E_loo_failure_driven",
        "source_store": str(SOURCE_STORE),
        "method": "rule-masking LOO (equivalent to full retraining under skip_gate=True)",
        "targets": TARGETS,
        "per_case": per_case_results,
        "summary": {
            "n_targets": len(TARGETS),
            "n_generalized": n_generalized,
            "n_memorized": n_memorized,
            "verdict": (
                "CLEAN_SWEEP_GENERALIZES" if n_generalized == len(TARGETS)
                else f"PARTIAL: {n_generalized}/{len(TARGETS)} generalized, {n_memorized}/{len(TARGETS)} memorized"
            ),
        },
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("=" * 70)
    print("Experiment E summary")
    print("=" * 70)
    for target_id in TARGETS:
        r = per_case_results[target_id]
        print(f"  {target_id}: "
              f"sparse={r['sparse_rag_no_memory']['correct']}, "
              f"full={r['sparse_rag_plus_full_memory']['correct']}, "
              f"LOO={r['sparse_rag_plus_loo_memory']['correct']} "
              f"→ {r['generalization_verdict']}")
    print()
    print(f"  OVERALL: {payload['summary']['verdict']}")
    print(f"  cost:    ${total_cost:.4f}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  saved:   {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
