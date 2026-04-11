"""Experiment C — held-out generalization tests on the v0.2 sparse-RAG win.

Background: in the v0.2 train results (30 cases), sparse-RAG + memory beat
sparse-RAG by +1 case (28/30 vs 27/30) — the only cleanly attributable
memory win in the project. The fix was on adv_010 (thunderclap headache /
brain MRI). All 3 retrieved rules were marked applied=True. BUT the rules
that fixed adv_010 were extracted from adv_010 itself during training.
That's train-eval overlap on the same case. The mechanism is real but the
generalization claim ("rules from CASE_A help CASE_B") is unproven.

Two experiments here, both held-out:

  Phase A2 — LOO on adv_010 (the headline test)
    Train memory on the 29 v0.2 train cases EXCLUDING adv_010.
    Eval sparse-RAG and sparse-RAG + memory on the single adv_010 case.
    Question: does memory still recover adv_010 when adv_010 was never
    seen during training? If yes, the v0.2 fix did generalize. If no,
    the v0.2 fix was train-eval overlap as feared.

  Phase A1 — Stratified random 20/10 split (gives a held-out number)
    Stratify v0.2 train (22 denied / 5 approved / 3 overturned) into
    20 train + 10 held-out, preserving outcome ratios. Train memory on
    the 20. Eval sparse-RAG and sparse-RAG + memory on the held-out 10.
    Question: across a randomly-chosen held-out batch, does memory still
    add accuracy above sparse-RAG?

Both experiments run sparse-RAG mode (--include-policy --policy-max-chars
200) since that's the regime where memory showed value in v0.2.

Cost estimate: ~$0.42 total
  A2 train (29 cases multi-policy skip-gate): ~$0.19
  A2 sparse-RAG eval (1 case):                 ~$0.003
  A2 sparse-RAG+memory eval (1 case):          ~$0.006
  A1 train (20 cases):                         ~$0.13
  A1 sparse-RAG eval (10 cases):               ~$0.032
  A1 sparse-RAG+memory eval (10 cases):        ~$0.060
"""

from __future__ import annotations

import json
import random
import sqlite3
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
from medreason_bench.splits import load_split
from medreason_bench.train import TrainingConfig, run_training


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
OUT_PATH = Path(__file__).parent / "experiment_c_heldout.json"

POLICY_MAX_CHARS = 200
SEED = 11  # eval seed (matches the v0.2 result)
TRAIN_SEED = 42


def _build_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


def _train_store(
    *,
    train_cases: list,
    store_path: Path,
    label: str,
) -> dict:
    """Train a fresh memory store on the given case list. Returns a
    summary dict (cost, n_rules_promoted, etc.)."""
    print(f"\n[{label}] training memory store on {len(train_cases)} cases")
    print(f"[{label}] store path: {store_path}")

    if store_path.exists():
        store_path.unlink()
    store_path.parent.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(include_policy=False, policy_max_chars=None)
    critic_llm = _build_llm()
    proposer_llm = _build_llm()

    conn = sqlite3.connect(str(store_path))
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
        version=f"v0.2_{label}",
        split="train",
        gate_k=5,
        gate_seed=TRAIN_SEED,
        skip_gate=True,  # required for sparse multi-policy fixture
        embedder=FakeEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=TRAIN_SEED,
    )
    report = run_training(config)
    conn.commit()
    conn.close()

    print(
        f"[{label}] training done: {report.n_rules_promoted} rules promoted, "
        f"cost ${report.cost_total:.4f}, {report.elapsed_seconds:.1f}s"
    )
    return {
        "label": label,
        "n_train_cases": len(train_cases),
        "n_agent_correct": report.n_agent_correct,
        "n_critic_agreed": report.n_critic_agreed,
        "n_rules_promoted": report.n_rules_promoted,
        "cost_train": round(report.cost_total, 5),
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }


def _eval_sparse_rag(eval_cases: list, label: str) -> dict:
    """Run sparse-RAG (no memory) on the given cases. Returns summary."""
    print(f"\n[{label}] eval sparse-RAG (no memory) on {len(eval_cases)} cases")
    runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)

    results = []
    total_cost = 0.0
    for i, case in enumerate(eval_cases, 1):
        result = runner.run(case, seed=SEED)
        ok = "OK" if result.correct else "WRONG"
        print(
            f"  [{i}/{len(eval_cases)}] {case.case_id:12s} "
            f"gt={case.ground_truth_outcome.value:22s} "
            f"pred={result.determination.value:22s} {ok}  "
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
        })
    n_correct = sum(1 for r in results if r["correct"])
    print(f"[{label}] sparse-RAG accuracy: {n_correct}/{len(results)}  "
          f"cost ${total_cost:.4f}")
    return {
        "mode": "sparse_rag_no_memory",
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else 0.0,
        "cost_usd": round(total_cost, 5),
        "results": results,
    }


def _eval_sparse_rag_memory(eval_cases: list, store_path: Path, label: str) -> dict:
    """Run sparse-RAG + memory on the given cases. Returns summary."""
    print(f"\n[{label}] eval sparse-RAG + memory on {len(eval_cases)} cases")
    print(f"[{label}] memory store: {store_path}")

    base_runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    conn = sqlite3.connect(str(store_path))
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
            f"  [{i}/{len(eval_cases)}] {case.case_id:12s} "
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
    print(f"[{label}] sparse-RAG + memory accuracy: {n_correct}/{len(results)}  "
          f"cost ${total_cost:.4f}")
    return {
        "mode": "sparse_rag_plus_memory",
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else 0.0,
        "cost_usd": round(total_cost, 5),
        "results": results,
    }


def _stratified_split(cases: list, *, n_holdout: int, seed: int) -> tuple[list, list]:
    """Stratified random split by ground_truth_outcome.

    Returns (train_subset, holdout_subset) where holdout has roughly
    `n_holdout` total cases, distributed proportionally across strata.
    """
    by_outcome: dict[str, list] = defaultdict(list)
    for c in cases:
        by_outcome[c.ground_truth_outcome.value].append(c)

    total = len(cases)
    holdout: list = []
    train: list = []
    rng = random.Random(seed)

    # Allocate per-stratum holdout count proportionally, then absorb
    # rounding into the largest stratum.
    alloc = {}
    for k, group in by_outcome.items():
        alloc[k] = round(len(group) * n_holdout / total)
    diff = n_holdout - sum(alloc.values())
    if diff != 0:
        # Adjust the largest stratum to make exact n_holdout
        biggest = max(by_outcome.keys(), key=lambda k: len(by_outcome[k]))
        alloc[biggest] += diff

    for k in sorted(by_outcome.keys()):
        group = by_outcome[k][:]
        rng.shuffle(group)
        n_h = max(0, min(len(group), alloc[k]))
        holdout.extend(group[:n_h])
        train.extend(group[n_h:])

    holdout.sort(key=lambda c: c.case_id)
    train.sort(key=lambda c: c.case_id)
    return train, holdout


def main():
    started = time.time()
    print("=" * 70)
    print("Experiment C — held-out generalization tests on v0.2 sparse-RAG")
    print("=" * 70)

    cases = load_split(SPLITS_ROOT / "v0.2", "train")
    print(f"loaded {len(cases)} v0.2 train cases")

    payload: dict = {
        "experiment": "C_heldout_generalization",
        "version": "v0.2",
        "policy_max_chars": POLICY_MAX_CHARS,
        "eval_seed": SEED,
        "train_seed": TRAIN_SEED,
    }

    # ─────────────────────────────────────────────────────────────────
    # Phase A2 — Leave-one-out on adv_010
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase A2 — LOO on adv_010 (the headline test)")
    print("=" * 70)

    target_case = next((c for c in cases if c.case_id == "adv_010"), None)
    if target_case is None:
        print("ERROR: adv_010 not found in v0.2 train", file=sys.stderr)
        return 2
    a2_train = [c for c in cases if c.case_id != "adv_010"]
    a2_eval = [target_case]
    print(f"A2 train: {len(a2_train)} cases (excluded adv_010)")
    print(f"A2 eval:  {len(a2_eval)} case  ({a2_eval[0].case_id}, "
          f"gt={a2_eval[0].ground_truth_outcome.value})")

    a2_store = STORES_ROOT / "v0.2_loo010.db"
    a2_train_summary = _train_store(
        train_cases=a2_train,
        store_path=a2_store,
        label="A2-loo010",
    )
    a2_sparse = _eval_sparse_rag(a2_eval, label="A2")
    a2_memory = _eval_sparse_rag_memory(a2_eval, a2_store, label="A2")

    payload["A2_loo_adv_010"] = {
        "train": a2_train_summary,
        "sparse_rag": a2_sparse,
        "sparse_rag_plus_memory": a2_memory,
        "interpretation": (
            "If sparse-RAG+memory still gets adv_010 right after LOO, the "
            "v0.2 fix generalized from rules learned on OTHER cases. If "
            "sparse-RAG+memory now matches plain sparse-RAG (both wrong or "
            "both right), the v0.2 fix was train-eval overlap on the same "
            "case."
        ),
    }

    # ─────────────────────────────────────────────────────────────────
    # Phase A1 — Stratified random 20/10 split
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase A1 — Stratified random 20/10 split (seed 42)")
    print("=" * 70)

    a1_train, a1_eval = _stratified_split(cases, n_holdout=10, seed=TRAIN_SEED)
    print(f"A1 train: {len(a1_train)} cases")
    for c in a1_train:
        print(f"   {c.case_id:12s} {c.ground_truth_outcome.value}")
    print(f"A1 held-out: {len(a1_eval)} cases")
    for c in a1_eval:
        print(f"   {c.case_id:12s} {c.ground_truth_outcome.value}")

    a1_store = STORES_ROOT / "v0.2_strat20.db"
    a1_train_summary = _train_store(
        train_cases=a1_train,
        store_path=a1_store,
        label="A1-strat20",
    )
    a1_sparse = _eval_sparse_rag(a1_eval, label="A1")
    a1_memory = _eval_sparse_rag_memory(a1_eval, a1_store, label="A1")

    payload["A1_stratified_20_10"] = {
        "train_case_ids": [c.case_id for c in a1_train],
        "holdout_case_ids": [c.case_id for c in a1_eval],
        "train": a1_train_summary,
        "sparse_rag": a1_sparse,
        "sparse_rag_plus_memory": a1_memory,
        "interpretation": (
            "Compare sparse_rag accuracy vs sparse_rag_plus_memory accuracy "
            "on the held-out 10. A delta > 0 means memory generalizes; "
            "delta == 0 means memory adds no value above RAG on truly "
            "unseen cases; delta < 0 means memory hurts."
        ),
    }

    # ─────────────────────────────────────────────────────────────────
    # Save + summary
    # ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - started
    total_cost = (
        a2_train_summary["cost_train"] + a2_sparse["cost_usd"] + a2_memory["cost_usd"]
        + a1_train_summary["cost_train"] + a1_sparse["cost_usd"] + a1_memory["cost_usd"]
    )
    payload["elapsed_seconds"] = round(elapsed, 1)
    payload["total_cost_usd"] = round(total_cost, 5)

    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print()
    print("=" * 70)
    print("Experiment C summary")
    print("=" * 70)
    print(
        f"A2 (LOO adv_010):  sparse-RAG={a2_sparse['n_correct']}/{a2_sparse['n_cases']}  "
        f"sparse-RAG+memory={a2_memory['n_correct']}/{a2_memory['n_cases']}"
    )
    print(
        f"A1 (strat 20/10):  sparse-RAG={a1_sparse['n_correct']}/{a1_sparse['n_cases']}  "
        f"sparse-RAG+memory={a1_memory['n_correct']}/{a1_memory['n_cases']}"
    )
    print(f"total cost: ${total_cost:.4f}")
    print(f"elapsed:    {elapsed:.1f}s")
    print(f"saved to:   {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
