"""Drug discovery v0.1 — 6-config matrix on 25 lead optimization cases.

The second vertical for MedReason. Proves the metacognitive reasoning
memory mechanism transfers from healthcare prior auth to drug discovery
lead optimization triage.

Pipeline:
  1. Load 25 drug discovery cases
  2. Stratify into train/test split
  3. Seed 8 hand-crafted metacognitive rules into a fresh store
  4. Run training pipeline on train split (metacognitive proposer +
     failure analyzer) to extract additional rules from agent traces
  5. Eval 6 configs × 3 seeds on held-out test cases
  6. Output comparison table + per-case matrix

Configs:
  1. Zero-shot            (no guidelines, no memory)
  2. Zero-shot + memory   (no guidelines, Store A memory)
  3. Sparse-RAG           (200-char truncated guidelines, no memory)
  4. Sparse-RAG + memory  (200-char truncated guidelines, Store A)
  5. Full-RAG             (complete guidelines, no memory)
  6. Full-RAG + memory    (complete guidelines, Store A memory)

System prompt: system_drugdisc.txt (drug discovery triage agent).

Cost estimate: ~$12-18 (25 cases × ~$0.005/call × 6 configs × 3 seeds
+ training).
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
from medreason.ontology import ReasoningRule
from medreason.retrieval.embedder import OpenAIEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.data.drugdisc_cases import build_drugdisc_cases
from medreason_bench.data.drugdisc_rules import build_drugdisc_seed_rules
from medreason_bench.eval.metrics import macro_f1
from medreason_bench.splits import SplitRatios, stratify, write_manifest, verify_manifest
from medreason_bench.train import TrainingConfig, run_training


SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
MANIFEST_DIR = SPLITS_ROOT / "dd_v01"
OUT_PATH = Path(__file__).parent / "experiment_drugdisc_v01_matrix.json"

SYSTEM_PROMPT = "system_drugdisc.txt"
POLICY_MAX_CHARS = 200
TRAIN_SEED = 11
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


def _build_runner(*, include_policy: bool, policy_max_chars: int | None):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
        system_prompt_file=SYSTEM_PROMPT,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


# ── Phase 1: Build manifest ────────────────────────────────────────────────

def _build_manifest() -> tuple[list, list]:
    """Build cases, stratify, write manifest. Returns (train, test)."""
    print("[dd v0.1] building manifest")
    cases = build_drugdisc_cases()
    print(f"  loaded {len(cases)} cases")

    # 60/40 train/test, no dev
    ratios = SplitRatios(train=0.6, dev=0.0, test=0.4)
    splits = stratify(cases, ratios=ratios, seed=42)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(splits, MANIFEST_DIR)
    verify_manifest(MANIFEST_DIR)

    from collections import Counter
    for name in ("train", "test"):
        oc = Counter(c.ground_truth_outcome.value for c in splits[name])
        print(f"  {name}: {len(splits[name])} cases  {dict(oc)}")

    return splits["train"], splits["test"]


# ── Phase 2: Seed rules + train ────────────────────────────────────────────

def _train_store(train_cases: list) -> dict:
    print(f"\n[dd v0.1] training store on {len(train_cases)} cases")
    print(f"  system prompt: {SYSTEM_PROMPT}")
    print(f"  proposer: metacognitive_rule_proposer.txt")
    print(f"  store: {STORE_PATH}")

    if STORE_PATH.exists():
        STORE_PATH.unlink()
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(STORE_PATH))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    # Seed hand-crafted rules
    seed_rules = build_drugdisc_seed_rules()
    for r in seed_rules:
        rule_store.put(r)
    print(f"  seeded {len(seed_rules)} hand-crafted rules")

    # Run training pipeline to extract more rules from agent traces
    runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    critic_llm = _build_llm()
    proposer_llm = _build_llm()
    failure_llm = _build_llm()

    config = TrainingConfig(
        runner=runner,
        critic_llm=critic_llm,
        proposer_llm=proposer_llm,
        store=rule_store,
        trace_store=trace_store,
        policy=None,
        train_cases=train_cases,
        version="dd_v01",
        split="train",
        gate_k=5,
        gate_seed=TRAIN_SEED,
        skip_gate=True,
        embedder=OpenAIEmbedder(),
        progress_hook=lambda m: print(f"  {m}"),
        seed=TRAIN_SEED,
        include_failures=True,
        failure_analyzer_llm=failure_llm,
        use_metacognitive_proposer=True,
    )
    report = run_training(config)
    conn.commit()

    from medreason.ontology import RuleStatus
    total_rules = rule_store.count(RuleStatus.ACTIVE)
    conn.close()

    print(f"\n  training done in {report.elapsed_seconds:.1f}s")
    print(f"  agent correct / wrong: {report.n_agent_correct} / {report.n_agent_wrong}")
    print(f"  extracted rules (normal): {report.n_rules_promoted}")
    print(f"  extracted rules (failure): {report.n_failure_rules_promoted}")
    print(f"  total rules in store: {len(seed_rules) + report.n_rules_promoted + report.n_failure_rules_promoted}")
    print(f"  training cost: ${report.cost_total:.4f}")

    return {
        "n_train_cases": len(train_cases),
        "n_seed_rules": len(seed_rules),
        "n_agent_correct": report.n_agent_correct,
        "n_agent_wrong": report.n_agent_wrong,
        "n_rules_extracted_normal": report.n_rules_promoted,
        "n_rules_extracted_failure": report.n_failure_rules_promoted,
        "n_rules_total": len(seed_rules) + report.n_rules_promoted + report.n_failure_rules_promoted,
        "cost_total": round(report.cost_total, 5),
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }


# ── Phase 3: Eval matrix ──────────────────────────────────────────────────

def _eval_one(label, cases, *, include_policy, policy_max_chars, use_memory, seed):
    base = _build_runner(include_policy=include_policy, policy_max_chars=policy_max_chars)
    if use_memory:
        conn = sqlite3.connect(str(STORE_PATH))
        reranker = ClaudeLLMClient(model=resolve_claude_model("haiku"))
        runner = MemoryRunner(
            base_runner=base, store=RuleStore(conn),
            embedder=OpenAIEmbedder(), reranker_llm=reranker, tier3_top_k=5)
    else:
        conn = None
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


def _run_matrix(test_cases):
    print(f"\n[dd v0.1] running 6-config x 3-seed matrix on {len(test_cases)} held-out")
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


def _case_matrix(matrix, test_cases):
    out = {}
    for c in test_cases:
        out[c.case_id] = {"ground_truth": c.ground_truth_outcome.value}
        for label, _, _, _ in CONFIGS:
            runs = matrix["configs"][label]["per_seed"]
            corrects = [
                next((pc["correct"] for pc in r["per_case_results"]
                      if pc["case_id"] == c.case_id), False)
                for r in runs]
            nr = sum(1 for x in corrects if x)
            out[c.case_id][label] = {"n_right": nr, "n_total": len(corrects),
                                     "majority_correct": nr >= 2}
    return out


def main():
    started = time.time()
    print("=" * 72)
    print("Drug Discovery v0.1 — metacognitive matrix")
    print("=" * 72)

    train_cases, test_cases = _build_manifest()
    train_summary = _train_store(train_cases)
    matrix = _run_matrix(test_cases)
    cm = _case_matrix(matrix, test_cases)

    elapsed = time.time() - started
    total_cost = train_summary["cost_total"] + matrix["total_cost_usd"]

    payload = {
        "experiment": "drugdisc_v01_metacognitive_matrix",
        "manifest": str(MANIFEST_DIR),
        "store": str(STORE_PATH),
        "system_prompt": SYSTEM_PROMPT,
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "train": train_summary,
        "matrix": matrix,
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
        m = matrix["configs"][label]
        n = m["n_cases"]
        acc_n = m["accuracy_mean"] * n
        print(f"  {label:<24s} {acc_n:.1f}/{n} ({m['accuracy_mean']*100:.1f}%)  "
              f"{m['macro_f1_mean']:.3f}    ${m['cost_per_case_mean']:.5f}")
    print()
    print(f"  total cost: ${total_cost:.4f}")
    print(f"  elapsed:    {elapsed:.1f}s")
    print(f"  saved:      {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
