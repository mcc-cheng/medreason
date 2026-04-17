#!/usr/bin/env python3
"""Ingest external lead-optimization CSV and run the MedReason benchmark.

Usage:
    py -u mvp_dashboard/ingest_external_csv.py data.csv [--train-frac 0.25] [--seeds 3]

CSV format (minimum required columns):
    compound_id     — unique identifier (e.g., "REC-001")
    outcome         — "approved" or "denied" (or "synthesize"/"reject")
    notes           — free-text property profile / compound description

Optional columns:
    therapeutic_area — comma-separated tags (e.g., "ONCOLOGY,KINASE")
    policy          — policy text for RAG (same for all rows, or per-row)
    difficulty      — "easy", "medium", or "hard" (defaults to "medium")

Output:
    1. Stratified train/test split
    2. Memory training on train cases
    3. 2-config × N-seed eval matrix (sparse_rag vs sparse_rag_memory)
    4. Results JSON + summary table

Total turnaround: ~2-4 hours for 1000 cases.
"""
from __future__ import annotations

import argparse
import csv
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
from medreason.ontology import (
    BenchmarkCase,
    Difficulty,
    FacilityType,
    Outcome,
    Payer,
    PriorAuthTaskConfig,
    RuleStatus,
)
from medreason.retrieval.embedder import OpenAIEmbedder
from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.store import RuleStore, TraceStore
from medreason_bench.data.drugdisc_rules import build_drugdisc_seed_rules
from medreason_bench.eval.metrics import macro_f1
from medreason_bench.splits import SplitRatios, stratify, write_manifest
from medreason_bench.train import TrainingConfig, run_training


STORES_ROOT = ROOT / "medreason_bench" / "data" / "stores"
SPLITS_ROOT = ROOT / "medreason_bench" / "data" / "splits"
SYSTEM_PROMPT = "system_drugdisc.txt"
POLICY_MAX_CHARS = 200
TRAIN_SEED = 11


# ── CSV parsing ───────────────────────────────────────────────────────────

OUTCOME_MAP = {
    "approved": Outcome.APPROVED,
    "synthesize": Outcome.APPROVED,
    "synthesized": Outcome.APPROVED,
    "advance": Outcome.APPROVED,
    "yes": Outcome.APPROVED,
    "pass": Outcome.APPROVED,
    "denied": Outcome.DENIED,
    "reject": Outcome.DENIED,
    "rejected": Outcome.DENIED,
    "deprioritize": Outcome.DENIED,
    "no": Outcome.DENIED,
    "fail": Outcome.DENIED,
}

DIFFICULTY_MAP = {
    "easy": Difficulty.EASY,
    "medium": Difficulty.MEDIUM,
    "hard": Difficulty.HARD,
}


def _parse_csv(csv_path: Path) -> tuple[list[BenchmarkCase], str]:
    """Parse CSV into BenchmarkCase list. Returns (cases, policy_text)."""
    cases = []
    policy_text = ""

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Normalize column names (lowercase, strip whitespace)
        if reader.fieldnames:
            reader.fieldnames = [c.strip().lower().replace(" ", "_") for c in reader.fieldnames]

        required = {"compound_id", "outcome", "notes"}
        if reader.fieldnames and not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames)
            print(f"ERROR: CSV missing required columns: {missing}")
            print(f"  Found columns: {reader.fieldnames}")
            print(f"  Required: {sorted(required)}")
            sys.exit(1)

        for i, row in enumerate(reader):
            cid = row["compound_id"].strip()
            outcome_raw = row["outcome"].strip().lower()
            notes = row["notes"].strip()

            if not cid or not outcome_raw or not notes:
                print(f"  WARNING: skipping row {i+2} — empty required field")
                continue

            outcome = OUTCOME_MAP.get(outcome_raw)
            if outcome is None:
                print(f"  WARNING: skipping row {i+2} — unknown outcome '{outcome_raw}'")
                print(f"    Valid values: {sorted(OUTCOME_MAP.keys())}")
                continue

            # Optional fields
            areas = [t.strip() for t in row.get("therapeutic_area", "GENERAL").split(",")]
            difficulty = DIFFICULTY_MAP.get(
                row.get("difficulty", "medium").strip().lower(), Difficulty.MEDIUM
            )
            row_policy = row.get("policy", "").strip()
            if row_policy and not policy_text:
                policy_text = row_policy

            case_id = f"ext_{i+1:04d}"
            cases.append(BenchmarkCase(
                case_id=case_id,
                task_config=PriorAuthTaskConfig(
                    payer=Payer.AETNA,  # placeholder
                    cpt_code=cid,
                    icd10_codes=areas,
                    facility_type=FacilityType.OUTPATIENT,
                ),
                clinical_notes=notes,
                policy_excerpt=policy_text or "No policy provided.",
                ground_truth_outcome=outcome,
                ground_truth_reasoning=[],
                difficulty=difficulty,
            ))

    return cases, policy_text


# ── Runner builders ───────────────────────────────────────────────────────

def _build_runner(*, include_policy, policy_max_chars):
    return ClaudeRunner(
        model=resolve_claude_model("haiku"),
        include_policy=include_policy,
        policy_max_chars=policy_max_chars,
        system_prompt_file=SYSTEM_PROMPT,
    )


def _build_llm():
    return ClaudeLLMClient(model=resolve_claude_model("haiku"))


# ── Training ──────────────────────────────────────────────────────────────

def _train(train_cases: list[BenchmarkCase], store_path: Path, run_name: str) -> dict:
    print(f"\n[{run_name}] Training on {len(train_cases)} cases")

    if store_path.exists():
        store_path.unlink()
    store_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(store_path))
    rule_store = RuleStore(conn)
    trace_store = TraceStore(conn)

    # Seed metacognitive rules
    seed_rules = build_drugdisc_seed_rules()
    for r in seed_rules:
        rule_store.put(r)
    print(f"  Seeded {len(seed_rules)} metacognitive rules")

    runner = _build_runner(include_policy=True, policy_max_chars=POLICY_MAX_CHARS)
    config = TrainingConfig(
        runner=runner,
        critic_llm=_build_llm(),
        proposer_llm=_build_llm(),
        store=rule_store,
        trace_store=trace_store,
        policy=None,
        train_cases=train_cases,
        version=run_name,
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

    n_rules = rule_store.count(RuleStatus.ACTIVE)
    conn.close()

    print(f"  Done: {n_rules} rules, {report.n_agent_correct}/{len(train_cases)} correct, "
          f"${report.cost_total:.4f}, {report.elapsed_seconds:.0f}s")

    return {
        "n_train": len(train_cases),
        "n_rules": n_rules,
        "n_correct": report.n_agent_correct,
        "n_wrong": report.n_agent_wrong,
        "cost": round(report.cost_total, 5),
        "elapsed_s": round(report.elapsed_seconds, 1),
    }


# ── Eval ──────────────────────────────────────────────────────────────────

def _eval_one(label, cases, store_path, *, include_policy, policy_max_chars, use_memory, seed):
    base = _build_runner(include_policy=include_policy, policy_max_chars=policy_max_chars)
    conn = None
    if use_memory:
        conn = sqlite3.connect(str(store_path))
        reranker = _build_llm()
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
            "compound_id": c.task_config.cpt_code,
            "ground_truth": c.ground_truth_outcome.value,
            "determination": r.determination.value,
            "correct": r.correct,
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
        "cost_per_case": round(cost / len(results), 5) if results else 0,
        "per_case_results": [t[2] for t in results],
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest external CSV and run MedReason benchmark")
    parser.add_argument("csv_path", type=Path, help="Path to CSV file")
    parser.add_argument("--train-frac", type=float, default=0.25, help="Fraction for training (default 0.25)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of eval seeds (default 3)")
    parser.add_argument("--run-name", type=str, default=None, help="Run name (default: CSV filename)")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: {args.csv_path} not found")
        sys.exit(1)

    run_name = args.run_name or args.csv_path.stem
    eval_seeds = list(range(11, 11 + args.seeds * 6, 6))  # [11, 17, 23, ...]
    store_path = STORES_ROOT / f"{run_name}.db"
    manifest_dir = SPLITS_ROOT / run_name
    out_path = Path(__file__).parent / f"experiment_{run_name}.json"

    started = time.time()
    print("=" * 72)
    print(f"MedReason External Benchmark — {run_name}")
    print("=" * 72)

    # Phase 1: Parse CSV
    print(f"\n[1/4] Parsing {args.csv_path}")
    cases, policy = _parse_csv(args.csv_path)
    if not cases:
        print("ERROR: No valid cases parsed from CSV")
        sys.exit(1)

    oc = Counter(c.ground_truth_outcome.value for c in cases)
    print(f"  Loaded {len(cases)} cases: {dict(oc)}")
    if policy:
        print(f"  Policy: {len(policy)} chars (from CSV)")

    # Phase 2: Split
    print(f"\n[2/4] Stratifying {args.train_frac:.0%} train / {1-args.train_frac:.0%} test")
    ratios = SplitRatios(train=args.train_frac, dev=0.0, test=1.0 - args.train_frac)
    splits = stratify(cases, ratios=ratios, seed=42)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(splits, manifest_dir)

    train_cases = splits["train"]
    test_cases = splits["test"]
    for name, cs in [("train", train_cases), ("test", test_cases)]:
        oc = Counter(c.ground_truth_outcome.value for c in cs)
        print(f"  {name}: {len(cs)} cases {dict(oc)}")

    # Phase 3: Train
    print(f"\n[3/4] Training memory store")
    train_report = _train(train_cases, store_path, run_name)

    # Phase 4: Eval
    configs = [
        ("sparse_rag",        True, POLICY_MAX_CHARS, False),
        ("sparse_rag_memory", True, POLICY_MAX_CHARS, True),
    ]

    n_test = len(test_cases)
    print(f"\n[4/4] Eval: {len(configs)} configs × {len(eval_seeds)} seeds × {n_test} cases")

    matrix = {}
    total_cost = 0.0
    for label, inc_pol, pmc, use_mem in configs:
        print(f"\n  === {label} ===")
        per_seed = []
        for seed in eval_seeds:
            print(f"    -- seed {seed} --")
            run = _eval_one(label, test_cases, store_path,
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
            "cost_per_case_mean": statistics.mean(r["cost_per_case"] for r in per_seed),
            "n_cases": per_seed[0]["n_cases"],
            "per_seed": per_seed,
        }
        n = per_seed[0]["n_cases"]
        print(f"  >>> {label}: {statistics.mean(accs)*n:.1f}/{n}  "
              f"f1={matrix[label]['macro_f1_mean']:.3f}")

    # Summary
    elapsed = time.time() - started
    print()
    print("=" * 72)
    print(f"MedReason Results — {run_name}")
    print(f"  {len(cases)} total cases ({len(train_cases)} train / {len(test_cases)} test)")
    print(f"  {train_report['n_rules']} rules extracted")
    print("=" * 72)
    print(f"  {'config':<24s} {'accuracy':<20s} {'F1':<8s} {'$/case'}")
    print("  " + "-" * 60)
    for label, _, _, _ in configs:
        m = matrix[label]
        n = m["n_cases"]
        acc_n = m["accuracy_mean"] * n
        std = m["accuracy_stdev"] * 100
        print(f"  {label:<24s} {acc_n:5.1f}/{n} ({m['accuracy_mean']*100:5.1f}%) ±{std:4.1f}%  "
              f"{m['macro_f1_mean']:5.3f}  ${m['cost_per_case_mean']:.5f}")

    delta = matrix["sparse_rag_memory"]["accuracy_mean"] - matrix["sparse_rag"]["accuracy_mean"]
    print(f"\n  Memory delta: {delta*100:+.1f}pp")
    print(f"  Training cost: ${train_report['cost']:.4f}")
    print(f"  Total cost: ${total_cost + train_report['cost']:.4f}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    payload = {
        "experiment": f"external_benchmark_{run_name}",
        "csv_path": str(args.csv_path),
        "n_total": len(cases),
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "training": train_report,
        "matrix": matrix,
        "memory_delta_pp": round(delta * 100, 1),
        "total_cost_usd": round(total_cost + train_report["cost"], 5),
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
