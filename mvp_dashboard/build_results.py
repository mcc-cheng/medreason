"""Build mvp_dashboard/results.json from the latest leaderboard artifacts.

Reads:
- medreason_bench/leaderboard/entries/*__v0.0__dev.json (one zero-shot, one memory)
- medreason_bench/leaderboard/entries/*__v0.0__dev__cases.json (per-case results)
- medreason_bench/data/training_reports/v0.0_haiku.json (training counts/cost)
- medreason_bench/data/stores/v0.0.db (top rules by use)

Writes:
- mvp_dashboard/results.json — single payload the static index.html consumes.

Run with:
    py mvp_dashboard/build_results.py [--version v0.0] [--model haiku] [--split dev]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "medreason_bench" / "leaderboard" / "entries"
TRAINING_REPORTS = ROOT / "medreason_bench" / "data" / "training_reports"
STORES = ROOT / "medreason_bench" / "data" / "stores"
OUT = Path(__file__).resolve().parent / "results.json"


def _load_entry_pair(version: str, split: str, model_alias: str) -> tuple[dict, dict]:
    """Find the zero-shot and memory leaderboard entries for the given run."""
    # Phase 4 saves entries with filenames like:
    #   <runner_id>__<version>__<split>.json
    # The runner_id encodes the model + ":memory" suffix when memory is on.
    # Memory entries replace ":" with "_" in the filename — see save_entry.
    candidates = list(LEADERBOARD.glob(f"*__{version}__{split}.json"))
    if not candidates:
        sys.exit(
            f"error: no leaderboard entries for version={version} split={split} in {LEADERBOARD}"
        )

    zs_entry = None
    mem_entry = None
    for path in candidates:
        data = json.loads(path.read_text())
        rid = data.get("runner_id", "")
        if "memory" in rid.lower():
            mem_entry = data
        else:
            zs_entry = data

    if zs_entry is None:
        sys.exit("error: no zero-shot leaderboard entry found")
    if mem_entry is None:
        sys.exit("error: no memory leaderboard entry found")
    return zs_entry, mem_entry


def _load_cases_pair(version: str, split: str) -> tuple[list, list]:
    """Load per-case results for both modes."""
    candidates = list(LEADERBOARD.glob(f"*__{version}__{split}__cases.json"))
    if not candidates:
        sys.exit(f"error: no per-case JSON files in {LEADERBOARD}")

    zs_cases = []
    mem_cases = []
    for path in candidates:
        data = json.loads(path.read_text())
        if data.get("mode") == "memory":
            mem_cases = data["cases"]
        else:
            zs_cases = data["cases"]

    if not zs_cases or not mem_cases:
        sys.exit("error: missing zero-shot or memory per-case JSON")
    return zs_cases, mem_cases


def _load_training_report(version: str, model: str) -> dict:
    path = TRAINING_REPORTS / f"{version}_{model}.json"
    if not path.exists():
        sys.exit(f"error: training report not found at {path}")
    return json.loads(path.read_text())


def _load_top_rules(version: str, limit: int = 10) -> list[dict]:
    """Top rules from the trained store, ranked by usage (seen_count)."""
    db_path = STORES / f"{version}.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    sys.path.insert(0, str(ROOT))
    from medreason.store import RuleStore
    from medreason.ontology import RuleStatus
    store = RuleStore(conn)
    rules = store.list_by_status(RuleStatus.ACTIVE)
    rules.sort(key=lambda r: (-r.seen_count, -r.posterior_mean))
    out = []
    for r in rules[:limit]:
        out.append({
            "rule_id": r.rule_id,
            "action": r.action,
            "posterior": r.posterior_mean,
            "trials": r.trials,
            "seen": r.seen_count,
        })
    conn.close()
    return out


def _per_case_payload(zs_cases: list, mem_cases: list) -> list[dict]:
    """Pair zero-shot and memory results by case_id."""
    zs_by_id = {c["case_id"]: c for c in zs_cases}
    mem_by_id = {c["case_id"]: c for c in mem_cases}
    all_ids = sorted(set(zs_by_id) | set(mem_by_id))
    out = []
    for cid in all_ids:
        zs = zs_by_id.get(cid, {})
        mem = mem_by_id.get(cid, {})
        gt = zs.get("ground_truth") or mem.get("ground_truth", "")
        applied_count = sum(
            1 for a in mem.get("applied_rules", []) if a.get("applied")
        )
        retrieved_count = len(mem.get("retrieved_rule_ids", []))
        out.append({
            "case_id": cid,
            "ground_truth": gt,
            "zero_shot": {
                "determination": zs.get("determination", "?"),
                "correct": bool(zs.get("correct", False)),
                "confidence": zs.get("confidence", 0.0),
                "tokens": zs.get("input_tokens", 0) + zs.get("output_tokens", 0),
                "cost_usd": zs.get("cost_usd", 0.0),
            },
            "memory": {
                "determination": mem.get("determination", "?"),
                "correct": bool(mem.get("correct", False)),
                "confidence": mem.get("confidence", 0.0),
                "tokens": mem.get("input_tokens", 0) + mem.get("output_tokens", 0),
                "cost_usd": mem.get("cost_usd", 0.0),
                "retrieved_count": retrieved_count,
                "applied_count": applied_count if retrieved_count > 0 else None,
            },
        })
    return out


def _entry_summary(entry: dict, n_correct: int) -> dict:
    return {
        "accuracy": entry["accuracy_mean"],
        "n_cases": entry["n_cases"],
        "correct": n_correct,
        "macro_f1": entry["macro_f1"],
        "brier": entry["brier"],
        "ece": entry["ece"],
        "avg_total_tokens": entry["avg_total_tokens"],
        "p50_latency_ms": entry["p50_latency_ms"],
        "p95_latency_ms": entry["p95_latency_ms"],
        "cost_per_case_usd": entry["cost_per_case_usd"],
        "total_cost_usd": entry["total_cost_usd"],
        "pattern_utilization": entry.get("pattern_utilization"),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v0.0")
    p.add_argument("--split", default="dev")
    p.add_argument("--model", default="haiku")
    args = p.parse_args(argv)

    zs_entry, mem_entry = _load_entry_pair(args.version, args.split, args.model)
    zs_cases, mem_cases = _load_cases_pair(args.version, args.split)
    training_report = _load_training_report(args.version, args.model)
    top_rules = _load_top_rules(args.version)

    # Compute correct counts from per-case
    zs_correct = sum(1 for c in zs_cases if c.get("correct"))
    mem_correct = sum(1 for c in mem_cases if c.get("correct"))

    payload = {
        "version": args.version,
        "split": args.split,
        "model": args.model,
        "seeds": zs_entry.get("seed_set", []),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "zero_shot": _entry_summary(zs_entry, zs_correct),
        "memory": _entry_summary(mem_entry, mem_correct),
        "training": {
            "n_cases_seen": training_report["n_cases_seen"],
            "n_agent_correct": training_report["n_agent_correct"],
            "n_critic_agreed": training_report["n_critic_agreed"],
            "n_traces_stored": training_report["n_traces_stored"],
            "n_rules_proposed": training_report["n_rules_proposed"],
            "n_rules_rejected": training_report["n_rules_rejected"],
            "n_rules_promoted": training_report["n_rules_promoted"],
            "n_rules_deprecated": training_report["n_rules_deprecated"],
            "n_rules_deferred": training_report["n_rules_deferred"],
            "cost_total": training_report["cost_total"],
            "elapsed_seconds": training_report["elapsed_seconds"],
        },
        "cases": _per_case_payload(zs_cases, mem_cases),
        "top_rules": top_rules,
    }

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
