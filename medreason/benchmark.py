"""Benchmark runner — orchestrates generation, evaluation, and reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from tabulate import tabulate

from .agent import run_with_memory, run_zero_shot
from .config import RESULTS_DIR
from .extractor import abstract_to_pattern, extract_trace, _extract_local
from .generator import generate_local, generate_with_api, load_cases, save_cases
from .injector import retrieve_and_inject
from .ontology import BenchmarkCase, BenchmarkResult
from .store import PatternStore


def run_benchmark(
    cases: list[BenchmarkCase],
    mode: str = "both",
    store: PatternStore | None = None,
) -> BenchmarkResult:
    """Run benchmark evaluation.

    mode: "zero_shot", "memory", or "both"
    """
    if store is None:
        store = PatternStore()

    result = BenchmarkResult(
        timestamp=datetime.now(timezone.utc),
        total_cases=len(cases),
    )

    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] Case {case.case_id} "
              f"(CPT {case.task_config.cpt_code}, {case.difficulty.value})",
              flush=True)

        # Zero-shot run
        if mode in ("zero_shot", "both"):
            try:
                zs = run_zero_shot(case)
                result.zero_shot_results.append(zs)
                status = "CORRECT" if zs.correct else "WRONG"
                print(f"    Zero-shot: {zs.determination.value} ({status}, "
                      f"conf={zs.confidence:.2f}, {zs.latency_ms:.0f}ms)")

                # Store patterns from correct zero-shot results too —
                # this builds the memory that helps future memory-augmented runs
                if zs.correct and mode == "both":
                    try:
                        trace = _extract_local(case, zs)
                        pattern = abstract_to_pattern(trace)
                        store.store_pattern(pattern)
                        result.patterns_created += 1
                    except Exception:
                        pass
            except Exception as e:
                print(f"    Zero-shot failed: {e}")

        # Memory-augmented run
        if mode in ("memory", "both"):
            try:
                injection, pattern_ids = retrieve_and_inject(
                    store, case.task_config, top_k=3
                )
                n_patterns = len(pattern_ids)
                mem = run_with_memory(case, injection, pattern_ids)
                result.memory_results.append(mem)
                status = "CORRECT" if mem.correct else "WRONG"
                print(f"    Memory ({n_patterns} patterns): {mem.determination.value} "
                      f"({status}, conf={mem.confidence:.2f}, {mem.latency_ms:.0f}ms)")

                # Feedback loop: correct results create patterns
                if mem.correct:
                    try:
                        trace = extract_trace(case, mem)
                        pattern = abstract_to_pattern(trace)
                        store.store_pattern(pattern)
                        result.patterns_created += 1
                    except Exception:
                        # Local extraction fallback
                        from .extractor import _extract_local
                        trace = _extract_local(case, mem)
                        pattern = abstract_to_pattern(trace)
                        store.store_pattern(pattern)
                        result.patterns_created += 1

                # Reinforce/weaken matched patterns
                for pid in pattern_ids:
                    store.reinforce(pid, success=mem.correct)
                    result.patterns_reinforced += 1

            except Exception as e:
                print(f"    Memory failed: {e}")

    return result


def compute_metrics(result: BenchmarkResult) -> dict:
    """Compute summary metrics from benchmark results."""

    def _metrics(results: list) -> dict:
        if not results:
            return {"accuracy": 0, "count": 0, "avg_confidence": 0,
                    "avg_tokens": 0, "avg_latency_ms": 0}
        correct = sum(1 for r in results if r.correct)
        return {
            "accuracy": correct / len(results),
            "count": len(results),
            "correct": correct,
            "avg_confidence": sum(r.confidence for r in results) / len(results),
            "avg_tokens": sum(r.input_tokens + r.output_tokens for r in results) / len(results),
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        }

    def _by_field(results: list, field_fn) -> dict:
        groups = defaultdict(list)
        for r in results:
            groups[field_fn(r)].append(r)
        return {k: _metrics(v) for k, v in sorted(groups.items())}

    # We need the cases to get difficulty/payer breakdowns
    zs = result.zero_shot_results
    mem = result.memory_results

    metrics = {
        "run_id": result.run_id,
        "timestamp": result.timestamp.isoformat(),
        "total_cases": result.total_cases,
        "zero_shot": _metrics(zs),
        "memory": _metrics(mem),
        "patterns_created": result.patterns_created,
        "patterns_reinforced": result.patterns_reinforced,
    }

    if zs and mem:
        metrics["delta_accuracy"] = metrics["memory"]["accuracy"] - metrics["zero_shot"]["accuracy"]
        metrics["delta_pp"] = round(metrics["delta_accuracy"] * 100, 1)

    return metrics


def print_report(metrics: dict, fmt: str = "table"):
    """Print benchmark report."""
    if fmt == "json":
        print(json.dumps(metrics, indent=2, default=str))
        return

    print("\n" + "=" * 60)
    print("  VERIDICUS BENCHMARK REPORT")
    print("=" * 60)

    rows = []
    for mode_key in ("zero_shot", "memory"):
        m = metrics.get(mode_key, {})
        if m.get("count", 0) == 0:
            continue
        rows.append([
            mode_key.replace("_", " ").title(),
            f"{m['accuracy']:.1%}",
            f"{m.get('correct', 0)}/{m['count']}",
            f"{m['avg_confidence']:.2f}",
            f"{m['avg_tokens']:.0f}",
            f"{m['avg_latency_ms']:.0f}ms",
        ])

    print(tabulate(
        rows,
        headers=["Mode", "Accuracy", "Correct", "Avg Conf", "Avg Tokens", "Avg Latency"],
        tablefmt="simple",
    ))

    if "delta_pp" in metrics:
        delta = metrics["delta_pp"]
        arrow = "+" if delta > 0 else ""
        print(f"\n  Memory advantage: {arrow}{delta} percentage points")

    print(f"  Patterns created: {metrics['patterns_created']}")
    print(f"  Patterns reinforced: {metrics['patterns_reinforced']}")
    print("=" * 60 + "\n")


def save_results(result: BenchmarkResult, metrics: dict):
    """Save results to JSON files for dashboard consumption."""
    # Full results
    full_path = RESULTS_DIR / f"run_{result.run_id}.json"
    full_data = {
        "run_id": result.run_id,
        "timestamp": result.timestamp.isoformat(),
        "total_cases": result.total_cases,
        "patterns_created": result.patterns_created,
        "patterns_reinforced": result.patterns_reinforced,
        "zero_shot_results": [r.model_dump(mode="json") for r in result.zero_shot_results],
        "memory_results": [r.model_dump(mode="json") for r in result.memory_results],
    }
    full_path.write_text(json.dumps(full_data, indent=2))

    # Dashboard summary
    dash_path = RESULTS_DIR / "latest_dashboard.json"
    dash_data = _build_dashboard_data(result, metrics)
    dash_path.write_text(json.dumps(dash_data, indent=2))

    print(f"  Results saved to {full_path}")
    print(f"  Dashboard data saved to {dash_path}")
    return full_path, dash_path


def _build_dashboard_data(result: BenchmarkResult, metrics: dict) -> dict:
    """Build dashboard-compatible JSON."""
    # Per-case comparison
    case_comparisons = []
    zs_map = {r.case_id: r for r in result.zero_shot_results}
    mem_map = {r.case_id: r for r in result.memory_results}

    all_ids = set(list(zs_map.keys()) + list(mem_map.keys()))
    for cid in sorted(all_ids):
        entry = {"case_id": cid}
        if cid in zs_map:
            zs = zs_map[cid]
            entry["zero_shot"] = {
                "determination": zs.determination.value,
                "correct": zs.correct,
                "confidence": zs.confidence,
                "reasoning": zs.reasoning_chain,
                "tokens": zs.input_tokens + zs.output_tokens,
            }
        if cid in mem_map:
            mem = mem_map[cid]
            entry["memory"] = {
                "determination": mem.determination.value,
                "correct": mem.correct,
                "confidence": mem.confidence,
                "reasoning": mem.reasoning_chain,
                "tokens": mem.input_tokens + mem.output_tokens,
                "injected_patterns": mem.injected_pattern_ids,
            }
        case_comparisons.append(entry)

    # Payer breakdown
    payer_stats = defaultdict(lambda: {"zs_correct": 0, "zs_total": 0, "mem_correct": 0, "mem_total": 0})
    for r in result.zero_shot_results:
        # We'd need the case to get payer, but we can get it from case_id patterns
        payer_stats["all"]["zs_total"] += 1
        if r.correct:
            payer_stats["all"]["zs_correct"] += 1
    for r in result.memory_results:
        payer_stats["all"]["mem_total"] += 1
        if r.correct:
            payer_stats["all"]["mem_correct"] += 1

    return {
        "run_id": result.run_id,
        "timestamp": result.timestamp.isoformat(),
        "metrics": metrics,
        "cases": case_comparisons,
        "compounding_curve": _compute_compounding_curve(result),
    }


def _compute_compounding_curve(result: BenchmarkResult) -> list[dict]:
    """Compute cumulative accuracy over cases to show compounding effect."""
    curve = []
    mem_correct = 0
    zs_correct = 0

    for i in range(len(result.memory_results)):
        if i < len(result.zero_shot_results):
            zs_correct += int(result.zero_shot_results[i].correct)
        mem_correct += int(result.memory_results[i].correct)
        n = i + 1
        curve.append({
            "case_number": n,
            "zero_shot_accuracy": zs_correct / n if n > 0 else 0,
            "memory_accuracy": mem_correct / n if n > 0 else 0,
        })

    return curve
