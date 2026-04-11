"""Experiment A: priming ablation.

Tests whether the +25pp accuracy lift in v0.1 memory mode comes from:
  (a) the actual retrieved rules being applied, OR
  (b) the injection's framing ("apply each criterion explicitly, do not
      paraphrase a rule away") priming the agent to read the policy
      excerpt more carefully on the second pass.

Method: run ClaudeRunner zero-shot on the v0.1 train split with a
synthetic system_extra containing ONLY the priming framing (NO retrieved
rules). If accuracy lifts toward the memory mode's 10/12, the rules
themselves are doing nothing — it's all framing. If accuracy stays near
the plain zero-shot baseline of 7/12, memory really is contributing.

Output: priming_results.json with per-case results, suitable for
comparison against the existing zero-shot and memory leaderboard
entries.

Cost: ~12 calls × $0.0035 ≈ $0.04 with Haiku.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medreason.runners import ClaudeRunner, resolve_claude_model
from medreason_bench.splits import load_split


# The framing is the SAME framing the memory injector uses, minus the
# actual rule list. We strip the rule lines and the rule_id-based
# applied_rules contract since there are no rules to apply. What's left
# is purely the priming language: "apply each criterion explicitly, do
# not paraphrase a rule away".
PRIMING_PREAMBLE = (
    "=== POLICY APPLICATION PROTOCOL ===\n"
    "For this case, apply each criterion in the policy excerpt explicitly. "
    "Do not paraphrase a rule away. The clinical notes either satisfy a "
    "criterion or they do not — read the policy excerpt section by section "
    "and check each criterion against the documented findings before "
    "reaching your determination. Pay attention to footnotes, exception "
    "clauses, frequency limits, perioperative prerequisites, and appeal-"
    "precedent paragraphs that may flip the obvious surface answer.\n"
    "=== END PROTOCOL ==="
)


def main():
    splits_root = ROOT / "medreason_bench" / "data" / "splits"
    cases = load_split(splits_root / "v0.1", "train")
    print(f"[experiment-A] loaded {len(cases)} v0.1 train cases")

    runner = ClaudeRunner(model=resolve_claude_model("haiku"))
    print(f"[experiment-A] runner: {runner.runner_id}")
    print(f"[experiment-A] priming preamble: {len(PRIMING_PREAMBLE)} chars")

    results = []
    total_cost = 0.0
    started = time.time()

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.case_id} (gt={case.ground_truth_outcome.value})  ...", end="", flush=True)
        result = runner.run(case, seed=11, system_extra=PRIMING_PREAMBLE)
        ok = "OK" if result.correct else "WRONG"
        print(f" {result.determination.value:24} {ok}  ${result.cost_usd:.5f}")
        total_cost += result.cost_usd
        results.append({
            "case_id": result.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "determination": result.determination.value,
            "correct": result.correct,
            "confidence": result.confidence,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "reasoning_chain": result.reasoning_chain[:600],
        })

    elapsed = time.time() - started
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / len(results)

    payload = {
        "experiment": "A_priming_only",
        "runner_id": runner.runner_id,
        "n_cases": len(results),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "total_cost_usd": round(total_cost, 5),
        "elapsed_seconds": round(elapsed, 1),
        "priming_preamble": PRIMING_PREAMBLE,
        "cases": results,
    }
    out = Path(__file__).parent / "experiment_a_priming.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print()
    print(f"[experiment-A] accuracy: {n_correct}/{len(results)} = {accuracy:.3f}")
    print(f"[experiment-A] cost: ${total_cost:.4f}, elapsed: {elapsed:.1f}s")
    print(f"[experiment-A] saved to {out}")


if __name__ == "__main__":
    main()
