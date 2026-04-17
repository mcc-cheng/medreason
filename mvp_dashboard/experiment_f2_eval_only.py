"""F2 eval only — uses the existing dd_v02_abstracted store (230 rules)."""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from medreason.runners import ClaudeRunner, MemoryRunner, resolve_claude_model
from medreason.retrieval.embedder import FakeEmbedder
from medreason.store import RuleStore
from medreason_bench.splits import load_split

MODEL = resolve_claude_model('haiku')
SPLITS = PROJ / "medreason_bench" / "data" / "splits"
STORE = PROJ / "medreason_bench" / "data" / "stores" / "dd_v02_abstracted.db"

test_cases = load_split(SPLITS / "dd_v02_synthetic", "test")
print(f"Test cases: {len(test_cases)}")

base = ClaudeRunner(model=MODEL, include_policy=True, policy_max_chars=200)

t0 = time.time()
total_cost = 0.0
all_results = {}

for seed in [11, 17, 23]:
    conn = sqlite3.connect(str(STORE))
    rs = RuleStore(conn)
    mem = MemoryRunner(
        base_runner=base, store=rs, embedder=FakeEmbedder(),
        reranker_llm=None, tier3_top_k=5,
    )

    n_sr = 0
    n_srm = 0
    seed_cases = []

    for i, case in enumerate(test_cases):
        try:
            sr = base.run(case, seed=seed)
            srm = mem.run(case, seed=seed)
        except Exception as e:
            print(f"  ERROR on {case.case_id} seed {seed}: {e}")
            time.sleep(5)
            try:
                sr = base.run(case, seed=seed)
                srm = mem.run(case, seed=seed)
            except Exception as e2:
                print(f"  RETRY FAILED {case.case_id}: {e2}")
                continue

        total_cost += sr.cost_usd + srm.cost_usd
        n_sr += int(sr.correct)
        n_srm += int(srm.correct)

        seed_cases.append({
            "case_id": case.case_id,
            "ground_truth": case.ground_truth_outcome.value,
            "sr_correct": sr.correct,
            "srm_correct": srm.correct,
            "sr_det": sr.determination.value,
            "srm_det": srm.determination.value,
            "applied_rules": [
                {"rule_id": a.rule_id, "applied": a.applied}
                for a in srm.applied_rules
            ],
            "retrieved_rule_ids": list(srm.retrieved_rule_ids),
        })

        if (i + 1) % 30 == 0:
            print(f"  seed {seed}: {i+1}/{len(test_cases)} done "
                  f"(SR={n_sr} SRM={n_srm})")

    all_results[seed] = {
        "n": len(test_cases),
        "sr": n_sr,
        "srm": n_srm,
        "sr_acc": n_sr / len(test_cases),
        "srm_acc": n_srm / len(test_cases),
        "cases": seed_cases,
    }
    print(f"Seed {seed}: SR={n_sr}/{len(test_cases)} ({n_sr/len(test_cases):.1%})  "
          f"SRM={n_srm}/{len(test_cases)} ({n_srm/len(test_cases):.1%})")
    conn.close()

sr_mean = sum(s["sr_acc"] for s in all_results.values()) / 3
srm_mean = sum(s["srm_acc"] for s in all_results.values()) / 3
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"F2 RESULT (ABSTRACTED):  SR={sr_mean:.1%}  SRM={srm_mean:.1%}  "
      f"delta={srm_mean - sr_mean:+.1%}")
print(f"Baseline (no abstract): SR=93.5%  SRM=98.4%  delta=+4.9%")
print(f"Cost: ${total_cost:.3f}  Elapsed: {elapsed:.0f}s")

out = {
    "experiment": "F2_drugdisc_abstraction_eval",
    "elapsed_seconds": round(elapsed, 1),
    "total_cost_usd": round(total_cost, 5),
    "sr_mean": sr_mean,
    "srm_mean": srm_mean,
    "delta_pp": round((srm_mean - sr_mean) * 100, 2),
    "per_seed": {str(k): v for k, v in all_results.items()},
}
out_path = PROJ / "mvp_dashboard" / "experiment_f2_results.json"
out_path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
print(f"Saved to {out_path}")
