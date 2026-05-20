# targetval Phase 2 session log (2026-05-20)

Production buildout of the medreason target-validation product, shipped by a 9-agent ruflo swarm on 2026-05-20. This document is the operator's guide + coordination lessons learned. The architectural plan it implements lives at [`targetval_phase2_plan.md`](targetval_phase2_plan.md); the 6 per-builder dispatch briefs live in [`dispatch/`](dispatch/); the curated retro biology spot-check lives at [`mapk_retro_review.md`](mapk_retro_review.md).

## Branch state

- Branch: **`targetval-phase2`** at commit `529ef14`, pushed to origin.
- Origin: **`https://github.com/mcc-cheng/medreason.git`** (repo was transferred from `ultimatem7/medreason` mid-session).
- `master` remains dashboard-only Next.js; the 232-file Python research tree lives ONLY on this branch.

## How to run the code

```bash
# Get on the branch
git fetch origin targetval-phase2
git checkout targetval-phase2

# Python env (project requires 3.12+; we run 3.13.x)
python --version
python -m pip install -r requirements.txt
# Runtime deps: anthropic, openai, duckdb, pydantic, click, tabulate, psycopg2-binary
# Dev: pytest

# Canonical targetval suite (118 tests, ~0.4s)
python -m pytest tests/test_targetval_*.py -v

# Full repo suite (567/568; the one failure is a pre-existing drugdisc off-by-one, see below)
python -m pytest tests/

# Targetval CLI
python -m medreason_bench.targetval --help
python -m medreason_bench.targetval data build-mapk-retro
python -m medreason_bench.targetval swarm dryrun
python -m medreason_bench.targetval card write
```

## What shipped

| Builder | Files | Purpose |
|---|---|---|
| `topology-builder` | `medreason/targetval/topology.py` (140→349 LoC) | Deterministic `PathwayCache.to_json/load` + `cache_version`; modality-aware `derive_bypass_signals`; `dominant_bypass_mechanism` helper; `feedback_reactivation` signal for MAPK rebound |
| `ingest-builder` | `medreason/targetval/evidence_ingest.py` (234 LoC); new `medreason_bench/targetval/customer_csv.py` | `EvidencePipeline` + `build_default_pipeline` factory; caching/retry/error-type plumbing on fake fetchers; `IngestReport` for customer-CSV boundary validation. Real APIs (Reactome, Open Targets, ChEMBL) deferred to Phase 3. |
| `swarm-llm-builder` | `medreason/targetval/swarm.py` (273 LoC); new `swarm_prompts.py` (251) + `swarm_parsing.py` (391) | Real LLM-driven memo flow; 4-strategy tolerant JSON parser; `safe_parse_memo` fallback that never raises; `TargetMemo` v0.2 adds `bypass_signals_seen` + `parse_warnings` |
| `layer-store-builder` | `medreason/targetval/layers.py` (250→482 LoC); new `medreason/targetval/layer_stores.py` (153) | `LayerRouter.ingest_rule` / `retrieve_for_case` / `promote_rule` wired to canonical sqlite `RuleStore` with leak guard. CAMPAIGN→UNIVERSAL forbidden; must go via DISEASE. `promote_rule` strips customer provenance and re-validates. |
| `proposer-builder` | `medreason/targetval/cross_agent_analyzer.py` (445 LoC); new `cross_agent_prompts.py` (199) | **The moat.** `propose_corrective_rules` is real, never raises, every emitted CANDIDATE rule passes `LayerPolicy(layer=Layer.UNIVERSAL).validate_rule(rule, customer_tag=None)` defense-in-depth before return. |
| `mapk-curator` | `medreason_bench/targetval/mapk_retro*.py` (4 modules, 1,024 LoC); new `tests/test_targetval_end_to_end.py` (369) | 22 curated MAPK retro targets (BRAF/MEK/ERK/KRAS/NRAS/EGFR/MET/ALK/SHP2/FGFR2/CDK4-6/PI3K/WEE1/AXL/MDM2/RAF1) covering 9 approved / 10 phase-2-failed / 1 safety / 1 active-unknown. Real NCT IDs, prose `per literature: <study>` citations (no fabricated DOIs). Universal-safe surface (empty `internal` sub-bundle). |

Tests: 43 (pre-Phase 2 baseline) → **118 canonical targetval tests passing** (+75 new). Phase 2 LoC: 3,873 across 18 files.

## The moat (architectural invariant)

The 3-layer memory architecture is documented in `targetval_phase2_plan.md`; the invariant Phase 2 enforces is that **no rule carrying a `customer_tag` may enter Universal or Disease layers, ever.** This is the cross-customer moat — the differentiator that lets the platform learn across pharma tenants without violating data-sharing constraints.

Enforcement:
- `LayerPolicy.validate_rule(rule, customer_tag)` runs at `LayerRouter.ingest_rule` boundary.
- `LayerRouter.promote_rule` strips customer provenance from `supporting_case_ids` / `proposer_run_id` and re-validates; raises `LayerPolicyViolation` if the tag survives stripping.
- `propose_corrective_rules` (the cross-agent rule proposer) runs each LLM-emitted rule through `LayerPolicy(layer=Layer.UNIVERSAL).validate_rule(rule, customer_tag=None)` before returning — defense in depth.
- 23 layer tests + 16 cross-agent tests exercise these boundaries.

## How the swarm coordinated (lessons for future swarm work)

This was the first session that ran a multi-agent ruflo swarm to completion. The originally-documented "SendMessage-first agent-to-agent comms" pattern did NOT work as advertised. Here's what actually worked:

1. **Subagent tool context is isolated.** Tools the lead loads via `ToolSearch` (e.g., `SendMessage`, `TaskUpdate`) are not visible inside spawned subagents unless they call `ToolSearch` themselves. Architect tried to `SendMessage` builders directly and failed silently with "SendMessage primitive isn't a registered tool in this environment" — produced dispatch files instead.

2. **Subagents terminate after producing output, even with `run_in_background: true`.** They don't idle waiting for messages; they end their turn after the initial "I'm ready" response. Don't assume an agent is still alive after it has reported once.

3. **Resume-by-agent-ID works.** `SendMessage` with the spawn-time agent ID resurrects a terminated subagent with full transcript context. Resume-by-name fails after termination. Track agent IDs from spawn notifications.

4. **Therefore: lead-orchestrated dispatch is the resilient topology.** The lead tracks agent IDs, dispatches builders sequentially or in safe parallel pairs per the architect's dependency order, and relays each builder's output to the next as inline context. Architect-to-builder direct messaging is fragile.

5. **Each builder dispatch must contain the full upstream contract inline.** Subagents can't ask the lead clarifying questions mid-execution. Bake every interface signature into the dispatch message — line counts, field names, enum values, signature corrections found by earlier builders.

6. **Dependency order that worked for Phase 2:**
   1. architect (solo — designs plan + writes 6 dispatch briefs)
   2. topology-builder
   3. ingest-builder (depends on topology PathwayCache shape)
   4. swarm-llm-builder + layer-store-builder (parallel, touch different files)
   5. proposer-builder (needs TargetMemo from swarm-llm + LayerRouter API from layer-store)
   6. mapk-curator (needs proposer wired for the end-to-end test)
   7. tester (comprehensive verification)
   8. reviewer (invariant + quality pass)

7. **macOS Desktop + iCloud + parallel agents = file conflict dupes.** Running parallel subagents that write the same files caused iCloud to create ~296 `<name> 2.<ext>` copies across the repo. Either disable iCloud Desktop sync or move the repo to a non-synced path. Phase 2 deleted all dupes pre-commit (confirmed byte-identical or older mid-session snapshots).

## Pre-existing failure NOT caused by Phase 2

`tests/test_drugdisc_cases.py::test_seed_rules_are_valid` asserts `len(rules) == 8` but `build_drugdisc_seed_rules()` returns 9. One-line off-by-one in either the seed or the assertion. Touches `medreason_bench/data/drugdisc_*.py`, not Phase 2 territory. Separate fix.

## Phase 3 follow-ups (from the reviewer's verdict)

1. Expand MAPK retro bypass-mechanism coverage. Current fixture covers 5/9 `BypassMechanism` enum values; missing `MICROENVIRONMENT_RESCUE`, `PHARMACOKINETIC_ESCAPE`, `OTHER_VALIDATED` (all are in `bypass_precision_recall.positive_set` so the metric isn't fully stressed).
2. Split `tests/test_targetval_layers.py` (541 LoC) along its three section banners (policy / round-trip / promotion) into ~3×180 LoC files.
3. Clean up the macOS-Finder-duplicate filenames repo-wide (already done in this PR, but recurs if the repo lives in iCloud).
4. Real external APIs (Reactome, Open Targets, ChEMBL) — deferred per the original Phase 2 scope decision.
5. Fix the drugdisc off-by-one assertion.

## File index (all paths relative to repo root)

**Modified:** `medreason/targetval/{topology,evidence_ingest,swarm,layers,cross_agent_analyzer}.py`; `medreason_bench/targetval/{mapk_retro,recursion_ingest,__main__}.py`.

**New source:** `medreason/targetval/{swarm_prompts,swarm_parsing,layer_stores,cross_agent_prompts}.py`; `medreason_bench/targetval/{customer_csv,mapk_retro_data,mapk_retro_entries,mapk_retro_entries_extra}.py`.

**New tests:** `tests/test_targetval_{evidence_ingest,end_to_end}.py`.

**Docs:** this file; [`targetval_phase2_plan.md`](targetval_phase2_plan.md) (architect's 796-line plan); [`dispatch/*.md`](dispatch/) (6 builder briefs); [`mapk_retro_review.md`](mapk_retro_review.md) (biology spot-check).
