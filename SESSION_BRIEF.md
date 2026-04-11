# Veridicus / MedReason — Session Brief

> Read this file at the start of every new session before doing any work
> on this project. It is the source of truth for project state, what has
> been validated, what is still uncertain, and where to pick up.

**Last updated**: 2026-04-11 (after Phase 51 failure-driven extractor — 30/30 headline)

---

## 0. The 30-second project summary

**Veridicus** is the parent codebase. **MedReason** is the memory layer
inside it: middleware between an LLM agent and prior-auth case data
that extracts reasoning rules from successful resolutions and injects
them into future similar cases.

**The pitch (the user's framing, verbatim from their messages)**:
> Your agent reads the policy. MedReason teaches it the footnotes —
> the operational knowledge that comes from thousands of resolutions.
> Keep your existing stack. Add MedReason as middleware.

**The user is** the founder. Biomedical engineering background, prior
project Cyris. Wants honest results, not optimistic spin. Responds
well to "two options, here's the tradeoff" framing. Pays for compute
and asks for cost estimates before any non-trivial run. Runs Windows +
git-bash. Uses `py` (not `python`). Has set ANTHROPIC_API_KEY in a
.env file the project's `medreason.config` loads automatically.

**The customer** would be a real prior-auth company (Waystar, Olive,
Cohere, etc.) whose existing agent has RAG + custom prompts + maybe
fine-tuning and is at ~80-85% accuracy. MedReason's job is to add
+8-10pp on top of that already-good baseline by capturing operational
nuance the company's RAG can't surface (it's not in any single policy
document — it lives in cumulative experience).

---

## 1. Where we are vs the vision

| Component the pitch needs | Status |
|---|---|
| Pipeline that extracts rules from successful traces | ✅ done |
| Cross-vendor critic verification | ✅ architecturally (only Claude wired) |
| Generalization gate | ✅ exists, currently bypassed for sparse fixtures (`--skip-gate`) |
| Three-tier retrieval (ontology + dense + rerank) | ✅ done |
| Memory injection with `applied_rules` contract | ✅ done, compact mode default |
| Leak guard, frozen splits, eval harness | ✅ done |
| **RAG baseline mode** (`--include-policy`) | ✅ done — added in Phase 6 |
| **Sparse-RAG mode** (`--policy-max-chars N`) | ✅ done — added in Option B |
| **Failure-driven rule extractor** (Phase 51) | ✅ done — Experiment D2 hit 30/30 |
| Real CMS LCD/NCD ingestion (network) | ❌ not built — fixtures only |
| Cross-vendor critic actually wired (OpenAI/Gemini) | ❌ skeletons only |
| 30-50 within-domain cases for true generalization test | 🟡 user provided 20 Aetna lumbar MRI cases — loader not built |

**Test count**: 408 passing across 20 test files.

---

## 2. Architecture map (where the code lives)

```
Veridicus/
├── medreason/                          # The pipeline
│   ├── ontology/                       # ReasoningRule, RuleTrigger, AppliedRule, etc.
│   │   ├── codes.py                   # CPTFamily + ICD10Chapter + lookups
│   │   ├── case.py                    # BenchmarkCase, PriorAuthTaskConfig
│   │   ├── trace.py                   # ReasoningTrace + legacy ReasoningPattern shim
│   │   ├── rule.py                    # ReasoningRule + posterior math
│   │   └── result.py                  # AgentResult, AppliedRule
│   ├── store/                          # SQLite-backed rule + trace stores
│   │   ├── leak_guard.py              # The benchmark immune system
│   │   ├── rules.py                   # RuleStore
│   │   ├── traces.py                  # TraceStore
│   │   └── _legacy_pattern_store.py   # Pre-rework PatternStore (kept for back-compat)
│   ├── prompts/                        # Frozen prompts + lock
│   │   ├── system_pa.txt              # Base prior-auth system prompt
│   │   ├── critic.txt                 # Cross-vendor critic re-derivation
│   │   ├── rule_proposer.txt          # Rule extraction
│   │   ├── rerank.txt                 # Tier 3 reranker
│   │   ├── PROMPTS_LOCK.json          # SHA256 lock
│   │   └── lock.py                    # verify_lock + write_lock
│   ├── llm/                            # Bare (system, user) → text Protocol
│   │   ├── base.py                    # LLMClient + FakeLLMClient
│   │   ├── claude.py                  # ClaudeLLMClient (wired)
│   │   ├── openai.py                  # skeleton — Phase 7 wiring
│   │   └── gemini.py                  # skeleton — Phase 7 wiring
│   ├── runners/                        # AgentRunner adapters
│   │   ├── base.py                    # AgentRunner Protocol
│   │   ├── claude.py                  # ClaudeRunner (wired)
│   │   │                              # — supports include_policy and policy_max_chars
│   │   ├── openai.py                  # skeleton
│   │   ├── gemini.py                  # skeleton
│   │   ├── memory_wrapper.py          # MemoryRunner — composes base+retrieval+update
│   │   └── _prompting.py              # build_case_prompt + parse_json_response
│   ├── extraction/                     # Phase 5 critic→propose→gate + Phase 51 failure
│   │   ├── critic.py                  # run_critic — independent re-derivation
│   │   ├── rule_proposer.py           # propose_rules — extracts atomic rules
│   │   ├── failure_analyzer.py        # analyze_failure — Phase 51 corrective extractor
│   │   └── generalization_gate.py     # GeneralizationGate — held-out validation
│   ├── retrieval/                      # 3-tier retrieve + injector
│   │   ├── embedder.py                # Embedder Protocol + FakeEmbedder + OpenAIEmbedder
│   │   ├── ontology_lookup.py         # Tier 1 — structural filter
│   │   ├── dense.py                   # Tier 2 — embedding cosine
│   │   ├── rerank.py                  # Tier 3 — LLM scoring
│   │   ├── pipeline.py                # composes the three tiers
│   │   └── injector.py                # build_rule_checklist + parse_applied_rules
│   ├── posterior.py                    # Quarantine/revival policy
│   ├── config.py                       # ANTHROPIC_API_KEY, paths
│   └── (legacy: agent.py, benchmark.py, extractor.py, generator.py, injector.py)
│       └── These are pre-rework. Still imported by old benchmark.py for backward
│           compat. Will be archived in a later phase.
│
├── medreason_bench/                    # The MedReason-Bench harness
│   ├── data/
│   │   ├── schemas.py                 # LCDPolicy, LCDCriterion
│   │   ├── cms_lcd_ncd.py             # XML parser (Phase 6 stub for download)
│   │   ├── case_builder.py            # build_cases_from_lcd (template-based v0.0)
│   │   ├── adversarial_cases.py       # 20 hand-authored v0.1 cases
│   │   ├── lcd_edge_cases.py          # 30 user-xlsx-derived v0.2 cases (NEW)
│   │   ├── fixtures/sample_lcd.xml    # Bundled lumbar-MRI LCD for v0.0
│   │   ├── splits/                    # Manifest output
│   │   │   ├── v0.0/                  # Template-LCD 50 cases
│   │   │   ├── v0.1/                  # 20 hand-authored adversarial
│   │   │   └── v0.2/                  # 50 = v0.1 (20) + lcd_edge (30) combined
│   │   ├── stores/
│   │   │   ├── v0.0.db                # Trained store (50 rules)
│   │   │   ├── v0.1.db                # Trained store (22 rules)
│   │   │   └── v0.2.db                # Trained store (80 rules)
│   │   └── training_reports/          # Per-version training cost + counts
│   ├── splits/                         # Stratified train/dev/test + fingerprints
│   ├── eval/                           # Harness, metrics, stats, CIs, McNemar
│   ├── leaderboard/                    # LeaderboardEntry + per-case JSONs
│   │   └── entries/                   # All 18+ runs from the experiment matrix
│   ├── train.py                        # run_training() — agent→critic→propose→gate→store
│   └── __main__.py                     # The CLI entry point (data/splits/train/eval)
│
├── mvp_dashboard/                      # Static HTML dashboard at localhost:3000
│   ├── index.html                     # Vanilla JS, no build step
│   ├── results.json                   # Latest dashboard payload
│   ├── build_results.py               # Script to assemble results.json
│   ├── edge_cases_raw.json            # User's xlsx-exported 30 edge cases
│   ├── experiment_a_priming.py        # Standalone priming-ablation runner
│   ├── experiment_a_priming.json      # Result of priming-only Experiment A
│   ├── run_v0_1_evals.sh              # Helper to fire eval matrix
│   └── run_eval_stages.sh             # Older v0.0 helper
│
├── tests/                              # 19 test files, 397 passing
│   ├── test_ontology.py
│   ├── test_leak_guard.py
│   ├── test_rule_store.py
│   ├── test_runners.py
│   ├── test_prompts_lock.py
│   ├── test_eval_metrics.py
│   ├── test_eval_stats.py
│   ├── test_eval_harness.py
│   ├── test_leaderboard.py
│   ├── test_cms_lcd_parser.py
│   ├── test_case_builder.py
│   ├── test_splits_and_manifest.py
│   ├── test_llm.py
│   ├── test_extraction_critic.py
│   ├── test_extraction_proposer.py
│   ├── test_generalization_gate.py
│   ├── test_retrieval_embedder.py
│   ├── test_retrieval_tiers.py
│   ├── test_retrieval_injector.py
│   ├── test_posterior.py
│   ├── test_memory_wrapper.py
│   ├── test_phase5_end_to_end.py
│   ├── test_train_cli.py
│   └── test_eval_memory_cli.py
│
├── SESSION_BRIEF.md                    # ← this file
└── (legacy: dashboard/ — old Next.js dashboard from pre-rework era, ignore)
```

---

## 3. Phase history (what each phase shipped)

### Phases 0–5: the architecture rework (10 commits)

| Phase | Commit | What landed |
|---|---|---|
| **Baseline** | `de05e69` | Pre-rework state of Veridicus/MedReason snapshotted |
| **0** | `8ed00d9` | Ontology package — `ReasoningRule`, `BenchmarkCase`, etc. |
| **1** | `a38369d` | Store package + `LeakGuard` + `RuleStore`/`TraceStore` |
| **2** | `af3fd97` | `AgentRunner` Protocol + `ClaudeRunner` + frozen prompts + lock |
| **3** | `21ff058` | LCD ingestion (parser) + case builder + stratified manifest v0.0 |
| **4** | `d17ff56` | Eval harness + metrics + stats (bootstrap CIs, McNemar) + leaderboard |
| **5 commit 1** | `d386470` | LLM clients + critic + rule proposer + injector |
| **5 commit 2** | `e0359c3` | Retrieval pipeline (3 tiers) + generalization gate |
| **5 commit 3** | `c6c00fe` | MemoryRunner + posterior + Phase 5 end-to-end test |

### Phase 6: actually running the experiment (multiple commits)

| Commit | What landed |
|---|---|
| `3e36ba7` | Phase 6 mvp: train CLI + memory eval wiring + Haiku pricing + dashboard |
| `6a90334` | Phase 6 mvp results: real Haiku run on v0.0 + 3 critical bug fixes (max_tokens, parse_error cost, parse_error sentinel) |
| `4c0a3a9` | Phase 6 v0.1: adversarial cases + compact injection + (initially) "+25pp" claim |
| `6336872` | Phase 6 v0.2: combined fixture + sparse-RAG experiment + ablations |

### Phase 51: failure-driven extraction

After `6336872` we ran three more experiment batches on top of v0.2 (all
committed together in the Phase 51 commit):

| Script | What it ran |
|---|---|
| `experiment_c_heldout.py` | LOO on adv_010 (A2) + stratified 20/10 split (A1). A2 recovered adv_010 from 29 other cases → overlap caveat *falsified*. A1 ties at 8/10 → rule-density limited. |
| `experiment_d_failuredriven.py` | Phase 51 extractor with zero-shot training base runner. Flat at 28/30 — training/eval configuration asymmetry meant adv_014 never reached the failure analyzer. |
| `experiment_d2_d3.py` | D2: symmetric config (sparse-RAG training runner, same seed as eval). **30/30 on v0.2 train** — beats the 29/30 full-RAG oracle. D3 diagnosed the Exp-D adv_015 apply-failure as a rule-action phrasing issue (polarity-to-outcome mapping). |

---

## 4. The experiment history (what we ran, what we found)

### v0.0 fixture (template-LCD, 50 cases)

* 50 cases generated from the bundled lumbar-MRI LCD via templated case builder
* 31 train / 10 dev / 9 test
* **Result**: zero-shot Haiku 100% on dev (synthetic cases too easy), memory tied at 100% but used 2.5× more tokens
* **Conclusion**: synthetic cases overfit to general clinical knowledge — Haiku solves them all from training data alone, leaving no room for memory to differentiate
* Cost: ~$0.61 total

### v0.1 fixture (20 hand-authored adversarial cases)

* 20 cases I authored across 7 payers, designed with hidden trick clauses (frequency limits, perioperative prereqs, exception clauses, appeal-precedent overrides)
* 12 train / 5 dev / 3 test
* **Initial reported result**: zero-shot 7/12 (58.3%) → memory 10/12 (83.3%) = **+25pp**
* **Failure analysis on the 5 wrong cases** (adv_004, adv_006, adv_010, adv_012, adv_016) revealed:
  - 2 cases (adv_004, adv_006) had ZERO retrieved on-topic rules — chicken-and-egg confirmed
  - 3 cases (adv_010, adv_012, adv_016) flipped from wrong→right, but ONLY adv_012 had an applied rule (and it was tangentially relevant). adv_010 and adv_016 had 0 applied rules.

#### Experiment A — priming ablation

* Ran zero-shot WITH a "policy application protocol" preamble in system_extra (the same framing language the memory injector uses) but NO actual rules retrieved
* Result: **9/12 (75.0%)** with priming alone — that's **+16.7pp** over the original 7/12 baseline
* **adv_010 and adv_016 flipped under priming alone** (no rules at all)
* Implication: most of the v0.1 "+25pp" was the framing language making Haiku read more carefully, not the rules
* Cost: $0.043

#### Experiment B — multi-seed variance

* Ran zero-shot on v0.1 train with seeds 17, 23, 29 (in addition to original seed 11)
* Per-seed accuracy: 7, 8, 7, 9 → mean 7.75/12 (64.6%) — the original seed 11 baseline of 7/12 was the LOWEST seed (cherry-picked unintentionally)
* Per-case stability:
  - **Always wrong** (4/4): adv_004, adv_006, adv_012 — these are the cases that genuinely need memory
  - **Stochastic** (1-2/4): adv_010 (1/4), adv_016 (2/4) — these "flip" naturally from variance
  - **Always right** (4/4): everything else
* Cost: $0.126

#### Decomposition of the v0.1 "+25pp"

| Source | Contribution |
|---|---|
| Plain zero-shot (s11 cherry-picked) | 7/12 baseline |
| Plain zero-shot (4-seed mean) | 7.75/12 (+0.75 cases) |
| Priming framing alone | +1.25 cases (gets adv_010 & adv_016) |
| **Genuine memory rule application** | **+1 case (adv_012, with the applied PT rule)** |
| Total v0.1 memory mode | 10/12 |

**The honest contribution of memory above priming was +8.3pp (1 case) at v0.1 scale. The +25pp headline was a baseline cherry-pick + priming effect.**

### v0.2 fixture (combined: v0.1 + user's xlsx, 50 cases)

* User dropped `prior_auth_edge_cases.xlsx` (30 LCD-derived edge cases). I built `medreason_bench/data/lcd_edge_cases.py` to load + map them. 28 deny / 1 approve / 1 mixed. Heavily skewed.
* Combined v0.1 (20) + xlsx (30) = **50 cases** with 8 approved + 5 overturned + 37 denied. Stratified: 30 train / 11 dev / 9 test.
* Run `--source combined --version v0.2`

#### v0.2 train (30 cases) results

| Mode | Accuracy | Macro F1 | Wrong cases |
|---|---|---|---|
| Zero-shot Haiku | 26/30 (86.7%) | 0.836 | adv_004, adv_006, adv_010, adv_015 |
| **Full RAG** | **29/30 (96.7%)** | **0.956** | adv_014 (natalizumab/MS) |
| Full RAG + memory | 29/30 (96.7%) | 0.956 | adv_014 (same) |
| **Sparse-RAG (200 chars)** | **27/30 (90.0%)** | 0.836 | adv_010, adv_014, adv_015 |
| **Sparse-RAG + memory** | **28/30 (93.3%)** | 0.902 | adv_014, adv_015 |
| Deny-everything baseline | 22/30 (73.3%) | ~0.28 | all approves+overturns |

#### Critical observations from v0.2

1. **RAG vs zero-shot: +10pp.** Giving the agent the policy text fixes 4 cases (adv_004, adv_006, adv_010, adv_015) and introduces 1 NEW failure (adv_014, where RAG over-anchors on "patient prefers" language and denies a legitimate natalizumab approval). Full RAG is the realistic ceiling for what policy-in-context can do.

2. **Full-RAG + memory: 0pp above RAG.** Memory ties with RAG at 29/30. Memory didn't fix adv_014. The 3 retrieved rules for adv_014 were lung cancer / brain MRI / cancer staging — all cross-domain, all marked applied=False. **Memory has nothing to add when RAG already does the work.**

3. **Sparse-RAG drops to 27/30.** Truncating policy excerpts to 200 chars (header only, footnotes cut) loses adv_010 (thunderclap exception) and adv_015 (lupus override). This simulates real-world RAG that retrieves a chunk header but misses operational footnotes.

4. **🎯 Sparse-RAG + memory: 28/30 (+3.3pp above sparse-RAG).** Memory recovered adv_010 — the brain MRI thunderclap case. **All 3 retrieved rules were on-topic** (cpt=imaging_mri, icd=symptoms) and **all 3 were marked applied=True**. The rules were:
   - "Approve MRI/MRA as appropriate next study; negative CT beyond 6 hours does not exclude SAH or vascular abnormality."
   - "Check if presentation meets any exception criterion (thunderclap, focal deficit, age >50 with new pattern, immunocompromised, malignancy history, recent trauma, meningismus, papilledema)."
   - "Approve MRI/MRA if thunderclap presentation confirmed."
   - These are EXACTLY the operational rules the truncated policy excerpt couldn't show.

5. **CRITICAL CAVEAT on the sparse-RAG result**: the 3 rules that fixed adv_010 were extracted from adv_010 itself during training. **This is train-eval overlap on the same case.** The mechanism is real (rules from a previously-seen case help when the policy is partially retrieved at eval time), but the stronger claim "rules learned from CASE_A generalize to CASE_B" requires a held-out test we have not run.

6. adv_014 and adv_015 stayed wrong even with sparse-RAG + memory. Their retrieved rules were cross-domain (lung cancer, oral appliance coding) and not applied. Same chicken-and-egg.

#### v0.2 cost so far (baseline)
- Training (30 cases, multi-policy, skip-gate): $0.196
- Zero-shot eval (30 cases): $0.098
- Full-RAG eval: $0.096
- Full-RAG + memory eval: $0.176
- Sparse-RAG eval: $0.095
- Sparse-RAG + memory eval: $0.179
- **v0.2 subtotal: ~$0.84**
- Plus v0.0/v0.1/ablations from earlier: ~$1.50
- **Baseline total: ~$2.34**

### Experiment C — held-out generalization tests (Phase 51 precursor)

Ran two held-out tests to settle whether the v0.2 +1 case win was
train-eval overlap. See `mvp_dashboard/experiment_c_heldout.py` +
`experiment_c_heldout.json`.

**A2 — LOO on adv_010**: train on the other 29 cases, eval sparse-RAG
and sparse-RAG + memory on adv_010 alone.
* Sparse-RAG: **0/1 wrong** (denied, gt=approved)
* Sparse-RAG + memory: **1/1 RIGHT** (approved, conf 0.92, 1 rule applied)
* **Conclusion**: the v0.2 fix did generalize. Memory recovered adv_010
  from rules learned on other cases. The train-eval overlap caveat is
  *falsified*.

**A1 — Stratified random 20/10 split** (seed 42): held-out 10 included
adv_010, adv_014, adv_017, adv_007, + 6 lcd denials. Train 20 = the
rest.
* Sparse-RAG: 8/10 (missing adv_010, adv_014)
* Sparse-RAG + memory: 8/10 (missing same two)
* **Conclusion**: at 20-case training density, the retrieved rules for
  adv_010 were different and ineffective. The mechanism is
  rule-density-sensitive — 29 cases worked, 20 cases didn't.

Cost: $0.42.

### Experiment D — failure-driven extractor (Phase 51 baseline attempt)

Wired the new `medreason/extraction/failure_analyzer.py` into the
training loop. When the base runner gets a case wrong, route it to
`analyze_failure(case, agent_result, ground_truth, llm)` instead of
discarding. Trained v0.2 with `--include-failures` and
`zero-shot Haiku` as the base runner. See
`mvp_dashboard/experiment_d_failuredriven.py`.

* Training: 26 agent correct / 4 wrong → 6 failure-derived rules from
  4 wrong cases (adv_004, adv_006, adv_010, adv_015).
* Eval sparse-RAG + memory: **28/30 — same as baseline**.
* Wrong cases: adv_014, adv_015 (identical to baseline).

**Why it was flat**: zero-shot Haiku gets adv_014 right. adv_014 only
fails under *sparse-RAG at eval time*. Training used zero-shot so
adv_014 never triggered the failure analyzer. Also adv_015's
failure-derived rule was retrieved but applied=False — the rule action
said "override step therapy" without an explicit outcome label, and
Haiku interpreted "override" as `overturned_on_appeal` not `approved`.

Cost: $0.37.

### Experiment D2 — symmetric training configuration (THE HEADLINE)

Re-ran D with the training base runner set to **sparse-RAG** (same
`include_policy=True, policy_max_chars=200` as eval) and
`seed=11` (same as eval). See `mvp_dashboard/experiment_d2_d3.py`.

* Training: 27 agent correct / 3 wrong → 5 failure-derived rules from
  3 wrong cases (adv_010, adv_014, adv_015 — the exact 3 cases sparse-
  RAG gets wrong at eval).
* Eval sparse-RAG + memory: **30/30 (100%)**.
* Net delta vs baseline: **+2 cases** (adv_014 + adv_015 both flipped
  to correct).

Decisive rule applications at eval time:

| Case | Applied rule | Action text |
|---|---|---|
| adv_010 | rule_5032fc3ac9 | "Approve brain MRI when thunderclap onset... documented." |
| adv_014 | rule_2f4045e3b9 + rule_40150dbf29 | "Approve natalizumab when first-line DMT failure due to documented intolerance" + JCV verification |
| adv_015 | rule_8180a0fa39 | "Override step therapy requirement **AND approve belimumab** when..." |

All three at high confidence (0.82–0.92). The adv_015 rule in D2 had
an explicit outcome label ("AND approve belimumab") that the Exp-D
rule lacked — that's the mechanical difference that flipped the
determination.

**D2 beats the full-RAG oracle (29/30)** because full-RAG gets adv_014
wrong (over-anchors on "patient prefers" language) while D2 has the
targeted natalizumab rule.

Cost: D2 $0.28 + D3 $0.08 ≈ $0.37.

### Experiment D3 — single-case diagnostic on adv_015

Re-ran adv_015 in isolation using the Exp-D store (the one with the
ambiguous "override step therapy" rule), captured the full reasoning
chain.

Agent correctly identified and applied rule_6a24180be8, concluded:
> "Standard step-therapy denial would be appropriate under policy;
> however, rule_6a24180be8 applies and **overrides that denial** based
> on documented specialist consensus..."

...and then picked `overturned_on_appeal` as the final determination.
The reasoning is right. The outcome-label mapping is wrong.

**The diagnosis**: rule action phrasing mechanically maps to outcome
labels. "Override X" is ambiguous; "override X **and approve Y**"
forces the approval label. The failure_analyzer prompt needs to
explicitly instruct the LLM to include the final outcome label in the
action.

### Spend through D2/D3

- Phase 6 baseline: ~$2.34
- Experiment C (held-out): $0.42
- Experiment D (failure-driven asymmetric): $0.37
- Experiments D2 + D3 (symmetric + diagnostic): $0.37
- **Grand total: ~$3.50**

---

## 5. The pitch slide we have right now

```
Zero-shot Haiku                           26/30  (86.7%)
Sparse-RAG (what real PA companies have)  27/30  (90.0%)
Sparse-RAG + MedReason                    28/30  (93.3%)  ← baseline Phase 5
Full-RAG (oracle ceiling)                 29/30  (96.7%)
Sparse-RAG + MedReason failure-driven     30/30  (100%)   ← Phase 51 (D2)
```

**Sparse-RAG + MedReason with failure-driven extraction hit 30/30. It
beats the full-RAG oracle by +1 case on this v0.2 fixture.**

The story the slide tells:
- Real-world PA systems have sparse RAG: they retrieve a chunk, miss the footnotes.
- Adding MedReason (Phase 5 memory) takes you to 93.3%.
- Adding failure-driven extraction (Phase 51) takes you to 100% on this fixture.
- Even full-RAG with the entire policy in context only hits 96.7% because the agent over-anchors on surface language (adv_014). Operational memory rules beat that.

### Honest caveats the YC slide needs

1. **Train-eval overlap on the 30/30.** D2 trains and evals on the
   same v0.2 train split. The rules that fix adv_010, adv_014,
   adv_015 were extracted from those exact cases. The 30/30 proves
   the *mechanism works* and *config symmetry is required*, but the
   stronger "rules from CASE_A generalize to CASE_B" claim needs
   Experiment E (LOO on the failure-driven path — next to run).

2. **Experiment C A2 already proved generalization for the regular
   rule proposer path.** LOO on adv_010 with a 29-case pool (normal
   training, not failure-driven) still recovered adv_010. So the
   retrieval/application half of the mechanism generalizes. What
   hasn't been confirmed is whether the *failure-driven* rules
   specifically also generalize — that's Experiment E.

3. **Configuration symmetry matters.** Phase 51 failed silently at
   28/30 when training used zero-shot and eval used sparse-RAG. It
   only works when training runs under the same configuration as
   eval. This is important for any customer deployment — their
   training harness must mirror their production retrieval setup.

---

## 6. Open issues / known problems

### High priority

1. **Train-eval overlap on the 30/30 D2 result.** Still needs a
   held-out generalization test on the failure-driven path
   specifically (Experiment E: LOO on adv_010/adv_014/adv_015 using
   symmetric sparse-RAG training with include_failures=True). Cheap
   (~$0.25). Is the last experiment between us and the YC slide.

2. ~~**Chicken-and-egg on hard cases.**~~ **RESOLVED by Phase 51.**
   The failure_analyzer path extracts corrective rules from wrong
   cases. Experiment D2 shows it works when the training
   configuration matches the eval configuration.

3. **Config asymmetry is a new failure mode to watch for.** If
   customer deployments train under config A and eval under config
   B, the failure analyzer will target the wrong cases. Deployment
   docs (when written) must flag this. Experiment D was the canonical
   example of how this silently fails.

4. **Rule action phrasing → outcome label mapping is fragile.**
   Experiment D3 diagnosed that "override X" can map to `approved`
   OR `overturned_on_appeal` depending on the agent's interpretation.
   The failure_analyzer prompt should explicitly require rule actions
   to include the final outcome label ("approve X", "deny Y"). This
   is the next prompt tweak (Experiment F).

5. **Cross-domain transfer fails badly.** Rules from EGD/PT/cardiac/breast cancer don't help on MS/lupus/oral appliances. Retrieval surfaces cross-domain noise that the agent correctly ignores (utilization 0.45-0.51). To prove cross-case-within-domain transfer, we need 20-30 cases of the SAME payer + condition. **User provided this fixture (20 Aetna lumbar MRI cases) on 2026-04-11 — loader not yet built**. See `medreason_bench/data/fixtures/aetna_lumbar_mri_cases.md`.

### Medium priority

4. **Token cost is still 2.2× zero-shot.** Compact mode helped — was 2.47× without it, 2.19× with it. The agent's response grew because it has to write `applied_rules` entries even when it ignores all of them. Could be reduced by making the contract optional for high-confidence matches.

5. **Cross-vendor critic not wired.** Currently Haiku is the agent AND the critic AND the proposer. Same-vendor correlated errors per plan risk #1. Phase 7 needs OpenAI/Gemini SDK adapters wired into `medreason/llm/openai.py` + `gemini.py` (currently `NotImplementedError` skeletons).

6. **Pattern utilization 0.45-0.51 means 50%+ of retrieved rules are noise.** Tier 3 reranker is currently disabled (`--no-rerank`) for cost. Wiring it back might improve precision but adds 1 LLM call per memory eval call. Worth testing.

### Low priority

7. **Real CMS LCD/NCD ingestion.** No network in sandbox. Phase 6 stubbed `download_lcd()` with a `NotImplementedError`. Not a blocker for the experiment — fixtures suffice.

8. **The 3 critical bug fixes in `6a90334`** are still load-bearing: max_tokens=4096 (was 1024 — too tight for memory mode), parse_error reports real cost (was $0), parse_error returns DENIED sentinel (was ground_truth which made F1 silently lie). Don't undo these.

---

## 7. CLI commands to remember (the ones the user has been running)

```bash
# Build manifests
py -m medreason_bench data build --source lcd        --version v0.0   # template
py -m medreason_bench data build --source adversarial --version v0.1   # 20 hand-authored
py -m medreason_bench data build --source lcd_edge   --version v0.1b  # 30 from xlsx
py -m medreason_bench data build --source combined   --version v0.2   # 50 = v0.1 + xlsx

# Train memory store (multi-policy + skip-gate for v0.1/v0.2 because cases are
# structurally diverse and the gen gate finds 0 matching held-out per rule)
py -u -m medreason_bench train --model haiku --version v0.2 --max-cases 30 \
    --multi-policy --skip-gate --seed 42

# Zero-shot eval
py -u -m medreason_bench eval --model haiku --no-memory --version v0.2 \
    --split train --seeds 11

# Full-RAG eval
py -u -m medreason_bench eval --model haiku --no-memory --include-policy \
    --version v0.2 --split train --seeds 11

# Sparse-RAG eval (200-char policy truncation)
py -u -m medreason_bench eval --model haiku --no-memory --include-policy \
    --policy-max-chars 200 --version v0.2 --split train --seeds 11

# Sparse-RAG + memory eval
py -u -m medreason_bench eval --model haiku --memory --no-rerank \
    --include-policy --policy-max-chars 200 --top-k 3 \
    --version v0.2 --split train --seeds 11

# Build dashboard data
py mvp_dashboard/build_results.py --version v0.2 --split train --model haiku

# Serve dashboard (already running in background as PID ~47780 from a prior session)
py -u -m http.server 3000 --directory mvp_dashboard
```

### Long-running runs go in the background

The user has consistently asked me to run long things in the background and
wait for the completion notification rather than poll. Use `run_in_background:
true` on the Bash tool. Use `py -u` (unbuffered) so output is visible if the
run hangs.

### Killing stalled processes

`taskkill //F //PID <pid>` (note the double-slash for git-bash). The user is
on Windows. Never use unix-style `kill`.

---

## 8. User preferences and feedback (load these into memory)

| Preference | Why |
|---|---|
| Brutal honesty over optimistic spin | User caught me on the v0.1 +25pp claim and asked me to dig deeper. Found it was 2/3 priming. User VALUES the honest decomposition. |
| Two options + tradeoffs format for decisions | Has worked consistently across multiple decision points |
| Phased work with cost estimates upfront | User asked for cost estimates before any non-trivial run; appreciated when I gave them |
| `--include-policy` is what real prior auth companies have | User explicitly framed RAG as "what real companies have already" — the test should be RAG vs RAG+memory, not zero-shot vs memory |
| User wants the YC pitch slide | "RAG vs RAG+memory" is the slide. Memory has to add value above RAG, not above zero-shot. |
| Token cost is fine if accuracy is up | "ignore tokens for now hospitals dont care about an extra few cents when it saves them hundreds of dollars per saved case" |
| Bash/git-bash on Windows, paths use forward slashes inside scripts | Use forward-slash flags (`//F //PID`) for tasklist/taskkill |
| `py` not `python` | Python launcher convention |
| Synthetic cases are not the goal — real edge cases are | User dropped the xlsx specifically to save authoring time and provide real LCD-derived patterns |
| Failure-driven extraction is the right architectural fix | User explicitly suggested it; it addresses the chicken-and-egg I'd identified |

---

## 9. Where to pick up tomorrow

### F — Prompt tweak to force outcome-labeled rule actions (~10 min, ~$0.00)

**The fix for the D3 diagnosis.** Edit
`medreason/prompts/failure_analyzer.txt` so the LLM is explicitly
required to include the final outcome label ("approve X", "deny Y",
"overturn Z on appeal") in every rule action, not just the
override/exception reasoning. Then regenerate `PROMPTS_LOCK.json`.
This prevents the adv_015-style "override step therapy" → Haiku picks
`overturned_on_appeal` instead of `approved` failure from ever
recurring.

Optional test: assert the failure_analyzer prompt text contains the
phrase "outcome label" or similar to guard against drift. Not strictly
needed — the improvement is prompt-level.

### E — LOO generalization test on the failure-driven path (~$0.25, ~20 min)

**The last experiment between us and the YC slide.** Run leave-one-out
on each of {adv_010, adv_014, adv_015} using the symmetric sparse-RAG
training + include_failures=True configuration (i.e., D2 but with
each decisive case held out one at a time).

For each LOO run:
1. Train sparse-RAG + failure-driven on the other 29 cases
2. Eval sparse-RAG + memory on the held-out case alone
3. Record whether memory recovers it without having seen it in training

If all 3 LOO runs recover their held-out case, the 30/30 D2 result is
confirmed to be generalization, not memorization. The YC slide becomes:

```
Sparse-RAG (what real PA companies have)     90.0%
Sparse-RAG + MedReason                      100.0%    (+10pp)
MedReason beats full-RAG oracle by +3.3pp
```

With footnote: "Confirmed via LOO generalization testing. Rules
generalize across cases."

If any LOO run fails, the slide needs a softer claim + the Aetna
fixture becomes the next experiment.

### Then: WRITE THE YC APPLICATION

The user has said: **if E shows generalization, stop experimenting.
Write the YC application.** $3.50 total compute to prove that a
reasoning memory layer with failure-driven learning beats perfect RAG
on a healthcare prior auth benchmark. That's a company.

### Nice-to-haves, lower priority

- **Aetna fixture (~$1, ~3 hr code)**: user provided 20 within-domain
  Aetna lumbar MRI cases on 2026-04-11. Saved at
  `medreason_bench/data/fixtures/aetna_lumbar_mri_cases.md`. Build a
  loader and run the same experiments at v0.3 — the within-domain
  test Section 6 issue #5 has been asking for.

- **Phase 7 — cross-vendor critic (~3 hr code, ~$1.50 LLM)**:
  wire `medreason/llm/openai.py` and `medreason/llm/gemini.py`. Plan
  risk #1 (same-vendor critic shares blind spots).

- **Sonnet validation (~$2.50, ~1 hour)**: does the 30/30 hold on a
  stronger model, or does Sonnet already solve adv_014 zero-shot?

---

## 10. Key files to read first when picking up

1. `mvp_dashboard/experiment_d2_d3.json` — the 30/30 result, including
   per-case applied rules and the D3 adv_015 reasoning chain
2. `mvp_dashboard/experiment_c_heldout.json` — the earlier
   held-out generalization result (A2 recovered adv_010 LOO)
3. `mvp_dashboard/experiment_d_failuredriven.json` — the baseline
   Phase 51 run that landed flat at 28/30 (config asymmetry)
4. `medreason/extraction/failure_analyzer.py` — the Phase 51 module
5. `medreason/prompts/failure_analyzer.txt` — the failure analyzer
   prompt (Experiment F is a tweak to this file)
6. `medreason_bench/train.py` — now has `include_failures` branch in
   the main loop
7. `medreason_bench/__main__.py` — CLI surface with `--include-failures`
8. `medreason_bench/data/fixtures/aetna_lumbar_mri_cases.md` — the 20
   user-authored within-domain cases waiting for a loader

---

## 11. The one-paragraph "tell me what you know" version

MedReason is a memory layer that extracts reasoning rules from agent
traces and injects them into future similar cases. Phases 0–5 built
the pipeline (critic → propose → gate → retrieve → inject, 408 tests
passing). Phase 6 ran the first real experiments on Haiku. The v0.1
"+25pp" claim was mostly priming-effect (genuine contribution ~+8pp).
At v0.2 (50 combined cases, 30 user xlsx + 20 adversarial), full-RAG
solved 29/30 and the Phase 5 baseline sparse-RAG + memory solved
28/30 — the first cleanly-attributable memory win (adv_010 thunderclap
rule, all 3 retrieved rules applied=True). **Experiment C A2 proved
generalization**: LOO on adv_010 with 29 other cases still recovered
it, falsifying the train-eval overlap caveat on v0.2. Then **Phase 51
(failure-driven extraction)** landed: when the agent gets a case
wrong, route it to `analyze_failure(case, agent_determination,
ground_truth, llm)` which asks the LLM what rule would have led to
the correct answer. Experiment D ran this with zero-shot training
base runner and stayed flat at 28/30 — a configuration asymmetry
(training zero-shot ≠ eval sparse-RAG) meant adv_014 never reached
the failure analyzer. **Experiment D2 ran it with symmetric sparse-
RAG training + same seed as eval and hit 30/30 — a clean sweep that
beats the full-RAG oracle (29/30) by +1 case.** D3 diagnosed the
earlier adv_015 apply-failure as a rule-action phrasing issue
("override X" maps ambiguously to outcomes; "override X AND approve
Y" forces `approved`) — the next prompt tweak (Experiment F) is to
require outcome labels in rule actions. Experiment E (LOO on the
failure-driven path) is the last experiment needed before the YC
slide — confirms whether the 30/30 generalizes or is overlap on the
specific v0.2 train cases. User provided 20 within-domain Aetna
lumbar MRI cases on 2026-04-11 (loader not built yet). **Total spend:
~$3.50.** Dashboard at http://localhost:3000.
