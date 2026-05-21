# medreason

Two sibling codebases sharing a repo:

1. **Drug Discovery Canvas** — Next.js dashboard for agentic in-silico drug-discovery simulation against a live knowledge graph. Lives on `master`. The default landing experience.
2. **Veridicus + MedReason-Bench + targetval** — the Python research codebase: institutional reasoning memory for healthcare AI agents, the prior-auth + lead-op + target-validation benchmarks, and the Phase 2 target-validation product with the 3-layer memory moat. Lives on the **`targetval-phase2`** branch (143 .py files, 4 CLIs, 567-test suite).

Pick the side you're working on below.

---

## Drug Discovery Canvas (Next.js dashboard)

An agentic in-silico drug discovery platform. A Gemini-powered agent simulates compound–protein interactions and stores outcomes in a live knowledge graph (PostgreSQL via Prisma). Each simulation run updates Bayesian confidence scores on interaction edges.

### Running locally

**Prerequisites:** Docker, Node.js 18+, [Ollama](https://ollama.com)

```bash
# 1. Pull the model (one-time)
ollama pull llama3.1

# 2. Start the postgres database (from repo root)
docker compose up -d

# 3. Install dependencies
cd dashboard
npm install

# 4. Configure environment

cp .env.example .env.local
# No API keys needed — Ollama runs fully local

# 5. Set up the database
npm run db:generate   # generate Prisma client
npm run db:push       # push schema to DB
npm run db:seed       # load real protein/drug seed data

# 6. Start the app
npm run dev           # → http://localhost:3000
```

### Sample prompts

Click any example in the UI, or paste these directly:

**Imatinib × BCR-ABL (CML targeted therapy)**
> Simulate Imatinib binding to BCR-ABL at 0.1 µM. Predict efficacy and toxicity, explain what the Bayesian confidence score means, and summarize what this simulation adds to our knowledge.

**Gefitinib × EGFR (lung cancer)**
> Analyze Gefitinib selectivity at EGFR at 1 µM. Assess its safety profile and predict on-target vs off-target effects.

**Aspirin × COX-2 (anti-inflammatory)**
> Model COX-2 inhibition by Aspirin at 5 µM. Explain the irreversible acetylation mechanism and quantify the expected confidence shift.

### Seeded knowledge graph

| Node | Type | Description |
|---|---|---|
| BCR-ABL | Protein | BCR-ABL fusion tyrosine kinase — driver of CML |
| EGFR | Protein | Epidermal Growth Factor Receptor — target in NSCLC |
| COX-2 | Protein | Cyclooxygenase-2 — mediator of inflammation |
| Imatinib | Compound | Gleevec / STI-571 · 493.6 Da — FDA-approved CML drug |
| Gefitinib | Compound | Iressa / ZD1839 · 446.9 Da — EGFR inhibitor for NSCLC |
| Aspirin | Compound | Acetylsalicylic acid · 180.2 Da — irreversible COX-2 inhibitor |

### Dashboard project structure

```
dashboard/           ← Next.js app
  app/
    api/simulate/    ← POST — runs the Gemini agent
    api/graph/       ← GET  — queries the knowledge graph
  components/        ← SimulationPanel, ResultPanel, GraphView
  lib/
    agentEngine.ts   ← Ollama tool-calling loop (llama3.1, fully local)
    memoryManager.ts ← Sub-graph extraction + Bayesian updates
    types.ts         ← Domain interfaces
  prisma/
    schema.prisma    ← Node, Edge, ProvenanceEntry, AgentSession tables
    seed.ts          ← Real drug/protein seed data
docker-compose.yml   ← pgvector/postgres on port 5432
```

---

## Python research codebase (`targetval-phase2` branch)

Veridicus is the institutional reasoning memory layer; MedReason-Bench is its public benchmark; the `targetval` module is the Phase 2 target-validation product with the 3-layer memory moat (Universal / Disease / Campaign). See [`docs/targetval_phase2_session.md`](docs/targetval_phase2_session.md) for the architecture overview and Phase 2 session log; [`docs/targetval_phase2_plan.md`](docs/targetval_phase2_plan.md) for the build plan; [`docs/mapk_retro_review.md`](docs/mapk_retro_review.md) for the curated MAPK retrospective biology spot-check.

### Setup

```bash
# 1. Switch to the branch (this content does not exist on master)
git checkout targetval-phase2

# 2. Install Python deps (3.12+ required; 3.13 tested)
python -m pip install -r requirements.txt
# Runtime: anthropic, openai, duckdb, pydantic, click, tabulate, psycopg2-binary
# Dev: pytest

# 3. Optional: postgres for the legacy prior-auth flows (same docker-compose as the dashboard)
docker compose up -d
```

### Test suite

```bash
# Canonical targetval suite — 118 tests, ~0.4s
python -m pytest tests/test_targetval_*.py -v

# Full repo suite — 567 passing / 568 total; the one remaining failure
# is an unrelated drugdisc off-by-one (test_seed_rules_are_valid: asserts 8, finds 9)
python -m pytest tests/

# Targetval slice only (no LLM, no network, no DB)
python -m pytest tests/test_targetval_topology.py \
                 tests/test_targetval_layers.py \
                 tests/test_targetval_swarm.py \
                 tests/test_targetval_cross_agent.py \
                 tests/test_targetval_end_to_end.py -v
```

### CLI: `python -m medreason` (Veridicus core)

| Command | Purpose |
|---|---|
| `python -m medreason generate --n 30 --seed 42 --local --output cases.jsonl` | Generate benchmark cases. `--local` uses pre-written cases (no API needed). |
| `python -m medreason run --mode {zero_shot,memory,both} --n 30 --input cases.jsonl --format {table,json}` | Run benchmark evaluation in zero-shot, memory-augmented, or both modes. |
| `python -m medreason report --format {table,json}` | View latest benchmark results. |
| `python -m medreason stats` | Show pattern store statistics. |

### CLI: `python -m medreason_bench` (MedReason-Bench harness)

| Command | Purpose |
|---|---|
| `python -m medreason_bench data build --source {lcd,adversarial,lcd_edge,aetna_lumbar,drugdisc,combined} --version v0.0 --seed 42` | Parse policy + build stratified manifest. Source choices: LCD template-expanded, v0.1 adversarial, 30-case LCD edge xlsx, 60-case Aetna lumbar MRI, 25-case drug-discovery lead-op, or combined. |
| `python -m medreason_bench splits verify --version v0.2` | Re-hash a manifest and confirm LeakGuard compatibility. |
| `python -m medreason_bench train --version v0.2 --split train --model haiku --gate-k 5 [--abstract-rules] [--include-failures] [--multi-policy]` | Populate the memory store from a training split. Honors `--critic-model` and `--proposer-model` overrides. |
| `python -m medreason_bench eval --runner claude --model sonnet --memory --split test --version v0.2 --seeds 1 2 3` | Run an AgentRunner against a split. Supports `--include-policy` (RAG), `--policy-max-chars` (sparse-RAG), `--rerank`, `--top-k`. Writes leaderboard entries under `medreason_bench/leaderboard/entries/`. |

### CLI: `python -m medreason_bench.targetval` (target validation product, Phase 2)

| Command | Purpose |
|---|---|
| `python -m medreason_bench.targetval data build-mapk-retro --version v0.2 --out mapk_v0_2.jsonl` | Materialize the curated MAPK retrospective fixture (~22 BRAF/MEK/ERK/KRAS/NRAS/EGFR/MET/ALK targets with public-literature outcomes + bypass mechanisms). Universal-layer-safe. |
| `python -m medreason_bench.targetval data ingest-customer --customer recursion --targets targets.csv [--outcomes outcomes.jsonl]` | Ingest a customer's CSV/JSONL of targets into Campaign-layer cases. Stamps `customer_tag` on every `InternalEvidence` so the layer policy can enforce per-tenant boundaries. Returns an `IngestReport` with `n_cases` / `n_rejected`. |
| `python -m medreason_bench.targetval swarm dryrun [--campaign recursion] [--seeds 1 2 3] [--no-memory]` | Run the 1-agent-per-target swarm against a campaign's case set using the fake LLM (`FakeLLMClient`). No real API calls. Prints per-target memos + aggregate ranking. |
| `python -m medreason_bench.targetval card write --out prediction.json` | Write a deterministic `TargetValPredictionCard` (SHA-256-stable serialization) for the current prediction state. Schema parallel to `medreason_bench/leadop/prediction_card.py`. |

### Python research project structure

```
medreason/                       ← Veridicus core (institutional reasoning memory)
  llm/                           ← LLMClient + FakeLLMClient + Claude/OpenAI/Gemini adapters
  ontology/                      ← BenchmarkCase, ReasoningRule, RuleStatus, codes
  extraction/                    ← critic + rule_proposer + rule_abstractor + generalization_gate + failure_analyzer
  retrieval/                     ← ontology_lookup + rerank
  runners/                       ← AgentRunner Protocol + Claude/OpenAI implementations + memory wrapper
  store/                         ← Sqlite-backed RuleStore + leak_guard + audit log
  prompts/                       ← Frozen prompt files + PROMPTS_LOCK.json
  posterior.py                   ← Bayesian rule confidence updates
  targetval/                     ← TARGET VALIDATION PRODUCT (Phase 2)
    case.py                      ← TargetValidationCase, TargetID, DiseaseContext, BypassMechanism
    evidence.py                  ← EvidenceBundle + per-source sub-bundles
    evidence_ingest.py           ← EvidencePipeline + fake fetchers + caching/retry plumbing
    topology.py                  ← PathwayCache + ParalogFamily + derive_bypass_signals
    swarm.py                     ← SwarmAgent + SwarmRunner (parallel via ThreadPoolExecutor)
    swarm_prompts.py             ← Per-target prompt template
    swarm_parsing.py             ← 4-strategy tolerant JSON parser + safe_parse_memo fallback
    layers.py                    ← Layer enum + LayerPolicy + LayerRouter (ingest_rule, promote_rule, retrieve_for_case)
    layer_stores.py              ← TargetvalRuleStore adapter + InMemoryRuleStore (no-persistence path)
    cross_agent_analyzer.py      ← THE MOAT: detect_systematic_errors + propose_corrective_rules
    cross_agent_prompts.py       ← Rule-proposer prompt template + repair suffix

medreason_bench/                 ← MedReason-Bench public benchmark
  data/                          ← LCD/NCD ingestion + case construction + fixtures
  splits/                        ← Stratified train/dev/test partitioning + fingerprints
  eval/                          ← Phase 4: harness + metrics + bootstrap CIs + McNemar
  leaderboard/                   ← Schema + entries/ JSON output + HF dataset card emitter
  leadop/                        ← Lead-op SAR direction benchmark (Crizotinib retro etc.)
  targetval/                     ← Phase 2 benchmark side
    mapk_retro.py                ← build_mapk_retro_seed() — 22 curated MAPK cases (v0.2)
    mapk_retro_data.py           ← Outcome / bypass mechanism mix
    mapk_retro_entries.py        ← Case entries 1-N
    mapk_retro_entries_extra.py  ← Case entries N+1-22
    schemas.py                   ← DuckDB tables (targets, bypass_outcomes, targetval_meta)
    recursion_ingest.py          ← Customer CSV/JSONL ingest with IngestReport
    customer_csv.py              ← parse_jsonl + parse_parquet (lazy pyarrow)
    synthetic.py                 ← 3 toy targets (BRAF/KRAS/EGFR) for smoke testing
    metrics.py                   ← top_k_target_hit + bypass_precision_recall + bootstrap_ci
    prediction_card.py           ← TargetValPredictionCard + TargetValPredictionEnvelope (SHA-256)
    __main__.py                  ← CLI surface (the table above)

tests/                           ← 567 tests; 118 are targetval (tests/test_targetval_*.py)
docs/
  targetval_phase2_plan.md       ← Architect's 796-line build plan
  targetval_phase2_session.md    ← Session log + run instructions + swarm lessons
  mapk_retro_review.md           ← Curated MAPK biology spot-check (one paragraph per case)
  dispatch/                      ← 6 per-builder dispatch briefs from Phase 2
```

### What the targetval moat enforces

The 3-layer memory split is the cross-customer defensibility story:

- **Universal layer** — mechanism rules trained only on public retrospectives. Customer-data-free. Cross-customer safe.
- **Disease layer** — disease/pathway-scoped rules (e.g., `oncology_mapk`). Cross-customer within a disease cluster.
- **Campaign layer** — per-customer, strictly tenant-scoped. Never leaves the tenant.

Enforcement: `LayerPolicy.validate_rule` runs at every `LayerRouter.ingest_rule` boundary. `LayerRouter.promote_rule` strips customer provenance + re-validates before writes to higher layers. The cross-agent rule proposer at `medreason/targetval/cross_agent_analyzer.py::propose_corrective_rules` runs `LayerPolicy(layer=Layer.UNIVERSAL).validate_rule(rule, customer_tag=None)` defense-in-depth on every emitted candidate.

---

## Which branch should I be on?

| If you want to… | Use |
|---|---|
| Run the dashboard / demo to a non-technical audience | `master` |
| Touch the Python research codebase, run benchmarks, extend targetval | `targetval-phase2` |
| See the Phase 2 build plan / coordination lessons | `targetval-phase2`, then read `docs/targetval_phase2_session.md` |
