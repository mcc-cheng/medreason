# medreason

Two sibling codebases sharing a repo:

1. **Drug Discovery Canvas** — Next.js dashboard for agentic in-silico drug-discovery simulation against a live knowledge graph. Lives on `master`. The default landing experience.
2. **Veridicus + MedReason-Bench + targetval** — the Python research codebase: institutional reasoning memory for healthcare AI agents, the prior-auth + lead-op + target-validation benchmarks, and the Phase 2 target-validation product with the 3-layer memory moat. Lives on the **`targetval-phase2`** branch (143 .py files, 4 CLIs, 567-test suite).

Pick the side you're working on below.

---

## Drug Discovery Canvas (Next.js dashboard)

An agentic in-silico drug discovery platform. A local Ollama agent (llama3.1) simulates compound–protein interactions and stores outcomes in a live knowledge graph (PostgreSQL via Prisma). Each simulation run updates Bayesian confidence scores on interaction edges, which propagate through protein–protein similarity networks to infer new drug–target relationships.

The UI is a full-screen interactive force-directed graph — nodes are proteins and compounds, edges are interactions. Running a simulation highlights the relevant pathway in the graph.

### Knowledge graph pipeline

Three-stage ingestion builds the graph from public datasets:

| Stage | Script | Source | Writes |
|---|---|---|---|
| 1. Drug–target binding | `scripts/ingest_chembl.py` | ChEMBL REST API | ~17 proteins, ~100 compounds, ~300 INHIBITS edges |
| 2. Protein similarity | `scripts/ingest_string.py` | STRING REST API | ~156 SIMILAR\_TO edges (bidirectional) |
| 3. Bayesian propagation | `POST /api/propagate` | Computed | ~700 inferred TARGETS edges |

Propagation math: `inferred_conf = direct_conf × similarity_score × 0.7`. If a drug inhibits protein A with confidence 0.9 and protein B is 80% similar to A, the system infers the drug likely targets B with confidence ~0.5.

### Running locally

**Prerequisites:** Docker, Node.js 18+, [Ollama](https://ollama.com)

```bash
# 1. Pull the model (one-time, ~4 GB)
ollama pull llama3.1

# 2. Start Postgres
docker compose up -d          # from repo root

# 3. Install and configure
cd dashboard
npm install
cp .env.example .env.local    # no API keys needed — fully local

# 4. Set up the database
npm run db:generate
npm run db:push
npm run db:seed               # seeds 6 real proteins/drugs with literature edges

# 5. Start the app
npm run dev                   # → http://localhost:3000
```

#### Optional: populate the full knowledge graph

After the app is running, run the ingestion pipeline to load real data from ChEMBL and STRING:

```bash
cd dashboard/scripts
python3 -m venv .venv && source .venv/bin/activate
pip install psycopg2-binary datasets

# ChEMBL drug–target binding (takes ~3 min, hits ChEMBL + UniProt APIs)
python ingest_chembl.py --limit 200

# STRING protein–protein similarity
python ingest_string.py --min-score 0.4

# Bayesian propagation (infers new drug→protein edges through similarity)
curl -X POST http://localhost:3000/api/propagate
```

Both scripts support `--dry-run` to preview without writing to the DB.

### Using the UI

- **Graph** — force-directed canvas fills the screen. Scroll to zoom, drag to pan, drag nodes to rearrange. Click any node for its detail card (name, molecular weight, external ID).
- **Edge type toggles** — switch between Drug→Protein (direct binding), Similarity (STRING), and Inferred (Bayesian propagation) edge layers.
- **Propagate button** — re-runs Bayesian propagation and refreshes the graph.
- **Run simulation** — select a protein and compound, pick an example prompt or write your own, submit. The selected nodes glow in the graph and the agent response appears in the right panel.

### Sample prompts

Click any example in the UI, or paste directly:

**Imatinib × BCR-ABL (CML targeted therapy)**
> Simulate Imatinib binding to BCR-ABL at 0.1 µM. Predict efficacy and toxicity, explain what the Bayesian confidence score means, and summarize what this simulation adds to our knowledge.

**Gefitinib × EGFR (lung cancer)**
> Analyze Gefitinib selectivity at EGFR at 1 µM. Assess its safety profile and predict on-target vs off-target effects.

**Aspirin × COX-2 (anti-inflammatory)**
> Model COX-2 inhibition by Aspirin at 5 µM. Explain the irreversible acetylation mechanism and quantify the expected confidence shift.

### Seed knowledge graph (db:seed)

| Node | Type | Details |
|---|---|---|
| BCR-ABL | Protein | BCR-ABL fusion tyrosine kinase — driver of CML |
| EGFR | Protein | Epidermal Growth Factor Receptor — target in NSCLC |
| COX-2 | Protein | Cyclooxygenase-2 — mediator of inflammation |
| Imatinib | Compound | Gleevec / STI-571 · 493.6 Da — FDA-approved CML drug |
| Gefitinib | Compound | Iressa / ZD1839 · 446.9 Da — EGFR inhibitor for NSCLC |
| Aspirin | Compound | Acetylsalicylic acid · 180.2 Da — irreversible COX-2 inhibitor |

After running the full pipeline: ~115 nodes (17 proteins + ~96 compounds) and ~980 edges.

### Dashboard project structure

```
dashboard/
  app/
    api/simulate/       ← POST — runs the Ollama agent
    api/graph/          ← GET  — queries the knowledge graph
    api/propagate/      ← POST — runs Bayesian confidence propagation
  components/
    GraphView.tsx       ← interactive force-directed graph (react-force-graph-2d)
    SimulationPanel.tsx ← node selector + prompt input
    ResultPanel.tsx     ← agent response + Bayesian update log
  lib/
    agentEngine.ts      ← Ollama tool-calling loop (llama3.1, fully local)
    memoryManager.ts    ← sub-graph extraction + Bayesian confidence updates
    propagation.ts      ← cross-target confidence propagation via SIMILAR_TO edges
    types.ts            ← domain interfaces
  prisma/
    schema.prisma       ← Node, Edge, ProvenanceEntry, AgentSession tables
    seed.ts             ← real drug/protein seed data (literature-sourced confidence)
  scripts/
    ingest_chembl.py    ← ChEMBL + UniProt ingestion (proteins, compounds, INHIBITS edges)
    ingest_string.py    ← STRING protein similarity ingestion (SIMILAR_TO edges)
    ingest_rxrx3.py     ← Recursion rxrx3-core phenomics ingestion (streaming, no images)
docker-compose.yml      ← pgvector/postgres on port 5432
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
| `python -m medreason_bench data build --source {lcd,adversarial,lcd_edge,aetna_lumbar,drugdisc,combined} --version v0.0 --seed 42` | Parse policy + build stratified manifest. |
| `python -m medreason_bench splits verify --version v0.2` | Re-hash a manifest and confirm LeakGuard compatibility. |
| `python -m medreason_bench train --version v0.2 --split train --model haiku --gate-k 5 [--abstract-rules] [--include-failures] [--multi-policy]` | Populate the memory store from a training split. |
| `python -m medreason_bench eval --runner claude --model sonnet --memory --split test --version v0.2 --seeds 1 2 3` | Run an AgentRunner against a split. Writes leaderboard entries under `medreason_bench/leaderboard/entries/`. |

### CLI: `python -m medreason_bench.targetval` (target validation product, Phase 2)

| Command | Purpose |
|---|---|
| `python -m medreason_bench.targetval data build-mapk-retro --version v0.2 --out mapk_v0_2.jsonl` | Materialize the curated MAPK retrospective fixture (~22 targets). |
| `python -m medreason_bench.targetval data ingest-customer --customer recursion --targets targets.csv [--outcomes outcomes.jsonl]` | Ingest a customer's CSV/JSONL of targets into Campaign-layer cases. |
| `python -m medreason_bench.targetval swarm dryrun [--campaign recursion] [--seeds 1 2 3] [--no-memory]` | Run the 1-agent-per-target swarm using the fake LLM. No real API calls. |
| `python -m medreason_bench.targetval card write --out prediction.json` | Write a deterministic `TargetValPredictionCard` (SHA-256-stable). |

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
    layers.py                    ← Layer enum + LayerPolicy + LayerRouter
    layer_stores.py              ← TargetvalRuleStore adapter + InMemoryRuleStore
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
    schemas.py                   ← DuckDB tables (targets, bypass_outcomes, targetval_meta)
    recursion_ingest.py          ← Customer CSV/JSONL ingest with IngestReport
    metrics.py                   ← top_k_target_hit + bypass_precision_recall + bootstrap_ci
    prediction_card.py           ← TargetValPredictionCard (SHA-256)
    __main__.py                  ← CLI surface

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
