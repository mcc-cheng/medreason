# Drug Discovery Canvas

An agentic in-silico drug discovery platform. A Gemini-powered agent simulates compound–protein interactions and stores outcomes in a live knowledge graph (PostgreSQL via Prisma). Each simulation run updates Bayesian confidence scores on interaction edges.

## Running locally

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

## Sample prompts

Click any example in the UI, or paste these directly:

**Imatinib × BCR-ABL (CML targeted therapy)**
> Simulate Imatinib binding to BCR-ABL at 0.1 µM. Predict efficacy and toxicity, explain what the Bayesian confidence score means, and summarize what this simulation adds to our knowledge.

**Gefitinib × EGFR (lung cancer)**
> Analyze Gefitinib selectivity at EGFR at 1 µM. Assess its safety profile and predict on-target vs off-target effects.

**Aspirin × COX-2 (anti-inflammatory)**
> Model COX-2 inhibition by Aspirin at 5 µM. Explain the irreversible acetylation mechanism and quantify the expected confidence shift.

## Seeded knowledge graph

| Node | Type | Description |
|---|---|---|
| BCR-ABL | Protein | BCR-ABL fusion tyrosine kinase — driver of CML |
| EGFR | Protein | Epidermal Growth Factor Receptor — target in NSCLC |
| COX-2 | Protein | Cyclooxygenase-2 — mediator of inflammation |
| Imatinib | Compound | Gleevec / STI-571 · 493.6 Da — FDA-approved CML drug |
| Gefitinib | Compound | Iressa / ZD1839 · 446.9 Da — EGFR inhibitor for NSCLC |
| Aspirin | Compound | Acetylsalicylic acid · 180.2 Da — irreversible COX-2 inhibitor |

## Project structure

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
