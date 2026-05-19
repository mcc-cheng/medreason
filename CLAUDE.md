# CLAUDE.md

Drug Discovery Canvas — project notes for Claude Code.

## What this project is

An agentic in-silico drug discovery platform. A Gemini-powered agent simulates
compound–protein knockout interactions and stores outcomes in a live knowledge
graph (PostgreSQL via Prisma). Each simulation run updates Bayesian confidence
scores on interaction edges. The UI is a Next.js App Router app in `dashboard/`.

## Project structure

```
dashboard/           ← Next.js app (run `npm run dev` from here)
  app/
    api/simulate/    ← POST — runs the Gemini agent
    api/graph/       ← GET  — queries the knowledge graph
    page.tsx         ← Main canvas UI
  components/        ← SimulationPanel, ResultPanel, GraphView
  lib/
    types.ts         ← All domain interfaces
    memoryManager.ts ← Sub-graph extraction + Bayesian confidence updates
    agentEngine.ts   ← Gemini tool-calling loop
  prisma/
    schema.prisma    ← Node, Edge, ProvenanceEntry, AgentSession tables
    seed.ts          ← Mock seed data (GLOW-SQUID-9, CHIPOTLE-MAYO-42, etc.)
  .env.example       ← Copy to .env.local with your keys
docker-compose.yml   ← pgvector/postgres on port 5432
```

## Setup commands (run from `dashboard/`)

```bash
npm install
cp .env.example .env.local   # then fill in GEMINI_API_KEY
docker compose up -d         # start postgres (run from repo root)
npm run db:generate          # prisma generate
npm run db:push              # push schema to DB
npm run db:seed              # load mock nodes + edge
npm run dev                  # start Next.js at localhost:3000
```

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
