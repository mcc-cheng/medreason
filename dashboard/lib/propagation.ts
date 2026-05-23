// ============================================================
// propagation.ts — Bayesian confidence propagation
//
// Core idea: if drug D inhibits protein A with confidence c,
// and protein B is similar to A with score s, we infer
// D likely targets B with confidence ≈ c × s × DECAY.
//
// We encode this as soft Beta-prior pseudo-counts so the
// inference blends naturally with experimental observations
// via the same Bayesian update machinery in MemoryManager.
// ============================================================

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const CROSS_TARGET_DECAY   = 0.7;  // similarity chain attenuation
const INFERRED_EVIDENCE_W  = 0.3;  // inferred evidence weight vs direct (1.0)
const MIN_INFERRED_CONF    = 0.15; // discard very weak inferences
const MIN_SIMILARITY_SCORE = 0.3;  // ignore low-similarity protein pairs

export interface PropagationSummary {
  edgesInspected:  number;
  similarityPairs: number;
  edgesInferred:   number;
  edgesUpdated:    number;
}

export async function propagateConfidence(): Promise<PropagationSummary> {
  // ── 1. Load all direct drug→protein edges ─────────────────
  const directEdges = await prisma.edge.findMany({
    where: { interactionType: { in: ['INHIBITS', 'ACTIVATES', 'BINDS'] } },
    select: { id: true, sourceId: true, targetId: true, confidenceScore: true },
  });

  // ── 2. Load all protein–protein similarity edges ───────────
  const simEdges = await prisma.edge.findMany({
    where: { interactionType: 'SIMILAR_TO' },
    select: { sourceId: true, targetId: true, confidenceScore: true },
  });

  // Build bidirectional map: proteinId → [{neighborId, score}]
  const simMap = new Map<string, { neighborId: string; score: number }[]>();
  for (const e of simEdges) {
    if (e.confidenceScore < MIN_SIMILARITY_SCORE) continue;
    const addEntry = (from: string, to: string) => {
      if (!simMap.has(from)) simMap.set(from, []);
      simMap.get(from)!.push({ neighborId: to, score: e.confidenceScore });
    };
    addEntry(e.sourceId, e.targetId);
    addEntry(e.targetId, e.sourceId); // similarity is symmetric
  }

  // ── 3. Propagate ───────────────────────────────────────────
  let edgesInferred = 0;
  let edgesUpdated  = 0;

  for (const direct of directEdges) {
    const drugId    = direct.sourceId;
    const proteinId = direct.targetId;
    const directConf = direct.confidenceScore;

    const neighbors = simMap.get(proteinId) ?? [];

    for (const { neighborId, score } of neighbors) {
      // Don't overwrite the original direct interaction
      if (neighborId === proteinId) continue;

      const inferredConf = directConf * score * CROSS_TARGET_DECAY;
      if (inferredConf < MIN_INFERRED_CONF) continue;

      // Pseudo-counts for the Beta prior (scaled by evidence weight)
      const alpha = inferredConf * INFERRED_EVIDENCE_W * 10;
      const beta  = (1 - inferredConf) * INFERRED_EVIDENCE_W * 10;

      const reasoning =
        `Bayesian propagation: ${drugId} → ${proteinId} (conf=${directConf.toFixed(3)}) ` +
        `× similarity(${proteinId}, ${neighborId})=${score.toFixed(3)} ` +
        `× decay=${CROSS_TARGET_DECAY} → inferred conf=${inferredConf.toFixed(3)}`;

      // Check if a TARGETS edge already exists
      const existing = await prisma.edge.findUnique({
        where: {
          sourceId_targetId_interactionType: {
            sourceId:        drugId,
            targetId:        neighborId,
            interactionType: 'TARGETS',
          },
        },
      });

      if (existing) {
        // Only update if our new inference is stronger
        if (inferredConf <= existing.confidenceScore) continue;

        await prisma.edge.update({
          where: { id: existing.id },
          data: {
            confidenceScore:  inferredConf,
            observationCount: { increment: 1 },
            priorAlpha:       { increment: alpha },
            priorBeta:        { increment: beta },
            provenanceLog: {
              create: {
                agentReasoningSnapshot: reasoning,
                simulationSource:       'BAYESIAN_PROPAGATION',
                evidenceWeight:         INFERRED_EVIDENCE_W,
                observedOutcome:        'SUPPORT',
              },
            },
          },
        });
        edgesUpdated++;
      } else {
        // Verify both nodes exist before creating edge
        const [drugNode, targetNode] = await Promise.all([
          prisma.node.findUnique({ where: { id: drugId },     select: { id: true } }),
          prisma.node.findUnique({ where: { id: neighborId }, select: { id: true } }),
        ]);
        if (!drugNode || !targetNode) continue;

        await prisma.edge.create({
          data: {
            sourceId:        drugId,
            targetId:        neighborId,
            interactionType: 'TARGETS',
            confidenceScore: inferredConf,
            observationCount: 1,
            priorAlpha:      alpha,
            priorBeta:       beta,
            provenanceLog: {
              create: {
                agentReasoningSnapshot: reasoning,
                simulationSource:       'BAYESIAN_PROPAGATION',
                evidenceWeight:         INFERRED_EVIDENCE_W,
                observedOutcome:        'SUPPORT',
              },
            },
          },
        });
        edgesInferred++;
      }
    }
  }

  return {
    edgesInspected:  directEdges.length,
    similarityPairs: simEdges.length,
    edgesInferred,
    edgesUpdated,
  };
}
