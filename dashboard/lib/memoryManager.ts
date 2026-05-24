// ============================================================
// memoryManager.ts — Drug Discovery Canvas
// Handles sub-graph extraction and Bayesian confidence updates.
// All DB access goes through Prisma; the in-memory Map is a
// read-through cache for the current request lifetime.
// ============================================================

import { PrismaClient, Prisma } from '@prisma/client';
import type {
  CustomNode,
  InteractionEdge,
  SubGraph,
  KnowledgeGraph,
  BayesianUpdateInput,
  BayesianUpdateResult,
  ProvenanceEntry,
} from './types';

const prisma = new PrismaClient();

// ─── Prisma row → domain type mappers ────────────────────────

function mapNode(row: Prisma.NodeGetPayload<object>): CustomNode {
  return {
    id: row.id,
    name: row.name,
    type: row.type as CustomNode['type'],
    metadata: {
      molecularWeight:  row.molecularWeight ?? undefined,
      dangerLevel:      (row.dangerLevel as CustomNode['metadata']['dangerLevel']) ?? undefined,
      originDepartment: row.originDepartment,
      smiles:      (row as Record<string, unknown>).smiles      as string   | undefined ?? undefined,
      externalId:  (row as Record<string, unknown>).externalId  as string   | undefined ?? undefined,
      sequence:    (row as Record<string, unknown>).sequence    as string   | undefined ?? undefined,
      goTerms:     (row as Record<string, unknown>).goTerms     as string[] | undefined ?? undefined,
    },
  };
}

function mapEdge(
  row: Prisma.EdgeGetPayload<{ include: { provenanceLog: true } }>
): InteractionEdge {
  return {
    id: row.id,
    sourceId: row.sourceId,
    targetId: row.targetId,
    interactionType: row.interactionType as InteractionEdge['interactionType'],
    confidenceScore: row.confidenceScore,
    observationCount: row.observationCount,
    priorAlpha: row.priorAlpha,
    priorBeta: row.priorBeta,
    provenanceLog: row.provenanceLog.map((p) => ({
      timestamp: p.timestamp.toISOString(),
      agentReasoningSnapshot: p.agentReasoningSnapshot,
      simulationSource: p.simulationSource as ProvenanceEntry['simulationSource'],
      evidenceWeight: p.evidenceWeight,
      observedOutcome: p.observedOutcome as ProvenanceEntry['observedOutcome'],
    })),
  };
}

// ─── MemoryManager ────────────────────────────────────────────

export class MemoryManager {
  private cache: KnowledgeGraph = {
    nodes: new Map(),
    edges: new Map(),
  };

  // ── Full graph load ─────────────────────────────────────────

  async loadFullGraph(): Promise<KnowledgeGraph> {
    const [nodes, edges] = await Promise.all([
      prisma.node.findMany(),
      prisma.edge.findMany({ include: { provenanceLog: true } }),
    ]);

    this.cache.nodes.clear();
    this.cache.edges.clear();

    for (const n of nodes) this.cache.nodes.set(n.id, mapNode(n));
    for (const e of edges) this.cache.edges.set(e.id, mapEdge(e));

    return this.cache;
  }

  // ── Sub-graph extraction ────────────────────────────────────

  /**
   * Extract the ego-network around focalNodeIds up to `depth` hops.
   * Bounded traversal so the LLM context window stays manageable.
   */
  async extractSubGraph(focalNodeIds: string[], depth: number = 2): Promise<SubGraph> {
    const visitedNodeIds = new Set<string>(focalNodeIds);
    const visitedEdgeIds = new Set<string>();
    let frontier = [...focalNodeIds];

    for (let hop = 0; hop < depth; hop++) {
      const edges = await prisma.edge.findMany({
        where: {
          OR: [
            { sourceId: { in: frontier } },
            { targetId: { in: frontier } },
          ],
        },
        include: { provenanceLog: true },
      });

      const nextFrontier: string[] = [];
      for (const e of edges) {
        if (!visitedEdgeIds.has(e.id)) {
          visitedEdgeIds.add(e.id);
          this.cache.edges.set(e.id, mapEdge(e));
        }
        if (!visitedNodeIds.has(e.sourceId)) {
          visitedNodeIds.add(e.sourceId);
          nextFrontier.push(e.sourceId);
        }
        if (!visitedNodeIds.has(e.targetId)) {
          visitedNodeIds.add(e.targetId);
          nextFrontier.push(e.targetId);
        }
      }
      frontier = nextFrontier;
      if (frontier.length === 0) break;
    }

    // Fetch any nodes not yet in cache
    const uncachedIds = [...visitedNodeIds].filter((id) => !this.cache.nodes.has(id));
    if (uncachedIds.length > 0) {
      const rows = await prisma.node.findMany({ where: { id: { in: uncachedIds } } });
      for (const r of rows) this.cache.nodes.set(r.id, mapNode(r));
    }

    return {
      focalNodeIds,
      depth,
      nodes: [...visitedNodeIds].map((id) => this.cache.nodes.get(id)!).filter(Boolean),
      edges: [...visitedEdgeIds].map((id) => this.cache.edges.get(id)!).filter(Boolean),
    };
  }

  // ── Bayesian confidence update ──────────────────────────────

  /**
   * Beta-distribution update rule:
   *   new_alpha = old_alpha + (outcome === SUPPORT ? weight : 0)
   *   new_beta  = old_beta  + (outcome === CONTRADICT ? weight : 0)
   *   confidence = new_alpha / (new_alpha + new_beta)
   *
   * AMBIGUOUS outcomes increment observationCount but do not shift alpha/beta.
   */
  async updateEdgeConfidence(input: BayesianUpdateInput): Promise<BayesianUpdateResult> {
    const edge = await prisma.edge.findUniqueOrThrow({ where: { id: input.edgeId } });

    const w = input.evidenceWeight;
    const deltaAlpha = input.observedOutcome === 'SUPPORT' ? w : 0;
    const deltaBeta = input.observedOutcome === 'CONTRADICT' ? w : 0;

    const newAlpha = edge.priorAlpha + deltaAlpha;
    const newBeta = edge.priorBeta + deltaBeta;
    const newConfidence = newAlpha / (newAlpha + newBeta);
    const newCount = edge.observationCount + 1;

    const updated = await prisma.edge.update({
      where: { id: input.edgeId },
      data: {
        priorAlpha: newAlpha,
        priorBeta: newBeta,
        confidenceScore: newConfidence,
        observationCount: newCount,
        provenanceLog: {
          create: {
            agentReasoningSnapshot: input.provenanceEntry.agentReasoningSnapshot,
            simulationSource: input.provenanceEntry.simulationSource,
            evidenceWeight: input.evidenceWeight,
            observedOutcome: input.observedOutcome,
          },
        },
      },
    });

    // Invalidate cache entry so next extraction reflects the update
    this.cache.edges.delete(input.edgeId);

    return {
      edgeId: input.edgeId,
      previousConfidence: edge.confidenceScore,
      updatedConfidence: updated.confidenceScore,
      updatedObservationCount: newCount,
    };
  }

  // ── Node upsert ─────────────────────────────────────────────

  async upsertNode(node: CustomNode): Promise<void> {
    await prisma.node.upsert({
      where: { id: node.id },
      update: {
        name: node.name,
        type: node.type,
        molecularWeight: node.metadata.molecularWeight ?? null,
        dangerLevel: node.metadata.dangerLevel ?? null,
        originDepartment: node.metadata.originDepartment,
      },
      create: {
        id: node.id,
        name: node.name,
        type: node.type,
        molecularWeight: node.metadata.molecularWeight ?? null,
        dangerLevel: node.metadata.dangerLevel ?? null,
        originDepartment: node.metadata.originDepartment,
      },
    });
    this.cache.nodes.set(node.id, node);
  }

  // ── Edge upsert (creates if missing, then updates confidence) ──

  async upsertEdge(edge: Omit<InteractionEdge, 'id' | 'provenanceLog' | 'priorAlpha' | 'priorBeta'>): Promise<string> {
    const row = await prisma.edge.upsert({
      where: {
        sourceId_targetId_interactionType: {
          sourceId: edge.sourceId,
          targetId: edge.targetId,
          interactionType: edge.interactionType,
        },
      },
      update: {
        confidenceScore: edge.confidenceScore,
        observationCount: { increment: 1 },
      },
      create: {
        sourceId: edge.sourceId,
        targetId: edge.targetId,
        interactionType: edge.interactionType,
        confidenceScore: edge.confidenceScore,
        observationCount: edge.observationCount,
      },
    });
    this.cache.edges.delete(row.id);
    return row.id;
  }

  // ── Compound–protein interaction analysis ────────────────────

  /**
   * Analyze the relationship between a specific compound and a target protein:
   *   1. Direct edge (if any) with full Bayesian confidence stats
   *   2. All other compounds that also target the same protein, ranked by confidence
   *   3. All other proteins that the compound targets, ranked by confidence
   */
  async analyzeCompoundProteinInteraction(
    compoundId: string,
    proteinId: string,
  ): Promise<{
    compound: CustomNode | null;
    protein: CustomNode | null;
    directEdge: InteractionEdge | null;
    competingCompounds: Array<{ node: CustomNode | null; edge: InteractionEdge }>;
    otherTargets: Array<{ node: CustomNode | null; edge: InteractionEdge }>;
  }> {
    const [compoundRow, proteinRow] = await Promise.all([
      prisma.node.findUnique({ where: { id: compoundId } }),
      prisma.node.findUnique({ where: { id: proteinId } }),
    ]);
    const compound = compoundRow ? mapNode(compoundRow) : null;
    const protein  = proteinRow  ? mapNode(proteinRow)  : null;

    // Direct edge between this compound and protein
    const directRow = await prisma.edge.findFirst({
      where: { sourceId: compoundId, targetId: proteinId },
      include: { provenanceLog: true },
    });
    const directEdge = directRow ? mapEdge(directRow) : null;

    // Other compounds targeting the same protein, ranked by confidence
    const competingRows = await prisma.edge.findMany({
      where: { targetId: proteinId, sourceId: { not: compoundId } },
      include: { provenanceLog: true },
      orderBy: { confidenceScore: 'desc' },
    });
    const competingSourceIds = [...new Set(competingRows.map((e) => e.sourceId))];
    const competingNodes = await prisma.node.findMany({ where: { id: { in: competingSourceIds } } });
    const competingNodeMap = new Map(competingNodes.map((r) => [r.id, mapNode(r)]));
    const competingCompounds = competingRows.map((e) => ({
      node: competingNodeMap.get(e.sourceId) ?? null,
      edge: mapEdge(e),
    }));

    // Other proteins this compound targets, ranked by confidence
    const otherTargetRows = await prisma.edge.findMany({
      where: { sourceId: compoundId, targetId: { not: proteinId } },
      include: { provenanceLog: true },
      orderBy: { confidenceScore: 'desc' },
    });
    const otherTargetIds = [...new Set(otherTargetRows.map((e) => e.targetId))];
    const otherTargetNodes = await prisma.node.findMany({ where: { id: { in: otherTargetIds } } });
    const otherTargetNodeMap = new Map(otherTargetNodes.map((r) => [r.id, mapNode(r)]));
    const otherTargets = otherTargetRows.map((e) => ({
      node: otherTargetNodeMap.get(e.targetId) ?? null,
      edge: mapEdge(e),
    }));

    return { compound, protein, directEdge, competingCompounds, otherTargets };
  }

  // ── Formatting helpers for LLM context injection ─────────────

  /**
   * Render a SubGraph as a compact text block suitable for
   * inserting into a Gemini prompt.
   */
  static formatSubGraphForPrompt(sg: SubGraph): string {
    const nodeLines = sg.nodes.map(
      (n) =>
        `  [${n.type}] ${n.id} | MW: ${n.metadata.molecularWeight ?? 'unknown'} | danger: ${n.metadata.dangerLevel ?? 'unknown'}`
    );
    const edgeLines = sg.edges.map(
      (e) =>
        `  ${e.sourceId} --[${e.interactionType} conf=${e.confidenceScore.toFixed(2)} n=${e.observationCount}]--> ${e.targetId}`
    );
    return [
      `=== KNOWLEDGE GRAPH CONTEXT (depth=${sg.depth}) ===`,
      'Nodes:',
      ...nodeLines,
      'Edges:',
      ...edgeLines,
      '=== END CONTEXT ===',
    ].join('\n');
  }
}
