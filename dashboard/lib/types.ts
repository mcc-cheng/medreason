// ============================================================
// types.ts — Drug Discovery Canvas
// Domain types for the stateful Knowledge Graph engine.
// ============================================================

// ─── Graph Node ──────────────────────────────────────────────

export type NodeType = 'BIOMOLECULE' | 'CHEMICAL_CANDIDATE' | 'METABOLIC_PATHWAY';

export interface CustomNode {
  id: string;
  name: string;
  type: NodeType;
  metadata: {
    molecularWeight?: number;
    dangerLevel?: 'LOW' | 'MEDIUM' | 'CHAOTIC';
    originDepartment: string;
    smiles?: string;
    externalId?: string;
    sequence?: string;
    goTerms?: string[];
  };
}

// ─── Graph Edge ──────────────────────────────────────────────

export type EdgeInteractionType =
  | 'INHIBITS'
  | 'ACTIVATES'
  | 'BINDS'
  | 'ALLOSTERIC_MODULATOR'
  | 'SIMILAR_TO'
  | 'TARGETS'
  | 'ASSOCIATED_WITH';

export interface ProvenanceEntry {
  timestamp: string; // ISO-8601
  agentReasoningSnapshot: string;
  simulationSource: 'MOCK_TOOL' | 'USER_DIRECT_OVERWRITE' | 'RAG_LITERATURE' | 'RXRX3_PHENOMICS' | 'CHEMBL' | 'UNIPROT_STRING' | 'BAYESIAN_PROPAGATION';
  evidenceWeight: number;
  observedOutcome: 'SUPPORT' | 'CONTRADICT' | 'AMBIGUOUS';
}

export interface InteractionEdge {
  id: string;
  sourceId: string;
  targetId: string;
  interactionType: EdgeInteractionType;
  confidenceScore: number; // [0.0, 1.0]
  observationCount: number;
  priorAlpha: number;  // Bayesian pseudo-count for successes
  priorBeta: number;   // Bayesian pseudo-count for failures
  provenanceLog: ProvenanceEntry[];
}

// ─── Simulation ──────────────────────────────────────────────

export interface SimulationResult {
  predictedEfficacy: number;  // [0.0, 1.0]
  predictedToxicity: number;  // [0.0, 1.0]
  primaryInteractionsDiscovered: InteractionEdge[];
  logSummary: string;
}

// ─── Knowledge Graph ─────────────────────────────────────────

export interface KnowledgeGraph {
  nodes: Map<string, CustomNode>;
  edges: Map<string, InteractionEdge>;
}

export interface SubGraph {
  focalNodeIds: string[];
  depth: number;
  nodes: CustomNode[];
  edges: InteractionEdge[];
}

// ─── Bayesian Confidence Update ──────────────────────────────

export interface BayesianUpdateInput {
  edgeId: string;
  observedOutcome: 'SUPPORT' | 'CONTRADICT' | 'AMBIGUOUS';
  evidenceWeight: number;
  provenanceEntry: Omit<ProvenanceEntry, 'timestamp'>;
}

export interface BayesianUpdateResult {
  edgeId: string;
  previousConfidence: number;
  updatedConfidence: number;
  updatedObservationCount: number;
}

// ─── Agent Tool Definitions ───────────────────────────────────

export interface AgentToolCall {
  toolName: string;
  toolCallId: string;
  input: Record<string, unknown>;
}

export interface AgentToolResult {
  toolCallId: string;
  toolName: string;
  output: unknown;
  isError: boolean;
}

// ─── Agent Session ────────────────────────────────────────────

export interface AgentMessage {
  role: 'user' | 'model' | 'tool_result';
  content: string;
}

export interface AgentRunInput {
  userPrompt: string;
  focalNodeIds: string[];
  subGraphDepth?: number;
}

export interface GraphMutationSummary {
  nodesUpserted: string[];
  edgesUpserted: string[];
  confidenceUpdates: BayesianUpdateResult[];
}

export interface AgentRunOutput {
  sessionId: string;
  finalResponse: string;
  toolCallsExecuted: AgentToolCall[];
  graphMutations: GraphMutationSummary;
  totalTokensUsed: number;
}

// ─── Mock Fixture Constants ───────────────────────────────────

export const SEED_NODE_IDS = {
  PROTEIN_BCR_ABL: 'BCR-ABL',
  PROTEIN_EGFR: 'EGFR',
  PROTEIN_COX2: 'COX-2',
  COMPOUND_IMATINIB: 'IMATINIB',
  COMPOUND_GEFITINIB: 'GEFITINIB',
  COMPOUND_ASPIRIN: 'ASPIRIN',
} as const;

export const SEED_DEPARTMENT = 'Computational Biology Lab' as const;

export type SeedNodeId = (typeof SEED_NODE_IDS)[keyof typeof SEED_NODE_IDS];
