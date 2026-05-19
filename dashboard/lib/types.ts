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
  };
}

// ─── Graph Edge ──────────────────────────────────────────────

export type EdgeInteractionType =
  | 'AGGRESSIVELY_TICKLES'
  | 'MUTUALLY_ANNOYS'
  | 'PERMANENTLY_MELTS'
  | 'BENIGN_IGNORE';

export interface ProvenanceEntry {
  timestamp: string; // ISO-8601
  agentReasoningSnapshot: string;
  simulationSource: 'MOCK_TOOL' | 'USER_DIRECT_OVERWRITE' | 'RAG_LITERATURE';
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

export const MOCK_NODE_IDS = {
  PROTEIN_ALPHA: 'GLOW-SQUID-9',
  PROTEIN_BETA: 'HONEY-BADGER-X',
  COMPOUND_ALPHA: 'CHIPOTLE-MAYO-42',
  COMPOUND_BETA: 'CAFFEINE-OVERDOSE-99',
} as const;

export const MOCK_DEPARTMENT = 'Department of Mad Science and Taco Logistics' as const;

export type MockNodeId = (typeof MOCK_NODE_IDS)[keyof typeof MOCK_NODE_IDS];
