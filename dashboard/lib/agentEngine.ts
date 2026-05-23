import { Ollama, type Message, type Tool } from 'ollama';
import { PrismaClient } from '@prisma/client';
import { MemoryManager } from './memoryManager';
import type {
  AgentRunInput,
  AgentRunOutput,
  AgentToolCall,
  AgentMessage,
  SimulationResult,
  BayesianUpdateResult,
} from './types';

const prisma = new PrismaClient();

// ─── Mock simulation tool implementations ─────────────────────

function runKnockoutSimulation(
  compoundId: string,
  proteinId: string,
  concentration: number
): SimulationResult {
  const efficacy = Math.min(1, Math.abs(Math.sin(concentration * 3.14)) * 0.8 + 0.1);
  const toxicity = Math.max(0, 1 - efficacy * 0.6 + 0.1);
  return {
    predictedEfficacy: parseFloat(efficacy.toFixed(3)),
    predictedToxicity: parseFloat(toxicity.toFixed(3)),
    primaryInteractionsDiscovered: [],
    logSummary: `Knockout: ${compoundId} on ${proteinId} at ${concentration}µM → efficacy=${efficacy.toFixed(2)}, toxicity=${toxicity.toFixed(2)}`,
  };
}

function assessCompoundSafety(
  compoundId: string,
  mw: number,
  dangerLevel: string
): { safetyScore: number; flags: string[] } {
  const flags: string[] = [];
  let score = 1.0;
  if (mw > 500) { flags.push('LIPINSKI_MW_VIOLATION'); score -= 0.2; }
  if (dangerLevel === 'CHAOTIC') { flags.push('CHAOTIC_DANGER_LEVEL'); score -= 0.4; }
  if (dangerLevel === 'MEDIUM')  { flags.push('MODERATE_RISK'); score -= 0.1; }
  return { safetyScore: parseFloat(Math.max(0, score).toFixed(3)), flags };
}

// ─── Ollama tool schema declarations ─────────────────────────

const TOOLS: Tool[] = [
  {
    type: 'function',
    function: {
      name: 'run_knockout_simulation',
      description: 'Run a compound-protein knockout simulation. Returns predicted efficacy and toxicity scores.',
      parameters: {
        type: 'object',
        properties: {
          compound_id:       { type: 'string', description: 'ID of the chemical compound (e.g. "IMATINIB", "GEFITINIB", "ASPIRIN")' },
          protein_id:        { type: 'string', description: 'ID of the protein/biomolecule (e.g. "BCR-ABL", "EGFR", "COX-2")' },
          concentration_um:  { type: 'number', description: 'Compound concentration in micromolar (µM)' },
        },
        required: ['compound_id', 'protein_id', 'concentration_um'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'assess_compound_safety',
      description: 'Assess the safety profile of a compound based on molecular weight and danger metadata.',
      parameters: {
        type: 'object',
        properties: {
          compound_id:      { type: 'string', description: 'ID of the compound node' },
          molecular_weight: { type: 'number', description: 'Molecular weight of the compound in Da' },
          danger_level:     { type: 'string', description: 'Danger level: LOW | MEDIUM | CHAOTIC' },
        },
        required: ['compound_id', 'molecular_weight', 'danger_level'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'query_graph',
      description: 'Query the knowledge graph for nodes and interactions. Pass empty node_ids to list everything.',
      parameters: {
        type: 'object',
        properties: {
          node_ids: {
            type: 'array',
            items: { type: 'string' },
            description: 'List of node IDs to query. Pass [] to list all nodes.',
          },
          depth: { type: 'number', description: 'Hops to traverse from each focal node (1–3). Default 1.' },
        },
        required: ['node_ids'],
      },
    },
  },
];

// ─── Tool dispatcher ──────────────────────────────────────────

async function dispatchTool(
  name: string,
  args: Record<string, unknown>,
  memory: MemoryManager
): Promise<unknown> {
  switch (name) {
    case 'run_knockout_simulation':
      return runKnockoutSimulation(
        args.compound_id as string,
        args.protein_id as string,
        args.concentration_um as number
      );
    case 'assess_compound_safety':
      return assessCompoundSafety(
        args.compound_id as string,
        args.molecular_weight as number,
        args.danger_level as string
      );
    case 'query_graph': {
      const ids   = (args.node_ids as string[]) ?? [];
      const depth = (args.depth as number) ?? 1;
      if (ids.length === 0) {
        const graph = await memory.loadFullGraph();
        return {
          nodes: [...graph.nodes.values()],
          edges: [...graph.edges.values()].map((e) => ({ ...e, provenanceLog: undefined })),
        };
      }
      const sg = await memory.extractSubGraph(ids, depth);
      return {
        nodes: sg.nodes,
        edges: sg.edges.map((e) => ({ ...e, provenanceLog: undefined })),
        formatted: MemoryManager.formatSubGraphForPrompt(sg),
      };
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ─── AgentEngine ──────────────────────────────────────────────

export class AgentEngine {
  private ollama: Ollama;
  private memory: MemoryManager;
  private model: string;

  constructor() {
    this.ollama = new Ollama({ host: process.env.OLLAMA_HOST ?? 'http://localhost:11434' });
    this.memory = new MemoryManager();
    this.model  = process.env.OLLAMA_MODEL ?? 'llama3.1';
  }

  async run(input: AgentRunInput): Promise<AgentRunOutput> {
    const { userPrompt, focalNodeIds, subGraphDepth = 2 } = input;

    const subGraph    = await this.memory.extractSubGraph(focalNodeIds, subGraphDepth);
    const graphContext = MemoryManager.formatSubGraphForPrompt(subGraph);

    const systemPrompt = [
      'You are an expert computational biologist operating an in-silico drug discovery simulation platform.',
      'You have access to a live knowledge graph of proteins, compounds, and interaction edges — each edge has a Bayesian confidence score updated by past simulations.',
      'When asked to simulate or analyze a compound-protein interaction:',
      '  1. Query the graph to understand what is already known.',
      '  2. Run the relevant simulation or safety tools.',
      '  3. Interpret the results: explain predicted efficacy, toxicity, and confidence scores.',
      '  4. Summarize what the simulation learned and how it updates the knowledge graph.',
      'Be concise, scientifically precise, and reason step by step.',
    ].join('\n');

    const initialContent = `${graphContext}\n\nUser request:\n${userPrompt}`;

    const messages: Message[] = [
      { role: 'system',  content: systemPrompt },
      { role: 'user',    content: initialContent },
    ];

    const toolCallsExecuted: AgentToolCall[]      = [];
    const confidenceUpdates: BayesianUpdateResult[] = [];
    const nodesUpserted: string[]                  = [];
    const edgesUpserted: string[]                  = [];
    const sessionMessages: AgentMessage[]          = [{ role: 'user', content: userPrompt }];

    let finalResponse = '';
    let iterations    = 0;
    const MAX_ITERATIONS = 8;

    while (iterations < MAX_ITERATIONS) {
      iterations++;

      const response = await this.ollama.chat({
        model:    this.model,
        messages,
        tools:    TOOLS,
      });

      const assistantMsg = response.message;
      messages.push(assistantMsg);

      // No tool calls — done
      if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
        finalResponse = assistantMsg.content ?? '';
        sessionMessages.push({ role: 'model', content: finalResponse });
        break;
      }

      // Execute each tool call
      for (const toolCall of assistantMsg.tool_calls) {
        const { name, arguments: args } = toolCall.function;
        const callId = `call_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

        toolCallsExecuted.push({
          toolName:   name,
          toolCallId: callId,
          input:      args as Record<string, unknown>,
        });

        let toolOutput: unknown;
        try {
          toolOutput = await dispatchTool(name, args as Record<string, unknown>, this.memory);

          if (name === 'run_knockout_simulation') {
            const sim        = toolOutput as SimulationResult;
            const compoundId = (args as Record<string, unknown>).compound_id as string;
            const proteinId  = (args as Record<string, unknown>).protein_id  as string;

            const edgeId = await this.memory.upsertEdge({
              sourceId:        compoundId,
              targetId:        proteinId,
              interactionType: 'INHIBITS',
              confidenceScore: sim.predictedEfficacy,
              observationCount: 1,
            });
            edgesUpserted.push(edgeId);

            const outcome =
              sim.predictedEfficacy > 0.5    ? 'SUPPORT'    :
              sim.predictedToxicity  > 0.7   ? 'CONTRADICT' : 'AMBIGUOUS';

            const update = await this.memory.updateEdgeConfidence({
              edgeId,
              observedOutcome: outcome,
              evidenceWeight: 1.0,
              provenanceEntry: {
                agentReasoningSnapshot: `Agent run: ${sim.logSummary}`,
                simulationSource: 'MOCK_TOOL',
                evidenceWeight: 1.0,
                observedOutcome: outcome,
              },
            });
            confidenceUpdates.push(update);
          }
        } catch (err) {
          toolOutput = { error: String(err) };
        }

        sessionMessages.push({ role: 'tool_result', content: JSON.stringify(toolOutput) });

        // Ollama tool results go as a 'tool' role message
        messages.push({ role: 'tool', content: JSON.stringify(toolOutput) });
      }
    }

    if (!finalResponse) {
      finalResponse = 'Agent completed tool calls but produced no final text response.';
    }

    const session = await prisma.agentSession.create({
      data: {
        userPrompt,
        finalResponse,
        totalTokens:   0, // Ollama doesn't report tokens the same way
        toolCallsJson: toolCallsExecuted as never,
        mutationsJson: { nodesUpserted, edgesUpserted, confidenceUpdates } as never,
        messages:      sessionMessages as never,
      },
    });

    return {
      sessionId:         session.id,
      finalResponse,
      toolCallsExecuted,
      graphMutations:    { nodesUpserted, edgesUpserted, confidenceUpdates },
      totalTokensUsed:   0,
    };
  }
}
