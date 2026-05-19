// ============================================================
// agentEngine.ts — Drug Discovery Canvas
// Gemini tool-calling loop: consumes a user prompt, evaluates
// it against the knowledge graph, runs mock simulation tools,
// and commits learned outcomes back to the database via
// MemoryManager.
// ============================================================

import {
  GoogleGenerativeAI,
  SchemaType,
  type Tool,
  type FunctionDeclaration,
  type GenerateContentRequest,
  type Content,
} from '@google/generative-ai';
import { PrismaClient } from '@prisma/client';
import { MemoryManager } from './memoryManager';
import type {
  AgentRunInput,
  AgentRunOutput,
  AgentToolCall,
  AgentMessage,
  GraphMutationSummary,
  SimulationResult,
  BayesianUpdateResult,
} from './types';

const prisma = new PrismaClient();

// ─── Mock simulation tool implementations ─────────────────────

/**
 * Simulates a compound→protein knockout interaction.
 * In production this would call a real docking/MD engine.
 */
function runKnockoutSimulation(
  compoundId: string,
  proteinId: string,
  concentration: number
): SimulationResult {
  const seed = (compoundId.length * proteinId.length * concentration) % 1;
  const efficacy = Math.min(1, Math.abs(Math.sin(concentration * 3.14)) * 0.8 + 0.1);
  const toxicity = Math.max(0, 1 - efficacy * 0.6 + seed * 0.2);

  return {
    predictedEfficacy: parseFloat(efficacy.toFixed(3)),
    predictedToxicity: parseFloat(toxicity.toFixed(3)),
    primaryInteractionsDiscovered: [],
    logSummary: `Mock knockout: ${compoundId} on ${proteinId} at ${concentration}µM → efficacy=${efficacy.toFixed(2)}, toxicity=${toxicity.toFixed(2)}`,
  };
}

/**
 * Scores compound safety from molecular weight and danger level metadata.
 */
function assessCompoundSafety(
  compoundId: string,
  mw: number,
  dangerLevel: string
): { safetyScore: number; flags: string[] } {
  const flags: string[] = [];
  let score = 1.0;

  if (mw > 500) { flags.push('LIPINSKI_MW_VIOLATION'); score -= 0.2; }
  if (dangerLevel === 'CHAOTIC') { flags.push('CHAOTIC_DANGER_LEVEL'); score -= 0.4; }
  if (dangerLevel === 'MEDIUM') { flags.push('MODERATE_RISK'); score -= 0.1; }

  return {
    safetyScore: parseFloat(Math.max(0, score).toFixed(3)),
    flags,
  };
}

// ─── Gemini tool schema declarations ─────────────────────────

const TOOLS: Tool[] = [
  {
    functionDeclarations: [
      {
        name: 'run_knockout_simulation',
        description:
          'Run a compound-protein knockout simulation. Returns predicted efficacy and toxicity scores.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            compound_id: {
              type: SchemaType.STRING,
              description: 'ID of the chemical compound node (e.g. "IMATINIB", "GEFITINIB", "ASPIRIN")',
            },
            protein_id: {
              type: SchemaType.STRING,
              description: 'ID of the protein/biomolecule node (e.g. "BCR-ABL", "EGFR", "COX-2")',
            },
            concentration_um: {
              type: SchemaType.NUMBER,
              description: 'Compound concentration in micromolar (µM)',
            },
          },
          required: ['compound_id', 'protein_id', 'concentration_um'],
        },
      } satisfies FunctionDeclaration,
      {
        name: 'assess_compound_safety',
        description:
          'Assess the safety profile of a compound based on its molecular weight and danger metadata.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            compound_id: {
              type: SchemaType.STRING,
              description: 'ID of the compound node',
            },
            molecular_weight: {
              type: SchemaType.NUMBER,
              description: 'Molecular weight of the compound in Da',
            },
            danger_level: {
              type: SchemaType.STRING,
              description: 'Danger level: LOW | MEDIUM | CHAOTIC',
            },
          },
          required: ['compound_id', 'molecular_weight', 'danger_level'],
        },
      } satisfies FunctionDeclaration,
      {
        name: 'query_graph',
        description:
          'Query the knowledge graph for nodes and their known interactions. Use this to look up what compounds or proteins are known and what their current confidence scores are.',
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            node_ids: {
              type: SchemaType.ARRAY,
              items: { type: SchemaType.STRING },
              description: 'List of node IDs to query. Pass empty array to list all nodes.',
            },
            depth: {
              type: SchemaType.NUMBER,
              description: 'How many hops to traverse from each focal node (1–3). Default 1.',
            },
          },
          required: ['node_ids'],
        },
      } satisfies FunctionDeclaration,
    ],
  },
];

// ─── Tool dispatcher ──────────────────────────────────────────

async function dispatchTool(
  name: string,
  args: Record<string, unknown>,
  memory: MemoryManager
): Promise<unknown> {
  switch (name) {
    case 'run_knockout_simulation': {
      return runKnockoutSimulation(
        args.compound_id as string,
        args.protein_id as string,
        args.concentration_um as number
      );
    }
    case 'assess_compound_safety': {
      return assessCompoundSafety(
        args.compound_id as string,
        args.molecular_weight as number,
        args.danger_level as string
      );
    }
    case 'query_graph': {
      const ids = (args.node_ids as string[]) ?? [];
      const depth = (args.depth as number) ?? 1;
      if (ids.length === 0) {
        const graph = await memory.loadFullGraph();
        return {
          nodes: [...graph.nodes.values()],
          edges: [...graph.edges.values()].map((e) => ({
            ...e,
            provenanceLog: undefined, // omit full log from tool response
          })),
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
  private genai: GoogleGenerativeAI;
  private memory: MemoryManager;

  constructor(apiKey: string) {
    this.genai = new GoogleGenerativeAI(apiKey);
    this.memory = new MemoryManager();
  }

  async run(input: AgentRunInput): Promise<AgentRunOutput> {
    const { userPrompt, focalNodeIds, subGraphDepth = 2 } = input;

    // 1. Extract relevant sub-graph and format it as context
    const subGraph = await this.memory.extractSubGraph(focalNodeIds, subGraphDepth);
    const graphContext = MemoryManager.formatSubGraphForPrompt(subGraph);

    const systemInstruction = [
      'You are an expert computational biologist operating an in-silico drug discovery simulation platform.',
      'You have access to a live knowledge graph of proteins, compounds, and their interaction edges — each edge has a Bayesian confidence score updated by past simulations.',
      'When a user asks you to simulate or analyze a compound-protein interaction:',
      '  1. Query the graph to understand what is already known.',
      '  2. Run the relevant simulation or safety tools.',
      '  3. Interpret the results: explain predicted efficacy, toxicity, and what the confidence scores mean.',
      '  4. Summarize what the simulation learned and how it should update our knowledge.',
      'Be concise, scientifically precise, and explain your reasoning step by step.',
    ].join('\n');

    const initialUserContent = [
      graphContext,
      '',
      'User request:',
      userPrompt,
    ].join('\n');

    const model = this.genai.getGenerativeModel({
      model: 'gemini-2.0-flash',
      systemInstruction,
      tools: TOOLS,
    });

    // 2. Agentic tool-calling loop
    const toolCallsExecuted: AgentToolCall[] = [];
    const confidenceUpdates: BayesianUpdateResult[] = [];
    const nodesUpserted: string[] = [];
    const edgesUpserted: string[] = [];
    const messages: AgentMessage[] = [{ role: 'user', content: userPrompt }];

    const history: Content[] = [];
    let currentUserContent = initialUserContent;
    let totalTokens = 0;
    let finalResponse = '';
    let iterations = 0;
    const MAX_ITERATIONS = 6;

    while (iterations < MAX_ITERATIONS) {
      iterations++;

      const request: GenerateContentRequest = {
        contents: [
          ...history,
          { role: 'user', parts: [{ text: currentUserContent }] },
        ],
      };

      const result = await model.generateContent(request);
      const response = result.response;

      if (response.usageMetadata) {
        totalTokens += response.usageMetadata.totalTokenCount ?? 0;
      }

      const candidate = response.candidates?.[0];
      if (!candidate) break;

      // Push the model turn into history
      history.push({ role: 'user', parts: [{ text: currentUserContent }] });
      history.push({ role: 'model', parts: candidate.content.parts });

      const functionCalls = candidate.content.parts.filter((p) => p.functionCall);

      if (functionCalls.length === 0) {
        // No tool calls — extract text response and exit
        finalResponse = candidate.content.parts
          .filter((p) => p.text)
          .map((p) => p.text)
          .join('');
        messages.push({ role: 'model', content: finalResponse });
        break;
      }

      // 3. Execute all tool calls in this turn
      const toolResultParts: Content['parts'] = [];

      for (const part of functionCalls) {
        if (!part.functionCall) continue;
        const { name, args } = part.functionCall;
        const callId = `call_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

        const toolCall: AgentToolCall = {
          toolName: name,
          toolCallId: callId,
          input: args as Record<string, unknown>,
        };
        toolCallsExecuted.push(toolCall);

        let toolOutput: unknown;
        let isError = false;
        try {
          toolOutput = await dispatchTool(name, args as Record<string, unknown>, this.memory);

          // 4. If this was a simulation, commit the result back to the graph
          if (name === 'run_knockout_simulation') {
            const simResult = toolOutput as SimulationResult;
            const compoundId = (args as Record<string, unknown>).compound_id as string;
            const proteinId = (args as Record<string, unknown>).protein_id as string;

            // Upsert edge and update Bayesian confidence
            const edgeId = await this.memory.upsertEdge({
              sourceId: compoundId,
              targetId: proteinId,
              interactionType: 'INHIBITS',
              confidenceScore: simResult.predictedEfficacy,
              observationCount: 1,
            });
            edgesUpserted.push(edgeId);

            const outcome =
              simResult.predictedEfficacy > 0.5
                ? 'SUPPORT'
                : simResult.predictedToxicity > 0.7
                ? 'CONTRADICT'
                : 'AMBIGUOUS';

            const update = await this.memory.updateEdgeConfidence({
              edgeId,
              observedOutcome: outcome,
              evidenceWeight: 1.0,
              provenanceEntry: {
                agentReasoningSnapshot: `Gemini agent run: ${simResult.logSummary}`,
                simulationSource: 'MOCK_TOOL',
                observedOutcome: outcome,
              },
            });
            confidenceUpdates.push(update);
          }
        } catch (err) {
          toolOutput = { error: String(err) };
          isError = true;
        }

        messages.push({
          role: 'tool_result',
          content: JSON.stringify(toolOutput),
        });

        toolResultParts.push({
          functionResponse: {
            name,
            response: { result: toolOutput },
          },
        });
      }

      // Feed tool results back as the next user turn
      history.push({ role: 'user', parts: toolResultParts });
      currentUserContent = ''; // Gemini reads history; empty placeholder
    }

    if (!finalResponse) {
      finalResponse = 'Agent completed tool calls but produced no final text response.';
    }

    // 5. Persist session
    const session = await prisma.agentSession.create({
      data: {
        userPrompt,
        finalResponse,
        totalTokens,
        toolCallsJson: toolCallsExecuted as unknown as Prisma.InputJsonValue,
        mutationsJson: {
          nodesUpserted,
          edgesUpserted,
          confidenceUpdates,
        } as unknown as Prisma.InputJsonValue,
        messages: messages as unknown as Prisma.InputJsonValue,
      },
    });

    return {
      sessionId: session.id,
      finalResponse,
      toolCallsExecuted,
      graphMutations: { nodesUpserted, edgesUpserted, confidenceUpdates },
      totalTokensUsed: totalTokens,
    };
  }
}
