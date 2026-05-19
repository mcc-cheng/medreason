import { NextRequest, NextResponse } from 'next/server';
import { MemoryManager } from '@/lib/memoryManager';

export const runtime = 'nodejs';

// GET /api/graph?nodeIds=A,B&depth=2
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const rawIds = searchParams.get('nodeIds');
  const depth = parseInt(searchParams.get('depth') ?? '2', 10);

  const memory = new MemoryManager();

  try {
    if (!rawIds) {
      // Return full graph
      const graph = await memory.loadFullGraph();
      return NextResponse.json({
        nodes: [...graph.nodes.values()],
        edges: [...graph.edges.values()].map((e) => ({
          ...e,
          provenanceLog: e.provenanceLog.slice(-3), // last 3 entries only
        })),
      });
    }

    const nodeIds = rawIds.split(',').map((id) => id.trim()).filter(Boolean);
    const subGraph = await memory.extractSubGraph(nodeIds, depth);
    return NextResponse.json({
      focalNodeIds: subGraph.focalNodeIds,
      depth: subGraph.depth,
      nodes: subGraph.nodes,
      edges: subGraph.edges.map((e) => ({
        ...e,
        provenanceLog: e.provenanceLog.slice(-3),
      })),
    });
  } catch (err) {
    console.error('[/api/graph]', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
