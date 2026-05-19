import { NextRequest, NextResponse } from 'next/server';
import { AgentEngine } from '@/lib/agentEngine';
import type { AgentRunInput } from '@/lib/types';

export const runtime = 'nodejs'; // Prisma requires Node.js runtime

export async function POST(req: NextRequest) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: 'GEMINI_API_KEY not set' }, { status: 500 });
  }

  let body: AgentRunInput;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (!body.userPrompt || typeof body.userPrompt !== 'string') {
    return NextResponse.json({ error: 'userPrompt is required' }, { status: 400 });
  }

  if (!Array.isArray(body.focalNodeIds)) {
    return NextResponse.json({ error: 'focalNodeIds must be an array' }, { status: 400 });
  }

  try {
    const engine = new AgentEngine(apiKey);
    const output = await engine.run({
      userPrompt: body.userPrompt,
      focalNodeIds: body.focalNodeIds,
      subGraphDepth: body.subGraphDepth ?? 2,
    });
    return NextResponse.json(output);
  } catch (err) {
    console.error('[/api/simulate]', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
