import { NextResponse } from 'next/server';
import { propagateConfidence } from '@/lib/propagation';

export const runtime = 'nodejs';

export async function POST() {
  try {
    const summary = await propagateConfidence();
    return NextResponse.json(summary);
  } catch (err) {
    console.error('[/api/propagate]', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Propagation failed' },
      { status: 500 }
    );
  }
}
