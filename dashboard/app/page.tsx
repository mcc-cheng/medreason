'use client';

import { useState } from 'react';
import SimulationPanel from '@/components/SimulationPanel';
import ResultPanel from '@/components/ResultPanel';
import GraphView from '@/components/GraphView';
import type { AgentRunOutput } from '@/lib/types';

export default function Home() {
  const [result, setResult] = useState<AgentRunOutput | null>(null);
  const [graphKey, setGraphKey] = useState(0);

  function handleResult(r: AgentRunOutput) {
    setResult(r);
    setGraphKey((k) => k + 1); // force GraphView to re-fetch after a simulation
  }

  return (
    <main className="max-w-6xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Drug Discovery Canvas</h1>
        <p className="text-sm text-gray-400 mt-1">
          Simulate compound–protein knockouts. Each run updates the knowledge graph&apos;s
          Bayesian confidence scores via the Gemini agent.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <SimulationPanel onResult={handleResult} />
          {result && <ResultPanel result={result} />}
        </div>
        <div>
          <GraphView key={graphKey} />
        </div>
      </div>
    </main>
  );
}
