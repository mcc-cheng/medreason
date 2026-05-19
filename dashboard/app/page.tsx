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
    setGraphKey((k) => k + 1);
  }

  return (
    <main className="min-h-screen" style={{ background: '#09090B' }}>
      {/* Top bar */}
      <header className="border-b border-white/[0.04] px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="4" cy="4" r="2" fill="white" fillOpacity="0.9" />
                <circle cx="10" cy="4" r="2" fill="white" fillOpacity="0.6" />
                <circle cx="7" cy="10" r="2" fill="white" fillOpacity="0.75" />
                <line x1="4" y1="4" x2="10" y2="4" stroke="white" strokeOpacity="0.4" strokeWidth="1" />
                <line x1="4" y1="4" x2="7" y2="10" stroke="white" strokeOpacity="0.4" strokeWidth="1" />
                <line x1="10" y1="4" x2="7" y2="10" stroke="white" strokeOpacity="0.4" strokeWidth="1" />
              </svg>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white tracking-tight">Drug Discovery Canvas</h1>
              <p className="text-xs text-zinc-500">In-silico compound–protein simulation</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-600 font-mono">llama3.1 · local</span>
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(52,211,153,0.6)]" />
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-5">
            <SimulationPanel onResult={handleResult} />
            {result && <ResultPanel result={result} />}
          </div>
          <div>
            <GraphView key={graphKey} />
          </div>
        </div>
      </div>
    </main>
  );
}
