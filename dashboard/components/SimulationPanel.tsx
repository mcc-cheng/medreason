'use client';

import { useState } from 'react';
import type { AgentRunOutput } from '@/lib/types';

const KNOWN_NODES = [
  { id: 'GLOW-SQUID-9', label: 'GLOW-SQUID-9 (protein)' },
  { id: 'HONEY-BADGER-X', label: 'HONEY-BADGER-X (protein)' },
  { id: 'CHIPOTLE-MAYO-42', label: 'CHIPOTLE-MAYO-42 (compound)' },
  { id: 'CAFFEINE-OVERDOSE-99', label: 'CAFFEINE-OVERDOSE-99 (compound)' },
];

interface Props {
  onResult: (result: AgentRunOutput) => void;
}

export default function SimulationPanel({ onResult }: Props) {
  const [prompt, setPrompt] = useState('');
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleNode(id: string) {
    setSelectedNodes((prev) =>
      prev.includes(id) ? prev.filter((n) => n !== id) : [...prev, id]
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || selectedNodes.length === 0) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userPrompt: prompt,
          focalNodeIds: selectedNodes,
          subGraphDepth: 2,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? 'Simulation failed');
      onResult(data as AgentRunOutput);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-5">
      <h2 className="text-lg font-semibold text-white">Run Simulation</h2>

      {/* Node selector */}
      <div>
        <p className="text-xs text-gray-400 mb-2 uppercase tracking-wider">Focal nodes</p>
        <div className="flex flex-wrap gap-2">
          {KNOWN_NODES.map((n) => (
            <button
              key={n.id}
              onClick={() => toggleNode(n.id)}
              className={`px-3 py-1.5 rounded-lg text-sm font-mono transition-colors ${
                selectedNodes.includes(n.id)
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {n.label}
            </button>
          ))}
        </div>
      </div>

      {/* Prompt input */}
      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          className="w-full bg-gray-800 border border-gray-600 rounded-lg p-3 text-gray-100 text-sm placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500 transition-colors"
          rows={3}
          placeholder='e.g. "Simulate CHIPOTLE-MAYO-42 binding to GLOW-SQUID-9 at 10µM and assess its safety profile."'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <button
          type="submit"
          disabled={loading || !prompt.trim() || selectedNodes.length === 0}
          className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors"
        >
          {loading ? 'Running agent…' : 'Run simulation'}
        </button>
      </form>

      {error && (
        <p className="text-red-400 text-sm bg-red-950 border border-red-800 rounded-lg px-3 py-2">
          {error}
        </p>
      )}
    </div>
  );
}
