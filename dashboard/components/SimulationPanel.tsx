'use client';

import { useState } from 'react';
import type { AgentRunOutput } from '@/lib/types';

const NODES = [
  { id: 'BCR-ABL',    label: 'BCR-ABL',    sub: 'Tyrosine kinase',        kind: 'protein'   },
  { id: 'EGFR',       label: 'EGFR',        sub: 'Growth factor receptor', kind: 'protein'   },
  { id: 'COX-2',      label: 'COX-2',       sub: 'Cyclooxygenase enzyme',  kind: 'protein'   },
  { id: 'IMATINIB',   label: 'Imatinib',    sub: 'Gleevec · 493.6 Da',     kind: 'compound'  },
  { id: 'GEFITINIB',  label: 'Gefitinib',   sub: 'Iressa · 446.9 Da',      kind: 'compound'  },
  { id: 'ASPIRIN',    label: 'Aspirin',     sub: 'ASA · 180.2 Da',         kind: 'compound'  },
];

const EXAMPLES = [
  {
    prompt: 'Simulate Imatinib binding to BCR-ABL at 0.1 µM. Predict efficacy and toxicity, explain what the Bayesian confidence score means, and summarize what this simulation adds to our knowledge.',
    nodes: ['IMATINIB', 'BCR-ABL'],
  },
  {
    prompt: 'Analyze Gefitinib selectivity at EGFR at 1 µM. Assess its safety profile and predict on-target vs off-target effects.',
    nodes: ['GEFITINIB', 'EGFR'],
  },
  {
    prompt: 'Model COX-2 inhibition by Aspirin at 5 µM. Explain the irreversible acetylation mechanism and quantify the expected confidence shift.',
    nodes: ['ASPIRIN', 'COX-2'],
  },
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

  function applyExample(ex: typeof EXAMPLES[0]) {
    setPrompt(ex.prompt);
    setSelectedNodes(ex.nodes);
    setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;
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

  const proteins  = NODES.filter((n) => n.kind === 'protein');
  const compounds = NODES.filter((n) => n.kind === 'compound');

  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse">
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/[0.04]">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Run Simulation</h2>
            <p className="text-xs text-zinc-500 mt-0.5">Select targets, then describe the experiment</p>
          </div>
          {loading && (
            <div className="flex items-center gap-1.5 text-xs text-violet-400">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
              Running agent
            </div>
          )}
        </div>
      </div>

      <div className="p-5 space-y-5">
        {/* Node selector */}
        <div className="space-y-3">
          <NodeGroup
            label="Proteins"
            nodes={proteins}
            selected={selectedNodes}
            onToggle={toggleNode}
            colorClass="violet"
          />
          <NodeGroup
            label="Compounds"
            nodes={compounds}
            selected={selectedNodes}
            onToggle={toggleNode}
            colorClass="emerald"
          />
        </div>

        {/* Example prompts */}
        <div>
          <p className="text-xs text-zinc-600 uppercase tracking-widest mb-2">Example experiments</p>
          <div className="space-y-1.5">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                type="button"
                onClick={() => applyExample(ex)}
                className="w-full text-left px-3 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.05] border border-white/[0.04] hover:border-white/[0.08] transition-all group"
              >
                <p className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors leading-relaxed line-clamp-2">
                  {ex.prompt}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* Prompt input */}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="relative">
            <textarea
              className="w-full rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 resize-none focus:outline-none transition-all"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
              onFocus={(e) => {
                e.currentTarget.style.border = '1px solid rgba(139,92,246,0.4)';
                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(139,92,246,0.08)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.border = '1px solid rgba(255,255,255,0.06)';
                e.currentTarget.style.boxShadow = 'none';
              }}
              rows={4}
              placeholder="Describe the simulation experiment…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>

          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            className="relative w-full py-2.5 rounded-xl text-sm font-medium text-white transition-all overflow-hidden disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: loading || !prompt.trim()
                ? 'rgba(255,255,255,0.05)'
                : 'linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%)',
              boxShadow: !loading && prompt.trim()
                ? '0 0 20px rgba(109,40,217,0.3), 0 1px 0 rgba(255,255,255,0.1) inset'
                : 'none',
            }}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Running agent…
              </span>
            ) : (
              'Run simulation'
            )}
          </button>
        </form>

        {error && (
          <div className="rounded-xl px-4 py-3 text-xs text-red-300 leading-relaxed"
            style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.15)' }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

function NodeGroup({
  label, nodes, selected, onToggle, colorClass,
}: {
  label: string;
  nodes: typeof NODES;
  selected: string[];
  onToggle: (id: string) => void;
  colorClass: 'violet' | 'emerald';
}) {
  const active = colorClass === 'violet'
    ? { bg: 'rgba(139,92,246,0.15)', border: 'rgba(139,92,246,0.4)', dot: '#8b5cf6' }
    : { bg: 'rgba(52,211,153,0.1)',  border: 'rgba(52,211,153,0.35)', dot: '#34d399' };

  const inactive = { bg: 'rgba(255,255,255,0.02)', border: 'rgba(255,255,255,0.05)' };

  return (
    <div>
      <p className="text-xs text-zinc-600 uppercase tracking-widest mb-1.5">{label}</p>
      <div className="grid grid-cols-3 gap-1.5">
        {nodes.map((n) => {
          const on = selected.includes(n.id);
          return (
            <button
              key={n.id}
              type="button"
              onClick={() => onToggle(n.id)}
              className="relative text-left px-3 py-2 rounded-lg transition-all"
              style={{
                background: on ? active.bg : inactive.bg,
                border: `1px solid ${on ? active.border : inactive.border}`,
                boxShadow: on ? `0 0 12px ${active.dot}22` : 'none',
              }}
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: on ? active.dot : '#3f3f46' }}
                />
                <span className={`text-xs font-mono font-medium ${on ? 'text-white' : 'text-zinc-400'}`}>
                  {n.label}
                </span>
              </div>
              <p className="text-[10px] text-zinc-600 leading-tight pl-3">{n.sub}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
