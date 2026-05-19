'use client';

import { useEffect, useState } from 'react';
import type { CustomNode, InteractionEdge } from '@/lib/types';

interface GraphData {
  nodes: CustomNode[];
  edges: (InteractionEdge & { provenanceLog?: unknown[] })[];
}

const TYPE_STYLE: Record<string, { dot: string; label: string }> = {
  BIOMOLECULE:        { dot: '#8b5cf6', label: 'Protein' },
  CHEMICAL_CANDIDATE: { dot: '#34d399', label: 'Compound' },
  METABOLIC_PATHWAY:  { dot: '#fbbf24', label: 'Pathway' },
};

const DANGER_STYLE: Record<string, string> = {
  LOW:     'text-emerald-400 bg-emerald-400/10',
  MEDIUM:  'text-amber-400   bg-amber-400/10',
  CHAOTIC: 'text-red-400     bg-red-400/10',
};

export default function GraphView() {
  const [data, setData]       = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  async function fetchGraph() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/graph');
      if (!res.ok) throw new Error('Failed to fetch graph');
      setData(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchGraph(); }, []);

  if (loading) return <GraphSkeleton />;

  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse">
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/[0.04] flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Knowledge Graph</h2>
          {data && (
            <p className="text-xs text-zinc-500 mt-0.5">
              {data.nodes.length} nodes · {data.edges.length} edges
            </p>
          )}
        </div>
        <button
          onClick={fetchGraph}
          className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors px-2.5 py-1 rounded-lg hover:bg-white/[0.04]"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="m-5 rounded-xl px-4 py-3 text-xs text-red-300"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.15)' }}>
          {error}
        </div>
      )}

      {data && (
        <div className="p-5 space-y-5">
          {/* Nodes */}
          <div>
            <p className="text-xs text-zinc-600 uppercase tracking-widest mb-2.5">Nodes</p>
            <div className="space-y-1.5">
              {data.nodes.map((n) => {
                const ts = TYPE_STYLE[n.type] ?? { dot: '#6b7280', label: n.type };
                const ds = n.metadata.dangerLevel ? DANGER_STYLE[n.metadata.dangerLevel] : null;
                return (
                  <div
                    key={n.id}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors hover:bg-white/[0.02]"
                    style={{ border: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: ts.dot, boxShadow: `0 0 6px ${ts.dot}66` }}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-mono font-medium text-white">{n.id}</span>
                        <span className="text-[10px] text-zinc-600">{ts.label}</span>
                        {ds && (
                          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${ds}`}>
                            {n.metadata.dangerLevel}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-zinc-500 truncate mt-0.5">{n.name}</p>
                    </div>
                    {n.metadata.molecularWeight && (
                      <span className="text-[10px] text-zinc-700 font-mono flex-shrink-0">
                        {n.metadata.molecularWeight >= 1000
                          ? `${(n.metadata.molecularWeight / 1000).toFixed(0)} kDa`
                          : `${n.metadata.molecularWeight} Da`}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Edges */}
          {data.edges.length > 0 && (
            <div>
              <p className="text-xs text-zinc-600 uppercase tracking-widest mb-2.5">Interactions</p>
              <div className="space-y-2">
                {data.edges.map((e) => (
                  <EdgeRow key={e.id} edge={e} />
                ))}
              </div>
            </div>
          )}

          {data.edges.length === 0 && (
            <p className="text-xs text-zinc-600 italic text-center py-4">
              No interactions yet — run a simulation to discover edges.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function EdgeRow({ edge }: { edge: InteractionEdge & { provenanceLog?: unknown[] } }) {
  const pct = Math.round(edge.confidenceScore * 100);
  const barColor = edge.confidenceScore > 0.7
    ? '#34d399'
    : edge.confidenceScore > 0.4
    ? '#fbbf24'
    : '#f87171';

  return (
    <div
      className="px-3 py-3 rounded-xl space-y-2"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
    >
      <div className="flex items-center gap-2 text-xs font-mono flex-wrap">
        <span className="text-violet-300">{edge.sourceId}</span>
        <span className="text-zinc-700 text-[10px] px-1.5 py-0.5 rounded"
          style={{ background: 'rgba(255,255,255,0.04)' }}>
          {edge.interactionType.toLowerCase()}
        </span>
        <span className="text-emerald-300">{edge.targetId}</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: barColor, boxShadow: `0 0 6px ${barColor}66` }}
          />
        </div>
        <span className="text-xs font-mono font-medium flex-shrink-0" style={{ color: barColor }}>
          {pct}%
        </span>
        <span className="text-[10px] text-zinc-600 flex-shrink-0">n={edge.observationCount}</span>
      </div>
    </div>
  );
}

function GraphSkeleton() {
  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse p-5 space-y-3 animate-pulse">
      <div className="h-4 bg-white/[0.04] rounded w-32" />
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div key={i} className="h-12 bg-white/[0.02] rounded-xl" />
      ))}
    </div>
  );
}
