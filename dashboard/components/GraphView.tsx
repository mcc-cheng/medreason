'use client';

import { useEffect, useState } from 'react';
import type { CustomNode, InteractionEdge } from '@/lib/types';

interface GraphData {
  nodes: CustomNode[];
  edges: (InteractionEdge & { provenanceLog?: unknown[] })[];
}

const TYPE_COLORS: Record<string, string> = {
  BIOMOLECULE: '#60a5fa',        // blue
  CHEMICAL_CANDIDATE: '#34d399', // green
  METABOLIC_PATHWAY: '#fbbf24',  // amber
};

const DANGER_BADGE: Record<string, string> = {
  LOW: 'text-green-400 bg-green-900',
  MEDIUM: 'text-amber-400 bg-amber-900',
  CHAOTIC: 'text-red-400 bg-red-900',
};

export default function GraphView() {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
  if (error) return <p className="text-red-400 text-sm">{error}</p>;
  if (!data) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Knowledge Graph</h2>
        <button
          onClick={fetchGraph}
          className="text-xs text-gray-400 hover:text-gray-200 transition-colors px-2 py-1 rounded border border-gray-700 hover:border-gray-500"
        >
          Refresh
        </button>
      </div>

      {/* Node list */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">
          Nodes ({data.nodes.length})
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {data.nodes.map((n) => (
            <div key={n.id} className="bg-gray-800 rounded-lg px-3 py-2 flex items-start gap-3">
              <span
                className="mt-0.5 w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: TYPE_COLORS[n.type] ?? '#9ca3af' }}
              />
              <div className="min-w-0">
                <p className="text-sm font-mono text-white truncate">{n.id}</p>
                <p className="text-xs text-gray-400 truncate">{n.name}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-gray-500">{n.type}</span>
                  {n.metadata.dangerLevel && (
                    <span
                      className={`text-xs px-1.5 py-0.5 rounded font-medium ${DANGER_BADGE[n.metadata.dangerLevel]}`}
                    >
                      {n.metadata.dangerLevel}
                    </span>
                  )}
                  {n.metadata.molecularWeight && (
                    <span className="text-xs text-gray-500">{n.metadata.molecularWeight} Da</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Edge list */}
      {data.edges.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">
            Edges ({data.edges.length})
          </p>
          <div className="space-y-2">
            {data.edges.map((e) => (
              <div key={e.id} className="bg-gray-800 rounded-lg px-3 py-2 text-xs font-mono">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-blue-300">{e.sourceId}</span>
                  <span className="text-gray-500">
                    —[{e.interactionType}]→
                  </span>
                  <span className="text-green-300">{e.targetId}</span>
                </div>
                <div className="flex items-center gap-4 mt-1">
                  <ConfidenceBar value={e.confidenceScore} />
                  <span className="text-gray-500">n={e.observationCount}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.edges.length === 0 && (
        <p className="text-sm text-gray-500 italic">
          No edges yet — run a simulation to discover interactions.
        </p>
      )}
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value > 0.7 ? '#34d399' : value > 0.4 ? '#fbbf24' : '#f87171';
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span style={{ color }}>{pct}%</span>
    </div>
  );
}

function GraphSkeleton() {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-3 animate-pulse">
      <div className="h-5 bg-gray-700 rounded w-40" />
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="h-14 bg-gray-800 rounded-lg" />
      ))}
    </div>
  );
}
