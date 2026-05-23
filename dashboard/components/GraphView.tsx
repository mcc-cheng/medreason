'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import dynamic from 'next/dynamic';
import type { CustomNode, InteractionEdge } from '@/lib/types';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => <GraphSkeleton />,
});

interface GraphData {
  nodes: CustomNode[];
  edges: (InteractionEdge & { provenanceLog?: unknown[] })[];
}

interface Props {
  highlightNodes?: Set<string>;
}

const NODE_COLORS: Record<string, string> = {
  BIOMOLECULE:        '#8b5cf6',
  CHEMICAL_CANDIDATE: '#34d399',
  METABOLIC_PATHWAY:  '#fbbf24',
};

const EDGE_COLORS: Record<string, string> = {
  INHIBITS:             '#f87171',
  ACTIVATES:            '#34d399',
  BINDS:                '#60a5fa',
  SIMILAR_TO:           '#a78bfa',
  TARGETS:              '#fb923c',
  ALLOSTERIC_MODULATOR: '#f472b6',
  ASSOCIATED_WITH:      '#94a3b8',
};

type EdgeGroup = 'direct' | 'similarity' | 'inferred';

const DIRECT_TYPES  = new Set(['INHIBITS', 'ACTIVATES', 'BINDS', 'ALLOSTERIC_MODULATOR']);
const SIM_TYPES     = new Set(['SIMILAR_TO']);
const INFER_TYPES   = new Set(['TARGETS', 'ASSOCIATED_WITH']);

export default function GraphView({ highlightNodes = new Set<string>() }: Props) {
  const [data, setData]                 = useState<GraphData | null>(null);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState<string | null>(null);
  const [propagating, setPropagating]   = useState(false);
  const [propagResult, setPropagResult] = useState<{ edgesInferred: number; edgesUpdated: number } | null>(null);
  const [activeGroups, setActiveGroups] = useState<Set<EdgeGroup>>(new Set(['direct', 'similarity']));
  const [selectedNode, setSelectedNode] = useState<Record<string, unknown> | null>(null);
  const [dimensions, setDimensions]     = useState({ width: 600, height: 500 });

  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef     = useRef<any>(null);

  // Track container size
  useEffect(() => {
    const obs = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setDimensions({ width: r.width, height: Math.max(350, r.height) });
    });
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  async function fetchGraph() {
    setLoading(true); setError(null);
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

  async function runPropagation() {
    setPropagating(true); setPropagResult(null);
    try {
      const res = await fetch('/api/propagate', { method: 'POST' });
      const d   = await res.json();
      if (!res.ok) throw new Error(d.error ?? 'Propagation failed');
      setPropagResult(d);
      await fetchGraph();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Propagation error');
    } finally {
      setPropagating(false);
    }
  }

  useEffect(() => { fetchGraph(); }, []);

  // Zoom to highlighted nodes when they change
  useEffect(() => {
    if (!graphRef.current || highlightNodes.size === 0) return;
    const t = setTimeout(() => {
      graphRef.current?.zoomToFit(600, 80, (n: unknown) => highlightNodes.has((n as { id: string }).id));
    }, 600);
    return () => clearTimeout(t);
  }, [highlightNodes]);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };

    const visible = new Set<string>();
    if (activeGroups.has('direct'))     DIRECT_TYPES.forEach(t => visible.add(t));
    if (activeGroups.has('similarity')) SIM_TYPES.forEach(t => visible.add(t));
    if (activeGroups.has('inferred'))   INFER_TYPES.forEach(t => visible.add(t));

    const links = data.edges
      .filter(e => visible.has(e.interactionType) && e.confidenceScore >= 0.3)
      .map(e => ({
        id:         e.id,
        source:     e.sourceId,
        target:     e.targetId,
        type:       e.interactionType,
        confidence: e.confidenceScore,
      }));

    const connectedIds = new Set<string>();
    links.forEach(l => { connectedIds.add(l.source); connectedIds.add(l.target); });

    const nodes = data.nodes
      .filter(n => connectedIds.has(n.id))
      .map(n => ({
        id:       n.id,
        name:     n.name,
        type:     n.type,
        val:      n.type === 'BIOMOLECULE' ? 14 : 6,
        metadata: n.metadata,
      }));

    return { nodes, links };
  }, [data, activeGroups]);

  const nodeCanvasObject = useCallback((
    node: Record<string, unknown>,
    ctx: CanvasRenderingContext2D,
    globalScale: number,
  ) => {
    const id          = node.id as string;
    const type        = node.type as string;
    const val         = node.val as number;
    const x           = node.x as number;
    const y           = node.y as number;
    const color       = NODE_COLORS[type] ?? '#6b7280';
    const isHL        = highlightNodes.has(id);
    const isSel       = (selectedNode as { id?: string } | null)?.id === id;
    const dimmed      = highlightNodes.size > 0 && !isHL && !isSel;
    const r           = Math.sqrt(val) * (isHL || isSel ? 1.5 : 1);

    // Glow halo
    if (isHL || isSel) {
      const grad = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
      grad.addColorStop(0, color + '60');
      grad.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(x, y, r * 3, 0, 2 * Math.PI);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // Node body
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = dimmed ? color + '28' : isHL ? color : color + 'bb';
    ctx.fill();

    if (isSel) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5 / globalScale;
      ctx.stroke();
    }

    // Label
    if (isHL || isSel || globalScale >= 2) {
      const fs = Math.max(7, 10 / globalScale);
      ctx.font = `${isHL ? 'bold ' : ''}${fs}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = dimmed ? '#ffffff22' : '#ffffffcc';
      ctx.fillText(id, x, y + r + 2 / globalScale);
    }
  }, [highlightNodes, selectedNode]);

  const getLinkColor = useCallback((link: Record<string, unknown>) => {
    const src  = (link.source as { id?: string } | string);
    const tgt  = (link.target as { id?: string } | string);
    const srcId = typeof src === 'string' ? src : src.id ?? '';
    const tgtId = typeof tgt === 'string' ? tgt : tgt.id ?? '';
    const isHL  = highlightNodes.has(srcId) || highlightNodes.has(tgtId);
    const dimmed = highlightNodes.size > 0 && !isHL;
    const base   = EDGE_COLORS[link.type as string] ?? '#6b7280';
    const conf   = link.confidence as number;
    const alpha  = dimmed ? '0a' : conf > 0.7 ? 'cc' : conf > 0.45 ? '77' : '44';
    return base + alpha;
  }, [highlightNodes]);

  const getLinkWidth = useCallback((link: Record<string, unknown>) => {
    const type = link.type as string;
    if (type === 'SIMILAR_TO') return 0.8;
    return Math.max(0.5, Math.min(3, (link.confidence as number) * 3));
  }, []);

  const getLinkArrowLen = useCallback((link: Record<string, unknown>) => {
    return link.type === 'SIMILAR_TO' ? 0 : 3;
  }, []);

  const getLinkCurvature = useCallback((link: Record<string, unknown>) => {
    return link.type === 'SIMILAR_TO' ? 0.25 : 0;
  }, []);

  const toggleGroup = (g: EdgeGroup) => {
    setActiveGroups(prev => {
      const next = new Set(prev);
      next.has(g) ? next.delete(g) : next.add(g);
      return next;
    });
  };

  if (loading) return <GraphSkeleton />;

  const GROUPS: [EdgeGroup, string, string][] = [
    ['direct',     'Drug→Protein', '#f87171'],
    ['similarity', 'Similarity',   '#a78bfa'],
    ['inferred',   'Inferred',     '#fb923c'],
  ];

  return (
    <div className="flex flex-col h-full glass-refract rounded-2xl overflow-hidden shadow-diffuse">
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-white/[0.04] flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Knowledge Graph</h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            {graphData.nodes.length} nodes · {graphData.links.length} edges
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={runPropagation}
            disabled={propagating}
            className="text-xs px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
            style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.2)', color: '#c4b5fd' }}
          >
            {propagating ? 'Propagating…' : 'Propagate'}
          </button>
          <button
            onClick={fetchGraph}
            className="text-xs text-zinc-500 hover:text-zinc-300 px-2.5 py-1 rounded-lg hover:bg-white/[0.04] transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Edge type toggles */}
      <div className="flex-shrink-0 px-4 py-2 border-b border-white/[0.04] flex items-center gap-2 flex-wrap">
        {GROUPS.map(([g, label, color]) => (
          <button
            key={g}
            onClick={() => toggleGroup(g)}
            className="text-[10px] px-2.5 py-0.5 rounded-full transition-all"
            style={{
              background: activeGroups.has(g) ? color + '22' : 'rgba(255,255,255,0.03)',
              border:     `1px solid ${activeGroups.has(g) ? color + '66' : 'rgba(255,255,255,0.06)'}`,
              color:      activeGroups.has(g) ? color : '#52525b',
            }}
          >
            {label}
          </button>
        ))}
        <span className="text-[10px] text-zinc-700 ml-auto hidden sm:block">scroll · drag · click</span>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="flex-1 relative min-h-0">
        {data && (
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="transparent"
            nodeCanvasObject={nodeCanvasObject}
            nodeCanvasObjectMode={() => 'replace'}
            linkColor={getLinkColor}
            linkWidth={getLinkWidth}
            linkDirectionalArrowLength={getLinkArrowLen}
            linkDirectionalArrowRelPos={1}
            linkCurvature={getLinkCurvature}
            onNodeClick={(node) => setSelectedNode(prev => (prev as { id?: string } | null)?.id === (node as { id: string }).id ? null : node as Record<string, unknown>)}
            cooldownTicks={120}
            d3AlphaDecay={0.025}
            d3VelocityDecay={0.35}
          />
        )}

        {/* Node detail card */}
        {selectedNode && (
          <div className="absolute bottom-3 left-3 w-64">
            <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
          </div>
        )}

        {/* Propagation toast */}
        {propagResult && (
          <div
            className="absolute top-3 left-1/2 -translate-x-1/2 text-xs text-violet-300 px-3 py-1.5 rounded-full whitespace-nowrap"
            style={{ background: 'rgba(139,92,246,0.18)', border: '1px solid rgba(139,92,246,0.3)' }}
          >
            ✦ {propagResult.edgesInferred} inferred · {propagResult.edgesUpdated} updated
          </div>
        )}

        {error && (
          <div
            className="absolute top-3 left-3 right-3 rounded-lg px-3 py-2 text-xs text-red-300"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
          >
            {error}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex-shrink-0 px-4 py-2 border-t border-white/[0.04] flex items-center gap-5">
        {([['#8b5cf6', 'Protein'], ['#34d399', 'Compound'], ['#fbbf24', 'Pathway']] as const).map(([c, l]) => (
          <div key={l} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: c, boxShadow: `0 0 4px ${c}88` }} />
            <span className="text-[10px] text-zinc-600">{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function NodeDetail({ node, onClose }: { node: Record<string, unknown>; onClose: () => void }) {
  const type      = String(node.type ?? '');
  const id        = String(node.id ?? '');
  const name      = String(node.name ?? '');
  const metadata  = node.metadata as Record<string, unknown> | undefined;
  const mw        = metadata?.molecularWeight as number | undefined;
  const extId     = metadata?.externalId as string | undefined;
  const danger    = metadata?.dangerLevel as string | undefined;
  const color     = NODE_COLORS[type] ?? '#6b7280';
  const typeLabel = type === 'BIOMOLECULE' ? 'Protein' : type === 'CHEMICAL_CANDIDATE' ? 'Compound' : 'Pathway';

  return (
    <div
      className="rounded-xl p-3 text-xs"
      style={{ background: 'rgba(9,9,11,0.92)', border: '1px solid rgba(255,255,255,0.1)', backdropFilter: 'blur(16px)' }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
          <span className="font-mono font-semibold text-white">{id}</span>
          <span className="text-zinc-500">{typeLabel}</span>
        </div>
        <button onClick={onClose} className="text-zinc-600 hover:text-zinc-300 ml-3 leading-none">✕</button>
      </div>
      <p className="text-zinc-400 mb-2 leading-snug">{name}</p>
      {mw !== undefined && (
        <p className="text-zinc-600 font-mono">
          MW:{' '}{mw >= 1000 ? `${(mw / 1000).toFixed(1)} kDa` : `${mw} Da`}
        </p>
      )}
      {extId && <p className="text-zinc-700 font-mono mt-0.5 truncate">{extId}</p>}
      {danger && (
        <span className="inline-block mt-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium text-red-400 bg-red-400/10">
          {danger}
        </span>
      )}
    </div>
  );
}

function GraphSkeleton() {
  return (
    <div className="glass-refract rounded-2xl h-full flex flex-col animate-pulse">
      <div className="px-4 py-3 border-b border-white/[0.04]">
        <div className="h-4 bg-white/[0.04] rounded w-32" />
        <div className="h-3 bg-white/[0.02] rounded w-20 mt-1" />
      </div>
      <div className="flex-1 flex items-center justify-center">
        <div className="text-xs text-zinc-700">Loading graph…</div>
      </div>
    </div>
  );
}
