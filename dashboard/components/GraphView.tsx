'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import dynamic from 'next/dynamic';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error — d3-force-3d ships JS without types but is bundled by react-force-graph-2d.
import { forceCollide } from 'd3-force-3d';
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

// Three-color palette: gold (compounds, positive action), ruby (pathways, inhibition),
// darker sapphire (proteins, structural / binding). Lighter sapphire for accents.
const NODE_COLORS: Record<string, string> = {
  BIOMOLECULE:        '#2C4A8E', // darker sapphire — proteins
  CHEMICAL_CANDIDATE: '#2D5A3F', // dark forest green — compounds
  METABOLIC_PATHWAY:  '#D03D5F', // ruby — pathways
};

// Polished-gem gradient stops for each node type. `hi` is the bright highlight ramp,
// `mid` is the saturated base, `lo` is the deep rim shadow. `spec` is reserved for
// selection ring color.
const NODE_GRADIENT: Record<string, { spec: string; hi: string; mid: string; lo: string }> = {
  BIOMOLECULE:        { spec: '#E8EEFF', hi: '#7DA0E8', mid: '#2C4A8E', lo: '#0F1F4A' }, // sapphire
  CHEMICAL_CANDIDATE: { spec: '#E5F4E8', hi: '#6FB783', mid: '#2D5A3F', lo: '#0E2A18' }, // forest green
  METABOLIC_PATHWAY:  { spec: '#FFE6EC', hi: '#F285A0', mid: '#D03D5F', lo: '#5A0918' }, // ruby
};

const EDGE_COLORS: Record<string, string> = {
  INHIBITS:             '#5B2D78', // dark purple — block
  ACTIVATES:            '#E5B547', // gold — promote
  BINDS:                '#4D6FC4', // lighter sapphire — neutral bond
  SIMILAR_TO:           '#4D6FC4', // lighter sapphire — homology
  TARGETS:              '#E5B547', // gold — drug action
  ALLOSTERIC_MODULATOR: '#5B2D78', // dark purple — modulation
  ASSOCIATED_WITH:      '#6B7B95', // muted slate — weak association
};

// Bright "shine" stroke colors layered on top of each edge so the wire glints like
// a polished filament of the same color family.
const EDGE_SHINE: Record<string, string> = {
  INHIBITS:             '#A07AC0', // bright purple
  ACTIVATES:            '#F7E388', // bright gold
  BINDS:                '#A8C0F4', // bright sapphire
  SIMILAR_TO:           '#A8C0F4',
  TARGETS:              '#F7E388',
  ALLOSTERIC_MODULATOR: '#A07AC0',
  ASSOCIATED_WITH:      '#A3AFC5',
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
  const [searchQuery, setSearchQuery]     = useState('');
  const [searchFocused, setSearchFocused] = useState(false);
  const [searchPickId, setSearchPickId]   = useState<string | null>(null);
  const [bbox, setBbox]                   = useState<{ xMin: number; xMax: number; yMin: number; yMax: number } | null>(null);

  const containerRef    = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef        = useRef<any>(null);
  const isClampingRef   = useRef(false);

  // graphReady tracks when the dynamically-imported ForceGraph2D instance is actually
  // mounted. Without this, the force-setup useEffect runs once on data load but bails
  // because graphRef.current is still null (dynamic import not resolved yet) — and
  // then never re-runs since no deps change.
  const [graphReady, setGraphReady] = useState(false);
  const setGraphRef = useCallback((instance: unknown) => {
    graphRef.current = instance;
    setGraphReady(!!instance);
  }, []);

  // Track container size — depends on `loading` so the observer re-attaches
  // once the real container mounts (the skeleton early-returns before it).
  useEffect(() => {
    if (loading) return;
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setDimensions({ width: r.width, height: Math.max(350, r.height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [loading]);

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

  // ESC closes any open detail/selection panels (search dropdown handled inline on input).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedNode(null);
        setSearchPickId(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

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
        val:      n.type === 'BIOMOLECULE' ? 5 : 2.5,
        metadata: n.metadata,
      }));

    return { nodes, links };
  }, [data, activeGroups]);

  // Physically spread layout — very strong repulsion, long links, big collision radius
  // so node bodies have clear airspace around them and labels stay readable.
  // SIMILAR_TO links are softened so protein-protein clusters don't fuse into one blob.
  // The fallback timer applies the initial zoom + bbox snapshot in case onEngineStop
  // doesn't fire (which happens in some react-force-graph versions after a manual reheat).
  useEffect(() => {
    if (!graphRef.current || !data) return;
    const charge = graphRef.current.d3Force('charge');
    const link   = graphRef.current.d3Force('link');
    if (charge) charge.strength(-8000).distanceMax(2000);
    if (link) {
      link
        .distance((l: { type?: string; confidence?: number }) => {
          const conf = typeof l.confidence === 'number' ? l.confidence : 0.5;
          if (l.type === 'SIMILAR_TO') return 150 + (1 - conf) * 250;
          return 200 + (1 - conf) * 400;
        })
        .strength((l: { type?: string }) => (l.type === 'SIMILAR_TO' ? 0.03 : 0.08));
    }
    graphRef.current.d3Force('collide', forceCollide(90));
    graphRef.current.d3ReheatSimulation?.();

    // Fallback: after ~2.2s the simulation should be settled. Snapshot positions and
    // apply initial zoom (smallest scale where node labels are readable, = 3).
    const t = setTimeout(() => {
      if (!graphRef.current) return;
      const xs: number[] = [];
      const ys: number[] = [];
      for (const n of graphData.nodes) {
        const nx = (n as { x?: number }).x;
        const ny = (n as { y?: number }).y;
        if (Number.isFinite(nx) && Number.isFinite(ny)) {
          xs.push(nx as number);
          ys.push(ny as number);
        }
      }
      if (xs.length === 0) return;
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      setBbox({ xMin, xMax, yMin, yMax });
      const bw = (xMax - xMin) + 2 * PAN_PAD;
      const bh = (yMax - yMin) + 2 * PAN_PAD;
      if (bw > 0 && bh > 0 && dimensions.width > 0 && dimensions.height > 0) {
        const fitZoom = Math.min(dimensions.width / bw, dimensions.height / bh);
        graphRef.current.centerAt?.((xMin + xMax) / 2, (yMin + yMax) / 2, 400);
        graphRef.current.zoom?.(Math.max(fitZoom, 6), 400);
      }
    }, 2200);
    return () => clearTimeout(t);
    // graphData / dimensions intentionally NOT in deps — we only run this on
    // data/activeGroups change. The timer captures whatever values were current then.
    // graphReady IS needed so this re-runs when the dynamic ForceGraph2D mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, activeGroups, graphReady]);

  // Zoom to highlighted nodes when they change
  useEffect(() => {
    if (!graphRef.current || highlightNodes.size === 0) return;
    const t = setTimeout(() => {
      graphRef.current?.zoomToFit(600, 80, (n: unknown) => highlightNodes.has((n as { id: string }).id));
    }, 600);
    return () => clearTimeout(t);
  }, [highlightNodes]);

  // Two highlight sets:
  //   edgeHighlights — only the picked / focal nodes. An edge lights up only when
  //     one of its endpoints is in this set. Neighbors are NOT here, so a neighbor's
  //     OTHER edges (to unrelated nodes) stay dimmed.
  //   allHighlights  — picked + focal + immediate neighbors. Used for node rendering
  //     (color, label visibility) so neighbors appear at full brightness too.
  const searchNeighbors = useMemo(() => {
    const s = new Set<string>();
    if (!searchPickId) return s;
    for (const l of graphData.links) {
      const src = l.source as string | { id?: string };
      const tgt = l.target as string | { id?: string };
      const srcId = typeof src === 'string' ? src : src?.id ?? '';
      const tgtId = typeof tgt === 'string' ? tgt : tgt?.id ?? '';
      if (srcId === searchPickId) s.add(tgtId);
      if (tgtId === searchPickId) s.add(srcId);
    }
    return s;
  }, [searchPickId, graphData.links]);

  const edgeHighlights = useMemo(() => {
    const s = new Set(highlightNodes);
    if (searchPickId) s.add(searchPickId);
    return s;
  }, [highlightNodes, searchPickId]);

  const allHighlights = useMemo(() => {
    const s = new Set(edgeHighlights);
    searchNeighbors.forEach(id => s.add(id));
    return s;
  }, [edgeHighlights, searchNeighbors]);

  // Prefix-matched search options (case-insensitive). Matches on either `name` or `id`,
  // both required to START WITH the typed text — not substring contains.
  const searchOptions = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return [];
    return graphData.nodes
      .filter(n => {
        const name = String((n as { name?: string }).name ?? '').toLowerCase();
        const id   = String(n.id).toLowerCase();
        return name.startsWith(q) || id.startsWith(q);
      })
      .slice(0, 12)
      .map(n => ({
        id:   n.id as string,
        name: String((n as { name?: string }).name ?? n.id),
        type: n.type as string,
      }));
  }, [graphData.nodes, searchQuery]);

  function pickSearchOption(nodeId: string) {
    setSearchPickId(nodeId);
    setSearchFocused(false);
    const node = graphData.nodes.find(n => n.id === nodeId) as { x?: number; y?: number } | undefined;
    if (node && Number.isFinite(node.x) && Number.isFinite(node.y) && graphRef.current) {
      graphRef.current.centerAt(node.x as number, node.y as number, 600);
      graphRef.current.zoom(2.5, 600);
    }
  }

  // After the simulation settles, snapshot the bounding box and fit the view to it exactly.
  // The fit zoom is the same formula as `dynamicMinZoom` so "initial view" === "fully zoomed out".
  const handleEngineStop = useCallback(() => {
    if (!graphRef.current || graphData.nodes.length === 0) return;
    const xs: number[] = [];
    const ys: number[] = [];
    for (const n of graphData.nodes) {
      const nx = (n as { x?: number }).x;
      const ny = (n as { y?: number }).y;
      if (Number.isFinite(nx) && Number.isFinite(ny)) {
        xs.push(nx as number);
        ys.push(ny as number);
      }
    }
    if (xs.length === 0) return;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    setBbox({ xMin, xMax, yMin, yMax });

    const bw = (xMax - xMin) + 2 * PAN_PAD;
    const bh = (yMax - yMin) + 2 * PAN_PAD;
    if (bw <= 0 || bh <= 0 || dimensions.width <= 0 || dimensions.height <= 0) {
      graphRef.current.zoomToFit(400, 40);
      return;
    }
    const fitZoom = Math.min(dimensions.width / bw, dimensions.height / bh);
    graphRef.current.centerAt((xMin + xMax) / 2, (yMin + yMax) / 2, 400);
    // Start at the smallest zoom where node labels are still legible. Text shows when
    // globalScale ≥ 3 (see nodeCanvasObject), so 3 is the floor — but if fitZoom is
    // already higher than that we use it instead (graph already small enough).
    graphRef.current.zoom(Math.max(fitZoom, 6), 400);
  }, [graphData, dimensions]);

  // Dynamic minZoom: at this zoom level, the (padded) bbox exactly fills the viewport.
  // Below this we'd be showing empty space outside the graph, which we don't want.
  // Pre-bbox fallback is intentionally conservative (0.7) so the user can't zoom into a
  // void during the brief moment between mount and first onEngineStop.
  const PAN_PAD = 120;
  const dynamicMinZoom = useMemo(() => {
    if (!bbox || dimensions.width === 0 || dimensions.height === 0) return 0.7;
    const bw = (bbox.xMax - bbox.xMin) + 2 * PAN_PAD;
    const bh = (bbox.yMax - bbox.yMin) + 2 * PAN_PAD;
    if (bw === 0 || bh === 0) return 0.7;
    return Math.min(dimensions.width / bw, dimensions.height / bh);
  }, [bbox, dimensions]);

  // Wheel inversion — scroll UP zooms IN, scroll DOWN zooms OUT.
  // Strategy: don't fight react-force-graph's d3-zoom (which uses its own internal
  // event handling that ignores stopPropagation). Instead, intercept wheel events in
  // capture phase, suppress the original, and dispatch a synthetic wheel with -deltaY
  // to the canvas. d3-zoom then handles the inverted event normally.
  useEffect(() => {
    if (loading) return;
    let synthetic = false;  // re-entry guard — skip the event we just dispatched

    const onWheel = (e: WheelEvent) => {
      if (synthetic) return;
      const target = e.target as Element | null;
      const el = containerRef.current;
      if (!el || !target || !el.contains(target)) return;

      e.preventDefault();
      e.stopImmediatePropagation();

      // Find the actual canvas (d3-zoom listens on it directly) and dispatch there.
      const canvas = el.querySelector('canvas') ?? target;

      synthetic = true;
      try {
        const synth = new WheelEvent('wheel', {
          bubbles: true,
          cancelable: true,
          view: window,
          deltaX: e.deltaX,
          deltaY: -e.deltaY,  // ← the actual inversion
          deltaZ: e.deltaZ,
          deltaMode: e.deltaMode,
          clientX: e.clientX,
          clientY: e.clientY,
          screenX: e.screenX,
          screenY: e.screenY,
          ctrlKey: e.ctrlKey,
          metaKey: e.metaKey,
          altKey: e.altKey,
          shiftKey: e.shiftKey,
          button: e.button,
          buttons: e.buttons,
        });
        canvas.dispatchEvent(synth);
      } finally {
        synthetic = false;
      }
    };

    window.addEventListener('wheel', onWheel, { passive: false, capture: true });
    return () => window.removeEventListener('wheel', onWheel, true);
  }, [loading]);

  // Clamp pan so the viewport edges can never cross the (padded) bbox edges.
  // When viewport is larger than the bbox in a given axis, center is locked to the bbox center on that axis.
  const handleZoom = useCallback(() => {
    if (isClampingRef.current || !bbox || !graphRef.current) return;
    const c = graphRef.current.centerAt();
    const k = graphRef.current.zoom();
    if (!c || !k) return;

    const bxMin = bbox.xMin - PAN_PAD, bxMax = bbox.xMax + PAN_PAD;
    const byMin = bbox.yMin - PAN_PAD, byMax = bbox.yMax + PAN_PAD;
    const bcx   = (bxMin + bxMax) / 2,  bcy   = (byMin + byMax) / 2;
    const bw    = bxMax - bxMin,        bh    = byMax - byMin;
    const vw    = dimensions.width  / k;
    const vh    = dimensions.height / k;

    let cx: number, cy: number;
    if (vw >= bw) cx = bcx;
    else { const slack = (bw - vw) / 2; cx = Math.max(bcx - slack, Math.min(bcx + slack, c.x)); }
    if (vh >= bh) cy = bcy;
    else { const slack = (bh - vh) / 2; cy = Math.max(bcy - slack, Math.min(bcy + slack, c.y)); }

    if (Math.abs(cx - c.x) > 0.5 || Math.abs(cy - c.y) > 0.5) {
      isClampingRef.current = true;
      graphRef.current.centerAt(cx, cy, 0);
      setTimeout(() => { isClampingRef.current = false; }, 0);
    }
  }, [bbox, dimensions]);

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
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const stops       = NODE_GRADIENT[type] ?? { spec: '#ffffff', hi: '#cccccc', mid: '#6b7280', lo: '#1a1a1a' };
    const isHL        = allHighlights.has(id);
    const isSel       = (selectedNode as { id?: string } | null)?.id === id;
    const dimmed      = allHighlights.size > 0 && !isHL && !isSel;
    const r           = Math.sqrt(val) * (isHL || isSel ? 1.5 : 1);

    // Outer glow halo (highlighted / selected only) — bloom of the metal's mid hue.
    if (isHL || isSel) {
      const halo = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
      halo.addColorStop(0, stops.mid + '70');
      halo.addColorStop(1, stops.mid + '00');
      ctx.beginPath();
      ctx.arc(x, y, r * 3, 0, 2 * Math.PI);
      ctx.fillStyle = halo;
      ctx.fill();
    }

    // Metallic gem body — radial gradient with off-center origin to simulate a
    // top-left light source. Stops: bright highlight → main color → deep rim shadow.
    const lightOffset = r * 0.4;
    const gradOrigin  = { x: x - lightOffset, y: y - lightOffset };
    const bodyGrad    = ctx.createRadialGradient(gradOrigin.x, gradOrigin.y, 0, x, y, r * 1.05);
    bodyGrad.addColorStop(0,    stops.hi);
    bodyGrad.addColorStop(0.35, stops.mid);
    bodyGrad.addColorStop(1,    stops.lo);

    if (dimmed) ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = bodyGrad;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Rim — thin dark outline anchors the gem against the black background.
    if (!dimmed) {
      ctx.strokeStyle = stops.lo;
      ctx.lineWidth = 0.6 / globalScale;
      ctx.stroke();
    }

    if (isSel) {
      ctx.strokeStyle = stops.spec;
      ctx.lineWidth = 1.5 / globalScale;
      ctx.stroke();
    }

    // Label — only show when zoomed in enough or when node is highlighted/selected.
    if (isHL || isSel || globalScale >= 3) {
      const isPicked = id === searchPickId;
      const fs = Math.max(4, 6 / globalScale) * (isPicked ? 1.3 : 1);
      ctx.font = `${isHL || isPicked ? 'bold ' : ''}${fs}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const labelY = y + r + 1.5 / globalScale;
      if (isPicked) {
        // Subtle glow — single pass, small blur — just enough to lift the label without
        // smearing it. canvas shadowBlur is in canvas pixels so it stays visually
        // consistent at any zoom level.
        ctx.save();
        ctx.shadowColor = '#B22A2A';
        ctx.shadowBlur = 5;
        ctx.fillStyle = '#B22A2A';
        ctx.fillText(id, x, labelY);
        ctx.restore();
      } else {
        ctx.fillStyle = dimmed ? '#ffffff22' : '#ffffffcc';
        ctx.fillText(id, x, labelY);
      }
    }
  }, [allHighlights, selectedNode, searchPickId]);

  const getLinkColor = useCallback((link: Record<string, unknown>) => {
    const src  = (link.source as { id?: string } | string);
    const tgt  = (link.target as { id?: string } | string);
    const srcId = typeof src === 'string' ? src : src.id ?? '';
    const tgtId = typeof tgt === 'string' ? tgt : tgt.id ?? '';
    const isHL  = edgeHighlights.has(srcId) || edgeHighlights.has(tgtId);
    const dimmed = edgeHighlights.size > 0 && !isHL;
    const base   = EDGE_COLORS[link.type as string] ?? '#6b7280';
    const conf   = link.confidence as number;
    // Dimmed edges stay faintly visible — they read as a background layer underneath
    // the selected node + its immediate neighbors, rather than disappearing.
    const alpha  = dimmed ? '20' : conf > 0.7 ? 'cc' : conf > 0.45 ? '77' : '44';
    return base + alpha;
  }, [edgeHighlights]);

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

  // Bright shine stroke drawn ON TOP of the default link — gives the wire a polished
  // filament look matching the gem palette. Skipped for dimmed links so highlight focus
  // is preserved.
  const linkCanvasObject = useCallback((
    link: Record<string, unknown>,
    ctx: CanvasRenderingContext2D,
    globalScale: number,
  ) => {
    const src = link.source as { x?: number; y?: number; id?: string } | undefined;
    const tgt = link.target as { x?: number; y?: number; id?: string } | undefined;
    if (!src || !tgt) return;
    const x1 = src.x, y1 = src.y, x2 = tgt.x, y2 = tgt.y;
    if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return;

    const type   = link.type as string;
    const conf   = link.confidence as number;
    const srcId  = src.id ?? '';
    const tgtId  = tgt.id ?? '';
    const isHL   = edgeHighlights.has(srcId) || edgeHighlights.has(tgtId);
    const dimmed = edgeHighlights.size > 0 && !isHL;
    if (dimmed) return;

    const shine    = EDGE_SHINE[type] ?? '#cccccc';
    const baseLw   = type === 'SIMILAR_TO' ? 0.8 : Math.max(0.5, Math.min(3, conf * 3));
    const shineLw  = baseLw * 0.42;
    const alpha    = isHL ? 'ee' : conf > 0.7 ? 'aa' : conf > 0.45 ? '7a' : '4a';

    ctx.strokeStyle = shine + alpha;
    ctx.lineWidth = shineLw / globalScale;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1 as number, y1 as number);
    if (type === 'SIMILAR_TO') {
      // Match the 0.25 curvature used by the default renderer.
      const mx = ((x1 as number) + (x2 as number)) / 2;
      const my = ((y1 as number) + (y2 as number)) / 2;
      const dx = (x2 as number) - (x1 as number);
      const dy = (y2 as number) - (y1 as number);
      const cx = mx + dy * 0.25;
      const cy = my - dx * 0.25;
      ctx.quadraticCurveTo(cx, cy, x2 as number, y2 as number);
    } else {
      ctx.lineTo(x2 as number, y2 as number);
    }
    ctx.stroke();
  }, [edgeHighlights]);

  const toggleGroup = (g: EdgeGroup) => {
    setActiveGroups(prev => {
      const next = new Set(prev);
      next.has(g) ? next.delete(g) : next.add(g);
      return next;
    });
  };

  if (loading) return <GraphSkeleton />;

  const GROUPS: [EdgeGroup, string, string][] = [
    ['direct',     'Drug→Protein', '#E5B547'], // gold
    ['similarity', 'Similarity',   '#4D6FC4'], // lighter sapphire
    ['inferred',   'Inferred',     '#D03D5F'], // ruby
  ];

  return (
    <>
      {/* Canvas — fills the entire viewport. All controls float on top as glass overlays. */}
      <div ref={containerRef} className="absolute inset-0 z-0">
        {data && (
          <ForceGraph2D
            ref={setGraphRef}
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="transparent"
            nodeLabel={(n: Record<string, unknown>) => {
              const t = String(n.type ?? '');
              const typeLabel = t === 'BIOMOLECULE' ? 'Protein' : t === 'CHEMICAL_CANDIDATE' ? 'Compound' : 'Pathway';
              return `${String(n.id)} — ${typeLabel}`;
            }}
            nodeCanvasObject={nodeCanvasObject}
            nodeCanvasObjectMode={() => 'replace'}
            linkColor={getLinkColor}
            linkWidth={getLinkWidth}
            linkDirectionalArrowLength={getLinkArrowLen}
            linkDirectionalArrowRelPos={1}
            linkCurvature={getLinkCurvature}
            linkCanvasObject={linkCanvasObject}
            linkCanvasObjectMode={() => 'after'}
            onNodeClick={(node) => setSelectedNode(prev => (prev as { id?: string } | null)?.id === (node as { id: string }).id ? null : node as Record<string, unknown>)}
            enableNodeDrag={false}
            onEngineStop={handleEngineStop}
            onZoom={handleZoom}
            minZoom={dynamicMinZoom}
            maxZoom={8}
            cooldownTicks={120}
            cooldownTime={2000}
            d3AlphaDecay={0.045}
            d3VelocityDecay={0.6}
          />
        )}

        {/* Empty state — when all edge groups are toggled off */}
        {data && graphData.nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none px-6">
            <div className="text-center max-w-[260px]">
              <p className="text-sm text-zinc-300 font-medium mb-1">No edges to show</p>
              <p className="text-xs text-zinc-500 leading-relaxed">
                Enable at least one edge type above (Drug→Protein, Similarity, or Inferred) to see the network.
              </p>
            </div>
          </div>
        )}

        {/* Node detail card — anchored above the legend so they don't overlap */}
        {selectedNode && (
          <div className="absolute w-64 z-20" style={{ bottom: '64px', left: '16px' }}>
            <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
          </div>
        )}

        {/* Propagation toast */}
        {propagResult && (
          <div
            role="status"
            aria-live="polite"
            className="absolute top-3 left-1/2 -translate-x-1/2 text-xs text-violet-200 px-3 py-1.5 rounded-full whitespace-nowrap flex items-center gap-1.5 tabular-nums"
            style={{ background: 'rgba(139,92,246,0.22)', border: '1px solid rgba(139,92,246,0.35)' }}
          >
            <SparkIcon size={11} />
            {propagResult.edgesInferred} inferred · {propagResult.edgesUpdated} updated
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="absolute rounded-lg px-3 py-2 text-xs text-red-200"
            style={{ top: '72px', left: '50%', transform: 'translateX(-50%)', background: 'rgba(239,68,68,0.18)', border: '1px solid rgba(239,68,68,0.35)', backdropFilter: 'blur(12px)' }}
          >
            {error}
          </div>
        )}
      </div>

      {/* Top-left floating control card — Knowledge Graph header + search + edge toggles */}
      <div
        className="absolute z-20 glass-refract rounded-2xl overflow-visible"
        style={{ top: '72px', left: '16px', width: '320px' }}
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-white/[0.04] flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Knowledge Graph</h2>
            <p className="text-xs text-zinc-400 mt-0.5 tabular-nums">
              {graphData.nodes.length} nodes · {graphData.links.length} edges
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={runPropagation}
              disabled={propagating}
              className="text-xs px-2.5 py-1.5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer min-h-[32px]"
              style={{ background: 'rgba(139,92,246,0.18)', border: '1px solid rgba(139,92,246,0.28)', color: '#c4b5fd' }}
            >
              {propagating ? 'Propagating…' : 'Propagate'}
            </button>
            <button
              onClick={fetchGraph}
              className="text-xs text-zinc-400 hover:text-zinc-200 px-2.5 py-1.5 rounded-lg hover:bg-white/[0.06] transition-colors cursor-pointer min-h-[32px]"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="px-3 py-2 border-b border-white/[0.04]">
          <div className="relative">
            {searchQuery && (
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  setSearchQuery('');
                  setSearchPickId(null);
                  setSearchFocused(true);
                }}
                aria-label="Clear search"
                className="absolute right-1 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.06] w-6 h-6 flex items-center justify-center rounded transition-colors z-10 cursor-pointer"
              >
                <CloseIcon size={12} />
              </button>
            )}
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setSearchPickId(null);
                setSearchFocused(true);
              }}
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setTimeout(() => setSearchFocused(false), 150)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setSearchQuery('');
                  setSearchPickId(null);
                  setSearchFocused(false);
                  (e.target as HTMLInputElement).blur();
                } else if (e.key === 'Enter' && searchOptions.length > 0) {
                  pickSearchOption(searchOptions[0].id);
                }
              }}
              placeholder="Search nodes by name…"
              aria-label="Search graph nodes"
              className="w-full text-xs pl-3 pr-7 py-1.5 rounded-lg bg-white/[0.05] border border-white/[0.08] text-white placeholder:text-zinc-500 focus:outline-none focus:border-violet-500/40 focus:ring-1 focus:ring-violet-500/20"
            />
            {searchFocused && searchQuery.trim() && (
              <div
                className="absolute left-0 right-0 top-full mt-1 z-20 rounded-lg overflow-hidden max-h-64 overflow-y-auto"
                style={{ background: 'rgba(9,9,11,0.96)', border: '1px solid rgba(255,255,255,0.08)', backdropFilter: 'blur(12px)' }}
              >
                {searchOptions.length === 0 ? (
                  <div className="px-3 py-2.5 text-xs text-red-300">No match in graph for &quot;{searchQuery.trim()}&quot;</div>
                ) : (
                  searchOptions.map(opt => (
                    <button
                      key={opt.id}
                      onMouseDown={(e) => { e.preventDefault(); pickSearchOption(opt.id); }}
                      className="w-full text-left px-3 py-2 text-xs hover:bg-white/[0.05] flex items-center gap-2 border-b border-white/[0.03] last:border-b-0 cursor-pointer"
                    >
                      <div
                        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: NODE_COLORS[opt.type] ?? '#6b7280' }}
                      />
                      <span className="font-mono text-white">{opt.id}</span>
                      {opt.name !== opt.id && <span className="text-zinc-500 truncate">{opt.name}</span>}
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
        </div>

        {/* Edge type toggles */}
        <div className="px-3 py-2.5 flex items-center gap-1.5 flex-wrap">
          {GROUPS.map(([g, label, color]) => {
            const on = activeGroups.has(g);
            return (
              <button
                key={g}
                onClick={() => toggleGroup(g)}
                aria-pressed={on}
                className="text-xs px-2.5 py-1 rounded-full transition-all cursor-pointer min-h-[28px]"
                style={{
                  background: on ? color + '22' : 'rgba(255,255,255,0.04)',
                  border:     `1px solid ${on ? color + '66' : 'rgba(255,255,255,0.08)'}`,
                  color:      on ? color : '#a1a1aa',
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Bottom-left floating legend */}
      <div
        className="absolute z-20 glass-refract rounded-full px-4 py-2 flex items-center gap-4"
        style={{ bottom: '16px', left: '16px' }}
      >
        {([['#2C4A8E', 'Protein'], ['#2D5A3F', 'Compound'], ['#D03D5F', 'Pathway']] as const).map(([c, l]) => (
          <div key={l} className="flex items-center gap-1.5">
            <div aria-hidden="true" className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: c, boxShadow: `0 0 4px ${c}88` }} />
            <span className="text-xs text-zinc-300">{l}</span>
          </div>
        ))}
      </div>

      {/* Pan/zoom hint — bottom-right */}
      <div
        className="absolute z-20 hidden sm:block text-xs text-zinc-400 px-3 py-1.5 rounded-full glass-refract"
        style={{ bottom: '16px', right: '388px' }}
        aria-hidden="true"
      >
        scroll · drag · click
      </div>
    </>
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
        <button
          onClick={onClose}
          aria-label="Close node details"
          className="text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.06] ml-3 w-7 h-7 flex items-center justify-center rounded transition-colors cursor-pointer"
        ><CloseIcon size={12} /></button>
      </div>
      <p className="text-zinc-300 mb-2 leading-snug">{name}</p>
      {mw !== undefined && (
        <p className="text-zinc-400 font-mono tabular-nums">
          MW:{' '}{mw >= 1000 ? `${(mw / 1000).toFixed(1)} kDa` : `${mw} Da`}
        </p>
      )}
      {extId && <p className="text-zinc-500 font-mono mt-0.5 truncate">{extId}</p>}
      {danger && (
        <span className="inline-block mt-1.5 px-2 py-0.5 rounded text-xs font-medium text-red-400 bg-red-400/10">
          {danger}
        </span>
      )}
    </div>
  );
}

function CloseIcon({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="6" y1="6" x2="18" y2="18" />
      <line x1="18" y1="6" x2="6" y2="18" />
    </svg>
  );
}

function SparkIcon({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2l1.6 6.4L20 10l-6.4 1.6L12 18l-1.6-6.4L4 10l6.4-1.6L12 2z" />
    </svg>
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
