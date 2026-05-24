'use client';

import { useState, useRef, useEffect } from 'react';
import type { AgentRunOutput } from '@/lib/types';

interface Props {
  result: AgentRunOutput;
  onClose: () => void;
}

// ─── Minimal markdown → JSX renderer ─────────────────────────
// Handles: ## headers, **bold**, - bullets, blank-line paragraphs.

function renderMarkdown(text: string): React.ReactNode[] {
  const blocks = text.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
  return blocks.map((block, bi) => {
    // ## Heading
    if (block.startsWith('## ')) {
      return (
        <h3 key={bi} className="text-xs font-semibold text-zinc-300 uppercase tracking-widest mt-3 mb-1 first:mt-0">
          {block.slice(3)}
        </h3>
      );
    }
    // ### Heading
    if (block.startsWith('### ')) {
      return (
        <h4 key={bi} className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mt-2 mb-0.5">
          {block.slice(4)}
        </h4>
      );
    }
    // Bullet list block
    const lines = block.split('\n');
    if (lines.every((l) => l.trimStart().startsWith('- '))) {
      return (
        <ul key={bi} className="space-y-1 pl-0 my-1">
          {lines.map((l, li) => (
            <li key={li} className="flex gap-2 text-sm text-zinc-300 leading-snug">
              <span className="mt-1.5 w-1 h-1 rounded-full bg-zinc-500 flex-shrink-0" />
              <span>{inlineBold(l.replace(/^[\s-]+/, ''))}</span>
            </li>
          ))}
        </ul>
      );
    }
    // Mixed block — some bullet, some not
    if (lines.some((l) => l.trimStart().startsWith('- '))) {
      return (
        <div key={bi} className="space-y-1 my-1">
          {lines.map((l, li) => {
            const isBullet = l.trimStart().startsWith('- ');
            return isBullet ? (
              <div key={li} className="flex gap-2 text-sm text-zinc-300 leading-snug">
                <span className="mt-1.5 w-1 h-1 rounded-full bg-zinc-500 flex-shrink-0" />
                <span>{inlineBold(l.replace(/^[\s-]+/, ''))}</span>
              </div>
            ) : (
              <p key={li} className="text-sm text-zinc-300 leading-relaxed">{inlineBold(l)}</p>
            );
          })}
        </div>
      );
    }
    // Plain paragraph
    return (
      <p key={bi} className="text-sm text-zinc-300 leading-relaxed">
        {inlineBold(block.replace(/\n/g, ' '))}
      </p>
    );
  });
}

function inlineBold(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith('**') && p.endsWith('**')
      ? <strong key={i} className="font-semibold text-white">{p.slice(2, -2)}</strong>
      : p
  );
}

// ─── Expandable response block ────────────────────────────────

const COLLAPSED_HEIGHT = 168; // px

function ResponseBlock({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const innerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (innerRef.current) {
      setOverflows(innerRef.current.scrollHeight > COLLAPSED_HEIGHT + 8);
    }
  }, [text]);

  return (
    <div>
      <div
        className="relative overflow-hidden transition-all duration-300"
        style={{ maxHeight: expanded ? 'none' : `${COLLAPSED_HEIGHT}px` }}
      >
        <div ref={innerRef} className="space-y-1.5 font-sans">
          {renderMarkdown(text)}
        </div>
        {/* Fade mask when collapsed and overflowing */}
        {!expanded && overflows && (
          <div
            className="absolute bottom-0 left-0 right-0 h-10 pointer-events-none"
            style={{ background: 'linear-gradient(to bottom, transparent, rgba(18,18,22,0.95))' }}
          />
        )}
      </div>
      {overflows && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 flex items-center gap-1 text-xs text-violet-400 hover:text-violet-200 transition-colors cursor-pointer"
        >
          <ChevronIcon expanded={expanded} />
          {expanded ? 'Collapse' : 'Show full analysis'}
        </button>
      )}
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────

export default function ResultPanel({ result, onClose }: Props) {
  const { finalResponse, toolCallsExecuted, graphMutations } = result;

  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-white/[0.04] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" aria-hidden="true" />
          <h2 className="text-xs font-semibold text-white tracking-wide uppercase">Analysis</h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-zinc-600 font-mono tabular-nums">
            {toolCallsExecuted.length} calls · {graphMutations.edgesUpserted.length} edges
          </span>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-5 h-5 flex items-center justify-center rounded text-zinc-600 hover:text-zinc-200 hover:bg-white/[0.06] transition-colors cursor-pointer"
          >
            <CloseIcon />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* LLM response */}
        <ResponseBlock text={finalResponse || 'No response.'} />

        {/* Bayesian confidence updates — compact */}
        {graphMutations.confidenceUpdates.length > 0 && (
          <div
            className="rounded-lg px-3 py-2.5 space-y-1.5"
            style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
          >
            <p className="text-xs text-zinc-500 uppercase tracking-widest font-medium mb-2">
              Confidence updates
            </p>
            {graphMutations.confidenceUpdates.map((u) => {
              const delta    = u.updatedConfidence - u.previousConfidence;
              const improved = delta > 0;
              return (
                <div key={u.edgeId} className="flex items-center justify-between gap-2">
                  <span className="text-xs text-zinc-500 font-mono truncate">{u.edgeId.slice(0, 12)}…</span>
                  <div className="flex items-center gap-1.5 text-xs font-mono tabular-nums flex-shrink-0">
                    <span className="text-zinc-500">{(u.previousConfidence * 100).toFixed(1)}%</span>
                    <span className="text-zinc-600">→</span>
                    <span className={improved ? 'text-emerald-400' : 'text-red-400'}>
                      {(u.updatedConfidence * 100).toFixed(1)}%
                    </span>
                    <span className={`text-xs ${improved ? 'text-emerald-500' : 'text-red-500'}`}>
                      {improved ? '+' : ''}{(delta * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
      className={`transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
