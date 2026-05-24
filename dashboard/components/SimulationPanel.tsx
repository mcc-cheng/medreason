'use client';

import { useState } from 'react';
import type { AgentRunOutput } from '@/lib/types';

const PROTEINS = [
  { id: 'BCR-ABL', label: 'BCR-ABL', sub: 'Tyrosine kinase'        },
  { id: 'EGFR',    label: 'EGFR',    sub: 'Growth factor receptor'  },
  { id: 'COX-2',   label: 'COX-2',   sub: 'Cyclooxygenase enzyme'   },
];

const COMPOUNDS = [
  { id: 'IMATINIB',  label: 'Imatinib',  sub: '493.6 Da' },
  { id: 'GEFITINIB', label: 'Gefitinib', sub: '446.9 Da' },
  { id: 'ASPIRIN',   label: 'Aspirin',   sub: '180.2 Da' },
];

interface Props {
  onResult: (result: AgentRunOutput, focalNodeIds: string[]) => void;
}

export default function SimulationPanel({ onResult }: Props) {
  const [compound, setCompound] = useState<string | null>(null);
  const [protein, setProtein]   = useState<string | null>(null);
  const [extraContext, setExtraContext] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);

  const canSubmit = !!compound && !!protein && !loading;

  function buildPrompt(compoundId: string, proteinId: string) {
    const base =
      `Analyze the interaction between compound ${compoundId} and protein ${proteinId}. ` +
      `Use the analyze_compound_protein_interaction tool to retrieve the direct interaction edge, ` +
      `competing compounds that also target ${proteinId}, and other proteins that ${compoundId} affects. ` +
      `Interpret the Bayesian confidence scores, rank all competing compounds by confidence, ` +
      `and explain the pharmacological significance of the findings.`;
    return extraContext.trim() ? `${base}\n\nAdditional context: ${extraContext.trim()}` : base;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userPrompt:    buildPrompt(compound!, protein!),
          focalNodeIds:  [compound!, protein!],
          subGraphDepth: 2,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? 'Analysis failed');
      onResult(data as AgentRunOutput, [compound!, protein!]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  const compoundLabel = compound ? COMPOUNDS.find((c) => c.id === compound)?.label : null;
  const proteinLabel  = protein  ? PROTEINS.find((p)  => p.id === protein)?.label  : null;

  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse">
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/[0.04]">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Interaction Analysis</h2>
            <p className="text-xs text-zinc-400 mt-0.5">Select a compound and target protein</p>
          </div>
          {loading && (
            <div className="flex items-center gap-1.5 text-xs text-violet-400">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
              Querying graph…
            </div>
          )}
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* Compound selector */}
        <NodeGroup
          label="Compound"
          items={COMPOUNDS}
          selected={compound}
          onSelect={setCompound}
          activeColor={{ bg: 'rgba(52,211,153,0.1)', border: 'rgba(52,211,153,0.35)', dot: '#34d399' }}
        />

        {/* Protein selector */}
        <NodeGroup
          label="Target Protein"
          items={PROTEINS}
          selected={protein}
          onSelect={setProtein}
          activeColor={{ bg: 'rgba(139,92,246,0.15)', border: 'rgba(139,92,246,0.4)', dot: '#8b5cf6' }}
        />

        {/* Optional extra context + submit */}
        <form onSubmit={handleSubmit} className="space-y-3 pt-1">
          <div>
            <label
              htmlFor="extra-context"
              className="block text-xs text-zinc-400 uppercase tracking-widest mb-1.5 font-medium"
            >
              Additional context{' '}
              <span className="normal-case text-zinc-600">(optional)</span>
            </label>
            <textarea
              id="extra-context"
              rows={2}
              className="w-full rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 resize-none focus:outline-none transition-all"
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
              placeholder="e.g. Focus on selectivity vs off-target effects…"
              value={extraContext}
              onChange={(e) => setExtraContext(e.target.value)}
            />
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            className="relative w-full py-3 rounded-xl text-sm font-medium text-white transition-all overflow-hidden disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer min-h-[44px]"
            style={{
              background: canSubmit
                ? 'linear-gradient(135deg, #6d28d9 0%, #0d9488 100%)'
                : 'rgba(255,255,255,0.05)',
              boxShadow: canSubmit
                ? '0 0 20px rgba(109,40,217,0.25), 0 1px 0 rgba(255,255,255,0.1) inset'
                : 'none',
            }}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing…
              </span>
            ) : canSubmit ? (
              `Analyze ${compoundLabel} → ${proteinLabel}`
            ) : (
              `Select ${!compound ? 'a compound' : 'a target protein'}`
            )}
          </button>
        </form>

        {error && (
          <div
            role="alert"
            className="rounded-xl px-4 py-3 text-xs text-red-300 leading-relaxed"
            style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.15)' }}
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

function NodeGroup({
  label, items, selected, onSelect, activeColor,
}: {
  label: string;
  items: { id: string; label: string; sub: string }[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  activeColor: { bg: string; border: string; dot: string };
}) {
  return (
    <div>
      <p className="text-xs text-zinc-400 uppercase tracking-widest mb-2 font-medium">{label}</p>
      <div className="grid grid-cols-3 gap-1.5">
        {items.map((item) => {
          const active = selected === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(active ? null : item.id)}
              aria-pressed={active}
              className="relative text-left px-3 py-2 rounded-lg transition-all cursor-pointer min-h-[44px]"
              style={{
                background: active ? activeColor.bg : 'rgba(255,255,255,0.02)',
                border: `1px solid ${active ? activeColor.border : 'rgba(255,255,255,0.05)'}`,
                boxShadow: active ? `0 0 10px ${activeColor.dot}22` : 'none',
              }}
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: active ? activeColor.dot : '#52525b' }}
                />
                <span className={`text-xs font-mono font-medium truncate ${active ? 'text-white' : 'text-zinc-300'}`}>
                  {item.label}
                </span>
              </div>
              <p className="text-xs text-zinc-500 leading-tight pl-3 truncate">{item.sub}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
