'use client';

import type { AgentRunOutput } from '@/lib/types';

interface Props {
  result: AgentRunOutput;
}

export default function ResultPanel({ result }: Props) {
  const { finalResponse, toolCallsExecuted, graphMutations, totalTokensUsed, sessionId } = result;

  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse">
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/[0.04] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" />
          <h2 className="text-sm font-semibold text-white">Agent Response</h2>
        </div>
        <span className="text-[10px] text-zinc-600 font-mono">{sessionId.slice(0, 14)}</span>
      </div>

      <div className="p-5 space-y-5">
        {/* Stats */}
        <div className="grid grid-cols-3 gap-2">
          <StatCard label="Tool calls" value={toolCallsExecuted.length} color="violet" />
          <StatCard label="Edges updated" value={graphMutations.edgesUpserted.length} color="emerald" />
          <StatCard label="Tokens" value={totalTokensUsed.toLocaleString()} color="zinc" />
        </div>

        {/* Response text */}
        <div
          className="rounded-xl px-4 py-4 text-sm text-zinc-200 leading-relaxed whitespace-pre-wrap"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
        >
          {finalResponse}
        </div>

        {/* Confidence updates */}
        {graphMutations.confidenceUpdates.length > 0 && (
          <div>
            <p className="text-xs text-zinc-600 uppercase tracking-widest mb-2">Bayesian updates</p>
            <div className="space-y-2">
              {graphMutations.confidenceUpdates.map((u) => {
                const improved = u.updatedConfidence > u.previousConfidence;
                const delta = (u.updatedConfidence - u.previousConfidence);
                return (
                  <div
                    key={u.edgeId}
                    className="flex items-center justify-between px-3 py-2.5 rounded-lg"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    <span className="text-xs text-zinc-500 font-mono">{u.edgeId.slice(0, 18)}…</span>
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <span className="text-zinc-500">{(u.previousConfidence * 100).toFixed(1)}%</span>
                      <span className="text-zinc-700">→</span>
                      <span className={improved ? 'text-emerald-400' : 'text-red-400'}>
                        {(u.updatedConfidence * 100).toFixed(1)}%
                      </span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        improved ? 'text-emerald-400 bg-emerald-400/10' : 'text-red-400 bg-red-400/10'
                      }`}>
                        {improved ? '+' : ''}{(delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tool call log */}
        {toolCallsExecuted.length > 0 && (
          <details className="group">
            <summary className="text-xs text-zinc-600 uppercase tracking-widest cursor-pointer hover:text-zinc-400 transition-colors select-none">
              Tool log · {toolCallsExecuted.length} calls
            </summary>
            <div className="mt-2 space-y-1.5">
              {toolCallsExecuted.map((tc) => (
                <div
                  key={tc.toolCallId}
                  className="px-3 py-2 rounded-lg text-xs font-mono"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <span className="text-violet-400">{tc.toolName}</span>
                  <span className="text-zinc-600 ml-2">{JSON.stringify(tc.input)}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: 'violet' | 'emerald' | 'zinc' }) {
  const accent = color === 'violet' ? 'text-violet-400' : color === 'emerald' ? 'text-emerald-400' : 'text-zinc-300';
  return (
    <div
      className="rounded-xl px-3 py-3 text-center"
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
    >
      <p className={`text-lg font-semibold ${accent}`}>{value}</p>
      <p className="text-[10px] text-zinc-600 mt-0.5 uppercase tracking-wider">{label}</p>
    </div>
  );
}
