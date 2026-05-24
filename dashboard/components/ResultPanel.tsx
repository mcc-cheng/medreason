'use client';

import type { AgentRunOutput } from '@/lib/types';

interface Props {
  result: AgentRunOutput;
}

export default function ResultPanel({ result }: Props) {
  const { finalResponse, toolCallsExecuted, graphMutations, totalTokensUsed, sessionId } = result;

  return (
    <div className="glass-refract rounded-2xl overflow-hidden shadow-diffuse flex-shrink-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/[0.04] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" aria-hidden="true" />
          <h2 className="text-sm font-semibold text-white">Agent Response</h2>
        </div>
        <div className="flex items-center gap-3 text-xs text-zinc-400 font-mono tabular-nums">
          <span>{toolCallsExecuted.length} calls</span>
          <span>{graphMutations.edgesUpserted.length} edges</span>
          <span>{totalTokensUsed.toLocaleString()} tok</span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* LLM response — primary content */}
        <div
          className="rounded-xl px-4 py-4 text-sm text-zinc-200 leading-relaxed whitespace-pre-wrap"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
        >
          {finalResponse || <span className="text-zinc-600 italic">No response.</span>}
        </div>

        {/* Bayesian confidence updates */}
        {graphMutations.confidenceUpdates.length > 0 && (
          <div>
            <p className="text-xs text-zinc-400 uppercase tracking-widest mb-2 font-medium">Bayesian updates</p>
            <div className="space-y-1.5">
              {graphMutations.confidenceUpdates.map((u) => {
                const delta    = u.updatedConfidence - u.previousConfidence;
                const improved = delta > 0;
                const arrow    = improved ? '↑' : '↓';
                return (
                  <div
                    key={u.edgeId}
                    className="flex items-center justify-between px-3 py-2 rounded-lg gap-3"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
                  >
                    <span className="text-xs text-zinc-400 font-mono truncate flex-1 min-w-0">
                      {u.edgeId.slice(0, 14)}…
                    </span>
                    <div className="flex items-center gap-2 text-xs font-mono flex-shrink-0 tabular-nums">
                      <span className="text-zinc-400">{(u.previousConfidence * 100).toFixed(1)}%</span>
                      <ArrowRightIcon />
                      <span className={improved ? 'text-emerald-300' : 'text-red-300'}>
                        {(u.updatedConfidence * 100).toFixed(1)}%
                      </span>
                      <span
                        aria-label={`${improved ? 'increased' : 'decreased'} by ${Math.abs(delta * 100).toFixed(1)} percent`}
                        className={`px-1.5 py-0.5 rounded inline-flex items-center gap-0.5 ${improved ? 'text-emerald-300 bg-emerald-400/10' : 'text-red-300 bg-red-400/10'}`}
                      >
                        <span aria-hidden="true">{arrow}</span>
                        <span>{improved ? '+' : ''}{(delta * 100).toFixed(1)}%</span>
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tool call log — collapsed by default */}
        {toolCallsExecuted.length > 0 && (
          <details>
            <summary className="text-xs text-zinc-400 uppercase tracking-widest cursor-pointer hover:text-zinc-200 transition-colors select-none font-medium">
              Tool log · {toolCallsExecuted.length} calls
            </summary>
            <div className="mt-2 space-y-1.5">
              {toolCallsExecuted.map((tc) => (
                <div
                  key={tc.toolCallId}
                  className="px-3 py-2 rounded-lg text-xs font-mono leading-relaxed break-words"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <span className="text-violet-300">{tc.toolName}</span>
                  <span className="text-zinc-500 ml-2">{JSON.stringify(tc.input)}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

function ArrowRightIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="text-zinc-500">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="13 6 19 12 13 18" />
    </svg>
  );
}
