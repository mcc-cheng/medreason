'use client';

import type { AgentRunOutput } from '@/lib/types';

interface Props {
  result: AgentRunOutput;
}

export default function ResultPanel({ result }: Props) {
  const { finalResponse, toolCallsExecuted, graphMutations, totalTokensUsed, sessionId } = result;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Agent Response</h2>
        <span className="text-xs text-gray-500 font-mono">{sessionId.slice(0, 12)}…</span>
      </div>

      {/* Final response */}
      <div className="bg-gray-800 rounded-lg p-4 text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">
        {finalResponse}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Tool calls" value={toolCallsExecuted.length} />
        <Stat label="Graph edges updated" value={graphMutations.edgesUpserted.length} />
        <Stat label="Tokens used" value={totalTokensUsed.toLocaleString()} />
      </div>

      {/* Confidence updates */}
      {graphMutations.confidenceUpdates.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Confidence updates</p>
          <div className="space-y-2">
            {graphMutations.confidenceUpdates.map((u) => (
              <div
                key={u.edgeId}
                className="flex items-center justify-between bg-gray-800 rounded-lg px-3 py-2 text-xs font-mono"
              >
                <span className="text-gray-400 truncate">{u.edgeId.slice(0, 16)}…</span>
                <span className="flex items-center gap-2">
                  <span className="text-gray-500">{u.previousConfidence.toFixed(3)}</span>
                  <span className="text-gray-600">→</span>
                  <span
                    className={
                      u.updatedConfidence > u.previousConfidence
                        ? 'text-green-400'
                        : 'text-red-400'
                    }
                  >
                    {u.updatedConfidence.toFixed(3)}
                  </span>
                  <span className="text-gray-600 ml-1">n={u.updatedObservationCount}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tool call log */}
      {toolCallsExecuted.length > 0 && (
        <details className="group">
          <summary className="text-xs text-gray-400 uppercase tracking-wider cursor-pointer hover:text-gray-300 transition-colors">
            Tool call log ({toolCallsExecuted.length})
          </summary>
          <div className="mt-2 space-y-2">
            {toolCallsExecuted.map((tc) => (
              <div
                key={tc.toolCallId}
                className="bg-gray-800 rounded-lg px-3 py-2 text-xs font-mono"
              >
                <span className="text-blue-400">{tc.toolName}</span>
                <span className="text-gray-500 ml-2">{JSON.stringify(tc.input)}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-800 rounded-lg px-3 py-2 text-center">
      <p className="text-lg font-semibold text-white">{value}</p>
      <p className="text-xs text-gray-400 mt-0.5">{label}</p>
    </div>
  );
}
