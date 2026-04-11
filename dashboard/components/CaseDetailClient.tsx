"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { CaseComparison } from "@/lib/data";

interface CaseDetailClientProps {
  caseId: string;
  comparison: CaseComparison;
  benchmarkCase?: {
    task_config: {
      payer: string;
      cpt_code: string;
      facility_type: string;
    };
    difficulty: string;
    clinical_notes: string;
    policy_excerpt: string;
    ground_truth_outcome: string;
    ground_truth_reasoning: string[];
  };
}

export default function CaseDetailClient({
  caseId,
  comparison,
  benchmarkCase,
}: CaseDetailClientProps) {
  return (
    <main className="min-h-[100dvh] px-4 md:px-6 pt-28 pb-16 max-w-[1100px] mx-auto">
      {/* Back link */}
      <motion.div
        initial={{ opacity: 0, x: -8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: [0.32, 0.72, 0, 1] }}
      >
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-xs text-text-secondary hover:text-accent transition-colors duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] mb-10 group"
        >
          <span className="transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:-translate-x-0.5">
            &larr;
          </span>
          <span>Back to Benchmark</span>
        </Link>
      </motion.div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 24, filter: "blur(12px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 0.8, ease: [0.32, 0.72, 0, 1] }}
        className="mb-12"
      >
        <div className="flex items-center gap-2.5 mb-4">
          <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] uppercase tracking-[0.15em] font-medium bg-surface border border-surface-border text-text-secondary">
            Case Detail
          </span>
        </div>
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tighter text-text-primary">
          {caseId}
        </h1>
      </motion.div>

      {/* Task Config */}
      {benchmarkCase && (
        <motion.section
          initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{
            duration: 0.8,
            delay: 0.1,
            ease: [0.32, 0.72, 0, 1],
          }}
          className="mb-8"
        >
          <div className="rounded-3xl p-1.5 bezel-outer">
            <div className="bezel-inner rounded-[calc(1.5rem-6px)] p-6 md:p-8">
              {/* Config eyebrow */}
              <div className="flex items-center gap-2.5 mb-6">
                <div className="w-1 h-1 rounded-full bg-accent/50" />
                <h2 className="text-[10px] uppercase tracking-[0.15em] font-medium text-text-secondary">
                  Task Configuration
                </h2>
              </div>

              {/* Config grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                <ConfigItem label="Payer" value={benchmarkCase.task_config.payer} />
                <ConfigItem
                  label="CPT"
                  value={benchmarkCase.task_config.cpt_code}
                  mono
                />
                <ConfigItem
                  label="Facility"
                  value={benchmarkCase.task_config.facility_type}
                />
                <ConfigItem label="Difficulty" value={benchmarkCase.difficulty} />
              </div>

              {/* Clinical Notes */}
              <div className="mt-6 pt-6 border-t border-surface-border/60">
                <h3 className="text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary mb-3">
                  Clinical Notes
                </h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  {benchmarkCase.clinical_notes}
                </p>
              </div>

              {/* Policy */}
              <div className="mt-6 pt-6 border-t border-surface-border/60">
                <h3 className="text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary mb-3">
                  Policy Excerpt
                </h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  {benchmarkCase.policy_excerpt}
                </p>
              </div>

              {/* Ground Truth */}
              <div className="mt-6 pt-6 border-t border-surface-border/60">
                <div className="flex items-center gap-2 mb-3">
                  <h3 className="text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Ground Truth
                  </h3>
                  <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide bg-accent/[0.08] text-accent border border-accent/[0.12]">
                    {benchmarkCase.ground_truth_outcome}
                  </span>
                </div>
                <ol className="list-decimal list-inside text-sm text-text-secondary space-y-1.5 leading-relaxed">
                  {benchmarkCase.ground_truth_reasoning.map(
                    (step: string, i: number) => (
                      <li key={i}>{step}</li>
                    )
                  )}
                </ol>
              </div>
            </div>
          </div>
        </motion.section>
      )}

      {/* Side-by-side Comparison */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Zero-Shot Panel */}
        <motion.div
          initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{
            duration: 0.8,
            delay: 0.2,
            ease: [0.32, 0.72, 0, 1],
          }}
          className="rounded-3xl p-1.5 bezel-outer"
        >
          <div className="bezel-inner rounded-[calc(1.5rem-6px)] p-6 md:p-8">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2.5">
                <div className="w-1 h-1 rounded-full bg-zero-shot" />
                <h2 className="text-[10px] uppercase tracking-[0.15em] font-medium text-text-secondary">
                  Zero-Shot
                </h2>
              </div>
              {comparison.zero_shot && (
                <ResultBadge correct={comparison.zero_shot.correct} />
              )}
            </div>

            {comparison.zero_shot ? (
              <>
                <div className="flex flex-wrap gap-4 text-xs text-text-secondary mb-4">
                  <MetricPill
                    label="Result"
                    value={comparison.zero_shot.determination}
                  />
                  <MetricPill
                    label="Confidence"
                    value={comparison.zero_shot.confidence.toFixed(2)}
                    mono
                  />
                  <MetricPill
                    label="Tokens"
                    value={String(comparison.zero_shot.tokens)}
                    mono
                  />
                </div>
                <div className="bg-bg rounded-2xl p-4 text-sm text-text-secondary leading-relaxed max-h-64 overflow-y-auto border border-surface-border/40">
                  {comparison.zero_shot.reasoning}
                </div>
              </>
            ) : (
              <p className="text-text-secondary text-sm">Not run</p>
            )}
          </div>
        </motion.div>

        {/* Memory Panel */}
        <motion.div
          initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{
            duration: 0.8,
            delay: 0.3,
            ease: [0.32, 0.72, 0, 1],
          }}
          className={`rounded-3xl p-1.5 ${
            comparison.memory?.correct
              ? "bg-accent/[0.03] ring-1 ring-accent/[0.1] glow-accent"
              : "bezel-outer"
          }`}
        >
          <div
            className={`rounded-[calc(1.5rem-6px)] p-6 md:p-8 ${
              comparison.memory?.correct
                ? "bg-accent/[0.03] border border-accent/[0.06] shadow-[inset_0_1px_1px_rgba(52,211,153,0.08)]"
                : "bezel-inner"
            }`}
          >
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2.5">
                <div className="w-1 h-1 rounded-full bg-accent" />
                <h2 className="text-[10px] uppercase tracking-[0.15em] font-medium text-accent/80">
                  Memory-Augmented
                </h2>
              </div>
              {comparison.memory && (
                <ResultBadge correct={comparison.memory.correct} accent />
              )}
            </div>

            {comparison.memory ? (
              <>
                <div className="flex flex-wrap gap-4 text-xs text-text-secondary mb-4">
                  <MetricPill
                    label="Result"
                    value={comparison.memory.determination}
                    accent
                  />
                  <MetricPill
                    label="Confidence"
                    value={comparison.memory.confidence.toFixed(2)}
                    mono
                    accent
                  />
                  <MetricPill
                    label="Tokens"
                    value={String(comparison.memory.tokens)}
                    mono
                  />
                </div>

                {comparison.memory.injected_patterns.length > 0 && (
                  <div className="mb-4 flex flex-wrap gap-1.5">
                    {comparison.memory.injected_patterns.map((pid) => (
                      <span
                        key={pid}
                        className="inline-block bg-accent/[0.06] text-accent/80 px-2 py-0.5 rounded-full text-[10px] font-mono border border-accent/[0.1]"
                      >
                        {pid}
                      </span>
                    ))}
                  </div>
                )}

                <div className="bg-bg rounded-2xl p-4 text-sm text-text-secondary leading-relaxed max-h-64 overflow-y-auto border border-surface-border/40">
                  {comparison.memory.reasoning}
                </div>
              </>
            ) : (
              <p className="text-text-secondary text-sm">Not run</p>
            )}
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="mt-16 pt-6 border-t border-surface-border">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-xs text-text-secondary hover:text-accent transition-colors duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group"
        >
          <span className="transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:-translate-x-0.5">
            &larr;
          </span>
          <span>Back to Benchmark</span>
        </Link>
      </footer>
    </main>
  );
}

function ConfigItem({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <span className="text-[10px] uppercase tracking-[0.12em] text-text-tertiary">
        {label}
      </span>
      <p
        className={`text-text-primary font-medium mt-0.5 ${
          mono ? "font-mono text-sm" : "text-sm"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function MetricPill({
  label,
  value,
  mono,
  accent,
}: {
  label: string;
  value: string;
  mono?: boolean;
  accent?: boolean;
}) {
  return (
    <span className="text-xs text-text-secondary">
      {label}:{" "}
      <strong
        className={`${mono ? "font-mono" : "font-medium"} ${
          accent ? "text-accent" : "text-text-primary"
        }`}
      >
        {value}
      </strong>
    </span>
  );
}

function ResultBadge({
  correct,
  accent,
}: {
  correct: boolean;
  accent?: boolean;
}) {
  if (correct) {
    return (
      <span
        className={`px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide ${
          accent
            ? "bg-accent/[0.08] text-accent border border-accent/[0.12]"
            : "bg-emerald-500/[0.08] text-emerald-400 border border-emerald-500/[0.12]"
        }`}
      >
        CORRECT
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide bg-rose-500/[0.08] text-rose-400 border border-rose-500/[0.12]">
      WRONG
    </span>
  );
}
