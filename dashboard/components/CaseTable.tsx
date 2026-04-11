"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { CaseComparison } from "@/lib/data";

interface CaseTableProps {
  cases: CaseComparison[];
}

export default function CaseTable({ cases }: CaseTableProps) {
  return (
    <section className="mt-16">
      {/* Section eyebrow */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.8, ease: [0.32, 0.72, 0, 1] }}
        className="flex items-center gap-2.5 mb-8"
      >
        <div className="w-1 h-1 rounded-full bg-accent/50" />
        <h2 className="text-[10px] uppercase tracking-[0.2em] font-medium text-text-secondary">
          Case Results
        </h2>
        <span className="text-[10px] font-mono text-text-tertiary ml-1">
          ({cases.length})
        </span>
        <div className="flex-1 h-px bg-surface-border ml-2" />
      </motion.div>

      {/* Table — double bezel */}
      <motion.div
        initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
        whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        viewport={{ once: true, margin: "-60px" }}
        transition={{ duration: 0.8, ease: [0.32, 0.72, 0, 1] }}
        className="rounded-3xl p-1.5 bezel-outer"
      >
        <div className="bezel-inner rounded-[calc(1.5rem-6px)] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border/80">
                  <th className="px-5 py-4 text-left text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Case
                  </th>
                  <th className="px-5 py-4 text-center text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Zero-Shot
                  </th>
                  <th className="px-5 py-4 text-center text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Memory
                  </th>
                  <th className="px-5 py-4 text-center text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    ZS Conf
                  </th>
                  <th className="px-5 py-4 text-center text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Mem Conf
                  </th>
                  <th className="px-5 py-4 text-center text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Patterns
                  </th>
                  <th className="px-5 py-4 text-right text-[10px] uppercase tracking-[0.15em] font-medium text-text-tertiary">
                    Detail
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border/40">
                {cases.map((c, i) => (
                  <motion.tr
                    key={c.case_id}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{
                      delay: Math.min(i * 0.02, 0.4),
                      duration: 0.5,
                      ease: [0.32, 0.72, 0, 1],
                    }}
                    className="group hover:bg-surface-hover/50 transition-colors duration-500 ease-spring"
                  >
                    <td className="px-5 py-3.5 font-mono text-xs text-text-secondary tracking-tight">
                      {c.case_id}
                    </td>
                    <td className="px-5 py-3.5 text-center">
                      {c.zero_shot && (
                        <StatusBadge correct={c.zero_shot.correct} />
                      )}
                    </td>
                    <td className="px-5 py-3.5 text-center">
                      {c.memory && (
                        <StatusBadge correct={c.memory.correct} accent />
                      )}
                    </td>
                    <td className="px-5 py-3.5 text-center font-mono text-xs text-text-secondary">
                      {c.zero_shot?.confidence.toFixed(2) ?? "\u2014"}
                    </td>
                    <td className="px-5 py-3.5 text-center font-mono text-xs text-accent/80">
                      {c.memory?.confidence.toFixed(2) ?? "\u2014"}
                    </td>
                    <td className="px-5 py-3.5 text-center font-mono text-xs text-text-secondary">
                      {c.memory?.injected_patterns?.length ?? 0}
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <Link
                        href={`/case/${c.case_id}`}
                        className="inline-flex items-center gap-1 text-xs text-text-secondary hover:text-accent transition-colors duration-500 ease-spring group/link"
                      >
                        <span>View</span>
                        <span className="transition-transform duration-500 ease-spring group-hover/link:translate-x-0.5">
                          &rarr;
                        </span>
                      </Link>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </motion.div>
    </section>
  );
}

function StatusBadge({
  correct,
  accent,
}: {
  correct: boolean;
  accent?: boolean;
}) {
  if (correct) {
    return (
      <span
        className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide ${
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
    <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide bg-rose-500/[0.08] text-rose-400 border border-rose-500/[0.12]">
      WRONG
    </span>
  );
}
