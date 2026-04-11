"use client";

import { motion } from "framer-motion";

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
  delta?: boolean;
  index?: number;
}

export default function StatCard({
  label,
  value,
  sub,
  accent,
  delta,
  index = 0,
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{
        duration: 0.8,
        delay: index * 0.12,
        ease: [0.32, 0.72, 0, 1],
      }}
      className={`rounded-3xl p-1.5 ${
        accent
          ? "bg-accent/[0.04] ring-1 ring-accent/[0.12] glow-accent"
          : "bezel-outer"
      }`}
    >
      <div
        className={`rounded-[calc(1.5rem-6px)] p-6 md:p-8 ${
          accent
            ? "bg-accent/[0.04] border border-accent/[0.08] shadow-[inset_0_1px_1px_rgba(52,211,153,0.1)]"
            : "bezel-inner"
        }`}
      >
        {/* Eyebrow tag */}
        <div className="flex items-center gap-2 mb-4">
          <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] uppercase tracking-[0.15em] font-medium bg-surface-hover text-text-secondary border border-surface-border">
            {label}
          </span>
          {accent && (
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-soft" />
          )}
        </div>

        {/* Value */}
        <p
          className={`text-4xl md:text-5xl font-semibold tracking-tighter leading-none ${
            accent || delta ? "text-accent" : "text-text-primary"
          }`}
        >
          {value}
        </p>

        {/* Subtitle */}
        {sub && (
          <p className="text-sm text-text-secondary mt-3 font-mono tracking-tight">
            {sub}
          </p>
        )}
      </div>
    </motion.div>
  );
}
