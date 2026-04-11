"use client";

import { motion } from "framer-motion";

interface HeroSectionProps {
  isDemo: boolean;
}

export default function HeroSection({ isDemo }: HeroSectionProps) {
  return (
    <header className="mb-16 md:mb-24 grid grid-cols-1 md:grid-cols-12 gap-8 items-end">
      {/* Left — Title block */}
      <motion.div
        initial={{ opacity: 0, y: 32, filter: "blur(12px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 1, ease: [0.32, 0.72, 0, 1] }}
        className="md:col-span-7"
      >
        {/* Eyebrow */}
        <div className="flex items-center gap-2.5 mb-6">
          <span className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.2em] font-medium bg-surface border border-surface-border text-text-secondary">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-soft" />
            {isDemo ? "Demo Mode" : "Live Benchmark"}
          </span>
        </div>

        {/* H1 */}
        <h1 className="text-4xl md:text-6xl font-semibold tracking-tighter leading-none text-text-primary mb-5">
          Veridicus
        </h1>
        <p className="text-base md:text-lg text-text-secondary leading-relaxed max-w-[52ch]">
          Institutional reasoning memory for healthcare AI. Benchmarking
          zero-shot against memory-augmented prior authorization decisions.
        </p>
      </motion.div>

      {/* Right — Ambient metric */}
      <motion.div
        initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 1, delay: 0.2, ease: [0.32, 0.72, 0, 1] }}
        className="md:col-span-5 flex md:justify-end"
      >
        <div className="flex items-baseline gap-3">
          <span className="text-5xl md:text-7xl font-semibold tracking-tighter text-accent leading-none">
            +20
          </span>
          <div className="flex flex-col">
            <span className="text-xs text-text-secondary tracking-wide uppercase">
              percentage
            </span>
            <span className="text-xs text-text-secondary tracking-wide uppercase">
              point lift
            </span>
          </div>
        </div>
      </motion.div>
    </header>
  );
}
