"use client";

import { motion } from "framer-motion";

const steps = [
  {
    num: "01",
    title: "Capture",
    description:
      "Extract full reasoning chains from resolved prior auth cases as structured traces.",
  },
  {
    num: "02",
    title: "Abstract",
    description:
      "Strip PHI, generalize reasoning into patterns keyed by payer, CPT, ICD-10, and facility.",
  },
  {
    num: "03",
    title: "Store",
    description:
      "Persist in hybrid store with exact config hash matching and vector similarity fallback.",
  },
  {
    num: "04",
    title: "Inject",
    description:
      "Retrieve top-k matching patterns and prepend as institutional memory before agent calls.",
  },
];

export default function MethodSection() {
  return (
    <section className="mt-24 mb-16">
      {/* Section eyebrow */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.8, ease: [0.32, 0.72, 0, 1] }}
        className="flex items-center gap-2.5 mb-10"
      >
        <div className="w-1 h-1 rounded-full bg-accent/50" />
        <h2 className="text-[10px] uppercase tracking-[0.2em] font-medium text-text-secondary">
          Method
        </h2>
        <div className="flex-1 h-px bg-surface-border ml-2" />
      </motion.div>

      {/* Steps — staggered asymmetric grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {steps.map((step, i) => (
          <motion.div
            key={step.title}
            initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
            whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{
              duration: 0.7,
              delay: i * 0.1,
              ease: [0.32, 0.72, 0, 1],
            }}
            className="group rounded-3xl p-1.5 bezel-outer"
          >
            <div className="bezel-inner rounded-[calc(1.5rem-6px)] p-6 md:p-8 relative overflow-hidden transition-all duration-700 ease-spring group-hover:border-accent/[0.08]">
              {/* Large ghost number */}
              <span className="absolute -top-2 -right-1 text-[5rem] font-semibold leading-none text-surface-border/50 select-none transition-colors duration-700 ease-spring group-hover:text-accent/[0.06]">
                {step.num}
              </span>

              {/* Number pill */}
              <div className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-surface-hover border border-surface-border text-[10px] font-mono text-text-secondary mb-4">
                {step.num}
              </div>

              {/* Title */}
              <h3 className="text-lg font-semibold tracking-tight text-text-primary mb-2 transition-colors duration-500 ease-spring group-hover:text-accent">
                {step.title}
              </h3>

              {/* Description */}
              <p className="text-sm text-text-secondary leading-relaxed relative z-10 max-w-[45ch]">
                {step.description}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
