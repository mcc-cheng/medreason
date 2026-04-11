"use client";

import { motion } from "framer-motion";
import Link from "next/link";

export default function Navbar() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: [0.32, 0.72, 0, 1] }}
      className="fixed top-5 left-1/2 -translate-x-1/2 z-40 glass-refract rounded-full px-6 py-2.5 flex items-center gap-6"
    >
      <Link href="/" className="flex items-center gap-2.5 group">
        <div className="w-2 h-2 rounded-full bg-accent animate-pulse-soft" />
        <span className="text-sm font-medium tracking-tight text-text-primary transition-colors duration-500 ease-spring">
          Veridicus
        </span>
      </Link>

      <div className="w-px h-4 bg-surface-border" />

      <div className="flex items-center gap-4">
        <Link
          href="/"
          className="text-xs text-text-secondary hover:text-text-primary transition-colors duration-500 ease-spring tracking-wide uppercase"
        >
          Benchmark
        </Link>
        <span className="text-xs text-text-tertiary font-mono">v0.1</span>
      </div>
    </motion.nav>
  );
}
