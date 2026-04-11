import { getDashboardData, getDemoData } from "@/lib/data";
import StatCard from "@/components/StatCard";
import {
  CompoundingChart,
  DifficultyChart,
  PayerChart,
  ProcedureRadar,
} from "@/components/Charts";
import MethodSection from "@/components/MethodSection";
import CaseTable from "@/components/CaseTable";
import HeroSection from "@/components/HeroSection";

const DEMO_DIFFICULTY = [
  { difficulty: "Easy", zero_shot: 0.82, memory: 0.95 },
  { difficulty: "Medium", zero_shot: 0.6, memory: 0.8 },
  { difficulty: "Hard", zero_shot: 0.42, memory: 0.68 },
];

const DEMO_PAYERS = [
  { payer: "Aetna", zero_shot: 0.65, memory: 0.85 },
  { payer: "UHC", zero_shot: 0.6, memory: 0.82 },
  { payer: "BCBS", zero_shot: 0.58, memory: 0.8 },
  { payer: "Cigna", zero_shot: 0.62, memory: 0.84 },
  { payer: "Medicare", zero_shot: 0.7, memory: 0.88 },
  { payer: "Humana", zero_shot: 0.55, memory: 0.78 },
  { payer: "Medicaid", zero_shot: 0.5, memory: 0.75 },
];

const DEMO_PROCEDURES = [
  { procedure: "99213", zero_shot: 0.8, memory: 0.95 },
  { procedure: "99214", zero_shot: 0.55, memory: 0.75 },
  { procedure: "27447", zero_shot: 0.5, memory: 0.8 },
  { procedure: "29881", zero_shot: 0.6, memory: 0.85 },
  { procedure: "72148", zero_shot: 0.45, memory: 0.7 },
  { procedure: "70553", zero_shot: 0.7, memory: 0.9 },
  { procedure: "64483", zero_shot: 0.4, memory: 0.65 },
  { procedure: "90837", zero_shot: 0.65, memory: 0.85 },
];

export const dynamic = "force-dynamic";

export default function Home() {
  const data = getDashboardData() ?? getDemoData();
  const { metrics } = data;
  const isDemo = data.run_id === "demo";

  return (
    <main className="min-h-[100dvh] px-4 md:px-6 pt-28 pb-16 max-w-[1400px] mx-auto">
      {/* Hero — Left-aligned asymmetric */}
      <HeroSection isDemo={isDemo} />

      {/* Stat Cards — Asymmetric bento: 2fr 1fr */}
      <section className="grid grid-cols-1 md:grid-cols-12 gap-3 mb-16">
        <div className="md:col-span-5">
          <StatCard
            label="Zero-Shot Accuracy"
            value={`${(metrics.zero_shot.accuracy * 100).toFixed(1)}%`}
            sub={`${metrics.zero_shot.correct ?? 0}/${metrics.zero_shot.count} correct`}
            index={0}
          />
        </div>
        <div className="md:col-span-4">
          <StatCard
            label="Memory-Augmented"
            value={`${(metrics.memory.accuracy * 100).toFixed(1)}%`}
            sub={`${metrics.memory.correct ?? 0}/${metrics.memory.count} correct`}
            accent
            index={1}
          />
        </div>
        <div className="md:col-span-3">
          <StatCard
            label="Advantage"
            value={`+${metrics.delta_pp?.toFixed(1) ?? "0"}pp`}
            sub={`${metrics.patterns_created} patterns`}
            delta
            index={2}
          />
        </div>
      </section>

      {/* Charts — Asymmetric 5/7 grid */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-3 mb-16">
        <PayerChart data={DEMO_PAYERS} />
        <CompoundingChart data={data.compounding_curve} />
        <DifficultyChart data={DEMO_DIFFICULTY} />
        <ProcedureRadar data={DEMO_PROCEDURES} />
      </section>

      {/* Method */}
      <MethodSection />

      {/* Case Table */}
      <CaseTable cases={data.cases} />

      {/* Footer */}
      <footer className="mt-32 pt-8 border-t border-surface-border">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <p className="text-xs text-text-tertiary tracking-wide">
            Veridicus v0.1.0
          </p>
          <p className="text-xs text-text-tertiary font-mono">
            {metrics.total_cases} cases / {metrics.patterns_created} patterns /
            run {metrics.run_id}
          </p>
        </div>
      </footer>
    </main>
  );
}
