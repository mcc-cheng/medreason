import fs from "fs";
import path from "path";

const RESULTS_DIR = path.join(process.cwd(), "..", "results");

export interface CaseComparison {
  case_id: string;
  zero_shot?: {
    determination: string;
    correct: boolean;
    confidence: number;
    reasoning: string;
    tokens: number;
  };
  memory?: {
    determination: string;
    correct: boolean;
    confidence: number;
    reasoning: string;
    tokens: number;
    injected_patterns: string[];
  };
}

export interface CompoundingPoint {
  case_number: number;
  zero_shot_accuracy: number;
  memory_accuracy: number;
}

export interface Metrics {
  accuracy: number;
  count: number;
  correct?: number;
  avg_confidence: number;
  avg_tokens: number;
  avg_latency_ms: number;
}

export interface DashboardData {
  run_id: string;
  timestamp: string;
  metrics: {
    run_id: string;
    timestamp: string;
    total_cases: number;
    zero_shot: Metrics;
    memory: Metrics;
    delta_accuracy?: number;
    delta_pp?: number;
    patterns_created: number;
    patterns_reinforced: number;
  };
  cases: CaseComparison[];
  compounding_curve: CompoundingPoint[];
}

export function getDashboardData(): DashboardData | null {
  const filePath = path.join(RESULTS_DIR, "latest_dashboard.json");
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw);
}

export function getRunData(runId: string) {
  const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.startsWith("run_"));
  for (const file of files) {
    if (file.includes(runId)) {
      const raw = fs.readFileSync(path.join(RESULTS_DIR, file), "utf-8");
      return JSON.parse(raw);
    }
  }
  return null;
}

export function getCaseDetail(caseId: string): CaseComparison | null {
  const data = getDashboardData();
  if (!data) return null;
  return data.cases.find((c) => c.case_id === caseId) ?? null;
}

export function getBenchmarkCases() {
  const filePath = path.join(
    process.cwd(),
    "..",
    "benchmark",
    "benchmark_cases.json"
  );
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw);
}

// Demo data for when no results exist yet
export function getDemoData(): DashboardData {
  return {
    run_id: "demo",
    timestamp: new Date().toISOString(),
    metrics: {
      run_id: "demo",
      timestamp: new Date().toISOString(),
      total_cases: 30,
      zero_shot: {
        accuracy: 0.633,
        count: 30,
        correct: 19,
        avg_confidence: 0.72,
        avg_tokens: 1842,
        avg_latency_ms: 2340,
      },
      memory: {
        accuracy: 0.833,
        count: 30,
        correct: 25,
        avg_confidence: 0.85,
        avg_tokens: 2156,
        avg_latency_ms: 2780,
      },
      delta_accuracy: 0.2,
      delta_pp: 20.0,
      patterns_created: 25,
      patterns_reinforced: 47,
    },
    cases: Array.from({ length: 30 }, (_, i) => ({
      case_id: `demo-${String(i + 1).padStart(3, "0")}`,
      zero_shot: {
        determination: i % 3 === 0 ? "denied" : "approved",
        correct: Math.random() > 0.37,
        confidence: 0.6 + Math.random() * 0.3,
        reasoning: "Demo reasoning chain...",
        tokens: 1500 + Math.floor(Math.random() * 800),
      },
      memory: {
        determination: i % 4 === 0 ? "denied" : "approved",
        correct: Math.random() > 0.17,
        confidence: 0.75 + Math.random() * 0.2,
        reasoning: "Demo reasoning with memory...",
        tokens: 1800 + Math.floor(Math.random() * 800),
        injected_patterns: ["p1", "p2"],
      },
    })),
    compounding_curve: Array.from({ length: 30 }, (_, i) => ({
      case_number: i + 1,
      zero_shot_accuracy: 0.55 + Math.random() * 0.15,
      memory_accuracy: 0.6 + (i / 30) * 0.25 + Math.random() * 0.05,
    })),
  };
}
