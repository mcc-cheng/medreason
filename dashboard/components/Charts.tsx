"use client";

import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { motion } from "framer-motion";
import { CompoundingPoint } from "@/lib/data";

const COLORS = {
  zeroShot: "#64748B",
  memory: "#34D399",
  grid: "rgba(255,255,255,0.03)",
  text: "#71717A",
  surface: "#131316",
  border: "rgba(255,255,255,0.04)",
};

const tooltipStyle = {
  backgroundColor: "#131316",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: "12px",
  color: "#FAFAFA",
  fontSize: "11px",
  fontFamily: "JetBrains Mono, monospace",
  boxShadow: "0 20px 40px -15px rgba(0,0,0,0.4)",
  padding: "10px 14px",
};

const legendStyle = {
  fontSize: "10px",
  color: COLORS.text,
  fontFamily: "JetBrains Mono, monospace",
};

function ChartShell({
  title,
  children,
  index = 0,
  span,
}: {
  title: string;
  children: React.ReactNode;
  index?: number;
  span?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{
        duration: 0.8,
        delay: 0.2 + index * 0.1,
        ease: [0.32, 0.72, 0, 1],
      }}
      className={`rounded-3xl p-1.5 bezel-outer ${span ?? ""}`}
    >
      <div className="bezel-inner rounded-[calc(1.5rem-6px)] p-5 md:p-6">
        <div className="flex items-center gap-2.5 mb-5">
          <div className="w-1 h-1 rounded-full bg-accent/50" />
          <h3 className="text-[10px] uppercase tracking-[0.15em] font-medium text-text-secondary">
            {title}
          </h3>
        </div>
        {children}
      </div>
    </motion.div>
  );
}

interface CompoundingChartProps {
  data: CompoundingPoint[];
}

export function CompoundingChart({ data }: CompoundingChartProps) {
  return (
    <ChartShell title="Compounding Accuracy Curve" index={1} span="lg:col-span-7">
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="memoryGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.memory} stopOpacity={0.15} />
              <stop offset="100%" stopColor={COLORS.memory} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="zsGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.zeroShot} stopOpacity={0.1} />
              <stop offset="100%" stopColor={COLORS.zeroShot} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} />
          <XAxis
            dataKey="case_number"
            stroke={COLORS.text}
            fontSize={10}
            fontFamily="JetBrains Mono"
            tickLine={false}
            axisLine={{ stroke: COLORS.border }}
            label={{
              value: "Cases Processed",
              position: "insideBottom",
              offset: -5,
              fill: COLORS.text,
              fontSize: 10,
              fontFamily: "JetBrains Mono",
            }}
          />
          <YAxis
            stroke={COLORS.text}
            fontSize={10}
            fontFamily="JetBrains Mono"
            domain={[0, 1]}
            tickLine={false}
            axisLine={{ stroke: COLORS.border }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
          />
          <Legend wrapperStyle={legendStyle} />
          <Area
            type="monotone"
            dataKey="zero_shot_accuracy"
            name="Zero-Shot"
            stroke={COLORS.zeroShot}
            fill="url(#zsGrad)"
            strokeWidth={1.5}
          />
          <Area
            type="monotone"
            dataKey="memory_accuracy"
            name="Memory-Augmented"
            stroke={COLORS.memory}
            fill="url(#memoryGrad)"
            strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

interface DifficultyChartProps {
  data: { difficulty: string; zero_shot: number; memory: number }[];
}

export function DifficultyChart({ data }: DifficultyChartProps) {
  return (
    <ChartShell title="Accuracy by Difficulty" index={2} span="lg:col-span-5">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} barGap={2} barSize={24}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} />
          <XAxis
            dataKey="difficulty"
            stroke={COLORS.text}
            fontSize={10}
            fontFamily="JetBrains Mono"
            tickLine={false}
            axisLine={{ stroke: COLORS.border }}
          />
          <YAxis
            stroke={COLORS.text}
            fontSize={10}
            fontFamily="JetBrains Mono"
            domain={[0, 1]}
            tickLine={false}
            axisLine={{ stroke: COLORS.border }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
          />
          <Legend wrapperStyle={legendStyle} />
          <Bar
            dataKey="zero_shot"
            name="Zero-Shot"
            fill={COLORS.zeroShot}
            radius={[6, 6, 0, 0]}
          />
          <Bar
            dataKey="memory"
            name="Memory"
            fill={COLORS.memory}
            radius={[6, 6, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

interface PayerChartProps {
  data: { payer: string; zero_shot: number; memory: number }[];
}

export function PayerChart({ data }: PayerChartProps) {
  return (
    <ChartShell title="Accuracy by Payer" index={0} span="lg:col-span-5">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical" barGap={2} barSize={14}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} />
          <XAxis
            type="number"
            stroke={COLORS.text}
            fontSize={10}
            fontFamily="JetBrains Mono"
            domain={[0, 1]}
            tickLine={false}
            axisLine={{ stroke: COLORS.border }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            type="category"
            dataKey="payer"
            stroke={COLORS.text}
            fontSize={10}
            fontFamily="JetBrains Mono"
            width={80}
            tickLine={false}
            axisLine={{ stroke: COLORS.border }}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
          />
          <Legend wrapperStyle={legendStyle} />
          <Bar
            dataKey="zero_shot"
            name="Zero-Shot"
            fill={COLORS.zeroShot}
            radius={[0, 4, 4, 0]}
          />
          <Bar
            dataKey="memory"
            name="Memory"
            fill={COLORS.memory}
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

interface ProcedureRadarProps {
  data: { procedure: string; zero_shot: number; memory: number }[];
}

export function ProcedureRadar({ data }: ProcedureRadarProps) {
  return (
    <ChartShell title="Procedure Coverage" index={3} span="lg:col-span-7">
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid stroke={COLORS.grid} />
          <PolarAngleAxis
            dataKey="procedure"
            stroke={COLORS.text}
            fontSize={9}
            fontFamily="JetBrains Mono"
          />
          <PolarRadiusAxis
            stroke={COLORS.text}
            fontSize={9}
            domain={[0, 1]}
            fontFamily="JetBrains Mono"
          />
          <Radar
            name="Zero-Shot"
            dataKey="zero_shot"
            stroke={COLORS.zeroShot}
            fill={COLORS.zeroShot}
            fillOpacity={0.12}
            strokeWidth={1.5}
          />
          <Radar
            name="Memory"
            dataKey="memory"
            stroke={COLORS.memory}
            fill={COLORS.memory}
            fillOpacity={0.12}
            strokeWidth={1.5}
          />
          <Legend wrapperStyle={legendStyle} />
        </RadarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
