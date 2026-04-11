import {
  getCaseDetail,
  getBenchmarkCases,
} from "@/lib/data";
import Link from "next/link";
import CaseDetailClient from "@/components/CaseDetailClient";

export const dynamic = "force-dynamic";

export default function CaseDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const caseId = params.id;
  const comparison = getCaseDetail(caseId);
  const allCases = getBenchmarkCases();
  const benchmarkCase = allCases.find(
    (c: { case_id: string }) => c.case_id === caseId
  );

  if (!comparison) {
    return (
      <main className="min-h-[100dvh] px-4 md:px-6 pt-28 pb-16 max-w-[1100px] mx-auto">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-xs text-text-secondary hover:text-accent transition-colors duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] mb-10 group"
        >
          <span className="transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:-translate-x-0.5">
            &larr;
          </span>
          <span>Back to Benchmark</span>
        </Link>

        <h1 className="text-3xl font-semibold tracking-tighter text-text-primary mb-4">
          Case {caseId}
        </h1>
        <p className="text-text-secondary text-sm">
          Case not found. Run a benchmark first.
        </p>
      </main>
    );
  }

  return (
    <CaseDetailClient
      caseId={caseId}
      comparison={comparison}
      benchmarkCase={benchmarkCase}
    />
  );
}
