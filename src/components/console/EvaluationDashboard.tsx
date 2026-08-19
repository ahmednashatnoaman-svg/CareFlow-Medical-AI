"use client";

import { useEffect, useState } from "react";
import { ApiError, evaluation, type EvaluationResults } from "@/lib/api";

/**
 * Quantitative RAG evaluation, split by pipeline stage.
 *
 * The previous version hardcoded fallbacks (`?? 0.92`, `?? 0.98`, and two invented
 * "data-driven improvement" entries) that rendered whenever the fetch failed — and the
 * fetch always failed, because the endpoint had a broken path. The dashboard therefore
 * displayed invented metrics as measured results.
 *
 * This version has no fallback values anywhere. A missing metric renders an em-dash and
 * a missing run renders instructions for producing one.
 */

/** Retrieval and generation are scored separately so each stage can be validated alone. */
const RETRIEVAL_METRICS = [
  { key: "context_precision", label: "Context precision", hint: "Share of retrieved chunks that are relevant" },
  { key: "context_recall", label: "Context recall", hint: "Share of needed evidence actually retrieved" },
] as const;

const GENERATION_METRICS = [
  { key: "faithfulness", label: "Faithfulness", hint: "Answer claims entailed by retrieved context (inverse hallucination)" },
  { key: "answer_relevancy", label: "Answer relevancy", hint: "How directly the answer addresses the question" },
  { key: "answer_similarity", label: "Answer similarity", hint: "Semantic agreement with ground truth" },
] as const;

function MetricBar({ label, value, hint }: { label: string; value?: number; hint: string }) {
  const measured = typeof value === "number" && Number.isFinite(value);
  return (
    <div className="py-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs" title={hint}>
          {label}
        </span>
        <span className="numeric text-sm" style={{ color: measured ? "var(--vital)" : "var(--muted-foreground)" }}>
          {measured ? value.toFixed(3) : "—"}
        </span>
      </div>
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full" style={{ background: "var(--muted)" }}>
        {measured && (
          <div
            className="h-full rounded-full transition-[width] duration-700"
            style={{ width: `${Math.min(Math.max(value, 0), 1) * 100}%`, background: "var(--vital)" }}
          />
        )}
      </div>
    </div>
  );
}

function Section({
  title,
  subtitle,
  metrics,
  averages,
}: {
  title: string;
  subtitle: string;
  metrics: readonly { key: string; label: string; hint: string }[];
  averages: Record<string, number | undefined>;
}) {
  return (
    <section className="panel p-4">
      <h3 className="text-sm font-medium" style={{ fontFamily: "var(--font-display)" }}>
        {title}
      </h3>
      <p className="mt-0.5 text-[11px] text-muted-foreground">{subtitle}</p>
      <div className="mt-2 divide-y" style={{ borderColor: "var(--border)" }}>
        {metrics.map((m) => (
          <MetricBar key={m.key} label={m.label} value={averages[m.key]} hint={m.hint} />
        ))}
      </div>
    </section>
  );
}

export function EvaluationDashboard() {
  const [data, setData] = useState<EvaluationResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    evaluation
      .get()
      .then((d) => alive && setData(d))
      .catch((e) => {
        if (!alive) return;
        // 404 means "no run yet" — a distinct, actionable state, not an error.
        if (e instanceof ApiError && e.status === 404) setMissing(true);
        else setError(e instanceof ApiError ? e.message : "Failed to load evaluation results.");
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="panel flex h-full items-center justify-center p-8">
        <p className="text-xs text-muted-foreground">Loading evaluation results…</p>
      </div>
    );
  }

  if (missing) {
    return (
      <div className="panel flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <h3 className="text-sm font-medium" style={{ fontFamily: "var(--font-display)" }}>
          No evaluation run recorded
        </h3>
        <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
          Metrics are shown only when they have actually been measured. Generate a run:
        </p>
        <code
          className="numeric rounded-md border px-3 py-1.5 text-[11px]"
          style={{ borderColor: "var(--border-strong)", color: "var(--vital)" }}
        >
          python scripts/evaluate_rag.py
        </code>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="panel flex h-full items-center justify-center p-8">
        <p className="text-xs" style={{ color: "var(--emergency)" }}>
          {error ?? "No data."}
        </p>
      </div>
    );
  }

  const avg = data.averages ?? {};
  const rows = data.detailed_results ?? [];
  const latency = avg.latency_sec;

  return (
    <div className="space-y-4 overflow-y-auto pr-1">
      <div className="grid gap-4 md:grid-cols-2">
        <Section
          title="Retrieval"
          subtitle="Did the vector search surface the right guideline passages?"
          metrics={RETRIEVAL_METRICS}
          averages={avg as Record<string, number | undefined>}
        />
        <Section
          title="Generation"
          subtitle="Given those passages, was the answer faithful and on-topic?"
          metrics={GENERATION_METRICS}
          averages={avg as Record<string, number | undefined>}
        />
      </div>

      <section className="panel p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h3 className="text-sm font-medium" style={{ fontFamily: "var(--font-display)" }}>
            Run detail
          </h3>
          <div className="flex gap-4 text-[11px] text-muted-foreground">
            <span>
              samples <span className="numeric text-foreground">{rows.length}</span>
            </span>
            <span>
              mean latency{" "}
              <span className="numeric text-foreground">
                {typeof latency === "number" ? `${latency.toFixed(2)}s` : "—"}
              </span>
            </span>
            {data.metadata?.generated_at && (
              <span>
                generated <span className="numeric text-foreground">{data.metadata.generated_at}</span>
              </span>
            )}
          </div>
        </div>

        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-[11px]">
            <thead>
              <tr className="eyebrow" style={{ borderBottom: "1px solid var(--border)" }}>
                <th className="pb-2 pr-3 font-normal">Question</th>
                <th className="pb-2 pr-3 text-right font-normal">Faith.</th>
                <th className="pb-2 pr-3 text-right font-normal">Relev.</th>
                <th className="pb-2 pr-3 text-right font-normal">Ctx prec.</th>
                <th className="pb-2 pr-3 text-right font-normal">Ctx rec.</th>
                <th className="pb-2 text-right font-normal">Latency</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td className="max-w-[280px] truncate py-1.5 pr-3" title={r.question}>
                    {r.question}
                  </td>
                  {(["faithfulness", "answer_relevancy", "context_precision", "context_recall"] as const).map(
                    (k) => (
                      <td key={k} className="numeric py-1.5 pr-3 text-right">
                        {typeof r[k] === "number" ? r[k]!.toFixed(2) : "—"}
                      </td>
                    ),
                  )}
                  <td className="numeric py-1.5 text-right">
                    {typeof r.latency_sec === "number" ? `${r.latency_sec.toFixed(2)}s` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
