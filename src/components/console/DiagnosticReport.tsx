"use client";

import type { DiagnosticReport as Report } from "@/lib/api";

/**
 * Differential diagnosis report rendered at interview completion.
 *
 * Urgency drives color semantically: an emergency-tier result must be visually
 * unmistakable, so the mapping below is the only place urgency colors are assigned.
 */

function urgencyToken(level: string): { color: string; label: string } {
  const l = level.toLowerCase();
  if (l.includes("emergency")) return { color: "var(--emergency)", label: "Emergency" };
  if (l.includes("urgent")) return { color: "var(--urgent)", label: "Urgent" };
  if (l.includes("routine")) return { color: "var(--routine)", label: "Routine" };
  if (l.includes("self")) return { color: "var(--selfcare)", label: "Self-care" };
  return { color: "var(--muted-foreground)", label: level };
}

export function DiagnosticReportCard({ report }: { report: Report }) {
  return (
    <section className="trace-in mt-3 space-y-3" aria-label="Diagnostic report">
      <div className="panel p-4">
        <h3 className="eyebrow">Differential diagnosis</h3>

        <ol className="mt-3 space-y-3">
          {report.top_diagnoses.map((dx, i) => {
            const u = urgencyToken(dx.urgency_level);
            return (
              <li key={`${dx.diagnosis}-${i}`} className="rounded-md border p-3"
                  style={{ borderColor: "var(--border)" }}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="numeric text-xs text-muted-foreground">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <h4 className="font-medium" style={{ fontFamily: "var(--font-display)" }}>
                      {dx.diagnosis}
                    </h4>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className="rounded-sm border px-1.5 py-0.5 text-[10px] uppercase tracking-wider"
                      style={{ color: u.color, borderColor: u.color }}
                    >
                      {u.label}
                    </span>
                    <span className="numeric text-sm" style={{ color: "var(--vital)" }}>
                      {dx.estimated_probability}
                    </span>
                  </div>
                </div>

                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{dx.reasoning}</p>

                {dx.graph_evidence && (
                  <p
                    className="numeric mt-2 overflow-x-auto rounded-sm border px-2 py-1.5 text-[10px] whitespace-nowrap"
                    style={{ borderColor: "var(--border)", color: "var(--muted-foreground)" }}
                    title="Knowledge-graph traversal supporting this diagnosis"
                  >
                    {dx.graph_evidence}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      </div>

      <div className="panel p-4">
        <h3 className="eyebrow">Triage recommendation</h3>
        <p className="mt-2 text-sm leading-relaxed">{report.triage_recommendation}</p>

        {report.clinical_summary && (
          <>
            <h3 className="eyebrow mt-4">Clinical summary</h3>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              {report.clinical_summary}
            </p>
          </>
        )}
      </div>

      {/* Required safety framing: this is decision support, not a diagnosis. */}
      <p
        className="rounded-md border px-3 py-2 text-[11px] leading-relaxed"
        style={{ borderColor: "var(--urgent)", color: "var(--urgent)" }}
        role="note"
      >
        Decision support only. This report does not constitute a medical diagnosis and does
        not replace assessment by a qualified clinician. In an emergency, contact local
        emergency services immediately.
      </p>
    </section>
  );
}
