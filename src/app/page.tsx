"use client";

import { useState } from "react";
import { EvaluationDashboard } from "@/components/console/EvaluationDashboard";
import { GuidelinesConsole } from "@/components/console/GuidelinesConsole";
import { SystemStatus } from "@/components/console/SystemStatus";
import { TelemetryRail } from "@/components/console/TelemetryRail";
import { TriageConsole } from "@/components/console/TriageConsole";
import type { Language, TriageResponse } from "@/lib/api";

type Mode = "triage" | "guidelines" | "evaluation";

const MODES: { id: Mode; label: string; sub: string }[] = [
  { id: "triage", label: "Triage", sub: "Graph RAG" },
  { id: "guidelines", label: "Guidelines", sub: "Vector RAG" },
  { id: "evaluation", label: "Evaluation", sub: "Metrics" },
];

export default function Home() {
  const [mode, setMode] = useState<Mode>("triage");
  const [language, setLanguage] = useState<Language>("en");
  // Lifted so the telemetry rail reflects the live interview rather than static text.
  const [triageState, setTriageState] = useState<TriageResponse | null>(null);

  return (
    <div className="relative flex min-h-screen flex-col">
      {/* Graph-paper ground: the recognizable signature of the instrument aesthetic. */}
      <div className="gridpaper pointer-events-none fixed inset-0 -z-10" aria-hidden />

      <header
        className="sticky top-0 z-10 border-b px-5 py-3 backdrop-blur-md"
        style={{ borderColor: "var(--border)", background: "color-mix(in oklab, var(--background) 85%, transparent)" }}
      >
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4">
          <div className="flex items-baseline gap-3">
            <h1 className="text-lg font-bold tracking-tight" style={{ fontFamily: "var(--font-display)" }}>
              Care<span style={{ color: "var(--vital)" }}>Flow</span>
            </h1>
            <p className="eyebrow hidden sm:block">Dual-mode clinical AI</p>
          </div>

          <nav className="flex gap-1" role="tablist" aria-label="Mode">
            {MODES.map((m) => {
              const active = mode === m.id;
              return (
                <button
                  key={m.id}
                  role="tab"
                  aria-selected={active}
                  onClick={() => setMode(m.id)}
                  className="rounded-md border px-3 py-1.5 text-left transition-colors"
                  style={{
                    borderColor: active ? "var(--vital)" : "var(--border)",
                    background: active ? "color-mix(in oklab, var(--vital) 12%, transparent)" : "transparent",
                  }}
                >
                  <span
                    className="block text-xs font-medium"
                    style={{ color: active ? "var(--vital)" : "var(--foreground)" }}
                  >
                    {m.label}
                  </span>
                  <span className="eyebrow block">{m.sub}</span>
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-2">
            <div className="flex rounded-md border" style={{ borderColor: "var(--border-strong)" }}>
              {(["en", "ar"] as Language[]).map((l) => (
                <button
                  key={l}
                  onClick={() => setLanguage(l)}
                  aria-pressed={language === l}
                  className="numeric px-2.5 py-1.5 text-[11px] uppercase transition-colors"
                  style={{
                    color: language === l ? "var(--vital)" : "var(--muted-foreground)",
                    background: language === l ? "color-mix(in oklab, var(--vital) 12%, transparent)" : "transparent",
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
            <SystemStatus />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] flex-1 px-5 py-5">
        {mode === "evaluation" ? (
          <EvaluationDashboard />
        ) : (
          <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
            <div className="h-[calc(100vh-9rem)] min-h-[520px]">
              {mode === "triage" ? (
                <TriageConsole language={language} onStateChange={setTriageState} />
              ) : (
                <GuidelinesConsole language={language} />
              )}
            </div>

            {/* The rail is meaningful only for the stateful triage interview. */}
            {mode === "triage" ? (
              <TelemetryRail state={triageState} />
            ) : (
              <aside className="panel p-4">
                <h2 className="eyebrow">Retrieval mode</h2>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                  Questions are embedded and matched against WHO guideline chunks in Qdrant.
                  Each answer is generated only from the passages returned, and those
                  passages are attached to every response so any claim can be traced back
                  to its source.
                </p>
              </aside>
            )}
          </div>
        )}
      </main>

      <footer className="border-t px-5 py-3" style={{ borderColor: "var(--border)" }}>
        <p className="mx-auto max-w-[1400px] text-[11px] text-muted-foreground">
          Clinical decision support for demonstration purposes. Not a medical device and
          not a substitute for professional care.
        </p>
      </footer>
    </div>
  );
}
