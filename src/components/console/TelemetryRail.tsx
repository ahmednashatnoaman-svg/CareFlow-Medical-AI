"use client";

import type { SocratesTracker, TriageResponse } from "@/lib/api";

/**
 * Live instrument readout for the triage interview.
 *
 * Every value here is read from the API response. The previous implementation displayed
 * hardcoded strings ("1.82", "0.14", a fixed 70% bar) that never changed, which presented
 * invented telemetry as if it were measured. Where a value is genuinely unavailable this
 * renders an em-dash rather than inventing one.
 */

const SOCRATES_AXES: { key: keyof SocratesTracker; label: string; short: string }[] = [
  { key: "site", label: "Site", short: "S" },
  { key: "onset", label: "Onset", short: "O" },
  { key: "character", label: "Character", short: "C" },
  { key: "radiation", label: "Radiation", short: "R" },
  { key: "associated_symptoms", label: "Associated symptoms", short: "A" },
  { key: "time_course", label: "Time course", short: "T" },
  { key: "exacerbating_relieving", label: "Exacerbating / relieving", short: "E" },
  { key: "severity", label: "Severity", short: "S" },
];

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="numeric text-sm text-foreground" title={hint}>
        {value}
      </span>
    </div>
  );
}

export function TelemetryRail({ state }: { state: TriageResponse | null }) {
  const tracker = state?.socrates_tracker ?? {};
  const covered = SOCRATES_AXES.filter((a) => tracker[a.key]).length;
  const turns = state?.turn_count ?? 0;
  const maxTurns = state?.max_turns ?? 8;

  return (
    <aside className="flex flex-col gap-4" aria-label="Interview telemetry">
      {/* SOCRATES coverage — the memorable anchor: eight discrete instrument segments
          that fill as the interview establishes each axis of the pain history. */}
      <section className="panel p-4">
        <div className="flex items-center justify-between">
          <h2 className="eyebrow">SOCRATES coverage</h2>
          <span className="numeric text-xs text-vital">
            {covered}/{SOCRATES_AXES.length}
          </span>
        </div>

        <div className="mt-3 grid grid-cols-8 gap-1" role="img"
             aria-label={`${covered} of ${SOCRATES_AXES.length} SOCRATES axes established`}>
          {SOCRATES_AXES.map((axis) => {
            const on = Boolean(tracker[axis.key]);
            return (
              <div key={axis.key} className="group relative" title={`${axis.label}: ${on ? "established" : "pending"}`}>
                <div
                  className="h-8 rounded-sm border transition-colors duration-300"
                  style={{
                    background: on ? "var(--vital)" : "var(--muted)",
                    borderColor: on ? "var(--vital)" : "var(--border)",
                    opacity: on ? 1 : 0.5,
                  }}
                />
                <div
                  className="numeric mt-1 text-center text-[9px]"
                  style={{ color: on ? "var(--vital)" : "var(--muted-foreground)" }}
                >
                  {axis.short}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Interview progress */}
      <section className="panel p-4">
        <h2 className="eyebrow">Interview state</h2>
        <div className="mt-2 divide-y divide-[var(--border)]">
          <Metric label="Turn" value={`${turns} / ${maxTurns}`} />
          <Metric
            label="Target symptom"
            value={state?.target_symptom ?? "—"}
            hint="Symptom the graph engine is currently evaluating"
          />
          <Metric
            label="Stop reason"
            value={state?.stop_reason ?? "—"}
            hint="Populated by the termination engine when the interview ends"
          />
        </div>
      </section>

      {/* Confirmed and ruled-out findings, straight from the extraction step. */}
      <section className="panel flex-1 p-4">
        <h2 className="eyebrow">Clinical findings</h2>

        <div className="mt-3 space-y-3">
          <div>
            <div className="mb-1.5 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--vital)" }} />
              <span className="text-[11px] text-muted-foreground">
                Confirmed ({state?.positive_symptoms.length ?? 0})
              </span>
            </div>
            {state?.positive_symptoms.length ? (
              <ul className="flex flex-wrap gap-1">
                {state.positive_symptoms.map((s) => (
                  <li
                    key={s}
                    className="rounded-sm border px-1.5 py-0.5 text-[11px]"
                    style={{ borderColor: "var(--vital-dim)", color: "var(--vital)" }}
                  >
                    {s}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[11px] text-muted-foreground">None recorded yet</p>
            )}
          </div>

          <div>
            <div className="mb-1.5 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--border-strong)" }} />
              <span className="text-[11px] text-muted-foreground">
                Ruled out ({state?.negated_symptoms.length ?? 0})
              </span>
            </div>
            {state?.negated_symptoms.length ? (
              <ul className="flex flex-wrap gap-1">
                {state.negated_symptoms.map((s) => (
                  <li key={s} className="rounded-sm border px-1.5 py-0.5 text-[11px] text-muted-foreground line-through">
                    {s}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[11px] text-muted-foreground">None recorded yet</p>
            )}
          </div>
        </div>
      </section>
    </aside>
  );
}
