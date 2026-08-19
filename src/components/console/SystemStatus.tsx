"use client";

import { useEffect, useState } from "react";
import { health, type Readiness } from "@/lib/api";

/**
 * Live dependency indicator driven by /api/v1/health/ready.
 *
 * This is the UI counterpart to removing the silent random-vector fallback: when the
 * embedder or vector store is down, that is now visible in the header instead of being
 * masked by plausible-looking answers.
 */
const TONE: Record<string, string> = {
  ok: "var(--vital)",
  degraded: "var(--urgent)",
  unavailable: "var(--emergency)",
  unknown: "var(--muted-foreground)",
};

export function SystemStatus() {
  const [state, setState] = useState<Readiness | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      health
        .ready()
        .then((r) => alive && (setState(r), setFailed(false)))
        .catch(() => alive && setFailed(true));
    void poll();
    const id = setInterval(poll, 30_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const status = failed ? "unavailable" : (state?.status ?? "unknown");
  const color = TONE[status];

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label={`System status: ${status}`}
        className="flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-[11px] transition-colors hover:bg-[var(--surface-raised)]"
        style={{ borderColor: "var(--border-strong)" }}
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${status === "ok" ? "pulse-vital" : ""}`}
          style={{ background: color }}
        />
        <span style={{ color }}>{status}</span>
      </button>

      {open && (
        <div
          className="panel trace-in absolute right-0 z-20 mt-2 w-64 p-3"
          style={{ background: "var(--surface-raised)" }}
        >
          <h3 className="eyebrow">Dependencies</h3>
          {failed || !state ? (
            <p className="mt-2 text-[11px]" style={{ color: "var(--emergency)" }}>
              Backend unreachable. Start it with{" "}
              <code className="numeric">npm run dev:api</code>.
            </p>
          ) : (
            <>
              <ul className="mt-2 space-y-1.5">
                {Object.entries(state.checks).map(([name, check]) => (
                  <li key={name} className="flex items-center justify-between gap-2 text-[11px]">
                    <span className="text-muted-foreground">{name.replace(/_/g, " ")}</span>
                    <span className="flex items-center gap-1.5" style={{ color: TONE[check.status] }}>
                      <span className="h-1 w-1 rounded-full" style={{ background: TONE[check.status] }} />
                      {check.status}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="numeric mt-2 border-t pt-2 text-[10px] text-muted-foreground"
                 style={{ borderColor: "var(--border)" }}>
                v{state.version} · {state.environment}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
