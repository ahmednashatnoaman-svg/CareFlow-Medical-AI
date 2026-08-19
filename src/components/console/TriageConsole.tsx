"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, triage, type Language, type TriageResponse } from "@/lib/api";
import { DiagnosticReportCard } from "./DiagnosticReport";
import {
  ErrorNotice,
  MessageBubble,
  TypingIndicator,
  useAutoScroll,
  type Message,
} from "./ChatPrimitives";

/**
 * Mode 1 — multi-turn Graph RAG triage interview.
 *
 * The backend is stateful per `session_id`, so this component owns exactly one session
 * and threads it through every call. It lifts each response to the parent so the
 * telemetry rail renders measured values instead of placeholders.
 */
export function TriageConsole({
  language,
  onStateChange,
}: {
  language: Language;
  onStateChange: (s: TriageResponse | null) => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  // Starts true: the component opens a session on mount, so it is busy from first paint.
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<string[]>([]);
  const [complete, setComplete] = useState(false);
  const sessionRef = useRef<string>("");
  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useAutoScroll(messages.length + (busy ? 1 : 0));

  const apply = useCallback(
    (data: TriageResponse, extraSlot?: React.ReactNode) => {
      onStateChange(data);
      setOptions(data.is_complete ? [] : data.options);
      setComplete(data.is_complete);
      setMessages((m) => [
        ...m,
        {
          id: `a-${Date.now()}-${m.length}`,
          role: "assistant",
          content: data.message,
          slot:
            extraSlot ??
            (data.diagnostic_report ? <DiagnosticReportCard report={data.diagnostic_report} /> : undefined),
        },
      ]);
    },
    [onStateChange],
  );

  /* Opens a session and renders the greeting. Contains no synchronous setState so it is
     safe to call straight from an effect -- every state write happens after an await. */
  const openSession = useCallback(async () => {
    // crypto.randomUUID is available in every browser Next 16 targets.
    const id = crypto.randomUUID();
    sessionRef.current = id;
    try {
      const data = await triage.start(id, language);
      // Ignore a response whose session was superseded by a newer start (fast language
      // toggling), otherwise a stale greeting could overwrite the current transcript.
      if (sessionRef.current !== id) return;
      apply(data);
    } catch (e) {
      if (sessionRef.current !== id) return;
      setError(e instanceof ApiError ? e.message : "Failed to start the triage session.");
    } finally {
      if (sessionRef.current === id) setBusy(false);
    }
  }, [language, apply]);

  /* Restart on language change: the greeting and option text come from the backend in the
     selected language, so continuing would leave the transcript bilingual. */
  useEffect(() => {
    // react-hooks/set-state-in-effect traces the call graph into openSession and flags the
    // setState calls inside it. Those all run in an async continuation *after* an await,
    // not synchronously in the effect body, so they cannot cascade renders during commit --
    // this is the rule's known false positive for fetch-on-mount. Opening the session is a
    // genuine external-system synchronization, which is what effects are for.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void openSession();
  }, [openSession]);

  /* Explicit user action, so resetting state synchronously here is correct. */
  const start = useCallback(() => {
    setBusy(true);
    setError(null);
    setMessages([]);
    setOptions([]);
    setComplete(false);
    void openSession();
  }, [openSession]);

  const send = useCallback(
    async (text: string, displayAs?: string) => {
      if (!text.trim() || busy || complete) return;
      setError(null);
      setBusy(true);
      setOptions([]);
      setMessages((m) => [
        ...m,
        { id: `u-${Date.now()}`, role: "user", content: displayAs ?? text },
      ]);
      try {
        apply(await triage.step(sessionRef.current, text, language));
      } catch (e) {
        setError(
          e instanceof ApiError
            ? `${e.message}${e.status ? ` (HTTP ${e.status})` : ""}`
            : "The triage engine did not respond.",
        );
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [busy, complete, language, apply],
  );

  return (
    <div className="panel flex h-full min-h-0 flex-col">
      <header className="flex items-center justify-between gap-3 border-b px-4 py-3"
              style={{ borderColor: "var(--border)" }}>
        <div>
          <h2 className="text-sm font-medium" style={{ fontFamily: "var(--font-display)" }}>
            Diagnostic triage interview
          </h2>
          <p className="text-[11px] text-muted-foreground">
            PrimeKG graph traversal with SOCRATES history taking
          </p>
        </div>
        <button
          onClick={start}
          disabled={busy}
          className="rounded-md border px-2.5 py-1 text-[11px] transition-colors hover:bg-[var(--surface-raised)] disabled:opacity-40"
          style={{ borderColor: "var(--border-strong)" }}
        >
          New session
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {busy && <TypingIndicator label="Traversing knowledge graph…" />}
        {error && <ErrorNotice error={error} onRetry={start} />}
        <div ref={bottomRef} />
      </div>

      {/* Numbered options map to the backend's "1"|"2"|"3" selection protocol, but the
          transcript shows the human-readable option text the patient actually chose. */}
      {options.length > 0 && !busy && (
        <div className="flex flex-wrap gap-2 border-t px-4 py-3" style={{ borderColor: "var(--border)" }}>
          {options.map((opt, i) => (
            <button
              key={`${opt}-${i}`}
              onClick={() => void send(String(i + 1), opt)}
              className="trace-in flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs transition-colors hover:border-[var(--vital)] hover:text-[var(--vital)]"
              style={{ borderColor: "var(--border-strong)" }}
            >
              <span className="numeric text-[10px] text-muted-foreground">{i + 1}</span>
              {opt}
            </button>
          ))}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          const v = input;
          setInput("");
          void send(v);
        }}
        className="flex gap-2 border-t px-4 py-3"
        style={{ borderColor: "var(--border)" }}
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy || complete}
          dir={language === "ar" ? "rtl" : "ltr"}
          placeholder={
            complete
              ? "Interview complete — start a new session"
              : language === "ar"
                ? "اكتب الأعراض التي تشعر بها…"
                : "Describe your symptom, or pick an option above…"
          }
          aria-label="Your response"
          className="min-w-0 flex-1 rounded-md border bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground disabled:opacity-50"
          style={{ borderColor: "var(--border-strong)" }}
        />
        <button
          type="submit"
          disabled={busy || complete || !input.trim()}
          className="rounded-md px-4 py-2 text-sm font-medium transition-opacity disabled:opacity-30"
          style={{ background: "var(--vital)", color: "var(--background)" }}
        >
          Send
        </button>
      </form>
    </div>
  );
}
