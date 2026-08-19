"use client";

import { useCallback, useRef, useState } from "react";
import { ApiError, dialogue, type ChatTurn, type Citation, type Language } from "@/lib/api";
import {
  ErrorNotice,
  MessageBubble,
  TypingIndicator,
  useAutoScroll,
  type Message,
} from "./ChatPrimitives";

/**
 * Mode 2 — WHO guideline retrieval.
 *
 * Every answer ships with the chunks it was grounded in. Citations are rendered inline
 * and always expandable, because an ungrounded medical claim and a grounded one must not
 * look identical: `chunks_retrieved === 0` is called out explicitly rather than silently
 * presenting an unsourced answer.
 */

function CitationList({ sources, retrieved }: { sources: Citation[]; retrieved: number }) {
  const [open, setOpen] = useState(false);

  if (retrieved === 0) {
    return (
      <p
        className="mt-2 rounded-md border px-2.5 py-1.5 text-[11px]"
        style={{ borderColor: "var(--urgent)", color: "var(--urgent)" }}
        role="note"
      >
        No guideline passages matched this query — the answer above is not grounded in
        retrieved sources. Treat it with caution.
      </p>
    );
  }

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-1.5 text-[11px] transition-colors hover:text-[var(--vital)]"
        style={{ color: "var(--muted-foreground)" }}
      >
        <span
          aria-hidden
          className="inline-block transition-transform"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        >
          ▸
        </span>
        {sources.length} grounding {sources.length === 1 ? "source" : "sources"}
      </button>

      {open && (
        <ol className="trace-in mt-2 space-y-1.5">
          {sources.map((s, i) => (
            <li
              key={`${s.source_file}-${i}`}
              className="rounded-md border p-2.5"
              style={{ borderColor: "var(--border)", background: "var(--background)" }}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[11px] font-medium">{s.source_file}</span>
                <span
                  className="numeric text-[10px]"
                  style={{ color: "var(--vital)" }}
                  title="Cosine similarity between the query and this chunk"
                >
                  {s.relevance_score.toFixed(3)}
                </span>
              </div>
              <p className="eyebrow mt-0.5">{s.section}</p>
              <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground">{s.snippet}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

const SUGGESTIONS = [
  "What are the WHO thresholds for diagnosing hypertension?",
  "First-line pharmacological treatment for adult hypertension?",
  "When should antihypertensive therapy be initiated?",
];

export function GuidelinesConsole({ language }: { language: Language }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Sent to the backend for follow-up context; kept separate from `messages`, which
  // carries React nodes that must never be serialized into a request body.
  const historyRef = useRef<ChatTurn[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useAutoScroll(messages.length + (busy ? 1 : 0));

  const ask = useCallback(
    async (query: string) => {
      if (!query.trim() || busy) return;
      setError(null);
      setBusy(true);
      setMessages((m) => [...m, { id: `u-${Date.now()}`, role: "user", content: query }]);

      try {
        const data = await dialogue.chat(query, historyRef.current);
        const appended: ChatTurn[] = [
          ...historyRef.current,
          { role: "user", content: query },
          { role: "assistant", content: data.answer },
        ];
        // Bound the payload; the backend only reads the last few turns.
        historyRef.current = appended.slice(-8);
        setMessages((m) => [
          ...m,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: data.answer,
            slot: <CitationList sources={data.sources} retrieved={data.chunks_retrieved} />,
          },
        ]);
      } catch (e) {
        setError(
          e instanceof ApiError
            ? `${e.message}${e.status ? ` (HTTP ${e.status})` : ""}`
            : "Guideline retrieval failed.",
        );
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [busy],
  );

  return (
    <div className="panel flex h-full min-h-0 flex-col">
      <header className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
        <h2 className="text-sm font-medium" style={{ fontFamily: "var(--font-display)" }}>
          WHO guideline assistant
        </h2>
        <p className="text-[11px] text-muted-foreground">
          Qdrant vector retrieval over official guideline documents — every answer cited
        </p>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !busy && (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <p className="max-w-sm text-xs text-muted-foreground">
              Ask a clinical question. Answers are generated strictly from retrieved
              guideline passages, and the passages are shown alongside every response.
            </p>
            <div className="flex flex-col gap-1.5">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => void ask(s)}
                  className="rounded-md border px-3 py-1.5 text-left text-[11px] transition-colors hover:border-[var(--vital)] hover:text-[var(--vital)]"
                  style={{ borderColor: "var(--border)" }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {busy && <TypingIndicator label="Retrieving guideline passages…" />}
        {error && <ErrorNotice error={error} />}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          const v = input;
          setInput("");
          void ask(v);
        }}
        className="flex gap-2 border-t px-4 py-3"
        style={{ borderColor: "var(--border)" }}
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          dir={language === "ar" ? "rtl" : "ltr"}
          placeholder={
            language === "ar" ? "اسأل عن إرشادات منظمة الصحة العالمية…" : "Ask about a clinical guideline…"
          }
          aria-label="Your question"
          className="min-w-0 flex-1 rounded-md border bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground disabled:opacity-50"
          style={{ borderColor: "var(--border-strong)" }}
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-md px-4 py-2 text-sm font-medium transition-opacity disabled:opacity-30"
          style={{ background: "var(--vital)", color: "var(--background)" }}
        >
          Ask
        </button>
      </form>
    </div>
  );
}
