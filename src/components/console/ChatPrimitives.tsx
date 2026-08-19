"use client";

import { useEffect, useRef } from "react";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  /** Rendered beneath the bubble — citations, options, or a report. */
  slot?: React.ReactNode;
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  if (isSystem) {
    return (
      <div
        role="status"
        className="trace-in mx-auto max-w-[90%] rounded-md border px-3 py-2 text-center text-xs"
        style={{ borderColor: "var(--emergency)", color: "var(--emergency)" }}
      >
        {message.content}
      </div>
    );
  }

  return (
    <div className={`trace-in flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        aria-hidden
        className="numeric mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border text-[10px]"
        style={{
          borderColor: isUser ? "var(--border-strong)" : "var(--vital-dim)",
          color: isUser ? "var(--muted-foreground)" : "var(--vital)",
        }}
      >
        {isUser ? "PT" : "AI"}
      </div>

      <div className={`min-w-0 max-w-[85%] ${isUser ? "items-end" : ""}`}>
        <div
          className="rounded-lg border px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap"
          style={{
            background: isUser ? "var(--surface-raised)" : "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          {message.content}
        </div>
        {message.slot}
      </div>
    </div>
  );
}

export function TypingIndicator({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3" role="status" aria-live="polite">
      <div
        aria-hidden
        className="numeric flex h-7 w-7 items-center justify-center rounded-md border text-[10px]"
        style={{ borderColor: "var(--vital-dim)", color: "var(--vital)" }}
      >
        AI
      </div>
      <div className="flex items-center gap-2">
        <div className="flex gap-1" aria-hidden>
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="blip h-1.5 w-1.5 rounded-full"
              style={{ background: "var(--vital)", animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
    </div>
  );
}

/** Scrolls to the newest message whenever `dep` changes. */
export function useAutoScroll<T>(dep: T) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [dep]);
  return ref;
}

export function ErrorNotice({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="rounded-md border px-3 py-2.5 text-xs"
      style={{ borderColor: "var(--emergency)", background: "color-mix(in oklab, var(--emergency) 8%, transparent)" }}
    >
      <p style={{ color: "var(--emergency)" }}>{error}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1.5 underline underline-offset-2 hover:no-underline"
          style={{ color: "var(--emergency)" }}
        >
          Retry
        </button>
      )}
    </div>
  );
}
