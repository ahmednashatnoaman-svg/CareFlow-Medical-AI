/**
 * Typed client for the CareFlow FastAPI backend.
 *
 * Types mirror careflow/schemas/chatbot.py exactly. Keeping them in one file means a
 * backend contract change surfaces as a TypeScript error at every call site rather than
 * as `undefined` rendered into the UI.
 */

/** Base path. Same-origin by default; both next.config rewrites and vercel.json map
 *  /api/* to the FastAPI app, so the browser never needs an absolute backend URL. */
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    // fetch() rejects only on network-layer failure. Surfaced distinctly from an HTTP
    // error so the UI can say "backend unreachable" instead of a misleading status code.
    throw new ApiError("Cannot reach the CareFlow API. Is the backend running?", 0, String(cause));
  }

  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.message;
    } catch {
      /* non-JSON error body; fall through to the status text */
    }
    throw new ApiError(detail ?? res.statusText, res.status, detail);
  }
  return res.json() as Promise<T>;
}

/* ------------------------------------------------------------------ */
/* Mode 1 — Graph RAG triage                                           */
/* ------------------------------------------------------------------ */

/** The eight SOCRATES pain-assessment axes tracked across the interview. */
export interface SocratesTracker {
  site: boolean;
  onset: boolean;
  character: boolean;
  radiation: boolean;
  associated_symptoms: boolean;
  time_course: boolean;
  exacerbating_relieving: boolean;
  severity: boolean;
}

export interface DiagnosisDetail {
  diagnosis: string;
  estimated_probability: string;
  urgency_level: string;
  reasoning: string;
  graph_evidence: string;
}

export interface DiagnosticReport {
  top_diagnoses: DiagnosisDetail[];
  triage_recommendation: string;
  clinical_summary?: string | null;
}

export interface TriageResponse {
  session_id: string;
  is_complete: boolean;
  message: string;
  options: string[];
  target_symptom?: string | null;
  socrates_tracker: Partial<SocratesTracker>;
  socrates_score: number;
  turn_count: number;
  max_turns: number;
  positive_symptoms: string[];
  negated_symptoms: string[];
  stop_reason?: string | null;
  diagnostic_report?: DiagnosticReport | null;
}

export type Language = "en" | "ar";

export const triage = {
  start: (sessionId: string, language: Language) =>
    request<TriageResponse>("/triage/start", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, language }),
    }),

  /** Advance one turn. `message` is free text, or "1"|"2"|"3" to pick an option. */
  step: (sessionId: string, message: string, language: Language) =>
    request<TriageResponse>("/triage/step", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message, language }),
    }),

  reset: (sessionId: string, language: Language) =>
    request<TriageResponse>("/triage/reset", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, language }),
    }),
};

/* ------------------------------------------------------------------ */
/* Mode 2 — WHO guidelines vector RAG                                  */
/* ------------------------------------------------------------------ */

export interface Citation {
  source_file: string;
  section: string;
  relevance_score: number;
  snippet: string;
}

export interface DialogueResponse {
  query: string;
  answer: string;
  sources: Citation[];
  chunks_retrieved: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export const dialogue = {
  chat: (query: string, history: ChatTurn[], topK = 5) =>
    request<DialogueResponse>("/dialogue/chat", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK, conversation_history: history }),
    }),
};

/* ------------------------------------------------------------------ */
/* Evaluation + health                                                 */
/* ------------------------------------------------------------------ */

/** Per-question row emitted by scripts/evaluate_rag.py. */
export interface EvaluationRow {
  question: string;
  latency_sec?: number;
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  answer_similarity?: number;
}

export interface EvaluationResults {
  averages: {
    faithfulness?: number;
    answer_relevancy?: number;
    context_precision?: number;
    context_recall?: number;
    answer_similarity?: number;
    latency_sec?: number;
  };
  detailed_results: EvaluationRow[];
  /** Present only when the run recorded provenance; absent in older artifacts. */
  metadata?: {
    generated_at?: string;
    sample_size?: number;
    retriever?: string;
    generator?: string;
  };
}

export const evaluation = {
  /** Rejects with ApiError(404) when no run exists. Callers must render "not measured"
   *  rather than substituting placeholder figures. */
  get: () => request<EvaluationResults>("/evaluation"),
};

export interface ReadinessCheck {
  status: "ok" | "degraded" | "unavailable";
  [k: string]: unknown;
}

export interface Readiness {
  status: "ok" | "degraded" | "unavailable";
  version: string;
  environment: string;
  checks: Record<string, ReadinessCheck>;
}

export const health = {
  ready: () => request<Readiness>("/health/ready"),
};
