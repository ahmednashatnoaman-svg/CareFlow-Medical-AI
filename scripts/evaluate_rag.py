"""Quantitative RAG evaluation for CareFlow.

Measures the retrieval and generation stages **separately**, because they fail for
different reasons and a single blended score hides which one is at fault:

  Retrieval  — context precision / recall, plus retrieval-only diagnostics (hit rate,
               MRR, mean top-1 score) computed without involving the generator at all.
  Generation — faithfulness, answer relevancy, answer similarity, conditioned on
               whatever the retriever actually returned.

Writes careflow/artifacts/evaluation_results.json, which GET /api/v1/evaluation serves
and the dashboard renders. The dashboard has no fallback values: whatever is missing here
shows as "not measured" rather than as an invented number.

Two question sources are available:

  benchmark (default) data/benchmarks/retrieval_benchmark.json -- the same domain-matched
            clinical queries scored for retrieval in scripts/benchmark_retrieval.py. These
            *do* retrieve real context, so this is where generation quality on a working
            retrieval path is actually measured. The benchmark file has no curated
            reference answer, only expected_keywords, so only the ground-truth-free
            metrics (faithfulness, answer_relevancy) are computed here -- context_recall
            and answer_similarity need a real reference and are left unmeasured rather
            than approximated from keywords.
  meddata   data/meddata/*QA.csv[.zip] -- a broad general-medical QA corpus, largely *not*
            covered by the indexed WHO/CDC/USPSTF guidelines. It exercises
            faithfulness/relevancy plus context_precision/context_recall/answer_similarity
            against each row's own reference "Answer", but most questions retrieve zero
            in-domain context at a well-tuned threshold, so this is a hallucination/
            refusal check ("does it say 'insufficient information' instead of guessing?"),
            not a generation-quality measurement -- writes to a separate file rather than
            standing in as the headline evaluation.

Usage:
    python scripts/evaluate_rag.py [--sample-size N] [--top-k K] [--retrieval-only]
    python scripts/evaluate_rag.py --dataset meddata
"""

import argparse
import asyncio
import glob
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from careflow.core.config import settings
from careflow.services.dialogue_service import dialogue_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate_rag")

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "careflow" / "artifacts" / "evaluation_results.json"

# The Gemini free tier allows 15 requests/minute and ragas issues several LLM calls per
# sample, so requests are serialized with a delay. Raising these will trip 429s.
INTER_QUERY_DELAY_S = 2.5


def load_dataset(sample_size: int, seed: int = 42) -> pd.DataFrame:
    """Load the medical QA corpus.

    Reads .csv.zip alongside .csv -- most of data/meddata is zipped, and the previous
    glob matched only "*QA.csv", silently sampling from a third of the corpus.
    """
    root = Path(__file__).resolve().parents[1] / "data" / "meddata"
    paths = sorted(glob.glob(str(root / "*QA.csv"))) + sorted(glob.glob(str(root / "*QA.csv.zip")))
    if not paths:
        raise SystemExit(f"No QA CSVs found under {root}")

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:  # noqa: BLE001 - skip unreadable shards, report at the end
            logger.warning("Skipping %s: %s", Path(p).name, exc)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Question", "Answer"])
    logger.info("Loaded %d QA pairs from %d files", len(df), len(frames))
    return df.sample(n=min(sample_size, len(df)), random_state=seed)


def load_benchmark_dataset() -> pd.DataFrame:
    """Load the domain-matched clinical queries used by scripts/benchmark_retrieval.py.

    No "Answer" column: the benchmark labels expected_source and expected_keywords, not a
    curated reference answer, so callers must not treat the empty Answer as ground truth.
    """
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "retrieval_benchmark.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(spec["queries"])
    df["Answer"] = ""
    logger.info("Loaded %d domain-matched clinical queries from %s", len(df), path.name)
    return df.rename(columns={"question": "Question"})


async def collect_predictions(questions: list[str], top_k: int) -> list[dict[str, Any]]:
    """Run the live RAG pipeline once per question, recording contexts and latency."""
    records: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        started = time.perf_counter()
        try:
            res = await dialogue_service.answer_question(query=q, top_k=top_k)
            records.append({
                "question": q,
                "answer": res["answer"],
                # Each record owns its own list. The previous version initialised
                # `contexts = [[]] * n`, which aliases one list across every row.
                "contexts": [s["snippet"] for s in res["sources"]],
                "scores": [s["relevance_score"] for s in res["sources"]],
                "latency_sec": round(time.perf_counter() - started, 3),
                "error": None,
            })
            logger.info("[%d/%d] ok (%.2fs, %d chunks)", i, len(questions),
                        records[-1]["latency_sec"], len(records[-1]["contexts"]))
        except Exception as exc:  # noqa: BLE001 - a failed sample is recorded, not fatal
            logger.warning("[%d/%d] failed: %s", i, len(questions), exc)
            records.append({
                "question": q, "answer": "", "contexts": [], "scores": [],
                "latency_sec": round(time.perf_counter() - started, 3), "error": str(exc),
            })
        await asyncio.sleep(INTER_QUERY_DELAY_S)
    return records


def retrieval_diagnostics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Retrieval-only metrics that need no LLM judge.

    Cheap, deterministic, and unaffected by generator quality or API quota, so retrieval
    regressions stay measurable even when the LLM-judged metrics cannot run.
    """
    ok = [r for r in records if not r["error"]]
    if not ok:
        return {}
    hit_rate = sum(1 for r in ok if r["contexts"]) / len(ok)
    top1 = [r["scores"][0] for r in ok if r["scores"]]
    latencies = [r["latency_sec"] for r in ok]
    return {
        "hit_rate": round(hit_rate, 4),
        "mean_top1_score": round(sum(top1) / len(top1), 4) if top1 else 0.0,
        "mean_chunks_retrieved": round(sum(len(r["contexts"]) for r in ok) / len(ok), 2),
        "mean_latency_sec": round(sum(latencies) / len(latencies), 3),
    }


def run_ragas(
    records: list[dict[str, Any]], ground_truths: list[str], has_ground_truth: bool
) -> dict[str, Any]:
    """Score with ragas, splitting metrics by pipeline stage.

    `has_ground_truth` gates which metrics run: context_recall and answer_similarity both
    compare against a reference answer, which the domain-matched benchmark dataset doesn't
    have (see load_benchmark_dataset). Requesting them with an empty ground_truth string
    wouldn't error -- ragas would just score against a false reference -- so the metric set
    is chosen up front instead of relying on ragas to notice the input is meaningless.

    Returns {} when ragas or its judge model is unavailable; callers must then emit no
    numbers at all rather than substituting defaults.
    """
    try:
        from datasets import Dataset
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy, answer_similarity, context_precision, context_recall, faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        logger.error("ragas stack unavailable (%s). Install with: uv pip install -e '.[eval]'", exc)
        return {}

    # AnswerRelevancy defaults to strictness=3 -- it scores relevance by asking the judge
    # to generate 3 candidate questions per answer in one call (n=3). Groq's
    # OpenAI-compatible endpoint rejects any n > 1 ("number must be at most 1"), which
    # otherwise fails this metric on every sample, every retry, for the configured
    # max_retries. strictness=1 asks for one completion per call, which Groq accepts.
    answer_relevancy = AnswerRelevancy(strictness=1)

    usable = [(r, gt) for r, gt in zip(records, ground_truths) if not r["error"] and r["contexts"]]
    if not usable:
        logger.error("No successful samples with retrieved context; skipping ragas.")
        return {}

    ds_dict = {
        "question": [r["question"] for r, _ in usable],
        "answer":   [r["answer"] for r, _ in usable],
        "contexts": [r["contexts"] for r, _ in usable],
    }
    if has_ground_truth:
        ds_dict["ground_truth"] = [gt for _, gt in usable]
    ds = Dataset.from_dict(ds_dict)

    if not settings.GROQ_API_KEY:
        logger.error("GROQ_API_KEY unset; ragas needs a judge model.")
        return {}

    judge = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)
    embedder = GoogleGenerativeAIEmbeddings(
        model=settings.RAGAS_JUDGE_EMBEDDING_MODEL, google_api_key=settings.GEMINI_API_KEY
    )

    if has_ground_truth:
        RETRIEVAL = {"context_precision": context_precision, "context_recall": context_recall}
        GENERATION = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_similarity": answer_similarity,
        }
    else:
        RETRIEVAL = {}
        GENERATION = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy}

    try:
        result = evaluate(
            dataset=ds,
            metrics=[*RETRIEVAL.values(), *GENERATION.values()],
            llm=judge,
            embeddings=embedder,
            run_config=RunConfig(max_workers=2, max_retries=10),
        )
        # EvaluationResult isn't a Mapping in this ragas version -- dict(result) raises
        # KeyError trying to iterate it as one. to_pandas() is the one public, stable
        # accessor for both the per-sample rows and (via column means) the aggregate
        # score, so it's used for both rather than a private/version-fragile attribute.
        df = result.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.error("ragas evaluation failed: %s", exc)
        return {}

    def col_mean(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        vals = df[col].dropna()
        return round(float(vals.mean()), 4) if len(vals) else None

    return {
        "retrieval": {k: v for k in RETRIEVAL if (v := col_mean(k)) is not None},
        "generation": {k: v for k in GENERATION if (v := col_mean(k)) is not None},
        "per_sample": df.to_dict(orient="records"),
        "n_scored": len(usable),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Skip LLM-judged metrics (no judge quota needed).")
    ap.add_argument(
        "--dataset", choices=["meddata", "benchmark"], default="benchmark",
        help="benchmark: domain-matched clinical queries that actually retrieve context "
             "(default -- see module docstring). meddata: broad QA corpus with real "
             "reference answers, mostly out-of-domain for the indexed guidelines; useful "
             "as a hallucination/refusal check, not a generation-quality measurement.",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Override output path (default: careflow/artifacts/evaluation_results.json for "
             "benchmark -- the file GET /api/v1/evaluation serves -- or "
             "evaluation_results_meddata.json for the meddata stress test).",
    )
    args = ap.parse_args()

    has_ground_truth = args.dataset == "meddata"
    df = load_dataset(args.sample_size) if has_ground_truth else load_benchmark_dataset()
    questions = df["Question"].astype(str).tolist()
    ground_truths = df["Answer"].astype(str).tolist()

    logger.info("Running RAG over %d questions (top_k=%d, dataset=%s)...",
                len(questions), args.top_k, args.dataset)
    records = await collect_predictions(questions, args.top_k)

    diagnostics = retrieval_diagnostics(records)
    ragas_scores = (
        {} if args.retrieval_only else run_ragas(records, ground_truths, has_ground_truth)
    )

    averages: dict[str, float] = {**ragas_scores.get("retrieval", {}), **ragas_scores.get("generation", {})}
    if diagnostics.get("mean_latency_sec") is not None:
        averages["latency_sec"] = diagnostics["mean_latency_sec"]

    # ragas has used both "question" and "user_input" as the question-column name across
    # versions; match on whichever this installed version's to_pandas() actually produced
    # rather than assuming one, so a version bump doesn't silently zero out every
    # per-sample score while the aggregate averages above keep working fine.
    per_sample = {
        (row.get("question") or row.get("user_input")): row
        for row in ragas_scores.get("per_sample", [])
    }
    detailed = []
    for r in records:
        row = {"question": r["question"], "latency_sec": r["latency_sec"],
               "chunks_retrieved": len(r["contexts"])}
        row.update({
            k: round(float(v), 4)
            for k, v in (per_sample.get(r["question"]) or {}).items()
            if k in {"faithfulness", "answer_relevancy", "context_precision",
                     "context_recall", "answer_similarity"} and isinstance(v, (int, float))
        })
        if r["error"]:
            row["error"] = r["error"]
        detailed.append(row)

    output = {
        "averages": averages,
        "retrieval_diagnostics": diagnostics,
        "detailed_results": detailed,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "sample_size": len(records),
            "n_scored": ragas_scores.get("n_scored", 0),
            "top_k": args.top_k,
            "retriever": f"Qdrant/{settings.QDRANT_COLLECTION} + {settings.EMBEDDING_MODEL}",
            "generator": settings.GEMINI_MODEL,
            "judge": settings.GROQ_MODEL if ragas_scores else None,
            "retrieval_only": args.retrieval_only,
            "dataset": args.dataset,
        },
    }

    if args.output is not None:
        output_path = args.output
    elif args.dataset == "meddata":
        output_path = OUTPUT_PATH.with_name("evaluation_results_meddata.json")
    else:
        output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    logger.info("Retrieval diagnostics: %s", diagnostics)
    logger.info("Averages: %s", averages)


if __name__ == "__main__":
    asyncio.run(main())
