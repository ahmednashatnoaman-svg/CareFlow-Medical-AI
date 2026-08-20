"""Labeled retrieval benchmark for the WHO guidelines collection.

Measures the retrieval stage in isolation -- no LLM, no judge model, no API quota. Because
every query carries a labeled `expected_source`, this reports true precision@k, MRR, and
hit rate rather than a bare similarity number.

This exists because scripts/evaluate_rag.py samples data/meddata, whose questions are about
rare genetic conditions ("Dystonia 18", "Benign schwannoma") that WHO public-health
guidelines do not cover. Scores there measure corpus/dataset mismatch, not retrieval
quality, and would look the same whether retrieval worked or not.

Usage:
    python scripts/benchmark_retrieval.py [--top-k K] [--output PATH]
    python scripts/benchmark_retrieval.py --no-rerank      # bi-encoder only, for the rerank ablation
    python scripts/benchmark_retrieval.py --sweep 0.20,0.35,0.45,0.50,0.55,0.60
                                                             # vector-score-threshold sweep (always rerank-free)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from careflow.core.config import settings
from careflow.services.dialogue_service import dialogue_service

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("benchmark_retrieval")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "benchmarks" / "retrieval_benchmark.json"
DEFAULT_OUTPUT = ROOT / "careflow" / "artifacts" / "retrieval_benchmark_results.json"

# Retrieve once with the filter wide open, then apply each candidate threshold offline.
# Qdrant applies `score_threshold` alongside `limit`, so filtering the unthresholded top-K
# is equivalent to re-querying at that threshold: any point scoring >= t that fell outside
# the top-K sits below K points that each score >= its own score, hence >= t, so those K
# would saturate the limit regardless. One embedding call per query covers the whole sweep.
#
# The sweep always measures the bi-encoder (rerank=False): it is isolating the vector
# score threshold specifically, and cross-encoder relevance scores live on a different
# scale, so filtering reranked results by this vector score would conflate two unrelated
# signals.
UNFILTERED = 0.0


def reciprocal_rank(sources: list[str], expected: str) -> float:
    """1/rank of the first correct source, or 0.0 if absent."""
    for i, s in enumerate(sources, 1):
        if s == expected:
            return 1.0 / i
    return 0.0


def keyword_recall(chunks: list[dict[str, Any]], keywords: list[str]) -> float:
    """Share of expected keywords present anywhere in the retrieved text.

    A content check independent of source labelling: retrieving the right *document* but
    the wrong *chunk* of it still fails to give the generator what it needs.
    """
    if not keywords:
        return 0.0
    blob = " ".join(c.get("text", "") for c in chunks).lower()
    return sum(1 for k in keywords if k.lower() in blob) / len(keywords)


async def retrieve_for_sweep(
    queries: list[dict[str, Any]], breadth: int
) -> list[list[dict[str, Any]]]:
    """One unthresholded, rerank-free retrieval pass per query, wide enough to sweep."""
    out: list[list[dict[str, Any]]] = []
    for q in queries:
        out.append(
            await dialogue_service.search_guidelines(
                q["question"], top_k=breadth, score_threshold=UNFILTERED, rerank=False
            )
        )
    return out


async def retrieve_production(
    queries: list[dict[str, Any]], top_k: int, rerank: bool, threshold: float
) -> list[list[dict[str, Any]]]:
    """Retrieval exactly as `search_guidelines` runs in the app: given threshold, optional
    cross-encoder rerank down to `top_k`."""
    out: list[list[dict[str, Any]]] = []
    for q in queries:
        out.append(
            await dialogue_service.search_guidelines(
                q["question"], top_k=top_k, score_threshold=threshold, rerank=rerank
            )
        )
    return out


def score_at(
    queries: list[dict[str, Any]],
    retrievals: list[list[dict[str, Any]]],
    threshold: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the benchmark as it would run with `RETRIEVAL_SCORE_THRESHOLD=threshold`."""
    rows: list[dict[str, Any]] = []

    for q, retrieved in zip(queries, retrievals):
        chunks = [c for c in retrieved if c["score"] >= threshold]
        sources = [c["source_file"] for c in chunks]
        expected = q["expected_source"]

        hits = sum(1 for s in sources if s == expected)
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_source": expected,
            "top1_source": sources[0] if sources else None,
            "top1_correct": bool(sources) and sources[0] == expected,
            "top1_score": chunks[0]["score"] if chunks else 0.0,
            f"precision_at_{top_k}": round(hits / len(sources), 4) if sources else 0.0,
            "reciprocal_rank": round(reciprocal_rank(sources, expected), 4),
            "keyword_recall": round(keyword_recall(chunks, q.get("expected_keywords", [])), 4),
            "chunks_returned": len(chunks),
        })

    n = len(rows)
    summary = {
        "score_threshold": threshold,
        "top1_accuracy": round(sum(r["top1_correct"] for r in rows) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
        f"mean_precision_at_{top_k}": round(sum(r[f"precision_at_{top_k}"] for r in rows) / n, 4),
        "mean_keyword_recall": round(sum(r["keyword_recall"] for r in rows) / n, 4),
        "mean_top1_score": round(sum(r["top1_score"] for r in rows) / n, 4),
        "mean_chunks_returned": round(sum(r["chunks_returned"] for r in rows) / n, 4),
        "hit_rate": round(sum(1 for r in rows if r["chunks_returned"] > 0) / n, 4),
        "full_keyword_recall_queries": sum(1 for r in rows if r["keyword_recall"] == 1.0),
    }
    return summary, rows


async def run(top_k: int, thresholds: list[float], primary: float, rerank: bool) -> dict[str, Any]:
    spec = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    queries = spec["queries"]

    # Production-shaped pass: exactly what the app does at the configured threshold, with
    # (or without, per --no-rerank) the cross-encoder stage. This is the headline summary.
    prod_retrievals = await retrieve_production(queries, top_k, rerank, primary)
    summary, rows = score_at(queries, prod_retrievals, primary, top_k)

    report: dict[str, Any] = {
        "summary": summary,
        "results": rows,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_queries": len(queries),
            "top_k": top_k,
            "score_threshold": primary,
            "rerank": rerank,
            "collection": settings.QDRANT_COLLECTION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
        },
    }

    if thresholds:
        # Sweep pass: bi-encoder only (see retrieve_for_sweep docstring), scored offline
        # at each candidate threshold from one shared retrieval per query.
        sweep_retrievals = await retrieve_for_sweep(queries, top_k)
        report["sweep"] = [score_at(queries, sweep_retrievals, t, top_k)[0] for t in thresholds]

    return report


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--sweep",
        type=str,
        default="",
        help="Comma-separated score thresholds to evaluate, e.g. 0.20,0.50,0.55",
    )
    ap.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder rerank stage (bi-encoder-only, for the rerank ablation)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Vector score_threshold for the production pass (default: settings.RETRIEVAL_SCORE_THRESHOLD)",
    )
    args = ap.parse_args()

    thresholds = [float(t) for t in args.sweep.split(",") if t.strip()]
    primary = args.threshold if args.threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD
    rerank = not args.no_rerank
    pk = f"mean_precision_at_{args.top_k}"

    report = await run(args.top_k, thresholds, primary, rerank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    n = report["metadata"]["n_queries"]
    print(f"\nRetrieval benchmark — {n} labeled queries, top_k={args.top_k}, "
          f"threshold={primary}, rerank={rerank}, collection='{report['metadata']['collection']}'\n")
    print(f"  top-1 accuracy        {s['top1_accuracy']:.3f}")
    print(f"  MRR                   {s['mrr']:.3f}")
    print(f"  precision@{args.top_k}          {s[pk]:.3f}")
    print(f"  keyword recall        {s['mean_keyword_recall']:.3f}")
    print(f"  mean top-1 score      {s['mean_top1_score']:.3f}")
    print(f"  chunks/query          {s['mean_chunks_returned']:.2f}")
    print(f"  hit rate              {s['hit_rate']:.3f}\n")

    for r in report["results"]:
        mark = "OK  " if r["top1_correct"] else "MISS"
        print(f"  [{mark}] {r['id']:<22} rr={r['reciprocal_rank']:.2f} "
              f"kw={r['keyword_recall']:.2f} score={r['top1_score']:.3f}")

    if thresholds:
        print(f"\nThreshold sweep (top_k={args.top_k}, {n} queries)\n")
        print(f"  {'thresh':>7}  {'prec@' + str(args.top_k):>7}  {'chunks/q':>8}  "
              f"{'kw recall':>9}  {'top-1 acc':>9}  {'MRR':>5}")
        for row in report["sweep"]:
            print(f"  {row['score_threshold']:>7.2f}  {row[pk]:>7.3f}  "
                  f"{row['mean_chunks_returned']:>8.2f}  "
                  f"{str(row['full_keyword_recall_queries']) + '/' + str(n):>9}  "
                  f"{row['top1_accuracy']:>9.3f}  {row['mrr']:>5.3f}")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
