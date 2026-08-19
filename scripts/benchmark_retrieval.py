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


async def run(top_k: int) -> dict[str, Any]:
    spec = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    queries = spec["queries"]
    rows: list[dict[str, Any]] = []

    for q in queries:
        chunks = await dialogue_service.search_guidelines(q["question"], top_k=top_k)
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
        "top1_accuracy": round(sum(r["top1_correct"] for r in rows) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
        f"mean_precision_at_{top_k}": round(sum(r[f"precision_at_{top_k}"] for r in rows) / n, 4),
        "mean_keyword_recall": round(sum(r["keyword_recall"] for r in rows) / n, 4),
        "mean_top1_score": round(sum(r["top1_score"] for r in rows) / n, 4),
        "hit_rate": round(sum(1 for r in rows if r["chunks_returned"] > 0) / n, 4),
    }

    return {
        "summary": summary,
        "results": rows,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_queries": n,
            "top_k": top_k,
            "collection": settings.QDRANT_COLLECTION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
        },
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    report = await run(args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    print(f"\nRetrieval benchmark — {report['metadata']['n_queries']} labeled queries, "
          f"top_k={args.top_k}, collection='{report['metadata']['collection']}'\n")
    print(f"  top-1 accuracy        {s['top1_accuracy']:.3f}")
    print(f"  MRR                   {s['mrr']:.3f}")
    print(f"  precision@{args.top_k}          {s[f'mean_precision_at_{args.top_k}']:.3f}")
    print(f"  keyword recall        {s['mean_keyword_recall']:.3f}")
    print(f"  mean top-1 score      {s['mean_top1_score']:.3f}")
    print(f"  hit rate              {s['hit_rate']:.3f}\n")

    for r in report["results"]:
        mark = "OK  " if r["top1_correct"] else "MISS"
        print(f"  [{mark}] {r['id']:<22} rr={r['reciprocal_rank']:.2f} "
              f"kw={r['keyword_recall']:.2f} score={r['top1_score']:.3f}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
