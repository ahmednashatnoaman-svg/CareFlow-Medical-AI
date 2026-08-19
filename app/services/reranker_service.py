"""Module 7: Cross Encoder Reranker Service.

Reranks candidate retrieved chunks (Top K=20) using cross-encoder scores (bge-reranker-v2-m3)
to output the Top N=10 highest precision context chunks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cached across instances -- CrossEncoder model loading is expensive (network fetch +
# weight loading), and CrossEncoderReranker() is constructed fresh in tests and possibly
# in multiple service instances, so a per-instance load would repeat that cost needlessly.
_cross_encoder_cache: Dict[str, Any] = {}


class CrossEncoderReranker:
    """Reranks candidate chunks with a real cross-encoder model when available.

    Falls back to a lexical-overlap heuristic if the model can't be loaded (offline,
    dependency missing, first-download failure) -- unlike the prior version, which never
    attempted to load `model_name` at all despite the class name and docstring claiming
    cross-encoder scoring; `model_name` was accepted and silently ignored.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model: Optional[Any] = None):
        """`model` lets callers (tests, alternate deployments) inject a pre-built
        cross-encoder-like object (anything exposing `.predict(pairs) -> scores`) instead
        of downloading `model_name` from the network -- without it, every test that
        constructs a bare `CrossEncoderReranker()` would otherwise trigger a real model
        download on first use, exactly the "unit test needs the network" problem this
        service previously avoided only by accident (it never loaded any model at all).
        """
        self.model_name = model_name
        self._model: Optional[Any] = model
        self._load_attempted = model is not None

    def _get_model(self) -> Optional[Any]:
        if self._load_attempted:
            return self._model
        self._load_attempted = True

        if self.model_name in _cross_encoder_cache:
            self._model = _cross_encoder_cache[self.model_name]
            return self._model

        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(self.model_name)
            _cross_encoder_cache[self.model_name] = model
            self._model = model
            logger.info("Loaded cross-encoder reranker model '%s'", self.model_name)
        except Exception as exc:
            logger.warning(
                "Could not load cross-encoder model '%s' (%s); using lexical-overlap "
                "heuristic reranker instead",
                self.model_name,
                exc,
            )
            self._model = None

        return self._model

    def _heuristic_score(self, query: str, chunk: Dict[str, Any]) -> float:
        """Lexical-overlap fallback used only when the real cross-encoder is unavailable."""
        keywords = set(query.lower().split())
        content_words = set(str(chunk.get("content", "")).lower().split())
        overlap_count = len(keywords & content_words)
        base_score = chunk.get("score", 0.5)
        return round(base_score * 0.6 + (overlap_count / max(len(keywords), 1)) * 0.4, 4)

    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate chunks by relevance score to patient query.

        Args:
            query (str): Patient query / symptoms in English.
            chunks (List[Dict[str, Any]]): Candidate retrieved chunks.
            top_n (int): Number of top reranked chunks to return (Default N=10).

        Returns:
            List[Dict[str, Any]]: Sorted top_n chunks.
        """
        if not chunks:
            return []

        logger.info("Executing Cross Encoder Reranker", extra={"input_chunks": len(chunks), "top_n": top_n})

        model = await asyncio.to_thread(self._get_model)

        if model is not None:
            pairs = [(query, str(chunk.get("content", ""))) for chunk in chunks]
            scores = await asyncio.to_thread(model.predict, pairs)
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                updated_chunk = dict(chunk)
                updated_chunk["rerank_score"] = round(float(score), 4)
                scored_chunks.append(updated_chunk)
        else:
            scored_chunks = [
                {**chunk, "rerank_score": self._heuristic_score(query, chunk)}
                for chunk in chunks
            ]

        sorted_chunks = sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_chunks[:top_n]
