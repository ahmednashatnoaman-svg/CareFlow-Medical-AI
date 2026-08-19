"""Module 6: Hierarchical Retrieval Engine Service.

Executes dense vector search over Qdrant medical_chunks collection,
applies metadata filters, retrieves Top-K medical chunks, and merges results.
Fully dynamic — queries Qdrant database without any hardcoded document pools or keyword fallbacks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient
from careflow.core.config import settings
from careflow.services.embedding_service import embed_text

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hierarchical hybrid retrieval engine over medical guidelines in Qdrant."""

    def __init__(self, qdrant_client: Optional[AsyncQdrantClient] = None):
        if qdrant_client:
            self.qdrant = qdrant_client
        elif settings.QDRANT_URL:
            self.qdrant = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_SEARCH_TIMEOUT,
            )
        elif settings.QDRANT_HOST:
            self.qdrant = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_SEARCH_TIMEOUT,
            )
        else:
            self.qdrant = None

    def _embed_text(self, text: str, vector_size: Optional[int] = None) -> List[float]:
        """Embed `text` via the shared embedding service.

        Delegates to careflow.services.embedding_service, which is the single provider
        chain for the whole app. This method used to carry its own four-provider cascade
        ending in a random vector -- a chain that diverged from the one in
        dialogue_service, so the two RAG modes could embed the same query differently.

        Raises EmbeddingUnavailableError when no provider works, rather than returning a
        placeholder that would make irrelevant retrieval look successful.
        """
        return embed_text(text, vector_size or settings.VECTOR_SIZE)

    async def search_chunks(
        self,
        query: str,
        top_k: int | None = None,
        chief_complaint_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Performs search to retrieve top candidate medical chunks from Qdrant vector database.

        Args:
            query (str): Canonical clinical English patient query / symptoms.
            top_k (int | None): Candidates to retrieve; defaults to settings.RETRIEVAL_CANDIDATE_K.
            chief_complaint_filter (Optional[str]): Metadata section filter.

        Returns:
            List[Dict[str, Any]]: Top candidate chunks retrieved directly from Qdrant.
        """
        top_k = top_k if top_k is not None else settings.RETRIEVAL_CANDIDATE_K

        logger.info("Executing Qdrant vector retrieval", extra={"query": query, "top_k": top_k})

        if not self.qdrant:
            logger.warning("Qdrant client is not initialized or configured. Returning empty chunks.")
            return []

        # Embedding failure is not a Qdrant/network hiccup -- it means there is no vector
        # to search with at all, so it must not be folded into the same "return []" path
        # as a collection-search failure below. Silently returning zero chunks here would
        # be indistinguishable from "the guidelines genuinely have nothing relevant," which
        # is dangerous given the orchestrator prompt asserts it reasons strictly from
        # retrieved evidence. Let it propagate so the caller can fail loudly.
        query_vector = self._embed_text(query)

        try:
            filter_cond = None
            if chief_complaint_filter:
                from qdrant_client.http import models
                filter_cond = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="section",
                            match=models.MatchValue(value=chief_complaint_filter)
                        )
                    ]
                )

            target_collections = [
                settings.QDRANT_COLLECTION_CHUNKS,
                settings.QDRANT_COLLECTION,
            ]
            target_collections = list(dict.fromkeys(c for c in target_collections if c))

            hits = []
            # P2-8 FIX: Use sentinel 'unknown' so the log message is never misleadingly
            # attributing results to the first collection when all searches failed.
            collection_name = "unknown"

            async def _query(c_name: str) -> list:
                """Execute vector search on a single Qdrant collection."""
                if hasattr(self.qdrant, "query_points"):
                    res = await self.qdrant.query_points(
                        collection_name=c_name,
                        query=query_vector,
                        query_filter=filter_cond,
                        limit=top_k,
                    )
                    return res.points if hasattr(res, "points") else res
                elif hasattr(self.qdrant, "search"):
                    return await self.qdrant.search(
                        collection_name=c_name,
                        query_vector=query_vector,
                        query_filter=filter_cond,
                        limit=top_k,
                    )
                return []

            for col in target_collections:
                try:
                    res_hits = await asyncio.wait_for(_query(col), timeout=settings.QDRANT_SEARCH_TIMEOUT)
                    if res_hits:
                        hits = res_hits
                        collection_name = col
                        break
                except Exception as col_err:
                    logger.debug("Search on collection '%s' failed: %s", col, col_err)

            qdrant_chunks: List[Dict[str, Any]] = []
            for hit in hits:
                payload = getattr(hit, "payload", {}) or {}
                chunk_id = str(payload.get("chunk_id") or getattr(hit, "id", ""))
                source = str(payload.get("source", "Qdrant Medical Knowledge"))
                section = str(payload.get("section", "General"))
                content = str(
                    payload.get("content")
                    or payload.get("raw_text")
                    or payload.get("text")
                    or ""
                )
                score = round(float(getattr(hit, "score", 0.0)), 4)

                qdrant_chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "section": section,
                    "content": content,
                    "score": score,
                    "metadata": payload,
                })

            logger.info(
                "Retrieved %d medical chunks from Qdrant collection '%s'",
                len(qdrant_chunks),
                collection_name,
            )
            return qdrant_chunks

        except Exception as exc:
            logger.warning(
                "Qdrant retrieval failed: %s.",
                exc,
            )
            return []
