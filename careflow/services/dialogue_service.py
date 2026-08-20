"""WHO Guidelines Dialogue Vector RAG Service.

Queries the remote Qdrant cloud collection 'who_guidelines' with BAAI/bge-m3 embeddings
and generates grounded, citation-backed medical responses.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from careflow.core.config import settings
from careflow.core.constants import LLM_GROUNDED_ANSWER, LOG_QUERY_PREVIEW_CHARS
from careflow.services.embedding_service import embed_text
from careflow.services.llm_client import llm_client
from careflow.services.reranker_service import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Shared across DialogueRAGService instances for the same reason retrieval_service's
# graph pipeline caches its reranker: CrossEncoder weight loading is expensive, and this
# is the second call site for the same model (BAAI/bge-reranker-v2-m3), so it must reuse
# the reranker_service module-level cache rather than trigger a second download.
_reranker = CrossEncoderReranker()


class DialogueRAGService:
    """Vector RAG Assistant grounded in WHO medical guidelines."""

    def __init__(self):
        self.qdrant_url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self.collection_name = settings.QDRANT_COLLECTION or "who_guidelines"
        self._async_client: Optional[AsyncQdrantClient] = None
        self._sync_client: Optional[QdrantClient] = None

    def get_sync_client(self) -> QdrantClient:
        if self._sync_client is None:
            self._sync_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                timeout=settings.QDRANT_TIMEOUT,
            )
        return self._sync_client

    def get_async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                timeout=settings.QDRANT_TIMEOUT,
            )
        return self._async_client

    async def search_guidelines(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Searches who_guidelines collection in Qdrant for semantic matches.

        `top_k` and `score_threshold` default to the configured values rather than to
        literals baked into the signature, so retrieval breadth is tunable per
        deployment without a code change.

        Retrieves `RETRIEVAL_CANDIDATE_K` candidates above `score_threshold`, then reranks
        with a cross-encoder (BAAI/bge-reranker-v2-m3) down to `top_k`. Vector similarity
        (bi-encoder) optimizes for recall over the whole collection, not precision on this
        one query -- it embeds query and chunk independently and never lets them attend to
        each other. The cross-encoder scores each (query, chunk) pair jointly, so it is a
        strictly better precision signal for choosing what actually reaches the generator.
        This was previously computed for the triage graph's chunk search
        (retrieval_service.py) but not for this guideline Q&A path, so the two RAG modes
        picked their final context differently. `rerank=False` is kept for the retrieval
        benchmark, which needs raw vector order to measure the bi-encoder in isolation.
        """
        top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        score_threshold = (
            score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD
        )
        candidate_k = max(settings.RETRIEVAL_CANDIDATE_K, top_k) if rerank else top_k
        logger.info(
            "Executing WHO guidelines vector search for query: '%s' (candidate_k=%d, top_k=%d, rerank=%s)",
            query[:80], candidate_k, top_k, rerank,
        )

        # Embed query text
        query_vector = await asyncio.to_thread(embed_text, query, settings.VECTOR_SIZE)

        client = self.get_async_client()
        hits = []

        try:
            # Query Qdrant with named vector 'dense'
            if hasattr(client, "query_points"):
                res = await client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="dense",
                    limit=candidate_k,
                    score_threshold=score_threshold,
                )
                hits = res.points if hasattr(res, "points") else res
            elif hasattr(client, "search"):
                hits = await client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", query_vector),
                    limit=candidate_k,
                    score_threshold=score_threshold,
                )
        except Exception as e:
            logger.warning("Primary query_points with using='dense' failed: %s. Retrying unnamed vector.", e)
            try:
                if hasattr(client, "query_points"):
                    res = await client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=candidate_k,
                        score_threshold=score_threshold,
                    )
                    hits = res.points if hasattr(res, "points") else res
            except Exception as e2:
                logger.error("Qdrant search completely failed: %s", e2)
                return []

        chunks: List[Dict[str, Any]] = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            score = round(float(getattr(hit, "score", 0.0)), 4)
            text = payload.get("text", "")
            source_file = payload.get("source_file", "WHO Guideline Document")
            section = payload.get("section", "General")
            position = payload.get("position", 0)

            chunks.append({
                "chunk_id": str(getattr(hit, "id", "")),
                "score": score,
                "text": text,
                "source_file": source_file,
                "section": section,
                "position": position,
            })

        logger.info("Retrieved %d candidate chunks from '%s'", len(chunks), self.collection_name)

        if rerank and len(chunks) > top_k:
            # CrossEncoderReranker reads chunk text from the "content" key (a convention
            # shared with retrieval_service's chunk shape); this service's chunks carry
            # "text" instead, so bridge the two without changing either service's schema.
            for c in chunks:
                c["content"] = c["text"]
            chunks = await _reranker.rerank(query, chunks, top_n=top_k)
            for c in chunks:
                c.pop("content", None)
        else:
            chunks = chunks[:top_k]

        return chunks

    async def answer_question(
        self,
        query: str,
        top_k: int | None = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Performs end-to-end RAG over WHO guidelines with source attribution."""
        top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K

        # 1. Retrieve relevant chunks
        chunks = await self.search_guidelines(query=query, top_k=top_k)

        # 2. Assemble context block
        context_blocks = []
        for i, c in enumerate(chunks, 1):
            context_blocks.append(
                f"[Document {i}] Source: {c['source_file']} | Section: {c['section']}\nContent: {c['text']}"
            )
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant WHO guideline documents found."

        # 3. Format prompt
        system_prompt = """You are an expert WHO (World Health Organization) Medical Guidelines Assistant.
Your mission is to provide accurate, evidence-based medical information strictly grounded in the official WHO guidelines provided in the context.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer ONLY using the facts from the provided WHO Guideline documents. If the provided context does not contain enough information to answer, state clearly: "The provided WHO guidelines do not contain sufficient information on this specific topic."
2. Language: If the user asks in Arabic, answer in professional, clear Arabic. If in English, answer in English.
3. Citations: At the end of your response or when quoting specific recommendations, cite the source guideline title and section.
4. Medical Safety: Remind users that this information is for educational and clinical guideline guidance and does not replace emergency clinical judgment."""

        history_text = ""
        if conversation_history:
            history_text = "\n[RECENT CONVERSATION HISTORY]\n" + "\n".join(
                f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
                for msg in conversation_history[-4:]
            ) + "\n"

        user_prompt = f"""{history_text}
[WHO GUIDELINE RETRIEVED CONTEXT]
{context_text}

[USER INQUIRY]
{query}

Please provide a well-structured, clear, and comprehensive answer grounded strictly in the WHO guidelines above:"""

        # 4. Generate grounded answer via LLM
        answer = await asyncio.to_thread(
            llm_client.generate_text,
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=LLM_GROUNDED_ANSWER.temperature,
            max_tokens=LLM_GROUNDED_ANSWER.max_tokens,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "source_file": c["source_file"],
                    "section": c["section"],
                    "relevance_score": c["score"],
                    "snippet": c["text"][:settings.CITATION_SNIPPET_CHARS]
                    + ("..." if len(c["text"]) > settings.CITATION_SNIPPET_CHARS else ""),
                    # Full, untruncated chunk text -- the generator saw this, not the
                    # 300-char UI snippet above. Evaluation harnesses that check
                    # faithfulness against "snippet" are checking the answer against less
                    # context than actually produced it, which understates faithfulness
                    # for reasons that have nothing to do with generation quality.
                    "full_text": c["text"],
                }
                for c in chunks
            ],
            "chunks_retrieved": len(chunks),
        }


# Global singleton instance
dialogue_service = DialogueRAGService()
