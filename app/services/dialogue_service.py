"""WHO Guidelines Dialogue Vector RAG Service.

Queries the remote Qdrant cloud collection 'who_guidelines' with BAAI/bge-m3 embeddings
and generates grounded, citation-backed medical responses.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from app.core.config import settings
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)

# Global cache for transformer model to avoid reload latency
_TOKENIZER = None
_TRANSFORMER_MODEL = None


def get_embedding_vector(text: str, vector_size: int = 1024) -> List[float]:
    """Generates 1024-dim BAAI/bge-m3 dense vector embedding for queries."""
    global _TOKENIZER, _TRANSFORMER_MODEL

    # Method 1: Hugging Face Transformers with safetensors
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if _TOKENIZER is None or _TRANSFORMER_MODEL is None:
            logger.info("Loading BGE-M3 embedding model '%s' (safetensors)...", settings.EMBEDDING_MODEL)
            _TOKENIZER = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
            _TRANSFORMER_MODEL = AutoModel.from_pretrained(settings.EMBEDDING_MODEL, use_safetensors=True)
            _TRANSFORMER_MODEL.eval()

        inputs = _TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = _TRANSFORMER_MODEL(**inputs)
            # CLS token representation normalized
            vec = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)[0].tolist()
            if len(vec) == vector_size:
                return vec
    except Exception as e:
        logger.warning("Local Transformers BGE-M3 embedding failed (%s). Checking fallbacks.", e)

    # Method 2: Cloud Modal Endpoint (if configured)
    raw_url = settings.EMBEDDER_ENDPOINT_URL
    if raw_url:
        try:
            import httpx
            base_url = raw_url.rstrip("/")
            endpoint = base_url if "/api/v1/embed" in base_url else f"{base_url}/api/v1/embed"
            with httpx.Client(timeout=10.0) as client:
                res = client.post(endpoint, json={"text": text, "normalize": True})
                if res.status_code == 200:
                    data = res.json()
                    vec = data.get("embedding") or (data.get("embeddings")[0] if data.get("embeddings") else None)
                    if vec and len(vec) == vector_size:
                        return vec
        except Exception as e:
            logger.warning("Modal cloud embedder endpoint failed (%s)", e)

    # Method 3: Deterministic pseudo-random embedding for test environments
    import hashlib, random
    logger.warning("Using deterministic fallback embedding.")
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    return [rng.uniform(-0.1, 0.1) for _ in range(vector_size)]


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
                timeout=10.0,
            )
        return self._sync_client

    def get_async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                timeout=10.0,
            )
        return self._async_client

    async def search_guidelines(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Searches who_guidelines collection in Qdrant for semantic matches."""
        logger.info("Executing WHO guidelines vector search for query: '%s' (top_k=%d)", query[:80], top_k)
        
        # Embed query text
        query_vector = await asyncio.to_thread(get_embedding_vector, query, settings.VECTOR_SIZE)

        client = self.get_async_client()
        hits = []

        try:
            # Query Qdrant with named vector 'dense'
            if hasattr(client, "query_points"):
                res = await client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="dense",
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                hits = res.points if hasattr(res, "points") else res
            elif hasattr(client, "search"):
                hits = await client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", query_vector),
                    limit=top_k,
                    score_threshold=score_threshold,
                )
        except Exception as e:
            logger.warning("Primary query_points with using='dense' failed: %s. Retrying unnamed vector.", e)
            try:
                if hasattr(client, "query_points"):
                    res = await client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=top_k,
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

        logger.info("Retrieved %d relevant chunks from '%s'", len(chunks), self.collection_name)
        return chunks

    async def answer_question(
        self,
        query: str,
        top_k: int = 5,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Performs end-to-end RAG over WHO guidelines with source attribution."""
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
            temperature=0.2,
            max_tokens=1500,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "source_file": c["source_file"],
                    "section": c["section"],
                    "relevance_score": c["score"],
                    "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                }
                for c in chunks
            ],
            "chunks_retrieved": len(chunks),
        }


# Global singleton instance
dialogue_service = DialogueRAGService()
