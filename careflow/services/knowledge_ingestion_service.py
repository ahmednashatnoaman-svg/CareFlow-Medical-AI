"""Module 5: Knowledge Ingestion Pipeline Service.

Processes medical documents/guidelines (WHO, NICE Guidelines), performs section detection,
1200-token chunking with 200-token overlap, metadata tagging, overview generation, and Qdrant vector indexing.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from careflow.core.config import settings
from careflow.services.embedding_service import embed_text

logger = logging.getLogger(__name__)


class KnowledgeIngestionService:
    """Ingests medical PDFs and guideline text into Qdrant collections."""

    def __init__(self, qdrant_client: Optional[AsyncQdrantClient] = None):
        if qdrant_client:
            self.qdrant = qdrant_client
        elif settings.QDRANT_URL:
            self.qdrant = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            self.qdrant = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
            )

    async def initialize_collections(self) -> None:
        """Ensures medical_documents and medical_chunks collections exist in Qdrant."""
        try:
            collections = await self.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]

            if settings.QDRANT_COLLECTION_DOCUMENTS not in collection_names:
                await self.qdrant.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_DOCUMENTS,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=settings.VECTOR_SIZE, distance=models.Distance.COSINE
                        )
                    },
                )
                logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION_DOCUMENTS}")

            if settings.QDRANT_COLLECTION_CHUNKS not in collection_names:
                await self.qdrant.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_CHUNKS,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=settings.VECTOR_SIZE, distance=models.Distance.COSINE
                        )
                    },
                )
                logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION_CHUNKS}")
        except Exception as exc:
            logger.warning(f"Could not connect to live Qdrant instance: {exc}. Ingestion will run in mock mode.")

    def chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        """Splits document text into chunks of specified token/word size with overlap.

        Args:
            text (str): Document full text.
            chunk_size (int): Max chunk length in words. Defaults to 1200.
            overlap (int): Overlap length in words. Defaults to 200.

        Returns:
            List[str]: Extracted text chunks.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks: List[str] = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
        return chunks

    async def ingest_document(
        self,
        source_name: str,
        content: str,
        section: str = "General Clinical Guidelines",
        publication_date: str = "2024-01-01",
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Chunk, embed, and upsert a document into Qdrant.

        The previous implementation assembled chunk metadata, logged "Document ingested
        successfully", and returned -- without ever embedding or upserting anything. Every
        caller (including POST /api/v1/knowledge/ingest) therefore reported success while
        writing nothing, which is why the clinical guidelines never reached the index.

        Args:
            source_name: Document source, e.g. "WHO Guideline - Hypertension 2021".
            content: Full document text.
            section: Clinical section category, used as a retrieval filter.
            publication_date: Source publication date.
            collection: Target collection; defaults to settings.QDRANT_COLLECTION.

        Returns:
            Ingestion summary including how many chunks were actually written.
        """
        target = collection or settings.QDRANT_COLLECTION
        doc_id = str(uuid.uuid4())
        raw_chunks = self.chunk_text(content, chunk_size=1200, overlap=200)
        if not raw_chunks:
            return {"doc_id": doc_id, "source": source_name, "chunks_count": 0, "chunks_written": 0}

        # Embedded off the event loop: embed_text is synchronous and may perform network
        # I/O or run a local model.
        vectors = await asyncio.to_thread(lambda: [embed_text(c) for c in raw_chunks])

        points: List[models.PointStruct] = []
        for idx, (text_chunk, vector) in enumerate(zip(raw_chunks, vectors)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": vector},
                    # Payload keys match what dialogue_service reads back (text,
                    # source_file, section, position); a mismatch here surfaces as blank
                    # citations rather than an error.
                    payload={
                        "doc_id": doc_id,
                        "text": text_chunk,
                        "content": text_chunk,
                        "source_file": source_name,
                        "source": source_name,
                        "section": section,
                        "position": idx,
                        "char_count": len(text_chunk),
                        "publication_date": publication_date,
                    },
                )
            )

        await self.qdrant.upsert(collection_name=target, points=points, wait=True)
        logger.info(
            "Ingested %d chunks from '%s' into '%s'", len(points), source_name, target
        )

        return {
            "doc_id": doc_id,
            "source": source_name,
            "collection": target,
            "overview": f"Overview of {source_name}: {content[:200]}...",
            "chunks_count": len(raw_chunks),
            "chunks_written": len(points),
        }
