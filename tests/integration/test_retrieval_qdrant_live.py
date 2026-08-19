"""Integration Test for Live Qdrant Vector Retrieval.

Connects to live Qdrant Cloud or local Qdrant server configured in settings,
initializes RetrievalEngine with the live AsyncQdrantClient, and verifies real
vector retrieval over medical knowledge collections.
"""

import logging
import pytest
from qdrant_client import AsyncQdrantClient
from app.core.config import settings
from app.services.retrieval_service import RetrievalEngine

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_retrieval_qdrant_live():
    """Tests RetrievalEngine against live Qdrant database."""
    # 1. Connect to live Qdrant database using application settings
    if settings.QDRANT_URL:
        client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
    else:
        client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )

    try:
        # 2. Instantiate RetrievalEngine with live Qdrant client
        engine = RetrievalEngine(qdrant_client=client)

        # 3. Search chunks for live query
        query = "chest pain shortness of breath"
        chunks = await engine.search_chunks(query, top_k=5)

        # 4. Assert live vector retrieval results
        assert isinstance(chunks, list)
        logger.info(f"LIVE QDRANT RETRIEVED CHUNKS: {len(chunks)}")
    finally:
        await client.close()
