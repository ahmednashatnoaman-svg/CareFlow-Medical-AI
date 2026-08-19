"""Unit tests for Retrieval Engine and Reranker (Modules 6 & 7)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from careflow.services import embedding_service
from careflow.services.reranker_service import CrossEncoderReranker
from careflow.services.retrieval_service import RetrievalEngine


def test_hierarchical_retrieval():
    mock_qdrant = AsyncMock()
    mock_hit = MagicMock()
    mock_hit.payload = {
        "chunk_id": "chunk-101",
        "source": "NICE Guideline CG95 - Chest Pain",
        "section": "Acute Coronary Syndromes",
        "content": "Evaluate chest pain onset and radiation to arm.",
    }
    mock_hit.score = 0.95
    mock_res = MagicMock()
    mock_res.points = [mock_hit]
    mock_qdrant.query_points.return_value = mock_res

    engine = RetrievalEngine(qdrant_client=mock_qdrant)
    engine._embed_text = MagicMock(return_value=[0.01] * 1024)
    query = "chest pain radiating to arm with shortness of breath"
    chunks = asyncio.run(engine.search_chunks(query, top_k=5))

    assert len(chunks) > 0
    assert any("Chest" in c["source"] for c in chunks)


def test_cross_encoder_rerank_uses_injected_model():
    # Inject a stub model instead of letting the service download the real cross-encoder
    # from the network -- this is a unit test and must be deterministic/offline.
    stub_model = MagicMock()
    stub_model.predict.return_value = [0.1, 0.9]  # scores chunk 2 (chest pain) higher

    reranker = CrossEncoderReranker(model=stub_model)
    query = "chest pain shortness of breath"
    chunks = [
        {"content": "General headache evaluation guidelines", "score": 0.8},
        {"content": "Acute chest pain evaluation and dyspnea symptoms", "score": 0.7},
    ]

    reranked = asyncio.run(reranker.rerank(query, chunks, top_n=2))
    assert len(reranked) == 2
    assert "chest pain" in reranked[0]["content"].lower()
    stub_model.predict.assert_called_once()


def test_cross_encoder_rerank_falls_back_to_heuristic_when_model_unavailable():
    # model=None with a name that cannot be loaded (no network access assumed here) must
    # degrade to the lexical-overlap heuristic rather than raising.
    reranker = CrossEncoderReranker(model_name="this-model-does-not-exist/invalid")
    query = "chest pain shortness of breath"
    chunks = [
        {"content": "General headache evaluation guidelines", "score": 0.8},
        {"content": "Acute chest pain evaluation and dyspnea symptoms", "score": 0.7},
    ]

    reranked = asyncio.run(reranker.rerank(query, chunks, top_n=2))
    assert len(reranked) == 2
    assert "chest pain" in reranked[0]["content"].lower()


def test_embed_text_returns_correct_vector_dimension(monkeypatch):
    # Pin the mock provider. Without this the auto chain reaches the local
    # sentence-transformers provider and downloads BAAI/bge-m3 (~2.3GB), which OOM-killed
    # the whole pytest process (exit 137) -- a unit test must never touch the network.
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "mock")
    engine = RetrievalEngine()
    vec = engine._embed_text("chest pain radiating to arm")
    assert isinstance(vec, list)
    assert len(vec) == 1024


def test_embed_text_raises_when_no_provider_available(monkeypatch):
    """An unavailable embedder must fail loudly, never return a usable-looking vector.

    Regression guard for the original behaviour: both embedding paths fell back to a
    deterministic random vector, so retrieval silently returned arbitrary chunks with
    plausible similarity scores instead of reporting the outage.
    """
    monkeypatch.setattr(embedding_service.settings, "EMBEDDING_PROVIDER", "remote")
    monkeypatch.setattr(embedding_service.settings, "EMBEDDER_ENDPOINT_URL", "")

    engine = RetrievalEngine()
    with pytest.raises(embedding_service.EmbeddingUnavailableError):
        engine._embed_text("chest pain")

