"""Knowledge Base Management Endpoint."""

from fastapi import APIRouter, status
from careflow.schemas.common import ApiResponse
from careflow.schemas.conversation import KnowledgeIngestRequest
from careflow.services.knowledge_ingestion_service import KnowledgeIngestionService

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])
ingest_service = KnowledgeIngestionService()


@router.post("/ingest", response_model=ApiResponse[dict])
async def ingest_guideline_document(payload: KnowledgeIngestRequest):
    """Ingests medical guidelines document text into Qdrant collection."""
    res = await ingest_service.ingest_document(
        source_name=payload.source_name,
        content=payload.content,
        section=payload.section or "General Guidelines",
    )
    return ApiResponse(
        success=True,
        message="Medical document ingested into knowledge base",
        data=res,
    )
