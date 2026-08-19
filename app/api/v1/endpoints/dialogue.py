"""WHO Guidelines Dialogue RAG API Endpoints (Mode 2 - Vector RAG)."""

import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.chatbot import Citation, DialogueChatRequest, DialogueChatResponse
from app.services.dialogue_service import dialogue_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialogue", tags=["Mode 2: WHO Guidelines Dialogue Vector RAG"])


@router.post("/chat", response_model=DialogueChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_who_guidelines(req: DialogueChatRequest) -> DialogueChatResponse:
    """Answers medical inquiries grounded strictly in WHO Guidelines using Qdrant vector retrieval."""
    if not req.query or not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        result = await dialogue_service.answer_question(
            query=req.query,
            top_k=req.top_k,
            conversation_history=req.conversation_history,
        )

        citations = [
            Citation(
                source_file=s["source_file"],
                section=s["section"],
                relevance_score=s["relevance_score"],
                snippet=s["snippet"],
            )
            for s in result.get("sources", [])
        ]

        return DialogueChatResponse(
            query=result["query"],
            answer=result["answer"],
            sources=citations,
            chunks_retrieved=result.get("chunks_retrieved", len(citations)),
        )

    except Exception as e:
        logger.error("WHO Guidelines Dialogue error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process inquiry: {str(e)}",
        )
