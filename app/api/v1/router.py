"""API v1 Router aggregation."""

from fastapi import APIRouter
from app.api.v1.endpoints import conversation, dialogue, health, knowledge, triage

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(triage.router)
api_router.include_router(dialogue.router)
api_router.include_router(conversation.router)
api_router.include_router(knowledge.router)
