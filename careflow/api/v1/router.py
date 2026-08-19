"""API v1 Router aggregation."""

from fastapi import APIRouter
from careflow.api.v1.endpoints import conversation, dialogue, health, knowledge, triage, evaluation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(triage.router)
api_router.include_router(dialogue.router)
api_router.include_router(conversation.router)
api_router.include_router(knowledge.router)
api_router.include_router(evaluation.router)
