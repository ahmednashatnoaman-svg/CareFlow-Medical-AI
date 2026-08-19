"""Centralized API Router for history-service exclusively serving /api/v1 routes."""

from fastapi import APIRouter
from careflow.api.v1.router import api_router as api_v1_router

api_router = APIRouter()

# Exclusive versioned router (/api/v1)
api_router.include_router(api_v1_router, prefix="/api/v1")
