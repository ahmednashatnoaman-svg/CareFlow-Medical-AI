"""Middleware package initialization."""

from app.core.middleware.logging_middleware import RequestIDLoggingMiddleware

__all__ = ["RequestIDLoggingMiddleware"]
