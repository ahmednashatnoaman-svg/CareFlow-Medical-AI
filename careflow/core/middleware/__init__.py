"""Middleware package initialization."""

from careflow.core.middleware.logging_middleware import RequestIDLoggingMiddleware

__all__ = ["RequestIDLoggingMiddleware"]
