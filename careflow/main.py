"""FastAPI Application Main Entry Point.

Initializes FastAPI app, middleware, routers, CORS, static files, and lifespan handlers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from careflow.api.router import api_router
from careflow.core.config import settings
from careflow.core.logging import logger
from careflow.core.dependencies.deps import init_db
from careflow.core.middleware.logging_middleware import RequestIDLoggingMiddleware
from careflow.schemas.common import ApiErrorResponse
from careflow.core.version import API_VERSION
from careflow.services.primekg_service import primekg_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application Lifespan Context Manager handling startup & shutdown."""
    settings.init_tracing()
    logger.info("Initializing PrimeKG clinical knowledge graph: %d nodes", primekg_service.graph.number_of_nodes())
    try:
        await init_db()
    except Exception as e:
        logger.warning("Database init skipped: %s", e)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=API_VERSION,
    description="Dual-Mode Medical Chatbot: Graph RAG Triage & WHO Guidelines Vector RAG",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS. A wildcard origin cannot be combined with credentialed requests (the Fetch spec
# forbids it and browsers reject the response), so credentials are enabled only when an
# explicit origin allowlist is configured.
_cors_origins = settings.allowed_origins_list
_wildcard = "*" in _cors_origins
if _wildcard and settings.APP_ENV == "production":
    logger.warning(
        "ALLOWED_ORIGINS is '*' in production. Set an explicit origin allowlist "
        "(e.g. https://your-app.vercel.app) to enable credentialed requests."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _wildcard,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Add Request ID & Access Logging Middleware
app.add_middleware(RequestIDLoggingMiddleware)

# Register API Router
app.include_router(api_router)

@app.get("/", tags=["Service"])
async def root_index():
    """Service descriptor. The user-facing UI is the Next.js app, not this API."""
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": API_VERSION,
        "docs": "/docs",
        "api_base": "/api/v1",
    }


@app.get("/health", tags=["Health"])
async def root_health():
    """Root health check endpoint."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": API_VERSION,
        "environment": settings.APP_ENV,
        "modes": {
            "mode_1_triage_graph_rag": "active",
            "mode_2_who_dialogue_vector_rag": "active",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "Unhandled exception on %s (request_id=%s): %s",
        request.url.path,
        request_id,
        exc,
        exc_info=True,
    )
    # Upstream driver/SDK exceptions routinely embed connection strings and API keys in
    # their message, so the raw text is exposed only under DEBUG. The request id gives
    # users something actionable to quote without leaking internals.
    detail = str(exc) if settings.DEBUG else "An internal server error occurred"
    if request_id and not settings.DEBUG:
        detail = f"{detail} (request_id={request_id})"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiErrorResponse(
            success=False,
            error_code="INTERNAL_SERVER_ERROR",
            message=detail,
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("careflow.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
