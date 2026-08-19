"""FastAPI Application Main Entry Point.

Initializes FastAPI app, middleware, routers, CORS, static files, and lifespan handlers.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api.router import api_router
from app.core.config import settings
from app.core.logging import logger
from app.core.dependencies.deps import init_db
from app.core.middleware.logging_middleware import RequestIDLoggingMiddleware
from app.schemas.common import ApiErrorResponse
from app.services.primekg_service import primekg_service


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
    version="2.5.0",
    description="Dual-Mode Medical Chatbot: Graph RAG Triage & WHO Guidelines Vector RAG",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request ID & Access Logging Middleware
app.add_middleware(RequestIDLoggingMiddleware)

# Register API Router
app.include_router(api_router)

# Mount Static Files for Web Interface
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["UI"])
async def root_index():
    """Serves the interactive web interface."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": "2.5.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def root_health():
    """Root health check endpoint."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": "2.5.0",
        "environment": settings.APP_ENV,
        "modes": {
            "mode_1_triage_graph_rag": "active",
            "mode_2_who_dialogue_vector_rag": "active",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiErrorResponse(
            success=False,
            error_code="INTERNAL_SERVER_ERROR",
            message=str(exc) if settings.DEBUG else "An internal server error occurred",
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
