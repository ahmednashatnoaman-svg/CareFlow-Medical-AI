"""Health and readiness endpoints.

`/health` is a liveness probe: cheap, dependency-free, safe for load balancers.
`/health/ready` is a readiness probe that actually exercises the external dependencies
the RAG pipeline needs, so a degraded retrieval path is visible here rather than being
discovered by a patient mid-consultation.
"""

import asyncio
import logging

from fastapi import APIRouter

from careflow.core.config import settings
from careflow.core.version import API_VERSION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", summary="Liveness probe")
async def health_check() -> dict:
    """Report that the process is up. Performs no I/O."""
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


async def _check_qdrant() -> dict:
    """Confirm the vector store is reachable and the target collection exists."""
    from careflow.services.dialogue_service import dialogue_service

    try:
        client = dialogue_service.get_async_client()
        collections = await asyncio.wait_for(client.get_collections(), timeout=5.0)
        names = [c.name for c in collections.collections]
        target = dialogue_service.collection_name
        return {
            "status": "ok" if target in names else "degraded",
            "collection": target,
            "collection_present": target in names,
        }
    except Exception as exc:  # noqa: BLE001 - probes report status, never raise
        return {"status": "unavailable", "error": str(exc)}


async def _check_embedder() -> dict:
    """Exercise the real embedding chain.

    This is the check that matters most: when no embedder is available the system used to
    fall back to random vectors and keep answering, so retrieval failure was invisible.
    """
    from careflow.services.embedding_service import embedding_health

    return await asyncio.to_thread(embedding_health)


async def _check_knowledge_graph() -> dict:
    """Confirm PrimeKG loaded with a usable number of nodes."""
    from careflow.services.primekg_service import primekg_service

    try:
        nodes = primekg_service.graph.number_of_nodes()
        return {"status": "ok" if nodes > 0 else "degraded", "nodes": nodes}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": str(exc)}


@router.get("/ready", summary="Readiness probe with dependency checks")
async def readiness_check() -> dict:
    """Probe every dependency the RAG pipeline needs.

    Checks run concurrently so the endpoint stays fast. Overall status is the worst
    individual status: `unavailable` on any hard failure, `degraded` if something is
    reachable but not correctly provisioned.
    """
    qdrant, embedder, kg = await asyncio.gather(
        _check_qdrant(), _check_embedder(), _check_knowledge_graph()
    )
    checks = {"qdrant": qdrant, "embedder": embedder, "knowledge_graph": kg}

    statuses = {c["status"] for c in checks.values()}
    overall = "unavailable" if "unavailable" in statuses else ("degraded" if "degraded" in statuses else "ok")

    return {
        "status": overall,
        "service": settings.APP_NAME,
        "version": API_VERSION,
        "environment": settings.APP_ENV,
        "checks": checks,
    }
