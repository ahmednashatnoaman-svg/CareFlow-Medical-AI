"""Unified embedding provider for CareFlow.

Single source of truth for turning text into dense vectors. Previously this logic was
duplicated across `dialogue_service.get_embedding_vector` and
`retrieval_service._embed_text`, each with a *different* provider chain, and both ending
in a deterministic pseudo-random vector.

That silent random fallback is the reason this module exists. Searching a random query
vector against a real BGE-M3 index returns arbitrary chunks with plausible-looking
similarity scores, which the LLM then answers over confidently. The failure is invisible:
no exception, no empty result, just quietly wrong medical retrieval. A loud failure is
strictly safer, so an unavailable embedder now raises instead of inventing numbers.

Providers are tried in order and the first success wins. `EMBEDDING_PROVIDER` pins a
single provider when you need determinism (notably `mock` for offline unit tests, which
must never download a ~2.3GB model).
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from typing import Callable, List, Optional

import httpx

from careflow.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """No embedding provider could produce a usable vector.

    Deliberately fatal. Callers must not substitute a placeholder vector: a wrong vector
    is indistinguishable from a correct one downstream, so retrieval would silently
    degrade to noise rather than reporting an outage.
    """


# Model handles are expensive to build (network fetch + weight load), so they are cached
# process-wide behind a lock rather than per RetrievalEngine/DialogueService instance.
_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def _mock_vector(text: str, size: int) -> List[float]:
    """Deterministic pseudo-random vector. Test fixture only — never a production path.

    Reachable only when EMBEDDING_PROVIDER == "mock" is set explicitly, so it can never be
    silently selected because a real provider happened to be unreachable.
    """
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    return [rng.uniform(-0.1, 0.1) for _ in range(size)]


def _remote_vector(text: str, size: int) -> Optional[List[float]]:
    """Embed via the hosted Modal BGE-M3 microservice."""
    raw_url = settings.EMBEDDER_ENDPOINT_URL
    if not raw_url:
        return None

    base = raw_url.rstrip("/")
    endpoint = base if "/api/v1/embed" in base else f"{base}/api/v1/embed"
    with httpx.Client(timeout=settings.EMBEDDING_TIMEOUT, follow_redirects=True) as client:
        res = client.post(endpoint, json={"text": text, "normalize": True})
        res.raise_for_status()
        data = res.json()
        vec = data.get("embedding")
        if vec is None:
            batch = data.get("embeddings")
            vec = batch[0] if batch else None

    if not vec:
        raise ValueError("remote embedder returned no vector")
    if len(vec) != size:
        # A dimension mismatch means this embedder is not the one the index was built
        # with. Using it anyway produces meaningless cosine scores, so reject it.
        raise ValueError(f"remote embedder returned dim={len(vec)}, expected {size}")
    return [float(x) for x in vec]


def _local_vector(text: str, size: int) -> Optional[List[float]]:
    """Embed with a locally installed sentence-transformers model.

    Optional: the heavy ML stack is intentionally absent from the serverless deployment
    (it blows the 250MB function limit), so an ImportError here is expected in production
    and simply advances to the next provider.
    """
    key = f"st::{settings.EMBEDDING_MODEL}"
    with _cache_lock:
        model = _model_cache.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading local embedding model '%s'", settings.EMBEDDING_MODEL)
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            _model_cache[key] = model

    vec = model.encode(text, normalize_embeddings=True).tolist()  # type: ignore[attr-defined]
    if len(vec) != size:
        raise ValueError(f"local model returned dim={len(vec)}, expected {size}")
    return [float(x) for x in vec]


_PROVIDERS: dict[str, Callable[[str, int], Optional[List[float]]]] = {
    "remote": _remote_vector,
    "local": _local_vector,
    "mock": _mock_vector,
}

# "mock" is excluded from the auto chain on purpose -- it must be opted into explicitly.
_AUTO_CHAIN = ("remote", "local")


def embed_text(text: str, size: Optional[int] = None) -> List[float]:
    """Return a dense embedding for `text`.

    Raises:
        EmbeddingUnavailableError: every configured provider failed. Callers should let
            this propagate; degrading to a placeholder vector would turn a visible outage
            into silently wrong retrieval.
    """
    if not text or not text.strip():
        raise ValueError("cannot embed empty text")

    dim = size or settings.VECTOR_SIZE
    configured = (settings.EMBEDDING_PROVIDER or "auto").strip().lower()
    chain = _AUTO_CHAIN if configured == "auto" else (configured,)

    failures: list[str] = []
    for name in chain:
        provider = _PROVIDERS.get(name)
        if provider is None:
            failures.append(f"{name}: unknown provider")
            continue
        try:
            vec = provider(text, dim)
            if vec:
                logger.debug("Embedded via '%s' (dim=%d)", name, len(vec))
                return vec
            failures.append(f"{name}: not configured")
        except Exception as exc:  # noqa: BLE001 - provider errors are reported in aggregate
            logger.warning("Embedding provider '%s' failed: %s", name, exc)
            failures.append(f"{name}: {exc}")

    raise EmbeddingUnavailableError(
        "No embedding provider produced a vector. Retrieval cannot run without one; "
        "refusing to substitute a placeholder because that would silently return "
        f"irrelevant medical context. Attempts: {'; '.join(failures)}"
    )


def embedding_health() -> dict:
    """Probe the embedder so /health can surface a degraded RAG path before users hit it."""
    try:
        vec = embed_text("health check", settings.VECTOR_SIZE)
        return {"status": "ok", "dimension": len(vec), "provider": settings.EMBEDDING_PROVIDER}
    except Exception as exc:  # noqa: BLE001 - health probes report, never raise
        return {"status": "unavailable", "provider": settings.EMBEDDING_PROVIDER, "error": str(exc)}
