import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application Settings dataclass."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Core Settings
    APP_NAME: str = os.getenv("APP_NAME", "CareFlow AI Medical History Collection Service")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    API_DEFAULT_VERSION: str = os.getenv("API_DEFAULT_VERSION", "v1")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    ENABLE_DIRECT_GEMINI_TESTING: bool = os.getenv("ENABLE_DIRECT_GEMINI_TESTING", "True").lower() in ("true", "1", "t")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # Database (Supabase Cloud PostgreSQL) Settings
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL", None)
    SUPABASE_ANON_KEY: Optional[str] = os.getenv("SUPABASE_ANON_KEY", None)
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY", None)
    SUPABASE_PROJECT_REF: Optional[str] = os.getenv("SUPABASE_PROJECT_REF", None)

    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: Optional[str] = os.getenv("POSTGRES_PASSWORD", None)
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "postgres")
    POSTGRES_HOST: Optional[str] = os.getenv("POSTGRES_HOST", None)
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", None)

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str) and v.strip():
            url = v.strip()
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        return None

    # Cache Settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Triage session persistence.
    #   auto   -> Redis when APP_ENV=production and REDIS_URL is reachable, else in-memory
    #   redis  -> always Redis (required for serverless / multi-replica correctness)
    #   memory -> always in-memory (single-process only)
    SESSION_BACKEND: str = os.getenv("SESSION_BACKEND", "auto")
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    SESSION_MAX: int = int(os.getenv("SESSION_MAX", "1000"))

    # Vector DB (Qdrant) Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
    QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH", None)
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_DOCUMENTS: str = os.getenv("QDRANT_COLLECTION_DOCUMENTS", "medical_documents")
    QDRANT_COLLECTION_CHUNKS: str = os.getenv("QDRANT_COLLECTION_CHUNKS", "medical_chunks")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "who_guidelines")

    # LLM, ASR & Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1024"))
    # Provider chain for careflow.services.embedding_service.
    #   auto   -> remote (Modal), then local (sentence-transformers)
    #   remote -> hosted endpoint only
    #   local  -> local model only
    #   mock   -> deterministic fake vectors; TESTS ONLY, never a production value.
    # "mock" is deliberately excluded from the auto chain so an unreachable embedder can
    # never silently degrade retrieval to random vectors.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "auto")
    EMBEDDING_TIMEOUT: float = float(os.getenv("EMBEDDING_TIMEOUT", "15.0"))
    SBG_API_KEY: str = os.getenv("SBG_API_KEY", "mock-key")
    SBG_BASE_URL: str = os.getenv("SBG_BASE_URL", "http://apiaccess.iti.net.eg")
    SBG_MODEL: str = os.getenv("SBG_MODEL", "openai.gpt-oss-120b-1:0")
    SBG_MAX_TOKENS: int = int(os.getenv("SBG_MAX_TOKENS", "2048"))
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", None)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # Used only by scripts/evaluate_rag.py's ragas judge (answer_similarity /
    # answer_relevancy need a real embedder, separate from the app's own retrieval
    # embedder in embedding_service.py). "models/embedding-001" -- the previous hardcoded
    # value -- 404s on the current Gemini API; verify with `curl
    # generativelanguage.googleapis.com/v1beta/models` before changing this again.
    RAGAS_JUDGE_EMBEDDING_MODEL: str = os.getenv(
        "RAGAS_JUDGE_EMBEDDING_MODEL", "models/gemini-embedding-001"
    )
    PRIMEKG_GRAPH_PATH: str = os.getenv("PRIMEKG_GRAPH_PATH", "data/primekg_clinical_graph.pkl")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    ASR_ENDPOINT_URL: str = os.getenv(
        "ASR_ENDPOINT_URL",
        "https://hossam-elsherbiny95--careflow-asr-service-fastapi-app.modal.run",
    )
    EMBEDDER_ENDPOINT_URL: str = os.getenv(
        "EMBEDDER_ENDPOINT_URL",
        "https://hossam-elsherbiny95--careflow-embedder-service-fastapi-app.modal.run",
    )

    # Web Disease-Info Search (Mayo Clinic) Settings
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "True").lower() in ("true", "1", "t")
    MAYO_CLINIC_SEARCH_URL: str = os.getenv(
        "MAYO_CLINIC_SEARCH_URL",
        "https://www.mayoclinic.org/diseases-conditions/search-results",
    )
    WEB_SEARCH_TIMEOUT: float = float(os.getenv("WEB_SEARCH_TIMEOUT", "10.0"))
    WEB_SEARCH_MAX_DISEASES_PER_TURN: int = int(os.getenv("WEB_SEARCH_MAX_DISEASES_PER_TURN", "2"))

    # Tracing & Observability (LangSmith) Settings
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "True").lower() in ("true", "1", "t")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY", None)
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "careflow-history-service")

    # --- Operational tuning -----------------------------------------------------
    # Values that legitimately differ per environment (network conditions, quota,
    # latency budget). Previously these were literals scattered across the service
    # layer, so tuning a timeout meant hunting through five files.
    QDRANT_TIMEOUT: float = float(os.getenv("QDRANT_TIMEOUT", "10.0"))
    QDRANT_SEARCH_TIMEOUT: float = float(os.getenv("QDRANT_SEARCH_TIMEOUT", "3.0"))
    HEALTH_CHECK_TIMEOUT: float = float(os.getenv("HEALTH_CHECK_TIMEOUT", "5.0"))
    SPEECH_TIMEOUT: float = float(os.getenv("SPEECH_TIMEOUT", "30.0"))

    # Retrieval shape. CANDIDATE_K is the wide first pass fed to the cross-encoder;
    # RERANK_TOP_N is what survives into the generation context.
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "10"))
    # Raised 0.20 -> 0.55 from benchmark evidence (scripts/benchmark_retrieval.py --sweep).
    # At 0.20 every query returned the full k=5, but only 1 chunk was relevant
    # (precision@5 0.200) -- the other 4 were noise from the administrative corpus, which
    # the generator still had to read. Sweeping the threshold on the 8 labeled clinical
    # queries: 0.20 -> 0.200 (5.0 chunks/query), 0.50 -> 0.692 (2.0), 0.55 -> 1.000 (1.0),
    # 0.60 -> 1.000 (1.0), 0.65 -> 0.625 (0.62, and keyword recall drops from 8/8 to 5/8:
    # past this point real answer content starts getting cut, not just noise). Keyword
    # recall held at 8/8 for every threshold through 0.60, so nothing clinically needed
    # was lost getting to precision@5 = 1.000. 0.55 is chosen over 0.60 for the larger
    # margin to that 0.65 cliff -- both score 1.000 on this benchmark, so there is no
    # accuracy reason to prefer the riskier value. The sweep covers 8 queries against 3
    # clinical documents; correct matches score 0.61-0.78 against ~0.42-0.49 for noise.
    # Retune with the benchmark after expanding the corpus.
    RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.55"))
    TRIAGE_GUIDELINE_TOP_K: int = int(os.getenv("TRIAGE_GUIDELINE_TOP_K", "3"))

    # Citation snippet length shown in the UI.
    CITATION_SNIPPET_CHARS: int = int(os.getenv("CITATION_SNIPPET_CHARS", "300"))

    # Interview Termination Engine Parameters
    MIN_QUESTIONS: int = int(os.getenv("MIN_QUESTIONS", "3"))
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "8"))
    MIN_SOCRATES_SCORE: int = int(os.getenv("MIN_SOCRATES_SCORE", "5"))
    CONFIDENCE_MARGIN_THRESHOLD: float = float(os.getenv("CONFIDENCE_MARGIN_THRESHOLD", "0.40"))
    MAX_ENTROPY_THRESHOLD: float = float(os.getenv("MAX_ENTROPY_THRESHOLD", "1.20"))
    MIN_COVERAGE_THRESHOLD: float = float(os.getenv("MIN_COVERAGE_THRESHOLD", "0.90"))
    MIN_GUIDELINE_THRESHOLD: float = float(os.getenv("MIN_GUIDELINE_THRESHOLD", "0.90"))
    MIN_INTERVIEW_SCORE_THRESHOLD: float = float(os.getenv("MIN_INTERVIEW_SCORE_THRESHOLD", "0.85"))
    MAX_INFO_GAIN_THRESHOLD: float = float(os.getenv("MAX_INFO_GAIN_THRESHOLD", "0.20"))

    # Interview Score Component Weights
    WEIGHT_COVERAGE: float = float(os.getenv("WEIGHT_COVERAGE", "0.40"))
    WEIGHT_GUIDELINE: float = float(os.getenv("WEIGHT_GUIDELINE", "0.25"))
    WEIGHT_CONSISTENCY: float = float(os.getenv("WEIGHT_CONSISTENCY", "0.15"))
    WEIGHT_RED_FLAGS: float = float(os.getenv("WEIGHT_RED_FLAGS", "0.10"))
    WEIGHT_INFO_GAIN: float = float(os.getenv("WEIGHT_INFO_GAIN", "0.10"))

    # Spoken answer-option matching
    OPTION_MATCH_THRESHOLD: float = float(os.getenv("OPTION_MATCH_THRESHOLD", "0.80"))
    MAX_ANSWER_OPTIONS: int = int(os.getenv("MAX_ANSWER_OPTIONS", "5"))

    # Auth (JWT bearer token verification)
    JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY", None)
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    def init_tracing(self) -> None:
        """Sets up LangSmith / LangChain tracing environment variables."""
        if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_TRACING"] = str(self.LANGSMITH_TRACING).lower()
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.LANGSMITH_TRACING).lower()
            os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT


settings = Settings()
settings.init_tracing()
