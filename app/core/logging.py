"""Structured Logging Module.

Configures application logger and silences extra log noise for clean Uvicorn output.
"""

import logging
import sys
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured production logging."""

    def __init__(self, service_name: str = "careflow-history-service"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
        }

        if hasattr(record, "request_id"):
            log_object["request_id"] = getattr(record, "request_id")

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_object.update(record.extra)

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object)


def setup_logging(
    service_name: str = "careflow-history-service",
    log_level: str = None,
    log_format: str = None,
) -> logging.Logger:
    """Configures application logger and suppresses extra noise for clean Uvicorn logs."""
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "WARNING")
    if log_format is None:
        log_format = os.getenv("LOG_FORMAT", "text")

    # Mute noisy 3rd-party loggers (httpx, httpcore, urllib3, asyncio, qdrant_client)
    for noisy in ["httpx", "httpcore", "urllib3", "asyncio", "qdrant_client"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Attach handler only if LOG_FORMAT=json for production
    if log_format.lower() == "json":
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter(service_name=service_name))
        root_logger.addHandler(handler)

    logger = logging.getLogger("careflow")
    logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
    return logger


logger = setup_logging()
