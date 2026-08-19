"""Quantitative evaluation metrics endpoint.

Serves the artifact written by `scripts/evaluate_rag.py`. Retrieval and generation
metrics are reported separately so each stage of the pipeline can be validated on its
own, per the project's evaluation criteria.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Evaluation Metrics"])

# careflow/api/v1/endpoints/evaluation.py -> parents[3] == careflow/
# The previous version used two `..` segments and resolved to careflow/api/static/,
# which does not exist, so this endpoint returned 404 unconditionally and the dashboard
# silently rendered its hardcoded placeholder numbers instead.
RESULTS_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "evaluation_results.json"


@router.get("", summary="Retrieve quantitative RAG evaluation metrics")
async def get_evaluation_metrics() -> dict:
    """Return the most recent evaluation run.

    A 404 here means the evaluation has not been run yet -- callers must surface that as
    "not measured" rather than falling back to invented figures.
    """
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No evaluation results found. Run `python scripts/evaluate_rag.py` to "
                "generate them."
            ),
        )

    try:
        with RESULTS_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Evaluation results file is malformed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation results file is present but not valid JSON.",
        ) from exc
