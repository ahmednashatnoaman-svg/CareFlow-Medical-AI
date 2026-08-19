import os
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/evaluation", tags=["Evaluation Metrics"])

@router.get("")
async def get_evaluation_metrics():
    """Retrieve quantitative evaluation metrics from the static JSON file."""
    # The evaluation file is located in app/static/evaluation_results.json
    results_path = os.path.join(os.path.dirname(__file__), "..", "..", "static", "evaluation_results.json")
    
    try:
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        else:
            return JSONResponse(status_code=404, content={"error": "Evaluation results not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
