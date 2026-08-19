"""Triage & Diagnosis API Endpoints (Mode 1 - Graph RAG)."""

import uuid
from fastapi import APIRouter, HTTPException, status
from app.schemas.chatbot import (
    DiagnosticReport,
    TriageResponse,
    TriageStartRequest,
    TriageStepRequest,
)
from app.services.triage_service import triage_service

router = APIRouter(prefix="/triage", tags=["Mode 1: Graph RAG Triage & Diagnosis"])


@router.post("/start", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def start_triage(req: TriageStartRequest) -> TriageResponse:
    """Initializes a new clinical triage interview session."""
    session_id = req.session_id or str(uuid.uuid4())
    session = triage_service.get_or_create_session(session_id=session_id, language=req.language)

    greeting_en = "Hello! I am your clinical triage assistant. What symptoms are you experiencing today?"
    greeting_ar = "أهلاً بيك. ألف سلامة عليك، تقدر تقولي حاسس بإيه أو بتشتكي من إيه النهاردة؟"

    message = greeting_ar if session.language == "ar" else greeting_en

    return TriageResponse(
        session_id=session.session_id,
        is_complete=False,
        message=message,
        options=[],
        target_symptom=None,
        socrates_tracker=session.socrates_tracker,
        socrates_score=0,
        turn_count=0,
        max_turns=session.max_turns,
        positive_symptoms=[],
        negated_symptoms=[],
        diagnostic_report=None,
    )


@router.post("/step", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def step_triage(req: TriageStepRequest) -> TriageResponse:
    """Processes one turn of patient input and advances the Graph RAG triage interview."""
    if not req.message or not req.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    result = await triage_service.process_user_turn(
        session_id=req.session_id,
        user_message=req.message,
        forced_language=req.language,
    )

    report_model = None
    if result.get("diagnostic_report"):
        report_model = DiagnosticReport(**result["diagnostic_report"])

    return TriageResponse(
        session_id=result["session_id"],
        is_complete=result["is_complete"],
        message=result["message"],
        options=result.get("options", []),
        target_symptom=result.get("target_symptom"),
        socrates_tracker=result.get("socrates_tracker", {}),
        socrates_score=result.get("socrates_score", 0),
        turn_count=result.get("turn_count", 0),
        max_turns=result.get("max_turns", 8),
        positive_symptoms=result.get("positive_symptoms", []),
        negated_symptoms=result.get("negated_symptoms", []),
        stop_reason=result.get("stop_reason"),
        diagnostic_report=report_model,
    )


@router.post("/reset", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def reset_triage(req: TriageStartRequest) -> TriageResponse:
    """Resets an existing triage session."""
    session_id = req.session_id or str(uuid.uuid4())
    session = triage_service.reset_session(session_id)
    session.language = req.language or session.language

    greeting_en = "Session reset. What symptoms or medical concerns would you like to evaluate?"
    greeting_ar = "تمت إعادة تعيين الجلسة. ما هي الأعراض أو الشكوى التي تود فحصها اليوم؟"

    message = greeting_ar if session.language == "ar" else greeting_en

    return TriageResponse(
        session_id=session.session_id,
        is_complete=False,
        message=message,
        options=[],
        target_symptom=None,
        socrates_tracker=session.socrates_tracker,
        socrates_score=0,
        turn_count=0,
        max_turns=session.max_turns,
        positive_symptoms=[],
        negated_symptoms=[],
        diagnostic_report=None,
    )
