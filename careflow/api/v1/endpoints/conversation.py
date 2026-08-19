"""Module 11: Conversation API Endpoints.

REST + WebSocket endpoints for starting, stepping (audio/text), querying state,
fetching structured history, finishing interviews, and real-time streaming.
"""

import logging
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from careflow.core.constants import ERR_CONVERSATION_NOT_FOUND, ERR_INVALID_AUDIO, STATUS_ACTIVE, STATUS_COMPLETED
from careflow.core.dependencies.deps import get_db
from careflow.services.graph.workflow import interview_app
from careflow.crud.conversation_repository import ConversationRepository
from careflow.crud.history_repository import HistoryRepository
from careflow.schemas.common import ApiResponse
from careflow.schemas.conversation import (
    AutomatedPipelineRequest,
    ConversationFinishRequest,
    ConversationStartData,
    ConversationStartRequest,
    ConversationStateData,
    ConversationStepData,
    ConversationTextRequest,
)
from careflow.schemas.history import StructuredMedicalHistory
from careflow.schemas.state import AnswerOption, ConversationTurn, EvaluationMetrics
from careflow.services.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["Conversation"])
conv_manager = ConversationManager()


def _build_options(target: Any) -> List[AnswerOption]:
    """Renders answer choices for the API response with automatic 1-based indexing."""
    if isinstance(target, dict):
        raw_list = target.get("question_options", [])
    elif isinstance(target, list):
        raw_list = target
    else:
        raw_list = []

    opts: List[AnswerOption] = []
    for idx, opt in enumerate(raw_list, start=1):
        if isinstance(opt, AnswerOption):
            opts.append(opt)
        elif isinstance(opt, dict):
            opt_dict = dict(opt)
            if opt_dict.get("index") is None:
                opt_dict["index"] = idx
            opts.append(AnswerOption(**opt_dict))
        elif isinstance(opt, str):
            opts.append(AnswerOption(index=idx, text_english=opt, text_arabic=opt))
    return opts


async def _run_turn(
    conversation_id: str,
    current_state: Dict[str, Any],
    db: AsyncSession,
    fallback_step: int = 1,
) -> Dict[str, Any]:
    """Runs one interview turn through the graph and persists the result.

    Shared by the text, audio, and automated-pipeline paths, which previously carried
    three near-identical copies of this sequence.
    """
    graph_output = await interview_app.ainvoke(current_state)
    graph_output.pop("raw_audio_bytes", None)
    step = graph_output.get("step_count", fallback_step)

    turn = ConversationTurn(
        turn_index=step,
        user_input_arabic=graph_output.get("user_input_arabic", ""),
        user_input_english=graph_output.get("user_input_english", ""),
        assistant_output_english=graph_output.get("generated_question_english", ""),
        assistant_output_arabic=graph_output.get("generated_question_arabic", ""),
        extracted_entities=graph_output.get("extracted_entities", {}),
        offered_options=_build_options(graph_output),
        selected_option_index=graph_output.get("selected_option_index"),
        yielded_new_info=graph_output.get("yielded_new_info", True),
    )

    updated_turns = list(current_state.get("turns", [])) + [turn.model_dump()]
    graph_output["turns"] = updated_turns

    await conv_manager.update_state(conversation_id, graph_output)
    await ConversationRepository(db).update_state(
        conversation_id=conversation_id,
        turns=updated_turns,
        state_snapshot=graph_output,
        step_count=step,
        status=graph_output.get("status", STATUS_ACTIVE),
    )
    return graph_output


def _step_data(conversation_id: str, graph_output: Dict[str, Any], fallback_step: int = 1) -> ConversationStepData:
    """Builds the turn response payload from the post-graph state."""
    return ConversationStepData(
        conversation_id=conversation_id,
        step_index=graph_output.get("step_count", fallback_step),
        user_input_arabic=graph_output.get("user_input_arabic", ""),
        user_input_english=graph_output.get("user_input_english", ""),
        next_question_arabic=graph_output.get("generated_question_arabic", ""),
        next_question_english=graph_output.get("generated_question_english", ""),
        should_continue=graph_output.get("should_continue", True),
        status=graph_output.get("status", STATUS_ACTIVE),
        metrics=EvaluationMetrics(**graph_output.get("evaluation_metrics", {})),
        structured_history=(
            StructuredMedicalHistory(**graph_output["structured_history"])
            if graph_output.get("structured_history")
            else None
        ),
        options=_build_options(graph_output),
        selected_option_index=graph_output.get("selected_option_index"),
    )


@router.post("/start", response_model=ApiResponse[ConversationStartData])
async def start_conversation(
    payload: ConversationStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """Starts a new medical history collection session."""
    repo = ConversationRepository(db)
    conv = await repo.create(
        patient_id=payload.patient_id,
        chief_complaint=payload.chief_complaint,
    )

    initial_q_en = "Hello. What symptom or medical concern brings you in today?"
    initial_q_ar = "أهلاً بك. ايه العرض أو المشكلة الطبية اللي بتعاني منها النهاردة؟"
    initial_options = [
        {"index": 1, "text_english": "Pain", "text_arabic": "وجع أو ألم", "is_free_text": False},
        {"index": 2, "text_english": "Fever", "text_arabic": "سخونية", "is_free_text": False},
        {"index": 3, "text_english": "Cough or breathing trouble", "text_arabic": "كحة أو صعوبة في التنفس", "is_free_text": False},
        {"index": 4, "text_english": "Something else / I'll describe it myself", "text_arabic": "حاجة تانية / هوصفها بنفسي", "is_free_text": True},
    ]

    initial_state: Dict[str, Any] = conv_manager.new_state(
        conversation_id=conv.id,
        patient_id=payload.patient_id,
        chief_complaint=payload.chief_complaint or "",
    )
    initial_state["previous_questions"] = [initial_q_en]
    initial_state["question_options"] = initial_options

    await conv_manager.update_state(conv.id, initial_state)
    await repo.update_state(
        conversation_id=conv.id,
        turns=[],
        state_snapshot=initial_state,
        step_count=0,
        status=STATUS_ACTIVE,
    )

    data = ConversationStartData(
        conversation_id=conv.id,
        patient_id=payload.patient_id,
        status=STATUS_ACTIVE,
        initial_question_arabic=initial_q_ar,
        initial_question_english=initial_q_en,
        options=_build_options(initial_options),
    )

    return ApiResponse(
        success=True,
        message="Conversation session started successfully",
        data=data,
    )


@router.post("/text", response_model=ApiResponse[ConversationStepData])
async def process_text_turn(
    payload: ConversationTextRequest,
    db: AsyncSession = Depends(get_db),
):
    """Processes a patient text turn in Egyptian Arabic or English."""
    repo = ConversationRepository(db)
    conv = await repo.get_by_id(payload.conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": ERR_CONVERSATION_NOT_FOUND, "message": "Conversation session not found"},
        )

    current_state = await conv_manager.get_state(payload.conversation_id)
    current_state["user_input_arabic"] = payload.text
    current_state["user_input_english"] = ""
    current_state["raw_audio_bytes"] = None

    graph_output = await _run_turn(payload.conversation_id, current_state, db)

    if graph_output.get("structured_history"):
        metrics = EvaluationMetrics(**graph_output.get("evaluation_metrics", {}))
        hist_repo = HistoryRepository(db)
        await hist_repo.save_history(
            conversation_id=payload.conversation_id,
            patient_id=conv.patient_id,
            chief_complaint=graph_output.get("chief_complaint", ""),
            structured_history=StructuredMedicalHistory(**graph_output["structured_history"]).model_dump(),
            interview_score=metrics.overall_interview_score,
            completion_status=STATUS_COMPLETED,
        )

    return ApiResponse(
        success=True,
        message="Turn processed successfully",
        data=_step_data(payload.conversation_id, graph_output),
    )


@router.post("/audio", response_model=ApiResponse[ConversationStepData])
async def process_audio_turn(
    conversation_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Processes an uploaded patient audio stream turn."""
    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": ERR_INVALID_AUDIO, "message": "Uploaded audio stream is invalid or empty"},
        )

    repo = ConversationRepository(db)
    conv = await repo.get_by_id(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": ERR_CONVERSATION_NOT_FOUND, "message": "Conversation session not found"},
        )

    current_state = await conv_manager.get_state(conversation_id)
    current_state["raw_audio_bytes"] = audio_bytes
    current_state["user_input_arabic"] = ""
    current_state["user_input_english"] = ""

    graph_output = await _run_turn(conversation_id, current_state, db)

    if graph_output.get("structured_history"):
        metrics = EvaluationMetrics(**graph_output.get("evaluation_metrics", {}))
        hist_repo = HistoryRepository(db)
        await hist_repo.save_history(
            conversation_id=conversation_id,
            patient_id=conv.patient_id,
            chief_complaint=graph_output.get("chief_complaint", ""),
            structured_history=StructuredMedicalHistory(**graph_output["structured_history"]).model_dump(),
            interview_score=metrics.overall_interview_score,
            completion_status=STATUS_COMPLETED,
        )

    return ApiResponse(
        success=True,
        message="Audio turn processed successfully",
        data=_step_data(conversation_id, graph_output),
    )


@router.get("/state", response_model=ApiResponse[ConversationStateData])
async def get_conversation_state(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieves current active conversation state and turn history."""
    state = await conv_manager.get_state(conversation_id)
    repo = ConversationRepository(db)
    conv = await repo.get_by_id(conversation_id)

    if not conv and not state.get("patient_id"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": ERR_CONVERSATION_NOT_FOUND, "message": "Conversation session not found"},
        )

    summary = await conv_manager.summarize_state(conversation_id)

    data = ConversationStateData(
        conversation_id=conversation_id,
        patient_id=state.get("patient_id", conv.patient_id if conv else "unknown"),
        status=state.get("status", STATUS_ACTIVE),
        step_count=state.get("step_count", 0),
        chief_complaint=state.get("chief_complaint", ""),
        turns=[ConversationTurn(**t) for t in state.get("turns", [])],
        metrics=EvaluationMetrics(**state.get("evaluation_metrics", {})),
        current_summary=summary,
    )

    return ApiResponse(
        success=True,
        message="Conversation state retrieved successfully",
        data=data,
    )


@router.get("/history", response_model=ApiResponse[StructuredMedicalHistory])
async def get_structured_history(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Fetches final structured medical history for a completed conversation."""
    hist_repo = HistoryRepository(db)
    history = await hist_repo.get_by_conversation_id(conversation_id)

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": ERR_CONVERSATION_NOT_FOUND, "message": "Structured history record not found"},
        )

    structured_data = StructuredMedicalHistory(**history.structured_history)
    return ApiResponse(
        success=True,
        message="Structured medical history retrieved successfully",
        data=structured_data,
    )


@router.post("/finish", response_model=ApiResponse[StructuredMedicalHistory])
async def finish_conversation(
    payload: ConversationFinishRequest,
    db: AsyncSession = Depends(get_db),
):
    """Forcefully finishes interview and outputs structured medical history JSON."""
    state = await conv_manager.get_state(payload.conversation_id)
    repo = ConversationRepository(db)
    conv = await repo.get_by_id(payload.conversation_id)

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": ERR_CONVERSATION_NOT_FOUND, "message": "Conversation session not found"},
        )

    # Trigger History Builder Node
    state["should_continue"] = False
    state["status"] = STATUS_COMPLETED

    from careflow.services.graph.nodes import history_builder_node
    res = await history_builder_node(state)
    structured_hist_dict = res.get("structured_history", {})

    hist_repo = HistoryRepository(db)
    history = await hist_repo.save_history(
        conversation_id=payload.conversation_id,
        patient_id=conv.patient_id,
        chief_complaint=state.get("chief_complaint", ""),
        structured_history=structured_hist_dict,
        interview_score=structured_hist_dict.get("interview_score", 0.90),
        completion_status=STATUS_COMPLETED,
    )

    await repo.update_state(
        conversation_id=payload.conversation_id,
        turns=state.get("turns", []),
        state_snapshot=state,
        step_count=state.get("step_count", 0),
        status=STATUS_COMPLETED,
    )

    return ApiResponse(
        success=True,
        message="Interview finished and structured history generated",
        data=StructuredMedicalHistory(**history.structured_history),
    )


@router.websocket("/ws")
async def websocket_conversation_endpoint(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
):
    """Realtime WebSocket endpoint for streaming patient audio/text turns.

    Shares `_run_turn` with the /text and /audio REST endpoints so every turn is
    persisted (Redis/in-memory state + DB row) exactly like the REST paths -- a
    prior version invoked the graph and discarded the result, which meant every
    WebSocket turn silently replayed from the same stale starting state.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "")
            conv_id = data.get("conversation_id", "")

            current_state = await conv_manager.get_state(conv_id)
            current_state["user_input_arabic"] = user_text
            current_state["user_input_english"] = ""
            current_state["raw_audio_bytes"] = None

            graph_output = await _run_turn(conv_id, current_state, db, fallback_step=current_state.get("step_count", 0) + 1)

            if graph_output.get("structured_history"):
                conv = await ConversationRepository(db).get_by_id(conv_id)
                if conv:
                    metrics = EvaluationMetrics(**graph_output.get("evaluation_metrics", {}))
                    await HistoryRepository(db).save_history(
                        conversation_id=conv_id,
                        patient_id=conv.patient_id,
                        chief_complaint=graph_output.get("chief_complaint", ""),
                        structured_history=StructuredMedicalHistory(**graph_output["structured_history"]).model_dump(),
                        interview_score=metrics.overall_interview_score,
                        completion_status=STATUS_COMPLETED,
                    )

            await websocket.send_json(
                {
                    "success": True,
                    "conversation_id": conv_id,
                    "step_index": graph_output.get("step_count", 1),
                    "next_question_arabic": graph_output.get("generated_question_arabic", ""),
                    "next_question_english": graph_output.get("generated_question_english", ""),
                    "options": [o.model_dump() for o in _build_options(graph_output)],
                    "selected_option_index": graph_output.get("selected_option_index"),
                    "should_continue": graph_output.get("should_continue", True),
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}", exc_info=True)
        await websocket.close()


@router.post("/pipeline/automated", response_model=ApiResponse[StructuredMedicalHistory])
async def run_automated_pipeline_endpoint(
    payload: AutomatedPipelineRequest,
    db: AsyncSession = Depends(get_db),
):
    """Executes the full automated multi-turn clinical interview pipeline in a single REST API call."""
    repo = ConversationRepository(db)
    hist_repo = HistoryRepository(db)

    conv = await repo.create(
        patient_id=payload.patient_id,
        chief_complaint=payload.chief_complaint,
    )

    state: Dict[str, Any] = conv_manager.new_state(
        conversation_id=conv.id,
        patient_id=payload.patient_id,
        chief_complaint=payload.chief_complaint or "",
    )
    state["previous_questions"] = ["What symptom or medical concern brings you in today?"]
    await conv_manager.update_state(conv.id, state)

    for turn_num, input_text in enumerate(payload.patient_turns, start=1):
        state["user_input_arabic"] = input_text
        state["raw_audio_bytes"] = None

        state = await _run_turn(conv.id, state, db, fallback_step=turn_num)

        if not state.get("should_continue", True):
            break

    if not state.get("structured_history"):
        from careflow.services.graph.nodes import history_builder_node

        state["should_continue"] = False
        state["status"] = STATUS_COMPLETED
        res = await history_builder_node(state)
        state["structured_history"] = res.get("structured_history", {})

    structured_dict = state.get("structured_history", {})
    final_history = StructuredMedicalHistory(**structured_dict)

    history = await hist_repo.save_history(
        conversation_id=conv.id,
        patient_id=payload.patient_id,
        chief_complaint=payload.chief_complaint or "Not specified",
        structured_history=final_history.model_dump(),
        interview_score=final_history.interview_score,
        completion_status=STATUS_COMPLETED,
    )

    return ApiResponse(
        success=True,
        message="Automated pipeline executed successfully and structured history generated",
        data=StructuredMedicalHistory(**history.structured_history),
    )
