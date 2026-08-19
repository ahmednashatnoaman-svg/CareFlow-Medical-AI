"""LangGraph State Definition for Clinical Interview Workflow."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.schemas.state import EvaluationMetrics, ExtractedEntities


class InterviewState(TypedDict, total=False):
    """State schema for the LangGraph clinical interview state machine."""

    conversation_id: str
    patient_id: str
    step_count: int
    raw_audio_bytes: Optional[bytes]
    user_input_arabic: str
    user_input_english: str
    chief_complaint: str
    extracted_entities: Dict[str, Any]
    symptoms: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_degraded: bool
    reranked_chunks: List[Dict[str, Any]]
    web_evidence: List[Dict[str, Any]]
    web_searched_diseases: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    previous_questions: List[str]
    previous_answers: List[str]
    asked_questions: List[Dict[str, Any]]
    exhausted_attributes: List[str]
    missing_info: List[str]
    generated_question_english: str
    generated_question_arabic: str
    question_options: List[Dict[str, Any]]
    selected_option_index: Optional[int]
    option_resolved: bool
    yielded_new_info: bool
    evaluation_metrics: Dict[str, Any]
    should_continue: bool
    status: str
    structured_history: Optional[Dict[str, Any]]
    turn_log: Dict[str, Any]
