"""Conversation API Request & Response Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.history import StructuredMedicalHistory
from app.schemas.state import AnswerOption, ConversationTurn, EvaluationMetrics


class ConversationStartRequest(BaseModel):
    """Payload for starting a new medical history conversation."""

    patient_id: str = Field(..., description="Unique patient identifier")
    chief_complaint: Optional[str] = Field(default=None, description="Initial chief complaint if available")


class ConversationStartData(BaseModel):
    """Data object returned upon starting a conversation."""

    conversation_id: str
    patient_id: str
    status: str
    initial_question_arabic: str
    initial_question_english: str
    options: List[AnswerOption] = Field(default_factory=list)


class ConversationTextRequest(BaseModel):
    """Payload for submitting a text response in Egyptian Arabic or English."""

    conversation_id: str
    text: str = Field(..., description="Patient input text")


class ConversationStepData(BaseModel):
    """Data object returned after processing a turn."""

    conversation_id: str
    step_index: int
    user_input_arabic: str
    user_input_english: str
    next_question_arabic: str
    next_question_english: str
    should_continue: bool
    status: str
    metrics: EvaluationMetrics
    structured_history: Optional[StructuredMedicalHistory] = None
    options: List[AnswerOption] = Field(default_factory=list)
    selected_option_index: Optional[int] = None


class ConversationStateData(BaseModel):
    """Data object returning active conversation state."""

    conversation_id: str
    patient_id: str
    status: str
    step_count: int
    chief_complaint: str
    turns: List[ConversationTurn]
    metrics: EvaluationMetrics
    current_summary: str


class ConversationFinishRequest(BaseModel):
    """Payload to forcefully finish a conversation."""

    conversation_id: str


class KnowledgeIngestRequest(BaseModel):
    """Payload for ingesting clinical guideline text/document."""

    source_name: str
    content: str
    section: Optional[str] = "General Guidelines"


class AutomatedPipelineRequest(BaseModel):
    """Payload for executing the automated multi-turn interview pipeline."""

    patient_id: str = Field(..., description="Unique patient identifier")
    chief_complaint: Optional[str] = Field(default=None, description="Initial chief complaint if available")
    patient_turns: List[str] = Field(
        default_factory=lambda: [
            "حاسس بوجع جامد في صدري من ساعتين وبيسمع في كتفي الشمال",
            "الالم شديد 8 من 10 ونزل عليا عرق بارد مع صعوبة تنفس",
            "لأ معنديش سكر بس باخد علاج لضغط الدم العالي",
        ],
        description="Sequential list of Egyptian Arabic patient statements for automated simulation",
    )
