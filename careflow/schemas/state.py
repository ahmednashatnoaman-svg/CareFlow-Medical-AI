"""Internal Conversation State and Score Pydantic Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractedEntities(BaseModel):
    """Extracted medical entities from patient turn."""

    symptoms: List[str] = Field(default_factory=list)
    diseases: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    body_locations: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    """Detailed evaluation score components from Termination Engine."""

    coverage_score: float = 0.0
    guideline_coverage_score: float = 0.0
    consistency_score: float = 1.0
    red_flag_coverage_score: float = 0.0
    expected_info_gain: float = 1.0
    overall_interview_score: float = 0.0

    collected_attributes_count: int = 0
    required_attributes_count: int = 1
    asked_guideline_questions_count: int = 0
    total_guideline_questions_count: int = 1
    unanswered_red_flags: List[str] = Field(default_factory=list)
    is_emergency: bool = False
    should_terminate: bool = False
    termination_reason: str = ""


class AnswerOption(BaseModel):
    """One selectable answer offered alongside a follow-up question."""

    index: int = Field(default=1, description="1-based position the patient says aloud")
    text_english: str = ""
    text_arabic: str = ""
    is_free_text: bool = Field(default=False, description="Escape hatch for an answer not listed")


class ConversationTurn(BaseModel):
    """Single turn in a medical history conversation."""

    turn_index: int
    user_input_arabic: str = ""
    user_input_english: str = ""
    assistant_output_english: str = ""
    assistant_output_arabic: str = ""
    extracted_entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    timestamp: str = ""
    offered_options: List[AnswerOption] = Field(default_factory=list)
    selected_option_index: Optional[int] = None
    yielded_new_info: bool = True
