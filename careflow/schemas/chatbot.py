"""Pydantic Schemas for Dual-Mode Medical Chatbot.

Defines schemas for:
- Mode 1: Graph RAG Triage & Differential Diagnosis
- Mode 2: WHO Guidelines Dialogue Vector RAG
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- MODE 1: TRIAGE & DIAGNOSIS SCHEMAS ---

class SocratesTracker(BaseModel):
    site: bool = False
    onset: bool = False
    character: bool = False
    radiation: bool = False
    associated_symptoms: bool = False
    time_course: bool = False
    exacerbating_relieving: bool = False
    severity: bool = False


class DiagnosisDetail(BaseModel):
    diagnosis: str
    estimated_probability: str
    urgency_level: str = "Urgent"
    reasoning: str
    graph_evidence: str


class DiagnosticReport(BaseModel):
    top_diagnoses: List[DiagnosisDetail] = []
    triage_recommendation: str
    clinical_summary: Optional[str] = None


class TriageStartRequest(BaseModel):
    session_id: Optional[str] = None
    language: str = Field(default="en", description="'en' or 'ar'")


class TriageStepRequest(BaseModel):
    session_id: str
    message: str = Field(..., description="User's complaint or option choice (e.g., '1', '2', '3' or text)")
    language: Optional[str] = Field(default=None, description="Optional language override ('en' or 'ar')")


class TriageResponse(BaseModel):
    session_id: str
    is_complete: bool = False
    message: str
    options: List[str] = []
    target_symptom: Optional[str] = None
    socrates_tracker: Dict[str, bool] = {}
    socrates_score: int = 0
    turn_count: int = 0
    max_turns: int = 8
    positive_symptoms: List[str] = []
    negated_symptoms: List[str] = []
    stop_reason: Optional[str] = None
    diagnostic_report: Optional[DiagnosticReport] = None


# --- MODE 2: WHO GUIDELINES DIALOGUE VECTOR RAG SCHEMAS ---

class Citation(BaseModel):
    source_file: str
    section: str
    relevance_score: float
    snippet: str


class DialogueChatRequest(BaseModel):
    query: str = Field(..., description="User medical question to answer from WHO guidelines")
    top_k: int = Field(default=5, description="Number of guideline chunks to retrieve")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="List of previous conversation turns [{'role': 'user', 'content': '...'}]"
    )


class DialogueChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[Citation] = []
    chunks_retrieved: int = 0
