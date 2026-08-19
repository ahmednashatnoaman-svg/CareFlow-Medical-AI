"""Structured Medical History Pydantic Schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HistoryOfPresentIllness(BaseModel):
    """Details of the present illness (HPI)."""

    onset: Optional[str] = Field(default="", description="When symptoms started")
    duration: Optional[str] = Field(default="", description="Total duration of symptoms")
    location: Optional[str] = Field(default="", description="Anatomical location of symptom")
    severity: Optional[str] = Field(default="", description="Severity rating or narrative")
    character: Optional[str] = Field(default="", description="Quality of symptom e.g. sharp, dull")
    radiation: Optional[str] = Field(default="", description="Radiation location if applicable")
    associated_symptoms: List[str] = Field(default_factory=list, description="Other accompanying symptoms")
    aggravating_factors: List[str] = Field(default_factory=list, description="Factors worsening symptom")
    relieving_factors: List[str] = Field(default_factory=list, description="Factors alleviating symptom")


class StructuredMedicalHistory(BaseModel):
    """Full structured medical history output model."""

    chief_complaint: str = Field(default="", description="Primary complaint of patient")
    history_of_present_illness: HistoryOfPresentIllness = Field(default_factory=HistoryOfPresentIllness)
    past_medical_history: List[str] = Field(default_factory=list, description="Known past medical conditions")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    allergies: List[str] = Field(default_factory=list, description="Known drug/food allergies")
    family_history: List[str] = Field(default_factory=list, description="Relevant family health history")
    social_history: List[str] = Field(default_factory=list, description="Smoking, alcohol, occupation, etc.")
    risk_factors: List[str] = Field(default_factory=list, description="Extracted risk factors")
    red_flags: List[str] = Field(default_factory=list, description="Evaluated or triggered red flags")
    interview_score: float = Field(default=0.0, description="Overall Interview Completion Score (0.0 to 1.0)")
