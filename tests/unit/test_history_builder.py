"""Unit tests for MedicalHistoryBuilder service."""

from app.services.history_builder_service import MedicalHistoryBuilder
from app.schemas.history import StructuredMedicalHistory


def test_build_structured_history_includes_risk_factors_and_symptoms():
    builder = MedicalHistoryBuilder()
    state = {
        "chief_complaint": "Chest pain",
        "extracted_entities": {
            "symptoms": ["chest pain", "shortness of breath"],
            "diseases": ["hypertension"],
            "medications": ["aspirin"],
            "allergies": ["penicillin"],
            "family_history": ["father had CAD"],
            "social_history": ["smoker"],
            "risk_factors": ["smoking", "high blood pressure", "family history of CAD"],
            "attributes": {
                "onset": "2 hours ago",
                "severity": "severe",
            },
        },
        "red_flags": ["radiating chest pain"],
    }

    history = builder.build_structured_history(conversation_state=state, overall_interview_score=0.85)

    assert isinstance(history, StructuredMedicalHistory)
    assert history.chief_complaint == "Chest pain"
    assert history.risk_factors == ["smoking", "high blood pressure", "family history of CAD"]
    assert history.history_of_present_illness.associated_symptoms == ["chest pain", "shortness of breath"]
    assert history.interview_score == 0.85
