"""Unit tests for Interview Termination Engine (Module 9)."""

import pytest
from app.services.termination_engine import InterviewTerminationEngine


def test_coverage_score():
    engine = InterviewTerminationEngine()
    extracted = {
        "onset": "2 hours ago",
        "duration": "2 hours",
        "location": "Chest",
        "severity": "8/10",
        "character": "crushing",
        "radiation": "left arm",
        "associated_symptoms": ["dyspnea", "sweating"],
        "aggravating_factors": ["exertion"],
        "relieving_factors": ["rest"],
    }
    score = engine.compute_coverage_score(extracted)
    assert score == 1.0

    partial_extracted = {"onset": "2 hours ago", "location": "Chest"}
    partial_score = engine.compute_coverage_score(partial_extracted)
    assert 0.0 < partial_score < 1.0

    synonym_extracted = {
        "attributes": {
            "time_of_onset": "2 hours ago",
            "body_location": "Lower abdomen",
            "severity_level": "7/10",
            "worsening_factors": ["movement"],
            "alleviating_factors": ["rest"],
        }
    }
    syn_score = engine.compute_coverage_score(synonym_extracted)
    assert syn_score >= (5.0 / 9.0)


def test_red_flag_coverage():
    engine = InterviewTerminationEngine()
    chief = "Severe Chest Pain"
    answered = [
        "Does the pain radiate to your arm, neck, or jaw?",
        "Are you experiencing any shortness of breath, dizziness, or sweating?",
        "Do you have a history of hypertension or diabetes?",
    ]
    score, unanswered = engine.compute_red_flag_coverage(chief, answered)
    assert isinstance(score, float)
    assert isinstance(unanswered, list)


def test_interview_score_and_stop_policy_under_min_questions():
    engine = InterviewTerminationEngine(min_questions=5)
    metrics = engine.evaluate_interview(
        question_count=2,  # < min_questions 5
        chief_complaint="Chest Pain",
        extracted_attributes={"onset": "today", "location": "chest"},
        answered_red_flags=["radiation", "dyspnea", "syncope", "sweating", "risk factors"],
        conversation_history=[],
    )

    assert metrics.should_terminate is False
    assert "Incomplete medical history" in metrics.termination_reason or "Mandatory red flags" in metrics.termination_reason


def test_interview_score_and_stop_policy_termination_reached():
    engine = InterviewTerminationEngine(min_questions=5, min_score=0.70)
    full_attrs = {
        "onset": "2 hours ago",
        "duration": "2 hours",
        "location": "chest",
        "severity": "8/10",
        "character": "crushing",
        "radiation": "left shoulder",
        "associated_symptoms": ["sweating"],
        "aggravating_factors": ["exertion"],
        "relieving_factors": ["none"],
    }
    red_flags = [
        "radiation to jaw/arm",
        "syncope or dizziness",
        "dyspnea / shortness of breath",
        "diaphoresis / profuse sweating",
        "cardiovascular risk factors",
    ]

    metrics = engine.evaluate_interview(
        question_count=6,
        chief_complaint="Chest Pain",
        extracted_attributes=full_attrs,
        answered_red_flags=red_flags,
        conversation_history=[],
    )

    assert metrics.coverage_score == 1.0
    assert metrics.should_terminate is True


def test_interview_hard_maximum_questions_termination():
    engine = InterviewTerminationEngine(max_questions=10)
    metrics = engine.evaluate_interview(
        question_count=10,
        chief_complaint="Chest Pain",
        extracted_attributes={},
        answered_red_flags=[],
        conversation_history=[],
    )

    assert metrics.should_terminate is True
    assert "Reached hard maximum of 10 questions." in metrics.termination_reason

