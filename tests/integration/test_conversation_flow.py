"""Integration tests for full conversation API workflow."""

import pytest

# Requires live external services (Qdrant, Gemini/Groq) and real API keys, and the
# Gemini free tier caps at 15 req/min -- these exhaust it. Excluded from the default
# run via pyproject addopts; invoke deliberately with `pytest -m integration`.
pytestmark = pytest.mark.integration


def test_full_conversation_lifecycle(client):
    # 1. Start Conversation
    response = client.post(
        "/api/v1/conversation/start",
        json={"patient_id": "patient_test_001", "chief_complaint": "Chest pain"},
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    data = res_json["data"]
    conv_id = data["conversation_id"]
    assert data["status"] == "ACTIVE"
    assert len(data["initial_question_arabic"]) > 0

    # 2. Submit Patient Text Turn (Egyptian Arabic)
    step_resp = client.post(
        "/api/v1/conversation/text",
        json={"conversation_id": conv_id, "text": "حاسس بوجع جامد في صدري من ساعتين وبيسمع في كتفي الشمال"},
    )
    assert step_resp.status_code == 200
    step_json = step_resp.json()
    assert step_json["success"] is True
    step_data = step_json["data"]
    assert step_data["step_index"] == 1
    assert len(step_data["next_question_arabic"]) > 0

    # 3. Query Conversation State
    state_resp = client.get(f"/api/v1/conversation/state?conversation_id={conv_id}")
    assert state_resp.status_code == 200
    state_json = state_resp.json()
    assert state_json["success"] is True
    state_data = state_json["data"]
    assert state_data["step_count"] == 1
    assert len(state_data["turns"]) == 1

    # 4. Finish Conversation & Verify Structured History JSON Output
    finish_resp = client.post(
        "/api/v1/conversation/finish",
        json={"conversation_id": conv_id},
    )
    assert finish_resp.status_code == 200
    finish_json = finish_resp.json()
    assert finish_json["success"] is True
    history_data = finish_json["data"]
    assert "chief_complaint" in history_data
    assert "history_of_present_illness" in history_data
    assert "interview_score" in history_data


def test_multiturn_no_repeated_questions_and_increasing_coverage(client):
    """Verifies that multi-turn history taking generates distinct questions and coverage increases dynamically."""
    # 1. Start session
    start_res = client.post(
        "/api/v1/conversation/start",
        json={"patient_id": "patient_multiturn_999", "chief_complaint": "Abdominal pain"},
    ).json()
    conv_id = start_res["data"]["conversation_id"]

    patient_responses = [
        "وجع شديد في بطني ناحية اليمين تحت من مبارح",  # Turn 1: location & duration & onset
        "الألم شديد جداً 8 من 10 وبيزيد لما بتحرك",     # Turn 2: severity & aggravating
        "الألم زي مغص حاد ومش بيسمع في ضهري",           # Turn 3: character & radiation
        "معنديش سخونية ولا ترجيع ولا أي أعراض تانية",    # Turn 4: associated symptoms
        "الراحة والكمادات الدافية بتهديه شوية",         # Turn 5: relieving factors
    ]

    asked_questions = []
    coverage_scores = []

    for idx, user_input in enumerate(patient_responses, start=1):
        resp = client.post(
            "/api/v1/conversation/text",
            json={"conversation_id": conv_id, "text": user_input},
        )
        assert resp.status_code == 200
        step_data = resp.json()["data"]

        q_en = step_data["next_question_english"]
        cov = step_data["metrics"]["coverage_score"]

        assert step_data["step_index"] == idx
        if q_en:
            assert q_en not in asked_questions, f"Question '{q_en}' was repeated at turn {idx}!"
            asked_questions.append(q_en)
        coverage_scores.append(cov)

    # Assert coverage score strictly increased from early turns to final turns
    assert coverage_scores[-1] > coverage_scores[0], f"Coverage score did not increase: {coverage_scores}"
    assert coverage_scores[-1] >= 0.5, f"Expected final coverage >= 0.5, got {coverage_scores[-1]}"

    # Verify conversation state
    state_res = client.get(f"/api/v1/conversation/state?conversation_id={conv_id}").json()
    assert state_res["data"]["step_count"] == 5
    assert len(state_res["data"]["turns"]) == 5

