"""Integration tests for live audio voice pipeline execution via API."""

import io


def test_live_audio_voice_pipeline(client):
    """Tests full live voice pipeline with audio file upload, ASR transcription, turn processing, and termination."""
    # 1. Start Conversation
    start_resp = client.post(
        "/api/v1/conversation/start",
        json={"patient_id": "patient_voice_test_001", "chief_complaint": "Acute central chest pain"},
    )
    assert start_resp.status_code == 200
    conv_id = start_resp.json()["data"]["conversation_id"]

    # 2. Upload Audio File (Simulated WAV stream)
    dummy_wav = io.BytesIO(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    audio_resp = client.post(
        f"/api/v1/conversation/audio?conversation_id={conv_id}",
        files={"file": ("sample_patient_voice.wav", dummy_wav, "audio/wav")},
    )
    assert audio_resp.status_code == 200
    audio_json = audio_resp.json()
    assert audio_json["success"] is True
    step_data = audio_json["data"]
    assert step_data["step_index"] == 1
    assert isinstance(step_data["user_input_arabic"], str)
    assert len(step_data["next_question_arabic"]) > 0
    assert step_data["should_continue"] in (True, False)

    # 3. Finish Interview & Verify History Generation
    finish_resp = client.post(
        "/api/v1/conversation/finish",
        json={"conversation_id": conv_id},
    )
    assert finish_resp.status_code == 200
    history_data = finish_resp.json()["data"]
    assert "chief_complaint" in history_data
    assert "history_of_present_illness" in history_data
    assert "interview_score" in history_data
