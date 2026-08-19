"""Unit tests for the LLM Orchestrator Service (Module 8)."""

import asyncio
import json
from langchain_core.messages import AIMessage
from careflow.services.orchestrator_service import LLMOrchestrator


def test_generate_next_question_with_web_evidence():
    orchestrator = LLMOrchestrator(api_key="mock-key")
    result = asyncio.run(
        orchestrator.generate_next_question(
            conversation_history=[],
            structured_state={"symptoms": ["Chest Pain"], "chief_complaint": "chest pain"},
            retrieved_chunks=[{"source": "NICE CG95", "content": "Assess onset and duration."}],
            previous_questions=[],
            missing_info=["duration"],
            web_evidence=[
                {
                    "disease_query": "Heart attack",
                    "title": "Heart attack",
                    "url": "https://www.mayoclinic.org/diseases-conditions/heart-attack",
                    "symptoms": "Chest pain, shortness of breath.",
                    "causes": "Coronary artery disease.",
                }
            ],
        )
    )

    assert result["question"]
    assert isinstance(result["missing_info"], list)


def test_generate_next_question_without_web_evidence():
    orchestrator = LLMOrchestrator(api_key="mock-key")
    result = asyncio.run(
        orchestrator.generate_next_question(
            conversation_history=[],
            structured_state={"symptoms": [], "chief_complaint": "chest pain"},
            retrieved_chunks=[],
            previous_questions=[],
            missing_info=[],
        )
    )

    assert result["question"]


def test_generate_next_question_with_mock_llm():
    class MockLLM:
        async def ainvoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "question": "When did the chest pain start?",
                        "question_arabic": "متى بدأ وجع الصدر؟",
                        "target_attribute": "onset",
                        "options": [{"text_english": "Suddenly", "text_arabic": "فجأة"}],
                        "updated_summary": "Patient experiencing chest pain.",
                        "missing_info": ["severity"],
                    }
                )
            )

    orchestrator = LLMOrchestrator(api_key="mock-key")
    orchestrator._llm = MockLLM()

    result = asyncio.run(
        orchestrator.generate_next_question(
            conversation_history=[],
            structured_state={"chief_complaint": "chest pain", "symptoms": ["chest pain"]},
            retrieved_chunks=[],
            previous_questions=[],
            missing_info=["onset"],
        )
    )

    assert result["question"] == "When did the chest pain start?"
    assert result["question_arabic"] == "متى بدأ وجع الصدر؟"
    assert result["target_attribute"] == "onset"
