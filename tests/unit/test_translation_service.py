"""Unit tests for Translation Service (Module 2)."""

import asyncio
from langchain_core.messages import AIMessage
from app.services.translation_service import TranslationService


def test_translate_ar_to_en_with_llm():
    class MockLLM:
        async def ainvoke(self, messages):
            return AIMessage(content="I feel severe pain in my chest")

    service = TranslationService()
    service._llm = MockLLM()
    ar_text = "حاسس بوجع جامد في صدري"
    en_text = asyncio.run(service.translate_ar_to_en(ar_text))
    assert en_text == "I feel severe pain in my chest"


def test_translate_en_to_ar_with_llm():
    class MockLLM:
        async def ainvoke(self, messages):
            return AIMessage(content="بقالك قد إيه بتعاني من وجع الصدر ده؟")

    service = TranslationService()
    service._llm = MockLLM()
    en_text = "How long have you been experiencing this chest pain?"
    ar_text = asyncio.run(service.translate_en_to_ar(en_text))
    assert ar_text == "بقالك قد إيه بتعاني من وجع الصدر ده؟"
