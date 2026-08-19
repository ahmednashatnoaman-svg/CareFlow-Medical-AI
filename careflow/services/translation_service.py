"""Module 2: Translation Service.

Maintains English as the canonical internal reasoning language.
Translates Egyptian Arabic to clinical English and English questions back to Egyptian Arabic.
"""

import json
import logging
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from careflow.core.config import settings
from careflow.services.llm_clients import GeminiChatModel, SBGChatModel, extract_text_content

logger = logging.getLogger(__name__)


class TranslationService:
    """Translates Arabic clinical input to English and English LLM outputs to Egyptian Arabic."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SBG_API_KEY
        self._llm = None

        # 1. Primary: Google Gemini
        if settings.GEMINI_API_KEY:
            try:
                self._llm = GeminiChatModel(
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                )
                logger.info(f"Initialized Gemini LLM for translation using model '{settings.GEMINI_MODEL}'")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini LLM for translation: {e}")

        # 2. Fallback: Groq API
        if self._llm is None and settings.GROQ_API_KEY and settings.GROQ_API_KEY != "mock-key":
            try:
                self._llm = ChatOpenAI(
                    openai_api_base=settings.GROQ_BASE_URL,
                    openai_api_key=settings.GROQ_API_KEY,
                    model_name=settings.GROQ_MODEL,
                    temperature=0.1,
                )
                logger.info(f"Initialized Groq LLM fallback using model '{settings.GROQ_MODEL}'")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM fallback: {e}")

        # 3. Secondary: SBG gateway (gpt-oss-120)
        if self._llm is None and self.api_key and self.api_key != "mock-key":
            try:
                self._llm = SBGChatModel(
                    base_url=settings.SBG_BASE_URL,
                    api_key=self.api_key,
                    model_id=settings.SBG_MODEL,
                )
                logger.info(f"Initialized SBG gpt-oss LLM using model '{settings.SBG_MODEL}'")
            except Exception as e:
                logger.warning(f"Failed to initialize SBG gpt-oss LLM: {e}")

    async def translate_ar_to_en(self, arabic_text: str) -> str:
        """Translates Egyptian Arabic patient input into precise clinical English."""
        if not arabic_text or not arabic_text.strip():
            return ""

        cleaned = arabic_text.strip()

        if self._llm:
            try:
                sys_msg = SystemMessage(
                    content=(
                        "You are an expert medical translator specializing in Egyptian Arabic to clinical English. Your tone is highly accurate and professional.\n"
                        "Translate colloquial Egyptian Arabic statements accurately into precise clinical English. "
                        "Preserve medical terminology, severity, and temporal details. Do not simplify clinical meaning. "
                        "Think step-by-step about the translation if needed, but CRITICAL: Output absolutely nothing but the final English translation string. Do not include any reasoning, preambles, or markdown."
                    )
                )
                user_msg = HumanMessage(content=cleaned)
                response = await self._llm.ainvoke([sys_msg, user_msg])
                return extract_text_content(response.content)
            except Exception as exc:
                logger.error("LLM translation (Ar->En) failed", exc_info=True)

        return cleaned

    async def translate_en_to_ar(self, english_text: str) -> str:
        """Translates an English follow-up clinical question into fluent Egyptian Arabic."""
        if not english_text or not english_text.strip():
            return ""

        cleaned = english_text.strip()

        if self._llm:
            try:
                sys_msg = SystemMessage(
                    content=(
                        """
                        # Role
                        You are an Egyptian physician speaking to a patient.

                        # Context
                        The input is an English clinical follow-up question written for healthcare professionals.

                        # Task
                        Rewrite the question as if you were asking it directly to an Egyptian patient in everyday Egyptian Arabic.

                        # Constraints
                        - Do NOT translate word-for-word.
                        - Express the meaning, not the wording.
                        - Replace every medical term with a plain-language Arabic explanation that a patient without medical knowledge can understand.
                        - Do not preserve or quote technical terminology.
                        - The output must contain Arabic characters only.
                        - If any Latin character appears in the output, discard it and rewrite the sentence.
                        - Preserve the clinical intent exactly.
                        - Output only the final Arabic question.

                        Step 1 (internal): Identify any technical or medical terminology.

                        Step 2 (internal): Replace each identified term with language that an average Egyptian patient would naturally understand.

                        Step 3: Produce the final question in Egyptian Arabic.

                        Output only Step 3.
                    """
                    )
                )
                user_msg = HumanMessage(content=cleaned)
                response = await self._llm.ainvoke([sys_msg, user_msg])
                return extract_text_content(response.content)
            except Exception as exc:
                logger.error("LLM translation (En->Ar) failed", exc_info=True)

        return cleaned

    async def translate_options_en_to_ar(self, options: List[str]) -> List[str]:
        """Translates short answer choices into Egyptian Arabic."""
        cleaned = [str(o).strip() for o in options]
        if not any(cleaned):
            return cleaned

        if self._llm:
            try:
                sys_msg = SystemMessage(
                    content=(
                        "You are an expert translator converting short answer choices for a patient questionnaire from English "
                        "into natural, everyday Egyptian Arabic (لهجة مصرية). Your tone is precise and context-aware.\n"
                        "Keep each choice as a SHORT phrase -- never turn it into a sentence or a question. "
                        "CRITICAL: Output ONLY a valid JSON array of strings, in the exact same order and with the same number "
                        "of items as the input. Do not include any reasoning, explanations, numbering, or markdown text outside the JSON array."
                    )
                )
                user_msg = HumanMessage(content=json.dumps(cleaned, ensure_ascii=False))
                response = await self._llm.ainvoke([sys_msg, user_msg])
                content = extract_text_content(response.content)

                start_idx = content.find("[")
                end_idx = content.rfind("]")
                if start_idx != -1 and end_idx > start_idx:
                    content = content[start_idx : end_idx + 1]

                translated = json.loads(content)
                if isinstance(translated, list) and len(translated) == len(cleaned):
                    return [
                        str(ar).strip() or en
                        for ar, en in zip(translated, cleaned)
                    ]
            except Exception as exc:
                logger.error("LLM option translation (En->Ar) failed", exc_info=True)

        return cleaned
