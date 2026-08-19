"""Module 4: Medical Entity Extraction Service.

Extracts structured medical concepts: Symptoms, Diseases, Medications, Procedures,
Body locations, Risk factors, and attribute details (severity, onset, duration).
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from careflow.core.config import settings
from careflow.core.constants import LLM_EXTRACTION
from careflow.schemas.state import ExtractedEntities
from careflow.services.llm_clients import GeminiChatModel, SBGChatModel, extract_text_content

logger = logging.getLogger(__name__)


class EntityExtractionService:
    """Extracts medical entities and attribute breakdowns from English text."""

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
                logger.info(f"Initialized Gemini LLM for entity extraction using model '{settings.GEMINI_MODEL}'")
            except Exception as exc:
                logger.warning(f"Failed to initialize Gemini LLM for entity extraction: {exc}")

        # 2. Fallback: Groq API (if uninitialized)
        if self._llm is None and settings.GROQ_API_KEY and settings.GROQ_API_KEY != "mock-key":
            try:
                self._llm = ChatOpenAI(
                    openai_api_base=settings.GROQ_BASE_URL,
                    openai_api_key=settings.GROQ_API_KEY,
                    model_name=settings.GROQ_MODEL,
                    temperature=LLM_EXTRACTION.temperature,
                )
                logger.info(f"Initialized Groq LLM fallback for entity extraction using model '{settings.GROQ_MODEL}'")
            except Exception as exc:
                logger.warning(f"Failed to initialize Groq LLM fallback for entity extraction: {exc}")

        # 3. Secondary: SBG gateway (gpt-oss-120)
        if self._llm is None and self.api_key and self.api_key != "mock-key":
            try:
                self._llm = SBGChatModel(
                    base_url=settings.SBG_BASE_URL,
                    api_key=self.api_key,
                    model_id=settings.SBG_MODEL,
                )
                logger.info(f"Initialized SBG gpt-oss LLM for entity extraction using model '{settings.SBG_MODEL}'")
            except Exception as exc:
                logger.warning(f"Failed to initialize SBG gpt-oss LLM for entity extraction: {exc}")

    async def extract_entities(self, english_text: str) -> ExtractedEntities:
        """Extracts medical entities from canonical clinical English text.

        Args:
            english_text (str): Patient statement in English.

        Returns:
            ExtractedEntities: Extracted structured concepts.
        """
        if not english_text or not english_text.strip():
            return ExtractedEntities()

        if self._llm:
            try:
                sys_msg = SystemMessage(
                    content=(
                        "You are a clinical NLP entity extraction engine. Your tone should be objective and precise.\n"
                        "Extract medical entities from the input text step-by-step in your reasoning if needed, but "
                        "return ONLY a valid JSON object in your final output with no conversational text or markdown.\n"
                        "Keys must be exactly: 'symptoms', 'diseases', 'medications', 'procedures', "
                        "'body_locations', 'risk_factors', 'allergies', 'family_history', 'social_history', "
                        "and 'attributes' (onset, duration, severity, location, radiation, character, aggravating_factors, relieving_factors).\n"
                        "'allergies' is drug/food/environmental allergies the patient names explicitly. "
                        "'family_history' is relevant conditions the patient attributes to a relative. "
                        "'social_history' is smoking, alcohol, occupation, or living-situation details. "
                        "Leave any of these as an empty list if the text does not mention them -- never infer or guess. "
                        "Prioritize extracting exactly what is in the patient text over your internal knowledge base."
                    )
                )
                user_msg = HumanMessage(content=english_text)
                response = await self._llm.ainvoke([sys_msg, user_msg])
                import json
                raw_str = extract_text_content(response.content)

                if raw_str.startswith("```"):
                    lines = raw_str.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    raw_str = "\n".join(lines).strip()
                start_i = raw_str.find("{")
                end_i = raw_str.rfind("}")
                if start_i != -1 and end_i != -1 and end_i > start_i:
                    raw_str = raw_str[start_i : end_i + 1]

                parsed = json.loads(raw_str)
                return ExtractedEntities(
                    symptoms=parsed.get("symptoms", []),
                    diseases=parsed.get("diseases", []),
                    medications=parsed.get("medications", []),
                    procedures=parsed.get("procedures", []),
                    body_locations=parsed.get("body_locations", []),
                    risk_factors=parsed.get("risk_factors", []),
                    allergies=parsed.get("allergies", []),
                    family_history=parsed.get("family_history", []),
                    social_history=parsed.get("social_history", []),
                    attributes=parsed.get("attributes", {}),
                )
            except Exception as exc:
                logger.error("LLM entity extraction failed: %s", exc, exc_info=True)
                return ExtractedEntities()

        return ExtractedEntities()
