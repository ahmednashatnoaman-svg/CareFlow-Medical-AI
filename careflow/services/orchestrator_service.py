"""Module 8: LLM Orchestrator Service.

Coordinates clinical reasoning, missing information identification, question generation,
and evidence synthesis using RAG context, Mayo Clinic reference data, and interview state.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from careflow.core.config import settings
from careflow.core.constants import LLM_INTERVIEW
from careflow.services.llm_clients import GeminiChatModel, SBGChatModel, extract_text_content

logger = logging.getLogger(__name__)


DEFAULT_REQUIRED_ATTRIBUTES = [
    "onset",
    "duration",
    "location",
    "severity",
    "character",
    "radiation",
    "aggravating_factors",
    "relieving_factors",
    "associated_symptoms",
]


def build_options(options_data: Any) -> List[Dict[str, str]]:
    """Normalizes option structures into standard list of dictionaries."""
    if not isinstance(options_data, list):
        return []

    result = []
    for opt in options_data:
        if isinstance(opt, dict):
            eng = str(opt.get("text_english") or opt.get("english") or opt.get("text") or "").strip()
            ara = str(opt.get("text_arabic") or opt.get("arabic") or "").strip()
            if eng or ara:
                result.append({"text_english": eng or ara, "text_arabic": ara or eng})
        elif isinstance(opt, str) and opt.strip():
            txt = opt.strip()
            result.append({"text_english": txt, "text_arabic": txt})
    return result


class LLMOrchestrator:
    """Orchestrates follow-up question generation using LLM and clinical context."""

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
                logger.info(f"Initialized Gemini LLM for orchestrator using model '{settings.GEMINI_MODEL}'")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini LLM for orchestrator: {e}")

        # 2. Fallback: Groq API
        if self._llm is None and settings.GROQ_API_KEY and settings.GROQ_API_KEY != "mock-key":
            try:
                self._llm = ChatOpenAI(
                    openai_api_base=settings.GROQ_BASE_URL,
                    openai_api_key=settings.GROQ_API_KEY,
                    model_name=settings.GROQ_MODEL,
                    temperature=LLM_INTERVIEW.temperature,
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

    def _pick_attribute(
        self,
        raw_target: Any,
        available_attributes: List[str],
        missing_info: List[str],
    ) -> str:
        """Sanitizes LLM attribute selection against allowed attributes."""
        if isinstance(raw_target, str) and raw_target.strip():
            candidate = raw_target.strip().lower()
            if candidate in available_attributes:
                return candidate

        for missing in missing_info:
            missing_clean = missing.strip().lower()
            if missing_clean in available_attributes:
                return missing_clean

        return available_attributes[0] if available_attributes else "onset"

    async def generate_next_question(
        self,
        conversation_history: List[Dict[str, Any]],
        structured_state: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        previous_questions: List[str],
        missing_info: List[str],
        web_evidence: Optional[List[Dict[str, Any]]] = None,
        exhausted_attributes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generates exactly ONE follow-up question grounded in retrieved evidence via LLM."""
        logger.info("Generating next follow-up question via LLM Orchestrator")

        if not self._llm:
            raise RuntimeError(
                "LLM provider is not configured. Please set GEMINI_API_KEY, GROQ_API_KEY, or SBG_API_KEY."
            )

        exhausted = list(exhausted_attributes or [])
        available_attributes = [a for a in DEFAULT_REQUIRED_ATTRIBUTES if a not in exhausted]

        evidence_text = "\n\n".join(
            [f"[{c.get('source')}] {c.get('content')}" for c in retrieved_chunks]
        )

        web_evidence_text = "\n\n".join(
            f"[Mayo Clinic: {w.get('title')}] Symptoms: {w.get('symptoms', '')} Causes: {w.get('causes', '')}"
            for w in (web_evidence or [])
        )

        system_prompt = (
            "You are CareFlow's Clinical AI Interview Orchestrator. Your identity is a clinical expert, and your tone is empathetic, concise, and precise.\n"
            "Your task is to generate EXACTLY ONE clinical follow-up question for the patient in both English AND natural Egyptian Arabic dialect (عامية مصري),\n"
            "together with a short list of answer choices the patient can pick from (each choice in both English and Egyptian Arabic).\n"
            "Think step-by-step about the patient's missing information and the conversation history before generating the final JSON.\n"
            "RULES:\n"
            "1. Ask ONLY ONE question.\n"
            "2. NEVER repeat a question listed in previous questions. Every question MUST provide added value by targeting a specific missing information gap.\n"
            "3. Base reasoning on clinical knowledge and patient responses in full conversation history.\n"
            "4. Never invent medical advice or diagnosis.\n"
            "5. Keep the question concise, empathetic, and clear in both English and Egyptian Arabic.\n"
            "6. Refer to the patient's complaint in their own words. Never assume a body site that the patient did not mention.\n"
            f"7. Set target_attribute to the ONE attribute the question probes, chosen from: {json.dumps(available_attributes)}.\n"
            "8. NEVER target an exhausted attribute -- those were already asked and yielded nothing.\n"
            "Output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "question": "string (English follow-up question)",\n'
            '  "question_arabic": "string (Egyptian Arabic translation)",\n'
            '  "target_attribute": "string (one attribute probed)",\n'
            '  "options": [{"text_english": "string", "text_arabic": "string"}],\n'
            '  "updated_summary": "string",\n'
            '  "missing_info": ["string"]\n'
            "}"
        )

        user_prompt = (
            f"Conversation History:\n{json.dumps(conversation_history, ensure_ascii=False)}\n\n"
            f"Structured Patient State:\n{json.dumps(structured_state, ensure_ascii=False)}\n\n"
            f"Retrieved Clinical Guidelines Evidence:\n{evidence_text or 'None'}\n\n"
            f"Mayo Clinic Disease Reference:\n{web_evidence_text or 'None'}\n\n"
            f"Previous Questions Asked:\n{json.dumps(previous_questions, ensure_ascii=False)}\n\n"
            f"Exhausted Attributes (DO NOT ASK): {json.dumps(exhausted)}\n"
            f"Available Target Attributes: {json.dumps(available_attributes)}\n"
            f"Missing Clinical Information: {json.dumps(missing_info)}\n"
        )

        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content_str = extract_text_content(response.content)

            if content_str.startswith("```"):
                lines = content_str.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                content_str = "\n".join(lines).strip()

            start_idx = content_str.find("{")
            end_idx = content_str.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content_str = content_str[start_idx : end_idx + 1]

            parsed = json.loads(content_str)
            question = parsed.get("question", "")
            question_arabic = parsed.get("question_arabic", "")

            if not question and not question_arabic:
                raise ValueError("LLM returned empty question content")

            target_attribute = self._pick_attribute(
                parsed.get("target_attribute"), available_attributes, missing_info
            )
            return {
                "question": question or question_arabic,
                "question_arabic": question_arabic or question,
                "target_attribute": target_attribute,
                "options": build_options(parsed.get("options")),
                "updated_summary": parsed.get("updated_summary", "Patient reporting symptoms."),
                "missing_info": parsed.get("missing_info", missing_info),
            }
        except Exception as exc:
            logger.error("LLM Orchestrator invocation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"LLM question generation failed: {exc}")
