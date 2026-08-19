"""Unified Multi-Provider LLM Client.

Supports Google Gemini (via google-genai), Groq (via groq), and OpenAI
with structured JSON extraction, automated fallbacks, and translation layers.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from careflow.core.config import settings

logger = logging.getLogger(__name__)


def clean_json_text(text: str) -> str:
    """Strips markdown code blocks, whitespace, and extracts valid JSON substring."""
    text = text.strip()
    # Remove markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text


class LLMClient:
    """Production LLM Client supporting Gemini, Groq, and OpenAI with auto-fallback."""

    def __init__(self):
        self.gemini_client = None
        self.groq_client = None
        self.openai_client = None
        self._init_clients()

    def _init_clients(self):
        # 1. Initialize Google Gemini Client
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                logger.info("Initialized Google Gemini client with model %s", settings.GEMINI_MODEL)
            except Exception as e:
                logger.warning("Failed to initialize Google Gemini client: %s", e)

        # 2. Initialize Groq Client
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "mock-key":
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Initialized Groq client with model %s", settings.GROQ_MODEL)
            except Exception as e:
                logger.warning("Failed to initialize Groq client: %s", e)

        # 3. Initialize OpenAI Client (if provided)
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("Initialized OpenAI client with model %s", settings.OPENAI_MODEL)
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", e)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        attempts: int = 2,
    ) -> str:
        """Generates plain text response using available LLMs with retry backoff."""
        import time

        for attempt in range(attempts):
            errors = []

            # Attempt 1: Gemini
            if self.gemini_client:
                try:
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    if system_prompt:
                        config.system_instruction = system_prompt

                    model_name = settings.GEMINI_MODEL
                    if model_name.startswith("models/"):
                        model_name = model_name.replace("models/", "", 1)

                    response = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    logger.warning("Gemini generate_text (attempt %d) failed: %s", attempt + 1, e)
                    errors.append(f"Gemini: {e}")

            # Attempt 2: Groq
            if self.groq_client:
                try:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    response = self.groq_client.chat.completions.create(
                        model=settings.GROQ_MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning("Groq generate_text (attempt %d) failed: %s", attempt + 1, e)
                    errors.append(f"Groq: {e}")

            # Attempt 3: OpenAI
            if self.openai_client:
                try:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    response = self.openai_client.chat.completions.create(
                        model=settings.OPENAI_MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning("OpenAI generate_text (attempt %d) failed: %s", attempt + 1, e)
                    errors.append(f"OpenAI: {e}")

            if attempt < attempts - 1:
                time.sleep(1.0)

        raise RuntimeError(f"All LLM providers failed after {attempts} attempts. Errors: {'; '.join(errors)}")

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """Generates structured JSON with automated schema retries and cleaning."""
        last_error = None
        for attempt in range(retries):
            # Instruct explicit JSON formatting
            enhanced_system = (
                (system_prompt + "\n\n") if system_prompt else ""
            ) + "IMPORTANT: You MUST respond ONLY with a single valid JSON object. Do not include markdown code block formatting (```json), commentary, or extra text."

            try:
                raw_text = self.generate_text(
                    prompt=prompt,
                    system_prompt=enhanced_system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                cleaned = clean_json_text(raw_text)
                return json.loads(cleaned)
            except Exception as e:
                last_error = e
                logger.warning(
                    "JSON parsing attempt %d/%d failed: %s. Raw output snippet: %s",
                    attempt + 1,
                    retries,
                    e,
                    raw_text[:200] if 'raw_text' in locals() else 'None',
                )

        raise ValueError(f"Failed to generate valid JSON after {retries} attempts: {last_error}")

    def translate_patient_to_english(self, arabic_text: str) -> str:
        """Translates colloquial Egyptian Arabic patient complaint to clinical English."""
        # Fast check if text is already primarily English / Latin
        if not re.search(r"[\u0600-\u06FF]", arabic_text):
            return arabic_text

        prompt = f"""Translate the following Egyptian Arabic medical complaint into concise, clear clinical English.
Accurately translate Egyptian colloquial medical terms and slang (for example: 'مغص' -> abdominal cramps/colic, 'دوخة' -> dizziness/vertigo, 'نهجان' -> shortness of breath/dyspnea, 'ترجيع' -> vomiting, 'سخونية' -> fever).
Output ONLY the clean English translation, without quotes or additional text.

Patient: "{arabic_text}"
Translation:"""
        return self.generate_text(prompt=prompt, temperature=0.1, max_tokens=200)

    def translate_agent_to_arabic(self, english_json: Dict[str, Any]) -> Dict[str, Any]:
        """Translates clinical English question & options JSON payload to natural Egyptian Arabic."""
        prompt = f"""Translate the clinical interview question and options in the JSON below into empathetic, natural Egyptian Arabic (لهجة مصرية عامية مهذبة وطبية).
Use polite phrasing (e.g., 'ألف سلامة عليك', 'حاسس بـ...').
Keep the exact same JSON schema and keys. Translate only the string values of 'empathy_and_question', 'options', and 'summary_if_complete'.

English JSON:
{json.dumps(english_json, indent=2, ensure_ascii=False)}
"""
        try:
            return self.generate_json(prompt=prompt, temperature=0.2, max_tokens=1000)
        except Exception as e:
            logger.warning("Agent outbound Arabic translation failed: %s. Returning English payload.", e)
            return english_json


# Global singleton instance
llm_client = LLMClient()
