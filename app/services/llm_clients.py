"""Async client for the ITI SBG LLM gateway.

Speaks the gateway's REST contract (model_id/messages/system_prompt/max_tokens)
and exposes an `ainvoke()` surface compatible with the LangChain chat message
call sites elsewhere in this codebase, so it's a drop-in for `ChatOpenAI`.
"""

import logging
from typing import Any, List, Optional

import httpx
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from app.core.config import settings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

logger = logging.getLogger(__name__)


class SBGChatModel:
    """Minimal async chat client for the SBG gateway's /student/chat endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        # Defaults to settings.SBG_MAX_TOKENS (configurable) rather than a bare literal --
        # callers that construct this directly can still override it explicitly.
        max_tokens = max_tokens if max_tokens is not None else settings.SBG_MAX_TOKENS
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Sends chat messages to the SBG gateway and returns the model's reply.

        Args:
            messages (List[BaseMessage]): LangChain messages; any SystemMessage
                is extracted as the gateway's separate `system_prompt` field.

        Returns:
            AIMessage: Reply content, matching the ChatOpenAI response surface.
        """
        system_prompt = ""
        chat_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            else:
                role = "assistant" if msg.type == "ai" else "user"
                chat_messages.append({"role": role, "content": str(msg.content)})

        payload = {
            "model_id": self.model_id,
            "messages": chat_messages,
            "system_prompt": system_prompt,
            "max_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/student/chat",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = data.get("output_text")
        if content is None:
            raise ValueError(
                f"SBG gateway response missing 'output_text'; keys: {sorted(data)}"
            )
        return AIMessage(content=content)


def extract_text_content(content: Any) -> str:
    """Extracts plain text string from LLM message content (handles str, list of dicts/blocks)."""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text")))
        return "".join(parts).strip()
    return str(content).strip()


class GeminiChatModel:
    """Async chat client for Google Gemini, matching `SBGChatModel`'s `ainvoke()` surface."""

    def __init__(self, api_key: str, model: str):
        if ChatGoogleGenerativeAI is None:
            raise ImportError(
                "langchain-google-genai package is required to use GeminiChatModel. "
                "Install it via `pip install langchain-google-genai`."
            )
        self._client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Sends chat messages to Gemini and returns the model's reply.

        Args:
            messages (List[BaseMessage]): LangChain messages, passed through as-is --
                `ChatGoogleGenerativeAI` natively understands SystemMessage/HumanMessage/AIMessage.

        Returns:
            AIMessage: Reply content, matching the ChatOpenAI response surface.
        """
        result = await self._client.ainvoke(messages)
        text = extract_text_content(result.content)
        return AIMessage(content=text)

