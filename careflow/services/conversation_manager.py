"""Module 3: Conversation Manager.

Maintains conversation state in-memory / Redis cache.
Exposes get_state(), update_state(), summarize_state().
"""

import json
import logging
from typing import Any, Dict, List, Optional
from redis.asyncio import Redis
from careflow.core.config import settings
from careflow.core.constants import STATUS_ACTIVE

logger = logging.getLogger(__name__)


from langchain_core.messages import BaseMessage


def sanitize_state(obj: Any) -> Any:
    """Recursively converts non-JSON-serializable objects (like LangChain BaseMessages and raw bytes) into primitives."""
    if isinstance(obj, BaseMessage):
        role = "user" if getattr(obj, "type", "") in ("human", "user") else "assistant"
        return {"role": role, "content": getattr(obj, "content", str(obj))}
    elif isinstance(obj, dict):
        return {k: sanitize_state(v) for k, v in obj.items() if not isinstance(v, (bytes, bytearray))}
    elif isinstance(obj, list):
        return [sanitize_state(item) for item in obj]
    elif isinstance(obj, (bytes, bytearray)):
        return None
    return obj


class ConversationManager:
    """Manages active conversation state and cached history."""

    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client
        # In-memory storage fallback for local development/testing
        self._memory_store: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def new_state(conversation_id: str = "", patient_id: str = "unknown", chief_complaint: str = "") -> Dict[str, Any]:
        """Builds a fresh conversation state with every tracked key present.

        Single source of truth for the seed shape -- the /start, /pipeline, and
        default-state paths had drifted apart and silently omitted keys.
        """
        return {
            "conversation_id": conversation_id,
            "patient_id": patient_id,
            "chief_complaint": chief_complaint,
            "symptoms": [],
            "timeline": {},
            "previous_questions": [],
            "previous_answers": [],
            "asked_questions": [],
            "exhausted_attributes": [],
            "question_options": [],
            "selected_option_index": None,
            "option_resolved": False,
            "yielded_new_info": True,
            "turns": [],
            "risk_factors": [],
            "missing_info": [],
            "missing_information": [],
            "retrieved_evidence": [],
            "extracted_entities": {},
            "step_count": 0,
            "status": STATUS_ACTIVE,
        }

    async def get_state(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieves active conversation state by conversation ID.

        Args:
            conversation_id (str): Unique conversation identifier.

        Returns:
            Dict[str, Any]: Complete conversation state dictionary.
        """
        if self.redis:
            try:
                data = await self.redis.get(f"conv_state:{conversation_id}")
                if data:
                    return json.loads(data)
            except Exception as exc:
                logger.warning(f"Redis get_state error: {exc}. Falling back to memory.")

        return self._memory_store.get(conversation_id, self.new_state(conversation_id))

    async def update_state(self, conversation_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates active conversation state.

        Args:
            conversation_id (str): Unique conversation identifier.
            updates (Dict[str, Any]): Dictionary of fields to update.

        Returns:
            Dict[str, Any]: Updated conversation state.
        """
        current_state = await self.get_state(conversation_id)
        current_state.update(updates)

        clean_state = sanitize_state(current_state)

        if self.redis:
            try:
                await self.redis.set(
                    f"conv_state:{conversation_id}",
                    json.dumps(clean_state),
                    ex=86400,  # 24 hour TTL
                )
            except Exception as exc:
                logger.warning(f"Redis update_state error: {exc}")

        self._memory_store[conversation_id] = clean_state
        return clean_state

    async def summarize_state(self, conversation_id: str) -> str:
        """Generates a concise textual narrative summary of current medical history state.

        Args:
            conversation_id (str): Unique conversation ID.

        Returns:
            str: Human-readable narrative summary.
        """
        state = await self.get_state(conversation_id)
        cc = state.get("chief_complaint", "Not specified")
        symptoms = ", ".join(state.get("symptoms", [])) or "None reported"
        risks = ", ".join(state.get("risk_factors", [])) or "None noted"
        step = state.get("step_count", 0)

        summary = (
            f"Chief Complaint: {cc}. "
            f"Reported Symptoms: {symptoms}. "
            f"Risk Factors: {risks}. "
            f"Total Turns: {step}."
        )
        return summary
