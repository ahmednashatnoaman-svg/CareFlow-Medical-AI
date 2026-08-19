"""Async Repository for Conversation Database entity."""

from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from careflow.core.constants import STATUS_ACTIVE
from careflow.models.conversation import ConversationModel


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


class ConversationRepository:
    """Handles CRUD operations for Conversation records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, patient_id: str, chief_complaint: Optional[str] = None) -> ConversationModel:
        """Creates a new conversation record."""
        conversation = ConversationModel(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            status=STATUS_ACTIVE,
            step_count=0,
            turns=[],
            state_snapshot={},
        )
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def get_by_id(self, conversation_id: str) -> Optional[ConversationModel]:
        """Retrieves conversation by ID."""
        stmt = select(ConversationModel).where(
            ConversationModel.id == conversation_id,
            ConversationModel.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def update_state(
        self,
        conversation_id: str,
        turns: List[Dict[str, Any]],
        state_snapshot: Dict[str, Any],
        step_count: int,
        status: str = STATUS_ACTIVE,
    ) -> Optional[ConversationModel]:
        """Updates conversation turns, state snapshot, and status."""
        conversation = await self.get_by_id(conversation_id)
        if not conversation:
            return None

        clean_snapshot = sanitize_state(state_snapshot)

        conversation.turns = turns
        conversation.state_snapshot = clean_snapshot
        conversation.step_count = step_count
        conversation.status = status

        if clean_snapshot.get("chief_complaint"):
            conversation.chief_complaint = clean_snapshot.get("chief_complaint")

        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation
