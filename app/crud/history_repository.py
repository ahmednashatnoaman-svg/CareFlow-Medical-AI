"""Async Repository for Patient History Database entity."""

from typing import Any, Dict, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.constants import STATUS_COMPLETED
from app.models.patient_history import PatientHistoryModel


class HistoryRepository:
    """Handles CRUD operations for Patient History records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_history(
        self,
        conversation_id: str,
        patient_id: str,
        chief_complaint: str,
        structured_history: Dict[str, Any],
        interview_score: float,
        completion_status: str = STATUS_COMPLETED,
    ) -> PatientHistoryModel:
        """Saves or updates final structured medical history."""
        stmt = select(PatientHistoryModel).where(
            PatientHistoryModel.conversation_id == conversation_id
        )
        result = await self.session.execute(stmt)
        history = result.scalars().first()

        if history:
            history.structured_history = structured_history
            history.interview_score = interview_score
            history.completion_status = completion_status
            history.chief_complaint = chief_complaint
        else:
            history = PatientHistoryModel(
                conversation_id=conversation_id,
                patient_id=patient_id,
                chief_complaint=chief_complaint,
                structured_history=structured_history,
                interview_score=interview_score,
                completion_status=completion_status,
            )
            self.session.add(history)

        await self.session.commit()
        await self.session.refresh(history)
        return history

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[PatientHistoryModel]:
        """Retrieves structured history by conversation ID."""
        stmt = select(PatientHistoryModel).where(
            PatientHistoryModel.conversation_id == conversation_id,
            PatientHistoryModel.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
