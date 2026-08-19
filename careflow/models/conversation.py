"""Conversation Database Model.

Stores conversation metadata, status, step indices, turn logs, and state snapshots.
"""

from typing import Any, Dict, List
from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from careflow.core.constants import STATUS_ACTIVE
from careflow.models.base import Base


class ConversationModel(Base):
    """OR Model representing a patient conversation session."""

    __tablename__ = "conversations"

    patient_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default=STATUS_ACTIVE)
    step_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chief_complaint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    turns: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    state_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
