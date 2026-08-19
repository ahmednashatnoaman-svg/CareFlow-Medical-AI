"""Patient History Database Model.

Stores final structured medical history JSON and completion metrics.
"""

from typing import Any, Dict
from sqlalchemy import JSON, Float, String
from sqlalchemy.orm import Mapped, mapped_column
from app.core.constants import STATUS_COMPLETED
from app.models.base import Base


class PatientHistoryModel(Base):
    """ORM Model storing final structured medical history per patient session."""

    __tablename__ = "patient_histories"

    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    patient_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    chief_complaint: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    structured_history: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    interview_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    completion_status: Mapped[str] = mapped_column(String(50), nullable=False, default=STATUS_COMPLETED)
