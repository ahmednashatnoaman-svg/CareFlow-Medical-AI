"""baseline: conversations, patient_histories

Revision ID: 20260726000001
Revises:
Create Date: 2026-07-26

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "20260726000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(100), nullable=True, server_default="system"),
        sa.Column("patient_id", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="ACTIVE"),
        sa.Column("step_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chief_complaint", sa.String(255), nullable=True),
        sa.Column("turns", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("state_snapshot", sa.JSON(), nullable=False, server_default="{}"),
    )
    op.create_index("idx_conversations_patient_id", "conversations", ["patient_id"])

    op.create_table(
        "patient_histories",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(100), nullable=True, server_default="system"),
        sa.Column("conversation_id", sa.String(36), nullable=False, unique=True),
        sa.Column("patient_id", sa.String(100), nullable=False),
        sa.Column("chief_complaint", sa.String(255), nullable=False, server_default=""),
        sa.Column("structured_history", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("interview_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("completion_status", sa.String(50), nullable=False, server_default="COMPLETED"),
    )
    op.create_index("idx_patient_histories_patient_id", "patient_histories", ["patient_id"])


def downgrade() -> None:
    op.drop_table("patient_histories")
    op.drop_table("conversations")
