"""Database Seeding Utility Script.

Populates initial database demo records.
"""

import asyncio
import logging
from careflow.core.dependencies.deps import get_session_context, init_db
from careflow.crud.conversation_repository import ConversationRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def seed_data():
    """Seeds demo conversation record."""
    logger.info("Initializing database schema...")
    await init_db()

    async with get_session_context() as session:
        repo = ConversationRepository(session)
        conv = await repo.create(
            patient_id="patient_demo_101",
            chief_complaint="Severe central chest pain",
        )
        logger.info(f"Seeded demo conversation: {conv.id} for patient {conv.patient_id}")


if __name__ == "__main__":
    asyncio.run(seed_data())
