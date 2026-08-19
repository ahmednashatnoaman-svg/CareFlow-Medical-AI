"""Automated Full Pipeline Execution Script.

Simulates a full end-to-end medical history collection conversation from initial audio/text
through the complete 10-node LangGraph pipeline to the final structured history JSON output.
"""

import asyncio
import json
import logging
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from careflow.core.constants import STATUS_ACTIVE, STATUS_COMPLETED

from careflow.core.dependencies.deps import get_session_context, init_db
from careflow.services.graph.workflow import interview_app
from careflow.crud.conversation_repository import ConversationRepository
from careflow.crud.history_repository import HistoryRepository
from careflow.schemas.history import StructuredMedicalHistory
from careflow.services.conversation_manager import ConversationManager
from careflow.services.knowledge_ingestion_service import KnowledgeIngestionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("automated_pipeline")


async def run_pipeline():
    logger.info("Step 1: Initializing Database & Vector Knowledge Base...")
    await init_db()

    ingest_service = KnowledgeIngestionService()
    await ingest_service.initialize_collections()
    await ingest_service.ingest_document(
        source_name="NICE Guideline CG95 - Acute Chest Pain",
        content=(
            "NICE CG95 recommends evaluating chest pain onset, location, severity, duration, "
            "and radiation to shoulder, arm, neck, or jaw. Assess associated dyspnea, sweating, "
            "nausea, syncope, and cardiovascular risk factors (hypertension, diabetes)."
        ),
        section="Chest Pain & ACS",
    )

    conv_manager = ConversationManager()

    async with get_session_context() as session:
        repo = ConversationRepository(session)
        hist_repo = HistoryRepository(session)

        patient_id = "patient_auto_999"
        chief_complaint = ""

        logger.info(f"Step 2: Starting Conversation for Patient '{patient_id}'...")
        conv = await repo.create(patient_id=patient_id, chief_complaint=chief_complaint)
        conv_id = conv.id

        initial_state = {
            "conversation_id": conv_id,
            "patient_id": patient_id,
            "step_count": 0,
            "chief_complaint": chief_complaint,
            "turns": [],
            "previous_questions": ["What symptom or medical concern brings you in today?"],
            "extracted_entities": {},
            "missing_info": [],
            "status": STATUS_ACTIVE,
        }
        await conv_manager.update_state(conv_id, initial_state)

        # Multi-turn Patient Inputs simulating an adaptive Egyptian Arabic interview
        patient_turns = [
            "حاسس بوجع جامد في صدري من ساعتين وبيسمع في كتفي الشمال",
            "الالم شديد 8 من 10 ونزل عليا عرق بارد مع صعوبة تنفس",
            "لأ معنديش سكر بس باخد علاج لضغط الدم العالي",
        ]

        state = await conv_manager.get_state(conv_id)

        for turn_num, input_arabic in enumerate(patient_turns, start=1):
            logger.info(f"\n--- Processing Turn {turn_num} ---")
            logger.info(f"Patient (Egyptian Arabic): '{input_arabic}'")

            state["user_input_arabic"] = input_arabic
            state["raw_audio_bytes"] = None

            # Execute full LangGraph State Machine pipeline turn
            state = await interview_app.ainvoke(state)

            logger.info(f"Clinical English Input: '{state.get('user_input_english')}'")
            logger.info(f"Generated Question (English): '{state.get('generated_question_english')}'")
            logger.info(f"Generated Question (Arabic): '{state.get('generated_question_arabic')}'")
            metrics = state.get("evaluation_metrics", {})
            logger.info(
                f"Evaluation Score: {metrics.get('overall_interview_score')} "
                f"(Coverage: {metrics.get('coverage_score')}, Guideline: {metrics.get('guideline_coverage_score')})"
            )

            await conv_manager.update_state(conv_id, state)
            await repo.update_state(
                conversation_id=conv_id,
                turns=state.get("turns", []),
                state_snapshot=state,
                step_count=state.get("step_count", turn_num),
                status=state.get("status", STATUS_ACTIVE),
            )

            if not state.get("should_continue", True):
                logger.info("Termination Engine triggered completion condition!")
                break

        # If not terminated, build final history
        if not state.get("structured_history"):
            from careflow.services.graph.nodes import history_builder_node
            state["should_continue"] = False
            state["status"] = STATUS_COMPLETED
            res = await history_builder_node(state)
            state["structured_history"] = res.get("structured_history", {})

        structured_dict = state.get("structured_history", {})
        final_history = StructuredMedicalHistory(**structured_dict)

        await hist_repo.save_history(
            conversation_id=conv_id,
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            structured_history=final_history.model_dump(),
            interview_score=final_history.interview_score,
            completion_status=STATUS_COMPLETED,
        )

        logger.info("\n=======================================================")
        logger.info("FINAL STRUCTURED MEDICAL HISTORY JSON OUTPUT:")
        logger.info("=======================================================")
        logger.info(json.dumps(final_history.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run_pipeline())
