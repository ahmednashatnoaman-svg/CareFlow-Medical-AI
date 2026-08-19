"""Live Voice & Audio End-to-End Clinical Interview Pipeline Script.

Tests the full multi-turn clinical interview workflow starting from patient voice/audio streams:
Audio Input (ASR) -> Arabic-to-English Translation -> Entity Extraction -> Guideline Retrieval (Qdrant) ->
Cross-Encoder Rerank -> LLM Question Generator -> Interview Termination Score Engine ->
English-to-Arabic Question Translation -> Final Structured History Generation.
"""

import asyncio
import json
import logging
import os
import sys
from typing import List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.core.constants import STATUS_ACTIVE, STATUS_COMPLETED
from app.core.dependencies.deps import get_session_context, init_db
from app.services.graph.workflow import interview_app
from app.crud.conversation_repository import ConversationRepository
from app.crud.history_repository import HistoryRepository
from app.schemas.history import StructuredMedicalHistory
from app.services.conversation_manager import ConversationManager
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.services.speech_service import SpeechService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_voice_pipeline")


def create_dummy_wav_bytes(duration_sec: float = 2.0) -> bytes:
    """Generates valid WAV audio binary data for testing ASR voice endpoint."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * 2
    file_size = 36 + data_size

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(file_size.to_bytes(4, "little"))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))  # Subchunk1Size
    header.extend((1).to_bytes(2, "little"))   # AudioFormat (PCM)
    header.extend((1).to_bytes(2, "little"))   # NumChannels (Mono)
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend((sample_rate * 2).to_bytes(4, "little"))  # ByteRate
    header.extend((2).to_bytes(2, "little"))   # BlockAlign
    header.extend((16).to_bytes(2, "little"))  # BitsPerSample
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))
    header.extend(b"\x00" * data_size)
    return bytes(header)


async def run_live_pipeline(
    audio_paths_or_texts: Optional[List[str]] = None,
    patient_id: str = "patient_live_voice_001",
    chief_complaint: str = "",
    interactive: bool = False,
):
    """Executes live end-to-end voice pipeline simulation or interactive CLI session."""
    logger.info("===============================================================")
    logger.info("Step 1: Initializing DB & Medical Knowledge Base (Qdrant)...")
    logger.info("===============================================================")
    await init_db()

    ingest_service = KnowledgeIngestionService()
    await ingest_service.initialize_collections()
    await ingest_service.ingest_document(
        source_name="NICE Guideline CG95 - Chest Pain Evaluation",
        content=(
            "NICE CG95 recommends evaluating chest pain onset, location, severity, duration, "
            "and radiation to shoulder, arm, neck, or jaw. Assess associated symptoms such as "
            "dyspnea, sweating (diaphoresis), nausea, and syncope. Mandatory red flag checks include "
            "ruling out acute coronary syndrome, hypertension, diabetes, and smoking history."
        ),
        section="Acute Coronary Syndromes",
    )

    conv_manager = ConversationManager()
    speech_service = SpeechService()

    async with get_session_context() as session:
        repo = ConversationRepository(session)
        hist_repo = HistoryRepository(session)

        logger.info(f"\nStep 2: Starting Live Session for Patient '{patient_id}'...")
        conv = await repo.create(patient_id=patient_id, chief_complaint=chief_complaint)
        conv_id = conv.id

        state = {
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
        await conv_manager.update_state(conv_id, state)

        # Default multi-turn inputs if none provided
        if not audio_paths_or_texts:
            audio_paths_or_texts = [
                "حاسس بوجع جامد في صدري من ساعتين وبيسمع في كتفي الشمال",
                "الالم شديد 8 من 10 ونزل عليا عرق بارد مع صعوبة تنفس",
                "لأ معنديش سكر بس باخد علاج لضغط الدم العالي",
            ]

        turn_index = 0
        while True:
            turn_index += 1
            logger.info(f"\n---------------------------------------------------------------")
            logger.info(f"🎙️  TURN {turn_index} - Processing Patient Input...")
            logger.info(f"---------------------------------------------------------------")

            raw_audio_bytes: Optional[bytes] = None
            user_input_text: str = ""

            if interactive:
                try:
                    logger.info("\nEnter audio file path OR type Egyptian Arabic patient response:")
                    user_entry = input("> ").strip()
                except (KeyboardInterrupt, EOFError):
                    logger.info("\nUser requested termination (Ctrl+C). Ending interview.")
                    break
                if not user_entry:
                    logger.info("Empty input received. Ending interview.")
                    break
                if os.path.exists(user_entry):
                    with open(user_entry, "rb") as f:
                        raw_audio_bytes = f.read()
                    logger.info(f"Loaded audio file: {user_entry} ({len(raw_audio_bytes)} bytes)")
                else:
                    user_input_text = user_entry
            else:
                if turn_index > len(audio_paths_or_texts):
                    logger.info("Completed all provided test turns.")
                    break
                current_item = audio_paths_or_texts[turn_index - 1]
                if isinstance(current_item, str) and os.path.exists(current_item):
                    with open(current_item, "rb") as f:
                        raw_audio_bytes = f.read()
                    logger.info(f"Using Audio File: '{current_item}' ({len(raw_audio_bytes)} bytes)")
                elif isinstance(current_item, bytes):
                    raw_audio_bytes = current_item
                    logger.info(f"Using Raw Audio Stream Bytes ({len(raw_audio_bytes)} bytes)")
                else:
                    user_input_text = current_item
                    logger.info(f"Using Patient Statement Input: '{user_input_text}'")

            # Load turn inputs into state machine
            state["raw_audio_bytes"] = raw_audio_bytes
            state["user_input_arabic"] = user_input_text

            # Execute 10-Node LangGraph State Machine Workflow
            state = await interview_app.ainvoke(state)

            # Extract results
            asr_text = state.get("user_input_arabic", "")
            en_text = state.get("user_input_english", "")
            extracted = state.get("extracted_entities", {})
            next_q_en = state.get("generated_question_english", "")
            next_q_ar = state.get("generated_question_arabic", "")
            metrics = state.get("evaluation_metrics", {})
            should_continue = state.get("should_continue", True)

            logger.info(f"🗣️  ASR Transcribed Text (Arabic) : '{asr_text}'")
            logger.info(f"🇬🇧 Clinical English Translation : '{en_text}'")
            logger.info(f"🔍 Extracted Medical Entities   : {json.dumps(extracted, ensure_ascii=False)}")
            logger.info(f"❓ Next Follow-Up Question (En) : '{next_q_en}'")
            logger.info(f"🇪🇬 Next Follow-Up Question (Ar) : '{next_q_ar}'")
            logger.info(
                f"📊 Interview Evaluation Metrics : Overall Score={metrics.get('overall_interview_score', 0):.4f} "
                f"| Coverage={metrics.get('coverage_score', 0):.4f} "
                f"| Guideline={metrics.get('guideline_coverage_score', 0):.4f}"
                f"| Termination Decision={metrics.get('termination_decision', 'N/A')}"
                f"| Step Count={metrics.get('step_count', 0)}"
                f" | Info Gain={metrics.get('information_gain', 0):.4f}"
                f" | Consistency={metrics.get('consistency_score', 0):.4f}"
                f" | Red Flags Covered={metrics.get('all_red_flags_covered', False)}"
                f" | Question Count={metrics.get('question_count', 0)}"
            )

            logger.info(f"🚦 Termination Engine Decision  : Should Continue = {should_continue}")

            await conv_manager.update_state(conv_id, state)
            await repo.update_state(
                conversation_id=conv_id,
                turns=state.get("turns", []),
                state_snapshot=state,
                step_count=state.get("step_count", turn_index),
                status=state.get("status", STATUS_ACTIVE),
            )

            if not should_continue:
                logger.info("\n✨ Termination condition met! Ending interview session.")
                break

        # Finalize and build structured medical history JSON if not already built
        if not state.get("structured_history"):
            from app.services.graph.nodes import history_builder_node
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

        logger.info("\n===============================================================")
        logger.info("🎉 FINAL STRUCTURED MEDICAL HISTORY JSON OUTPUT:")
        logger.info("===============================================================")
        logger.info(json.dumps(final_history.model_dump(), indent=2, ensure_ascii=False))
        logger.info("===============================================================")
        logger.info("Pipeline execution completed successfully with 0 errors!")


if __name__ == "__main__":
    is_interactive = "--interactive" in sys.argv or "-i" in sys.argv
    try:
        asyncio.run(run_live_pipeline(interactive=is_interactive))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("\nSession terminated by user (Ctrl+C). Exiting cleanly.")
