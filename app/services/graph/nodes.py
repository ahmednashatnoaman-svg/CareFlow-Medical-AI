"""LangGraph State Machine Nodes.

Each node is independently testable, loosely coupled, and replaceable.
"""

import json
import logging
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from app.core.arabic_text import match_option
from app.core.config import settings
from app.core.constants import STATUS_ACTIVE, STATUS_COMPLETED
from app.services.graph.state import InterviewState
from app.services.entity_extraction_service import EntityExtractionService
from app.services.history_builder_service import MedicalHistoryBuilder
from app.services.orchestrator_service import LLMOrchestrator
from app.services.reranker_service import CrossEncoderReranker
from app.services.retrieval_service import RetrievalEngine
from app.services.speech_service import SpeechService
from app.services.termination_engine import InterviewTerminationEngine
from app.services.translation_service import TranslationService
from app.services.web_search_service import MayoClinicSearchService

logger = logging.getLogger(__name__)

# Content-free replies that must never be recorded as a collected attribute value --
# doing so would both inflate coverage_score and mask an exhausted attribute.
_NON_ANSWER_PHRASES = frozenset(
    {
        "i don't know", "i dont know", "not sure", "i'm not sure", "im not sure",
        "no idea", "not applicable", "n/a", "na", "don't know", "dont know", "unknown",
    }
)

# Service Instances
speech_service = SpeechService()
translation_service = TranslationService()
entity_service = EntityExtractionService()
retrieval_engine = RetrievalEngine()
reranker_service = CrossEncoderReranker()
orchestrator_service = LLMOrchestrator()
termination_engine = InterviewTerminationEngine()
history_builder = MedicalHistoryBuilder()
web_search_service = MayoClinicSearchService()


async def speech_node(state: InterviewState) -> Dict[str, Any]:
    """Node 1: Converts incoming audio stream into Egyptian Arabic transcript if audio present."""
    audio_bytes = state.get("raw_audio_bytes")
    if audio_bytes:
        res = await speech_service.transcribe_audio(audio_bytes)
        transcript = res.get("transcript", "").replace("@@@فراغ", "").strip()
        out: Dict[str, Any] = {"raw_audio_bytes": None}
        if transcript:
            out["user_input_arabic"] = transcript
        return out
    return {"raw_audio_bytes": None}


async def resolve_option_node(state: InterviewState) -> Dict[str, Any]:
    """Node 1b: Resolves a spoken/typed answer against the options offered last turn.

    Runs after ASR so it sees the transcript from both the audio and text paths. A hit
    supplies the option's English directly, skipping the AR->EN round trip. A miss is
    never an error -- the utterance simply flows on as free text.
    """
    options = state.get("question_options", [])
    transcript = state.get("user_input_arabic", "")
    if not options or not transcript:
        return {"option_resolved": False, "selected_option_index": None}

    index, confidence = match_option(
        transcript, options, threshold=settings.OPTION_MATCH_THRESHOLD
    )
    if index is None:
        logger.info("No option matched (best score %.2f); treating as free text", confidence)
        return {"option_resolved": False, "selected_option_index": None}

    selected = next((o for o in options if o.get("index") == index), None)
    if not selected:
        return {"option_resolved": False, "selected_option_index": None}

    logger.info("Resolved option %d ('%s') at confidence %.2f", index, selected.get("text_english", ""), confidence)
    text_en = selected.get("text_english", "")
    return {
        "user_input_english": text_en,
        "selected_option_index": index,
        "option_resolved": True,
        "messages": [HumanMessage(content=text_en)],
    }


async def translate_input_node(state: InterviewState) -> Dict[str, Any]:
    """Node 2: Translates patient Arabic transcript into canonical clinical English."""
    if state.get("option_resolved"):
        return {}

    ar_text = state.get("user_input_arabic", "")
    if ar_text:
        en_text = await translation_service.translate_ar_to_en(ar_text)
        return {
            "user_input_english": en_text,
            "messages": [HumanMessage(content=en_text)],
        }
    return {}


async def conversation_update_node(state: InterviewState) -> Dict[str, Any]:
    """Node 3: Increments step count and updates chief complaint if unassigned."""
    step = state.get("step_count", 0) + 1
    en_text = state.get("user_input_english", "")
    chief = state.get("chief_complaint", "")
    if not chief and en_text:
        chief = en_text
    return {"step_count": step, "chief_complaint": chief}


def _knowledge_signature(extracted: Dict[str, Any]) -> str:
    """Snapshot of what is known, used to detect whether an answer added anything.

    Includes attribute *values*, not just which keys are present -- refining a known
    attribute (location "Abdomen" -> "Lower right") is new information too.
    """
    attributes = extracted.get("attributes", {}) or {}
    return json.dumps(
        {
            "symptoms": sorted(extracted.get("symptoms", []) or []),
            "diseases": sorted(extracted.get("diseases", []) or []),
            "risk_factors": sorted(extracted.get("risk_factors", []) or []),
            "body_locations": sorted(extracted.get("body_locations", []) or []),
            "medications": sorted(extracted.get("medications", []) or []),
            "procedures": sorted(extracted.get("procedures", []) or []),
            "allergies": sorted(extracted.get("allergies", []) or []),
            "family_history": sorted(extracted.get("family_history", []) or []),
            "social_history": sorted(extracted.get("social_history", []) or []),
            "attributes": {
                k: v for k, v in sorted(attributes.items()) if v not in (None, "", [], {})
            },
        },
        sort_keys=True,
        default=str,
    )


def _extract_history_logs(messages: Any) -> List[Dict[str, Any]]:
    """Normalizes messages (BaseMessage, dict, or string) into standard [{role: 'user'|'assistant', content: str}] list."""
    history_logs: List[Dict[str, Any]] = []
    for m in (messages or []):
        if isinstance(m, BaseMessage):
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            content = getattr(m, "content", "")
        elif isinstance(m, dict):
            role = m.get("role") or ("user" if m.get("type") in ("human", "user") else "assistant")
            content = m.get("content", "")
        else:
            role = "user"
            content = str(m)
        if content:
            history_logs.append({"role": role, "content": str(content)})
    return history_logs


async def entity_extraction_node(state: InterviewState) -> Dict[str, Any]:
    """Node 4: Extracts medical entities (Symptoms, Diseases, Risk Factors, Attributes)."""
    en_text = state.get("user_input_english", "")
    current_extracted = dict(state.get("extracted_entities", {}))
    before = _knowledge_signature(current_extracted)

    if en_text:
        extracted = await entity_service.extract_entities(en_text)

        # Merge extracted entities across all categories
        merged_symptoms = list(set(current_extracted.get("symptoms", []) + extracted.symptoms))
        merged_diseases = list(set(current_extracted.get("diseases", []) + extracted.diseases))
        merged_risks = list(set(current_extracted.get("risk_factors", []) + extracted.risk_factors))
        merged_locations = list(set(current_extracted.get("body_locations", []) + extracted.body_locations))
        merged_meds = list(set(current_extracted.get("medications", []) + extracted.medications))
        merged_procs = list(set(current_extracted.get("procedures", []) + extracted.procedures))
        merged_allergies = list(set(current_extracted.get("allergies", []) + extracted.allergies))
        merged_family_history = list(set(current_extracted.get("family_history", []) + extracted.family_history))
        merged_social_history = list(set(current_extracted.get("social_history", []) + extracted.social_history))

        attrs = dict(current_extracted.get("attributes", {}))
        # Extraction emits the full attribute schema with nulls for whatever this answer
        # didn't mention; a blind update would erase everything learned in earlier turns.
        attrs.update(
            {k: v for k, v in extracted.attributes.items() if v not in (None, "", [], {})}
        )

        # If a question targeted an attribute, associate patient answer with that attribute.
        # A content-free non-answer ("I don't know", "not sure") must NOT be stamped in as
        # the attribute's value -- otherwise it inflates coverage_score as if real clinical
        # data were collected, and it always looks like "new information" on the attribute's
        # first ask (the slot goes from empty to non-empty), which defeats the exhaustion
        # check below and lets the orchestrator believe a dead-end attribute was answered.
        asked = state.get("asked_questions", [])
        target = asked[-1].get("target_attribute") if asked else ""
        is_non_answer = en_text.strip().lower() in _NON_ANSWER_PHRASES
        if target and not is_non_answer:
            if state.get("option_resolved") or not attrs.get(target):
                attrs[target] = en_text

        current_extracted["symptoms"] = merged_symptoms
        current_extracted["diseases"] = merged_diseases
        current_extracted["risk_factors"] = merged_risks
        current_extracted["body_locations"] = merged_locations
        current_extracted["medications"] = merged_meds
        current_extracted["procedures"] = merged_procs
        current_extracted["allergies"] = merged_allergies
        current_extracted["family_history"] = merged_family_history
        current_extracted["social_history"] = merged_social_history
        current_extracted["attributes"] = attrs

    yielded_new_info = _knowledge_signature(current_extracted) != before
    if not yielded_new_info:
        logger.info("Answer yielded no new information; its attribute will be marked exhausted")

    return {"extracted_entities": current_extracted, "yielded_new_info": yielded_new_info}


async def web_search_node(state: InterviewState) -> Dict[str, Any]:
    """Node 4b: Looks up newly-identified diseases on Mayo Clinic to enrich (not replace) guideline RAG."""
    if settings.ENABLE_DIRECT_GEMINI_TESTING:
        return {}

    extracted = state.get("extracted_entities", {})
    diseases = extracted.get("diseases", [])
    already_searched = set(state.get("web_searched_diseases", []))
    new_diseases = [d for d in diseases if d not in already_searched]

    if not new_diseases:
        return {}

    results = await web_search_service.search_disease_info(new_diseases)
    combined = state.get("web_evidence", []) + results
    updated_searched = list(already_searched | set(new_diseases))
    return {"web_evidence": combined, "web_searched_diseases": updated_searched}


def _build_retrieval_query(state: InterviewState) -> str:
    """Builds the clinical retrieval query from the whole known picture, not just this turn."""
    en_text = state.get("user_input_english", "")
    chief = state.get("chief_complaint", "")
    extracted = state.get("extracted_entities", {})
    symptoms = extracted.get("symptoms", [])

    parts = [p for p in (chief, ", ".join(symptoms), en_text) if p]
    return " ".join(dict.fromkeys(parts)) or chief or en_text


async def retrieval_node(state: InterviewState) -> Dict[str, Any]:
    """Node 5: Performs hierarchical search over medical guideline documents & chunks."""
    if settings.ENABLE_DIRECT_GEMINI_TESTING:
        return {"retrieved_chunks": [], "retrieval_degraded": False}

    query = _build_retrieval_query(state)
    try:
        chunks = await retrieval_engine.search_chunks(query, top_k=20)
    except RuntimeError:
        logger.error("Embedding generation failed; proceeding with zero retrieved evidence", exc_info=True)
        return {"retrieved_chunks": [], "retrieval_degraded": True}
    return {"retrieved_chunks": chunks, "retrieval_degraded": False}


async def rerank_node(state: InterviewState) -> Dict[str, Any]:
    """Node 6: Cross-Encoder reranking of Top 20 retrieved chunks to Top 10 context chunks."""
    if settings.ENABLE_DIRECT_GEMINI_TESTING:
        return {"reranked_chunks": []}

    query = _build_retrieval_query(state)
    chunks = state.get("retrieved_chunks", [])
    reranked = await reranker_service.rerank(query, chunks, top_n=10)
    return {"reranked_chunks": reranked}


async def generate_question_node(state: InterviewState) -> Dict[str, Any]:
    """Node 7: LLM Orchestrator generates single evidence-grounded follow-up question."""
    prev_questions = state.get("previous_questions", [])
    prev_answers = state.get("previous_answers", [])
    asked_questions = list(state.get("asked_questions", []))
    exhausted = list(state.get("exhausted_attributes", []))
    missing_info = list(state.get("missing_info", []))
    pending_red_flags = state.get("evaluation_metrics", {}).get("unanswered_red_flags", [])
    for flag in pending_red_flags:
        if flag not in missing_info:
            missing_info.append(flag)
    reranked = state.get("reranked_chunks", [])
    extracted = state.get("extracted_entities", {})
    web_evidence = state.get("web_evidence", [])
    answer_en = state.get("user_input_english", "")
    messages = state.get("messages", [])

    history_logs = _extract_history_logs(messages)

    if asked_questions:
        last = dict(asked_questions[-1])
        yielded = state.get("yielded_new_info", True)
        last["yielded_new_info"] = yielded
        last["answer_english"] = answer_en
        asked_questions[-1] = last

        target = last.get("target_attribute")
        if not yielded and target and target not in exhausted:
            exhausted.append(target)
            logger.info("Marking attribute '%s' exhausted -- last answer added no information", target)

    result = await orchestrator_service.generate_next_question(
        conversation_history=history_logs,
        structured_state={**extracted, "chief_complaint": state.get("chief_complaint", "")},
        retrieved_chunks=reranked,
        previous_questions=prev_questions,
        missing_info=missing_info,
        web_evidence=web_evidence,
        exhausted_attributes=exhausted,
    )

    question_en = result.get("question", "")
    options = result.get("options", [])
    asked_questions.append(
        {
            "question_english": question_en,
            "target_attribute": result.get("target_attribute", ""),
            "yielded_new_info": None,
            "answer_english": "",
        }
    )

    output: Dict[str, Any] = {
        "messages": [AIMessage(content=question_en)],
        "generated_question_english": question_en,
        "question_options": options,
        "previous_questions": prev_questions + [question_en],
        "previous_answers": prev_answers + ([answer_en] if answer_en else []),
        "asked_questions": asked_questions,
        "exhausted_attributes": exhausted,
        "missing_info": result.get("missing_info", []),
    }
    if result.get("question_arabic"):
        output["generated_question_arabic"] = result.get("question_arabic")
    return output


async def interview_evaluation_node(state: InterviewState) -> Dict[str, Any]:
    """Node 8: Interview Termination Engine computes score metrics & decides continue/stop."""
    step = state.get("step_count", 1)
    chief = state.get("chief_complaint", "")
    extracted = state.get("extracted_entities", {})
    messages = state.get("messages", [])

    history_logs = _extract_history_logs(messages)

    prev_questions = state.get("previous_questions", [])
    entity_labels = (
        extracted.get("symptoms", [])
        + extracted.get("risk_factors", [])
        + extracted.get("diseases", [])
    )
    red_flag_corpus = prev_questions + entity_labels
    metrics = termination_engine.evaluate_interview(
        question_count=step,
        chief_complaint=chief,
        extracted_attributes=extracted,
        answered_red_flags=red_flag_corpus,
        conversation_history=history_logs,
    )

    metrics_dict = metrics.model_dump()
    metrics_dict["asked_guideline_questions_count"] = len(state.get("asked_questions", []))
    should_cont = not metrics.should_terminate
    status_str = STATUS_COMPLETED if metrics.should_terminate else STATUS_ACTIVE

    return {
        "evaluation_metrics": metrics_dict,
        "should_continue": should_cont,
        "status": status_str,
    }


async def translate_output_node(state: InterviewState) -> Dict[str, Any]:
    """Node 9: Translates the generated question and its answer options into Egyptian Arabic."""
    q_en = state.get("generated_question_english", "")
    q_ar = state.get("generated_question_arabic", "")
    options = state.get("question_options", [])
    updates: Dict[str, Any] = {}

    if settings.ENABLE_DIRECT_GEMINI_TESTING and q_ar and all(o.get("text_arabic") for o in options):
        return {}

    if q_en and not q_ar:
        updates["generated_question_arabic"] = await translation_service.translate_en_to_ar(q_en)

    if options and any(not o.get("text_arabic") for o in options):
        arabic_texts = await translation_service.translate_options_en_to_ar(
            [str(o.get("text_english", "")) for o in options]
        )
        updates["question_options"] = [
            {**option, "text_arabic": option.get("text_arabic") or arabic}
            for option, arabic in zip(options, arabic_texts)
        ]

    return updates


async def history_builder_node(state: InterviewState) -> Dict[str, Any]:
    """Node 10: Constructs final structured medical history JSON on interview completion."""
    metrics_dict = state.get("evaluation_metrics", {})
    score = metrics_dict.get("overall_interview_score", 0.0)

    structured_obj = history_builder.build_structured_history(
        conversation_state=state,
        overall_interview_score=score,
    )
    return {"structured_history": structured_obj.model_dump()}
