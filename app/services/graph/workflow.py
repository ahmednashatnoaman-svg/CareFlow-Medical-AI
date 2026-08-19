"""LangGraph Workflow State Machine Compilation."""

import logging
from typing import Any, Dict
from langgraph.graph import END, StateGraph
from app.services.graph.nodes import (
    conversation_update_node,
    entity_extraction_node,
    generate_question_node,
    history_builder_node,
    interview_evaluation_node,
    rerank_node,
    resolve_option_node,
    retrieval_node,
    speech_node,
    translate_input_node,
    translate_output_node,
    web_search_node,
)
from app.services.graph.state import InterviewState

logger = logging.getLogger(__name__)


def should_continue_router(state: InterviewState) -> str:
    """Routing function evaluating whether to build history & end or translate output question."""
    if state.get("should_continue", True):
        return "translate_output"
    return "build_history"


def create_interview_workflow() -> StateGraph:
    """Builds and compiles the LangGraph clinical interview state machine graph.

    Returns:
        Compiled LangGraph runner graph.
    """
    workflow = StateGraph(InterviewState)

    # Add Nodes
    workflow.add_node("speech_asr", speech_node)
    workflow.add_node("resolve_option", resolve_option_node)
    workflow.add_node("translate_input", translate_input_node)
    workflow.add_node("conversation_update", conversation_update_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("retrieve_chunks", retrieval_node)
    workflow.add_node("rerank_chunks", rerank_node)
    workflow.add_node("generate_question", generate_question_node)
    workflow.add_node("evaluate_interview", interview_evaluation_node)
    workflow.add_node("translate_output", translate_output_node)
    workflow.add_node("build_history", history_builder_node)

    # Set Entry Point
    workflow.set_entry_point("speech_asr")

    # Add Linear Sequential Edges
    workflow.add_edge("speech_asr", "resolve_option")
    workflow.add_edge("resolve_option", "translate_input")
    workflow.add_edge("translate_input", "conversation_update")
    workflow.add_edge("conversation_update", "entity_extraction")
    workflow.add_edge("entity_extraction", "web_search")
    workflow.add_edge("web_search", "retrieve_chunks")
    workflow.add_edge("retrieve_chunks", "rerank_chunks")
    workflow.add_edge("rerank_chunks", "generate_question")
    workflow.add_edge("generate_question", "evaluate_interview")

    # Add Conditional Branching
    workflow.add_conditional_edges(
        "evaluate_interview",
        should_continue_router,
        {
            "translate_output": "translate_output",
            "build_history": "build_history",
        },
    )

    workflow.add_edge("translate_output", END)
    workflow.add_edge("build_history", END)

    app = workflow.compile()
    logger.info("LangGraph clinical interview workflow compiled successfully")
    return app


interview_app = create_interview_workflow()
