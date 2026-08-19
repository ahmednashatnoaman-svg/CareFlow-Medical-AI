"""Interactive Command-Line Interface for Dual-Mode Medical Chatbot.

Usage:
    python scripts/run_chatbot_cli.py --mode triage
    python scripts/run_chatbot_cli.py --mode dialogue
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from careflow.services.dialogue_service import dialogue_service
from careflow.services.triage_service import triage_service


async def run_triage_cli(language: str = "en"):
    print("=================================================================")
    print("🩺 CareFlow Medical AI — Graph RAG Diagnostic Triage (Mode 1)")
    print("=================================================================")
    print("Type 'exit' or 'quit' at any time to end the interview.\n")

    session_id = f"cli_{os.urandom(4).hex()}"
    triage_service.get_or_create_session(session_id=session_id, language=language)

    greeting = (
        "المساعد: أهلاً بيك. ألف سلامة عليك، تقدر تقولي حاسس بإيه أو بتشتكي من إيه النهاردة؟"
        if language == "ar"
        else "Assistant: Hello! I am your clinical triage assistant. What symptoms are you experiencing today?"
    )
    print(f"{greeting}\n")

    while True:
        prompt_label = "المريض: " if language == "ar" else "Patient: "
        try:
            user_input = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding interview.")
            break

        if not user_input or user_input.lower() in ["exit", "quit", "q", "خروج"]:
            print("\nEnding interview.")
            break

        res = await triage_service.process_user_turn(session_id, user_input, forced_language=language)

        print(f"\nAssistant: {res['message']}\n")

        if res.get("is_complete"):
            print("=================================================================")
            print("📋 CLINICAL DIFFERENTIAL DIAGNOSIS REPORT (For Attending Doctor)")
            print("=================================================================")
            report = res.get("diagnostic_report") or {}
            top_dx = report.get("top_diagnoses", [])
            for idx, dx in enumerate(top_dx, 1):
                print(f"[{idx}] Diagnosis: {dx.get('diagnosis', '').upper()} (Probability: {dx.get('estimated_probability', 'N/A')})")
                print(f"    Urgency:   {dx.get('urgency_level', 'Urgent')}")
                print(f"    Reasoning: {dx.get('reasoning', '')}")
                print(f"    Evidence:  {dx.get('graph_evidence', '')}\n")
            print(f"Triage Recommendation: {report.get('triage_recommendation', 'N/A')}")
            print(f"SOCRATES Score: {res.get('socrates_score', 0)}/8")
            print("=================================================================\n")
            break

        options = res.get("options", [])
        if options:
            prompt_opts = "اختر رقم الإجابة (1، 2، 3) أو اكتب ردك:" if language == "ar" else "Please choose an option (1, 2, or 3) or type your response:"
            print(prompt_opts)
            for idx, opt in enumerate(options, 1):
                print(f"  [{idx}] {opt}")
            print()


async def run_dialogue_cli():
    print("=================================================================")
    print("📚 CareFlow Medical AI — WHO Guidelines Vector RAG (Mode 2)")
    print("=================================================================")
    print("Connected to Qdrant Cloud (who_guidelines collection, 1024d BGE-M3)")
    print("Type 'exit' or 'quit' at any time to end.\n")

    history = []

    while True:
        try:
            query = input("User Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting dialogue.")
            break

        if not query or query.lower() in ["exit", "quit", "q", "خروج"]:
            print("\nExiting dialogue.")
            break

        print("\nSearching WHO Guidelines & Synthesizing Grounded Answer...\n")
        res = await dialogue_service.answer_question(query=query, top_k=4, conversation_history=history)

        print("=== WHO GUIDELINES ANSWER ===")
        print(res["answer"])
        print("\n=== SOURCES & CITATIONS ===")
        for s in res.get("sources", []):
            print(f"• Document: {s['source_file']}")
            print(f"  Section:  {s['section']} (Relevance Match: {s['relevance_score'] * 100:.1f}%)")
            print(f"  Snippet:  {s['snippet'][:150]}...\n")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": res["answer"]})


def main():
    parser = argparse.ArgumentParser(description="CareFlow Dual-Mode Medical Chatbot CLI")
    parser.add_argument(
        "--mode",
        choices=["triage", "dialogue"],
        default="triage",
        help="Operating mode: 'triage' (Mode 1: Graph RAG) or 'dialogue' (Mode 2: WHO Guidelines RAG)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ar"],
        default="en",
        help="Language for triage mode: 'en' (English) or 'ar' (Egyptian Arabic)",
    )
    args = parser.parse_args()

    if args.mode == "triage":
        asyncio.run(run_triage_cli(language=args.lang))
    else:
        asyncio.run(run_dialogue_cli())


if __name__ == "__main__":
    main()
