"""Interactive Clinical Triage & Diagnosis Orchestration Service (Graph RAG).

Implements the multi-turn Graph RAG triage interview engine with SOCRATES tracking,
PrimeKG clinical evidence ranking, Shannon entropy stopping metrics,
multilingual Egyptian Arabic + English translation layers, and doctor's diagnostic reporting.
Based on the architecture in CF_KG_RAG.ipynb.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set
from careflow.core.config import settings
from careflow.services.llm_client import llm_client
from careflow.services.primekg_service import primekg_service
from careflow.services.session_store import SessionStore, build_session_store

logger = logging.getLogger(__name__)

# Structured System Prompt for Agent Question Formulation
TRIAGE_AGENT_PROMPT = """You are an expert clinical triage AI interviewing a patient to gather their History of Present Illness (HPI).
You will be provided with:
- The patient's latest message
- Confirmed positive symptoms
- Denied symptoms (Do NOT ask about these)
- Knowledge Graph suggested symptoms to evaluate
- Current SOCRATES clinical history slots

YOUR GOAL:
1. Choose ONE primary target symptom from the suggested symptoms to evaluate next.
2. Formulate a polite, empathetic question about it.
3. Provide EXACTLY 3 numbered options for the patient to choose from:
   - Option 1: Positive / Severe statement confirming the symptom.
   - Option 2: Moderate / Partial statement about the symptom.
   - Option 3: Negative statement denying the symptom.
4. Evaluate SOCRATES criteria for the patient's complaint (Site, Onset, Character, Radiation, Associated symptoms, Timing/Duration, Exacerbating/Relieving factors, Severity).

OUTPUT FORMAT:
You MUST reply in pure, valid JSON with this exact structure:
{
  "target_symptom": "name of symptom being evaluated",
  "empathy_and_question": "Brief empathetic remark + single question",
  "options": [
    "Option 1 description",
    "Option 2 description",
    "Option 3 description"
  ],
  "socrates_tracker": {
    "site": true/false,
    "onset": true/false,
    "character": true/false,
    "radiation": true/false,
    "associated_symptoms": true/false,
    "time_course": true/false,
    "exacerbating_relieving": true/false,
    "severity": true/false
  },
  "socrates_score": 0,
  "is_complete": false,
  "summary_if_complete": ""
}
"""

DIAGNOSTIC_REPORT_PROMPT = """You are an expert Chief Medical Officer. The triage interview is now complete.
You are provided with:
1. The Patient's Confirmed Positive Symptoms and Denied Symptoms.
2. Top Diagnostic Matches retrieved from the PrimeKG Knowledge Graph, with heuristic confidence scores and matched/conflicting evidence.

YOUR GOAL:
Generate a formal clinical diagnostic report evaluating the differential diagnoses (top 3 probable conditions).
For each diagnosis, translate the graph confidence score into a realistic estimated probability percentage (e.g. 78%).
State clinical reasoning and format the graph evidence path.
Assign an urgency level: 'Emergency (Immediate ER)', 'Urgent (Within 24h)', 'Routine Outpatient', or 'Self-Care/Home Monitoring'.

OUTPUT FORMAT:
You MUST reply in pure, valid JSON with this exact structure:
{
  "top_diagnoses": [
    {
      "diagnosis": "Name of disease",
      "estimated_probability": "XX%",
      "urgency_level": "Emergency | Urgent | Routine | Self-Care",
      "reasoning": "Clinical justification based on matched vs conflicting symptoms.",
      "graph_evidence": "(Patient) -> [presents_with] -> (Symptom A) <- [associated_with] <- (Disease X)"
    }
  ],
  "triage_recommendation": "Clear, actionable recommendation for the patient and human physician.",
  "clinical_summary": "Concise summary of positive symptoms, red flags ruled out, and duration."
}
"""

EXTRACTION_PROMPT = """You are a clinical NLP entity extractor.
Given a patient's utterance and previous context (last evaluated symptom, if any), identify:
1. 'positive_symptoms': List of symptoms the patient affirmatively reports or confirms having.
2. 'negated_symptoms': List of symptoms the patient denies, rules out, or states they do NOT have.

OUTPUT FORMAT:
Respond with pure JSON:
{
  "positive_symptoms": ["symptom1", "symptom2"],
  "negated_symptoms": ["symptom3"]
}
"""


class TriageSession:
    """Represents a single clinical triage session state."""

    def __init__(self, session_id: str, language: str = "en"):
        self.session_id = session_id
        self.language = language  # 'en' or 'ar'
        self.positive_symptoms: Set[str] = set()
        self.negated_symptoms: Set[str] = set()
        self.turn_count = 0
        self.max_turns = settings.MAX_QUESTIONS
        self.last_target_symptom: Optional[str] = None
        self.last_options_english: List[str] = []
        self.last_options_arabic: List[str] = []
        self.socrates_tracker: Dict[str, bool] = {
            "site": False,
            "onset": False,
            "character": False,
            "radiation": False,
            "associated_symptoms": False,
            "time_course": False,
            "exacerbating_relieving": False,
            "severity": False,
        }
        self.socrates_score = 0
        self.is_complete = False
        self.history: List[Dict[str, str]] = []
        self.diagnostic_report: Optional[Dict[str, Any]] = None
        self.last_graph_evidence: Optional[Dict[str, Any]] = None
        self.last_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the session store. Sets become lists to stay JSON-encodable."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "positive_symptoms": sorted(self.positive_symptoms),
            "negated_symptoms": sorted(self.negated_symptoms),
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "last_target_symptom": self.last_target_symptom,
            "last_options_english": self.last_options_english,
            "last_options_arabic": self.last_options_arabic,
            "socrates_tracker": self.socrates_tracker,
            "socrates_score": self.socrates_score,
            "is_complete": self.is_complete,
            "history": self.history,
            "diagnostic_report": self.diagnostic_report,
            "last_graph_evidence": self.last_graph_evidence,
            "last_stats": self.last_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriageSession":
        """Rebuild from stored state, tolerating fields absent in older payloads."""
        s = cls(session_id=data["session_id"], language=data.get("language", "en"))
        s.positive_symptoms = set(data.get("positive_symptoms", []))
        s.negated_symptoms = set(data.get("negated_symptoms", []))
        s.turn_count = data.get("turn_count", 0)
        s.max_turns = data.get("max_turns", settings.MAX_QUESTIONS)
        s.last_target_symptom = data.get("last_target_symptom")
        s.last_options_english = data.get("last_options_english", [])
        s.last_options_arabic = data.get("last_options_arabic", [])
        s.socrates_tracker = data.get("socrates_tracker", s.socrates_tracker)
        s.socrates_score = data.get("socrates_score", 0)
        s.is_complete = data.get("is_complete", False)
        s.history = data.get("history", [])
        s.diagnostic_report = data.get("diagnostic_report")
        s.last_graph_evidence = data.get("last_graph_evidence")
        s.last_stats = data.get("last_stats")
        return s


class TriageOrchestratorService:
    """Manages triage sessions, graph reasoning, and diagnostic reporting."""

    def __init__(self, store: Optional[SessionStore] = None):
        # Injectable so tests can supply a deterministic store without touching Redis.
        self._store = store or build_session_store()

    def get_or_create_session(self, session_id: Optional[str] = None, language: str = "en") -> TriageSession:
        """Load an existing session or start a new one.

        Callers must persist mutations with `save_session`; the store may live out of
        process, so mutating the returned object is no longer sufficient on its own.
        """
        sid = session_id or str(uuid.uuid4())
        stored = self._store.get(sid)
        if stored is not None:
            return TriageSession.from_dict(stored)

        session = TriageSession(session_id=sid, language=language)
        self._store.set(sid, session.to_dict())
        return session

    def save_session(self, session: TriageSession) -> None:
        """Write session state back. Required after every turn."""
        self._store.set(session.session_id, session.to_dict())

    def reset_session(self, session_id: str) -> TriageSession:
        existing = self._store.get(session_id)
        lang = (existing or {}).get("language", "en")
        session = TriageSession(session_id=session_id, language=lang)
        self._store.set(session_id, session.to_dict())
        return session

    def extract_symptoms_from_input(
        self,
        user_input: str,
        last_target_symptom: Optional[str] = None,
        last_options: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """Extracts confirmed and negated symptoms with option mapping and fallback rules."""
        resolved_input = user_input.strip()

        # Check numeric selection (1, 2, 3)
        if resolved_input in ["1", "2", "3", "١", "٢", "٣"] and last_options and len(last_options) == 3:
            idx = int(resolved_input.translate(str.maketrans("١٢٣", "123"))) - 1
            resolved_input = last_options[idx]

        lower_text = resolved_input.lower()
        extracted = {"positive_symptoms": [], "negated_symptoms": []}

        # Handle simple Yes/No affirmations
        is_yes = any(w in lower_text for w in ["yes", "yeah", "yep", "sure", "positive", "true", "i do", "ايوه", "اه", "نعم", "عندي", "موجود"])
        is_no = any(w in lower_text for w in ["no", "nope", "nah", "none", "don't", "do not", "denied", "false", "لا", "مافيش", "مش عندي", "لا يوجد"])

        if last_target_symptom:
            if is_yes and not is_no:
                extracted["positive_symptoms"].append(last_target_symptom)
                return extracted
            elif is_no and not is_yes:
                extracted["negated_symptoms"].append(last_target_symptom)
                return extracted

        # Use LLM for deeper entity extraction
        try:
            prompt = f"""Context Last Evaluated Symptom: "{last_target_symptom or 'None'}"
Patient Message: "{resolved_input}"
Extract positive and negated symptoms:"""
            res = llm_client.generate_json(
                prompt=prompt,
                system_prompt=EXTRACTION_PROMPT,
                temperature=0.0,
                max_tokens=300,
            )
            extracted["positive_symptoms"].extend(res.get("positive_symptoms", []))
            extracted["negated_symptoms"].extend(res.get("negated_symptoms", []))
        except Exception as e:
            logger.warning("LLM symptom extraction failed: %s. Falling back to keyword matches.", e)
            for word in resolved_input.split():
                if len(word) > 3:
                    extracted["positive_symptoms"].append(word.lower())

        return extracted

    async def process_user_turn(
        self,
        session_id: str,
        user_message: str,
        forced_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes one turn of the interactive triage interview."""
        session = self.get_or_create_session(session_id)

        # Detect language (Arabic if contains Arabic characters)
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", user_message))
        if forced_language:
            session.language = forced_language
        elif has_arabic:
            session.language = "ar"

        session.history.append({"role": "user", "content": user_message})

        # --- Step 1: Inbound Resolution & Translation ---
        resolved_english = user_message
        resolved_arabic = user_message

        # Check if user typed option number 1, 2, or 3
        clean_input = user_message.strip()
        if clean_input in ["1", "2", "3", "١", "٢", "٣"]:
            idx = int(clean_input.translate(str.maketrans("١٢٣", "123"))) - 1
            if session.last_options_english and len(session.last_options_english) > idx:
                resolved_english = session.last_options_english[idx]
            if session.last_options_arabic and len(session.last_options_arabic) > idx:
                resolved_arabic = session.last_options_arabic[idx]
        elif session.language == "ar":
            resolved_english = llm_client.translate_patient_to_english(user_message)

        # --- Step 2: Clinical Entity Extraction & State Tracking ---
        extracted = self.extract_symptoms_from_input(
            resolved_english,
            last_target_symptom=session.last_target_symptom,
            last_options=session.last_options_english,
        )

        session.positive_symptoms.update(s.lower() for s in extracted["positive_symptoms"])
        session.negated_symptoms.update(s.lower() for s in extracted["negated_symptoms"])
        session.positive_symptoms -= session.negated_symptoms

        current_positives = list(session.positive_symptoms)
        current_negated = list(session.negated_symptoms)

        session.turn_count += 1

        # --- Step 3: PrimeKG Evidence & Graph Traversal ---
        graph_evidence = primekg_service.calculate_diagnostic_evidence(
            {"positive_symptoms": current_positives, "negated_symptoms": current_negated},
            top_k=3,
        )
        stats = primekg_service.calculate_statistical_confidence(graph_evidence)
        graph_candidates_res = primekg_service.get_next_symptom_candidates(current_positives, top_k_diseases=3)
        candidate_symptoms = graph_candidates_res.get("Suggested Next Symptoms to Ask", [])

        session.last_graph_evidence = graph_evidence
        session.last_stats = stats

        # --- Step 4: Check Stopping Criteria ---
        stop_flag, stop_reason = primekg_service.should_stop_interview(
            turn_count=session.turn_count,
            stats=stats,
            socrates_score=session.socrates_score,
            red_flags_cleared=True,
        )

        # --- Step 5: If Complete -> Generate Doctor's Diagnostic Report ---
        if stop_flag:
            session.is_complete = True
            report = self._generate_final_report(session, graph_evidence)
            session.diagnostic_report = report

            closing_message_en = "Triage interview complete. I have gathered your clinical history and generated the differential diagnosis report for the medical team."
            closing_message_ar = "تم اكتمال الفحص المبدئي بنجاح. تم تسجيل التاريخ المرضي وإعداد التقرير التشخيصي المبدئي للطبيب المعالج."

            final_text = closing_message_ar if session.language == "ar" else closing_message_en
            session.history.append({"role": "assistant", "content": final_text})
            self.save_session(session)

            return {
                "session_id": session.session_id,
                "is_complete": True,
                "message": final_text,
                "options": [],
                "target_symptom": None,
                "socrates_tracker": session.socrates_tracker,
                "socrates_score": session.socrates_score,
                "stop_reason": stop_reason,
                "positive_symptoms": current_positives,
                "negated_symptoms": current_negated,
                "stats": stats,
                "diagnostic_report": report,
            }

        # --- Step 6: Generate Next Structured Question via LLM ---
        context_block = f"""
[CLINICAL INTERVIEW STATE]
Turn Count: {session.turn_count}/{session.max_turns}
Latest Patient Utterance: "{resolved_english}"
Confirmed Symptoms: {current_positives}
Denied Symptoms: {current_negated}
Knowledge Graph Suggested Symptoms: {candidate_symptoms}
Current SOCRATES Slots: {json.dumps(session.socrates_tracker)}
"""
        agent_json = llm_client.generate_json(
            prompt=context_block + "\nGenerate the next structured triage question. Output ONLY raw JSON.",
            system_prompt=TRIAGE_AGENT_PROMPT,
            temperature=0.2,
            max_tokens=800,
        )

        # Update SOCRATES tracking from agent evaluation
        if "socrates_tracker" in agent_json and isinstance(agent_json["socrates_tracker"], dict):
            for k, v in agent_json["socrates_tracker"].items():
                if k in session.socrates_tracker:
                    session.socrates_tracker[k] = session.socrates_tracker[k] or bool(v)
            session.socrates_score = sum(1 for v in session.socrates_tracker.values() if v)

        session.last_target_symptom = agent_json.get("target_symptom")
        session.last_options_english = agent_json.get("options", [
            f"Yes, I have severe {session.last_target_symptom or 'symptoms'}",
            f"Yes, mild or moderate {session.last_target_symptom or 'symptoms'}",
            f"No, I do not have {session.last_target_symptom or 'symptoms'}",
        ])

        # --- Step 7: Outbound Translation if Arabic ---
        outbound_payload = agent_json
        if session.language == "ar":
            arabic_json = llm_client.translate_agent_to_arabic(agent_json)
            session.last_options_arabic = arabic_json.get("options", session.last_options_english)
            outbound_payload = arabic_json
        else:
            session.last_options_arabic = session.last_options_english

        assistant_msg = outbound_payload.get("empathy_and_question", "Could you please describe how you are feeling?")
        session.history.append({"role": "assistant", "content": assistant_msg})
        self.save_session(session)

        return {
            "session_id": session.session_id,
            "is_complete": False,
            "message": assistant_msg,
            "options": outbound_payload.get("options", []),
            "target_symptom": session.last_target_symptom,
            "socrates_tracker": session.socrates_tracker,
            "socrates_score": session.socrates_score,
            "turn_count": session.turn_count,
            "max_turns": session.max_turns,
            "positive_symptoms": current_positives,
            "negated_symptoms": current_negated,
            "stats": stats,
            "stop_reason": stop_reason,
            "diagnostic_report": None,
        }

    def _generate_final_report(self, session: TriageSession, graph_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Generates structured differential diagnosis report using PrimeKG evidence."""
        prompt = f"""
[PATIENT CLINICAL HISTORY]
Confirmed Positive Symptoms: {list(session.positive_symptoms)}
Denied Symptoms: {list(session.negated_symptoms)}
SOCRATES Criteria Profile: {json.dumps(session.socrates_tracker)}
Turns Conducted: {session.turn_count}

[PRIMEKG KNOWLEDGE GRAPH EVIDENCE]
{json.dumps(graph_evidence, indent=2)}

Generate the formal clinical diagnostic report in valid JSON."""

        try:
            report = llm_client.generate_json(
                prompt=prompt,
                system_prompt=DIAGNOSTIC_REPORT_PROMPT,
                temperature=0.1,
                max_tokens=1000,
            )
            return report
        except Exception as e:
            logger.error("Failed to generate clinical diagnostic report: %s", e)
            # Fallback structured report
            top_dx = []
            for dx, data in list(graph_evidence.items())[:3]:
                top_dx.append({
                    "diagnosis": dx.title(),
                    "estimated_probability": f"{int(data.get('raw_confidence_score', 0.5) * 100)}%",
                    "urgency_level": "Urgent",
                    "reasoning": f"Matched symptoms: {', '.join(data.get('matched_evidence', []))}.",
                    "graph_evidence": f"(Patient) -> [has] -> {data.get('matched_evidence', [])} <- [linked] <- ({dx})",
                })
            return {
                "top_diagnoses": top_dx,
                "triage_recommendation": "Please consult a healthcare professional for physical examination and definitive evaluation.",
                "clinical_summary": f"Confirmed symptoms: {list(session.positive_symptoms)}. Ruled out: {list(session.negated_symptoms)}.",
            }


# Global singleton instance
triage_service = TriageOrchestratorService()
