"""Safety guardrails for the RAG path.

A grounding *instruction* in a prompt is a request, not a guarantee: the model can still
answer from parametric memory when retrieval returns nothing, invent a dosage that appears
in no retrieved document, or be talked out of its instructions by text in the user's query.
These checks run outside the model, so they hold regardless of what it decides to emit.

Three stages:

  check_query()   -- before retrieval. Escalates medical emergencies (a guideline lookup is
                     the wrong response to "I'm having crushing chest pain") and refuses
                     prompt-injection attempts before they reach the grounding prompt.
  check_answer()  -- after generation. Blocks answers that were produced with no retrieved
                     context at all, and flags clinical numbers (doses, thresholds) that
                     appear in the answer but in none of the retrieved passages.
  enforce_disclaimer() -- guarantees the medical disclaimer survives, even if the model
                     drops it.

Each check is deterministic and LLM-free: guardrails that depend on a model call fail
exactly when the model is failing, and add latency to every request.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from careflow.core.config import settings

logger = logging.getLogger(__name__)


# The exact sentence the grounding prompt instructs the model to emit when context is
# insufficient. Also the canned answer used when a guardrail blocks an ungrounded answer,
# so users see one consistent refusal rather than two differently-worded ones.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "The provided WHO guidelines do not contain sufficient information on this specific topic."
)

MEDICAL_DISCLAIMER = (
    "This information is for educational and clinical guideline guidance and does not "
    "replace emergency clinical judgment or a consultation with a qualified clinician."
)

EMERGENCY_MESSAGE = (
    "**⚠️ THIS MAY BE A MEDICAL EMERGENCY — SEEK IMMEDIATE HELP.**\n\n"
    "Based on what you have described, please contact emergency services now "
    "(**123** in Egypt, **112** across the EU, **911** in the US) or go to the nearest "
    "emergency department immediately.\n\n"
    "Do not wait for an online answer, and do not drive yourself if you feel faint, "
    "breathless, or confused. If someone is with you, ask them to stay with you.\n\n"
    "This assistant retrieves clinical guidelines for educational use. It cannot assess, "
    "diagnose, or treat an emergency."
)

INJECTION_REFUSAL_MESSAGE = (
    "This request could not be processed. This assistant answers clinical questions "
    "strictly from retrieved WHO/CDC/USPSTF guideline passages, and cannot be instructed "
    "to disregard that grounding or reveal its internal configuration.\n\n"
    "Please rephrase as a clinical question and it will be answered from the guidelines."
)


# --- Emergency detection ----------------------------------------------------------------
#
# Requires BOTH a first-person distress signal AND a red-flag symptom. A guidelines
# assistant is *expected* to field questions like "what does WHO recommend for chest pain?"
# -- keying on the symptom alone would escalate every one of them into a false alarm and
# train users to ignore the warning, which is worse than not having it.

# Deliberately does NOT require a verb after the pronoun: "I'm bleeding heavily" carries the
# subject inside the contraction, and demanding a separate verb token silently dropped it.
# A bare first-person pronoun is enough, because escalation additionally requires a
# red-flag symptom -- this regex only has to answer "is the speaker talking about
# themselves or someone in front of them?", not "is this urgent?".
_FIRST_PERSON_DISTRESS = re.compile(
    r"\b(i|i'?m|im|i'?ve|ive|my|me|mine|we|we'?re|us|our)\b"
    # Third person needs an accompanying verb, so "her dose" (informational) stays out
    # while "she is unconscious" (someone present) comes in.
    r"|\b(he|she|they|his|her|their)\b.{0,30}?\b(is|are|was|were|has|have|got|just)\b"
    r"|\b(help|emergency|urgent|right now|can'?t|cannot|cant)\b",
    re.IGNORECASE,
)

# Each entry: (label, pattern). Patterns target acute presentations, not the condition name
# in the abstract -- "myocardial infarction" as a topic is informational; "crushing chest
# pain radiating to my arm" is a presentation.
_EMERGENCY_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    (
        "acute coronary syndrome",
        re.compile(
            r"(crushing|severe|squeezing|tight(ness)?|pressure).{0,30}chest"
            r"|chest.{0,30}(pain|pressure|tight).{0,60}(arm|jaw|neck|back|sweat|breath)"
            r"|heart attack",
            re.IGNORECASE,
        ),
    ),
    (
        "stroke",
        re.compile(
            r"face.{0,20}(droop|drooping|numb)"
            r"|slurred speech|can'?t speak|cannot speak|sudden.{0,20}(weakness|numbness).{0,30}(one side|arm|leg)"
            r"|stroke symptoms",
            re.IGNORECASE,
        ),
    ),
    (
        "respiratory distress",
        re.compile(
            r"can'?t breathe|cannot breathe|struggling to breathe|not breathing"
            r"|turning blue|lips are blue|choking",
            re.IGNORECASE,
        ),
    ),
    (
        "anaphylaxis",
        re.compile(
            r"anaphyla|throat.{0,20}(closing|swelling|swollen)|tongue.{0,20}swell",
            re.IGNORECASE,
        ),
    ),
    (
        "severe haemorrhage",
        re.compile(
            r"bleeding.{0,30}(won'?t stop|heavily|profusely|a lot)"
            r"|severe bleeding|haemorrhag|hemorrhag|coughing up blood|vomiting blood",
            re.IGNORECASE,
        ),
    ),
    (
        "altered consciousness",
        re.compile(
            r"unconscious|unresponsive|passed out|fainted|won'?t wake|not waking"
            r"|seizure|convulsion|convulsing",
            re.IGNORECASE,
        ),
    ),
    (
        "self-harm risk",
        re.compile(
            r"kill myself|suicid|end my life|hurt myself|harm myself|overdose[d]?|took too many",
            re.IGNORECASE,
        ),
    ),
    (
        "obstetric emergency",
        re.compile(
            r"(heavy|severe).{0,20}bleeding.{0,30}pregnan|pregnan.{0,30}(severe|heavy).{0,20}bleed"
            r"|water broke.{0,30}(blood|pain)",
            re.IGNORECASE,
        ),
    ),
]

# Self-harm bypasses the first-person-distress requirement: "how do I overdose on X" reads
# as informational to the distress regex, and the cost of a false negative here is not
# symmetric with the cost of a false positive.
_ALWAYS_ESCALATE = {"self-harm risk"}


# --- Prompt injection -------------------------------------------------------------------
#
# The RAG prompt's entire safety property is "answer only from retrieved context". Text in
# the query that redirects the model away from that instruction attacks the one guarantee
# this system makes about its medical output.

_INJECTION_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    (
        "instruction override",
        re.compile(
            r"ignore.{0,30}(previous|prior|above|earlier|all).{0,20}(instruction|prompt|rule|direction)"
            r"|disregard.{0,30}(previous|prior|above|the).{0,20}(instruction|prompt|rule|context|guideline)"
            r"|forget.{0,30}(previous|prior|above|everything|your).{0,20}(instruction|prompt|rule)",
            re.IGNORECASE,
        ),
    ),
    (
        "role reassignment",
        re.compile(
            r"you are now|from now on,? you|act as (if you|a|an)|pretend (to be|you)"
            r"|new (system )?(prompt|instruction|role)|developer mode|jailbreak",
            re.IGNORECASE,
        ),
    ),
    (
        "system prompt exfiltration",
        re.compile(
            r"(reveal|show|print|repeat|output|tell me).{0,25}(your |the )?(system|initial|original).{0,15}"
            r"(prompt|instruction|message)",
            re.IGNORECASE,
        ),
    ),
    (
        "grounding bypass",
        re.compile(
            r"(without|don'?t|do not|no need to).{0,25}(use|using|cite|citing|refer|referring).{0,25}"
            r"(the )?(context|guideline|source|document|retriev)"
            r"|answer from (your|general) knowledge"
            r"|make (it |something )?up",
            re.IGNORECASE,
        ),
    ),
]


# --- Clinical numeric verification ------------------------------------------------------
#
# A wrong dose is the highest-severity failure this system can produce, and it is also the
# most cheaply checkable: a dose the model invented will not appear anywhere in the passages
# it was given. Bare numbers are ignored (list markers, years, "1 in 4"); only numbers
# carrying a clinical unit are checked.

_CLINICAL_NUMBER = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(mg/dl|mmol/l|mg/kg|mcg/kg|mmhg|mg|mcg|µg|ug|kg|ml|iu|bpm|%|°c|units?)\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailVerdict:
    """Outcome of a guardrail stage.

    `blocked=True` means the caller must return `replacement_answer` instead of whatever it
    was going to return. `warnings` are non-blocking observations surfaced to the caller
    (and, for the numeric check, to the user) rather than silently logged.
    """

    blocked: bool = False
    replacement_answer: Optional[str] = None
    triggered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "triggered": self.triggered,
            "warnings": self.warnings,
            "reason": self.reason,
        }


def _matches(text: str, patterns: List[tuple[str, re.Pattern[str]]]) -> List[str]:
    return [label for label, pattern in patterns if pattern.search(text)]


def check_query(query: str) -> GuardrailVerdict:
    """Pre-retrieval input guardrails.

    Returns a blocking verdict for a medical emergency (escalate to emergency services
    rather than answer) or a prompt-injection attempt. Everything else passes through.
    """
    if not settings.GUARDRAILS_ENABLED or not query or not query.strip():
        return GuardrailVerdict()

    text = query.strip()

    # Emergency escalation takes precedence over every other check, including injection:
    # if someone is describing an emergency, that is the only thing that matters.
    if settings.GUARDRAIL_EMERGENCY_ESCALATION:
        emergencies = _matches(text, _EMERGENCY_PATTERNS)
        if emergencies:
            always = [e for e in emergencies if e in _ALWAYS_ESCALATE]
            if always or _FIRST_PERSON_DISTRESS.search(text):
                logger.warning("Guardrail: emergency escalation triggered (%s)", ", ".join(emergencies))
                return GuardrailVerdict(
                    blocked=True,
                    replacement_answer=EMERGENCY_MESSAGE,
                    triggered=emergencies,
                    reason="emergency_escalation",
                )
            # Symptom named without any first-person distress signal -- an informational
            # question about the condition. Answer it normally.
            logger.info(
                "Guardrail: emergency terms present but read as informational (%s)",
                ", ".join(emergencies),
            )

    if settings.GUARDRAIL_BLOCK_INJECTION:
        injections = _matches(text, _INJECTION_PATTERNS)
        if injections:
            logger.warning("Guardrail: prompt injection blocked (%s)", ", ".join(injections))
            return GuardrailVerdict(
                blocked=True,
                replacement_answer=INJECTION_REFUSAL_MESSAGE,
                triggered=injections,
                reason="prompt_injection",
            )

    return GuardrailVerdict()


def unsupported_clinical_numbers(answer: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """Clinical figures asserted in `answer` that appear in none of the retrieved passages.

    Compares on the numeric token rather than the full "130 mmHg" string, so a context
    reading "systolic blood pressure of 130 mmHg or higher" supports an answer saying
    "≥130 mmHg" -- the unit is often reformatted by the model even when the value is
    faithfully carried over.
    """
    context = " ".join(
        str(c.get("full_text") or c.get("text") or "") for c in chunks
    )
    context_numbers = {n.replace(",", ".") for n, _ in _CLINICAL_NUMBER.findall(context)}
    # Also accept any bare occurrence of the digits in context, covering values written
    # there without a unit (e.g. a table cell, or "130/80").
    bare_context_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", context))
    bare_context_numbers = {n.replace(",", ".") for n in bare_context_numbers}

    unsupported = []
    for value, unit in _CLINICAL_NUMBER.findall(answer):
        norm = value.replace(",", ".")
        if norm not in context_numbers and norm not in bare_context_numbers:
            unsupported.append(f"{value} {unit}")
    return unsupported


def check_answer(answer: str, chunks: List[Dict[str, Any]]) -> GuardrailVerdict:
    """Post-generation output guardrails.

    Blocks an answer generated with zero retrieved context that nonetheless asserts
    something -- the model answering from parametric memory despite the grounding
    instruction, which is precisely the failure the instruction exists to prevent and
    cannot itself detect.
    """
    if not settings.GUARDRAILS_ENABLED:
        return GuardrailVerdict()

    verdict = GuardrailVerdict()
    text = (answer or "").strip()

    if settings.GUARDRAIL_BLOCK_UNGROUNDED and not chunks:
        # With no context, the only acceptable answer is the refusal. Accept it in any
        # language/phrasing by looking for the instructed sentence; anything else is the
        # model substituting its own knowledge.
        looks_like_refusal = (
            "do not contain sufficient information" in text.lower()
            or "does not contain sufficient information" in text.lower()
            or "لا تحتوي" in text
        )
        if text and not looks_like_refusal:
            logger.warning(
                "Guardrail: blocked ungrounded answer (%d chars generated with 0 retrieved chunks)",
                len(text),
            )
            return GuardrailVerdict(
                blocked=True,
                replacement_answer=INSUFFICIENT_CONTEXT_MESSAGE,
                triggered=["ungrounded_answer"],
                reason="no_retrieved_context",
            )

    if settings.GUARDRAIL_NUMERIC_CHECK and chunks and text:
        unsupported = unsupported_clinical_numbers(text, chunks)
        if unsupported:
            logger.warning(
                "Guardrail: %d clinical value(s) not found in retrieved context: %s",
                len(unsupported),
                ", ".join(unsupported),
            )
            verdict.triggered.append("unverified_clinical_values")
            verdict.warnings.append(
                "Some clinical values in this answer could not be verified against the "
                "retrieved guideline passages: " + ", ".join(unsupported) +
                ". Confirm against the cited source before acting on them."
            )

    return verdict


def enforce_disclaimer(answer: str) -> str:
    """Guarantee the medical disclaimer is present.

    The prompt asks for it, but prompt compliance is probabilistic and this is the one
    element that must never be missing from clinical output.
    """
    if not settings.GUARDRAILS_ENABLED or not answer:
        return answer
    lowered = answer.lower()
    if "does not replace" in lowered or "لا يغني عن" in answer:
        return answer
    return f"{answer.rstrip()}\n\n---\n*{MEDICAL_DISCLAIMER}*"
