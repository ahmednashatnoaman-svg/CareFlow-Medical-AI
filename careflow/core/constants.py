"""Application Constants module.

Contains error codes, clinical defaults, and system constants.
"""

from typing import List

# Error Codes
ERR_CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
ERR_INVALID_AUDIO = "INVALID_AUDIO"
ERR_TRANSLATION_FAILED = "TRANSLATION_FAILED"
ERR_RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
ERR_LLM_FAILED = "LLM_FAILED"
ERR_INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
ERR_VALIDATION_ERROR = "VALIDATION_ERROR"

# Conversation Statuses
STATUS_ACTIVE = "ACTIVE"
STATUS_COMPLETED = "COMPLETED"
STATUS_TERMINATED = "TERMINATED"
STATUS_EMERGENCY = "EMERGENCY"

# Default Required Attributes per Symptom
DEFAULT_REQUIRED_ATTRIBUTES = [
    "onset",
    "duration",
    "location",
    "severity",
    "character",
    "radiation",
    "associated_symptoms",
    "aggravating_factors",
    "relieving_factors",
]

# Default Red Flag Requirements by Complaint Type
DEFAULT_RED_FLAGS_MAP = {
    "chest pain": [
        "radiation to jaw/arm",
        "syncope or dizziness",
        "dyspnea / shortness of breath",
        "diaphoresis / profuse sweating",
        "cardiovascular risk factors",
    ],
    "headache": [
        "thunderclap onset",
        "fever or stiff neck",
        "focal neurological deficit",
        "worse with valsalva",
    ],
    "abdominal pain": [
        "fever or chills",
        "persistent vomiting",
        "gastrointestinal bleeding / melena",
        "peritoneal signs / rigidity",
    ],
    "default": [
        "unexplained weight loss",
        "high fever",
        "loss of consciousness",
    ],
}


def get_mandatory_red_flags(chief_complaint: str) -> List[str]:
    """Returns the mandatory red-flag checklist for a chief complaint.

    Single source of truth for the "which flags apply to this complaint" lookup --
    shared by the Termination Engine (to score coverage) and the History Builder
    (to report which flags were actually addressed), so the two can never drift apart.
    """
    cc_lower = chief_complaint.lower()
    for key, flags in DEFAULT_RED_FLAGS_MAP.items():
        if key != "default" and key in cc_lower:
            return flags
    return DEFAULT_RED_FLAGS_MAP["default"]


# ---------------------------------------------------------------------------
# LLM sampling profiles
#
# Temperature and length are properties of the *task*, not of the deployment, so they
# live here as named profiles rather than in Settings. Naming them also documents intent:
# EXTRACTION is deterministic because entity extraction must be reproducible, while
# CLINICAL_REPORT allows a little variance for readable prose.
#
# Previously these were bare literals repeated across six services, where
# `temperature=0.0` in one file and `temperature=0.2` in another carried no explanation.
# ---------------------------------------------------------------------------

class LLMProfile:
    """A (temperature, max_tokens) pair for one class of LLM call."""

    __slots__ = ("temperature", "max_tokens")

    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens


# Structured extraction and option matching: must be reproducible run to run.
LLM_EXTRACTION = LLMProfile(temperature=0.0, max_tokens=300)

# Translation: near-deterministic, but a little slack for natural phrasing.
LLM_TRANSLATION = LLMProfile(temperature=0.1, max_tokens=200)

# Translating a structured JSON payload keeps the schema, so it needs more room.
LLM_TRANSLATION_JSON = LLMProfile(temperature=0.2, max_tokens=1000)

# Interview question generation: conversational, slight variation is desirable.
LLM_INTERVIEW = LLMProfile(temperature=0.2, max_tokens=800)

# Diagnostic report synthesis: low variance, long output.
LLM_CLINICAL_REPORT = LLMProfile(temperature=0.1, max_tokens=1000)

# Grounded guideline answers: low variance so claims track the retrieved context.
LLM_GROUNDED_ANSWER = LLMProfile(temperature=0.2, max_tokens=1500)

# Truncation used only in log lines, to keep records readable.
LOG_QUERY_PREVIEW_CHARS = 80
LOG_PAYLOAD_PREVIEW_CHARS = 200
