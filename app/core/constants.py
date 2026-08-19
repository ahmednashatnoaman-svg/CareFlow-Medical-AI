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
