"""Module 10: Medical History Builder Service.

Transforms collected conversation state and extracted entities into final structured JSON
conforming to CareFlow medical history standards.
"""

import logging
from typing import Any, Dict
from app.core.constants import get_mandatory_red_flags
from app.schemas.history import HistoryOfPresentIllness, StructuredMedicalHistory

logger = logging.getLogger(__name__)


class MedicalHistoryBuilder:
    """Builds final validated structured medical history object."""

    def build_structured_history(
        self,
        conversation_state: Dict[str, Any],
        overall_interview_score: float = 0.0,
    ) -> StructuredMedicalHistory:
        """Constructs StructuredMedicalHistory from state dictionary.

        Args:
            conversation_state (Dict[str, Any]): Active conversation state.
            overall_interview_score (float): Final calculated interview score.

        Returns:
            StructuredMedicalHistory: Validated structured history object.
        """
        logger.info("Building final structured medical history")

        entities = conversation_state.get("extracted_entities", {})
        attributes = entities.get("attributes", {})
        symptoms = entities.get("symptoms", [])

        def _to_str(val: Any) -> str:
            if isinstance(val, list):
                return ", ".join(str(x) for x in val if x)
            return str(val) if val is not None else ""

        def _to_list(val: Any) -> list:
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val.strip():
                return [val.strip()]
            return []

        hpi = HistoryOfPresentIllness(
            onset=_to_str(attributes.get("onset")),
            duration=_to_str(attributes.get("duration")),
            location=_to_str(attributes.get("location")) or ", ".join(entities.get("body_locations", [])),
            severity=_to_str(attributes.get("severity")),
            character=_to_str(attributes.get("character")),
            radiation=_to_str(attributes.get("radiation")),
            associated_symptoms=_to_list(symptoms),
            aggravating_factors=_to_list(attributes.get("aggravating_factors")),
            relieving_factors=_to_list(attributes.get("relieving_factors")),
        )

        # "red_flags" reports which mandatory flags for this complaint were actually
        # addressed during the interview -- derived from the Termination Engine's own
        # unanswered-flags list (evaluation_metrics), never a state key nothing else writes.
        chief_complaint = conversation_state.get("chief_complaint") or ""
        mandatory_flags = get_mandatory_red_flags(chief_complaint)
        unanswered_flags = set(
            conversation_state.get("evaluation_metrics", {}).get("unanswered_red_flags", [])
        )
        addressed_flags = [flag for flag in mandatory_flags if flag not in unanswered_flags]

        history = StructuredMedicalHistory(
            chief_complaint=chief_complaint or "Not specified",
            history_of_present_illness=hpi,
            past_medical_history=entities.get("diseases", []),
            medications=entities.get("medications", []),
            allergies=entities.get("allergies", []),
            family_history=entities.get("family_history", []),
            social_history=entities.get("social_history", []),
            risk_factors=entities.get("risk_factors", []),
            red_flags=addressed_flags,
            interview_score=overall_interview_score,
        )

        return history
