"""Module 9: Interview Termination Engine.

Evaluates measurable clinical metrics to decide whether to stop or continue the interview.
Computes Coverage, Guideline Coverage, Consistency, Red Flag Coverage, and Expected Info Gain.
Calculates Interview Score = 0.40*Coverage + 0.25*GuidelineCoverage + 0.15*Consistency + 0.10*RedFlagCoverage + 0.10*InfoGain.
Enforces configurable thresholds and mandatory red flag completion.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.constants import DEFAULT_REQUIRED_ATTRIBUTES, get_mandatory_red_flags
from app.schemas.state import EvaluationMetrics

logger = logging.getLogger(__name__)


class InterviewTerminationEngine:
    """Calculates measurable clinical completeness metrics and evaluates interview stop policy."""

    def __init__(
        self,
        min_questions: int = settings.MIN_QUESTIONS,
        max_questions: int = settings.MAX_QUESTIONS,
        min_coverage: float = settings.MIN_COVERAGE_THRESHOLD,
        min_guideline: float = settings.MIN_GUIDELINE_THRESHOLD,
        min_score: float = settings.MIN_INTERVIEW_SCORE_THRESHOLD,
        max_info_gain: float = settings.MAX_INFO_GAIN_THRESHOLD,
    ):
        self.min_questions = min_questions
        self.max_questions = max_questions
        self.min_coverage = min_coverage
        self.min_guideline = min_guideline
        self.min_score = min_score
        self.max_info_gain = max_info_gain

    def compute_coverage_score(self, extracted_attributes: Dict[str, Any], required_attributes: Optional[List[str]] = None) -> float:
        """Computes symptom attribute coverage ratio (Collected / Required).

        Args:
            extracted_attributes (Dict[str, Any]): Attributes or ExtractedEntities dictionary gathered so far.
            required_attributes (Optional[List[str]]): List of required clinical fields.

        Returns:
            float: Coverage score between 0.0 and 1.0.
        """
        required = required_attributes or DEFAULT_REQUIRED_ATTRIBUTES
        if not required:
            return 1.0

        attrs: Dict[str, Any] = {}
        if isinstance(extracted_attributes.get("attributes"), dict):
            attrs.update(extracted_attributes["attributes"])

        for k, v in extracted_attributes.items():
            if k != "attributes" and v:
                attrs[k] = v

        if extracted_attributes.get("symptoms") and "associated_symptoms" not in attrs:
            attrs["associated_symptoms"] = extracted_attributes["symptoms"]
        if (extracted_attributes.get("body_locations") or extracted_attributes.get("location")) and "location" not in attrs:
            attrs["location"] = extracted_attributes.get("body_locations") or extracted_attributes.get("location")

        SYNONYMS: Dict[str, List[str]] = {
            "aggravating_factors": ["aggravating", "aggravating_factor", "worsening_factors", "triggers"],
            "relieving_factors": ["relieving", "relieving_factor", "alleviating_factors", "mitigating_factors"],
            "associated_symptoms": ["associated", "associated_symptom", "other_symptoms", "symptoms"],
            "location": ["body_location", "body_locations", "site", "area"],
            "onset": ["time_of_onset", "onset_time", "start"],
            "severity": ["severity_level", "pain_scale", "scale"],
            "character": ["quality", "description"],
            "radiation": ["spread", "radiating"],
            "duration": ["length", "time_period"],
        }

        def _has_valid_value(key: str) -> bool:
            candidates = [key] + SYNONYMS.get(key, [])
            for cand in candidates:
                val = attrs.get(cand)
                if val not in (None, "", [], {}, ()) and not (isinstance(val, str) and val.strip().lower() in ("null", "not specified", "unspecified")):
                    return True
            return False

        collected_count = sum(1 for attr in required if _has_valid_value(attr))
        return min(round(collected_count / len(required), 4), 1.0)

    def compute_guideline_coverage_score(self, questions_asked_count: int, target_guideline_questions: int = 6) -> float:
        """Computes proportion of guideline-recommended questions addressed.

        Args:
            questions_asked_count (int): Questions asked so far.
            target_guideline_questions (int): Target guideline questions.

        Returns:
            float: Guideline coverage score between 0.0 and 1.0.
        """
        if target_guideline_questions <= 0:
            return 1.0
        return min(round(questions_asked_count / target_guideline_questions, 4), 1.0)

    _STOPWORDS = frozenset({"or", "and", "of", "to", "a", "an", "the", "/"})

    def compute_red_flag_coverage(
        self, chief_complaint: str, answered_red_flags: List[str]
    ) -> tuple[float, List[str]]:
        """Evaluates mandatory red flag coverage.

        A flag counts as covered once its topic has actually been raised in the
        conversation -- callers should pass the questions asked (which name the topic,
        e.g. "Are you experiencing any shortness of breath, dizziness, or sweating?"),
        not the patient's terse answer picks ("Fever"), which rarely restate it.

        Args:
            chief_complaint (str): Patient chief complaint.
            answered_red_flags (List[str]): Text (asked questions and/or extracted
                symptom/risk-factor labels) to search for each mandatory flag's topic.

        Returns:
            tuple[float, List[str]]: (Red flag score 0.0-1.0, list of missing mandatory red flags).
        """
        mandatory_flags = get_mandatory_red_flags(chief_complaint)

        answered_corpus = " ".join(f.lower() for f in answered_red_flags)
        unanswered = []

        for flag in mandatory_flags:
            significant_words = [
                w for w in re.split(r"[\s/]+", flag.lower()) if w and w not in self._STOPWORDS
            ]
            if not significant_words:
                continue
            matched_words = sum(
                1 for w in significant_words if re.search(rf"\b{re.escape(w)}\b", answered_corpus)
            )
            # Majority of the flag's own meaningful words must appear as whole words --
            # a single incidental overlap (e.g. "or") must never count as coverage.
            if matched_words < max(1, (len(significant_words) + 1) // 2):
                unanswered.append(flag)

        total_flags = len(mandatory_flags)
        if total_flags == 0:
            return 1.0, []

        covered_count = total_flags - len(unanswered)
        score = round(covered_count / total_flags, 4)
        return score, unanswered

    _CONTRADICTION_CUES = frozenset(
        {"no", "not", "never", "none", "nothing", "without", "isn't", "wasn't", "don't", "doesn't", "didn't"}
    )
    _CONTRADICTION_STOPWORDS = frozenset(
        {"a", "an", "the", "i", "have", "has", "had", "with", "and", "or", "of", "to", "in", "on", "my"}
    )

    def compute_consistency_score(self, conversation_history: List[Dict[str, Any]]) -> float:
        """Flags direct affirm/deny contradictions across patient turns.

        Heuristic, not a clinical NLI model: for every pair of patient turns, if one turn
        affirms a phrase (e.g. "I have a fever") and another later turn denies the same
        phrase's significant words (e.g. "no fever"), that is one contradiction. This
        catches the clearest case -- the patient reversing an earlier statement -- and
        deliberately does not attempt subtler clinical inconsistency.

        Args:
            conversation_history (List[Dict[str, Any]]): Turn history logs.

        Returns:
            float: Consistency score (1.0 = no detected contradictions, decreases per conflict).
        """
        patient_turns = [
            str(t.get("content", "")).lower()
            for t in conversation_history
            if t.get("role") in ("user", "human")
        ]
        if len(patient_turns) < 2:
            return 1.0

        def significant_words(text: str) -> set:
            return {
                w for w in re.findall(r"[a-zA-Z]+", text)
                if w not in self._CONTRADICTION_STOPWORDS and w not in self._CONTRADICTION_CUES
            }

        contradictions = 0
        for i, turn_a in enumerate(patient_turns):
            words_a = significant_words(turn_a)
            negated_a = bool(set(turn_a.split()) & self._CONTRADICTION_CUES)
            if not words_a:
                continue
            for turn_b in patient_turns[i + 1 :]:
                words_b = significant_words(turn_b)
                negated_b = bool(set(turn_b.split()) & self._CONTRADICTION_CUES)
                if negated_a == negated_b:
                    continue
                overlap = words_a & words_b
                if len(overlap) >= max(1, min(len(words_a), len(words_b)) // 2):
                    contradictions += 1

        return max(0.0, round(1.0 - 0.15 * contradictions, 4))

    def estimate_information_gain(
        self, question_count: int, coverage_score: float
    ) -> float:
        """Estimates expected information gain from asking an additional question.

        Args:
            question_count (int): Current question count.
            coverage_score (float): Current attribute coverage.

        Returns:
            float: Information gain score (0.0 = low gain, 1.0 = high gain).
        """
        if question_count >= self.max_questions:
            return 0.0
        remaining_gap = 1.0 - coverage_score
        return round(remaining_gap * (1.0 - (question_count / self.max_questions)), 4)

    def evaluate_interview(
        self,
        question_count: int,
        chief_complaint: str,
        extracted_attributes: Dict[str, Any],
        answered_red_flags: List[str],
        conversation_history: List[Dict[str, Any]],
        is_emergency_triggered: bool = False,
    ) -> EvaluationMetrics:
        """Evaluates complete interview termination metrics and stop policy.

        Args:
            question_count (int): Total questions asked.
            chief_complaint (str): Primary complaint.
            extracted_attributes (Dict[str, Any]): Attributes collected.
            answered_red_flags (List[str]): Checked red flags.
            conversation_history (List[Dict[str, Any]]): Turn history.
            is_emergency_triggered (bool): Whether emergency alert triggered.

        Returns:
            EvaluationMetrics: Comprehensive metrics and termination decision.
        """
        coverage = self.compute_coverage_score(extracted_attributes)
        guideline_cov = self.compute_guideline_coverage_score(question_count, target_guideline_questions=6)
        red_flag_score, unanswered_red_flags = self.compute_red_flag_coverage(chief_complaint, answered_red_flags)
        consistency = self.compute_consistency_score(conversation_history)
        info_gain = self.estimate_information_gain(question_count, coverage)

        # Weighted Sum Interview Score formula (SRS Section 9).
        #
        # info_gain is "how much is still learnable by asking one more question" (0.0 =
        # nothing left to learn, 1.0 = a lot). The stop policy below requires info_gain
        # to be LOW (<= MAX_INFO_GAIN_THRESHOLD) before stopping -- readiness-to-stop rises
        # as info_gain falls. The score must move the same direction, so it rewards
        # (1 - info_gain), not info_gain itself; otherwise a high-info-gain interview
        # (mid-interview, lots left to learn) would score *higher* while simultaneously
        # being barred from stopping, which is an internal contradiction.
        interview_score = round(
            settings.WEIGHT_COVERAGE * coverage
            + settings.WEIGHT_GUIDELINE * guideline_cov
            + settings.WEIGHT_CONSISTENCY * consistency
            + settings.WEIGHT_RED_FLAGS * red_flag_score
            + settings.WEIGHT_INFO_GAIN * (1.0 - info_gain),
            4,
        )

        # Hard Maximum Check
        if question_count >= self.max_questions:
            return EvaluationMetrics(
                coverage_score=coverage,
                guideline_coverage_score=guideline_cov,
                consistency_score=consistency,
                red_flag_coverage_score=red_flag_score,
                expected_info_gain=info_gain,
                overall_interview_score=interview_score,
                unanswered_red_flags=unanswered_red_flags,
                is_emergency=is_emergency_triggered,
                should_terminate=True,
                termination_reason=f"Reached hard maximum of {self.max_questions} questions.",
            )

        # Mandatory Red Flag Check
        all_red_flags_covered = len(unanswered_red_flags) == 0

        

        # Stop Policy Evaluation
        should_stop = (
            question_count >= self.min_questions
            and coverage >= self.min_coverage
            and guideline_cov >= self.min_guideline
            and all_red_flags_covered
            and consistency >= 0.90
            and info_gain <= self.max_info_gain
            and interview_score >= self.min_score
        )

        termination_reason = "Clinical interview completeness threshold satisfied." if should_stop else "Incomplete medical history, continuing interview."

        if not all_red_flags_covered and question_count >= self.min_questions:
            termination_reason = f"Mandatory red flags remaining: {', '.join(unanswered_red_flags)}"

        return EvaluationMetrics(
            coverage_score=coverage,
            guideline_coverage_score=guideline_cov,
            consistency_score=consistency,
            red_flag_coverage_score=red_flag_score,
            expected_info_gain=info_gain,
            overall_interview_score=interview_score,
            unanswered_red_flags=unanswered_red_flags,
            is_emergency=is_emergency_triggered,
            should_terminate=should_stop,
            termination_reason=termination_reason,
        )
