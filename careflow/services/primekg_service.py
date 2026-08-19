"""PrimeKG Clinical Knowledge Graph Service.

Implements graph-based clinical reasoning, symptom exploration traversal,
evidence scoring, Shannon entropy calculation, and SOCRATES stopping evaluation.
Based on the Graph RAG architecture in CF_KG_RAG.ipynb.
"""

import json
import logging
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import pandas as pd
from careflow.core.config import settings

logger = logging.getLogger(__name__)


# Comprehensive clinical seed graph dataset for instant out-of-the-box operation
CLINICAL_GRAPH_SEED = [
    # Cardiovascular
    ("acute coronary syndrome", "chest pain", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "chest tightness", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "radiation of pain to left arm", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "radiation of pain to jaw", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "diaphoresis", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "shortness of breath", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "nausea", "disease", "effect/phenotype"),
    ("acute coronary syndrome", "dizziness", "disease", "effect/phenotype"),
    ("angina pectoris", "chest pain", "disease", "effect/phenotype"),
    ("angina pectoris", "exertional chest pain", "disease", "effect/phenotype"),
    ("angina pectoris", "shortness of breath", "disease", "effect/phenotype"),
    ("angina pectoris", "fatigue", "disease", "effect/phenotype"),
    ("heart failure", "shortness of breath", "disease", "effect/phenotype"),
    ("heart failure", "orthopnea", "disease", "effect/phenotype"),
    ("heart failure", "paroxysmal nocturnal dyspnea", "disease", "effect/phenotype"),
    ("heart failure", "peripheral edema", "disease", "effect/phenotype"),
    ("heart failure", "fatigue", "disease", "effect/phenotype"),
    ("hypertensive crisis", "headache", "disease", "effect/phenotype"),
    ("hypertensive crisis", "blurred vision", "disease", "effect/phenotype"),
    ("hypertensive crisis", "dizziness", "disease", "effect/phenotype"),
    ("hypertensive crisis", "chest pain", "disease", "effect/phenotype"),
    ("hypertensive crisis", "epistaxis", "disease", "effect/phenotype"),

    # Respiratory
    ("pneumonia", "cough", "disease", "effect/phenotype"),
    ("pneumonia", "productive cough", "disease", "effect/phenotype"),
    ("pneumonia", "fever", "disease", "effect/phenotype"),
    ("pneumonia", "chills", "disease", "effect/phenotype"),
    ("pneumonia", "pleuritic chest pain", "disease", "effect/phenotype"),
    ("pneumonia", "shortness of breath", "disease", "effect/phenotype"),
    ("pneumonia", "fatigue", "disease", "effect/phenotype"),
    ("bronchial asthma", "wheezing", "disease", "effect/phenotype"),
    ("bronchial asthma", "shortness of breath", "disease", "effect/phenotype"),
    ("bronchial asthma", "chest tightness", "disease", "effect/phenotype"),
    ("bronchial asthma", "nocturnal cough", "disease", "effect/phenotype"),
    ("bronchial asthma", "dry cough", "disease", "effect/phenotype"),
    ("pulmonary embolism", "sudden onset dyspnea", "disease", "effect/phenotype"),
    ("pulmonary embolism", "pleuritic chest pain", "disease", "effect/phenotype"),
    ("pulmonary embolism", "hemoptysis", "disease", "effect/phenotype"),
    ("pulmonary embolism", "tachycardia", "disease", "effect/phenotype"),
    ("pulmonary embolism", "unilateral leg swelling", "disease", "effect/phenotype"),
    ("chronic obstructive pulmonary disease", "chronic cough", "disease", "effect/phenotype"),
    ("chronic obstructive pulmonary disease", "sputum production", "disease", "effect/phenotype"),
    ("chronic obstructive pulmonary disease", "progressive dyspnea", "disease", "effect/phenotype"),
    ("chronic obstructive pulmonary disease", "wheezing", "disease", "effect/phenotype"),
    ("chronic obstructive pulmonary disease", "fatigue", "disease", "effect/phenotype"),

    # Gastrointestinal
    ("gastroesophageal reflux disease", "heartburn", "disease", "effect/phenotype"),
    ("gastroesophageal reflux disease", "acid regurgitation", "disease", "effect/phenotype"),
    ("gastroesophageal reflux disease", "epigastric burning pain", "disease", "effect/phenotype"),
    ("gastroesophageal reflux disease", "dysphagia", "disease", "effect/phenotype"),
    ("gastroesophageal reflux disease", "chronic cough", "disease", "effect/phenotype"),
    ("gastroesophageal reflux disease", "chest pain", "disease", "effect/phenotype"),
    ("peptic ulcer disease", "epigastric pain", "disease", "effect/phenotype"),
    ("peptic ulcer disease", "pain relieved by food or antacids", "disease", "effect/phenotype"),
    ("peptic ulcer disease", "nocturnal abdominal pain", "disease", "effect/phenotype"),
    ("peptic ulcer disease", "nausea", "disease", "effect/phenotype"),
    ("peptic ulcer disease", "bloating", "disease", "effect/phenotype"),
    ("acute appendicitis", "periumbilical abdominal pain", "disease", "effect/phenotype"),
    ("acute appendicitis", "right lower quadrant pain", "disease", "effect/phenotype"),
    ("acute appendicitis", "pain migration", "disease", "effect/phenotype"),
    ("acute appendicitis", "nausea", "disease", "effect/phenotype"),
    ("acute appendicitis", "vomiting", "disease", "effect/phenotype"),
    ("acute appendicitis", "anorexia", "disease", "effect/phenotype"),
    ("acute appendicitis", "low grade fever", "disease", "effect/phenotype"),
    ("acute cholecystitis", "right upper quadrant pain", "disease", "effect/phenotype"),
    ("acute cholecystitis", "pain radiation to right scapula", "disease", "effect/phenotype"),
    ("acute cholecystitis", "nausea", "disease", "effect/phenotype"),
    ("acute cholecystitis", "vomiting", "disease", "effect/phenotype"),
    ("acute cholecystitis", "fever", "disease", "effect/phenotype"),
    ("acute cholecystitis", "pain worse after fatty meals", "disease", "effect/phenotype"),
    ("acute gastroenteritis", "watery diarrhea", "disease", "effect/phenotype"),
    ("acute gastroenteritis", "abdominal cramps", "disease", "effect/phenotype"),
    ("acute gastroenteritis", "nausea", "disease", "effect/phenotype"),
    ("acute gastroenteritis", "vomiting", "disease", "effect/phenotype"),
    ("acute gastroenteritis", "fever", "disease", "effect/phenotype"),
    ("acute gastroenteritis", "dehydration", "disease", "effect/phenotype"),

    # Neurological
    ("migraine", "unilateral pulsating headache", "disease", "effect/phenotype"),
    ("migraine", "headache", "disease", "effect/phenotype"),
    ("migraine", "photophobia", "disease", "effect/phenotype"),
    ("migraine", "phonophobia", "disease", "effect/phenotype"),
    ("migraine", "nausea", "disease", "effect/phenotype"),
    ("migraine", "vomiting", "disease", "effect/phenotype"),
    ("migraine", "visual aura", "disease", "effect/phenotype"),
    ("tension headache", "bilateral band-like headache", "disease", "effect/phenotype"),
    ("tension headache", "headache", "disease", "effect/phenotype"),
    ("tension headache", "neck stiffness", "disease", "effect/phenotype"),
    ("tension headache", "pericranial muscle tenderness", "disease", "effect/phenotype"),
    ("ischemic stroke", "sudden facial droop", "disease", "effect/phenotype"),
    ("ischemic stroke", "unilateral arm weakness", "disease", "effect/phenotype"),
    ("ischemic stroke", "slurred speech", "disease", "effect/phenotype"),
    ("ischemic stroke", "sudden dizziness", "disease", "effect/phenotype"),
    ("ischemic stroke", "sudden severe headache", "disease", "effect/phenotype"),
    ("ischemic stroke", "ataxia", "disease", "effect/phenotype"),
    ("meningitis", "severe headache", "disease", "effect/phenotype"),
    ("meningitis", "fever", "disease", "effect/phenotype"),
    ("meningitis", "neck stiffness", "disease", "effect/phenotype"),
    ("meningitis", "photophobia", "disease", "effect/phenotype"),
    ("meningitis", "altered mental status", "disease", "effect/phenotype"),

    # Infectious / Systemic
    ("influenza", "fever", "disease", "effect/phenotype"),
    ("influenza", "myalgia", "disease", "effect/phenotype"),
    ("influenza", "fatigue", "disease", "effect/phenotype"),
    ("influenza", "dry cough", "disease", "effect/phenotype"),
    ("influenza", "sore throat", "disease", "effect/phenotype"),
    ("influenza", "headache", "disease", "effect/phenotype"),
    ("influenza", "rhinorrhea", "disease", "effect/phenotype"),
    ("covid-19", "fever", "disease", "effect/phenotype"),
    ("covid-19", "dry cough", "disease", "effect/phenotype"),
    ("covid-19", "fatigue", "disease", "effect/phenotype"),
    ("covid-19", "loss of taste or smell", "disease", "effect/phenotype"),
    ("covid-19", "sore throat", "disease", "effect/phenotype"),
    ("covid-19", "shortness of breath", "disease", "effect/phenotype"),
    ("urinary tract infection", "dysuria", "disease", "effect/phenotype"),
    ("urinary tract infection", "urinary frequency", "disease", "effect/phenotype"),
    ("urinary tract infection", "urinary urgency", "disease", "effect/phenotype"),
    ("urinary tract infection", "suprapubic pelvic pain", "disease", "effect/phenotype"),
    ("urinary tract infection", "hematuria", "disease", "effect/phenotype"),
    ("urinary tract infection", "fever", "disease", "effect/phenotype"),
    ("pyelonephritis", "flank pain", "disease", "effect/phenotype"),
    ("pyelonephritis", "high fever", "disease", "effect/phenotype"),
    ("pyelonephritis", "chills", "disease", "effect/phenotype"),
    ("pyelonephritis", "nausea", "disease", "effect/phenotype"),
    ("pyelonephritis", "dysuria", "disease", "effect/phenotype"),
]


class PrimeKGService:
    """Manages the clinical knowledge graph network for triage and diagnostic reasoning."""

    def __init__(self, graph_path: Optional[str] = None):
        self.graph_path = graph_path or settings.PRIMEKG_GRAPH_PATH
        self.graph: nx.Graph = nx.Graph()
        self._initialize_graph()

    def _initialize_graph(self):
        """Loads cached graph or builds from seed dataset."""
        os.makedirs(os.path.dirname(self.graph_path) or "data", exist_ok=True)

        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(
                    "Loaded PrimeKG clinical graph from '%s': %d nodes, %d edges",
                    self.graph_path,
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
                return
            except Exception as e:
                logger.warning("Failed to load cached graph from '%s': %s. Rebuilding from seed.", self.graph_path, e)

        # Build from seed dataset
        self._build_graph_from_tuples(CLINICAL_GRAPH_SEED)
        try:
            with open(self.graph_path, "wb") as f:
                pickle.dump(self.graph, f)
            logger.info("Saved initialized clinical knowledge graph to '%s'", self.graph_path)
        except Exception as e:
            logger.warning("Failed to persist graph to '%s': %s", self.graph_path, e)

    def _build_graph_from_tuples(self, edges: List[Tuple[str, str, str, str]]):
        """Builds NetworkX bipartite graph from (x_name, y_name, x_type, y_type) tuples."""
        self.graph = nx.Graph()
        node_types = {}

        for x_name, y_name, x_type, y_type in edges:
            x_clean = x_name.strip().lower()
            y_clean = y_name.strip().lower()
            self.graph.add_edge(x_clean, y_clean, relation="associated_with")
            node_types[x_clean] = x_type
            node_types[y_clean] = y_type

        nx.set_node_attributes(self.graph, node_types, "node_type")
        logger.info(
            "Constructed clinical graph: %d nodes (%d diseases, %d symptoms), %d edges",
            self.graph.number_of_nodes(),
            sum(1 for t in node_types.values() if t == "disease"),
            sum(1 for t in node_types.values() if t == "effect/phenotype"),
            self.graph.number_of_edges(),
        )

    def find_matching_graph_symptoms(self, query_symptom: str) -> List[str]:
        """Fuzzy and substring match for raw clinical strings to known graph symptoms."""
        query_clean = query_symptom.strip().lower()
        if not query_clean:
            return []

        if query_clean in self.graph and self.graph.nodes[query_clean].get("node_type") == "effect/phenotype":
            return [query_clean]

        matches = []
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "effect/phenotype":
                node_lower = node.lower()
                if query_clean in node_lower or node_lower in query_clean:
                    matches.append(node)

        return matches or [query_clean]

    def get_next_symptom_candidates(
        self,
        patient_symptoms: List[str],
        top_k_diseases: int = 3,
        max_candidates: int = 10,
    ) -> Dict[str, Any]:
        """Performs graph traversal to identify candidate diseases and unasked diagnostic symptoms."""
        possible_diseases: Dict[str, int] = {}
        matched_symptoms_set = set()

        # Step 1: Find all diseases connected to confirmed symptoms
        for raw_symptom in patient_symptoms:
            matched_nodes = self.find_matching_graph_symptoms(raw_symptom)
            for symptom_node in matched_nodes:
                if symptom_node in self.graph:
                    matched_symptoms_set.add(symptom_node)
                    for neighbor in self.graph.neighbors(symptom_node):
                        if self.graph.nodes[neighbor].get("node_type") == "disease":
                            possible_diseases[neighbor] = possible_diseases.get(neighbor, 0) + 1

        if not possible_diseases:
            return {
                "Top Possible Diagnoses": [],
                "Suggested Next Symptoms to Ask": [],
                "message": "No matching diseases found in knowledge graph for these symptoms."
            }

        # Step 2: Rank diseases by symptom match count
        sorted_diseases = sorted(possible_diseases.items(), key=lambda x: x[1], reverse=True)
        top_diseases = [d[0] for d in sorted_diseases[:top_k_diseases]]

        # Step 3: Find unasked symptoms for the top candidate diseases
        symptoms_to_ask = []
        seen = set()

        for disease in top_diseases:
            for neighbor in self.graph.neighbors(disease):
                if (
                    self.graph.nodes[neighbor].get("node_type") == "effect/phenotype"
                    and neighbor not in matched_symptoms_set
                    and neighbor not in seen
                ):
                    symptoms_to_ask.append(neighbor)
                    seen.add(neighbor)
                    if len(symptoms_to_ask) >= max_candidates:
                        break
            if len(symptoms_to_ask) >= max_candidates:
                break

        return {
            "Top Possible Diagnoses": top_diseases,
            "Suggested Next Symptoms to Ask": symptoms_to_ask[:max_candidates]
        }

    def calculate_diagnostic_evidence(
        self,
        patient_state: Dict[str, Any],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Calculates mathematical confidence score for diseases based on PrimeKG overlaps.
        Penalizes diseases when the patient explicitly denied associated symptoms.
        """
        raw_positives = list(patient_state.get("positive_symptoms", []))
        raw_negated = list(patient_state.get("negated_symptoms", []))

        positives_set = set()
        for s in raw_positives:
            for m in self.find_matching_graph_symptoms(s):
                positives_set.add(m)

        negated_set = set()
        for s in raw_negated:
            for m in self.find_matching_graph_symptoms(s):
                negated_set.add(m)

        disease_metrics: Dict[str, Any] = {}

        # Scan all connected diseases
        for symptom in positives_set:
            if symptom in self.graph:
                for neighbor in self.graph.neighbors(symptom):
                    if self.graph.nodes[neighbor].get("node_type") == "disease":
                        if neighbor not in disease_metrics:
                            # Full symptom profile for this disease
                            disease_profile = set(
                                n for n in self.graph.neighbors(neighbor)
                                if self.graph.nodes[n].get("node_type") == "effect/phenotype"
                            )

                            matched = positives_set.intersection(disease_profile)
                            conflicting = negated_set.intersection(disease_profile)

                            # Raw score: matched minus penalty for conflicting symptoms
                            raw_score = len(matched) - (len(conflicting) * 1.5)
                            confidence = raw_score / max(len(disease_profile), 1)

                            disease_metrics[neighbor] = {
                                "raw_confidence_score": max(round(confidence, 3), 0.001),
                                "matched_evidence": list(matched),
                                "conflicting_evidence": list(conflicting),
                                "total_profile_symptoms": len(disease_profile),
                            }

        if not disease_metrics:
            return {
                "unspecified clinical syndrome": {
                    "raw_confidence_score": 0.001,
                    "matched_evidence": list(positives_set),
                    "conflicting_evidence": list(negated_set),
                    "total_profile_symptoms": 1,
                }
            }

        # Sort by highest confidence and extract Top K
        sorted_diseases = sorted(
            disease_metrics.items(),
            key=lambda x: x[1]["raw_confidence_score"],
            reverse=True,
        )

        top_diagnoses = {}
        for disease, metrics in sorted_diseases[:top_k]:
            top_diagnoses[disease] = metrics

        return top_diagnoses

    def calculate_statistical_confidence(self, disease_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Converts raw graph scores into probability distribution and computes Shannon entropy & margin."""
        scores = [data["raw_confidence_score"] for data in disease_metrics.values()]

        if not scores or sum(scores) == 0:
            return {"entropy": 1.0, "margin": 0.0, "probabilities": []}

        total_score = sum(scores)
        probabilities = [s / total_score for s in scores]

        # Shannon Entropy (lower entropy = higher concentration/certainty)
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        # Margin separation between top 1 and top 2
        margin = (probabilities[0] - probabilities[1]) if len(probabilities) > 1 else probabilities[0]

        return {
            "entropy": round(entropy, 3),
            "margin": round(margin, 3),
            "probabilities": [round(p, 3) for p in probabilities]
        }

    def should_stop_interview(
        self,
        turn_count: int,
        stats: Dict[str, Any],
        socrates_score: int,
        red_flags_cleared: bool = True,
    ) -> Tuple[bool, str]:
        """Evaluates the 4 pillars to decide if the triage interview should terminate."""
        min_turns = settings.MIN_QUESTIONS
        max_turns = settings.MAX_QUESTIONS
        min_socrates = settings.MIN_SOCRATES_SCORE
        margin_thresh = settings.CONFIDENCE_MARGIN_THRESHOLD
        entropy_thresh = settings.MAX_ENTROPY_THRESHOLD

        # 1. Hard Bounds
        if turn_count < min_turns:
            return False, f"Minimum turns ({min_turns}) not reached yet."

        if turn_count >= max_turns:
            return True, f"Maximum turns ({max_turns}) reached to prevent patient fatigue."

        # 2. Safety Rule
        if not red_flags_cleared:
            return False, "Critical red flag symptoms still need to be ruled out."

        # 3. Diagnostic Convergence (High Margin & Low Entropy)
        if stats.get("margin", 0.0) >= margin_thresh and stats.get("entropy", 2.0) <= entropy_thresh:
            return True, f"High diagnostic certainty reached (Top separation: {stats.get('margin', 0.0):.2f})."

        # 4. Clinical History Completeness (SOCRATES)
        if socrates_score >= min_socrates and stats.get("margin", 0.0) >= 0.25:
            return True, f"Sufficient SOCRATES clinical history gathered ({socrates_score}/8 criteria met)."

        return False, "Continuing clinical investigation to gather evidence."


# Global singleton instance
primekg_service = PrimeKGService()
