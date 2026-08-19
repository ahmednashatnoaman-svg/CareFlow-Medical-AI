"""Comprehensive Automated Test Suite for Dual-Mode Medical Chatbot.

Tests:
1. Mode 1: PrimeKG Clinical Knowledge Graph reasoning, traversal, scoring, entropy, and SOCRATES stopping.
2. Mode 1 API: Triage session start, multi-turn step, and doctor report generation.
3. Mode 2: WHO Guidelines Dialogue Vector RAG with Qdrant retrieval.
4. Mode 2 API: Dialogue chat endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from careflow.main import app
from careflow.services.primekg_service import primekg_service
from careflow.services.triage_service import triage_service
from careflow.services.dialogue_service import dialogue_service


@pytest.fixture
def client():
    return TestClient(app)


# --- TEST SUITE 1: PRIME-KG GRAPH RAG ENGINE ---

def test_primekg_graph_structure():
    """Verifies that the knowledge graph is loaded and contains disease and phenotype nodes."""
    assert primekg_service.graph.number_of_nodes() > 20
    assert primekg_service.graph.number_of_edges() > 20

    # Verify node types
    diseases = [n for n, d in primekg_service.graph.nodes(data=True) if d.get("node_type") == "disease"]
    phenotypes = [n for n, d in primekg_service.graph.nodes(data=True) if d.get("node_type") == "effect/phenotype"]

    assert len(diseases) > 5
    assert len(phenotypes) > 10
    assert "acute coronary syndrome" in diseases
    assert "chest pain" in phenotypes


def test_primekg_candidate_traversal():
    """Verifies graph traversal and candidate discovery for reported symptoms."""
    candidates = primekg_service.get_next_symptom_candidates(["chest pain", "shortness of breath"], top_k_diseases=3)

    assert "Top Possible Diagnoses" in candidates
    assert "Suggested Next Symptoms to Ask" in candidates
    assert len(candidates["Top Possible Diagnoses"]) > 0
    assert "acute coronary syndrome" in candidates["Top Possible Diagnoses"]
    assert len(candidates["Suggested Next Symptoms to Ask"]) > 0


def test_primekg_evidence_and_entropy():
    """Verifies diagnostic evidence scoring and Shannon entropy calculation."""
    patient_state = {
        "positive_symptoms": ["chest pain", "diaphoresis", "radiation of pain to left arm"],
        "negated_symptoms": ["fever"],
    }
    evidence = primekg_service.calculate_diagnostic_evidence(patient_state, top_k=3)
    assert len(evidence) > 0
    assert "acute coronary syndrome" in evidence

    acs_evidence = evidence["acute coronary syndrome"]
    assert "raw_confidence_score" in acs_evidence
    assert acs_evidence["raw_confidence_score"] > 0
    assert "chest pain" in acs_evidence["matched_evidence"]

    stats = primekg_service.calculate_statistical_confidence(evidence)
    assert "entropy" in stats
    assert "margin" in stats
    assert len(stats["probabilities"]) > 0
    assert 0.0 <= stats["entropy"] <= 3.0


def test_primekg_stopping_criteria():
    """Verifies termination evaluation bounds."""
    # Under minimum turns (turn 1) -> should not stop
    stats = {"entropy": 0.5, "margin": 0.6, "probabilities": [0.8, 0.2]}
    stop, reason = primekg_service.should_stop_interview(turn_count=1, stats=stats, socrates_score=6)
    assert stop is False

    # Max turns (turn 8) -> should stop
    stop, reason = primekg_service.should_stop_interview(turn_count=8, stats=stats, socrates_score=3)
    assert stop is True
    assert "Maximum turns" in reason

    # High margin + low entropy after turn 3 -> should stop
    stop, reason = primekg_service.should_stop_interview(turn_count=4, stats=stats, socrates_score=4)
    assert stop is True
    assert "High diagnostic certainty" in reason


# --- TEST SUITE 2: FASTAPI ENDPOINTS ---

def test_api_health(client):
    """Verifies root health endpoint."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "mode_1_triage_graph_rag" in data["modes"]
    assert "mode_2_who_dialogue_vector_rag" in data["modes"]


def test_triage_api_start_and_step(client):
    """Verifies Mode 1 API flow: start session and step."""
    # 1. Start session
    start_res = client.post("/api/v1/triage/start", json={"language": "en"})
    assert start_res.status_code == 200
    start_data = start_res.json()
    session_id = start_data["session_id"]
    assert session_id is not None
    assert "symptoms" in start_data["message"].lower()

    # 2. Step 1: user reports headache
    step_res = client.post(
        "/api/v1/triage/step",
        json={"session_id": session_id, "message": "I have had a severe throbbing headache and nausea since morning"},
    )
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["session_id"] == session_id
    assert len(step_data["options"]) > 0
    assert step_data["turn_count"] == 1
    assert "headache" in step_data["positive_symptoms"] or "nausea" in step_data["positive_symptoms"]


def test_dialogue_api_chat(client):
    """Verifies Mode 2 API chat with WHO guidelines."""
    res = client.post(
        "/api/v1/dialogue/chat",
        json={"query": "What are WHO recommendations for poliovirus containment?", "top_k": 3},
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 20
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "source_file" in data["sources"][0]
