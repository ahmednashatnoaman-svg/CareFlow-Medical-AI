"""Tests for triage session persistence.

Guards the serverless failure this store was introduced to fix: state written by
/triage/start must still be readable by the process that handles /triage/step.
"""

import time

import pytest

from careflow.services.session_store import InMemorySessionStore
from careflow.services.triage_service import TriageOrchestratorService, TriageSession


def test_session_round_trips_through_store():
    store = InMemorySessionStore(ttl_seconds=60, max_sessions=10)
    svc = TriageOrchestratorService(store=store)

    session = svc.get_or_create_session("abc", language="ar")
    session.positive_symptoms.add("chest pain")
    session.turn_count = 3
    session.socrates_tracker["site"] = True
    svc.save_session(session)

    # A *new* service instance reading the same store stands in for the next serverless
    # invocation: previously this returned an empty session and the interview restarted.
    reloaded = TriageOrchestratorService(store=store).get_or_create_session("abc")
    assert reloaded.positive_symptoms == {"chest pain"}
    assert reloaded.turn_count == 3
    assert reloaded.language == "ar"
    assert reloaded.socrates_tracker["site"] is True


def test_sets_survive_json_serialization():
    """positive/negated symptoms are sets in memory but must be JSON-encodable."""
    s = TriageSession("s1")
    s.positive_symptoms = {"fever", "cough"}
    s.negated_symptoms = {"rash"}
    restored = TriageSession.from_dict(s.to_dict())
    assert restored.positive_symptoms == {"fever", "cough"}
    assert restored.negated_symptoms == {"rash"}


def test_expired_session_is_evicted():
    store = InMemorySessionStore(ttl_seconds=0, max_sessions=10)
    store.set("gone", {"session_id": "gone"})
    time.sleep(0.01)
    assert store.get("gone") is None


def test_capacity_ceiling_bounds_memory():
    """Unbounded growth was the second half of the original leak."""
    store = InMemorySessionStore(ttl_seconds=300, max_sessions=3)
    for i in range(10):
        store.set(f"s{i}", {"session_id": f"s{i}"})
    assert len(store._data) <= 3


def test_reset_preserves_language_but_clears_state():
    store = InMemorySessionStore(ttl_seconds=60, max_sessions=10)
    svc = TriageOrchestratorService(store=store)
    s = svc.get_or_create_session("r1", language="ar")
    s.positive_symptoms.add("headache")
    s.turn_count = 5
    svc.save_session(s)

    reset = svc.reset_session("r1")
    assert reset.language == "ar"
    assert reset.positive_symptoms == set()
    assert reset.turn_count == 0
