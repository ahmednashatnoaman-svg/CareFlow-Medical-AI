"""Unit tests for Entity Extraction Service (Module 4)."""

import asyncio
from careflow.services.entity_extraction_service import EntityExtractionService


def test_extract_entities_chest_pain():
    service = EntityExtractionService()
    text = "I have severe chest pain radiating to left shoulder for 2 hours. Pain is 8 out of 10."
    entities = asyncio.run(service.extract_entities(text))

    assert any("chest pain" in s.lower() for s in entities.symptoms)
    assert any("chest" in b.lower() for b in entities.body_locations)
    assert "2 hours" in str(entities.attributes.get("duration"))
