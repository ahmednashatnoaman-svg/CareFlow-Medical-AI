"""Unit tests for the Mayo Clinic Web Disease-Info Search Service."""

import asyncio
from app.services.web_search_service import MayoClinicSearchService

SEARCH_RESULTS_HTML = """
<html><body>
<ul class="search-results">
  <li><a href="/diseases-conditions/heart-attack/symptoms-causes/syc-20373106">Heart attack</a></li>
  <li><a href="/diseases-conditions/hypertension/symptoms-causes/syc-20373410">High blood pressure (hypertension)</a></li>
  <li><a href="/diseases-conditions/search-results?q=heart">More results</a></li>
</ul>
</body></html>
"""

DETAIL_PAGE_HTML = """
<html><body>
<article>
  <h2>Symptoms</h2>
  <p>Chest pain or discomfort.</p>
  <p>Shortness of breath.</p>
  <h2>Causes</h2>
  <p>Coronary artery disease is the usual cause.</p>
  <h2>Risk factors</h2>
  <p>Age, smoking, high blood pressure.</p>
</article>
</body></html>
"""


def test_parse_search_results_extracts_condition_links():
    service = MayoClinicSearchService()
    hits = service._parse_search_results(SEARCH_RESULTS_HTML)

    assert len(hits) == 2
    assert hits[0]["title"] == "Heart attack"
    assert hits[0]["url"] == "/diseases-conditions/heart-attack/symptoms-causes/syc-20373106"


def test_parse_detail_sections_extracts_symptoms_and_causes():
    service = MayoClinicSearchService()
    sections = service._parse_detail_sections(DETAIL_PAGE_HTML)

    assert "Chest pain" in sections["symptoms"]
    assert "Shortness of breath" in sections["symptoms"]
    assert "Coronary artery disease" in sections["causes"]


def test_search_disease_info_disabled_returns_empty_without_network():
    service = MayoClinicSearchService(enabled=False)
    results = asyncio.run(service.search_disease_info(["Hypertension"]))

    assert results == []


def test_search_disease_info_empty_input_returns_empty():
    service = MayoClinicSearchService()
    results = asyncio.run(service.search_disease_info([]))

    assert results == []
