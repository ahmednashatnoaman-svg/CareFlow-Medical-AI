"""Web Disease-Info Search Service.

Looks up newly-identified diseases/conditions on Mayo Clinic's public search to provide
supplementary background context (symptoms, causes) for follow-up question generation.
This is purely additive to the existing guideline RAG (Modules 6/7) -- it never replaces
or filters retrieved guideline evidence, and degrades to an empty result on any failure.
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings

logger = logging.getLogger(__name__)

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class MayoClinicSearchService:
    """Fetches supplementary disease background from mayoclinic.org search results."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: Optional[float] = None,
        max_diseases: Optional[int] = None,
    ):
        self.base_url = base_url or settings.MAYO_CLINIC_SEARCH_URL
        self.enabled = settings.WEB_SEARCH_ENABLED if enabled is None else enabled
        self.timeout = timeout or settings.WEB_SEARCH_TIMEOUT
        self.max_diseases = max_diseases or settings.WEB_SEARCH_MAX_DISEASES_PER_TURN

    async def search_disease_info(self, disease_names: List[str]) -> List[Dict[str, Any]]:
        """Looks up background info for the given disease names.

        Args:
            disease_names (List[str]): Disease/condition names to look up.

        Returns:
            List[Dict[str, Any]]: One entry per successfully resolved disease, each with
                'disease_query', 'title', 'url', 'symptoms', and 'causes'. Diseases that
                fail to resolve (network error, no results, blocked request) are silently
                skipped rather than failing the whole lookup.
        """
        if not self.enabled or not disease_names:
            return []

        results: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=self.timeout, headers=_REQUEST_HEADERS) as client:
            for name in disease_names[: self.max_diseases]:
                try:
                    entry = await self._lookup_single_disease(client, name)
                    if entry:
                        results.append(entry)
                except Exception:
                    logger.warning(
                        "Mayo Clinic lookup failed for disease '%s', skipping", name, exc_info=True
                    )

        return results

    async def _lookup_single_disease(
        self, client: httpx.AsyncClient, disease_name: str
    ) -> Optional[Dict[str, Any]]:
        """Fetches search results then the top matching detail page for one disease name."""
        search_response = await client.get(self.base_url, params={"q": disease_name})
        search_response.raise_for_status()
        search_hits = self._parse_search_results(search_response.text)

        if not search_hits:
            logger.info("No Mayo Clinic search results found for '%s'", disease_name)
            return None

        top_hit = search_hits[0]
        detail_url = urljoin(self.base_url, top_hit["url"])

        detail_response = await client.get(detail_url)
        detail_response.raise_for_status()
        sections = self._parse_detail_sections(detail_response.text)

        return {
            "disease_query": disease_name,
            "title": top_hit["title"],
            "url": detail_url,
            "symptoms": sections.get("symptoms", ""),
            "causes": sections.get("causes", ""),
        }

    def _parse_search_results(self, html: str) -> List[Dict[str, str]]:
        """Extracts {title, url} entries from a Mayo Clinic search-results page.

        Args:
            html (str): Raw HTML of the search-results page.

        Returns:
            List[Dict[str, str]]: Ordered list of matched result links.
        """
        soup = BeautifulSoup(html, "html.parser")
        hits: List[Dict[str, str]] = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/diseases-conditions/" not in href or "search-results" in href:
                continue
            title = link.get_text(strip=True)
            if not title:
                continue
            hits.append({"title": title, "url": href})

        return hits

    def _parse_detail_sections(self, html: str) -> Dict[str, str]:
        """Extracts 'Symptoms' and 'Causes' section text from a disease detail page.

        Args:
            html (str): Raw HTML of the disease detail page.

        Returns:
            Dict[str, str]: Keys 'symptoms' and 'causes' mapped to concatenated section text
                (empty string if a section isn't found).
        """
        soup = BeautifulSoup(html, "html.parser")
        sections: Dict[str, str] = {"symptoms": "", "causes": ""}

        for heading in soup.find_all(["h2", "h3"]):
            heading_text = heading.get_text(strip=True).lower()
            key = next((k for k in sections if k in heading_text), None)
            if not key:
                continue

            paragraphs: List[str] = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ("h2", "h3"):
                    break
                text = sibling.get_text(" ", strip=True)
                if text:
                    paragraphs.append(text)

            sections[key] = " ".join(paragraphs).strip()

        return sections
