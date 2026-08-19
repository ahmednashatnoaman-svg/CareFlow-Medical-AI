"""Knowledge Base Ingestion CLI Utility Script.

Ingests clinical medical guideline documents into Qdrant collections.
"""

import asyncio
import logging
from app.services.knowledge_ingestion_service import KnowledgeIngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_NICE_GUIDELINE = """
NICE Guideline CG95: Chest pain of recent onset - Assessment and diagnosis.

1. Clinical Evaluation of Acute Chest Pain:
Assess the characteristics of the pain: onset (sudden or gradual), duration, location (retrosternal, epigastric),
severity, and radiation (left shoulder, arm, neck, jaw).
Evaluate associated symptoms: breathlessness (dyspnea), sweating (diaphoresis), nausea, vomiting, dizziness, or syncope.

2. Red Flag Symptoms:
Severe chest pain with radiation to back (suspect aortic dissection), associated severe dyspnea, hypotension, or syncope.
Patients presenting with chest pain and diaphoresis require immediate emergency referral for acute coronary syndrome (ACS).

3. Risk Factors Assessment:
Elicit history of diabetes mellitus, hypertension, hyperlipidemia, tobacco smoking, past CAD, and family history of CAD.
"""


async def main():
    logger.info("Initializing Knowledge Base Ingestion...")
    service = KnowledgeIngestionService()
    await service.initialize_collections()

    res = await service.ingest_document(
        source_name="NICE Guideline CG95 - Chest Pain",
        content=SAMPLE_NICE_GUIDELINE,
        section="Chest Pain & ACS",
    )
    logger.info(f"Ingestion finished successfully: {res.get('doc_id')}, chunks: {res.get('chunks_count')}")


if __name__ == "__main__":
    asyncio.run(main())
