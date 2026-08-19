
import sys
from pathlib import Path

# Scripts are run directly (`python scripts/x.py`), so the repo root is not on sys.path
# by default and `import careflow` would fail. Four of the seven scripts lacked this and
# crashed on import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asyncio
import logging
from careflow.services.knowledge_ingestion_service import KnowledgeIngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEW_GUIDELINES = [
    {
        "source_name": "WHO Guideline - Hypertension 2021",
        "section": "Pharmacological Treatment",
        "content": """
World Health Organization (WHO) Guideline for the pharmacological treatment of hypertension in adults (2021).

1. Blood Pressure Thresholds:
For adults with confirmed hypertension and cardiovascular disease (CVD), WHO recommends initiating pharmacological antihypertensive treatment at systolic blood pressure (SBP) ≥130 mmHg.
For individuals without CVD but with high cardiovascular risk, diabetes mellitus, or chronic kidney disease (CKD), treatment should also begin at SBP ≥130 mmHg.

2. First-line Medications:
WHO recommends using any of the following three classes of pharmacological antihypertensive medications as an initial treatment:
- Thiazide and thiazide-like agents
- Angiotensin-converting enzyme (ACE) inhibitors / Angiotensin-receptor blockers (ARBs)
- Long-acting dihydropyridine calcium channel blockers (CCBs)

3. Combination Therapy:
Combination therapy, preferably as a single-pill combination (SPC) to improve adherence, is recommended for initial treatment or if BP is not controlled with monotherapy.
"""
    },
    {
        "source_name": "CDC Guideline - Diabetes Screening 2023",
        "section": "Screening and Diagnosis",
        "content": """
Centers for Disease Control and Prevention (CDC) - National Diabetes Prevention Program Guidelines.

1. Screening Criteria:
Screening for prediabetes and type 2 diabetes should be considered in adults aged 35 to 70 years who are overweight or obese.
If results are normal, testing should be repeated at a minimum of 3-year intervals.

2. High-Risk Individuals:
Individuals with a family history of diabetes, history of gestational diabetes, or belonging to high-risk racial/ethnic groups should be screened regardless of age or BMI.

3. Diagnostic Thresholds:
- Fasting Plasma Glucose (FPG) ≥ 126 mg/dL (7.0 mmol/L)
- 2-hour PG ≥ 200 mg/dL (11.1 mmol/L) during OGTT
- A1C ≥ 6.5% (48 mmol/mol)
"""
    },
    {
        "source_name": "USPSTF Guideline - Asthma Management",
        "section": "Asthma Control and Exacerbations",
        "content": """
U.S. Preventive Services Task Force (USPSTF) related clinical recommendations for Asthma.

1. Initial Assessment:
Asthma control should be assessed using standardized questionnaires (e.g., Asthma Control Test - ACT).
Poor control is characterized by frequent daytime symptoms, nighttime awakenings, and limitation of activities.

2. Maintenance Therapy:
Inhaled corticosteroids (ICS) remain the cornerstone of persistent asthma management. 
For moderate-to-severe persistent asthma, the addition of a long-acting beta-agonist (LABA) is recommended.

3. Acute Exacerbations:
Patients presenting with severe acute exacerbations (PEF < 50% predicted, severe dyspnea) should receive immediate supplemental oxygen, high-dose short-acting beta-agonists (SABA), and systemic corticosteroids.
"""
    }
]

async def main():
    logger.info("Initializing Extra Knowledge Base Ingestion...")
    service = KnowledgeIngestionService()
    
    # We do NOT initialize collections here to avoid wiping out the existing ones, 
    # assuming initialize_collections() might reset it. 
    # Let's just insert directly.
    
    for g in NEW_GUIDELINES:
        logger.info(f"Ingesting: {g['source_name']}")
        res = await service.ingest_document(
            source_name=g["source_name"],
            content=g["content"],
            section=g["section"],
        )
        logger.info(f"Ingested {g['source_name']}: chunks={res.get('chunks_count')}")

if __name__ == "__main__":
    asyncio.run(main())
