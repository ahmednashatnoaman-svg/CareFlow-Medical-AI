# Software Requirements Specification (SRS)

# CareFlow Module 1

# Intelligent Voice-Based Medical History Collection Service

**Version:** 2.0

**Project:** CareFlow

**Purpose:** Implementation Specification

---

# 1. Introduction

## 1.1 Purpose

This service is responsible for collecting a complete and structured medical history from a patient through an intelligent voice conversation.

Unlike conventional symptom checkers that rely on predefined decision trees, this system performs an adaptive interview where every follow-up question is generated dynamically based on:

* Previous patient responses
* Conversation memory
* Trusted medical guidelines
* Retrieved clinical evidence
* Current interview completeness

The service should produce a structured medical history suitable for physician review and downstream AI modules.

---

# 2. High-Level Goals

The system must:

* Support natural voice conversations.
* Support Egyptian Arabic patients.
* Perform all medical reasoning in English.
* Ground every generated question in trusted medical evidence.
* Avoid fixed questionnaires.
* Avoid hallucinated medical advice.
* Produce structured medical history.
* Decide autonomously when sufficient information has been collected.

---

# 3. Overall Architecture

The internal language of the system is **English**.

Arabic is only used at the system boundaries.

```text
                    Patient
             (Egyptian Arabic)
                     │
                     ▼
              Speech Service
          (Speech-to-Text ASR)
                     │
                     ▼
            Arabic Transcript
                     │
                     ▼
          Translation Service
        Arabic → English (LLM)
                     │
                     ▼
       Canonical English Transcript
                     │
                     ▼
         Conversation Manager
                     │
                     ▼
      Medical Entity Extraction
                     │
                     ▼
      Hierarchical Retrieval Engine
                     │
                     ▼
        Cross Encoder Reranker
                     │
                     ▼
          LLM Orchestrator
                     │
                     ▼
 Interview Termination Engine
         │                    │
         │ Continue           │ Stop
         ▼                    ▼
Translation LLM         Structured History
English → Arabic
         │
         ▼
Patient
```

---

# 4. System Modules

The implementation must be divided into independent services.

## Module 1 — Speech Service

Responsibilities

* Audio Batching
* ASR
* Timestamp Generation
* Transcript Normalization

Input

Audio stream

Output

Arabic transcript

Recommended Models

* Cohere Transcribe Arabic ASR

---

## Module 2 — Translation Service

Purpose

Maintain English as the canonical reasoning language.

Pipeline

Arabic

↓

English

↓

Reasoning

↓

English

↓

Arabic

Responsibilities

* Translate Egyptian Arabic to clinical English.
* Normalize colloquial expressions.
* Preserve medical terminology.
* Translate generated questions back into fluent Egyptian Arabic.
* Never simplify clinical meaning.

Examples

Input

"حاسس بوجع جامد في صدري"

Output

"I have severe chest pain."

---

## Module 3 — Conversation Manager

Maintains conversation state.

Responsibilities

Store

* Chief complaint
* Symptoms
* Timeline
* Previous questions
* Previous answers
* Risk factors
* Missing information
* Retrieved evidence
* Confidence metrics

Must expose

```
get_state()

update_state()

summarize_state()
```

---

## Module 4 — Medical Entity Extraction

Extract structured concepts.

Example

Input

"I've had severe chest pain for two hours."

Output

```
Symptom:
Chest Pain

Severity:
Severe

Duration:
2 hours
```

Entities

* Symptoms
* Diseases
* Medications
* Procedures
* Body locations
* Risk factors

---

## Module 5 — Knowledge Ingestion Pipeline

Responsible for building the knowledge base.

Pipeline

Medical PDFs

↓

OCR

↓

Cleaning

↓

Section Detection

↓

Chunking

↓

Overview Generation

↓

Embedding

↓

Vector Database

Knowledge Sources

* WHO
* NICE Guidelines

---

## Module 6 — Retrieval Engine

The system contains approximately

3000+

medical documents.


Chunk Collection

Each document is split into chunks.

Chunk size

1200 tokens

Overlap

200 tokens

Metadata

```
doc_id

section

subsection

page

chunk_id

source

publication_date
```

Collection

```
medical_chunks
```

---

## Retrieval Pipeline

```
Patient Query

↓

Embedding

↓

Hyprid Search medical_documents

↓

Metadata Filter

↓

Search medical_chunks

↓

Top K Chunks

↓

Merge Results

↓

Cross Encoder

↓

Top N Chunks

↓

LLM
```

Recommended

K = 20

Final Context = 10 chunks

---

## Module 7 — Cross Encoder Reranker

Purpose

Improve retrieval precision.

Recommended

BAAI bge-m3

Input

Retrieved chunks

Output

Sorted chunks

---

## Module 8 — LLM Orchestrator

The orchestrator is responsible for generating exactly one follow-up question.

Prompt inputs

* Conversation history
* Current structured history
* Retrieved guideline chunks
* Previous questions
* Missing information
* Confidence metrics

Prompt rules

The model MUST

* Ask only one question.
* Never repeat questions.
* Base reasoning only on retrieved evidence.
* Never invent medical facts.
* Keep questions concise.
* Maintain conversational flow.

Output

```
Question

Updated Summary

Updated State
```

---

## Module 9 — Interview Termination Engine

Purpose

Determine whether additional questions are clinically valuable.

The interview must NOT stop based on an arbitrary LLM confidence score.

Instead compute an Interview Score using measurable metrics.

---

### Coverage Score

Measures collected symptom attributes.

Example

Chest pain requires

* onset
* duration
* location
* severity
* radiation
* associated symptoms

Coverage

Collected

8

Required

9

Coverage

8 / 9

---

### Guideline Coverage

Measures how many evidence-based questions from retrieved guidelines have been addressed.

Example

NICE recommends

12 history questions.

Asked

11

Coverage

11 / 12

---

### Red Flag Coverage

Every mandatory red flag must be evaluated.

Example

Chest Pain

Required

* Radiation
* Syncope
* Dyspnea
* Sweating
* Risk factors

If any mandatory red flag is missing,

interview cannot terminate.

---

### Consistency Score

Detect contradictions.

Example

Patient says

Pain started yesterday.

Later

Pain started two weeks ago.

Consistency decreases.

---

### Expected Information Gain

Estimate whether another question would significantly reduce uncertainty.

Prompt

```
Would asking another question
meaningfully improve the medical history?

Return

ExpectedInformationGain

Reason
```

---

### Interview Score

```
InterviewScore

=

0.40 × Coverage

+

0.25 × GuidelineCoverage

+

0.15 × Consistency

+

0.10 × RedFlagCoverage

+

0.10 × InformationGain
```

Weights must remain configurable.

---

### Interview Stop Policy

Interview terminates only if

Minimum Questions

≥ 5

AND

Coverage

≥ 90%

AND

Guideline Coverage

≥ 90%

AND

All Red Flags Covered

AND

Consistency Acceptable

AND

Expected Information Gain

Below threshold

AND

Interview Score

Above threshold

Hard Maximum

20 questions

Emergency symptoms

Immediately alert physician while completing current history.

---

## Module 10 — Medical History Builder

Produces structured JSON.

Example

```json
{
  "chief_complaint": "",
  "history_of_present_illness": {
    "onset": "",
    "duration": "",
    "location": "",
    "severity": "",
    "character": "",
    "radiation": "",
    "associated_symptoms": []
  },
  "past_medical_history": [],
  "medications": [],
  "allergies": [],
  "family_history": [],
  "social_history": [],
  "red_flags": [],
  "interview_score": 0.93
}
```

---

## Module 11 — API Layer

REST + WebSocket

Endpoints

```
POST /conversation/start

POST /conversation/audio

POST /conversation/text

GET /conversation/state

GET /conversation/history

POST /conversation/finish
```

---

## Module 12 — Monitoring

Collect

* LLM latency
* Retrieval latency
* ASR latency
* Translation latency
* Retrieval precision
* Interview length
* Number of questions
* Tokens
* Prompt traces
* Errors

Recommended

LangSmith

OpenTelemetry

---

# Data Storage

PostgreSQL

Conversation

Patient History

Metadata

Redis

Conversation cache

Qdrant

medical_documents

medical_chunks

---

# Recommended Technology Stack

Backend

Python

FastAPI

Workflow

LangGraph

Embeddings

hyprid (Dense, Sparse) BGE-M3

Vector Database

Qdrant

Reranker

BAAI bge-reranker-v2-m3

Speech Recognition

Cohere Trascribe Arabic ASR

Translation

LLM-based translation service

Primary LLM

Configurable (GPT, Qwen, Llama, Gemini)

Document Parsing

Docling

Database

PostgreSQL

Cache

Redis

Monitoring

Langsmith

---

# LangGraph Workflow

The entire interview should be implemented as a LangGraph state machine.

State Flow

```
START

↓

Speech Input

↓

Speech-to-Text

↓

Arabic→English Translation

↓

Conversation Update

↓

Medical Entity Extraction

↓

Retrieve Chunks

↓

Rerank

↓

Generate Question

↓

Interview Evaluation

↓

Continue?

├── Yes → English→Arabic Translation → WAIT FOR NEXT ANSWER
│
└── No → Build Structured History → END
```

Every node should be independently testable and replaceable.

---

# Design Principles

* English is the canonical internal language.
* Arabic is only used at system boundaries.
* Every question must be grounded in retrieved clinical evidence.
* No fixed decision trees.
* No medical reasoning without retrieval.
* Interview completion must be determined by measurable clinical completeness rather than a raw LLM confidence score.
* Every module must expose a clear API, be independently deployable, and support future extension as additional CareFlow modules (EMR retrieval, radiology analysis, diagnosis support) are integrated.
* Seperate the business logic from the code
* Use design patterns to make the code maintainable
* Keep the code clean and modular
* Use SOLID principles
* Use DRY (Don't Repeat Yourself) principle