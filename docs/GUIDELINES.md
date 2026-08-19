# CareFlow Development Guidelines

Team Development Standards & Best Practices

## Table of Contents

```
1. Organization Strategy
2. Standard Microservice Folder Structure
3. AI Microservice Special Structure
4. Naming Conventions
5. API Design Standards
6. Database Standards
7. Documentation Standards
8. Git Workflow
9. Scrum + Jira Guidelines
10. Testing Standards
11. Logging Standards
12. Security Standards
13. Docker Standards
14. CI/CD Standards
15. AI Microservice Standards
```
# 1 ) Organization Strategy

```
careflow-mans-org/
```
```
├── api-gateway/
```
```
├── shared-libraries/
```
```
├── infrastructure/
```
```
├── docs/
```
```
└── *frontend-sevices/
```
```
└── *backend-sevices/
```
```
└── *mobile-services/
```
```
└── *ai-sevices/
```

# 2 ) Standard Microservice Folder Structure

Example for FastAPI service:

```
service-name/
│
├── app/
│ ├── api/ # API routes
│ │ ├── v1/
│ │ │ ├── endpoints/
│ │ │ │ └── *.py
│ │ │ └── router.py
│
│ ├── core/ # configs, security, constants
│ │ ├── config.py
│ │ ├── security.py
│ │ └── constants.py
│
│ ├── models/ # DB models (ORM)
│ │ ├── patient.py
│ │ ├── doctor.py
│
│ ├── schemas/ # request/response schemas
│ │ ├── patient.py
│ │ ├── doctor.py
│
│ ├── services/ # business logic
│ │ ├── patient_service.py
│ │ ├── insurance_service.py
│
│ ├── repositories/ # DB access layer
│ │ ├── patient_repository.py
│
│ ├── dependencies/ # dependency injection
│ │ └── *.py
│
│ ├── middleware/ # auth, logging middleware
│ │ └── *.py
│
│ ├── utils/ # helper functions
│ │ └── *.py
│
│ ├── tests/ # unit/integration tests
│ │ ├── unit/
│ │ └── integration/
│
│ └── main.py # app entry point
│
├── migrations/ # DB migrations (alembic)
│
├── scripts/ # utility scripts NOT part of the API
```

```
│ ├── seed_database.py
│ ├── create_admin.py
│ ├── migrate_data.py
│
├── Dockerfile # build this service
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
# 3 ) AI Microservice Special Structure

Example: medical-nlp-service

```
medical-nlp-service/
│
├── app/
│ ├── api/
│
│ ├── models/ # ML models
│ │ ├── ner/
│ │ │ ├── model.py
│ │ │ └── inference.py
│ │ │
│ │ ├── classifier/
│ │ │ ├── model.py
│ │ │ └── inference.py
│
│ ├── pipelines/ # combine models
│ │ └── medical_pipeline.py
│
│ ├── preprocessing/
│ ├── postprocessing/
│
│ ├── services/
│ │ └── nlp_service.py
│
│ ├── core/ # configs, security, constants
│ │ ├── config.py
│ │ ├── security.py
│ │ └── constants.py
│
│ └── main.py
│
├── model_store/ # local dev models only
│
├── Dockerfile
├── requirements.txt
├── .env.example
```

```
├── .gitignore
└── README.md
```
# 4 ) Naming Conventions

Follow PEP 8

## Files

Use snake_case:

```
patient_service.py
medical_record.py
auth_utils.py
```
## Classes

Use PascalCase:

```
PatientService
MedicalRecordModel
AuthMiddleware
```
## Functions

Use snake_case + verb-first naming:

```
create_patient()
update_medical_record()
validate_token()
calculate_bmi()
```
## Variables

```
patient_name
doctor_id
appointment_date
```
Avoid:


```
x
temp
data
```
## Constants

#### MAX_LOGIN_ATTEMPTS

#### JWT_SECRET_KEY

#### DEFAULT_TIMEOUT

# 5 ) API Design Standards

Use RESTful APIs.

### Good endpoints

```
GET /patients
GET /patients/{id}
POST /patients
PUT /patients/{id}
DELETE /patients/{id}
```
### Version APIs

```
/api/v1/patients
/api/v1/doctors
```
### Response format standard

#### {

```
"success": true,
"message": "Patient created successfully",
"data": {}
}
```
Error:


#### {

```
"success": false,
"error_code": "PATIENT_NOT_FOUND",
"message": "Patient does not exist"
}
```
# 6 ) Database Standards

## Table naming

Use plural nouns:

```
patients
doctors
appointments
medical_records
```
## Primary keys

Avoid incremental IDs

```
id UUID PRIMARY KEY
```
## Common columns

Every table should have:

```
created_at
updated_at
deleted_at
created_by
```
# 7 ) Documentation Standards

Every service must contain:

```
README.md
```

Contains:

```
purpose
setup
environment variables
API endpoints
deployment steps
```
## Function/Method Docstring Format

Use Google style:

```
def create_patient(patient_data: dict):
"""
Creates a new patient.
```
```
Args:
patient_data (dict): Patient information.
```
```
Returns:
dict: Created patient object.
"""
```
## Class Docstring

```
class PatientService:
"""
Handles patient business logic.
```
```
Attributes:
repository: Database repository layer
"""
```
## API documentation

Use:

Swagger UI or Postman

# 8 ) Git Workflow

## Main branches

```
main
```

```
dev
```
## Feature branches

```
feature/auth-login
feature/patient-crud
feature/ai-chatbot
```
## Bug branches

Bug found during development (not production): Merge only to dev

```
bugfix/token-expiration
```
## Hotfix branches

Production bug: Merge to main and dev

```
hotfix/payment-failure
```
## Commit Convention

Use Conventional Commits:

```
feat: add patient registration
fix: resolve token validation issue
docs: update API docs
refactor: improve medical service logic
test: add appointment tests
```
## Pull Request Rules

Minimum:

```
1 reviewer approval
CI tests pass
No direct push to main
```
# 9 ) Scrum + Jira Guidelines


Each Micro-sevice is an Epic

```
Authentication
Patient Management
Doctor Management
Medical Records
Notifications
Infrastructure
Lab Agent
Rag Service
...
```
## Story Format

```
As a patient
I want to upload medical reports
So that doctors can review them
```
## Task Breakdown

Story: Build patient registration

Tasks:

```
Create DB schema
Build API
Write tests
Update documentation
```
## Definition of Done

Task is done only if:

```
Code Completed
Tested
Reviewed
Extracted to an Endpoint
Documented
Deployed to Staging
```
# 10 ) Testing Standards

Use: Pytest


```
tests/
unit/
integration/
e2e/
```
### Coverage target

Minimum:

#### 80%

# 11 ) Logging Standards

Use: ELK Stack or Grafana

## Log Levels

```
DEBUG → detailed internal info
INFO → normal operations
WARNING → unexpected behavior
ERROR → failures
CRITICAL → system crash
```
## Rules

```
Use structured logging (JSON format)
Never use print statements
Always include service name
Always include request_id
Log meaningful messages
```
```
logger.info(
"Patient created",
extra={
"patient_id": patient.id,
"service": "patient-service"
})
```
Avoid:

```
print("error happened")
```

```
logger.error("Something went wrong")
```
## Required Fields

```
timestamp
level
service
message
request_id
```
#### {

```
"timestamp": "...",
"level": "INFO",
"service": "patient-service",
"message": "Patient created",
"request_id": "...",
"user_id": "...",
"extra": {}
}
```
## Error Handling

```
Always log exceptions
Never suppress errors silently
```
```
try:
...
except Exception as e:
logger.exception("Unexpected error occurred")
raise
```
## Security

```
Do NOT log sensitive data (passwords, tokens, medical data)
```
## Centralized Logging

```
Use ELK Stack or Grafana for log aggregation
```
## Container Logging

```
Log to stdout/stderr only
```
## Request Tracing


```
Pass request_id across services via headers
```
# 12 ) Security Standards

Use:

```
JWT authentication
Role-based access control
Encrypt medical files
HTTPS only
Secrets in Vault/.env
Audit logs
```
Tools:

HashiCorp Vault OAuth 2. 0

# 13 ) Docker Standards

Each service:

```
Dockerfile
docker-compose.yml
```
Example:

```
app container
db container
redis container
```
# 14 ) CI/CD Standards

Use: GitHub

Pipeline:

```
Lint
→ Test
→ Build
→ Security Scan
→ Deploy
```

# 15 ) AI Microservice Standards

Since your platform includes AI:

## Separate model serving from business APIs

Bad:

```
patient-service handles ML model
```
Good:

```
medical-nlp-service
diagnosis-service
recommendation-service
chat-agent-service
```
## Model versioning

Use: MLflow

Track:

```
model version
dataset version
metrics
experiments
```
## Store large models separately

Use:

```
S 3
MinIO
HuggingFace storage
```
Amazon Web Services Hugging Face


