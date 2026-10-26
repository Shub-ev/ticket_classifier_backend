# Ticket Classifier Backend

A FastAPI service that classifies IT support tickets, assigns confident predictions to an approved support group, and escalates uncertain cases through LLM and human review.

## Ticket lifecycle

1. A ticket description is checked by the IT relevance classifier.
2. Relevant tickets are classified into an assignment group by the TF-IDF model.
3. High-confidence predictions that meet the vocabulary-coverage threshold are assigned directly.
4. Uncertain predictions and tickets the relevance model does not recognize are submitted to the LLM review flow. LangGraph coordinates a structured LLM decision to assign an approved group, reject clearly irrelevant content, or request human review.
5. Human reviewers can assign tickets to an approved group or reject them.

LLM assignment and rejection decisions below the 75% confidence threshold are routed to human review. The TF-IDF model uses an 80% confidence threshold and 20% vocabulary coverage threshold.

## Project layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Creates the FastAPI application and configures CORS. |
| `endpoint/triage.py` | Exposes ticket, queue, assignment, rejection, metadata, and test endpoints. |
| `helper_functions.py` | Runs local model inference and persists ticket states and training examples. |
| `utils.py` | Defines request schemas and the set of supported assignment groups. |
| `agents/langchain_agents/` | Configures LLM providers and structured invocation with provider fallback. |
| `agents/langgraph_agents/` | Defines the LangGraph ticket-triage workflow. |
| `agents/tools/` | Builds the classification request and applies the LLM confidence policy. |
| `agents/structured_response/` | Validates structured LLM triage decisions. |
| `agents/prompts/` | Contains the ticket-classification system prompt. |
| `config/settings.py` | Loads provider credentials from environment variables or the local `.env` file. |
| `models/` | Contains serialized classifiers and model-training utilities. |
| `data/` | Contains model inputs, test cases, metadata, queues, and review-training data. |

## Setup

Use Python 3.10 or later. Create a virtual environment, install dependencies, and provide both LLM provider credentials:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` and `GEMINI_API_KEY` in the environment or in a local `.env` file at the repository root. Do not commit credentials. The three serialized model files in `models/` must be present before importing the inference module.

Start the API from the repository root:

```powershell
uvicorn main:app --reload
```

FastAPI's interactive API documentation is available at `/docs`.

## API

All routes are prefixed with `/triage`.

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/tests/` | Classifies a ticket description and persists the resulting ticket state. |
| `GET` | `/human_review` | Lists tickets awaiting human review. |
| `POST` | `/human_review` | Assigns a human-review ticket to an approved group. |
| `POST` | `/human_review/reject` | Rejects a ticket awaiting human review. |
| `GET` | `/llm_review` | Lists tickets currently pending LLM processing. |
| `POST` | `/llm_review/process` | Retries LLM processing for tickets currently in the LLM queue. |
| `GET` | `/assigned` | Lists assigned tickets. |
| `POST` | `/assigned/reject` | Rejects an assigned ticket. |
| `GET` | `/assignment_groups` | Lists approved assignment groups. |
| `GET` | `/meta_data` | Returns model metadata, queue counts, and supported groups. |
| `GET` | `/tests/internal` | Runs the configured internal test cases and updates the recorded classifier accuracy. |
| `GET` | `/tests/internal_tests` | Returns the internal test-case dataset. |

The ticket endpoint accepts a JSON body such as:

```json
{
  "short_description": "Unable to access Outlook"
}
```

To assign a human-review ticket, send the ticket ID and one of the approved group names:

```json
{
  "id": "ticket-id",
  "assignment_group": "DWE Remote Support M365"
}
```

Rejection endpoints accept a ticket ID and an optional reason:

```json
{
  "id": "ticket-id",
  "reason": "Not an IT support request"
}
```

Human assignment returns `404` if the ticket is not in the human-review queue; rejection endpoints return `404` when the ticket is not in the relevant queue. Group names are restricted to the values defined in `utils.py`.

## Configuration and decision fields

Provider credentials are required when the settings module is loaded. LLM providers are attempted in configured priority order; a successful structured response must satisfy the Pydantic decision schema. Assignment decisions include the selected `assignment_group`; rejection decisions include `rejection_reason`; uncertain cases are persisted in the human-review queue. Ticket records preserve the action, confidence, rationale, timestamps, and assigning or rejecting actor where applicable.

## Data and operational notes

Ticket queues and outcomes are stored as JSON arrays under `data/`; human corrections are appended to `data/human_review_training.csv`. JSON files are suitable for local development and single-process use, but do not provide transactional or concurrent-write guarantees. Use a transactional database before running multiple workers or handling production traffic.

The model-training scripts under `models/IT_relevence_classifier/` are offline utilities and are not part of API startup. The API's internal test endpoint writes ticket outcomes and metadata; use a disposable dataset when running it outside development.
