"""Ticket inference, triage persistence, and review-queue operations.

The fitted classifiers are loaded once at module import. Ticket outcomes and
metadata are persisted as JSON arrays under ``data/`` for local development.
"""

from datetime import datetime, timezone
from pathlib import Path
from utils import ASSIGNMENT_GROUPS
import numpy as np
import joblib
import json
import csv
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
print("##### BASE PATH: ", BASE_DIR)
TEST_FILE = DATA_DIR / "test_tickets.json"
HUMAN_REVIEW_FILE = DATA_DIR / "human_review_queue.json"
LLM_REVIEW_FILE = DATA_DIR / "llm_review_queue.json"
ASSIGNED_TICKETS_FILE = DATA_DIR / "assigned_tickets.json"
REJECTED_TICKETS_FILE = DATA_DIR / "rejected_tickets.json"
META_DATA_FILE = DATA_DIR / "meta_data.json"
HUMAN_REVIEW_TRAINING_FILE = DATA_DIR / "human_review_training.csv"
HUMAN_REVIEW_THRESHOLD = 80.0

model = joblib.load(BASE_DIR / "models" / "tfidf_classifier.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
relevance_model = joblib.load(BASE_DIR / "models" / "relevance_classifier.pkl")


def predict_ticket_relevance(text: str):
    """Predict whether a ticket is in IT scope and return its confidence.

    Args:
        text: Ticket description to evaluate.

    Returns:
        A tuple containing the predicted ``YES``/``NO`` label and confidence
        as a percentage.
    """
    prediction = relevance_model.predict([text])[0]
    probabilities = relevance_model.predict_proba([text])[0]
    class_probabilities = dict(zip(relevance_model.classes_, probabilities))

    return prediction, round(float(class_probabilities[prediction] * 100), 2)


def predict_ticket_details(
    text: str,
    save_for_review: bool = False,
    save_assignment: bool = False,
):
    """Classify a ticket and optionally persist its resulting workflow state.

    Relevant tickets are evaluated by the assignment-group model. Cases below
    its confidence or vocabulary-coverage thresholds are routed through the
    LLM review flow when ``save_for_review`` is enabled.

    Args:
        text: Ticket description.
        save_for_review: Submit uncertain tickets for LLM triage.
        save_assignment: Persist confident model assignments.

    Returns:
        A prediction payload including confidence, relevance, and group hints.
    """
    relevance_prediction, relevance_confidence = predict_ticket_relevance(text)

    if relevance_prediction != "YES":
        response = {
            "predicted_group": "Human Review",
            "confidence": 0.0,
            "human_review_required": True,
            "reason": "Relevance classifier marked this ticket as out of scope",
            "relevance_prediction": relevance_prediction,
            "relevance_confidence": relevance_confidence,
            "top_predictions": [],
        }

        if save_for_review:
            ticket = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "text": text,
                **response,
            }
            response.update(_submit_to_llm_review(ticket))

        return response

    tokens = text.lower().split()

    known_tokens = [
        token
        for token in tokens
        if token in vectorizer.vocabulary_
    ]

    unknown_tokens = [
        token
        for token in tokens
        if token not in vectorizer.vocabulary_
    ]

    coverage = (
        len(known_tokens) / len(tokens)
        if len(tokens) > 0
        else 0
    )

    text_tfidf = vectorizer.transform([text])

    # Check if any known words exist
    if text_tfidf.nnz == 0:

        response = {
            "predicted_group": "Human Review",
            "confidence": 0.0,
            "human_review_required": True,
            "reason": "No known vocabulary terms found in training data",
            "relevance_prediction": relevance_prediction,
            "relevance_confidence": relevance_confidence,
            "top_predictions": []
        }

        if save_for_review:
            ticket = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "text": text,
                **response
            }
            response.update(_submit_to_llm_review(ticket))

        return response

    # Calculate vocabulary coverage
    tokens = text.lower().split()

    known_tokens = sum(
        1
        for token in tokens
        if token in vectorizer.vocabulary_
    )

    coverage = (
        known_tokens / len(tokens)
        if len(tokens) > 0
        else 0
    )

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)[0]

    confidence = round(float(np.max(probabilities) * 100), 2)

    top_predictions = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # Human review if:
    # 1. Low confidence
    # 2. Not enough known words
    human_review_required = (
        confidence < HUMAN_REVIEW_THRESHOLD
        or coverage < 0.20
    )

    response = {
        "predicted_group": (
            "Human Review"
            if human_review_required
            else prediction
        ),
        "model_prediction": prediction,
        "confidence": confidence,
        "vocabulary_coverage": round(coverage * 100, 2),
        "human_review_required": human_review_required,
        "relevance_prediction": relevance_prediction,
        "relevance_confidence": relevance_confidence,
        "top_predictions": [
            {
                "group": group,
                "probability": round(float(prob * 100), 2)
            }
            for group, prob in top_predictions
        ]
    }

    if save_for_review and human_review_required:
        ticket = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
            **response
        }
        response.update(_submit_to_llm_review(ticket))
    elif save_assignment and not human_review_required:
        ticket_id = str(uuid4())
        response["id"] = ticket_id
        save_assigned_ticket({
            "id": ticket_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "assigned_at": datetime.now(timezone.utc).isoformat(),
            "assigned_by": "model",
            "assignment_group": prediction,
            "text": text,
            **response,
        })

    return response



def _read_json_array(file_path: Path):
    """Read a JSON array, returning an empty list when the file is absent.

    Raises:
        ValueError: If the file exists but does not contain a JSON array.
    """
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {file_path}")
    return data


def _write_json_array(file_path: Path, data):
    """Persist a JSON-compatible list to ``file_path`` with readable spacing."""
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def get_human_review_queue():
    """Return human-review tickets and repair missing or duplicate IDs."""
    queue = _read_json_array(HUMAN_REVIEW_FILE)
    ids = set()
    changed = False
    for ticket in queue:
        ticket_id = ticket.get("id")
        if not ticket_id or ticket_id in ids:
            ticket_id = str(uuid4())
            ticket["id"] = ticket_id
            changed = True
        ids.add(ticket_id)

    if changed:
        _write_json_array(HUMAN_REVIEW_FILE, queue)
    return queue


def get_llm_review_queue():
    """Return tickets waiting for LLM triage."""
    return _read_json_array(LLM_REVIEW_FILE)


def add_to_llm_review(ticket):
    """Append a ticket with a new ID and pending status to the LLM queue."""
    ticket_id = str(uuid4())
    ticket["id"] = ticket_id
    ticket["status"] = "pending_llm_review"

    llm_review_queue = _read_json_array(LLM_REVIEW_FILE)
    llm_review_queue.append(ticket.copy())
    _write_json_array(LLM_REVIEW_FILE, llm_review_queue)
    return ticket_id


def process_llm_review_ticket(ticket_id: str) -> dict:
    """Run one queued ticket through LangGraph and persist its final outcome.

    A failed LLM invocation leaves the ticket in the LLM queue so it can be
    retried. Successful outcomes are moved to the assigned, rejected, or human
    review collection.

    Args:
        ticket_id: Identifier of the queued ticket to process.

    Returns:
        The processed ticket, including its LLM decision and final status.

    Raises:
        LookupError: If no queued ticket has the requested ID.
        RuntimeError: If all configured LLM providers fail.
    """
    queue = _read_json_array(LLM_REVIEW_FILE)
    ticket_index = next(
        (index for index, ticket in enumerate(queue) if ticket.get("id") == ticket_id),
        None,
    )
    if ticket_index is None:
        raise LookupError(f"Ticket {ticket_id} not found in LLM review queue")

    from agents.langgraph_agents.ticket_triage_graph import ticket_triage_graph

    ticket = queue[ticket_index]
    decision = ticket_triage_graph.invoke({
        "ticket": {
            **ticket,
            "allowed_assignment_groups": list(ASSIGNMENT_GROUPS),
        }
    })["decision"]
    now = datetime.now(timezone.utc).isoformat()
    ticket.update({
        "llm_action": decision.action,
        "llm_confidence": decision.confidence,
        "llm_rationale": decision.rationale,
        "reason": decision.rationale,
        "processed_at": now,
    })

    if decision.action == "assign":
        ticket.update({
            "status": "assigned",
            "assigned_at": now,
            "assigned_by": "llm",
            "assignment_group": decision.assignment_group,
            "human_review_required": False,
            "predicted_group": decision.assignment_group,
        })
        save_assigned_ticket(ticket.copy())
    elif decision.action == "reject":
        ticket.update({
            "status": "rejected",
            "rejected_at": now,
            "rejected_by": "llm",
            "rejection_reason": decision.rejection_reason,
            "human_review_required": False,
            "predicted_group": "Rejected",
        })
        rejected_tickets = _read_json_array(REJECTED_TICKETS_FILE)
        if not any(item.get("id") == ticket_id for item in rejected_tickets):
            rejected_tickets.append(ticket.copy())
        _write_json_array(REJECTED_TICKETS_FILE, rejected_tickets)
    else:
        ticket.update({
            "status": "human_review",
            "assigned_by": "human",
            "human_review_required": True,
            "predicted_group": "Human Review",
        })
        human_review_queue = get_human_review_queue()
        if not any(item.get("id") == ticket_id for item in human_review_queue):
            human_review_queue.append(ticket.copy())
        _write_json_array(HUMAN_REVIEW_FILE, human_review_queue)

    queue.pop(ticket_index)
    _write_json_array(LLM_REVIEW_FILE, queue)
    return ticket


def _submit_to_llm_review(ticket) -> dict:
    """Queue and immediately process a ticket through LLM triage."""
    ticket_id = add_to_llm_review(ticket)
    return process_llm_review_ticket(ticket_id)


def process_pending_llm_review_tickets():
    """Process each ticket currently pending in the LLM review queue."""
    results = []
    for ticket in _read_json_array(LLM_REVIEW_FILE):
        results.append(process_llm_review_ticket(ticket["id"]))
    return results


def save_assigned_ticket(ticket):
    """Persist an assignment once, using its ticket ID as the idempotency key."""
    assigned_tickets = _read_json_array(ASSIGNED_TICKETS_FILE)
    if not any(item.get("id") == ticket.get("id") for item in assigned_tickets):
        assigned_tickets.append(ticket)
        _write_json_array(ASSIGNED_TICKETS_FILE, assigned_tickets)


def save_human_review_training_example(text: str, assignment_group: str):
    """Append a human correction for future assignment-model training."""
    file_exists = HUMAN_REVIEW_TRAINING_FILE.exists()
    write_header = not file_exists or HUMAN_REVIEW_TRAINING_FILE.stat().st_size == 0

    with HUMAN_REVIEW_TRAINING_FILE.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "Assignment Group"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "text": text,
            "Assignment Group": assignment_group,
        })


def get_assigned_tickets():
    """Return all persisted assigned tickets."""
    return _read_json_array(ASSIGNED_TICKETS_FILE)


def reject_assigned_ticket(ticket_id: str, reason: str | None = None):
    """Move an assigned ticket to the rejected collection.

    Returns ``None`` when no assigned ticket has the requested ID.
    """
    assigned_tickets = _read_json_array(ASSIGNED_TICKETS_FILE)
    ticket_index = next(
        (index for index, ticket in enumerate(assigned_tickets) if ticket.get("id") == ticket_id),
        None,
    )
    if ticket_index is None:
        return None

    ticket = assigned_tickets.pop(ticket_index)
    ticket.pop("assignment_group", None)
    ticket.pop("assigned_at", None)
    ticket.pop("assigned_by", None)
    ticket.update({
        "status": "rejected",
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "rejected_by": "client",
    })
    if reason and reason.strip():
        ticket["rejection_reason"] = reason.strip()

    rejected_tickets = _read_json_array(REJECTED_TICKETS_FILE)
    rejected_tickets.append(ticket)
    _write_json_array(REJECTED_TICKETS_FILE, rejected_tickets)
    _write_json_array(ASSIGNED_TICKETS_FILE, assigned_tickets)
    return ticket


def reject_human_review_ticket(ticket_id: str, reason: str | None = None):
    """Reject a ticket from human review and persist the optional reason.

    Returns ``None`` when no human-review ticket has the requested ID.
    """
    queue = get_human_review_queue()
    ticket_index = next(
        (index for index, ticket in enumerate(queue) if ticket.get("id") == ticket_id),
        None,
    )
    if ticket_index is None:
        return None

    ticket = queue.pop(ticket_index)
    ticket.pop("assignment_group", None)
    ticket.update({
        "status": "rejected",
        "human_review_required": False,
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "rejected_by": "client",
    })
    if reason and reason.strip():
        ticket["rejection_reason"] = reason.strip()

    rejected_tickets = _read_json_array(REJECTED_TICKETS_FILE)
    rejected_tickets.append(ticket)
    _write_json_array(REJECTED_TICKETS_FILE, rejected_tickets)
    _write_json_array(HUMAN_REVIEW_FILE, queue)
    return ticket


def get_meta_data():
    """Return configured metadata enriched with live queue and group counts."""
    metadata = {}
    if META_DATA_FILE.exists():
        with META_DATA_FILE.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a JSON object in {META_DATA_FILE}")

    if HUMAN_REVIEW_TRAINING_FILE.exists():
        with HUMAN_REVIEW_TRAINING_FILE.open("r", newline="", encoding="utf-8-sig") as file:
            training_examples = sum(1 for _ in csv.DictReader(file))
    else:
        training_examples = 0

    metadata["ticket_counts"] = {
        "waiting_for_human_review": len(get_human_review_queue()),
        "waiting_for_llm_review": len(_read_json_array(LLM_REVIEW_FILE)),
        "assigned": len(get_assigned_tickets()),
        "rejected": len(_read_json_array(REJECTED_TICKETS_FILE)),
        "human_review_training_examples": training_examples,
    }
    metadata["assignment_groups"] = list(ASSIGNMENT_GROUPS)
    return metadata


def update_classifier_model_accuracy(accuracy: float):
    """Update the recorded accuracy while preserving other metadata fields."""
    metadata = get_meta_data()
    metadata["classifier_model_accuracy"] = accuracy
    with META_DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def get_assignment_groups():
    """Return the approved assignment-group names."""
    return list(ASSIGNMENT_GROUPS)


def assign_human_review_ticket(ticket_id: str, assignment_group: str):
    """Assign a human-reviewed ticket and record its correction for training.

    Returns ``None`` when no ticket with the requested ID is awaiting review.
    """
    queue = get_human_review_queue()
    ticket_index = next(
        (index for index, ticket in enumerate(queue) if ticket.get("id") == ticket_id),
        None,
    )
    if ticket_index is None:
        return None

    ticket = queue.pop(ticket_index)
    ticket.update({
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "assigned_by": "human",
        "assignment_group": assignment_group,
        "human_review_required": False,
    })

    save_human_review_training_example(ticket.get("text", ""), assignment_group)
    save_assigned_ticket(ticket)
    _write_json_array(HUMAN_REVIEW_FILE, queue)
    return ticket
