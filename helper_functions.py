from datetime import datetime, timezone
from pathlib import Path
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
ASSIGNED_TICKETS_FILE = DATA_DIR / "assigned_tickets.json"
REJECTED_TICKETS_FILE = DATA_DIR / "rejected_tickets.json"
ASSIGNMENT_GROUPS_FILE = DATA_DIR / "assignment_groups.json"
META_DATA_FILE = DATA_DIR / "meta_data.json"
HUMAN_REVIEW_TRAINING_FILE = DATA_DIR / "human_review_training.csv"
HUMAN_REVIEW_THRESHOLD = 80.0

model = joblib.load(BASE_DIR / "models" / "tfidf_classifier.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
relevance_model = joblib.load(BASE_DIR / "models" / "relevance_classifier.pkl")


def predict_ticket_relevance(text: str):
    prediction = relevance_model.predict([text])[0]
    probabilities = relevance_model.predict_proba([text])[0]
    class_probabilities = dict(zip(relevance_model.classes_, probabilities))

    return prediction, round(float(class_probabilities[prediction] * 100), 2)


def predict_ticket_details(
    text: str,
    save_for_review: bool = False,
    save_assignment: bool = False,
):
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
            ticket["id"] = add_to_human_review(ticket)
            response["id"] = ticket["id"]

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
            ticket["id"] = add_to_human_review(ticket)
            response["id"] = ticket["id"]

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
        ticket["id"] = add_to_human_review(ticket)
        response["id"] = ticket["id"]
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
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {file_path}")
    return data


def _write_json_array(file_path: Path, data):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def get_human_review_queue():
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


def add_to_human_review(ticket):
    queue = get_human_review_queue()
    ticket_id = str(uuid4())
    ticket["id"] = ticket_id
    queue.append(ticket)
    _write_json_array(HUMAN_REVIEW_FILE, queue)
    return ticket_id


def save_assigned_ticket(ticket):
    assigned_tickets = _read_json_array(ASSIGNED_TICKETS_FILE)
    assigned_tickets.append(ticket)
    _write_json_array(ASSIGNED_TICKETS_FILE, assigned_tickets)
    if ticket.get("assignment_group"):
        _store_assignment_group(ticket["assignment_group"])


def save_human_review_training_example(text: str, assignment_group: str):
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
    return _read_json_array(ASSIGNED_TICKETS_FILE)


def reject_assigned_ticket(ticket_id: str, reason: str | None = None):
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

    stored_groups = _read_json_array(ASSIGNMENT_GROUPS_FILE)
    if any(not isinstance(group, str) for group in stored_groups):
        raise ValueError(f"Expected assignment group names in {ASSIGNMENT_GROUPS_FILE}")
    assignment_groups = list(dict.fromkeys(
        [str(group) for group in model.classes_] + stored_groups
    ))
    metadata["ticket_counts"] = {
        "waiting_for_human_review": len(get_human_review_queue()),
        "assigned": len(get_assigned_tickets()),
        "rejected": len(_read_json_array(REJECTED_TICKETS_FILE)),
        "human_review_training_examples": training_examples,
    }
    metadata["assignment_groups"] = assignment_groups
    return metadata


def update_classifier_model_accuracy(accuracy: float):
    metadata = get_meta_data()
    metadata["classifier_model_accuracy"] = accuracy
    with META_DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def _store_assignment_group(assignment_group: str):
    assignment_group = assignment_group.strip()
    if not assignment_group:
        return

    groups = _read_json_array(ASSIGNMENT_GROUPS_FILE)
    if assignment_group not in groups:
        groups.append(assignment_group)
        _write_json_array(ASSIGNMENT_GROUPS_FILE, groups)


def get_assignment_groups():
    groups = _read_json_array(ASSIGNMENT_GROUPS_FILE)
    if any(not isinstance(group, str) for group in groups):
        raise ValueError(f"Expected assignment group names in {ASSIGNMENT_GROUPS_FILE}")

    known_groups = [str(group) for group in model.classes_]
    known_groups.extend(
        ticket["assignment_group"]
        for ticket in get_assigned_tickets()
        if ticket.get("assignment_group")
    )

    for group in known_groups:
        if group not in groups:
            groups.append(group)

    if groups != _read_json_array(ASSIGNMENT_GROUPS_FILE):
        _write_json_array(ASSIGNMENT_GROUPS_FILE, groups)
    return groups


def assign_human_review_ticket(ticket_id: str, assignment_group: str):
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
