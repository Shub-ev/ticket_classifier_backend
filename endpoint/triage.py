from fastapi import APIRouter, HTTPException
from utils import TicketRequest, HumanReviewAssignment
from helper_functions import (
    assign_human_review_ticket,
    get_assignment_groups,
    get_assigned_tickets,
    get_human_review_queue,
    get_meta_data,
    predict_ticket_details,
    update_classifier_model_accuracy,
)
from pathlib import Path
import json
import os

triage_router = APIRouter(prefix="/triage", tags=["Triage"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = os.path.join(BASE_DIR, "data", "test_tickets.json")


## This endpoint runs internal tests and return the results
@triage_router.get("/tests/internal")
def run_internal_tests():

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    passed = 0
    failed = 0
    human_review = 0

    results = []

    for ticket in test_data:

        text = ticket["short_description"]
        expected = ticket["expected_group"]

        prediction_result = predict_ticket_details(text, save_for_review=True)

        prediction = prediction_result["predicted_group"]
        confidence = prediction_result["confidence"]

        status = ""

        # Human review
        if confidence < 80:
            status = "human_review"
            human_review += 1

        # Correct prediction with confidence >= 80
        elif prediction == expected:
            status = "passed"
            passed += 1

        # Wrong prediction with confidence >= 80
        else:
            status = "failed"
            failed += 1

        results.append({
            "text": text,
            "expected": expected,
            **prediction_result,
            "status": status
        })

    total_evaluated = passed + failed

    accuracy = (
        round((passed / total_evaluated) * 100, 2)
        if total_evaluated > 0
        else 0
    )

    update_classifier_model_accuracy(accuracy)

    return {
        "total_tests": len(test_data),
        "passed": passed,
        "failed": failed,
        "human_review": human_review,
        "evaluated_tests": total_evaluated,
        "accuracy_percent": accuracy,
        "results": results
    }



@triage_router.post("/tests/")
def predict_ticket(ticket: TicketRequest):
    return predict_ticket_details(
        ticket.short_description,
        save_for_review=True,
        save_assignment=True,
    )


@triage_router.get("/human_review")
def get_tickets_for_human_review():
    return get_human_review_queue()


@triage_router.post("/human_review")
def assign_human_review_ticket_endpoint(assignment: HumanReviewAssignment):
    if not assignment.assignment_group.strip():
        raise HTTPException(status_code=422, detail="assignment_group must not be empty")

    ticket = assign_human_review_ticket(
        assignment.id,
        assignment.assignment_group.strip(),
    )
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found in human review queue")
    return {"message": "Ticket assigned successfully", "ticket": ticket}


@triage_router.get("/assigned")
def get_assigned_tickets_endpoint():
    return get_assigned_tickets()


@triage_router.get("/assignment_groups")
def get_assignment_groups_endpoint():
    return get_assignment_groups()


@triage_router.get("/meta_data")
def get_meta_data_endpoint():
    return get_meta_data()



@triage_router.get("/tests/internal_tests")
def get_internal_test_cases():
    """
    Returns the JSON file used for testing
    """

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)