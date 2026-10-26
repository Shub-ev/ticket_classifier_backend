from typing import Literal, get_args

from pydantic import BaseModel


# Keep this list authoritative for both model validation and client-facing APIs.
AssignmentGroup = Literal[
    "DWE Remote AV conference Support",
    "DWE Remote Support Compute HW_SW",
    "DWE Remote Support M365",
    "Low Volume Support Groups",
    "TCS APEX Centre L1 Support",
    "TCS Service Desk",
]
ASSIGNMENT_GROUPS = get_args(AssignmentGroup)


class TicketRequest(BaseModel):
    """Input payload for ticket-classification requests."""

    short_description: str


class HumanReviewAssignment(BaseModel):
    """Request to assign a human-reviewed ticket to an approved group."""

    id: str
    assignment_group: AssignmentGroup


class AssignedTicketRejection(BaseModel):
    """Request to reject an assigned or human-reviewed ticket."""

    id: str
    reason: str | None = None