from pydantic import BaseModel

class TicketRequest(BaseModel):
    short_description: str


class HumanReviewAssignment(BaseModel):
    id: str
    assignment_group: str


class AssignedTicketRejection(BaseModel):
    id: str
    reason: str | None = None