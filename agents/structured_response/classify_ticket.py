from typing import Literal
from pydantic import BaseModel, Field, model_validator
from utils import AssignmentGroup


class ClassifyTicket(BaseModel):
    """Validated LLM decision for the ticket-triage workflow."""

    action: Literal["assign", "reject", "human_review"]
    assignment_group: AssignmentGroup | None
    confidence: float = Field(ge=0, le=100)
    rationale: str
    rejection_reason: str | None

    @model_validator(mode="after")
    def validate_action_fields(self):
        """Ensure fields are consistent with the selected action."""
        if self.action == "assign" and self.assignment_group is None:
            raise ValueError("An assignment group is required when action is assign")
        if self.action != "assign" and self.assignment_group is not None:
            raise ValueError("An assignment group is only valid when action is assign")
        if self.action == "reject" and not self.rejection_reason:
            raise ValueError("A rejection reason is required when action is reject")
        return self
