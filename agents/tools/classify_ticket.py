import json

from agents.langchain_agents.agent_interface import AgentInterface
from agents.prompts.system_prompt.classify_ticket import (
    CLASSIFY_TICKET_SYSTEM_PROMPT,
)
from agents.structured_response.classify_ticket import ClassifyTicket


GROUP_GUIDANCE = {
    "DWE Remote AV conference Support": (
        "conference-room AV, Teams room devices, room consoles, and meeting-room equipment"
    ),
    "DWE Remote Support Compute HW_SW": (
        "laptops, Windows, endpoint hardware, operating systems, and software installation"
    ),
    "DWE Remote Support M365": (
        "Outlook, Microsoft 365, OneDrive, and user-facing Teams collaboration problems"
    ),
    "Low Volume Support Groups": (
        "specialized or lower-volume applications with no better listed owner"
    ),
    "TCS APEX Centre L1 Support": (
        "infrastructure, platform monitoring, cloud, server, and integration alerts"
    ),
    "TCS Service Desk": (
        "passwords, account access, onboarding, and general service requests"
    ),
}


def classify_ticket(ticket: dict) -> ClassifyTicket:
    """Ask the configured LLM to choose an allowed ticket-triage outcome.

    Args:
        ticket: Ticket fields and classifier hints. The caller must include
            ``allowed_assignment_groups``.

    Returns:
        A schema-validated decision. Low-confidence assignment and rejection
        decisions are converted to human review.
    """
    allowed_groups = "\n".join(
        f"- {group}: {GROUP_GUIDANCE.get(group, 'Use only if it is the best match.')}"
        for group in ticket["allowed_assignment_groups"]
    )
    ticket_context = {
        key: value
        for key, value in ticket.items()
        if key != "allowed_assignment_groups"
    }
    prompt = (
        f"Allowed assignment groups and their scope:\n{allowed_groups}\n"
        f"Ticket and classifier context (JSON): {json.dumps(ticket_context)}"
    )
    decision = AgentInterface().structured_invoke(
        prompt,
        ClassifyTicket,
        system_prompt=CLASSIFY_TICKET_SYSTEM_PROMPT,
    )
    if decision.action in {"assign", "reject"} and decision.confidence < 75:
        return decision.model_copy(update={
            "action": "human_review",
            "assignment_group": None,
            "rejection_reason": None,
            "rationale": (
                f"{decision.rationale} Confidence below the 75% automation "
                "threshold; routed to human review."
            ),
        })
    return decision
