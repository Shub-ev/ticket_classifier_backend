from typing_extensions import NotRequired, TypedDict
from langgraph.graph import END, StateGraph
from agents.structured_response.classify_ticket import ClassifyTicket
from agents.tools.classify_ticket import classify_ticket


class TicketTriageState(TypedDict):
    """State passed between ticket-classification and action-routing nodes."""

    ticket: dict
    decision: NotRequired[ClassifyTicket]
    status: NotRequired[str]


def classify_node(state: TicketTriageState):
    """Attach the validated LLM decision to the workflow state."""
    return {"decision": classify_ticket(state["ticket"])}


def assign_node(state: TicketTriageState):
    """Mark the workflow outcome as an assignment."""
    return {"status": "assigned"}


def reject_node(state: TicketTriageState):
    """Mark the workflow outcome as a rejection."""
    return {"status": "rejected"}


def human_review_node(state: TicketTriageState):
    """Mark the workflow outcome as requiring human review."""
    return {"status": "human_review"}


def route_action(state: TicketTriageState):
    """Select the action branch matching the validated LLM decision."""
    return state["decision"].action


# Classification produces one of three explicit outcomes; each terminal node
# records that outcome before the graph returns to the persistence layer.
workflow = StateGraph(TicketTriageState)
workflow.add_node("classify", classify_node)
workflow.add_node("assign", assign_node)
workflow.add_node("reject", reject_node)
workflow.add_node("human_review", human_review_node)
workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    route_action,
    {
        "assign": "assign",
        "reject": "reject",
        "human_review": "human_review",
    },
)
workflow.add_edge("assign", END)
workflow.add_edge("reject", END)
workflow.add_edge("human_review", END)

ticket_triage_graph = workflow.compile()
