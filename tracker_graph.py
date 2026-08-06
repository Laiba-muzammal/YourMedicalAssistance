"""
tracker_graph.py
LangGraph StateGraph version of the medicine-tracking flow.

Flow:
  START -> check_website -> (available?) 
      yes -> found_node -> END
      no  -> not_found_node -> (is_recheck AND 12hrs passed?)
                  yes -> send_reminder_node -> END
                  no  -> END

Same graph is reused for both:
  - is_recheck=False : first-time check from the MCP tool (LLM.py) -> chat reply, saves pending if not found
  - is_recheck=True  : periodic check from scheduler.py -> email on found, reminder email if 12hrs passed
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from tracker_db import add_pending_record, mark_found, record_check_attempt, should_send_reminder, mark_reminder_sent
from tracker_utils import check_availability, send_email


# ---------------------------------------------------------
# STATE - the data that flows through the graph, node to node
# ---------------------------------------------------------
class TrackerState(TypedDict):
    medicine_name: str
    url: str
    recipient_email: str
    is_recheck: bool                 # False = first-time tool call, True = scheduler recheck
    record: Optional[dict]           # full DB row, only present when is_recheck=True
    is_available: bool
    detail: str
    result_message: str              # what gets returned to the caller at the end


# ---------------------------------------------------------
# NODES
# ---------------------------------------------------------
def check_website_node(state: TrackerState) -> TrackerState:
    """Fetch the page and determine availability. Never raises - errors become 'not available'."""
    try:
        is_available, detail = check_availability(state["url"], state["medicine_name"])
    except ConnectionError as e:
        is_available, detail = False, f"Could not check right now: {e}"
        if state.get("is_recheck") and state.get("record"):
            record_check_attempt(state["record"]["id"], error=str(e))

    return {**state, "is_available": is_available, "detail": detail}


def route_after_check(state: TrackerState) -> str:
    """Conditional edge: decide which branch to take after checking the website."""
    return "found" if state["is_available"] else "not_found"


def found_node(state: TrackerState) -> TrackerState:
    """Medicine is available: on recheck, email + mark found. On first check, just message."""
    message = f"{state['medicine_name']} is available at {state['url']}. {state['detail']}"

    if state["is_recheck"]:
        send_email(
            state["recipient_email"],
            subject=f"✅ {state['medicine_name']} is now available!",
            body=f"Good news! '{state['medicine_name']}' is now available at:\n{state['url']}\n\n{state['detail']}",
        )
        mark_found(state["record"]["id"])
        record_check_attempt(state["record"]["id"], error=None)
        message = "Found on recheck - email sent, tracking closed."

    return {**state, "result_message": message}


def not_found_node(state: TrackerState) -> TrackerState:
    """
    Not available. First-time check -> save a pending record.
    Recheck -> just log the attempt, reminder decision happens in the next node.
    """
    if not state["is_recheck"]:
        add_pending_record(state["medicine_name"], state["url"], state["recipient_email"])
        message = (
            f"{state['medicine_name']} isn't available at {state['url']} right now. "
            f"I'll keep checking in the background and email {state['recipient_email']} "
            f"when it's available (or a reminder every 12 hours)."
        )
    else:
        record_check_attempt(state["record"]["id"], error=None)
        message = "Still not available - checked, no email sent yet."

    return {**state, "result_message": message}


def route_after_not_found(state: TrackerState) -> str:
    """Only rechecks that are 12+ hours since the last reminder go on to send one."""
    if state["is_recheck"] and should_send_reminder(state["record"]):
        return "send_reminder"
    return "done"


def send_reminder_node(state: TrackerState) -> TrackerState:
    send_email(
        state["recipient_email"],
        subject=f"⏳ Still checking: {state['medicine_name']}",
        body=(
            f"Just a reminder - you asked me to track '{state['medicine_name']}' at:\n{state['url']}\n\n"
            f"It's still not confirmed available. I'll keep checking and notify you as soon as it is."
        ),
    )
    mark_reminder_sent(state["record"]["id"])
    return {**state, "result_message": "Still not available - 12hr reminder email sent."}


# ---------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------
graph_builder = StateGraph(TrackerState)

graph_builder.add_node("check_website", check_website_node)
graph_builder.add_node("found_node", found_node)
graph_builder.add_node("not_found_node", not_found_node)
graph_builder.add_node("send_reminder_node", send_reminder_node)

graph_builder.add_edge(START, "check_website")
graph_builder.add_conditional_edges(
    "check_website", route_after_check,
    {"found": "found_node", "not_found": "not_found_node"},
)
graph_builder.add_edge("found_node", END)
graph_builder.add_conditional_edges(
    "not_found_node", route_after_not_found,
    {"send_reminder": "send_reminder_node", "done": END},
)
graph_builder.add_edge("send_reminder_node", END)

tracker_graph = graph_builder.compile()


# ---------------------------------------------------------
# Helper entry points - what LLM.py and scheduler.py actually call
# ---------------------------------------------------------
def run_immediate_check(medicine_name: str, url: str, recipient_email: str) -> str:
    """Used by the MCP tool - first-time check, no email, just a chat reply."""
    result = tracker_graph.invoke({
        "medicine_name": medicine_name,
        "url": url,
        "recipient_email": recipient_email,
        "is_recheck": False,
        "record": None,
        "is_available": False,
        "detail": "",
        "result_message": "",
    })
    return result["result_message"]


def run_recheck(record: dict) -> str:
    """Used by scheduler.py - one pending DB record goes through the graph."""
    result = tracker_graph.invoke({
        "medicine_name": record["medicine_name"],
        "url": record["url"],
        "recipient_email": record["recipient_email"],
        "is_recheck": True,
        "record": record,
        "is_available": False,
        "detail": "",
        "result_message": "",
    })
    return result["result_message"]