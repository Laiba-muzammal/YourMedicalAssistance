"""
Dr. MedAssist — LangChain/LangGraph ReAct Agent
=================================================
A medical-assisting conversational agent built on LangGraph's prebuilt
ReAct architecture. The agent has access to three tools: a time lookup,
a safe numeric calculator (for dosage/unit calculations), and a live
web search (for up-to-date medical information).

Session state (conversation history) is persisted in-memory per thread_id
using LangGraph's InMemorySaver checkpointer — no external database
required. Note: state is lost when the process exits; for production use,
swap InMemorySaver for a persistent checkpointer (e.g. SqliteSaver).
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun  # pip install duckduckgo-search

# Load environment variables (expects OPENROUTER_API_KEY in a .env file)
load_dotenv()

# ---------------------------------------------------------
# STEP 1: Model Setup (via OpenRouter)
# ---------------------------------------------------------
# Using Meta's Llama 3.3 70B Instruct model through the OpenRouter API,
# which exposes an OpenAI-compatible interface.
#
# Tuned parameters explained:
# - temperature=0.3 : Lower = more consistent/deterministic answers.
#   Important for medical info — we don't want dosage numbers changing
#   randomly between runs of the same question.
# - max_tokens=500  : Caps response length. High enough that dosage +
#   safety warnings + disclaimer aren't cut off mid-sentence, but not
#   so high that responses ramble or cost/latency balloon.
# - timeout=25      : Max seconds to wait for a model/API response
#   before raising an error, instead of hanging indefinitely.
# - max_retries=2   : Automatically retries failed API calls (e.g.
#   transient network issues) before giving up.
model = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=500,
    timeout=25,
    max_retries=2,
)

# ---------------------------------------------------------
# STEP 2: Tool Definitions
# ---------------------------------------------------------

@tool
def get_current_time() -> str:
    """Return the current date and time as an ISO-formatted string.

    Useful for timestamping responses or answering time-relative
    questions (e.g. 'how long until my next dose?').
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def safe_calculator(expression: str) -> str:
    """Evaluate a basic numeric expression safely, without using eval().

    Intended for dosage and unit calculations (e.g. mg/kg dosing,
    tablet counts, unit conversions). Uses `numexpr`, which only
    parses numeric math expressions and cannot execute arbitrary
    Python code — this avoids the remote-code-execution risk that
    a raw eval()-based calculator would introduce.

    Example:
        '25 * 4' -> 'Result: 100'

    Args:
        expression: A plain numeric expression (e.g. '5 * 70').

    Returns:
        A string containing the computed result, or an error message
        if the expression could not be evaluated.
    """
    import numexpr
    try:
        result = numexpr.evaluate(expression).item()
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the internet for up-to-date medical information.

    Use this for anything that may not be in the model's training data
    or requires current information — e.g. dosage guidelines, drug
    interactions, recent recalls, or disease outbreak updates.

    Powered by DuckDuckGo (free, no API key required).

    Args:
        query: The search query string.

    Returns:
        A string of search result snippets, or an error/fallback
        message if the search fails or returns nothing.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        print(f"🌐 Raw search result: {result[:300]}")  # debug — remove for production
        return result if result else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


# Register all tools available to the agent
tools = [get_current_time, safe_calculator, web_search]

# ---------------------------------------------------------
# STEP 3: Persona / System Prompt
# ---------------------------------------------------------
# Defines the agent's role, scope, and behavioral guidelines — including
# when to use which tool, when to defer to real healthcare professionals,
# and when a query is out of scope entirely.
PERSONA = """You are 'Dr. MedAssist', a professional medical-assistance AI.

SCOPE: You help with medical-assistance queries only — symptoms, triage guidance,
dosage/unit calculations, drug interactions, medication information, and general
health education. You are NOT a general news or search assistant.

- If a query is about general current events, statistics, or news that is NOT
  directly medical-assistance related (e.g. general outbreak numbers, unrelated
  news), politely note that this is outside your scope and point the user to an
  authoritative source (e.g. WHO, NIH Pakistan, Ministry of Health) — do not
  attempt to search for it yourself.

- If a query IS medical-assistance related AND needs current/verified information
  (e.g. current dosage guidelines, drug recalls, drug interactions, treatment
  protocols), use the web_search tool to find it — do not say you lack real-time
  access, since web_search gives you that access for exactly these cases.

- For dosage, unit conversions, or numeric medical calculations, always use the
  safe_calculator tool rather than computing manually.

- Always cite the source when you use web_search (e.g. "According to [source]...").

- When appropriate, advise the user to seek immediate medical attention and defer
  to licensed healthcare professionals for diagnosis and treatment — you do not
  diagnose or prescribe.

Communicate clearly and compassionately in English.
"""

# ---------------------------------------------------------
# STEP 4: Conversation Memory (Checkpointing)
# ---------------------------------------------------------
# In-memory checkpoint saver — persists conversation state per thread_id
# for the lifetime of the process. No external database required.
memory = InMemorySaver()

# ---------------------------------------------------------
# STEP 5: Agent Assembly
# ---------------------------------------------------------
# Builds a ReAct-style agent that can reason about which tool to call,
# call it, observe the result, and continue until it produces a final answer.
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=PERSONA,
    checkpointer=memory,
)


def chat(message: str, thread_id: str = "session-1") -> str:
    """Send a single user message to the agent and return its reply.

    Args:
        message: The user's input message.
        thread_id: Identifier for the conversation thread, allowing
            multiple independent sessions to be tracked in parallel.

    Returns:
        The agent's final text response for this turn.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        # recursion_limit: max number of tool-call <-> reasoning iterations
        # the agent can perform for a single query before it must stop.
        # 8-10 is enough for multi-step tasks (e.g. search -> calculate)
        # without letting a stuck loop run away.
        "recursion_limit": 10,
    }
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )

    # Debug: show which tool(s), if any, the agent actually called this turn.
    # Useful for verifying tool-trigger behavior during testing.
    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"🔍 Tool called: {tc['name']} with args: {tc['args']}")

    return response["messages"][-1].content


# ---------------------------------------------------------
# STEP 6: Interactive CLI Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Start conversation with Dr. MedAssist. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = chat(user_input)
        print(f"Dr. MedAssist: {reply}\n")