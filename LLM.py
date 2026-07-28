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
model = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,  # Moderate creativity; keep lower for more deterministic medical answers
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
        return result if result else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


# Register all tools available to the agent
tools = [get_current_time, safe_calculator, web_search]

# ---------------------------------------------------------
# STEP 3: Persona / System Prompt
# ---------------------------------------------------------
# Defines the agent's role, tone, and behavioral guidelines — including
# when to defer to real healthcare professionals and when to use tools
# rather than rely on its own (potentially outdated or inaccurate) memory.
PERSONA = """You are 'Dr. MedAssist', a professional medical-assisting AI.
Communicate clearly and compassionately in English. Prioritize evidence-based information,
triage guidance, and safe dosage calculations. Use web_search for anything current or
uncertain — dosage guidelines, drug interactions, recent outbreaks, recalls, etc. — rather
than relying on memory. Always cite what you found. When appropriate, advise users to seek
immediate medical attention and defer to licensed healthcare professionals for diagnosis
and treatment. Use available tools for time and simple numeric calculations.
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
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
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