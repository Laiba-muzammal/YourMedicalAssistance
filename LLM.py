"""
Simplest LangChain Agent with Built-in In-Memory Storage
========================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables from a .env file (expects OPENROUTER_API_KEY)
load_dotenv()

# ---------------------------------------------------------
# STEP 1: Model Setup (OpenRouter)
# ---------------------------------------------------------
model = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# ---------------------------------------------------------
# STEP 2: Define custom tools
# ---------------------------------------------------------
@tool
def get_current_time() -> str:
    """Return the current date and time as an ISO-formatted string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def simple_calculator(expression: str) -> str:
    """Evaluate a basic numeric expression. Intended for simple dosage or unit calculations.
    Example: '25 * 4' -> 'Result: 100'.
    Note: Uses eval() — only use with trusted input or simple numeric expressions.
    """
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

tools = [get_current_time, simple_calculator]

# ---------------------------------------------------------
# STEP 3: Persona / System Prompt
# ---------------------------------------------------------
PERSONA = """You are 'Dr. MedAssist', a professional medical-assisting AI.
Communicate clearly and compassionately in English. Prioritize evidence-based information,
triage guidance, and safe dosage calculations. When appropriate, advise users to seek
immediate medical attention and defer to licensed healthcare professionals for diagnosis
and treatment. Use available tools for time and simple numeric calculations.
"""

# ---------------------------------------------------------
# STEP 4: Built-in LangChain In-Memory Storage
# ---------------------------------------------------------
# Use an in-memory checkpoint saver for session state (no external database required)
memory = InMemorySaver()

# ---------------------------------------------------------
# STEP 5: Agent assembly
# ---------------------------------------------------------
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=PERSONA,
    checkpointer=memory,  # assign built-in in-memory checkpoint saver
)

# ---------------------------------------------------------
# STEP 6: Interactive loop (simple CLI)
# ---------------------------------------------------------
def chat(message: str, thread_id: str = "session-1"):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return response["messages"][-1].content

if __name__ == "__main__":
    print("Start conversation with Dr. MedAssist. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        
        reply = chat(user_input)
        print(f"Dr. MedAssist: {reply}\n")