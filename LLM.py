"""
Dr. MedAssist - MCP Server
"""

import os
import sys
import asyncio

# ---------------------------------------------------------
# WINDOWS FIX: default asyncio event loop on Windows doesn't
# handle stdio pipes reliably -> FastMCP stdio server hangs
# after "initialize" and never responds. ProactorEventLoop
# fixes this. Must be set before anything else touches asyncio.
# ---------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent  # LangGraph v1.0 / LangChain agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------
# Logging helper - write diagnostics to stderr (stdout is reserved for stdio MCP)
# ---------------------------------------------------------
def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def ensure_required_env() -> None:
    missing = [
        name
        for name in ("OPENROUTER_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        log(f"Missing required environment variables: {', '.join(missing)}")
        raise RuntimeError("Required environment variables are missing")
    log("Required environment variables loaded successfully")


ensure_required_env()

# ---------------------------------------------------------
# STEP 1: Vector Store & Model Setup
# ---------------------------------------------------------
log("Loading FastEmbed Model & Local FAISS Index...")
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
INDEX_PATH = str(BASE_DIR / "faiss_mplus_health_topics")

try:
    vector_store = FAISS.load_local(
        INDEX_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    log(f"FAISS index loaded successfully from {INDEX_PATH}")
except Exception as e:
    log(f"FAISS loading failed: {e}")
    vector_store = None

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
# STEP 2: Agent Tools Definitions
# ---------------------------------------------------------
@tool
def get_current_time() -> str:
    """Return the current date and time as an ISO-formatted string."""
    log("Tool invoked: get_current_time")
    result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"get_current_time result: {result}")
    return result

@tool
def safe_calculator(expression: str) -> str:
    """Evaluate a basic numeric expression safely (for dosage and unit calculations)."""
    log(f"Tool invoked: safe_calculator - expression={expression}")
    import numexpr
    try:
        result = numexpr.evaluate(expression).item()
        log(f"safe_calculator result: {result}")
        return f"Result: {result}"
    except Exception as e:
        log(f"safe_calculator error: {e}")
        return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the internet for up-to-date medical information (via Tavily)."""
    log(f"Tool invoked: web_search — query={query}")
    try:
        search = TavilySearch(max_results=3, api_key=os.getenv("TAVILY_API_KEY"))
        result = search.invoke(query)
        log(f"web_search returned result (length={len(str(result))} chars)")
        return str(result)
    except Exception as e:
        log(f"web_search error: {e}")
        return f"Search failed: {e}"

@tool
def search_medical_knowledge(query: str, k: int = 2) -> str:
    """
    Searches FAISS vector store and returns complete vector pipeline steps:
    Query Vector, Raw DB Docs, Embeddings, Similarity Scores, and Top-K Matches.
    """
    if vector_store is None:
        return "Medical Knowledge Base index is not available."

    try:
        # 1. Generate Query Vector Embedding
        query_embedding = embedding_model.embed_query(query)
        query_vector_sample = [round(float(val), 4) for val in query_embedding[:8]]
        
        # 2. Perform Cosine Similarity / Distance Search in FAISS
        # Returns list of tuples: (Document, Score)
        results_with_scores = vector_store.similarity_search_with_score(query, k=k)
        
        if not results_with_scores:
            return "No matching medical records found in knowledge base."

        pipeline_output = []
        pipeline_output.append("=== ⚙️ INTERNAL RAG VECTOR PIPELINE EXECUTION ===\n")
        
        # PIPELINE STEP 1: INPUT & EMBEDDING
        pipeline_output.append(f"1️⃣ USER INPUT QUERY: '{query}'")
        pipeline_output.append(
            f"   └─► Query Vector Embedding (Dim={len(query_embedding)}, Sample First 8 Dimensions): {query_vector_sample}...\n"
        )
        
        # PIPELINE STEP 2 & 3: COSINE SIMILARITY & TOP-K RETRIEVAL
        pipeline_output.append(f"2️⃣ FAISS VECTOR MATCHING & COSINE SIMILARITY (Top-{k} Matches):")
        
        for idx, (doc, score) in enumerate(results_with_scores, 1):
            condition = doc.metadata.get("condition", "N/A")
            source = doc.metadata.get("source", "MedlinePlus")
            content_snippet = doc.page_content.replace("\n", " ")[:150]
            
            # Retrieve embedding sample for the stored document
            doc_embedding = embedding_model.embed_query(doc.page_content)
            doc_vector_sample = [round(float(val), 4) for val in doc_embedding[:8]]
            
            doc_block = (
                f"\n   --- [TOP-{idx} RETRIEVED DOCUMENT] ---\n"
                f"   • Condition/Focus: {condition}\n"
                f"   • Source Database: {source}\n"
                f"   • Vector Distance/Score: {score:.4f} (Lower = Higher Similarity)\n"
                f"   • Stored Doc Vector Sample: {doc_vector_sample}...\n"
                f"   • Raw Content Snippet:\n     \"{content_snippet}...\""
            )
            pipeline_output.append(doc_block)

        final_text = "\n".join(pipeline_output)
        
        # Terminal Log Output
        log(f"[RAG PIPELINE] Executed query '{query}' | Top-{k} Scores: {[round(s,4) for _, s in results_with_scores]}")        
        return final_text

    except Exception as e:
        return f"Error executing FAISS similarity search pipeline: {str(e)}"
# Registering Tools for Agent
tools = [get_current_time, search_medical_knowledge, safe_calculator, web_search]

# ---------------------------------------------------------
# STEP 3: Persona
# ---------------------------------------------------------
PERSONA = """You are 'Dr. MedAssist', a professional medical-assistance AI.

SCOPE: Assist only with medical-assistance related queries such as symptoms,
triage guidance, dosage and unit calculations, drug interactions, medication
information, and general health education. Do not act as a general news or
non-medical search assistant.

- ALWAYS use the `search_medical_knowledge` tool FIRST whenever the user asks about medical topics, symptoms, health conditions, or treatments.
- If current or verified external medical information is required and not found in local DB, use the `web_search` tool and cite sources in the response.
- If a query concerns general non-medical events, state that it is outside the scope.
- For dosage, unit conversions, and numeric medical calculations, use the `safe_calculator` tool rather than manual computation.
- When appropriate, advise users to seek immediate medical attention and defer diagnosis or prescribing to licensed healthcare professionals.
- Always prioritize patient safety and well-being in all interactions.

STRICT TOOL USAGE RULES:
1. ALWAYS call `search_medical_knowledge` FIRST for any user query related to health, medical conditions, symptoms, or search requests.
2. Even if the user query is slightly unclear or contains typos (e.g. "FIASS", "ALT USE"), extract the core medical/search keywords and execute the `search_medical_knowledge` tool.
3. IN YOUR FINAL RESPONSE: You MUST start or end your response with a clear badge showing which tools were executed, like this:
   `[Tool Used: search_medical_knowledge]` or `[Tool Used: None / Direct Answer]

Communicate clearly, concisely, and compassionately in English.
"""

# ---------------------------------------------------------
# STEP 4: Memory & Agent Initialization
# ---------------------------------------------------------
memory = InMemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=PERSONA,
    checkpointer=memory,
)

def chat(message: str, thread_id: str = "session-1") -> str:
    """Send a single user message to the agent and return its reply."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )

    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                log(f"Agent performed internal tool call: {tc['name']} | args: {tc['args']}")

    return response["messages"][-1].content

# ---------------------------------------------------------
# STEP 5: FastMCP Server Definitions
# ---------------------------------------------------------
mcp_server = FastMCP("Dr-MedAssist")

@mcp_server.tool()
def ask_medassist(message: str, thread_id: str = "default-session") -> str:
    """Ask Dr. MedAssist a medical question."""
    log(f"[ask_medassist] Incoming request: {message!r}")
    result = chat(message, thread_id=thread_id)
    log(f"[ask_medassist] Sending response (length={len(result)} chars)")
    return result

@mcp_server.tool()
def check_drug_interaction(drug_a: str, drug_b: str) -> str:
    """Assess potential interactions between two medications."""
    log(f"[check_drug_interaction] drugs={drug_a!r}, {drug_b!r}")
    query = (
        f"Is there a known drug interaction between {drug_a} and {drug_b}? "
        f"Give severity (mild/moderate/severe) and a brief explanation, with source."
    )
    result = chat(query, thread_id=f"interaction-{drug_a}-{drug_b}")
    log(f"[check_drug_interaction] Prepared response (length={len(result)} chars)")
    return result

@mcp_server.tool()
def bmi_calculator(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI from weight and height."""
    log(f"[bmi_calculator] weight={weight_kg} kg, height={height_cm} cm")
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    result = f"BMI: {bmi} - Category: {category}"
    log(f"[bmi_calculator] result: {result}")
    return result

@mcp_server.tool()
def emergency_symptom_checker(symptoms: str) -> str:
    """Perform a concise triage assessment from symptoms."""
    log(f"[emergency_symptom_checker] symptoms={symptoms!r}")
    query = (
        f"A person reports these symptoms: {symptoms}. "
        f"Classify urgency as EMERGENCY, URGENT, or NON-URGENT, and give one-line next steps. "
        f"Be direct and brief."
    )
    result = chat(query, thread_id=f"triage-{hash(symptoms) % 10000}")
    log(f"[emergency_symptom_checker] Prepared response (length={len(result)} chars)")
    return result

@mcp_server.tool()
def get_medassist_history(thread_id: str = "default-session") -> str:
    """Retrieve the full conversation history for a specific thread."""
    log(f"[get_medassist_history] Request for history: thread={thread_id}")
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)

    if "messages" in state.values and state.values["messages"]:
        return "\n".join(
            f"[{msg.type}]: {msg.content}" for msg in state.values["messages"]
        )
    return "No conversation found for this thread."

@mcp_server.tool()
def search_medical_db_mcp(query: str) -> str:
    """Direct MCP endpoint to query MedlinePlus/MedQuAD vector database."""
    return search_medical_knowledge.invoke({"query": query})

# ---------------------------------------------------------
# STEP 6: Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    log("Dr. MedAssist MCP server starting...")
    mcp_server.run(transport="stdio")