# Dr. MedAssist 🩺

A conversational medical-assistance AI agent built with **LangChain** and **LangGraph**, exposed as an **MCP (Model Context Protocol) server** for use with Claude Desktop — featuring live internet access, a local medical knowledge base (RAG), safe dosage calculations, drug-interaction checks, BMI calculation, symptom triage, and persistent per-session conversation memory.

> ⚠️ **Disclaimer:** Dr. MedAssist is an informational assistant only. It does **not** provide medical diagnoses or treatment plans, and is not a substitute for a licensed healthcare professional. Always consult a doctor for medical decisions.

---

## Overview

Dr. MedAssist is a ReAct-style AI agent that can reason about which tool to use, call it, and respond with grounded, up-to-date information. Rather than running as a standalone CLI script, it is packaged as a **single-file MCP server**, so it can be plugged directly into Claude Desktop and called like any other native tool.

It's designed to assist with:

- Answering general medical questions using evidence-based reasoning
- **Retrieving grounded answers from a local medical knowledge base** (RAG over a FAISS vector store built from NIH's MedQuAD dataset)
- Looking up **current** medical information via live web search (drug interactions, guidelines, recalls, outbreaks)
- Performing **safe** dosage and unit calculations
- Checking interactions between two medications
- Calculating BMI and its category
- Performing quick symptom-based urgency triage (EMERGENCY / URGENT / NON-URGENT)
- Maintaining conversation context across a session (per `thread_id`)

---

## Features

| Feature | Description |
|---|---|
| 🔌 **MCP Server** | Runs as an MCP-compatible server (`FastMCP`) over stdio, so Claude Desktop can call it as a native tool |
| 📚 **RAG / Vector Knowledge Base** | Local **FAISS** vector store built from NIH's **MedQuAD** dataset (thousands of curated medical Q&A pairs); the agent retrieves relevant context before answering, instead of relying purely on the LLM's parametric knowledge |
| 🌐 **Web Search** | Uses **Tavily** (API key required) to fetch current medical information beyond the model's training data and the local knowledge base |
| 🧮 **Safe Calculator** | Numeric-only expression evaluation via `numexpr` — no `eval()`, no code-execution risk |
| 🕒 **Time Tool** | Returns current date/time for time-relative queries |
| 💊 **Drug Interaction Checker** | Dedicated tool to assess interaction severity between two medications |
| ⚖️ **BMI Calculator** | Deterministic Python calculation (no LLM call) — returns BMI value and category |
| 🚨 **Symptom Triage** | Classifies reported symptoms as EMERGENCY / URGENT / NON-URGENT with next steps |
| 💾 **Session Memory** | In-memory checkpointing (per `thread_id`) so the agent remembers context within a conversation, retrievable via a dedicated history tool |
| 🤖 **ReAct Agent** | Built on LangGraph's `create_react_agent` — the model autonomously decides which internal tool to call and when (knowledge base first, web search if needed) |

---

## Tech Stack

- **[LangChain](https://python.langchain.com/)** — tool definitions & LLM wrapper
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — agent orchestration & checkpointing
- **[OpenRouter](https://openrouter.ai/)** — LLM API gateway (using `meta-llama/llama-3.3-70b-instruct`)
- **[FAISS](https://github.com/facebookresearch/faiss)** — local vector similarity search for the medical knowledge base
- **[MedQuAD](https://github.com/abachaa/MedQuAD)** — NIH-sourced medical Q&A dataset used to build the knowledge base
- **HuggingFace / FastEmbed embeddings** (`BAAI/bge-small-en-v1.5`) — converts text into vectors for FAISS
- **[numexpr](https://github.com/pydata/numexpr)** — safe numeric expression evaluation
- **[langchain-tavily](https://pypi.org/project/langchain-tavily/)** — web search via the Tavily API
- **[mcp](https://pypi.org/project/mcp/)** (`FastMCP`) — exposes the agent as an MCP server for Claude Desktop

---

## Project Structure

```
AIAgent/
├── LLM.py                     # MCP server: model, tools, persona, memory, agent, and MCP tool endpoints
├── parser_data.py             # Parses raw MedQuAD XML files into a single JSON knowledge base
├── build_vector_db.py         # Builds and saves the FAISS vector index from the parsed knowledge base
├── faiss_mplus_health_topics/ # Generated FAISS index (not committed — see .gitignore)
├── MedQuAD/                   # Raw dataset, cloned separately (not committed — see .gitignore)
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed — see .gitignore)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Laiba-muzammal/YourMedicalAssistance.git
cd YourMedicalAssistance
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install mcp langchain langchain-openai langgraph langchain-tavily python-dotenv numexpr faiss-cpu langchain-community langchain-huggingface
```

Or, if you maintain a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
EMAIL_SENDER=your_gmail_address@gmail.com
EMAIL_APP_PASSWORD=your_16_character_gmail_app_password
```

- Get a free OpenRouter key at [openrouter.ai](https://openrouter.ai/).
- Get a free Tavily key at [tavily.com](https://tavily.com/).
- For Gmail sending, enable 2-step verification and create a Gmail App Password; regular Gmail passwords will fail.

### 5. Build the medical knowledge base (one-time setup)

Clone the MedQuAD dataset and generate the FAISS index:
```bash
git clone https://github.com/abachaa/MedQuAD.git
python parser_data.py         # -> creates the parsed knowledge base JSON
python build_vector_db.py     # -> creates faiss_mplus_health_topics/
```
This only needs to be run once. `LLM.py` loads the saved index at startup rather than rebuilding it every time.

### 6. Register the server with Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` (on Windows) and add:

```json
{
    "mcpServers": {
        "dr-medassist": {
            "command": "python",
            "args": ["C:/path/to/your/AIAgent/LLM.py"]
        }
    }
}
```

Restart Claude Desktop. The tools below will appear as available tools that Claude can call.

### 7. (Optional) Run standalone for testing
```bash
python LLM.py
```
This starts the MCP server over stdio. It's intended to be launched by Claude Desktop rather than run interactively — there is no CLI chat loop in this version.

---

## Exposed MCP Tools

| Tool | Purpose |
|---|---|
| `ask_medassist(message, thread_id)` | General-purpose entry point for any medical question — checks the local knowledge base first, then web search if needed |
| `check_drug_interaction(drug_a, drug_b)` | Assesses interaction severity between two named medications |
| `bmi_calculator(weight_kg, height_cm)` | Computes BMI and category directly (no LLM call) |
| `emergency_symptom_checker(symptoms)` | Classifies symptoms as EMERGENCY / URGENT / NON-URGENT with next steps |
| `get_medassist_history(thread_id)` | Retrieves the full conversation history for a given thread |

---

## Example Usage (from Claude Desktop)

```
User: What's the recommended dosage of paracetamol for a 70kg adult?
→ calls ask_medassist → agent checks the local knowledge base, may also use web_search + safe_calculator internally
→ responds with guidance and a reminder to consult a healthcare professional

User: Check interaction between ibuprofen and warfarin
→ calls check_drug_interaction("ibuprofen", "warfarin")
→ returns severity (mild/moderate/severe) with explanation and source

User: My weight is 70kg and height is 175cm, what's my BMI?
→ calls bmi_calculator(70, 175)
→ "BMI: 22.9 — Category: Normal weight"
```

---

## Logging

Since the MCP stdio transport reserves `stdout` for protocol frames, the server never uses `print()` for diagnostics. Instead, a `log()` helper writes timestamped messages to `stderr`, which Claude Desktop captures at:

```
%APPDATA%\Claude\logs\mcp-server-dr-medassist.log
```

Check this file to confirm the server started correctly and to see tool invocation activity, including knowledge-base retrieval calls.

---

## Security Notes

- The calculator tool uses `numexpr` instead of Python's built-in `eval()` to prevent arbitrary code execution from malicious or malformed input — an important safeguard given the sensitivity of dosage-related calculations.
- API keys are loaded from a `.env` file and excluded from version control via `.gitignore`. Never commit real API keys.
- The raw MedQuAD dataset and the generated FAISS index are excluded from version control (see `.gitignore`) — they are large and fully reproducible via `parser_data.py` and `build_vector_db.py`.
- The agent's persona restricts it to medical-assistance topics only; non-medical queries (general news, unrelated statistics) are declined and redirected to authoritative sources (e.g. WHO, NIH).

---

## Limitations

- Conversation memory is **in-memory only** (`InMemorySaver`) — it resets when the server process stops. For persistent, multi-session memory, swap it for a database-backed checkpointer (e.g. SQLite, Postgres).
- The knowledge base is a static snapshot of MedQuAD; it does not update automatically and won't reflect information newer than the dataset itself.
- Web search results are unverified third-party content; always cross-check critical medical information with authoritative sources.
- Requires a Tavily API key (no longer using the free, key-less DuckDuckGo search).
- This project is intended for educational/assistive purposes and has not been clinically validated.

---

## Future Improvements

- [ ] Persistent checkpointing (SQLite/Postgres)
- [ ] Source citation formatting for search and knowledge-base results
- [ ] Structured, database-backed drug-interaction lookup tool
- [ ] Periodic knowledge base refresh/expansion beyond MedQuAD
- [ ] Web-based chat UI (Streamlit/Gradio) as an alternative front-end to the MCP interface

---

## Author

**Laiba** — Software Engineering Student, UET Taxila