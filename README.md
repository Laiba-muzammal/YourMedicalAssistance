# Dr. MedAssist 🩺

A conversational medical-assistance AI agent built with **LangChain** and **LangGraph**, exposed as an **MCP (Model Context Protocol) server** for use with Claude Desktop — featuring live internet access, a local medical knowledge base (RAG), safe dosage calculations, drug-interaction checks, BMI calculation, symptom triage, **automated medicine-availability tracking with email alerts**, and persistent per-session conversation memory.

> ⚠️ **Disclaimer:** Dr. MedAssist is an informational assistant only. It does **not** provide medical diagnoses or treatment plans, and is not a substitute for a licensed healthcare professional. Always consult a doctor for medical decisions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Medicine-Tracking Workflow (LangGraph)](#medicine-tracking-workflow-langgraph)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Exposed MCP Tools](#exposed-mcp-tools)
- [Example Usage](#example-usage-from-claude-desktop)
- [Screenshots](#screenshots)
- [Logging](#logging)
- [Security Notes](#security-notes)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

Dr. MedAssist is a ReAct-style AI agent (built with LangChain's `create_agent`) that reasons about which tool to use, calls it, and responds with grounded, up-to-date information. It is packaged as a **single-file MCP server** (`LLM.py`), so it can be plugged directly into Claude Desktop and called like any other native tool. A companion **LangGraph state-machine workflow** handles a second, independent capability: tracking medicine availability on pharmacy websites and notifying the user by email — driven either on-demand (via chat) or automatically in the background (via a scheduler).

It's designed to assist with:

- Answering general medical questions using evidence-based reasoning
- **Retrieving grounded answers from a local medical knowledge base** (RAG over a FAISS vector store built from NIH's MedQuAD/MedlinePlus dataset), with full visibility into the retrieval pipeline (query embedding, similarity scores, top-k matches)
- Looking up **current** medical information via live web search (drug interactions, guidelines, recalls, outbreaks)
- Performing **safe** dosage and unit calculations
- Checking interactions between two medications
- Calculating BMI and its category
- Performing quick symptom-based urgency triage (EMERGENCY / URGENT / NON-URGENT)
- **Tracking whether a specific medicine is in stock at a given pharmacy website/URL**, and automatically emailing the user when it becomes available (or every 12 hours as a reminder while still unavailable)
- Maintaining conversation context across a session (per `thread_id`)

---

## Features

| Feature | Description |
|---|---|
| 🔌 **MCP Server** | Runs as an MCP-compatible server (`FastMCP`) over stdio, so Claude Desktop can call it as a native tool |
| 📚 **RAG / Vector Knowledge Base** | Local **FAISS** vector store built from NIH's **MedQuAD/MedlinePlus** dataset; the agent retrieves relevant context before answering, and the pipeline (query embedding → similarity search → top-k docs) is transparent rather than a black box |
| 🌐 **Web Search** | Uses **Tavily** (API key required) to fetch current medical information beyond the model's training data and the local knowledge base |
| 🧮 **Safe Calculator** | Numeric-only expression evaluation via `numexpr` — no `eval()`, no code-execution risk |
| 🕒 **Time Tool** | Returns current date/time for time-relative queries |
| 💊 **Drug Interaction Checker** | Dedicated tool to assess interaction severity between two medications |
| ⚖️ **BMI Calculator** | Deterministic Python calculation (no LLM call) — returns BMI value and category |
| 🚨 **Symptom Triage** | Classifies reported symptoms as EMERGENCY / URGENT / NON-URGENT with next steps |
| 📦 **Medicine Availability Tracking** | A **LangGraph state machine** checks a pharmacy URL for a medicine, auto-resolves a product link via search if only a website/name is given, persists pending trackers in SQLite, and emails the user when found — with 12-hourly reminder emails via a background scheduler until then |
| 💾 **Session Memory** | In-memory checkpointing (per `thread_id`) so the agent remembers context within a conversation, retrievable via a dedicated history tool |
| 🤖 **ReAct Agent** | Built on LangChain's `create_agent` — the model autonomously decides which internal tool to call and when (knowledge base first, web search if needed) |

---

## Tech Stack

- **[LangChain](https://python.langchain.com/)** — tool definitions, `create_agent`, and LLM wrapper
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — powers the medicine-tracking `StateGraph` (`tracker_graph.py`) and the agent's checkpointing
- **[OpenRouter](https://openrouter.ai/)** — LLM API gateway (using `meta-llama/llama-3.3-70b-instruct`)
- **[FAISS](https://github.com/facebookresearch/faiss)** — local vector similarity search for the medical knowledge base
- **[MedQuAD](https://github.com/abachaa/MedQuAD)** — NIH-sourced medical Q&A dataset used to build the knowledge base
- **HuggingFace / FastEmbed embeddings** (`BAAI/bge-small-en-v1.5`) — converts text into vectors for FAISS
- **[numexpr](https://github.com/pydata/numexpr)** — safe numeric expression evaluation
- **[langchain-tavily](https://pypi.org/project/langchain-tavily/)** — web search via the Tavily API
- **[mcp](https://pypi.org/project/mcp/)** (`FastMCP`) — exposes the agent as an MCP server for Claude Desktop
- **[APScheduler](https://apscheduler.readthedocs.io/)** (`BlockingScheduler`) — drives the 12-hourly background recheck job in `scheduler.py`
- **SQLite** (`medicine_tracker.db`) — persists pending medicine-tracking records across restarts
- **`smtplib` (Gmail)** — sends availability/reminder emails from `tracker_utils.py`

---

## Architecture

```
                         ┌─────────────────────┐
                         │   Claude Desktop      │
                         │   (MCP Client)         │
                         └──────────┬───────────┘
                                    │ stdio (MCP protocol)
                                    ▼
                         ┌─────────────────────┐
                         │       LLM.py           │
                         │   FastMCP Server        │
                         │   (7 exposed tools)      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌───────────────────────────┐
                     │   LangChain Agent          │
                     │   (create_agent, ReAct)     │
                     └──────────────┬─────────────┘
                                    │
        ┌───────────────┬──────────┼──────────────┬────────────────┐
        ▼               ▼          ▼               ▼                ▼
 ┌────────────┐  ┌────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────────┐
 │ FAISS RAG   │  │ Tavily Web  │ │  Safe    │ │  get_current_ │ │ track_medicine_    │
 │ Knowledge   │  │ Search      │ │ Calculator│ │  time         │ │ availability tool  │
 │ Base        │  │             │ │          │ │               │ │ (kicks off graph)  │
 └────────────┘  └────────────┘ └──────────┘ └──────────────┘ └─────────┬─────────┘
                                                                          │
                                                                          ▼
                                                          ┌───────────────────────────┐
                                                          │   tracker_graph.py          │
                                                          │   LangGraph StateGraph       │
                                                          └──────────────┬─────────────┘
                                                                          │
                                                    ┌─────────────────────┼─────────────────────┐
                                                    ▼                     ▼                     ▼
                                          ┌────────────────┐   ┌────────────────┐   ┌────────────────────┐
                                          │ tracker_db.py   │   │ tracker_utils.py│   │ scheduler.py         │
                                          │ (SQLite:        │   │ (site check +   │   │ (APScheduler,        │
                                          │  pending records)│   │  email sending) │   │  reruns graph every   │
                                          └────────────────┘   └────────────────┘   │  12h for is_recheck)  │
                                                                                     └────────────────────┘
```

The general-purpose agent (RAG → web search → calculator) checks the local knowledge base first for grounded answers, falling back to live web search only when needed. Medicine tracking is a **separate, deterministic LangGraph workflow** — not left to LLM improvisation — so availability checks, persistence, and email notifications behave predictably every time.

---

## Medicine-Tracking Workflow (LangGraph)

The medicine-availability tracker is implemented as its own **LangGraph `StateGraph`** in `tracker_graph.py`. It's the main non-conversational piece of this project — the persona/RAG agent talks to the user, but this graph does the actual repeated-checking-and-notifying work in the background.

**Flow:**
```
START → check_website
             │
     ┌───────┴────────┐
  available?        not available?
     │                    │
 found_node          not_found_node
     │                    │
    END          is_recheck AND 12h passed?
                          │
                 ┌────────┴────────┐
                yes                no
                 │                  │
        send_reminder_node        END
                 │
                END
```

The **same graph** is reused for two different callers:

| Caller | `is_recheck` | Behavior |
|---|---|---|
| `track_medicine_mcp` tool (via chat, in `LLM.py`) | `False` | First-time check. If available → tells the user immediately. If not → saves a **pending record** in SQLite (`add_pending_record`) and tells the user it will keep checking and email them. |
| `scheduler.py` (background job) | `True` | Recheck of an existing pending record. If now available → sends a **"found" email** and marks the record closed (`mark_found`). If still not available and **12+ hours** have passed since the last reminder → sends a **reminder email** (`should_send_reminder` / `mark_reminder_sent`). Otherwise just logs the attempt (`record_check_attempt`). |

**Key design details:**

- **URL resolution with fallback** — `track_medicine_availability` (the internal `@tool`) accepts either a direct product URL or just a pharmacy/site name. If given a name, it uses `web_search` with a `site:` query to auto-find the product page. If a direct URL check comes back negative or errors out, it retries once via search before giving up, to avoid false "not available" reports from a bad/stale link.
- **Errors never crash the flow** — `check_website_node` treats connection errors as "not available" rather than raising, so a temporarily-down site doesn't break tracking.
- **Persistent, not in-memory** — pending trackers live in `medicine_tracker.db` (SQLite via `tracker_db.py`), so they survive an MCP server or scheduler restart — unlike the chat conversation memory, which is in-memory only.
- **Background automation** — `scheduler.py` runs as a **separate long-running process** (`python scheduler.py`) using **APScheduler's `BlockingScheduler`**, firing `run_all_pending_checks()` once immediately on startup and then every **12 hours**, pulling all pending records (`get_pending_records`) and pushing each through `run_recheck()`.
- **Email delivery** — `tracker_utils.py` sends the "available" and "reminder" emails via Gmail SMTP, using the `EMAIL_SENDER` / `EMAIL_APP_PASSWORD` credentials from `.env`.

---

## Project Structure

```
YourMedicalAssistance/
├── LLM.py                     # MCP server: model, tools, persona, memory, agent, and 7 MCP tool endpoints
├── tracker_graph.py           # LangGraph StateGraph for the medicine-tracking workflow
├── tracker_db.py              # SQLite persistence for pending tracking records (init_db, add/mark/query helpers)
├── tracker_utils.py           # Website availability checks + email sending (Gmail SMTP)
├── scheduler.py               # APScheduler background process — rechecks pending records every 12 hours
├── medicine_tracker.db        # SQLite database file (pending/completed tracking records)
├── parser_data.py             # Parses raw MedQuAD XML files into a single JSON knowledge base
├── build_vector_db.py         # Builds and saves the FAISS vector index from the parsed knowledge base
├── faiss_mplus_health_topics/ # Generated FAISS index (not committed — see .gitignore)
├── MedQuAD/                   # Raw dataset, cloned separately (not committed — see .gitignore)
├── backups/                   # Backup copies of source files
├── requirements.txt           # Python dependencies
├── .env                       # API keys & email credentials (not committed — see .gitignore)
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
pip install -r requirements.txt
```

Or, if installing manually:
```bash
pip install mcp langchain langchain-openai langgraph langchain-tavily python-dotenv numexpr faiss-cpu langchain-community langchain-huggingface apscheduler
pip freeze > requirements.txt
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
- `OPENROUTER_API_KEY` and `TAVILY_API_KEY` are required at startup — `LLM.py` will raise an error if either is missing.

### 5. Build the medical knowledge base (one-time setup)

Clone the MedQuAD dataset and generate the FAISS index:
```bash
git clone https://github.com/abachaa/MedQuAD.git
python parser_data.py         # -> creates the parsed knowledge base JSON
python build_vector_db.py     # -> creates faiss_mplus_health_topics/
```
This only needs to be run once. `LLM.py` loads the saved index at startup rather than rebuilding it every time.

### 6. Register the MCP server with Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` (on Windows) and add:

```json
{
    "mcpServers": {
        "dr-medassist": {
            "command": "python",
            "args": ["C:/path/to/YourMedicalAssistance/LLM.py"]
        }
    }
}
```

Restart Claude Desktop. The tools listed below will appear as available tools that Claude can call.

> **Windows note:** `LLM.py` explicitly sets `asyncio.WindowsProactorEventLoopPolicy()` at startup. The default Windows event loop doesn't reliably handle stdio pipes, which causes the FastMCP stdio server to hang after the `initialize` handshake and never respond — this fix must run before anything else touches `asyncio`.

### 7. Run the background tracker scheduler (separate process)

For medicine-tracking reminders/emails to actually fire in the background, `scheduler.py` needs to run continuously, **separately** from the MCP server:
```bash
python scheduler.py
```
This starts an APScheduler `BlockingScheduler` that rechecks all pending records once on startup, then every 12 hours. Leave this running (e.g. in its own terminal, or as a background service) alongside Claude Desktop.

### 8. (Optional) Run the MCP server standalone for testing
```bash
python LLM.py
```
This starts the MCP server over stdio. It's intended to be launched by Claude Desktop rather than run interactively — there is no CLI chat loop in this version.

---

## Exposed MCP Tools

| Tool | Purpose |
|---|---|
| `ask_medassist(message, thread_id)` | General-purpose entry point for any medical question — agent decides whether to use the knowledge base, web search, or calculator |
| `check_drug_interaction(drug_a, drug_b)` | Assesses interaction severity between two named medications |
| `bmi_calculator(weight_kg, height_cm)` | Computes BMI and category directly (no LLM call) |
| `emergency_symptom_checker(symptoms)` | Classifies symptoms as EMERGENCY / URGENT / NON-URGENT with next steps |
| `get_medassist_history(thread_id)` | Retrieves the full conversation history for a given thread |
| `search_medical_db_mcp(query)` | Direct endpoint to query the FAISS/MedQuAD knowledge base, bypassing the agent |
| `track_medicine_mcp(medicine_name, url, recipient_email)` | Checks medicine availability now; if unavailable, starts background tracking via the LangGraph workflow and emails updates |

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

User: Let me know when Panadol is back in stock at dvago.pk, email me at me@example.com
→ calls track_medicine_mcp("Panadol", "dvago.pk", "me@example.com")
→ if unavailable now: "I'll keep checking in the background and email you when it's available
   (or a reminder every 12 hours)."
→ scheduler.py silently rechecks every 12h and emails when found
```

---

## Screenshots

> 📸 *Add your own screenshots below by placing image files in a `screenshots/` folder in the repo root and updating the paths.*

**1. `dr-medassist` connected as an MCP tool in Claude Desktop**
```markdown
![MCP server connected](./screenshots/mcp-connected.png)
```

**2. `ask_medassist` — general medical query in action**
```markdown
![Ask MedAssist demo](./screenshots/ask-medassist-demo.png)
```

**3. `track_medicine_mcp` — tracking started + confirmation email**
```markdown
![Medicine tracking demo](./screenshots/track-medicine-demo.png)
```

**4. `emergency_symptom_checker` — triage classification**
```markdown
![Symptom triage demo](./screenshots/symptom-triage-demo.png)
```

Once you have real screenshots, replace each `![...](...)` code block above with the actual image markdown (remove the surrounding ```` ``` ```` fences) so they render inline.

---

## Logging

Since the MCP stdio transport reserves `stdout` for protocol frames, `LLM.py` never uses `print()` for diagnostics. Instead, a `log()` helper writes timestamped messages to `stderr`, which Claude Desktop captures at:

```
%APPDATA%\Claude\logs\mcp-server-dr-medassist.log
```

`scheduler.py` uses the same `stderr`-based `log()` pattern, printed to its own terminal/process output since it doesn't run under the MCP stdio transport.

Check these logs to confirm both processes started correctly and to see tool invocations, RAG retrieval calls, and scheduler recheck ticks.

---

## Security Notes

- The calculator tool uses `numexpr` instead of Python's built-in `eval()` to prevent arbitrary code execution from malicious or malformed input — an important safeguard given the sensitivity of dosage-related calculations.
- API keys and email credentials are loaded from a `.env` file and excluded from version control via `.gitignore`. Never commit real API keys or app passwords.
- The raw MedQuAD dataset and the generated FAISS index are excluded from version control (see `.gitignore`) — they are large and fully reproducible via `parser_data.py` and `build_vector_db.py`.
- The agent's persona restricts it to medical-assistance topics only; non-medical queries (general news, unrelated statistics) are declined and redirected to authoritative sources (e.g. WHO, NIH).
- The persona explicitly requires **both** a URL and a recipient email before invoking the tracking tool — it won't silently start tracking or emailing without explicit user-provided contact info.

---

## Limitations

- **Conversation memory** is in-memory only (`InMemorySaver`) — it resets when the MCP server process stops.
- **Medicine-tracking records**, by contrast, persist in SQLite (`medicine_tracker.db`) and survive restarts — but the recheck loop only runs while `scheduler.py` is actively running.
- The knowledge base is a static snapshot of MedQuAD/MedlinePlus; it does not update automatically and won't reflect information newer than the dataset itself.
- Web search results are unverified third-party content; always cross-check critical medical information with authoritative sources.
- Requires a Tavily API key (no longer using the free, key-less DuckDuckGo search).
- Medicine-availability checking depends on the target website's HTML being scrapeable/parseable — layout changes on the pharmacy site can affect detection accuracy.
- This project is intended for educational/assistive purposes and has not been clinically validated.

---

## Future Improvements

- [ ] Persistent, database-backed conversation checkpointing (SQLite/Postgres) to match the tracker's persistence
- [ ] Source citation formatting for search and knowledge-base results
- [ ] Structured, database-backed drug-interaction lookup tool
- [ ] Periodic knowledge base refresh/expansion beyond MedQuAD
- [ ] Configurable recheck interval for the scheduler (currently fixed at 12 hours)
- [ ] Web-based chat UI (Streamlit/Gradio) as an alternative front-end to the MCP interface

---

## Contributing

This is currently a personal learning/portfolio project. Issues and suggestions are welcome via GitHub Issues on the repository. If you'd like to propose a change, please open an issue first to discuss what you'd like to modify.

---

## License

This project is available for educational and non-commercial use. If you plan to reuse or build on it, please credit the original author. *(Add a formal license file, e.g. MIT, if you intend to open-source this more broadly.)*

---

## Author

**Laiba** — Software Engineering Student, UET Taxila

---