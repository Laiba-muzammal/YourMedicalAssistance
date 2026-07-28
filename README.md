# Dr. MedAssist 🩺

A conversational medical-assistance AI agent built with **LangChain** and **LangGraph**, featuring live internet access, safe dosage calculations, and persistent conversation memory.

> ⚠️ **Disclaimer:** Dr. MedAssist is an informational assistant only. It does **not** provide medical diagnoses or treatment plans, and is not a substitute for a licensed healthcare professional. Always consult a doctor for medical decisions.

---

## Overview

Dr. MedAssist is a ReAct-style AI agent that can reason about which tool to use, call it, and respond with grounded, up-to-date information. It's designed to assist with:

- Answering general medical questions using evidence-based reasoning
- Looking up **current** medical information via live web search (drug interactions, guidelines, recalls, outbreaks)
- Performing **safe** dosage and unit calculations
- Maintaining conversation context across a session

---

## Features

| Feature | Description |
|---|---|
| 🌐 **Web Search** | Uses DuckDuckGo (free, no API key) to fetch current medical information beyond the model's training data |
| 🧮 **Safe Calculator** | Numeric-only expression evaluation via `numexpr` — no `eval()`, no code-execution risk |
| 🕒 **Time Tool** | Returns current date/time for time-relative queries |
| 💾 **Session Memory** | In-memory checkpointing (per `thread_id`) so the agent remembers context within a conversation |
| 🤖 **ReAct Agent** | Built on LangGraph's `create_react_agent` — the model autonomously decides which tool to call and when |

---

## Tech Stack

- **[LangChain](https://python.langchain.com/)** — tool definitions & LLM wrapper
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — agent orchestration & checkpointing
- **[OpenRouter](https://openrouter.ai/)** — LLM API gateway (using `meta-llama/llama-3.3-70b-instruct`)
- **[numexpr](https://github.com/pydata/numexpr)** — safe numeric expression evaluation
- **[duckduckgo-search](https://pypi.org/project/duckduckgo-search/)** — free web search, no API key required

---

## Project Structure

```
AIAgent/
├── main.py              # Agent definition, tools, and CLI entry point
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed — see .gitignore)
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
pip install langchain langchain-openai langchain-community langgraph python-dotenv numexpr duckduckgo-search
```

Or, if you maintain a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a free API key at [openrouter.ai](https://openrouter.ai/).

### 5. Run the agent
```bash
python main.py
```

You'll see:
```
Start conversation with Dr. MedAssist. Type 'exit' or 'quit' to stop.

You: 
```

---

## Example Usage

```
You: What's the recommended dosage of paracetamol for a 70kg adult?
Dr. MedAssist: [searches web, calculates if needed, responds with guidance
                and a reminder to consult a healthcare professional]

You: Calculate 500mg x 3 doses per day
Dr. MedAssist: Result: 1500 — that totals 1500mg per day...
```

---

## Security Notes

- The calculator tool uses `numexpr` instead of Python's built-in `eval()` to prevent arbitrary code execution from malicious or malformed input — an important safeguard given the sensitivity of dosage-related calculations.
- API keys are loaded from a `.env` file and excluded from version control via `.gitignore`. Never commit real API keys.

---

## Limitations

- Conversation memory is **in-memory only** — it resets when the program stops. For persistent, multi-session memory, swap `InMemorySaver` for a database-backed checkpointer (e.g. SQLite, Postgres).
- Web search results are unverified third-party content; always cross-check critical medical information with authoritative sources.
- This project is intended for educational/assistive purposes and has not been clinically validated.

---

## Future Improvements

- [ ] Persistent checkpointing (SQLite/Postgres)
- [ ] Source citation formatting for search results
- [ ] Structured drug-interaction lookup tool
- [ ] Web-based chat UI (Streamlit/Gradio)

---

## Author

**Laiba** — Software Engineering Student, UET Taxila