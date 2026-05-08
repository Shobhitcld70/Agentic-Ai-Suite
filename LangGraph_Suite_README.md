# 🤖 LangGraph Agentic AI Suite

[![Python](https://img.shields.io/badge/Python-3.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Google_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![LangSmith](https://img.shields.io/badge/LangSmith-LLMOps-FF6B2B?style=for-the-badge&logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)

> A production-grade suite of **LangGraph-powered agentic AI systems** — covering short-term and long-term memory, tool-calling agents, Human-in-the-Loop workflows, MCP integration, multi-agent orchestration with parallel execution, and full LLMOps observability via LangSmith. All frontends built with Streamlit.

---

## 📦 What's Inside

| Module | Description |
|---|---|
| `langgraph_backend.py` | Basic chatbot with STM via `MemorySaver` |
| `langgraph_database_backend.py` | Persistent chatbot with LTM via `SqliteSaver` |
| `langgraph_tool_backend.py` | Tool-calling agent: DuckDuckGo + Stock API + Calculator |
| `langgraph_mcp_backend.py` | MCP-integrated async agent with `MultiServerMCPClient` |
| `chatbot_with_hitl.py` | Human-in-the-Loop agent with `interrupt()` + `Command(resume=)` |
| `chatbot_async.py` | Async tool-calling agent with `ainvoke` |
| `bwa_basic.ipynb` | Blog Writer Agent — Orchestrator-Worker-Reducer pattern |
| `bwa_improved_prompting.ipynb` | Blog Writer Agent — structured planning with Pydantic |
| `bwa_research.ipynb` | Blog Writer Agent — with routing + live Tavily research |
| `docker-compose.yml` | PostgreSQL 16 container for production LTM |

---

## 🧠 Memory Architecture

One of the core design goals of this suite is implementing a **two-tier memory system** that mirrors how production AI assistants work.

```
┌─────────────────────────────────────────────────────┐
│                   MEMORY TIERS                       │
│                                                      │
│  STM — Short-Term Memory                            │
│  ┌─────────────────────────────────────────────┐    │
│  │  MemorySaver (in-process)                   │    │
│  │  • Lives in RAM for current session         │    │
│  │  • Message trimming (sliding window)        │    │
│  │  • Summarization for long contexts          │    │
│  │  • Lost when process restarts               │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  LTM — Long-Term Memory                             │
│  ┌─────────────────────────────────────────────┐    │
│  │  SqliteSaver / AsyncSqliteSaver             │    │
│  │  • Persists across sessions on disk         │    │
│  │  • PostgreSQL (Docker) for production       │    │
│  │  • UUID-based thread isolation              │    │
│  │  • Retrieve all past threads                │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### UUID-based Multi-threading
Every conversation is assigned a unique `thread_id` (UUID). This isolates conversation contexts — multiple users or sessions can run simultaneously without state bleeding between threads.

```python
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = chatbot.invoke(state, config=config)
```

---

## 🔧 Modules — Deep Dive

### 1. Basic Chatbot — Short-Term Memory
**File:** `langgraph_backend.py` + `streamlit_frontend.py`

LangGraph `StateGraph` with `MemorySaver` checkpointer. Messages accumulate within a session using the `add_messages` reducer. Clean baseline for understanding graph-based conversation state.

```
User → chat_node (Gemini 2.5 Flash) → Response
         ↑____________MemorySaver_______↑
```

---

### 2. Persistent Chatbot — Long-Term Memory
**File:** `langgraph_database_backend.py` + `streamlit_database_frontend.py`

Replaces `MemorySaver` with `SqliteSaver` backed by `chatbot.db`. Conversations survive restarts. The `retrieve_all_threads()` function lists all past sessions by scanning checkpointer state — the Streamlit sidebar displays them as a conversation history panel.

```python
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
```

For production: swap SQLite with the **PostgreSQL 16 container** (Docker Compose):
```bash
docker-compose up   # Starts postgres on port 5442
```

---

### 3. Tool-Calling Agent
**File:** `langgraph_tool_backend.py` + `streamlit_tool_frontend.py`

LLM decides autonomously whether to answer directly or invoke one of three tools. Uses `ToolNode` + `tools_condition` for the routing loop.

**Tools:**
- 🔍 **DuckDuckGoSearchRun** — live web search
- 📈 **get_stock_price** — Alpha Vantage API (real-time stock quotes)
- 🧮 **calculator** — arithmetic (add, sub, mul, div) with error handling

```
User → chat_node → [tool needed?]
                        │ Yes          │ No
                        ▼             ▼
                   tool_node        END
                        │
                        └→ chat_node (loop)
```

---

### 4. Human-in-the-Loop (HITL) Agent
**File:** `chatbot_with_hitl.py`

The most nuanced pattern in this suite. The `purchase_stock` tool uses LangGraph's `interrupt()` to **pause graph execution** and surface a human approval prompt before any irreversible action is taken.

```python
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    if decision.lower() == "yes":
        return {"status": "success", ...}
    return {"status": "cancelled", ...}
```

After the human responds, execution resumes via `Command(resume=decision)` — the graph picks up exactly where it paused with full state intact.

**Flow:**
```
User: "Buy 10 shares of AAPL"
    → chat_node → tool_node → interrupt() ← PAUSED
                                    ↓
                           Human: "yes" / "no"
                                    ↓
                           Command(resume=decision)
                                    ↓
                           tool_node completes → chat_node → Response
```

---

### 5. MCP-Integrated Async Agent
**File:** `langgraph_mcp_backend.py` + `streamlit_mcp_frontend.py`

Integrates external MCP (Model Context Protocol) servers as tool providers. Uses `MultiServerMCPClient` with two transport modes:

```python
client = MultiServerMCPClient({
    "arith": {
        "transport": "stdio",           # Local MCP server via subprocess
        "command": "python3",
        "args": ["mcp_server.py"],
    },
    "expense": {
        "transport": "streamable_http", # Remote MCP server via HTTP
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    }
})
```

**Threading architecture** — because Streamlit runs synchronously but MCP requires async, a dedicated async event loop runs in a background thread:

```python
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()
```

This allows Streamlit's sync frontend to call async graph operations without blocking or creating new event loops.

---

### 6. Blog Writer Agent (Multi-Agent Orchestrator-Worker-Reducer)
**Files:** `bwa_basic.ipynb`, `bwa_improved_prompting.ipynb`, `bwa_research.ipynb`

Three progressive versions of a multi-agent blog writing system. The core pattern:

```
topic → Orchestrator (plans N sections)
              ↓
         LangGraph Send()  ← fans out N parallel tasks
         /    |    |    \
    Worker  Worker  Worker  Worker   ← write sections in parallel
         \    |    |    /
              ↓
           Reducer (assembles + saves .md file)
```

**Version 1 — Basic (`bwa_basic.ipynb`)**
Simple plan → parallel write → assemble. GPT-4.1-mini.

**Version 2 — Improved Prompting (`bwa_improved_prompting.ipynb`)**
Structured `Task` schema with `goal`, `bullets`, `target_words`, `section_type`. Senior technical writer persona. Word count enforcement (±15%).

**Version 3 — Research Mode (`bwa_research.ipynb`)**
Adds a **Router** node that classifies topics as:
- `closed_book` — no research needed (fundamentals, concepts)
- `hybrid` — some live data needed
- `open_book` — fully research-dependent (news, rankings, latest models)

For research-enabled topics, a **Research** node fires Tavily searches with 3–10 queries before planning, injecting evidence into every worker's context. Workers cite sources. Saves structured `.md` output.

---

## 📊 LangSmith — LLMOps Observability

Every LLM call, tool invocation, and graph transition is traced automatically:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=chatbot-project
```

**What gets traced:**
- Full message history per run
- Tool call inputs and outputs
- Interrupt events and resume decisions (HITL)
- Token usage and latency per node
- Graph execution path visualization
- Error traces with full stack context

View all traces at [smith.langchain.com](https://smith.langchain.com)

---

## 🐳 PostgreSQL — Production LTM

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    ports:
      - "5442:5432"
```

```bash
docker-compose up    # Start PostgreSQL
docker-compose down  # Stop
```

Swap `SqliteSaver` → `PostgresSaver` in `langgraph_database_backend.py` to enable production-grade persistent memory with concurrent access support.

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/Shobhitcld70/LangGraph-Agentic-AI-Suite.git
cd LangGraph-Agentic-AI-Suite
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Environment
Create `.env`:
```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=chatbot-project
```

### 4. Run any frontend
```bash
# Basic chatbot (STM)
streamlit run app/streamlit_frontend.py

# Persistent chatbot (LTM)
streamlit run app/streamlit_database_frontend.py

# Tool-calling agent
streamlit run app/streamlit_tool_frontend.py

# MCP agent
streamlit run app/streamlit_mcp_frontend.py

# HITL agent (terminal)
python chatbot_with_hitl.py
```

### 5. PostgreSQL (optional)
```bash
docker-compose up
```

---

## 📁 Project Structure

```
LangGraph-Agentic-AI-Suite/
├── app/
│   ├── langgraph_backend.py              # STM chatbot backend
│   ├── langgraph_database_backend.py     # LTM chatbot backend (SQLite)
│   ├── langgraph_tool_backend.py         # Tool-calling agent backend
│   ├── langgraph_mcp_backend.py          # MCP async agent backend
│   ├── streamlit_frontend.py             # Basic chatbot UI
│   ├── streamlit_frontend_stream.py      # Streaming responses UI
│   ├── streamlit_frontend_threading.py   # Multi-thread UUID UI
│   ├── streamlit_database_frontend.py    # LTM chatbot UI
│   ├── streamlit_tool_frontend.py        # Tool agent UI
│   └── streamlit_mcp_frontend.py         # MCP agent UI
├── chatbot_with_hitl.py                  # HITL agent (terminal)
├── chatbot_mcp.py                        # Minimal MCP chatbot
├── chatbot_async.py                      # Async tool agent
├── bwa_basic.ipynb                       # Blog Writer v1
├── bwa_improved_prompting.ipynb          # Blog Writer v2
├── bwa_research.ipynb                    # Blog Writer v3 (research)
├── docker-compose.yml                    # PostgreSQL 16 container
├── .env                                  # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```txt
langgraph
langchain
langchain-community
langchain-google-genai
langchain-openai
langchain-mcp-adapters
streamlit
aiosqlite
python-dotenv
pydantic
requests
duckduckgo-search
```

---

## 🔑 Key Design Patterns

**Why LangGraph over plain LangChain?**
LangGraph gives you explicit state management, conditional routing, checkpointing, and the ability to interrupt and resume — none of which are possible with simple chains.

**Why separate STM and LTM?**
STM (MemorySaver) is fast and zero-setup — ideal for single-session demos. LTM (SqliteSaver/PostgreSQL) persists state across restarts — essential for production chatbots where users expect conversation history.

**Why a dedicated async thread for MCP?**
Streamlit's execution model is synchronous. MCP tools require async coroutines. Running a persistent `asyncio` event loop in a daemon thread bridges both worlds cleanly without `asyncio.run()` conflicts.

**Why `Send()` for blog writing?**
Each blog section is independent — there's no reason to write them sequentially. `Send()` fans out N parallel worker tasks, all executing simultaneously, dramatically reducing total generation time compared to sequential writing.

---

## 👤 Author

**Shobhit Krishnan**
- 📧 krishnanshobhit@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/shobhit-krishnan)
- 💻 [GitHub](https://github.com/Shobhitcld70)

---

⭐ Star this repo if you found it useful!
