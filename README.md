# Intelligent Employee Onboarding Copilot

> Built with LangChain · ChromaDB · Llama3 · sentence-transformers · Ollama

---

## Overview

An Intelligent Employee Onboarding Copilot that reduces manual HR overhead by combining:

- **RAG Pipeline** — answers new hire questions by retrieving information from onboarding documents
- **AI Agent** — executes onboarding tasks autonomously (IT provisioning, email, orientation scheduling)
- **Evaluation Module** — measures RAG quality using faithfulness and answer relevance metrics

---

## Project Structure

```
rag_onboarding/
├── data/
│   ├── docs/                    # Mock onboarding documents (markdown)
│   │   ├── employee_handbook.md
│   │   ├── it_setup_guide.md
│   │   ├── faq.md
│   │   ├── onboarding_checklist.md
│   │   └── department_docs.md
│   ├── chroma_db/               # ChromaDB vector store (auto-generated)
│   └── eval_dataset.json        # 10 Q&A pairs with ground truth
├── src/
│   ├── rag/
│   │   └── pipeline.py          # RAG pipeline (load, chunk, embed, retrieve, generate)
│   ├── agent/
│   │   └── agent.py             # ReAct-style onboarding agent with 4 tools
│   └── evaluation/
│       ├── evaluator.py         # Faithfulness + relevance metrics (LLM-as-judge)
│       └── run_evaluation.py    # Evaluation driver script
├── logs/                        # Structured logs and eval results
├── notebooks/
│   ├── demo.ipynb               # Demo Jupyter notebook
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Llama3 model pulled via Ollama

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag_onboarding
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install langchain langchain-community langchain-ollama langchain-text-splitters
pip install chromadb sentence-transformers python-dotenv pypdf tiktoken
pip install "unstructured[md]" jupyter
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your values (see `.env.example` for reference).

### 5. Pull the LLM model via Ollama

```bash
ollama pull llama3
```

Verify Ollama is running:
```bash
ollama list
```

---

## Running the Project

### Task 1 — RAG Pipeline

```bash
python3 src/rag/pipeline.py
```

This will:
- Load 5 onboarding documents from `data/docs/`
- Chunk them using RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
- Embed and store in ChromaDB
- Answer 5 test questions including one out-of-scope query

### Task 2 — AI Agent

```bash
python3 src/agent/agent.py
```

This will:
- Run the ReAct-style agent on a full onboarding scenario
- Execute 4 tools: IT ticket, welcome email, orientation, status update
- Print step-by-step reasoning and actions
- Display the final onboarding database state

### Task 3 — Evaluation

```bash
python3 src/evaluation/run_evaluation.py
```

This will:
- Load 10 Q&A pairs from `data/eval_dataset.json`
- Generate answers using the RAG pipeline
- Score each answer on faithfulness and answer relevance
- Print a full evaluation report
- Save results to `logs/eval_results.json`

---

## Environment Variables

```bash
# .env.example

# LLM Model (default: llama3 via Ollama — free, local)
LLM_MODEL=llama3

# Embedding model (default: sentence-transformers — free, local)
EMBEDDING_MODEL=all-MiniLM-L6-v2

LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=rag-onboarding
```

---

## Evaluation Results

| Metric | Score |
|---|---|
| Avg Faithfulness | 1.000 / 1.000 |
| Avg Answer Relevance | 0.830 / 1.000 |
| Avg Overall Score | 0.856 / 1.000 |
| Hallucination Risks | 1 / 10 (intentional out-of-scope) |
| Avg Eval Latency | 7,422 ms / sample |

> Q09 ("Who is the CEO of Apple?") was correctly declined by the system. It was flagged as a hallucination risk by the evaluator because relevance scored 0.0 — this is a known false positive for out-of-scope rejection.

---

## Technology Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | Llama3 via Ollama | Free, local, no API key needed |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, fast, strong semantic similarity |
| Vector Store | ChromaDB | Zero-config local setup |
| Framework | LangChain | Handles both RAG and agent orchestration |
| Evaluation | LLM-as-judge (custom) | No external API, consistent grading |
| Observability | Structured logging | Lightweight, zero dependencies |

---

## System Architecture Design

graph TD
    User([User: New Hire / HR Admin]) -- Natural Language Query --> Router[Orchestration Layer: Router]

    Router -- Q&A Query --> RAG[RAG Pipeline]
    Router -- Action Request --> Agent[Onboarding Agent]

    subgraph "Knowledge Retrieval"
    RAG --> Processing[Embed & Retrieve]
    Processing --> Synthesis[Synthesize with Llama3]
    Synthesis --> VectorStore[(ChromaDB Vector Store)]
    end

    subgraph "Autonomous Action"
    Agent --> ReAct[ReAct Loop: Reason-Act-Observe]
    ReAct --> Tools{Tools}
    Tools --> Tool1[IT Ticket]
    Tools --> Tool2[Email]
    Tools --> Tool3[Calendar]
    Tools --> Tool4[Status Update]
    end

    style User fill:#f9f,stroke:#333,stroke-width:2px
    style Router fill:#bbf,stroke:#333,stroke-width:2px
    style VectorStore fill:#dfd,stroke:#333,stroke-width:2px
---

## Key Design Decisions

**Document Splitting and Chunking Strategy:** Onboarding documents (Employee Handbook, IT Setup Guide, FAQ, Onboarding Checklist, Department Docs) are loaded from the local filesystem using LangChain's TextLoader. Documents are split using RecursiveCharacterTextSplitter with a chunk size of 500 tokens and 50-token overlap. RecursiveCharacterTextSplitter was chosen over fixed-size chunking because it respects natural text boundaries, producing semantically coherent chunks and it is more efficient for PDF documents, keeping in mind that the documents going into the RAG mostly are in PDF format. The 50-token overlap ensures context is preserved across boundaries. Employs cosine similarity to fetch the top 3 semantically relevant chunks. 

**Hallucination prevention:** The RAG prompt(Guardrails) explicitly instructs the LLM to answer only from retrieved context and decline out-of-scope questions with a standardized fallback message. 

**Agent pattern:** ReAct (Reason → Act → Observe) for its interpretability — each step is logged with the model's reasoning, making it easy to debug and audit in production. Accesses four mocked tools: create_it_ticket, send_welcome_email, schedule_orientation, and update_onboarding_status. 

**Mocked tools:** Agent tools are intentionally mocked. Each mock is a drop-in replacement — swapping the body with a real API call (Jira, SendGrid, Google Calendar, HiBob) requires no changes to the agent architecture.


## Trade-Offs

* Local vs API LLM: Using Llama3 via Ollama eliminated API costs but introduced occasional looping behavior in the agent compared to GPT-4o.

* Recursive vs Semantic: Recursive splitting was chosen for efficiency; semantic chunking would improve quality but requires an additional embedding pass.

## Scalability and Production

**Vector Store:** Migrate to Pinecone or Weavite for managed and distributed search.

**Framework Evolution:** To handle 10,000+ documents, we will adopt a hybrid approach: LlamaIndex will manage high-performance data indexing and retrieval , while LangChain will remain the primary orchestrator for complex, multi-step agent actions and tool-calling.

**LLM/Embeddings:** Shift to GPT-4o or Claude APIs for superior tool-calling and higher throughput.

**API Layer:** Wrap logic in FASTAPI service with async endpoints and request queueing.

**Caching:** Add semantic caching (eg, Redis) to reduce costs as it will serve repeat queries.

**Hallucinations:** In production, scores below 0.5 would trigger an automatic escalation to an HR coordinator using Slack. 

**Security:** Implement PII detection, role-based access control and audit logging.

---

## AI Tools Used

This project was built with assistance from Claude (Anthropic) as a coding assistant for scaffolding, debugging, and code review. All code has been reviewed by the author.

---

## Notes

- ChromaDB vector store is persisted to `data/chroma_db/` after first run — subsequent runs load from disk
- All agent tool calls are logged to `logs/agent.log`
- Evaluation logs are saved to `logs/evaluation.log` and `logs/eval_results.json`
- Never commit `.env` — use `.env.example` as the template