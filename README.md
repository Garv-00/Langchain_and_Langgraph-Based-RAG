# RAG Assistant: Production Agentic RAG Pipeline

> A stateful **Agentic RAG system** built with LangGraph, featuring intelligent query routing, semantic reranking, multi-layer security guardrails, and real-time token streaming.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   NeMo Guardrails Gate      │  ← Blocks jailbreaks, off-topic, prompt injections
└────────────┬────────────────┘
             │ (clean query)
             ▼
┌─────────────────────────────┐
│     LangGraph Agent         │
│  ┌────────────────────────┐ │
│  │   Planner Node         │ │  ← Qwen 3.8-27B classifies intent
│  │   (intent routing)     │ │     CONVERSATIONAL vs TECHNICAL
│  └──────────┬─────────────┘ │
│             │               │
│    ┌────────┴────────┐      │
│    ▼                 ▼      │
│  CONVERSATIONAL   TECHNICAL │
│  (memory recall)  (retrieval)│
│                      │      │
│  ┌───────────────────▼───┐  │
│  │   Retriever Node      │  │  ← Qdrant vector search (top 15)
│  │   + FlashRank Rerank  │  │     → FlashRank cross-encoder (top 5)
│  └───────────┬───────────┘  │
│              │              │
│  ┌───────────▼───────────┐  │
│  │   Responder Node      │  │  ← Qwen 3.8-27B synthesizes answer
│  └───────────────────────┘  │
└─────────────────────────────┘
             │
             ▼
    SSE Token Streaming → React UI
```

---

## Key Features

| Feature                 | Implementation                                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Agentic Routing**     | LangGraph planner classifies queries as conversational or technical before retrieval                                               |
| **Semantic Reranking**  | FlashRank cross-encoder reranks top-15 Qdrant results to top-5 for higher precision                                                |
| **Jina Embeddings**     | `jina-embeddings-v3` (1024-dim) for high-quality semantic search                                                                   |
| **Guardrails**          | NeMo Guardrails blocks jailbreaks, off-topic queries, and prompt injections                                                        |
| **Conversation Memory** | LangGraph `MemorySaver` with thread IDs enables multi-turn conversations                                                           |
| **SSE Streaming**       | Real-time token streaming from FastAPI to React UI                                                                                 |
| **Observability**       | Logfire spans across every pipeline stage                                                                                          |
| **Noisy Corpus**        | 5,449 vectors: ~36% true LangGraph/LangChain docs, ~64% semantically similar noise (LlamaIndex, Haystack) to stress-test retrieval |

---

## Tech Stack

**AI/ML**

- [LangGraph](https://github.com/langchain-ai/langgraph): stateful agent orchestration
- [Jina AI](https://jina.ai): `jina-embeddings-v3` embeddings (1024-dim)
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank): cross-encoder reranking
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails): safety rails
- [Groq](https://groq.com): `qwen/qwen3.8-27b` inference

**Infrastructure**

- [Qdrant Cloud](https://qdrant.tech): vector database (5,449 points, cosine similarity)
- [FastAPI](https://fastapi.tiangolo.com): async REST API with SSE streaming
- [Logfire](https://logfire.pydantic.dev): observability and tracing
- [Docker](https://www.docker.com): containerized backend

**Frontend**

- React + Vite: chat UI with markdown rendering
- Server-Sent Events (SSE): real-time token streaming

---

## Project Structure

```
RAG/
├── main.py                    # FastAPI app: lifespan, /query, /stream, /graph
├── Dockerfile                 # Backend container image
├── src/
│   ├── agents/
│   │   ├── graph.py           # LangGraph StateGraph definition
│   │   ├── state.py           # AgentState TypedDict
│   │   └── nodes/
│   │       ├── planner.py     # Intent classification (Qwen)
│   │       ├── retriever.py   # Qdrant search + FlashRank reranking
│   │       └── responder.py   # Answer synthesis
│   ├── retrieval/
│   │   ├── embeddings.py      # Jina → Gemini → sentence-transformers fallback chain
│   │   ├── qdrant_service.py  # Vector search + reranking pipeline
│   │   └── ranking_service.py # FlashRank cross-encoder
│   ├── ingestion/
│   │   ├── processor.py       # Universal ingestion pipeline
│   │   ├── chunking/          # Text splitter
│   │   ├── loaders/           # HTML, PDF, Office, text, markdown
│   │   └── connectors/        # Base connector
│   ├── guardrails/
│   │   ├── rails.py           # NeMo LLMRails singleton
│   │   └── colang_rules.py    # Colang intent definitions + flows
│   ├── gateways/
│   │   └── client.py          # LLM gateway (Portkey-ready)
│   └── config/
│       └── config.py          # Centralised settings
├── data/
│   └── corpus/
│       ├── true/              # LangGraph + LangChain docs
│       └── noise/             # LlamaIndex + Haystack docs
└── ui/                        # React + Vite frontend
```

---

## Setup & Running

The backend runs in **Docker**. The frontend runs with Node.

### Prerequisites

- Python 3.12+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) package manager
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1. Clone

```bash
git clone https://github.com/Garv-00/Langchain_and_Langgraph-Based-RAG
cd Langchain_and_Langgraph-Based-RAG
uv sync
```

### 2. Environment Variables

`APP_API_KEY` is a custom secret used to authenticate requests to the application's API endpoints. It is not issued by an external service, so you generate it yourself:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Backend:** create a `.env` file in the project root:

```env
# Application Security
APP_API_KEY=your_generated_secret_key

# Groq (LLM inference)
GROQ_API_KEY_1=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_MODEL_PLANNER=qwen/qwen3.8-27b
GROQ_FALLBACK_API_KEY=your_fallback_groq_key

# Qdrant (vector database)
QDRANT_CLUSTER_ENDPOINT=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# Jina (embeddings)
JINA_API_KEY=your_jina_api_key

# Logfire (observability)
LOGFIRE_TOKEN=your_logfire_token
```

**Frontend:** create `ui/.env` with the **same** key:

```env
VITE_API_KEY_NEW=your_generated_secret_key
```

> Write values without quotes (`KEY=value`).

### 3. Ingest documents

```bash
uv run python -m src.ingestion.processor data/corpus --wipe
```

### 4. Run the backend with Docker

```bash
docker build -t rag-backend .
docker run --rm --name rag-backend -p 8000:8000 --env-file .env rag-backend
```

The API is now available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Run the frontend

```bash
cd ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## How the Pipeline Works

### Query Routing

Every message passes through two gates before reaching the LLM:

1. **NeMo Guardrails**: classifies the message using Colang intent matching. Off-topic queries, jailbreak attempts, and prompt injections are blocked immediately with a canned response. The LangGraph pipeline is never invoked.

2. **LangGraph Planner**: for queries that pass the guardrail gate, the planner node uses Qwen to classify intent as either `CONVERSATIONAL` (can be answered from memory) or `TECHNICAL` (requires document retrieval). This prevents unnecessary vector searches for simple follow-up questions.

### Retrieval & Reranking

Technical queries trigger a two-stage retrieval:

1. **Vector search**: Qdrant retrieves the top 15 candidate chunks using cosine similarity over 1024-dim Jina embeddings.
2. **Cross-encoder reranking**: FlashRank's `ms-marco-MiniLM-L-12-v2` reranks the 15 candidates using a cross-encoder that jointly attends to the query and each document, selecting the top 5 most semantically relevant chunks.

This two-stage approach is intentional: cosine similarity is fast but "fuzzy", so it finds candidates. The cross-encoder is slower but precise, so it picks winners.

### Noisy Corpus Design

The corpus deliberately includes 64% noise (semantically related but wrong-framework documentation from LlamaIndex and Haystack). This stress-tests the retrieval pipeline's ability to surface true LangGraph/LangChain content in a high-noise environment, mirroring real enterprise deployments where only a fraction of indexed data is relevant to any given query.

---

## Security

The guardrail system handles four threat categories:

- **Off-topic queries**: non-AI-engineering questions are redirected
- **Jailbreak attempts**: persona reassignment instructions are blocked
- **Prompt injections**: "ignore your instructions" patterns are caught
- **Identity probing**: "what model are you / are you ChatGPT" questions are handled without revealing underlying model details

---

## API Reference

All endpoints require the header `x-api-key: <APP_API_KEY>`.

### `POST /stream`

Streams the response token by token via Server-Sent Events.

```json
// Request
{ "q": "How does MemorySaver work in LangGraph?", "thread_id": "user_123" }

// SSE stream
data: {"token": "Memory"}
data: {"token": "Saver"}
...
data: [DONE]
```

### `POST /query`

Returns the full response in one JSON payload (no streaming).

### `GET /graph`

Returns a Mermaid PNG diagram of the LangGraph agent workflow.

---

## Roadmap

- [ ] **Eval pipeline**: Ragas (faithfulness, answer relevancy, context recall) on a golden Q&A dataset
- [ ] **Portkey gateway**: LLM fallback routing + semantic caching
- [ ] **Fly.io deployment**: live public demo URL
- [ ] **AWS IaC**: Terraform for ECS Fargate + ALB + Secrets Manager
- [ ] **Fine-tuning**: LoRA/QLoRA on a domain-specific task

---

## Authors

**Garv** · [GitHub](https://github.com/Garv-00) · [LinkedIn](https://www.linkedin.com/in/garv-sharma-157796395)

**Anshul Nangru** · [GitHub](https://github.com/anshulnangru) · [LinkedIn](https://www.linkedin.com/in/anshul-nangru/)

---

_Built as a portfolio project demonstrating production-grade agentic AI engineering._
