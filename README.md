# 🧠 RAGenius v2 — Multi-Provider Retrieval-Augmented Generation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-blueviolet)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RAGenius v2** is a next-gen **Retrieval-Augmented Generation (RAG)** system supporting **multiple AI providers**:
- 🧠 **Azure OpenAI** — enterprise-grade LLMs  
- 🌐 **Google Gemini** — efficient, high-quality embeddings  
- ⚡ **Groq** — ultra-fast inference  

Built with **FastAPI**, **ChromaDB**, and **LangChain**, it enables hybrid document retrieval, intelligent context enrichment, and real-time streaming answers — across multiple AI providers.

---

## 🔭 Architecture Overview

```mermaid
flowchart TD
    subgraph Client["User / Frontend / API Client"]
        Q[User Query or File Upload]
    end

    subgraph API["FastAPI App: multi_provider_rag_api.py"]
        A1[/Request Handler/]
        A2{{"Engine Registry"}}
        A3[[RAG Engine]]
    end

    subgraph Embedding["Embedding Pipeline"]
        E1[Gemini]
        E2[Azure OpenAI]
    end

    subgraph LLM["LLM Provider"]
        L1[Azure GPT-4]
        L2[Groq GPT-OSS]
    end

    subgraph Store["Vector Store (ChromaDB)"]
        C1[(chroma_gemini)]
        C2[(chroma_azure)]
    end

    Q --> A1
    A1 --> A2
    A2 -->|Select Combo| A3
    A3 -->|Embeddings| Embedding
    A3 -->|Retrieval| Store
    A3 -->|LLM Query| LLM
    Embedding --> Store
    LLM --> A1

````

---

## 🔁 End-to-End System Flow (Sequence Diagram)

```mermaid
sequenceDiagram
    participant U as 🧍 User
    participant F as ⚙️ FastAPI (multi_provider_rag_api)
    participant E as 🧩 RAG Engine
    participant V as 💾 ChromaDB
    participant L as 🧠 LLM (Azure / Groq)

    U->>F: POST /rag/basic (query)
    F->>E: get_or_create_engine(provider combo)
    E->>V: query_db(question, top_k)
    V-->>E: relevant chunks (context)
    E->>L: generate answer(context + query)
    L-->>E: final response
    E-->>F: formatted result
    F-->>U: JSON / stream response
```

---

## 🌟 Key Features

### 🧩 Multi-Provider RAG Engine

* Supports **Azure OpenAI**, **Google Gemini**, and **Groq**
* Dynamic engine creation per provider combination
* Separate Chroma collections for embedding isolation

### 📄 Document Processing

* Multi-format support: PDF, Excel, JSON, CSV, DOCX, TXT
* Automatic metadata tagging
* Chunking with overlap and semantic boundaries

### 🔍 Query & Retrieval

* **Basic Mode:** Synchronous query
* **Streaming Mode:** Real-time token responses
* Vector similarity search with ChromaDB
* Summarization and top-K retrieval

### 🗄️ Vector Store Management

* Persistent ChromaDB storage
* Provider-specific directories (`chroma_azure`, `chroma_gemini`)
* Incremental updates without full reindexing

---

## 🧱 Project Structure

```
RAG/
├── api/
│   ├── app.py                         # Azure-only FastAPI app
│   ├── multi_provider_rag_api.py      # Multi-provider RAG API
│   ├── config.py                      # Configuration and defaults
│   ├── helper_function.py             # Engine management helpers
│   ├── models.py                      # Pydantic models
│   └── src/
│       ├── __init__.py
│       ├── data_loader.py             # Document ingestion
│       ├── embedding.py               # Embedding pipeline (Gemini & Azure)
│       ├── vectorstore.py             # ChromaDB integration
│       └── search.py                  # RAG engine (basic & streaming)
├── chromadb_store/
├── data/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

* Python **3.10+**
* API keys for **Azure OpenAI**, **Google Gemini**, and/or **Groq**
* Optional: Docker for deployment

---

### 2️⃣ Installation

#### Option 1: Using UV (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/AquibPy/RAGenius.git
cd RAGenius

uv sync
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

#### Option 2: Using pip

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### 3️⃣ Environment Configuration

```bash
cp .env.example .env
```

Example `.env`:

```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
```

---

### 4️⃣ Run API Server

#### Start the Multi-Provider API

```bash
uvicorn api.multi_provider_rag_api:app --reload --host 0.0.0.0 --port 8000
```

#### For Azure-only API

```bash
uv run uvicorn api.multi_provider_rag_api:app --reload
```

Access:

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔧 CLI Usage

```bash
# Basic RAG query
python main.py --query "What is attention mechanism?" --mode basic

# Streaming RAG query
python main.py --query "Explain transformers in NLP" --mode streaming
```

---

## 🔌 API Reference (Highlights)

| Endpoint              | Description              | Method | Output |
| --------------------- | ------------------------ | ------ | ------ |
| `/rag/basic`          | Non-streaming RAG query  | POST   | JSON   |
| `/rag/stream`         | Streaming RAG output     | POST   | SSE    |
| `/rag/upload`         | Upload documents         | POST   | JSON   |
| `/rag/engines`        | List active RAG engines  | GET    | JSON   |
| `/rag/switch-default` | Switch default providers | POST   | JSON   |
| `/rag/stats`          | Vector store analytics   | GET    | JSON   |
| `/health`             | Service health check     | GET    | JSON   |

---

## 🧩 Example Query

```bash
curl -X POST "http://localhost:8000/rag/basic" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "top_k": 5}'
```

Response:

```json
{
  "query": "What is deep learning?",
  "answer": "Deep learning is a subfield of AI using neural networks with many layers...",
  "sources": [...],
  "providers_used": {"embedding": "gemini", "llm": "groq"}
}
```

---

## ⚙️ Configuration Options

| Parameter                    | Default          | Description                |
| ---------------------------- | ---------------- | -------------------------- |
| `DEFAULT_EMBEDDING_PROVIDER` | `gemini`         | Default embedding provider |
| `DEFAULT_LLM_PROVIDER`       | `groq`           | Default LLM provider       |
| `MAX_FILE_SIZE`              | 50 MB            | Maximum upload size        |
| `VECTOR_DIR`                 | `chromadb_store` | Vector database directory  |

---

## 🧠 Core Components

### 1. `data_loader.py`

Loads and preprocesses multiple file formats with LangChain loaders.

### 2. `embedding.py`

Handles embeddings via **Gemini** or **Azure OpenAI** with batching and retry logic.

### 3. `vectorstore.py`

ChromaDB wrapper with provider-specific stores.

### 4. `search.py`

RAGEngine for hybrid retrieval and generation (sync and streaming).

---

## ⚖️ Provider Comparison

| Provider | Speed        | Cost      | Quality      |
| -------- | ------------ | --------- | ------------ |
| Gemini   | ⚡ Fast       | 💰 Low    | 🧠 High      |
| Azure    | ⚖️ Medium    | 💰 Medium | 🌟 Excellent |
| Groq     | 🚀 Very Fast | 💰 Low    | 👍 Good      |

---

## 📊 Monitoring

```bash
curl http://localhost:8000/health
curl http://localhost:8000/rag/engines
curl http://localhost:8000/rag/stats
```

---

## 🐳 Docker Deployment

```bash
docker build -t ragenius:latest .
docker run -p 8000:8000 --env-file .env ragenius:latest
```

Or with Docker Compose:

```bash
docker-compose up -d
```

---

## 🧱 Best Practices

| Goal                 | Recommendation                        |
| -------------------- | ------------------------------------- |
| 💰 Cost Optimization | Use Gemini + Groq                     |
| 🌟 Accuracy          | Use Azure + Azure                     |
| ⚡ Speed              | Use Groq for LLM                      |
| 🔐 Isolation         | One embedding provider per dataset    |
| 📁 File Size         | Keep <50 MB or adjust `MAX_FILE_SIZE` |

---

## 🤝 Contributing

```bash
git clone https://github.com/AquibPy/RAGenius.git
cd RAGenius
uv sync --dev
pre-commit install
```

---

## 📝 License

MIT License © 2025 [**Mohd Aquib**](https://github.com/AquibPy)

---

## 🙏 Acknowledgments

* **FastAPI** – Modern async API framework
* **LangChain** – LLM orchestration
* **ChromaDB** – Vector store for embeddings
* **Azure OpenAI** – GPT-4 model suite
* **Google Gemini** – Embedding provider
* **Groq** – Low-latency LLM engine

---

<div align="center">

**Built with ❤️ by [MOHD AQUIB](https://github.com/AquibPy)**
⭐ Star the repo if you find it useful!

</div>

---