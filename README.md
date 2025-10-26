# 🧠 RAGenius — A Smart Retrieval-Augmented Generation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAGenius is an enterprise-grade **Retrieval-Augmented Generation (RAG)** system that combines the power of **Azure OpenAI** with **ChromaDB** vector storage to deliver intelligent document querying capabilities. Built with **FastAPI**, it provides a robust API for processing, embedding, and querying documents with support for both real-time streaming and batch processing modes.

---

## 🌟 Key Features

### 📄 Document Processing
- **Multi-format Support**: Process PDF, Excel (.xlsx), JSON, CSV, DOCX, and TXT files
- **Batch Upload**: Upload multiple documents simultaneously via API
- **Smart Chunking**: Intelligent document splitting with configurable chunk size and overlap
- **Metadata Preservation**: Automatic file metadata tracking (filename, type, source)

### 🔍 Advanced RAG Capabilities
- **Dual Query Modes**: 
  - **Basic Mode**: Standard request-response for quick queries
  - **Streaming Mode**: Real-time token-by-token responses for enhanced UX
- **Vector Similarity Search**: Powered by ChromaDB for fast, accurate retrieval
- **Azure OpenAI Integration**: Leverages GPT-4 for high-quality answer generation
- **Context-Aware Responses**: Retrieves relevant document chunks before generating answers

### 🗄️ Vector Store Management
- **Persistent Storage**: ChromaDB-backed vector database with disk persistence
- **Incremental Updates**: Add new documents without rebuilding the entire index
- **Configurable Embeddings**: Support for Azure OpenAI embedding models
- **Automatic Initialization**: Vector store setup on first run

### 🛠️ Developer-Friendly
- **RESTful API**: Clean, documented endpoints via FastAPI
- **Async Support**: Built with Python's asyncio for concurrent operations
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Health Checks**: Built-in endpoint for service monitoring
- **Conversation History**: Track query-answer pairs across sessions

---

## 📁 Project Structure

```
RAG/
├── chromadb_store/          # Persistent vector database storage
├── data/                    # Default directory for documents
│   ├── pdf_files/
│   └── text_files/
├── notebooks/
│   └── document.ipynb       # Jupyter notebook for experimentation
├── src/                     # Core application modules
│   ├── __init__.py
│   ├── data_loader.py       # Multi-format document loader
│   ├── embedding.py         # Azure OpenAI embedding pipeline
│   ├── vectorstore.py       # ChromaDB vector store wrapper
│   └── search.py            # RAG engine (basic + streaming)
├── .env                     # Environment variables (Azure credentials)
├── .gitignore
├── app.py                   # FastAPI application
├── main.py                  # CLI interface for RAG operations
├── pyproject.toml           # Project dependencies (UV/Poetry)
├── requirements.txt         # Pip requirements
├── README.md                # This file
└── uv.lock                  # UV lockfile
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Azure OpenAI Account** with:
  - Deployed GPT-4 model
  - Text embedding model (e.g., `text-embedding-3-large`)
  - API keys and endpoints

### Installation

#### Option 1: Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/AquibPy/RAGenius.git
cd RAGenius

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/AquibPy/RAGenius.git
cd RAGenius

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

---

## 💻 Usage

### 1️⃣ Command-Line Interface

#### Initialize Vector Store with Documents

```bash
# Load documents from 'data' directory and build vector store
python main.py --query "What is attention mechanism?" --mode basic
```

#### Basic RAG Query

```bash
python main.py --query "Explain transformers in NLP" --mode basic
```

#### Streaming RAG Query

```bash
python main.py --query "What are the key components of BERT?" --mode streaming
```

### 2️⃣ FastAPI Server

#### Start the Server

```bash
# Using Uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or with UV
uv run uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

#### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3️⃣ API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "vectorstore_count": 247
}
```

#### Upload Documents

```bash
curl -X POST "http://localhost:8000/rag/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@spreadsheet.xlsx" \
  -F "files=@data.json"
```

**Response:**
```json
{
  "message": "✅ Uploaded 3 files successfully.",
  "files": ["document1.pdf", "spreadsheet.xlsx", "data.json"],
  "new_chunks_added": 42,
  "total_chunks_in_db": 289
}
```

#### Basic RAG Query

```bash
curl -X POST "http://localhost:8000/rag/basic" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is attention mechanism?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "What is attention mechanism?",
  "answer": "The attention mechanism is a neural network component that allows models to focus on relevant parts of the input when processing sequences..."
}
```

#### Streaming RAG Query

```bash
curl -X POST "http://localhost:8000/rag/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain BERT architecture", "top_k": 3}' \
  --no-buffer
```

**Response (Server-Sent Events):**
```
data: BERT
data:  (
data: Bid
data: irectional
data:  Encoder
data:  Representations
...
data: [DONE]
```

---

## 🧩 Core Components

### 1. Data Loader (`data_loader.py`)

Handles multi-format document ingestion:

```python
from src.data_loader import load_all_documents

# Load all supported files from a directory
docs = load_all_documents("data")
print(f"Loaded {len(docs)} documents")
```

**Supported Formats:**
- PDF (`.pdf`) via PyPDFLoader
- Text (`.txt`) with UTF-8 encoding
- CSV (`.csv`) via CSVLoader
- Word (`.docx`) via Docx2txtLoader
- JSON (`.json`) via JSONLoader
- Excel (`.xlsx`) via UnstructuredExcelLoader

### 2. Embedding Pipeline (`embedding.py`)

Generates embeddings using Azure OpenAI:

```python
from src.embedding import EmbeddingPipeline

emb_pipe = EmbeddingPipeline(
    model_name="text-embedding-3-large",
    chunk_size=1000,
    chunk_overlap=200
)

# Chunk documents
chunks = emb_pipe.chunk_documents(documents)

# Generate embeddings
embeddings = emb_pipe.generate_embeddings(chunks)
```

**Features:**
- Configurable chunking with overlap
- Batch embedding generation
- Automatic text splitting with semantic boundaries

### 3. Vector Store (`vectorstore.py`)

ChromaDB wrapper for vector operations:

```python
from src.vectorstore import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="pdf_documents",
    persist_directory="chromadb_store",
    chunk_size=1000,
    chunk_overlap=200
)

# Add documents
store.add_documents(documents)

# Query
results = store.query("What is machine learning?", top_k=5)
```

**Operations:**
- `add_documents()`: Incrementally add document chunks
- `query()`: Semantic search with configurable top-k
- `list_documents()`: Retrieve all stored documents
- `delete_collection()`: Remove entire collection

### 4. RAG Engine (`search.py`)

Unified interface for RAG operations:

```python
from src.search import RAGEngine

rag = RAGEngine(
    persist_dir="chromadb_store",
    llm_model="gpt-4",
    temperature=0.7,
    streaming=True
)

# Basic query
result = rag.query("Explain neural networks", top_k=3)
print(result["answer"])

# Streaming query
async for token in rag.stream_query("What is deep learning?", top_k=5):
    print(token, end="", flush=True)
```

**Query Modes:**
- **Basic**: Synchronous query with complete response
- **Streaming**: Async generator for token-by-token output

---

## ⚙️ Configuration Options

### Vector Store Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collection_name` | `pdf_documents` | ChromaDB collection identifier |
| `persist_directory` | `chromadb_store` | Vector database storage location |
| `chunk_size` | `1000` | Characters per document chunk |
| `chunk_overlap` | `200` | Overlapping characters between chunks |

### RAG Engine Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_model` | `gpt-4` | Azure OpenAI model name |
| `temperature` | `0.7` | Response creativity (0.0-2.0) |
| `streaming` | `True` | Enable token streaming |
| `top_k` | `5` | Number of context chunks to retrieve |

### Embedding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `text-embedding-3-large` | Azure embedding model |
| `chunk_size` | `1000` | Text chunk size for embeddings |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |

---

## 🔧 Advanced Usage

### Programmatic Document Processing

```python
from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore

# Load documents
docs = load_all_documents("my_documents")

# Initialize vector store
store = ChromaVectorStore(persist_directory="custom_store")

# Add with custom chunking
store.chunk_size = 500
store.chunk_overlap = 100
store.add_documents(docs)

# Query with filters
results = store.query("machine learning", top_k=10)
```

### Custom RAG Pipeline

```python
import asyncio
from src.search import RAGEngine

async def custom_rag_workflow():
    rag = RAGEngine(temperature=0.3)  # More deterministic
    
    questions = [
        "What is attention mechanism?",
        "Explain transformers",
        "What is BERT?"
    ]
    
    for q in questions:
        print(f"\n🔍 Query: {q}")
        async for token in rag.stream_query(q, top_k=3):
            print(token, end="", flush=True)
        print("\n" + "-"*60)

asyncio.run(custom_rag_workflow())
```

### Batch Document Upload via API

```python
import requests

files = [
    ('files', open('doc1.pdf', 'rb')),
    ('files', open('doc2.xlsx', 'rb')),
    ('files', open('doc3.json', 'rb'))
]

response = requests.post(
    'http://localhost:8000/rag/upload',
    files=files
)

print(response.json())
```

---

## 📊 Performance Considerations

### Embedding Generation
- **Batch Processing**: Embeddings are generated in batches for efficiency
- **Caching**: ChromaDB caches embeddings to avoid recomputation
- **Parallel Processing**: Document loading uses concurrent operations

### Query Optimization
- **Top-K Selection**: Adjust `top_k` parameter to balance context vs. speed
- **Chunk Size**: Smaller chunks = more precise retrieval, larger = better context
- **Temperature**: Lower values (0.1-0.5) for factual answers, higher (0.7-1.0) for creative responses

### Scaling Tips
- **Persistent Storage**: ChromaDB stores vectors on disk for fast restarts
- **Incremental Updates**: Add documents without rebuilding entire index
- **Async Operations**: Use streaming mode for better UX in production

---

## 🐳 Docker Deployment

### Dockerfile


### Build and Run

```bash
# Build image
docker build -t ragenius:latest .

# Run container
docker run -p 8000:8000 --env-file .env ragenius:latest

# Or use Docker Compose
docker-compose up -d
```



---

## 🛡️ Security Best Practices

1. **Environment Variables**: Never commit `.env` files to version control
2. **API Keys**: Rotate Azure OpenAI keys regularly
3. **Input Validation**: FastAPI automatically validates request payloads
4. **File Upload Limits**: Configure max file size in production
5. **Rate Limiting**: Implement rate limiting for public APIs

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/AquibPy/RAGenius.git
cd RAGenius

# Install development dependencies
uv sync --dev

# Run pre-commit hooks
pre-commit install
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastAPI** - Modern, fast web framework for building APIs
- **LangChain** - Framework for LLM-powered applications
- **ChromaDB** - AI-native open-source embedding database
- **Azure OpenAI** - Enterprise-grade language models
- **Astral UV** - Next-generation Python package manager

---

## 📮 Support

- **Documentation**: [Full API Docs](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/AquibPy/RAGenius/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AquibPy/RAGenius/discussions)

---

## 🗺️ Roadmap

- [ ] Support for additional LLM providers (OpenAI, Anthropic, Cohere)
- [ ] Web UI for document management and querying
- [ ] Multi-language support for document processing
- [ ] Advanced filtering and metadata search
- [ ] Integration with cloud storage (S3, Azure Blob)
- [ ] Conversation memory and context management
- [ ] Fine-tuned embedding models
- [ ] Kubernetes deployment manifests

---

<div align="center">

**Built with ❤️ by MOHD AQUIB**

⭐ Star us on GitHub if you find this project useful!

</div>