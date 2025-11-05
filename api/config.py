VECTOR_DIR = "chromadb_store"
DATA_DIR = "upload_files"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx", ".json", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024 # 50MB per files
SUPPORTED_COMBINATIONS = {
    "gemini-azure": {"embedding": "gemini", "llm": "azure"},
    "gemini-groq": {"embedding": "gemini", "llm": "groq"},
    "azure-azure": {"embedding": "azure", "llm": "azure"},
    "azure-groq" : {"embedding": "azure", "llm": "groq"}
}
RAG_ENGINES = {}
DEFAULT_ENGINE_KEY = None