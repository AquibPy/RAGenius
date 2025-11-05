import logging
from api import config
from fastapi import HTTPException, UploadFile
from typing import Optional
from pathlib import Path
from api.src.search import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_engine_key(embedding_provider: str, llm_provider: str) -> str:
    return f"{embedding_provider}-{llm_provider}"

def get_or_create_engine(embedding_provider: str, llm_provider:str):
    
    engine_key = get_engine_key(embedding_provider,llm_provider)

    if engine_key not in config.RAG_ENGINES:
        logger.info(f"[ENGINE] Creating new RAG engine: {engine_key}")
        try:
            config.RAG_ENGINES[engine_key] = RAGEngine(
                persist_dir= config.VECTOR_DIR,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider
            )
            logger.info(f"[ENGINE] ✅ Created engine: {engine_key}")
        except Exception as e:
            logger.error(f"[ENGINE] Failed to create engine {engine_key}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize RAG Engine with {engine_key}: {str(e)}"
            )
    return config.RAG_ENGINES[engine_key]

def validate_file(file: UploadFile) -> tuple[bool, Optional[str]]:
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type: {file_ext}. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    return True, None




def ensure_default_engine():
       if config.DEFAULT_ENGINE_KEY is None or config.DEFAULT_ENGINE_KEY not in config.RAG_ENGINES:
           raise HTTPException(
               status_code= 503,
               detail="NO RAG Engine initialized. Please wait for the startup to complete."
           )
