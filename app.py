from fastapi import FastAPI, HTTPException,UploadFile,File, Request
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.search import RAGEngine
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import os
import shutil
from typing import List


load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Global RAG engine instance
rag_engine = None

VECTOR_DIR = "chromadb_store"
DATA_DIR = "upload_files"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces the deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    # Startup
    global rag_engine
    try:
        logger.info("[BOOT] Initializing RAG Engine...")
        rag_engine = RAGEngine(persist_dir=VECTOR_DIR)
        logger.info(f"[BOOT] RAG Engine ready! Vector store contains {rag_engine.vectorstore.collection.count()} documents.")
    except Exception as e:
        logger.error(f"[BOOT] Failed to initialize RAG Engine: {e}")
        raise
    
    yield
    logger.info("[SHUTDOWN] Cleaning up resources...")


app = FastAPI(
    title="RAG Search API",
    description="An API for Basic and Streaming RAG Search using Azure OpenAI + ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return RedirectResponse("/ragui")

@app.get("/ragui", description="Provides a simple web interface to interact with the RAG")
async def chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/rag/basic")
async def basic_rag(request: QueryRequest):
    """Non-streaming RAG endpoint"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        result = rag_engine.query(request.query, top_k=request.top_k)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        response = {
            "query": request.query,
            "answer": result.get("answer") if isinstance(result, dict) else result,
            "sources": result.get("sources", []) if isinstance(result, dict) else [],
        }
        return JSONResponse(content=response, status_code=200)
    
    except Exception as e:
        logger.error(f"Error in basic_rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/stream")
async def streaming_rag(request: QueryRequest):
    """Streaming RAG endpoint"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    async def stream_response():
        try:
            async for token in rag_engine.stream_query(
                question=request.query,
                top_k=request.top_k,
                summarize=False
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/rag/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents (PDF, JSON, Excel, etc.)
    and append them to the existing Chroma vector store.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Create a temporary directory for this batch upload
        import uuid
        temp_dir = os.path.join(DATA_DIR, f"temp_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_names = []
        
        try:
            # Step 1: Save all uploaded files to temp directory
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                file_names.append(file.filename)
                
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                
                logger.info(f"Saved: {file.filename}")
            
            # Step 2: Load ALL documents from temp directory at once
            logger.info(f"Loading {len(files)} files from {temp_dir}...")
            docs = load_all_documents(temp_dir)
            
            if not docs:
                raise HTTPException(
                    status_code=400, 
                    detail="No valid documents could be extracted from uploaded files"
                )
            
            # Step 3: Add to vector store
            before = rag_engine.vectorstore.collection.count()
            rag_engine.vectorstore.add_documents(docs)
            after = rag_engine.vectorstore.collection.count()
            
            added_total = after - before
            logger.info(f"Added {added_total} chunks to vector store")
            
            return {
                "message": f"✅ Uploaded {len(files)} files successfully.",
                "files": file_names,
                "new_chunks_added": added_total,
                "total_chunks_in_db": after,
            }
        
        finally:
            # Step 4: Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_engine is None:
        return {"status": "initializing", "vectorstore_count": 0}
    
    try:
        count = rag_engine.vectorstore.collection.count()
        return {
            "status": "healthy",
            "vectorstore_count": count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }