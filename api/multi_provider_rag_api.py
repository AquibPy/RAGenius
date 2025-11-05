from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Query
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from api.src.data_loader import load_all_documents
from contextlib import asynccontextmanager
import logging
import os
import shutil
from typing import List, Optional, Literal
from pathlib import Path
import uuid
from api import config
from api.models import QueryRequest, ProviderConfig, UploadResponse, EngineInfo, HealthResponse
from api.helper_function import get_engine_key, get_or_create_engine, ensure_default_engine, validate_file
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_engine_key
    try:
        logger.info("=" * 80)
        logger.info("[STARTUP] Initializing Multi-Provider RAG System")
        logger.info(f"[STARTUP] Vector Directory: {config.VECTOR_DIR}")
        logger.info(f"[STARTUP] Upload Directory: {config.DATA_DIR}")

        default_embedding = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "gemini").lower()
        default_llm = os.getenv("DEFAULT_LLM_PROVIDER", "groq").lower()

        logger.info(f"[STARTUP] Default Embedding: {default_embedding.upper()}")
        logger.info(f"[STARTUP] Default LLM: {default_llm.upper()}")

        config.DEFAULT_ENGINE_KEY = get_engine_key(default_embedding,default_llm)
        default_engine = get_or_create_engine(default_embedding, default_llm)

        doc_count = default_engine.vectorstore.collection.count()
        logger.info(f"[STARTUP] ✅ Default RAG Engine initialized: {config.DEFAULT_ENGINE_KEY}")
        logger.info(f"[STARTUP] Vector store contains {doc_count} document chunks")

        logger.info(f"[STARTUP] Supported provider combinations:")
        for combo_name, cfg in config.SUPPORTED_COMBINATIONS.items():
            logger.info(f"  - {combo_name}: Embedding={cfg['embedding']}, LLM={cfg['llm']}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"[STARTUP]  ❌ Failed to initialize RAG System: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize RAG System: {e}")
    yield
    logger.info("[SHUTDOWN] Cleaning up resources...")
    config.RAG_ENGINES.clear()
    logger.info("[SHUTDOWN] ✅ Cleanup complete")

app = FastAPI(
    title="Multi-Provider RAG Search API",
    description="RAG API supporting Azure OpenAI, Groq (LLM) and Azure OpenAI, Gemini (Embeddings)",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers= ["*"]
)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to UI"""
    return RedirectResponse(url="/ragui")

@app.get("/ragui", include_in_schema=False)
async def chat_ui(request: Request):
    """Render the chat interface"""
    return templates.TemplateResponse("mpr_index.html", {"request": request})

@app.post("/rag/basic",
          response_model=dict,
          summary="Basic RAG Query",
          description="Perform a non-streaming RAG query with configurable providers")
async def basic_rag(request:QueryRequest):
    ensure_default_engine()
    try:
        # embedding_provider = request.embedding_provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER","azure")
        # llm_provider = request.llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "azure")
        if request.embedding_provider and request.llm_provider:
            embedding_provider = request.embedding_provider
            llm_provider = request.llm_provider
        else:
            # Use current global default from config
            embedding_provider, llm_provider = config.DEFAULT_ENGINE_KEY.split("-")


        engine = get_or_create_engine(embedding_provider,llm_provider)

        engine_key = get_engine_key(embedding_provider,llm_provider)
        logger.info(f"[BASIC RAG] Query: '{request.query}' | Engine: {engine_key} | top_k: {request.top_k}")

        result = engine.query(
            question= request.query,
            top_k = request.top_k,
            summarize = request.summarize
        )

        if "error" in result:
            logger.error(f"[BASIC RAG] Error: {result['error']}")
            raise HTTPException(status_code=500, detail= result["error"])
        
        response = {
            "query": request.query,
            "answer": result.get("answer",""),
            "sources": result.get("sources",[]),
            "top_k": request.top_k,
            "provider_used": {
                "embedding": embedding_provider,
                "llm": llm_provider
            }

        }
        logger.info(f"[BASIC RAG] ✅ Query processed successfully using {engine_key}")
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[BASIC RAG] Unexpected error : {e}",exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/rag/stream",
          summary="Streaming RAG Query",
          description="Perform a streaming RAG query with configurable providers")
async def streaming_rag(request: QueryRequest):
    ensure_default_engine()
    # embedding_provider = request.embedding_provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER","azure")
    # llm_provider = request.llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "azure")
    if request.embedding_provider and request.llm_provider:
        embedding_provider = request.embedding_provider
        llm_provider = request.llm_provider
    else:
            # Use current global default from config
        embedding_provider, llm_provider = config.DEFAULT_ENGINE_KEY.split("-")


    engine = get_or_create_engine(embedding_provider,llm_provider)

    engine_key = get_engine_key(embedding_provider,llm_provider)
    logger.info(f"[STREAM RAG] Query: '{request.query}' | Engine: {engine_key} | top_k: {request.top_k}")

    async def stream_response():
        try:
            yield f"data: {{\"type\": \"providers\", \"embedding\": \"{embedding_provider}\", \"llm\": \"{llm_provider}\"}}\n\n"
            
            async for token in engine.stream_query(
                question = request.query,
                top_k = request.top_k,
                summarize = request.summarize
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
            logger.info(f"[STREAM RAG] ✅ Streaming completed using {engine_key}")
        except Exception as e:
            logger.error(f"[STREAM RAG] Error during streaming: {e}", exc_info=True)
            yield f"data: [ERROR] {str(e)}\n\n"
        
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/rag/upload",
          response_model=UploadResponse,
          summary="Upload Documents",
          description="Upload Documents to add to the vector store")

async def upload_files(
    files: List[UploadFile] = File(...),
    embedding_provider: Literal["azure","gemini"] = Query(
        default=None,
        description="Embedding Provider to use for these documents"
    )
    ):
    """
    Use Query() when the value comes from the URL query string. 
    And Use Field() inside a Pydantic model to define metadata or validation for body data (not query).
    """

    ensure_default_engine()
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if embedding_provider:
        embed_provider = embedding_provider
        _, default_llm = config.DEFAULT_ENGINE_KEY.split("-")
    else:
        embed_provider, default_llm = config.DEFAULT_ENGINE_KEY.split("-")
    
    logger.info(f"[UPLOAD] Received {len(files)} file(s) for embedding with {embed_provider.upper()}")

    # default_llm = os.getenv("DEFAULT_LLM_PROVIDER", "azure")
    engine = get_or_create_engine(embed_provider,default_llm)

    temp_dir = os.path.join(config.DATA_DIR,f"temp_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir,exist_ok=True)

    file_names = []
    rejected_files = []

    try:
        for file in files:
            is_valid, error_msg = validate_file(file)
            if not is_valid:
                rejected_files.append({"filename": file.filename,"reason": error_msg})
                logger.warning(f"[UPLOAD] Rejected: {file.filename} - {error_msg}")
                continue
            
            file_path = os.path.join(temp_dir,file.filename)
            try:
                with open(file_path,"wb") as f:
                    content = await file.read()
                    if len(content)> config.MAX_FILE_SIZE:
                        rejected_files.append(
                            {
                                "filename": file.filename,
                                "reason": f"File too large (max {config.MAX_FILE_SIZE//(1024*1024)}MB)"
                            }
                        )
                        os.remove(file_path)
                        continue
                    f.write(content)
                file_names.append(file.filename)
                logger.info(f"[UPLOAD] ✅ Saved: {file.filename}")
            except Exception as e:
                rejected_files.append(
                    {
                        "filename": file.filename,
                        "reason": str(e)
                    }
                )
                logger.error(f"[UPLOAD] Failed to save {file.filename}: {e}")
        
        if not file_names:
            raise HTTPException(
                status_code=400,
                detail=f"No valid files to process. Rejected: {rejected_files}"
            )
        
        logger.info(f"[UPLOAD] Loading {len(file_names)} files(s)...")
        docs = load_all_documents(temp_dir)

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No valid documents could be extracted from upload files"
            )
        before = engine.vectorstore.collection.count()
        logger.info(f"[UPLOAD] Adding documents with {embed_provider.upper()} embeddings (current count: {before})...")

        engine.vectorstore.add_documents(docs)

        after = engine.vectorstore.collection.count()
        added_total = after - before

        logger.info(f"[UPLOAD] ✅ Added {added_total} chunks using {embed_provider.upper()} embeddings")

        response = UploadResponse(
            message=f"✅ Successfully uploaded {len(file_names)} file(s) using {embed_provider.upper()} embeddings",
            files= file_names,
            new_chunks_added=added_total,
            total_chunks_in_db= after,
            rejected_files= rejected_files if rejected_files else None
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] Error: {e}",exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"[UPLOAD] Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"[UPLOAD] Failed to clean up {temp_dir}: {e}")

@app.get("/rag/engines",
          summary="List Active Engine",
          description="List all currently active RAG engine configurations")
async def list_engines():
    ensure_default_engine()
    
    engine_info = []
    for engine_key,engine in config.RAG_ENGINES.items():
        embedding_prov,llm_prov = engine_key.split("-")
        engine_info.append(
            EngineInfo(
                combination= engine_key,
                embedding_provider=embedding_prov,
                llm_provider=llm_prov,
                vectorstore_count=engine.vectorstore.collection.count(),
                is_default=engine_key==config.DEFAULT_ENGINE_KEY
            )
        )
    return JSONResponse(content={
    "active_engine": [e.dict() for e in engine_info],
    "total_engines": len(engine_info),
    "default_engine": config.DEFAULT_ENGINE_KEY})

@app.post(
    "/rag/switch-default",
    summary="Switch Default Provider",
    description="Change the default RAG provider configuration (embedding + LLM)"
)
async def switch_default_provider(prov_cfg: ProviderConfig):

    ensure_default_engine()

    new_key = get_engine_key(prov_cfg.embedding_provider, prov_cfg.llm_provider)

    try:
        engine = get_or_create_engine(prov_cfg.embedding_provider, prov_cfg.llm_provider)
    except Exception as e:
        logger.error(f"[CONFIG] Failed to initialize engine {new_key}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize engine for {new_key}: {str(e)}"
        )

    old_key = config.DEFAULT_ENGINE_KEY
    config.DEFAULT_ENGINE_KEY = new_key

    logger.info(f"[CONFIG] ✅ Default engine switched from {old_key} → {new_key}")

    response = {
        "message": f"✅ Default provider switched successfully.",
        "previous_default": old_key,
        "new_default": new_key,
        "embedding_provider": prov_cfg.embedding_provider,
        "llm_provider": prov_cfg.llm_provider,
        "vectorstore_chunks": engine.vectorstore.collection.count(),
        "note": (
            "This change affects only the current process. "
            "If running multiple FastAPI workers, each worker must be updated separately."
        ),
    }

    return JSONResponse(content=response, status_code=200)


@app.get(
    "/health",
    summary="Health check",
    description="Check the API health status"
)
async def health_check():
    if not config.RAG_ENGINES:
        return HealthResponse(
            status="initializing",
            active_engines={},
            default_config={},
            error= "RAG system is still initializing"
        )
    
    try:
        engine_status = {}
        for engine_key,engine in config.RAG_ENGINES.items():
            embedding_prov,llm_prov = engine_key.split("-")
            engine_status[engine_key] = {
                "embedding": embedding_prov,
                "llm": llm_prov,
                "chunks": engine.vectorstore.collection.count()
            }
        
        default_embedding, default_llm = config.DEFAULT_ENGINE_KEY.split("-")
        return HealthResponse(
            status="healthy",
            active_engines= engine_status,
            default_config={
                "key": config.DEFAULT_ENGINE_KEY,
                "embedding":default_embedding,
                "llm": default_llm
            }
        )
    
    except Exception as e:
        logger.error(f"[HEALTH] Check failed: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            active_engines={},
            default_config={},
            error=str(e)
        )

@app.get(
    "/rag/stats",
    summary="Vector Store Statistics",
    description="Get detailed statistics about the vector stores"
)
async def get_stats():
    ensure_default_engine()
    try:
        all_stats = {}
        for engine_key,engine in config.RAG_ENGINES.items():
            collection = engine.vectorstore.collection
            count = collection.count()
            sample = collection.get(limit = min(100,count),include = ["metadatas"]) if count>0 else {"metadatas": []}

            file_types = {}
            source_files = set()

            for metadata in sample.get("metadatas",[]):
                if metadata:
                    file_type = metadata.get("file_type","unknown")
                    source_file = metadata.get("source_file","unknown")

                    file_types[file_type] = file_types.get(file_type,0)+1
                    source_files.add(source_file)
            embedding_prov,llm_prov = engine_key.split("-")
            all_stats[engine_key] = {
                "embedding_provider": embedding_prov,
                "llm_provider": llm_prov,
                "total_chunks": count,
                "unique_source_files": len(source_files),
                "file_type_distribution": file_types,
                 "sample_source_files": list(source_files)[:10] if source_files else []
            }
        return {
            "engines": all_stats,
            "total_engines": len(all_stats)
        }
    except Exception as e:
        logger.error(f"[STATS] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/providers",
         summary="Supported Providers",
         description="List all supported provider combinations")

async def list_providers():
    return JSONResponse(
        content={
            "supported_combinations": config.SUPPORTED_COMBINATIONS,
            "embedding_providers": ["azure","gemini"],
            "llm_provider": ["azure","groq"],
            "description": "You can mix any embedding provider with any LLM provider"
        },
        status_code=200
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request:Request,exec:HTTPException):
    logger.warning(f"HTTP {exec.status_code}:{exec.detail} | Path:{request.url.path}")
    return JSONResponse(
        status_code=exec.status_code,
        content={"details": exec.detail,"path": str(request.url.path)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request:Request,exc:Exception):
    logger.error(f"Unhandled Exception: {exc}",exc_info=True)
    return JSONResponse(
        status_code= 500,
        content={
            "detail":"Internal Server Error",
            "error": str(exc),
            "path": str(request.url.path)
        }
    )