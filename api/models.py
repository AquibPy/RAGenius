from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

class QueryRequest(BaseModel):
    query: str = Field(...,min_length=2, max_length=1000, description="The Search Query")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of Results to Return")
    summarize: bool = Field(default=False,description="Whether to Include Summary")
    embedding_provider: Optional[Literal["azure", "gemini"]] = Field(
        default=None,
        description="Override embedding provider for this query"
    )
    llm_provider: Optional[Literal["azure", "groq"]] = Field(
        default=None,
        description="Override LLM provider for this query"
    )

    @field_validator("query")
    def query_not_empty(cls,v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
class ProviderConfig(BaseModel):
    embedding_provider: Literal["azure","gemini"] = Field(
        default="azure",
        description="Embedding Provider to use"
    )
    llm_provider: Literal["azure","groq"] = Field(
        default="azure",
        description="LLM Provider to use"
    )

class UploadResponse(BaseModel):
    message: str
    files: List[str]
    new_chunks_added: int
    total_chunks_in_db: int
    rejected_files: Optional[List[dict]] = None

class EngineInfo(BaseModel):
    combination: str
    embedding_provider: str
    llm_provider: str
    vectorstore_count: int
    is_default: bool

class HealthResponse(BaseModel):
    status: str
    active_engines: dict
    default_config: dict
    error: Optional[str] = None