import os
from openai import AzureOpenAI
import numpy as np
from typing import List
from src.data_loader import load_all_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(override=True)

class EmbeddingPipeline:
    def __init__(self, model_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Read environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if not azure_endpoint or not api_key or not api_version:
            raise ValueError("AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_API_VERSION must all be set in environment variables.")
        
        self.model_name = model_name or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
    
    def chunk_documents(self,documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators= ["\n\n","\n"," ",""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        chunk_texts = [doc.page_content for doc in split_docs]
        return chunk_texts

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Input must be a list of strings.")

        print(f"Generating embeddings for {len(texts)} texts using model: {self.model_name}")

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.generate_embeddings(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)