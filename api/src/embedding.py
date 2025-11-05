import os
from openai import AzureOpenAI
import numpy as np
from typing import List
from api.src.data_loader import load_all_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time

load_dotenv(override=True)

class EmbeddingPipeline:
    def __init__(self,provider:str = "azure", chunk_size: int = 1000, chunk_overlap: int = 200):
        # Read environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.provider = provider.lower()

        # Initialize Azure OpenAI client
        if self.provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.client = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif self.provider=='azure':
            if not azure_endpoint or not api_key or not api_version:
                raise ValueError("AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_API_VERSION must all be set in environment variables.")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        else:
            raise ValueError(f"❌ Unsupported embedding provider: {self.provider}")
    
    def chunk_documents(self,documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators= ["\n\n","\n"," ",""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        # chunk_texts = [doc.page_content for doc in split_docs]
        return split_docs

    def generate_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
            
            try:
                # --- Gemini ---
                if self.provider == "gemini":
                    batch_emb = self.client.embed_documents(batch)

                # --- Azure OpenAI ---
                else:
                    response = self.client.embeddings.create(model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"), input=batch)
                    batch_emb = [item.embedding for item in response.data]

                all_embeddings.extend(batch_emb)

            except Exception as e:
                error_msg = str(e).lower()

                # --- Rate limit / quota handling ---
                if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                    print(f"[WARN] Rate limit or quota error encountered: {e}")
                    print("[INFO] Pausing for 60 seconds before retrying...")
                    time.sleep(60)

                    # Retry once after cooldown
                    try:
                        if self.provider == "gemini":
                            batch_emb = self.client.embed_documents(batch)
                        else:
                            response = self.client.embeddings.create(model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"), input=batch)
                            batch_emb = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_emb)
                        print("[INFO] Retry succeeded.")
                    except Exception as retry_err:
                        print(f"[ERROR] Retry failed due to: {retry_err}")
                        print("[WARN] Skipping this batch.")
                        continue

                else:
                    # Unexpected error — don’t fail silently
                    print(f"[ERROR] Unexpected error while processing batch {i // batch_size + 1}: {e}")
                    print("[WARN] Skipping this batch.")
                    continue

        embeddings = np.array(all_embeddings, dtype=np.float32)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline(provider="gemini")
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.generate_embeddings([chunk.page_content for chunk in chunks])
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)