import os
import uuid
import chromadb
from typing import List, Any
from api.src.embedding import EmbeddingPipeline

class ChromaVectorStore:
    def __init__(self,
                 collection_name: str = "pdf_documents",
                 persist_directory: str = "chromadb_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 provider: str = "azure"
                #  embedding_model: str = None,
                ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_provider = provider.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.provider = provider
        # self.embedding_model = embedding_model

        os.makedirs(self.persist_directory, exist_ok=True)


        # Initialize embedding pipeline
        self.emb_pipe = EmbeddingPipeline(
            provider=self.embedding_provider,
            # model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Use provider-specific Chroma folder to avoid mixing embeddings
        provider_folder = "chroma_gemini" if self.embedding_provider=='gemini'else "chroma_azure"
        self.store_path = os.path.join(self.persist_directory,provider_folder)
        os.makedirs(self.store_path,exist_ok=True)


        # Initialize Chroma client and collection
        self.client = chromadb.PersistentClient(path=self.store_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": f"PDF embeddings using {self.provider.upper()}"}
        )
        print(f"[INFO] ✅ Initialized Chroma vector store with provider: {self.provider.upper()}")
        print(f"[INFO] Storage directory: {self.store_path}")
        print(f"[INFO] Chroma vector store initialized: {self.collection_name}")
        print(f"[INFO] Existing documents in collection: {self.collection.count()}")

    def add_documents(self, documents: List[Any]):
        """
        Chunks, embeds, and adds documents to ChromaDB.
        """
        print(f"[INFO] Adding {len(documents)} documents to Chroma store using {self.provider.upper()} embeddings...")

        # Split documents into chunks
        chunks = self.emb_pipe.chunk_documents(documents)

        # Generate embeddings using Azure OpenAI
        chunk_text = [c.page_content for c in chunks]
        embeddings = self.emb_pipe.generate_embeddings(chunk_text)

        ids, metadatas = [], []

        for i, (chunk_doc, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(getattr(chunk_doc, "metadata", {}))
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=chunk_text
        )
        print(f"[INFO] Successfully added {len(chunks)} chunks using {self.provider.upper()} embeddings..")
        print(f"[INFO] Total documents in collection: {self.collection.count()}")

    def query_db(self, query_text: str, top_k: int = 5):
        """
        Query ChromaDB using Azure embeddings.
        """
        print(f"[INFO] Querying collection '{self.collection_name}' with {self.provider.upper()} embeddings...")
        query_emb = self.emb_pipe.generate_embeddings([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def list_documents(self):
        return self.collection.get(include=["metadatas", "documents"])

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
        print(f"[INFO] Collection '{self.collection_name}' deleted.")


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")

    store = ChromaVectorStore(persist_directory="chromadb_store",provider="azure")
    store.add_documents(docs)

    query_result = store.query_db("what is llm poisoning", top_k=3)
    for doc, meta, dist in zip(query_result['documents'][0],
                               query_result['metadatas'][0],
                               query_result['distances'][0]):
        print(f"Distance: {dist:.4f}, Text snippet: {doc[:150]}...")
    print(query_result)